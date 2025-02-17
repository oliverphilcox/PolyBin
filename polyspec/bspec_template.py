### Code for binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2025)
## This module contains the bispectrum template estimation code

import numpy as np
import time
from scipy.special import gamma, p_roots, lpmn
from .cython.k_integrals import *
from .cython.ideal_fisher import *
from .cython.fNL_utils import *

class BSpecTemplate():
    """
    Bispectrum estimation class for measuring the amplitudes of separable primoridal bispectrum templates. 
    We also feed in a function that applies the S^-1 operator (which is ideally beam.mask.C_l^{tot,-1}, where C_l^tot includes the beam and noise). 
    
    Inputs:
    - base: PolyBin class
    - mask: HEALPix mask applied to data. We can optionally specify a vector of three masks for [T, Q, U].
    - applySinv: function which returns S^-1 ~ P^dag Cov^{-1} in harmonic space, when applied to a given input map, where P = Mask * Beam.
    - templates: types of templates to compute e.g. [fNL-loc, isw-lensing]
    - k_arr, Tl_arr: k-array, plus T- and (optionally) E-mode transfer functions for all ell. Required for all primordial templates.
    - lmin, lmax: minimum/maximum ell (inclusive)
    - ns, As, k_pivot: primordial power spectrum parameters
    - r_values, r_weights: radial sampling points and weights for 1-dimensional integrals
    - C_Tphi: cross spectrum of temperature and lensing  [C^Tphi_0, C^Tphi_1, etc.]. Required if 'isw-lensing' is in templates.
    - C_lens_weight: dictionary of lensed power spectra (TT, TE, etc.). Required if 'isw-lensing' is in templates.
    - r_star, r_hor: Comoving distance to last-scattering and the horizon (default: Planck 2018 values).
    """
    def __init__(self, base, mask, applySinv, templates, lmin, lmax,  k_arr=[], Tl_arr=[], r_arr=[], ns=0.96, As=2.1e-9, k_pivot=0.05, r_values = [], r_weights = {}, C_Tphi=[], C_lens_weight = {}, r_star=None, r_hor=None):
        # Read in attributes
        self.base = base
        self.mask = mask
        self.applySinv = applySinv
        self.pol = self.base.pol
        self.templates = templates
        self.k_arr = k_arr
        self.lmin = lmin
        self.lmax = lmax
        self.ns = ns
        self.As = As
        self.k_pivot = k_pivot
        
        # Create primordial power spectrum function
        print("Primordial Spectrum: n_s = %.2f, A_s = %.2e, k_pivot = %.2f"%(self.ns, self.As, self.k_pivot));
        self.Pzeta = lambda k: 2.*np.pi**2./k**3*self.As*(k/self.k_pivot)**(self.ns-1)
        
        # Check ell ranges
        assert self.lmax<=base.lmax, "Maximum l can't be larger than HEALPix resolution!"
        assert self.lmin>=2, "Minimum l can't be less than 2"
        if self.lmax>(base.lmax+1)*2/3: print("## Caution: Maximum l is greater than (2/3)*HEALPix-lmax; this might cause boundary effects.")
        
        # Compute filters for ell range of interest
        self.lfilt = (self.base.l_arr>=self.lmin)&(self.base.l_arr<=self.lmax)
        self.ls, self.ms = self.base.l_arr[self.lfilt], self.base.m_arr[self.lfilt]
        self.m_weight = np.asarray(self.base.m_weight[self.lfilt],order='C')
        self.Cinv = np.asarray(self.base.inv_Cl_tot_lm_mat[:,:,self.lfilt],order='C')
        
        # Define beam (in our ell-range)
        self.beam_lm = np.asarray(self.base.beam_lm)[:,self.lfilt]
        
        # Print ell ranges
        print("l-range: %s"%[self.lmin,self.lmax])
        
        # Print polarizations
        if self.pol:
            print("Polarizations: ['T', 'E']")
        else:
            print("Polarizations: ['T']")
        
        # Configure template parameters and limits
        self._configure_templates(templates, C_Tphi, C_lens_weight)
        
        # Check mask properties
        if not type(mask)==float or type(mask)==int:
            if len(mask)==1 or len(mask)==3:
                assert len(mask[0])==self.base.Npix, f'Mask has incorrect shape: {mask.shape}'
            else:
                assert len(mask)==self.base.Npix, f'Mask has incorrect shape: {mask.shape}'
        if np.std(self.mask)<1e-12 and np.abs(np.mean(self.mask)-1)<1e-12:
            print("Mask: ones")
            self.ones_mask = True
        else:
            print("Mask: spatially varying")
            self.ones_mask = False
        
        # Define fixed points for radial sampling
        if r_star is None:
            self.r_star = 13882.607764400758 # recombination distance (Planck 2018 cosmology)
        else:
            self.r_star = r_star
        if r_hor is None:
            self.r_hor = 14163.032903769747 # horizon distance (Planck 2018 cosmology)
        else:
            self.r_hor = r_hor
        
        # Check transfer function and initialize arrays
        if self.ints_1d:
            self.Tl_arr = np.asarray(Tl_arr,dtype=np.float64,order='C')
            if not self.pol:
                assert len(self.Tl_arr)==1, "Transfer function should contain only T components"
            else:
                assert len(self.Tl_arr)==2, "Transfer function should contain both T and E components"
            assert len(self.Tl_arr[0])>=self.lmax+1, "Transfer function must contain all ell modes of interest"
            assert self.Tl_arr[0][:2].sum()==0., "Transfer function should return zero for ell = 0, 1"
            assert len(self.Tl_arr[0][0])==len(k_arr), "Transfer function must be computed on the input k-grid"
            if self.pol: assert self.Tl_arr[1][:2].sum()==0., "Transfer function should return zero for ell = 0, 1"
        
        # Initialize timers
        self.timers = {x: 0. for x in ['precomputation','numerator','fisher','optimization','analytic_fisher',
                                       'fish_grfs', 'fish_outer', 'fish_deriv', 'fish_products',
                                       'Sinv','Ainv','map_transforms','fNL_summation','lensing_summation']}
        self.base.time_sht = 0.
        
        # Check sampling points
        if len(r_values)==0 and self.ints_1d:
            print("# No input radial sampling points supplied; these can be computed with the optimize_radial_sampling_1d() function\n") 
            self.N_r = 0
        elif self.ints_1d:
            print("Reading in precomputed radial integration points")
            
            assert len(r_values)>0, "Must supply radial sampling points!"
            for t in templates:
                if (t in self.all_templates_1d) or ('tauNL-direc' in t) or ('tauNL-even' in t) or ('tauNL-odd' in t):
                    assert t in r_weights.keys(), "Must supply weight for template %s"%t
            self.r_arr = r_values
            self.N_r = len(self.r_arr)
            self.r_weights = r_weights
        else:
            self.N_r = 0
        
        # Precompute k-space integrals if r arrays have been supplied
        if self.N_r>0:
            self._prepare_templates(self.ints_1d)
            
    ### UTILITY FUNCTIONS
    def _configure_templates(self, templates, C_Tphi, C_lens_weight):
        """Check input templates and log which quantities to compute."""
        
        # Check correct templates are being used and print them
        self.all_templates_1d = ['fNL-loc']
        self.all_templates = self.all_templates_1d+['isw-lensing']
        ii = 0
        for t in templates:
            ii += 1
            assert t in self.all_templates, "Unknown template %s supplied!"%t
        print("Templates: %s"%templates)
        
        def _merge_dict(d1,d2):
            """Merge dictionaries and drop duplicates"""
            for key in d2.keys():
                if key in d1.keys():
                    new_list = []
                    for item in d2[key]:
                        if item not in new_list: new_list.append(item)
                    for item in d1[key]:
                        if item not in new_list: new_list.append(item)
                    d1[key] = new_list
                else:
                    new_list = []
                    for item in d2[key]:
                        if item not in new_list: new_list.append(item)
                    d1[key] = new_list
        
        # Store list of quantities to compute
        self.ints_1d = False
        self.to_compute = []
        
        # Check which integrals to compute
        if 'fNL-loc' in templates:
            self.to_compute.append(['p','q'])
            self.ints_1d = True
        if 'isw-lensing' in templates:
            # Check inputs
            assert len(C_Tphi)>0, "Must supply temperature-lensing cross spectrum!"
            assert len(C_Tphi)>=self.lmax+1, "Must specify C^T-phi(L) up to at least Lmax."
            assert not self.pol, "ISW-lensing not implemented for polarization!"
            if not self.pol:
                assert 'TT' in C_lens_weight.keys(), "Must specify lensed TT power spectrum!"
            else:
                assert 'TE' in C_lens_weight.keys(), "Must specify lensed TE power spectrum!"
                assert 'EE' in C_lens_weight.keys(), "Must specify lensed EE power spectrum!"
                assert 'BB' in C_lens_weight.keys(), "Must specify lensed BB power spectrum!"
                for k in C_lens_weight.keys():
                    assert len(C_lens_weight[k])>=self.lmax+1, "Must specify C_lens_weight(l) up to at least lmax."
                    
            # Reshape and store
            self.C_Tphi = C_Tphi[:self.lmax+1]
            self.C_lens_weight = {k: C_lens_weight[k][:self.lmax+1] for k in C_lens_weight.keys()}
            self.to_compute.append(['u','v','v-isw'])
        
        # Identify unique components 
        self.to_compute = np.unique(np.concatenate(self.to_compute))
        
        # Create filtering for minimum ls
        self.lminfilt = self.base.l_arr[self.base.l_arr<=self.lmax]>=self.lmin
        
    def report_timings(self):
        """Report timings for various steps of the computation."""
        print("\n## Timings ##\n")
        
        print("Precomputation: %.2fs"%self.timers['precomputation'])
        if self.timers['numerator']!=0:
            print("Numerator: %.2fs"%self.timers['numerator'])
        if self.timers['fisher']!=0:
            print("Fisher: %.2fs"%self.timers['fisher'])
        if self.timers['optimization']!=0:
            print("Optimization: %.2fs"%self.timers['optimization'])
        
        print("\n# Timing Breakdown")
        if self.timers['Sinv']!=0:
            print("S^-1 filtering: %.2fs"%self.timers['Sinv'])
        if self.timers['map_transforms']!=0:
            print("1-field transforms: %.2fs"%self.timers['map_transforms'])
        if self.timers['numerator']!=0:
            if np.any([('fNL' in t) for t in self.templates]):
                print("fNL -- 3-field summation: %.2fs"%self.timers['fNL_summation'])
            if 'isw-lensing' in self.templates:
                print("Lensing -- 3-field summation: %.2fs"%self.timers['lensing_summation'])
        if (self.timers['fisher']!=0 or self.timers['optimization']!=0):
            if self.timers['analytic_fisher']!=0:
                print("Analytic Fisher Matrices: %.2fs"%self.timers['analytic_fisher'])
            if self.timers['fish_grfs']!=0:
                print("Fisher -- creating GRFs: %.2fs"%self.timers['fish_grfs'])
            if self.timers['Ainv']!=0:
                print("Fisher -- A^-1 filtering: %.2fs"%self.timers['Ainv'])
            print("Fisher -- 3-field derivatives: %.2fs"%self.timers['fish_deriv'])
            print("Outer product: %.2fs"%self.timers['fish_outer'])  
            
        print("\n## Harmonic Transforms ##")
        print("Forward: %d"%self.base.n_SHTs_forward)
        print("Reverse: %d"%self.base.n_SHTs_reverse)
        print("Time: %.2fs"%self.base.time_sht)
        print("\n")
        
    def reset_timings(self):
        """Reset all the timers to zero."""
        for f in self.timers.keys():
            self.timers[f] = 0.
        self.base.n_SHTs_forward = 0
        self.base.n_SHTs_reverse = 0
        self.base.time_sht = 0.
    
    def _timer_func(counter):
        """Decorator to compute the executation time of a function and add it to a counter."""
        def _timer_func_int(func): 
            def wrap_func(self,*args, **kwargs): 
                t1 = time.time() 
                result = func(self,*args, **kwargs)
                t2 = time.time() 
                self.timers[counter] += t2-t1
                return result 
            return wrap_func 
        return _timer_func_int
        
    @_timer_func('precomputation')
    def _prepare_templates(self, ints_1d=True):
        """Compute necessary k-integrals over the transfer functions for template estimation.

        This fills arrays such as plXs and qlXs arrays. Note that values outside the desired ell & field range will be set to zero.
        """
        # Print dimensions of k and r
        print("N_k: %d"%len(self.k_arr))
        if ints_1d: print("N_r: %d"%self.N_r)
        
        # Clear saved quantities, if necessary
        if hasattr(self, 't0_num'): delattr(self, 't0_num')
        if ints_1d:
            if hasattr(self, 'plXs'): delattr(self, 'plXs')
            if hasattr(self, 'qlXs'): delattr(self, 'qlXs')
        
        # Precompute all spherical Bessel functions on a regular grid
        print("Precomputing Bessel functions")
        max_kr = max(self.k_arr)*max(self.r_arr)
        
        x_arr = list(np.arange(0,self.lmax*2,0.01))+list(np.arange(self.lmax*2,min(max_kr*1.01,self.lmax*100),0.1))
        if max_kr>100*self.lmax:
            x_arr += list(np.linspace(self.lmax*100,max_kr*1.01,1000))
        x_arr = np.asarray(x_arr,dtype=np.float64)
        
        # Compute Bessel function in range of interest in Cython
        jlxs = np.zeros((self.lmax-self.lmin+1,len(x_arr)),dtype=np.float64,order='C')
        compute_bessel(x_arr,self.lmin,self.lmax,jlxs,self.base.nthreads)
        if np.isnan(jlxs).any(): raise Exception("Spherical Bessel calculation failed!")
        
        # Interpolate to the values of interest
        print("Interpolating Bessel functions")
        if ints_1d:
            jlkr = interpolate_jlkr(x_arr, self.k_arr, self.r_arr, jlxs, self.base.nthreads)

        # Set up arrays
        Pzeta_arr = self.Pzeta(self.k_arr)

        if 'q' in self.to_compute and ints_1d:
            
            # Compute q integrals in Cython
            print("Computing q_l^X(r) integrals")
            self.qlXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            q_integral(self.k_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.qlXs)
            
        if 'p' in self.to_compute and ints_1d:
            
            # Compute p integrals in Cython
            print("Computing p_l^X(r) integrals")
            self.plXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral(self.k_arr, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.plXs)
            
        if ints_1d: del jlkr
        
        # Define Cython utility class
        self.utils = fNL_utils(self.base.nthreads, self.N_r, self.base.l_arr.astype(np.int32),self.base.m_arr.astype(np.int32),
                                self.ls.astype(np.int32), self.ms.astype(np.int32))            
        
        print("Precomputation complete")
        
    ### MAP TRANSFORMATIONS
    @_timer_func('map_transforms')
    def _compute_weighted_maps(self, h_lm_filt, flX_arr, spin=0):
        """
        Compute [Sum_lm {}_sY_lm(i) f_l^X(i) h_lm^X] maps for each sampling point i, given the relevant weightings. These are used in the bispectrum numerators and Fisher matrices.
        """
        if not (hasattr(self,'r_arr') or hasattr(self,'rtau_arr')):
            raise Exception("Radial arrays have not been computed!")
        
        # Sum over polarizations (only filling non-zero elements)
        summ = np.zeros((len(flX_arr[0,0]),len(self.lminfilt)),order='C',dtype='complex128')
        summ[:,self.lminfilt] = self.utils.apply_fl_weights(flX_arr, h_lm_filt, 1.)
        
        # Compute SHTs 
        if spin!=0:
            return self.base.to_map_vec(summ, output_spin=spin, lmax=self.lmax)[0]
        else:
            return self.base.to_map_vec(summ, output_spin=spin, lmax=self.lmax)

    @_timer_func('map_transforms')
    def _compute_lensing_U_map(self, h_lm_filt):
        """
        Compute lensing U map from a given data vector. These are used in the ISW-lensing bispectrum numerators.
        
        The U^T map is also used in the point-source estimator. (If "isw-lensing" is not in self.to_compute, we only compute U^T.)
        
        We return [U^T, U^E, U^B].
        """
        
        # Output array
        U = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
        
        # Compute X = T piece for point-source + lensing estimation
        inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
        inp_lm[self.lminfilt] = h_lm_filt[0]
        U[0] = self.base.to_map(inp_lm[None],lmax=self.lmax)[0]
        
        # Compute X = E, B if implementing lensing estimators
        if 'isw-lensing' in self.templates:
            
            if self.pol:
            
                # Compute X = E piece
                inp_lm[self.lminfilt] = h_lm_filt[1]
                U[1] = self.base.to_map_spin(inp_lm, inp_lm, spin=2, lmax=self.lmax)[0]
                
                # Compute X = B piece
                inp_lm[self.lminfilt] = h_lm_filt[2]
                U[2] = self.base.to_map_spin(inp_lm, inp_lm, spin=2, lmax=self.lmax)[0]
          
        # Return output
        return U

    @_timer_func('map_transforms')
    def _compute_lensing_V_map(self, h_lm_filt, isw=False):
        """
        Compute lensing V map from a given data vector. These are used in the ISW-lensing bispectrum numerators. This can also compute the ISW-weighted fields.
        """
        
        # Output array
        V = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
            
        if not self.pol:
            if not isw:
                pref = np.sqrt(self.ls*(self.ls+1.))*self.C_lens_weight['TT'][self.ls]
                inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
                inp_lm[self.lminfilt] = pref*h_lm_filt[0]
                V[0] = self.base.to_map_spin(-inp_lm,inp_lm,spin=1,lmax=self.lmax)[1] # h_lm (-1)Y_lm
                del pref, inp_lm
            else:
                # Apply C_l^{Tphi} filtering for ISW maps
                pref = np.sqrt(self.ls*(self.ls+1.))*self.C_Tphi[self.ls]
                inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
                inp_lm[self.lminfilt] = pref*h_lm_filt[0]
                V[0] = self.base.to_map_spin(-inp_lm,inp_lm,spin=1,lmax=self.lmax)[1] # h_lm (-1)Y_lm
                del pref, inp_lm
        
        else:
            if isw: raise Exception("Not yet implemented!")
            
            # Output array
            V = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
            
            # Spin-0, X = T
            pref = np.sqrt(self.ls*(self.ls+1.))
            wienerT = (self.C_lens_weight['TT'][self.ls]*h_lm_filt[0]+self.C_lens_weight['TE'][self.ls]*h_lm_filt[1])
            inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
            inp_lm[self.lminfilt] = pref*wienerT
            V[0] = self.base.to_map_spin(-inp_lm,inp_lm,spin=1,lmax=self.lmax)[1] # h_lm (-1)Y_lm
            del inp_lm
            
            # Spin-2
            pref_p = np.sqrt((self.ls+2.)*(self.ls-1.))
            pref_m = np.sqrt((self.ls-2.)*(self.ls+3.))
            wienerE = (self.C_lens_weight['TE'][self.ls]*h_lm_filt[0]+self.C_lens_weight['EE'][self.ls]*h_lm_filt[1])
            wienerB = self.C_lens_weight['BB'][self.ls]*h_lm_filt[2]
            inp_lm_re = np.zeros(len(self.lminfilt),dtype=np.complex128)
            inp_lm_im = np.zeros(len(self.lminfilt),dtype=np.complex128)
            
            # X = E,B(+)
            inp_lm_re[self.lminfilt] = pref_p*wienerE
            inp_lm_im[self.lminfilt] = 1.0j*pref_p*wienerB
            V[1] = self.base.to_map_spin(inp_lm_re+inp_lm_im,-inp_lm_re+inp_lm_im,spin=1,lmax=self.lmax)[0] # (h^R_lm + i h^I_lm)(+1)Y_lm
            
            # X = E,B(-)
            inp_lm_re[self.lminfilt] = pref_m*wienerE
            inp_lm_im[self.lminfilt] = 1.0j*pref_m*wienerB
            V[2] = self.base.to_map_spin(inp_lm_re+inp_lm_im,-inp_lm_re+inp_lm_im,spin=3,lmax=self.lmax)[0] # (h^R_lm + i h^I_lm)(+3)Y_lm
            
            del pref_p, pref_m, inp_lm_re, inp_lm_im
            
        # Return output
        return V

    def _filter_pair(self, input_maps, filtering = 'Q'):
        """Compute the processed field with a given filtering for a pair of input maps."""
        
        if filtering=='P':
            return np.asarray([self._compute_weighted_maps(imap, self.plXs) for imap in input_maps],order='C')     
        
        elif filtering=='Q':
            return np.asarray([self._compute_weighted_maps(imap, self.qlXs) for imap in input_maps],order='C')     
        
        elif filtering=='U':
            return np.asarray([self._compute_lensing_U_map(imap) for imap in input_maps], order='C')        
            
        elif filtering=='V':
            return np.asarray([self._compute_lensing_V_map(imap, isw=False) for imap in input_maps], order='C')        
        
        elif filtering=='V-ISW':
            return np.asarray([self._compute_lensing_V_map(imap, isw=True) for imap in input_maps], order='C')        
        
        else:
            raise Exception("Filtering %s is not implemented!"%filtering)

    def _apply_all_filters(self, input_map):
        """Compute the processed fields with all relevant filterings for a single input map."""
        
        # Output array
        output = {}
        
        # Compute local maps
        if 'q' in self.to_compute:
            output['q'] = self._compute_weighted_maps(input_map, self.qlXs)
        
        if 'p' in self.to_compute:
            output['p'] = self._compute_weighted_maps(input_map, self.plXs)
              
        # Compute lensing maps
        if 'u' in self.to_compute:
            output['u'] = self._compute_lensing_U_map(input_map)        
            
        if 'v' in self.to_compute:
            output['v'] = self._compute_lensing_V_map(input_map, isw=False)
            
        if 'v-isw' in self.to_compute:
            output['v-isw'] = self._compute_lensing_V_map(input_map, isw=True)
        
        return output
    
    ### SIMULATION FUNCTIONS
    def _process_sim(self, sim, input_type='map'):
        """
        Process a single input simulation. This is used for the linear term of the bispectrum estimator.
        
        We return a set of weighted maps for this simulation (filtered by e.g. p_l^X).
        """
        # Transform to Fourier space and normalize appropriately
        t_init = time.time()
        h_sim_lm = np.asarray(self.applySinv(sim, input_type=input_type, lmax=self.lmax)[:,self.lminfilt],order='C')
        self.timers['Sinv'] += time.time()-t_init
        
        # Compute processed maps
        proc_maps = self._apply_all_filters(h_sim_lm)
        return proc_maps

    def load_sims(self, load_sim, N_sims, verb=False, preload=True, input_type='map'):
        """
        Load in and preprocess N_sim Monte Carlo simulations used in the linear term of the bispectrum estimator.

        The input is a function which loads the simulation in map- or harmonic-space given an index (0 to N_sims-1).

        If preload=False, the simulation products will not be stored in memory, but instead accessed when necessary. This greatly reduces memory usage, but is less CPU efficient if many datasets are analyzed together.
        
        These can alternatively be generated with a fiducial spectrum using the generate_sims script.
        """
        
        self.N_it = N_sims
        print("Using %d Monte Carlo simulations"%self.N_it)

        if preload:
            self.preload = True

            # Check we have initialized correctly
            if self.ints_1d and (not hasattr(self,'r_arr')):
                raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
            
            #  Define list of maps
            self.proc_maps = []

            # Iterate over simulations and preprocess appropriately    
            for ii in range(self.N_it):
                if ii%5==0 and verb: print("Processing bias simulation %d of %d"%(ii+1,self.N_it))

                # Load and process simulation
                self.proc_maps.append(self._process_sim(load_sim(ii), input_type=input_type))
        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")

            # Simply save iterator and continue (simulations will be processed in serial later) 
            self.load_sim_data = lambda ii: self._process_sim(load_sim(ii), input_type=input_type)
            
    def generate_sims(self, N_sims, Cl_input=[], preload=True, verb=False):
        """
        Generate Monte Carlo simulations used in the linear term of the bispectrum generator. 
        These are pure GRFs. By default, they are generated with the input survey mask.
        
        If preload=True, we create N_sims simulations and store the relevant transformations into memory.
        If preload=False, we store only the function used to generate the sims, which will be processed later. This is cheaper on memory, but less CPU efficient if many datasets are analyzed together.
        
        We can alternatively load custom simulations using the load_sims script.
        """

        self.N_it = N_sims
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        # Define input power spectrum (with noise)
        if len(Cl_input)==0:
            Cl_input = self.base.Cl_tot

        if preload:
            self.preload = True

            # Check we have initialized correctly
            if self.ints_1d and (not hasattr(self,'r_arr')):
                raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
            
            # Define lists of maps
            self.proc_maps = []
            
            # Iterate over simulations
            for ii in range(self.N_it):
                if ii%5==0 and verb: print("Generating bias simulation %d of %d"%(ii+1,self.N_it))
                
                # Generate simulation and compute P, Q maps
                if self.ones_mask:
                    sim_lm = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, output_type='harmonic', lmax=self.lmax, deconvolve_beam=False)
                    self.proc_maps.append(self._process_sim(sim_lm, input_type='harmonic'))
                else:
                    sim = self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False)
                    self.proc_maps.append(self._process_sim(sim))

        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            if self.ones_mask:
                self.load_sim_data = lambda ii: self._process_sim(self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, output_type='harmonic', lmax=self.lmax, deconvolve_beam=False), input_type='harmonic')
            else:
                self.load_sim_data = lambda ii: self._process_sim(self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False))
    
    ### FISHER MATRIX FUNCTIONS
    @_timer_func('fish_outer')
    def _assemble_fish(self, Q3_a, Q3_b, sym=False):
        """Compute Fisher matrix between two Q arrays as an outer product. This is parallelized across the l,m axis."""
        return self.utils.outer_product(Q3_a, Q3_b, sym)

    def _weight_Q_maps(self, tmp_Q, weighting='Ainv'):
        """Apply inplace weighting to a Q map to form output array. This includes factors of S^-1.P if necessary."""
        
        for index in range(len(self.templates)):
            if weighting=='Ainv':
                # Construct l-space map down to l=0
                full_Q = np.zeros((1+2*self.pol,len(self.lminfilt)),dtype=np.complex128)
                full_Q[:,self.lminfilt] = self.beam_lm*tmp_Q[index]
                # Compute S^-1.P.Q
                t_init = time.time()
                if self.ones_mask:
                    tmp_Q[index] = self.applySinv(full_Q,input_type='harmonic', lmax=self.lmax)[:,self.lminfilt]
                else:
                    tmp_Q[index] = self.applySinv(self.mask*self.base.to_map(full_Q,lmax=self.lmax), lmax=self.lmax)[:,self.lminfilt]
                self.timers['Sinv'] += time.time()-t_init
            elif weighting=='Sinv':
                tmp_Q[index] = self.m_weight*tmp_Q[index]   

    @_timer_func('fish_deriv')
    def _transform_maps(self, map12, flXs, weights, spin=0):
        """Compute Sum_i w_i M_LM f^X_L(i) for real-space map M(n). We optionally average over spins."""
        output = np.zeros((1+2*self.pol,np.sum(self.lfilt)),dtype='complex')
        if spin==0:
            lm_map = np.asarray(self.base.to_lm_vec(map12,lmax=self.lmax)[:,self.lminfilt],order='C')
            return self.utils.radial_sum(lm_map, weights, flXs)
            # return np.sum(self.base.to_lm_vec(map12,lmax=self.lmax).T[self.lminfilt,None,:]*flXs*weights,axis=2).T
        elif spin==1:
            lm_map = np.asarray(self.base.to_lm_vec([map12,map12.conj()],spin=1,lmax=self.lmax)[:,:,self.lminfilt],order='C')
            return self.utils.radial_sum_spin1(lm_map, weights, flXs)
            # return 0.5*np.sum((np.array([1,-1])[:,None,None]*self.base.to_lm_vec([map123,map123.conj()],spin=1,lmax=self.lmax)).sum(axis=0).T[self.lminfilt,None,:]*flXs*weights,axis=2).T
        else:
            raise Exception(f"Wrong spin s = {spin}!")

    def _compute_fisher_derivatives(self, templates, N_fish_optim=None, verb=False):
        """Compute the derivative of the ideal Fisher matrix with respect to the weights for each template of interest."""

        # Output array
        output = {}
        
        # Compute arrays
        # NB: using exact Gauss-Legendre integration in mu
        [mus, w_mus] = p_roots(2*self.lmax+1)
        ls = np.arange(self.lmin,self.lmax+1)
        legs = np.asarray([lpmn(0,self.lmax,mus[i])[0][0,self.lmin:] for i in range(len(mus))])
            
        t_init = time.time()
        for template in templates:
                
            if template=='fNL-loc':
                if verb: print("\tComputing fNL-loc Fisher matrix derivative exactly")
                deriv_matrix = np.asarray(fisher_deriv_fNL_loc(self.plXs, self.qlXs, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                
            else:
                raise Exception("Template %s not implemented!"%template)
                
            output[template] = np.sum(deriv_matrix), deriv_matrix

        self.timers['analytic_fisher'] += time.time()-t_init

        return output

    ### NUMERATOR
    @_timer_func('numerator')
    def Bl_numerator(self, data, include_linear_term=True, verb=False, input_type='map'):
        """
        Compute the numerator of the quasi-optimal bispectrum estimator for all templates.

        We optionally include the linear terms, which can reduce the estimator variance.
        """
        # Check we have initialized correctly
        if self.ints_1d and (not hasattr(self,'r_arr')):
            raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
        
        # Check if simulations have been supplied
        if not hasattr(self, 'preload') and include_linear_term:
            raise Exception("Need to generate or specify bias simulations!")

        # Check input data format
        if self.pol:
            assert len(data)==3, "Data must contain T, Q, U components!"
        else:
            assert (len(data)==1 and len(data[0])==self.base.Npix) or len(data)==self.base.Npix, "Data must contain T only!"

        # Apply S^-1 to data and transform to harmonic space
        t_init = time.time()
        h_data_lm = np.asarray(self.applySinv(data, input_type=input_type, lmax=self.lmax)[:,self.lminfilt], order='C')
        self.timers['Sinv'] += time.time()-t_init
           
        # Compute all relevant weighted maps
        proc_maps = self._apply_all_filters(h_data_lm)
        
        # Define 3- and 1-field arrays
        b3_num = np.zeros(len(self.templates))
        if include_linear_term:
            b1_num = np.zeros(len(self.templates))
            
        if verb: print("# Assembling bispectrum numerator (3-field term)")
        for ii,t in enumerate(self.templates):
            
            if t=='fNL-loc':
                # fNL-local template
                print("Computing fNL template")
                
                t_init = time.time()
                b3_num[ii] = 3./5.*self.utils.fnl_loc_sum(self.r_weights[t], proc_maps['p'], proc_maps['p'], proc_maps['q'])*self.base.A_pix
                self.timers['fNL_summation'] += time.time()-t_init

            elif t=='isw-lensing':
                # ISW-Lensing template
                print("Computing ISW-lensing template")
                
                t_init = time.time()
                b3_num[ii] = np.sum(proc_maps['u'][0]*proc_maps['v'][0]*proc_maps['v-isw'][0].conjugate()).real*self.base.A_pix
                self.timers['lensing_summation'] += time.time()-t_init
               
        if include_linear_term:

            # Iterate over simulations
            for isim in range(self.N_it):
                if verb: print("# Assembling bispectrum linear term for simulation %d of %d"%(isim+1,self.N_it))

                # Load processed bias simulations
                if self.preload:
                    this_proc_maps = self.proc_maps[isim]
                else:
                    this_proc_maps = self.load_sim_data(isim)

                # Compute templates
                for ii,t in enumerate(self.templates):
                    if t=='fNL-loc':
                        t_init = time.time()
                        
                        # Sum over permutations
                        summ  = 2.*self.utils.fnl_loc_sum(self.r_weights[t], proc_maps['p'], this_proc_maps['p'], this_proc_maps['q'])
                        summ += self.utils.fnl_loc_sum(self.r_weights[t], this_proc_maps['p'], this_proc_maps['p'], proc_maps['q'])
                        b1_num[ii] += -3./5.*summ*self.base.A_pix/self.N_it
                        self.timers['fNL_summation'] += time.time()-t_init
                        
                    if t=='isw-lensing':
                        t_init = time.time()
               
                        # Sum over 3 permutations
                        summ  = np.sum(proc_maps['u'][0]*this_proc_maps['v'][0]*this_proc_maps['v-isw'][0].conjugate()).real
                        summ += np.sum(this_proc_maps['u'][0]*proc_maps['v'][0]*this_proc_maps['v-isw'][0].conjugate()).real
                        summ += np.sum(this_proc_maps['u'][0]*this_proc_maps['v'][0]*proc_maps['v-isw'][0].conjugate()).real
                        b1_num[ii] += -summ*self.base.A_pix/self.N_it
                        self.timers['lensing_summation'] += time.time()-t_init
                                            
        if include_linear_term:
            b_num = b3_num+b1_num
        else:
            b_num = b3_num

        return b_num

    ### OPTIMIZATION
    @_timer_func('optimization')
    def optimize_radial_sampling_1d(self, reduce_r=1, tolerance=1e-3, N_split=None, split_index=None, initial_r_points=None, verb=False):
        """
        Compute the 1D radial sampling points and weights via optimization (as in Smith & Zaldarriaga 06), up to some tolerance in the Fisher distance.
        Optimization will be done for each template, analytically computing the 'distance' between template approximations
        
        Main Inputs:
            - reduce_r: Downsample the number of points in the starting radial integral grid (default: 1)
            - tolerance: Convergence threshold for the optimization (default 1e-3). This indicates the approximate error in the Fisher matrix induced by the optimization.
            
        For large problems, it is too expensive to optimize the whole matrix at once. Instead, we can split the optimization into N_split pieces, each of which is optimized separately.
        Following this, we perform a final optimization of all N_split components, using the union of all previously obtained radial points. 
        
        Additional Inputs (for chunked computations):
            - N_split (optional): Number of chunks to split the optimization into. If None, no splitting is performed.
            - split_index (optional): Index of the chunk to optimize. 
            - initial_r_points (optional): Starting set of radial points (used for the final optimization step).
        
        """
        assert self.ints_1d, "No 1D optimization is required for these templates!"
        t_init = time.time()

        # Check precision parameters
        assert reduce_r>0, "reduce_r parameter must be positive"
        
        if reduce_r<0.5:
            print("## Caution: very dense r-sampling requested; computation may be very slow") 
        if reduce_r>3:
            print("## Caution: very sparse r-sampling requested; computation may be inaccurate")
        
        # Create radial array
        r_raw = np.asarray(list(np.arange(1,self.r_star*0.95,50*reduce_r))+list(np.arange(self.r_star*0.95,self.r_hor*1.05,5*reduce_r))+list(np.arange(self.r_hor*1.05,self.r_hor+5000,50*reduce_r)))
        r_init = 0.5*(r_raw[1:]+r_raw[:-1])
        self.quad_weights_1d = r_init**2*np.diff(r_raw)
        r_weights = {}
        
        # Partition the radial indices if required or read in precomputed points
        if initial_r_points is not None:
            assert split_index is None, "Cannot specify both initial_r_points and index_split"
            assert len(initial_r_points)==len(np.unique(initial_r_points)), "initial_r_points cannot contain repeated points"
            inds = np.asarray([np.where(r==r_init)[0][0] for r in initial_r_points])
            r_init = r_init[inds]
            self.quad_weights_1d = self.quad_weights_1d[inds]
        else:
            if N_split is not None:
                print("Partitioning sampling grid into %d pieces"%(N_split))                
                r_init = r_init[split_index::N_split]
                self.quad_weights_1d = self.quad_weights_1d[split_index::N_split]
        
        # Precompute k-integrals with initial r grid
        self.r_arr = r_init
        self.N_r = len(r_init)
        print("# Computing k integrals with fiducial radial grid")
        self._prepare_templates(ints_1d=True)
        
        # Reorder templates (keeping only those that require 1D optimization)
        ordered_templates = [tem for tem in self.templates if tem in self.all_templates_1d]
        
        # Create list of radial indices in the optimized representation
        inds = []
        inds_init = np.arange(self.N_r)
        
        # Compute all Fisher matrix derivatives of interest
        if verb: print("Computing all Fisher matrix derivatives")
        derivs = self._compute_fisher_derivatives(ordered_templates, verb=verb)
        
        # Save ideal Fisher matrices
        if not hasattr(self, 'ideal_fisher'):
            self.ideal_fisher = {}
        for t in ordered_templates:
            self.ideal_fisher[t] = derivs[t][0]
        
        for template in ordered_templates:
            
            if verb: print("\nRunning optimization for template %s"%template)
            
            # Compute Fisher matrix derivative
            init_score, deriv_matrix = derivs[template][0], derivs[template][1]
            if verb: print("Initial score: %.2e"%init_score)
            
            def _compute_score(w_vals, full_score=False):
                """
                Compute the Fisher distance between templates given weights w_vals. This optionally computes the gradients.
                """
                if full_score:
                    return np.sum(G_mat), np.sum(np.outer(w_vals,w_vals)*deriv_matrix[inds][:,inds])
                else:
                    return np.sum(G_mat)
                
            def _test_inds(inds, score_old, w_vals):
                """Test the current set of indices"""
                score = _compute_score(w_vals)
                return score, w_vals
            
            # Check zeroth iteration
            if len(inds)!=0:
                # Compute quadratic weights
                notinds = [i for i in np.arange(self.N_r) if i not in inds]
                inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
                G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]
                w_vals = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))
                
                # Compute score
                score, w_vals = _test_inds(inds, init_score, w_vals)
                if verb: print("Unoptimized relative score: %.2e"%(score/init_score))
            else:
                score = init_score
                w_vals = []
                
            # Set up iteration
            if score/init_score >= tolerance:
                
                # Define starting indices
                if len(inds)==0:
                    next_ind = np.argsort(np.sum(deriv_matrix,axis=1)**2/np.diag(deriv_matrix))[-1]
                else:
                    next_ind = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[-1]]
                inds.append(next_ind)
                print(inds)
                
                # Set-up memory
                w_vals_old = w_vals
                score_old = score
                    
                # Iterate until convergence
                for iit,iteration in enumerate(range(len(inds),self.N_r)):
                    
                    # Define indices  
                    inds[-1] = next_ind
                    notinds = [i for i in np.arange(self.N_r) if i not in inds]
                    
                    # Set-up weights
                    inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
                    G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]
                    
                    # Compute optimal quadratic weights
                    w_vals = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))
                      
                    # Compute score
                    score, w_vals = _test_inds(inds, score_old, w_vals)
                    if verb: print("Iteration %d, relative score: %.2e, old score: %.2e"%(iteration, score/init_score, score_old/init_score))
                    
                    # Check for numerical errors
                    if score<0:
                        print("## Score is negative; this indicates a numerical error!")
                        break
                    
                    # Finish if converged
                    if score/init_score < tolerance:
                        break
                    
                    # Update memory when score is accepted
                    w_vals_old = w_vals
                    score_old = score
                    
                    # Compute indices for next iteration
                    next_ind = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[-1]]
                    inds.append(next_ind)
                    
            if len(G_mat)==0:
                raise Exception("Failed to converge after %d iterations; this indicates a bug!"%N_fish_optim)
                
            if verb: print("\nScore threshold met with %d indices"%len(inds))
            w_opt = np.asarray(w_vals.copy())
            
            # Check final Fisher matrix
            score, fish = _compute_score(w_opt, full_score=True)
            if verb: print("Ideal %s Fisher: %.4e (initial), %.4e (optimized). Relative score: %.2e\n"%(template, init_score, fish, score/init_score))

            # Store attributes
            r_weights[template] = w_opt*self.quad_weights_1d[inds]
        
        # Store attributes
        self.r_arr = r_init[inds]
        self.N_r = len(self.r_arr)
        self.r_weights = {}
        for template in ordered_templates:
            # add weights, padding with zeros
            self.r_weights[template] = np.zeros(len(w_opt)) 
            self.r_weights[template][:len(r_weights[template])] = r_weights[template]
        
        # Precompute k-space integrals with new radial integration
        if verb: print("Computing k integrals with optimized radial grid")
        self._prepare_templates(ints_1d=True)

        print("\nOptimization complete after %.2f seconds"%(time.time()-t_init))
        
        return self.r_arr, self.r_weights
        
    ### NORMALIZATION
    @_timer_func('fisher')
    def compute_fisher_contribution(self, seed, verb=False):
        """
        This computes the contribution to the Fisher matrix from a single pair of GRF simulations, created internally.
        """
        # Check we have initialized correctly
        if self.ints_1d and (not hasattr(self,'r_arr')):
            raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
        
        print("Computing Fisher matrix with seed %d"%seed)
        
        # Initialize output
        fish = np.zeros((len(self.templates),len(self.templates)),dtype='complex')

        # Compute two random realizations with known power spectrum, removing the beam
        if verb: print("# Generating GRFs")
        t_init = time.time()
        a_maps = []
        for ii in range(2):
            if self.ones_mask:
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e9), output_type='harmonic',lmax=self.lmax, deconvolve_beam=True))
            else:
                # we can't truncate to l<=lmax here, since we need to apply the mask!
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e9), output_type='harmonic', deconvolve_beam=True))
        self.timers['fish_grfs'] += time.time()-t_init
        
        # Define Q map code
        def compute_Q3(weighting):
            """
            Assemble and return an array of Q3 maps in real- or harmonic-space, for S^-1 or A^-1 weighting. 

            Schematically Q ~ (A[x]B[y,z]_lm + perms., and we dynamically compute each permutation of (A B)_lm.

            The outputs are either Q_i or S^-1.P.Q_i.
            """
            # Weight maps by S^-1.P or A^-1
            if verb: print("Weighting maps")
            
            t_init = time.time()
            if weighting=='Sinv':
                # Compute S^-1.P.a
                if self.ones_mask:
                    Uinv_a_lms = [np.asarray(self.applySinv(self.base.beam_lm[:,self.base.l_arr<=self.lmax]*a_lm, input_type='harmonic', lmax=self.lmax)[:,self.lminfilt],order='C') for a_lm in a_maps]
                else:
                    Uinv_a_lms = [np.asarray(self.applySinv(self.mask*self.base.to_map(self.base.beam_lm*a_lm), lmax=self.lmax)[:,self.lminfilt],order='C') for a_lm in a_maps]
                self.timers['Sinv'] += time.time()-t_init
            elif weighting=='Ainv':
                # Compute A^-1.a
                if self.ones_mask:
                    Uinv_a_lms = [np.asarray(self.base.applyAinv(a_lm, input_type='harmonic', lmax=self.lmax)[:,self.lminfilt],order='C') for a_lm in a_maps]
                else:
                    Uinv_a_lms = [np.asarray(self.base.applyAinv(a_lm, input_type='harmonic')[:,self.lfilt],order='C') for a_lm in a_maps]
                self.timers['Ainv'] += time.time()-t_init 
            
            # Filter maps
            if verb: print("Computing filtered maps")
            if 'q' in self.to_compute:
                if verb: print("Creating Q maps")
                Q_maps = self._filter_pair(Uinv_a_lms, 'Q')   
            if 'p' in self.to_compute:
                if verb: print("Creating P maps")
                P_maps = self._filter_pair(Uinv_a_lms, 'P')   
            if 'u' in self.to_compute:
                if verb: print("Creating U maps")
                U_maps = self._filter_pair(Uinv_a_lms, 'U')
            if 'v' in self.to_compute:
                if verb: print("Creating V maps")
                V_maps = self._filter_pair(Uinv_a_lms, 'V')
            if 'v-isw' in self.to_compute:
                if verb: print("Creating ISW V maps")
                V_isw_maps = self._filter_pair(Uinv_a_lms, 'V-ISW')
            
            # Define output arrays (Q11, Q22)
            Qs = np.zeros((2,len(self.templates),1+2*self.pol,np.sum(self.lfilt)),dtype=np.complex128,order='C')
            
            # Compute products (with symmetries)
            for ii,t in enumerate(self.templates):
                if t=='fNL-loc':
                    
                    if verb: print("Computing Q-derivative for fNL-loc")

                    # 11
                    Qs[0,ii]  = 12./5.*self._transform_maps(self.utils.multiply(P_maps[0],Q_maps[0]),self.plXs,self.r_weights[t])
                    Qs[0,ii] += 6./5.*self._transform_maps(self.utils.multiply(P_maps[0],P_maps[0]),self.qlXs,self.r_weights[t])
                    
                    # 22
                    Qs[1,ii]  = 12./5.*self._transform_maps(self.utils.multiply(P_maps[1],Q_maps[1]),self.plXs,self.r_weights[t])
                    Qs[1,ii] += 6./5.*self._transform_maps(self.utils.multiply(P_maps[1],P_maps[1]),self.qlXs,self.r_weights[t])
                    
                if t=='isw-lensing':
                    if verb: print("Computing Q-derivative for isw-lensing")
                    
                    assert not self.pol
                    
                    pref1 = self.C_Tphi[self.ls]*np.sqrt(self.ls*(self.ls+1.))
                    pref2 = self.C_lens_weight['TT'][self.ls]*np.sqrt(self.ls*(self.ls+1.))
                    
                    # First term
                    input_map = np.real(V_maps[0]*V_isw_maps[0].conjugate())
                    Qs[0,ii,0] = 2*self.base.to_lm(input_map[None],lmax=self.lmax)[0,self.lminfilt]
                    input_map = np.real(V_maps[1]*V_isw_maps[1].conjugate())
                    Qs[1,ii,0] = 2*self.base.to_lm(input_map[None],lmax=self.lmax)[0,self.lminfilt]
                    
                    # Second term
                    input_map = U_maps[0]*V_maps[0] 
                    Qs[0,ii,0] += pref1*np.sum(np.array([-1,1])[:,None]*self.base.to_lm_spin(input_map.conj(), input_map,spin=1,lmax=self.lmax)[:,self.lminfilt],axis=0)
                    input_map = U_maps[1]*V_maps[1] 
                    Qs[1,ii,0] += pref1*np.sum(np.array([-1,1])[:,None]*self.base.to_lm_spin(input_map.conj(), input_map,spin=1,lmax=self.lmax)[:,self.lminfilt],axis=0)
                    
                    # Third term
                    input_map = U_maps[0]*V_isw_maps[0] 
                    Qs[0,ii,0] += pref2*np.sum(np.array([-1,1])[:,None]*self.base.to_lm_spin(input_map.conj(), input_map,spin=1,lmax=self.lmax)[:,self.lminfilt],axis=0)
                    input_map = U_maps[1]*V_isw_maps[1] 
                    Qs[1,ii,0] += pref2*np.sum(np.array([-1,1])[:,None]*self.base.to_lm_spin(input_map.conj(), input_map,spin=1,lmax=self.lmax)[:,self.lminfilt],axis=0)
            
            if weighting=='Ainv' and verb: print("Applying S^-1 weighting to output")
            for qindex in range(2):
                self._weight_Q_maps(Qs[qindex], weighting)
            return Qs.reshape(2,len(self.templates),-1)

        # Compute Q3 maps
        if verb: print("\n# Computing Q3 map for S^-1 weighting")
        Q3_Sinv = compute_Q3('Sinv')
        if verb: print("\n# Computing Q3 map for A^-1 weighting")
        Q3_Ainv = compute_Q3('Ainv')

        # Assemble Fisher matrix
        if verb: print("\n# Assembling Fisher matrix\n")

        # Compute Fisher matrix as an outer product
        fish = self._assemble_fish(Q3_Sinv, Q3_Ainv, sym=False)
        if verb: print("\n# Fisher matrix contribution %d computed successfully!"%seed)
        
        return fish
        
    def compute_fisher(self, N_it, verb=False):
        """
        Compute the Fisher matrix using N_it realizations. These are run in serial (since the code is already parallelized).
        
        For high-dimensional problems, it is usually preferred to split the computation across a cluster with MPI, calling compute_fisher_contribution for each instead of this function.
        """
        # Initialize output
        fish = np.zeros((len(self.templates),len(self.templates)))
        
        # Iterate over N_it seeds
        for seed in range(N_it):
            print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
            fish += self.compute_fisher_contribution(seed, verb=verb*(seed==0))/N_it
        
        # Store matrix in class attributes
        self.fish = fish
        self.inv_fish = np.linalg.inv(fish)

        return fish
    
    ### WRAPPER
    def Bl_full(self, data, fish=[], include_linear_term=True, verb=False, input_type='map'):
        """
        Compute the quasi-optimal bispectrum estimator. This is a wrapper of Bl_numerator, including the Fisher matrix multiplication.
        
        The code either uses pre-computed Fisher matrices or reads them in on input. 
        
        We can also optionally switch off the linear terms.
        """
        if verb: print("")

        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)

        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix first!")
        
        # Compute numerator
        Bl_num = self.Bl_numerator(data, verb=verb, include_linear_term=include_linear_term, input_type=input_type)

        # Apply normalization
        Bl_out = np.matmul(self.inv_fish,Bl_num)

        # Create output dictionary
        Bl_dict = {}
        index = 0
        # Iterate over fields
        for t in self.templates:
            Bl_dict[t] = Bl_out[index]
            index += 1
            
        return Bl_dict