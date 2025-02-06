### Code for ideal and unwindowed binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2025)
## This module contains the trispectrum template estimation code

import numpy as np
import time
import multiprocessing as mp
import tqdm
from scipy.interpolate import interp1d
import pywigxjpf as wig
from scipy.integrate import trapezoid
from scipy.special import loggamma, gamma, hyp2f1, spherical_jn, p_roots, lpmn
from scipy.optimize import minimize
from scipy.linalg import pinv
from .cython.k_integrals import *
from .cython.tauNL_utils import *
from .cython.ideal_fisher import *
        
class TSpecTemplate():
    """
    Trispectrum estimation class for measuring the amplitudes of separable primoridal trispectrum templates. 
    We also feed in a function that applies the S^-1 operator (which is ideally beam.mask.C_l^{tot,-1}, where C_l^tot includes the beam and noise). 
    
    Inputs:
    - base: PolySpec class
    - mask: HEALPix mask applied to data. We can optionally specify a vector of three masks for [T, Q, U].
    - applySinv: function which returns S^-1 ~ P^dag Cov^{-1} in harmonic space, when applied to a given input map, where P = Mask * Beam.
    - templates: types of templates to compute e.g. [gNL-loc, tauNL-loc]
    - k_arr, Tl_arr: k-array, plus T- and (optionally) E-mode transfer functions for all ell
    - lmin, lmax: minimum/maximum ell (inclusive)
    - Lmin, Lmax: minimum/maximum internal L (inclusive)
    - Lmin_lens, Lmax_lens: minimum/maximum internal L for lensing (inclusive)
    - ns, As, k_pivot: primordial power spectrum parameters. 
    - r_values, r_weights: radial sampling points and weights for 1-dimensional integrals
    - rtau_values, rtau_weights: list of (r,tau) sampling points for 2-dimensional EFTI integrals
    - weights: weights. These should be a dictionary for each template in question. For integrals involving tau or kappa, these should be two-dimensional.
    - C_phi: lensing power spectrum [C^phiphi_0, C^phiphi_1, etc.]. Required if 'lensing' is in templates.
    - C_lens_weight: dictionary of lensed power spectra (TT, TE, etc.). Required if 'lensing' is in templates.
    - K_coll, k_coll: cut-off scale for collider templates (restricting to k > k_coll, K < K_coll (default: 0.01, 0.01).
    - r_star, r_hor: Comoving distance to last-scattering and the horizon (default: Planck 2018 values).
    """
    def __init__(self, base, mask, applySinv, templates, k_arr, Tl_arr, lmin, lmax,  
                 Lmin=None, Lmax=None, Lmin_lens=None, Lmax_lens=None, ns=0.96, As=2.1e-9, k_pivot=0.05, 
                 r_values = [], r_weights = {}, rtau_values = [], rtau_weights = {}, 
                 C_phi=[], C_lens_weight = {}, K_coll=0.1, k_coll=0.1, r_star=None, r_hor=None):
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
        self.K_coll = K_coll
        self.k_coll = k_coll 
        
        # Define Gaunt factor and initialize 3js and 6js
        wig.wig_table_init(2*self.base.lmax,3)
        wig.wig_temp_init(2*self.base.lmax)
        self.wig3 = lambda l1,l2,l3,m1,m2,m3: wig.wig3jj(l1*2,l2*2,l3*2,m1*2,m2*2,m3*2)
        self.wig3zero = lambda l1,l2,l3: wig.wig3jj(l1*2,l2*2,l3*2,0,0,0)
        self.wig6 = lambda l1,l2,l3,l4,l5,l6: wig.wig6jj(l1*2,l2*2,l3*2,l4*2,l5*2,l6*2)
        def _gaunt(l1,l2,l3,m1,m2,m3): 
            if l1<0 or l2<0 or l3<0:
                return np.nan
            else:
                return np.sqrt((2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)/(4.*np.pi))*wig.wig3jj(2*l1,2*l2,2*l3,2*m1,2*m2,2*m3)*wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)
        self.gaunt = np.vectorize(_gaunt)
        
        # Create primordial power spectrum function
        print("Primordial Spectrum: n_s = %.3f, A_s = %.3e, k_pivot = %.3f"%(self.ns, self.As, self.k_pivot));
        self.Pzeta = lambda k: 2.*np.pi**2./k**3*self.As*(k/self.k_pivot)**(self.ns-1)
        
        # Check ell ranges
        assert self.lmax<=base.lmax, "Maximum l can't be larger than HEALPix resolution!"
        assert self.lmin>=2, "Minimum l can't be less than 2"
        if self.lmax>(base.lmax+1)*2/3: print("## Caution: Maximum l is greater than (2/3)*HEALPix-lmax; this might cause boundary effects.")
        
        # Check L ranges
        if Lmin==None: 
            self.Lmin = 1
        else:
            self.Lmin = Lmin
        if Lmax==None: 
            self.Lmax = min([base.lmax,2*self.lmax])
        else:
            self.Lmax = Lmax
        assert self.Lmin>=1, "Minimum L can't be less than 1"
        assert self.Lmax<=2*self.lmax, "Lmax <= 2*lmax by the triangle conditions"
        assert self.Lmax<=base.lmax, "Maximum L can't be larger than HEALPix resolution"
        
        # Check lensing L ranges
        if 'lensing' in self.templates:
            if Lmin_lens==None:
                if np.any(['gNL' in t for t in self.templates]):
                    self.Lmin_lens = 1
                elif np.any(['tauNL' in t for t in self.templates]) :
                    self.Lmin_lens = self.Lmin # only these modes are used!
                else:
                    self.Lmin_lens = 1
            else:
                self.Lmin_lens = Lmin_lens
            if Lmax_lens==None:
                if np.any(['gNL' in t for t in self.templates]):
                    self.Lmax_lens = min([base.lmax,2*self.lmax])
                elif np.any(['tauNL' in t for t in self.templates]) :
                    self.Lmax_lens = self.Lmax # only these modes are used!
                else:
                    self.Lmax_lens = min([base.lmax,2*self.lmax])
            else:
                self.Lmax_lens = Lmax_lens
            assert self.Lmin_lens>=1, "Minimum L_lens can't be less than 1"
            assert self.Lmax_lens<=2*self.lmax, "Lmax_lens <= 2*lmax by the triangle conditions"
            assert self.Lmax_lens<=base.lmax, "Maximum L can't be larger than HEALPix resolution"
            
        # Compute filters for ell range of interest
        self.lfilt = (self.base.l_arr>=self.lmin)&(self.base.l_arr<=self.lmax)
        self.Lfilt = (self.base.l_arr>=self.Lmin)&(self.base.l_arr<=self.Lmax)
        self.ls, self.ms = self.base.l_arr[self.lfilt], self.base.m_arr[self.lfilt]
        if 'lensing' in self.templates:
            self.Lfilt_lens = (self.base.l_arr>=self.Lmin_lens)&(self.base.l_arr<=self.Lmax_lens)
        self.m_weight = np.asarray(self.base.m_weight[self.lfilt],order='C')
        self.Cinv = np.asarray(self.base.inv_Cl_tot_lm_mat[:,:,self.lfilt],order='C')
                
        # Define beam (in our ell-range)
        self.beam_lm = np.asarray(self.base.beam_lm)[:,self.lfilt]
        
        # Print ell ranges
        print("l-range: %s"%[self.lmin,self.lmax])
        if np.any(['tauNL' in template for template in templates]):
            print("L-range: %s"%[self.Lmin,self.Lmax])
        if 'lensing' in self.templates:
            print("Lensing L-range: %s"%[self.Lmin_lens,self.Lmax_lens])
        
        # Check correct polarizations are being used
        if self.pol:
            print("Polarizations: ['T', 'E']")
        else:
            print("Polarizations: ['T']")
        
        # Configure template parameters and limits
        self._configure_templates(templates, C_phi, C_lens_weight)
    
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
            
        # Check collider parameters
        if self.ints_coll:
            print("Collider k-range: K < %.2e, k > %.2e"%(self.K_coll,self.k_coll))
            assert self.K_coll <= 2*np.max(self.k_arr), "Collider K-cut can't exceed K_max (= %.3e)"%(2*np.max(self.k_arr))
            assert self.K_coll >= 0, "Collider K-cut can't be below 0"
            assert self.k_coll <= np.max(self.k_arr), "Collider k-cut can't exceed k_max (= %.3e)"%np.max(self.k_arr)
            assert self.k_coll >= np.min(self.k_arr), "Collider k-cut can't be below k_min (= %.3e)"%np.min(self.k_arr)
            assert self.K_coll <= self.k_coll, "Collider K_cut can't exceed collider k_cut"
            
        # Check transfer function and initialize arrays
        if self.ints_coll or self.ints_1d or self.ints_2d:
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
                                       'fish_grfs', 'fish_outer', 'fish_weighting', 'fish_convolve', 'fish_deriv', 'fish_products',
                                       'Sinv','Ainv','map_transforms','gNL_summation',
                                       'tauNL_products','tauNL_summation',
                                       'lensing_products','lensing_summation']}
        self.base.time_sht = 0.
        
        # Check sampling points
        if len(r_values)==0 and self.ints_1d: # ints_coll always requires ints_1d!
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
        
        if len(rtau_values)==0 and self.ints_2d:
            print("# No input r/tau sampling points supplied; these can be computed with the optimize_radial_sampling_2d() function\n")
            self.N_rtau = 0. 
        elif self.ints_2d:
            print("Reading in precomputed r/tau integration points")
          
            assert (len(rtau_values)>0) and (len(rtau_values[0])==2), "Must supply r/tau sampling points for 2d integrals shapes!"
            for t in templates:
                if t in self.all_templates_2d: assert t in rtau_weights.keys(), "Must supply weight for template %s"%t
            self.rtau_arr = rtau_values
            self.N_rtau = len(self.rtau_arr)

            # Check bounds
            assert max(self.rtau_arr[:,0])>=0, "r should be in range (0, inf)"
            assert max(self.rtau_arr[:,1])<=0, "tau should be in range (-inf, 0)"

            self.rtau_weights = rtau_weights
        else:
            self.N_rtau = 0
        
        # Precompute k-space integrals if r or rtau arrays have been supplied
        if self.N_r>0:
            self._prepare_templates(self.ints_1d, False, self.ints_coll)
        if self.N_rtau>0:
            self._prepare_templates(False, self.ints_2d, False)
        
    ### UTILITY FUNCTIONS
    def _configure_templates(self, templates, C_phi, C_lens_weight):
        """Check input templates and log which quantities to compute."""
        
        # Check correct templates are being used and print them
        self.all_templates_1d = ['gNL-loc','gNL-con','tauNL-loc','tauNL-direc','tauNL-even','tauNL-odd']
        self.all_templates_2d = ['gNL-dotdot','gNL-dotdel','gNL-deldel']
        self.all_templates_coll = ['tauNL-light','tauNL-heavy']
        self.all_templates = self.all_templates_1d+self.all_templates_2d+self.all_templates_coll+['lensing','point-source']
        ii = 0
        for t in templates:
            ii += 1
            if 'tauNL-direc' in t: continue # direction-dependent templates
            if 'tauNL-even' in t: continue # direction-dependent even templates
            if 'tauNL-odd' in t: continue # direction-dependent odd templates
            if 'tauNL-light' in t: continue # light massive particle templates
            if 'tauNL-heavy' in t: continue # heavy massive particle templates
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
        self.ints_1d, self.ints_2d, self.ints_coll = False, False, False
        self.to_compute, self.n1n3n, self.coll_params = [], [], {}
        
        # Check each template in turn
        if 'gNL-loc' in templates:
            self.to_compute.append(['q'])
            self.ints_1d = True
        if 'gNL-con' in templates:
            self.to_compute.append(['r'])
            self.ints_1d = True
        if 'gNL-dotdot' in templates:
            self.to_compute.append(['a'])
            self.ints_2d = True
        if 'gNL-dotdel' in templates:
            self.to_compute.append(['a','b','c'])
            self.ints_2d = True
        if 'gNL-deldel' in templates:
            self.to_compute.append(['b','c'])
            self.ints_2d = True
        if 'tauNL-loc' in templates:
            self.to_compute.append(['q'])
            self.n1n3n += [[0,0,0]]
            self.ints_1d = True
        for t in templates:
            if 'tauNL-direc' in t:
                # Compute indices
                n1,n3,n = np.asarray((t).split(':')[1].split(','),dtype=int)
                # Check inputs
                if (n1==0)&(n3==0)&(n==0)&('tauNL-loc' in self.templates):
                    raise Exception("tauNL-loc and tauNL-direc:0,0,0 are fully degenerate!")
                if ('tauNL-direc:%d,%d,%d'%(n3,n1,n) in self.templates) and (n1<n3):
                    raise Exception("tauNL-direc:%d,%d,%d and tauNL-direc:%d,%d,%d are fully degenerate!"%(n1,n3,n,n3,n1,n))
                assert (n>=abs(n1-n3))and(n<=n1+n3),"Direction-dependent indices %d,%d,%d do not satisfy triangle conditions"%(n1,n3,n)
                assert (n1<=5)and(n3<=5)and(n<=5), "n>5 has not yet been implemented!"
                self.to_compute.append(['q'])
                self.n1n3n += [[n1,n3,n]]
                self.ints_1d = True
            if 'tauNL-even' in t:
                # Compute indices
                n = int(t.split(':')[1])
                # Check inputs
                if (n==0)&('tauNL-loc' in self.templates):
                   raise Exception("tauNL-even:0 and tauNL-loc are fully degenerate!")
                if (n==0)&('tauNL-direc:0,0,0' in self.templates):
                    raise Exception("tauNL-even:0 and tauNL-direc:0,0,0 are fully degenerate!")
                if ('tauNL-direc:%d,%d,%d'%(0,n,n) in self.templates)and('tauNL-direc:%d,%d,%d'%(n,n,0) in self.templates):
                    raise Exception("tauNL-even:%d is fully degenerate with tauNL-direc:%d,%d,%d and tauNL-direc:%d,%d,%d"%(n,0,n,n,n,n,0))
                self.to_compute.append(['q'])
                self.n1n3n += [[0,n,n],[n,n,0]]
                self.ints_1d = True
            if 'tauNL-odd' in t:
                # Compute indices
                n = int(t.split(':')[1])
                # Check inputs
                if (n==0)&('tauNL-direc:1,1,1' in self.templates):
                    raise Exception("tauNL-odd:0 and tauNL-direc:1,1,1 are fully degenerate!")
                # Check all possible n1, n3, n choices
                n1n3n0_list = []
                n1n3n0_list += [[n-1,n-1,1],[n-1,n+1,1],[n+1,n-1,1],[n+1,n+1,1]]
                n1n3n0_list += [[n-1,1,n-1],[n-1,1,n+1],[n+1,1,n-1],[n+1,1,n+1]]
                n1n3n0_list += [[1,n-1,n-1],[1,n-1,n+1],[1,n+1,n-1],[1,n+1,n+1]]
                n1n3n0_good = []
                for nn in n1n3n0_list:
                    if ((nn[2]>=np.abs(nn[0]-nn[1]))&(nn[2]<=nn[0]+nn[1])&((np.asarray(nn)>=0).all())):
                       n1n3n0_good.append(nn)
                self.to_compute.append(['q'])
                self.n1n3n += n1n3n0_good
                self.ints_1d = True
            if 'tauNL-light' in t:
                # compute indices
                s = int(t.split(':')[1].split(',')[0])
                nu_s = float(t.split(':')[1].split(',')[1])
                # Check inputs
                if s==0 and nu_s==1.5 and ('tauNL-loc' in self.templates):
                    raise Exception("tauNL-loc and tauNL-light:0,1.5 are fully degenerate!")
                if nu_s==1.5 and ('tauNL-direc:%d,%d,0'%(s,s) in self.templates):
                    raise Exception("tauNL-direc:%d,%d,0 and tauNL-light:%d,1.5 are fully degenerate!"%(s,s,s))
                
                # Check bounds
                if s==0:
                    assert nu_s>=0. and nu_s<=1.5, "Light spin-0 particles have nu_s in (0, 3/2)"
                else:
                    assert nu_s>=0 and nu_s<=s-0.5, "Light spin-%d particles have nu_s in (0, %.1f)"%(s,s-1./2.)
                    if nu_s>0.5:
                        print("## Warning: Spin = %d nu_s = %d violates the Higuchi bound!"%(s,nu_s))
                assert type(s)==int, "Spin must be an integer!"
                this_params = []
                for S in range(0,2*s+1,2):
                    this_params.append([s,S])
                self.to_compute.append(['q'])
                _merge_dict(self.coll_params,{nu_s-3./2.:this_params})
                self.ints_coll = True
                self.ints_1d = True
            if 'tauNL-heavy' in t:
                # compute indices
                s = int(t.split(':')[1].split(',')[0])
                mu_s = float(t.split(':')[1].split(',')[1])
                # Check inputs
                if mu_s==0 and ('tauNL-light:%d,%s'%(s,mu_s) in self.templates):
                   raise Exception("tauNL-light:s,0 and tauNL-heavy:s,0 are fully degenerate!")
                
                # Check bounds
                assert mu_s>=0., "Heavy spin-0 particles have mu_s >=0"
                assert type(s)==int, "Spin must be an integer!"
                this_params = []
                for S in range(0,2*s+1,2):
                    this_params.append([s,S])
                self.to_compute.append(['q'])
                _merge_dict(self.coll_params,{1.0j*mu_s-3./2.:this_params, -1.0j*mu_s-3./2.: this_params})
                self.ints_coll = True
                self.ints_1d = True
            if 'lensing' in t:
                
                # Check inputs
                assert len(C_phi)>0, "Must supply lensing power spectrum!"
                assert len(C_phi)>=self.Lmax_lens+1, "Must specify C^phi-phi(L) up to at least Lmax."
                if not self.pol:
                    assert 'TT' in C_lens_weight.keys(), "Must specify unlensed TT power spectrum!"
                else:
                    assert 'TE' in C_lens_weight.keys(), "Must specify unlensed TE power spectrum!"
                    assert 'EE' in C_lens_weight.keys(), "Must specify unlensed EE power spectrum!"
                    assert 'BB' in C_lens_weight.keys(), "Must specify unlensed BB power spectrum!"
                    for k in C_lens_weight.keys():
                        assert len(C_lens_weight[k])>=self.lmax+1, "Must specify C_lens_weight(l) up to at least lmax."
                        
                # Reshape and store
                self.C_phi = C_phi
                self.C_lens_weight = {k: C_lens_weight[k][:self.lmax+1] for k in C_lens_weight.keys()}
                self.to_compute.append(['u','v'])
            if 'point-source' in t:
                self.to_compute.append(['u'])
               
        # Identify unique components 
        self.n1n3n = np.asarray(np.unique(self.n1n3n,axis=0))
        self.coll_params = {k: np.asarray(self.coll_params[k]) for k in self.coll_params.keys()}
        self.to_compute = np.unique(np.concatenate(self.to_compute))
                
        if len(self.n1n3n)>0:
            self.nmax = np.max(self.n1n3n[:,:2])
        else:
            self.nmax = 0
        if len(self.coll_params.keys())>0:
            self.nmax_coll = np.max([np.max(self.coll_params[k][:,0]) for k in self.coll_params.keys()])
            self.nmax_coll_F = np.max([np.max(self.coll_params[k][:,1]) for k in self.coll_params.keys()])
        else:
            self.nmax_coll = 0
            self.nmax_coll_F = 0
        
        # Check which p indices to compute
        self.p_inds = []
        if 'gNL-loc' in self.templates:
            self.p_inds = [0]
        if len(self.n1n3n)>0:
            self.p_inds = list(np.unique(self.p_inds+list(np.unique(self.n1n3n[:,:2]))))
        
        # Define maximum n
        if len(self.n1n3n)!=0:
            self.nmax_F = np.max(self.n1n3n[:,2])
        else:
            self.nmax_F = 0
            
        # Define wider L filtering for non-zero n harmonics
        self.nLmax = self.Lmax+max([self.nmax_F,self.nmax_coll_F])
        self.nLmin = self.Lmin-max([self.nmax_F,self.nmax_coll_F])
        self.nLfilt = (self.base.l_arr>=self.nLmin)&(self.base.l_arr<=self.nLmax)
        self.Ls, self.Ms = self.base.l_arr[self.nLfilt], self.base.m_arr[self.nLfilt]
        
        # Create filtering for minimum ls
        self.nLminfilt = self.base.l_arr[self.base.l_arr<=self.nLmax]>=self.nLmin
        self.Lminfilt = self.base.l_arr[self.base.l_arr<=self.Lmax]>=self.Lmin
        if 'lensing' in self.templates:
            self.Lminfilt_lens = self.base.l_arr[self.base.l_arr<=self.Lmax_lens]>=self.Lmin_lens
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
            if np.any([('gNL' in t) for t in self.templates]):
                print("gNL -- 4-field summation: %.2fs"%self.timers['gNL_summation'])
            if np.any([('tauNL' in t) for t in self.templates]):
                print("tauNL -- 2-field transforms: %.2fs"%self.timers['tauNL_products'])
                print("tauNL -- 4-field summation: %.2fs"%self.timers['tauNL_summation'])
            if 'lensing' in self.templates:
                print("Lensing -- 2-field transforms: %.2fs"%self.timers['lensing_products'])
                print("Lensing -- 4-field summation: %.2fs"%self.timers['lensing_summation'])
        if (self.timers['fisher']!=0 or self.timers['optimization']!=0):
            if self.timers['analytic_fisher']!=0:
                print("Analytic Fisher Matrices: %.2fs"%self.timers['analytic_fisher'])
            if self.timers['fish_grfs']!=0:
                print("Fisher -- creating GRFs: %.2fs"%self.timers['fish_grfs'])
            if self.timers['Ainv']!=0:
                print("Fisher -- A^-1 filtering: %.2fs"%self.timers['Ainv'])
            if self.timers['fish_products']!=0:
                print("Fisher -- 2-field transforms: %.2fs"%self.timers['fish_products'])
            if self.timers['fish_convolve']!=0:
                print("Fisher -- 2-field convolution: %.2fs"%self.timers['fish_convolve'])
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
    def _prepare_templates(self, ints_1d=True, ints_2d=True, ints_coll=True):
        """Compute necessary k-integrals over the transfer functions for template estimation.

        This fills arrays such as plLXs, qlXs, and FLLs arrays. Note that values outside the desired ell & field range will be set to zero.
        """
        # Print dimensions of k and r
        print("N_k: %d"%len(self.k_arr))
        if ints_1d: print("N_r: %d"%self.N_r)
        if ints_2d: print("N_rtau: %d"%self.N_rtau)
        
        # Clear saved quantities, if necessary
        if hasattr(self, 't0_num'): delattr(self, 't0_num')
        if ints_1d:
            if hasattr(self, 'plLXs'): delattr(self, 'plLXs')
            if hasattr(self, 'qlXs'): delattr(self, 'qlXs')
            if hasattr(self, 'rlXs'): delattr(self, 'rlXs')
            if hasattr(self, 'FLLs'): delattr(self, 'FLLs')
        if ints_2d:
            if hasattr(self, 'alXs'): delattr(self, 'alXs')
            if hasattr(self, 'blXs'): delattr(self, 'blXs')
            if hasattr(self, 'clXs'): delattr(self, 'clXs')
        if ints_coll:
            if hasattr(self, 'coll_plLXs'): delattr(self, 'coll_plLXs')
            if hasattr(self, 'coll_FLLs'): delattr(self, 'coll_FLLs')
            
        # Check which lmin/lmax we need
        lmax = max([self.lmax+1,self.lmax+max([self.nmax,self.nmax_coll])])
        lmin = max([min([self.lmin-1,self.lmin-max([self.nmax,self.nmax_coll])]),0])
        
        # Precompute all spherical Bessel functions on a regular grid
        print("\nPrecomputing Bessel functions")
        
        # Check maximum k*r required
        max_kr = 0.
        if ints_1d:
            max_kr = max(self.k_arr)*max([max_kr,max(self.r_arr)])
        if ints_2d:
            max_kr = max(self.k_arr)*max([max_kr,self.r_hor+10000+5e6])
        
        x_arr = list(np.arange(0,lmax*2,0.01))+list(np.arange(lmax*2,min(max_kr*1.01,lmax*100),0.1))
        if max_kr>100*lmax:
            x_arr += list(np.linspace(lmax*100,max_kr*1.01,1000))
        x_arr = np.asarray(x_arr,dtype=np.float64)
        
        # Compute Bessel function in range of interest in Cython
        jlxs = np.zeros((lmax-lmin+1,len(x_arr)),dtype=np.float64,order='C')
        compute_bessel(x_arr,lmin,lmax,jlxs,self.base.nthreads)
        if np.isnan(jlxs).any(): raise Exception("Spherical Bessel calculation failed!")
        
        # Interpolate to the values of interest
        print("Interpolating Bessel functions")
        if ints_1d:
            assert np.max(x_arr)>=np.max(self.k_arr)*np.max(self.r_arr)
            jlkr_all = interpolate_jlkr(x_arr, self.k_arr, self.r_arr, jlxs, self.base.nthreads)
            jlkr = jlkr_all[self.lmin-lmin:self.lmax-lmin+1]
        
        if ints_2d:
            assert np.max(x_arr)>=np.max(self.k_arr)*np.max(self.rtau_arr[:,0])

            # Compute j_l(x)
            jlkrtau = interpolate_jlkr(x_arr, self.k_arr, self.rtau_arr[:,0], jlxs, self.base.nthreads)
            
            # Compute j_l'(x)
            if 'b' in self.to_compute:
                jlkrtau_prime = compute_jl_prime(lmin, lmax, self.lmin, self.lmax, jlkrtau, self.base.nthreads)
            
            # Trim excess values
            jlkrtau = jlkrtau[self.lmin-lmin:self.lmax-lmin+1]
        del x_arr
        
        # Set up arrays
        Pzeta_arr = self.Pzeta(self.k_arr)
        
        if 'a' in self.to_compute and ints_2d:
            # Compute a integrals in Cython
            print("Computing a_l^X(r, tau) integrals")
            self.alXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_rtau),dtype=np.float64,order='C')
            a_integral(self.k_arr,self.rtau_arr[:,1],Pzeta_arr,self.Tl_arr, 
                       jlkrtau, self.lmin, self.lmax, self.base.nthreads, self.alXs)
            
        if 'b' in self.to_compute and ints_2d:
            # Compute b integrals in Cython
            print("Computing b_l^X(r, tau) integrals")
            self.blXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_rtau),dtype=np.float64,order='C')
            b_integral(self.k_arr,self.rtau_arr[:,1],Pzeta_arr,self.Tl_arr, 
                       jlkrtau_prime, self.lmin, self.lmax, self.base.nthreads, self.blXs)
            del jlkrtau_prime
            
        if 'c' in self.to_compute and ints_2d:
            # Compute c integrals in Cython
            print("Computing c_l^X(r, tau) integrals")
            self.clXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_rtau),dtype=np.float64,order='C')
            c_integral(self.k_arr,self.rtau_arr[:,0],self.rtau_arr[:,1],Pzeta_arr,self.Tl_arr, 
                       jlkrtau, self.lmin, self.lmax, self.base.nthreads, self.clXs)
            
        if ints_2d: del jlkrtau
        
        if len(self.p_inds)>0 and ints_1d:
            
            # Compute p integrals in Cython
            print("Computing p_lL^X(r) integrals")
            self.plLXs = np.zeros((2*self.nmax+1,lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            this_nmax = max(self.p_inds)
            p_integral(self.k_arr, Pzeta_arr, self.Tl_arr, jlkr_all, self.lmin, self.lmax, lmin, lmax, this_nmax, self.base.nthreads, self.plLXs)
            
        if 'q' in self.to_compute and (ints_1d or ints_coll):
            
            # Compute q integrals in Cython
            print("Computing q_l^X(r) integrals")
            self.qlXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            q_integral(self.k_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.qlXs)
            
        if 'r' in self.to_compute and ints_1d:
            
            # Compute r integrals in Cython
            print("Computing r_l^X(r) integrals")
            self.rlXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            r_integral(self.k_arr, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.rlXs)
            
        if ints_coll and len(self.coll_params.keys())>0: # check there's something to compute!
            # Compute p_lL^{beta,X}(r) integrals on a flattened two-dimensional grid (vectorized).
            # We restrict to k >= k_coll in the integrals!
            print("Computing collider p_lL^{beta,X}(r) integrals")
                
            # We store [p_l(l-nmax), ..., p_l(l+n), ..., p_l(l+nmax)] for each beta = nu_s-3/2
            self.coll_plLXs = {}
            for beta in self.coll_params.keys():
                if np.imag(beta)!=0 and np.imag(beta)>0: continue # can get from complex conjugate 
                
                # Compute p integrals in Cython
                this_nmax = max(self.coll_params[beta][:,0])
                plLXs = np.zeros((2*this_nmax+1,lmax+1,1+2*self.pol,self.N_r),dtype=np.complex128,order='C')
                collider_p_integral(self.k_arr, Pzeta_arr, self.Tl_arr, jlkr_all, beta, self.k_coll, self.lmin, self.lmax, lmin, lmax, this_nmax, self.base.nthreads, plLXs)
                self.coll_plLXs[beta] = plLXs
                del plLXs
                    
        if (ints_1d or ints_coll): del jlkr, jlkr_all
        
        if ints_1d and len(self.n1n3n)>0: # check there's at least one thing to compute!
            
            print("Computing F_LL'(r, r') integrals")
            # Compute F integral for each L in the required range, using exact result
            FLLs = np.zeros((2*self.nmax_F+1,self.nLmax+1,self.N_r,self.N_r))
            pref = 2.*np.pi**2 * self.As / self.k_pivot**(self.ns-1)
            compute_F_integral(self.r_arr, FLLs, self.Lmin, self.Lmax, self.nmax_F, self.ns, pref, self.base.nthreads)
            self.FLLs = np.asarray(FLLs,dtype=np.float64,order='C')
            del FLLs
            
        if ints_coll and len(self.coll_params.keys())>0: # check there's at least one thing to compute!
            
            # Define K array (truncating at K_coll)
            K_arr = np.geomspace(0.1/(self.r_hor+5000),self.K_coll*1.01,5000)
            
            # Define 1d array to compute Bessel functions
            this_Lmax = self.Lmax+self.nmax_coll_F
            this_Lmin = max([self.Lmin-self.nmax_coll_F,0])
            max_Kr = self.K_coll*np.max(self.r_arr)
            y_arr = list(np.arange(0,this_Lmax*2,0.01))+list(np.arange(this_Lmax*2,max_Kr*1.01,0.01))
            y_arr = np.asarray(y_arr,dtype=np.float64)
            
            # Compute Bessel function in range of interest in Cython and interpolate to K, r grid
            jlys = np.zeros((this_Lmax-this_Lmin+1,len(y_arr)),dtype=np.float64,order='C')
            compute_bessel(y_arr,this_Lmin,this_Lmax,jlys,self.base.nthreads)
            jlKs = interpolate_jlkr(y_arr, K_arr, self.r_arr, jlys, self.base.nthreads)
            
            print("Computing F_LL'^beta(r, r') integrals")
            self.coll_FLLs = {}
            
            for beta in self.coll_params.keys():
                this_nmax_F = max(self.coll_params[beta][:,1])
                betaF = -2.*beta # exponent!
                if np.imag(beta)!=0 and np.imag(beta)>0: continue # can get from complex conjugate 
                
                # Compute F integral for each L in the required range, using a numerical approximation
                FLLs = np.zeros((2*this_nmax_F+1,self.base.lmax+1,self.N_r,self.N_r),dtype='complex128')
                compute_collider_F_integral(K_arr, self.Pzeta(K_arr), jlKs, -2.*beta, self.K_coll, self.Lmin, self.Lmax,
                                            this_nmax_F, self.nmax_coll_F, self.base.nthreads, FLLs)
                self.coll_FLLs[beta] = np.asarray(FLLs,dtype=np.complex128,order='C')
                del FLLs
            del jlKs
            
        if (ints_1d or ints_coll): del jlxs
        
        # Define Cython utility class
        self.utils = tauNL_utils(self.base.nthreads, self.N_r, self.base.l_arr.astype(np.int32),self.base.m_arr.astype(np.int32),
                                              self.Lmin, self.Lmax, self.ls.astype(np.int32), self.ms.astype(np.int32), 
                                              self.Ls.astype(np.int32),self.Ms.astype(np.int32),self.nmax_F)            
        
        print("Precomputation complete")
        
    def _calC(self, s, S, mu_s):
        """Compute the C_s(S, mu_s) coupling coefficient for collider templates"""
        
        def W_lam(lam, s, mu_s):
            """Compute the W_lambda(s, mu_s) phase function"""
            assert abs(lam)<=s
            if np.real(mu_s)==0.:
                mu_s += 1e-12 # for stability!
            W = gamma(0.5+s-1.0j*mu_s)*gamma(0.5+lam+1.0j*mu_s)/(gamma(0.5+s+1.0j*mu_s)*gamma(0.5+lam-1.0j*mu_s))
            if np.real(mu_s)==1e-12:
                return W.real
            else:
                return W
        
        # Check conditions!
        if s==0 and np.real(mu_s)==0:
            nu_s = np.real(-1.0j*mu_s)
            assert nu_s>=0. and nu_s<=1.5, "Light spin-0 particles have nu_s in (0, 3/2)"
        elif np.real(mu_s)==0:
            nu_s = np.real(-1.0j*mu_s)
            assert nu_s>=0 and nu_s<=s-0.5, "Light spin-%d particles have nu_s in (0, %.1f)"%(s,s-1./2.)
            assert nu_s<=0.5, "Couplings undefined for Higuchi-bound violating particles!"
        else:
            assert np.imag(mu_s)==0, "mu_s can't be complex!"
       
        if mu_s==0.:
            return (S==0)*(4.*np.pi)**(3./2.)/np.sqrt(2.*s+1.)*(-1.)**s
        else:
            pref = (4.*np.pi)**(3./2.)/(2.*s+1.)*np.sqrt(2.*S+1.)
            out = 0.
            for lam in range(-s,s+1):
                out += (-1.)**(S+lam)*W_lam(lam, s, mu_s)*self.wig3(s,s,S,lam,-lam,0)   
            return np.real_if_close(pref*out) 
    
    def _omega_prime(self, s, mu_s):
        """Compute the S-independent piece of the omega_s(mu_s) coupling coefficient for collider templates."""

        if np.real(mu_s)==0:
            return 0. # no phase for light particles!
        else:
            val = 2.**(-4.*mu_s*1.0j)*(1.+1.0j*np.sinh(np.pi*mu_s))*(2.5+s+1.0j*mu_s)**2./(1.5+s-1.0j*mu_s)**2.
            val *= gamma(-1.0j*mu_s)**2*gamma(0.5+s+1.0j*mu_s)**2
            return np.angle(-1.*val)
    
    def _decompose_tauNL_even(self, n):
        """Compute the (n1, n3, n) contributions to the tauNL-even:n template, and their associated weights."""
        pref = (-1.)**n/3.*(4.*np.pi)**(3./2.)/np.sqrt(2.*n+1.)
        if n==0:
            return [[0,0,0]],[3*pref]
        else:
            return [[n,n,0],[0,n,n]],[pref,2*pref]
    
    def _decompose_tauNL_odd(self, n):
            """Compute the (n1, n3, n) contributions to the tauNL-odd:n template, and their associated weights."""
            pref = (-1.)**n*np.sqrt(2.)/3.*(4.*np.pi)**(3./2.)
        
            all_n1n3ns = []
            all_weights = []
            for N in range(abs(n-1),n+2):
                tj_fac1 = self.wig3zero(N,1,n)
                if tj_fac1==0: continue
                for Np in range(abs(n-1),n+2):
                    tj_fac12 = tj_fac1*self.wig3zero(Np,1,n)
                    if tj_fac12==0: continue 
                    tj_fac12 *= self.wig6(N,1,Np,1,n,1)
                    if tj_fac12==0: continue
                    
                    weight = pref*np.sqrt(2.*N+1.)*np.sqrt(2.*Np+1.)*tj_fac12
                        
                    # Check each combination
                    if Np>=N:
                        all_n1n3ns.append([N,Np,1])
                        all_weights.append(weight)
                    else:
                        all_n1n3ns.append([Np,N,1])
                        all_weights.append((-1)**(Np+N)*weight)
                    all_n1n3ns.append([1,N,Np])
                    all_weights.append(weight*(1.+(-1)**(n+N+1)))
            
            # Compute unique weights
            uniq_n1n3ns = []
            uniq_weights = []
            for this_n1n3n in np.unique(all_n1n3ns,axis=0):
                sum_weight = np.sum([all_weights[i] for i in range(len(all_weights)) if (np.asarray(all_n1n3ns[i])==this_n1n3n).all()])
                uniq_n1n3ns.append(this_n1n3n)
                uniq_weights.append(sum_weight)
            return uniq_n1n3ns, uniq_weights
    
    @_timer_func('map_transforms')
    def _compute_weighted_maps(self, h_lm_filt, flX_arr, spin=0):
        """
        Compute [Sum_lm {}_sY_lm(i) f_l^X(i) h_lm] maps for each sampling point i, given the relevant weightings. These are used in the trispectrum numerators and Fisher matrices.
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
    def _compute_weighted_gaunt_maps(self, h_lm_filt, flLX_arr_re, n, mu, flLX_arr_im=[]):
        """
        Compute F_nmu(ang, i)  = [Sum_LM Y_LM(ang) Sum_lmX (-1)^mu i^{L-l} f_{lL}^X(i) G^{lLn}_{nM(-mu)} h^*_lm] maps for each sampling point i, given the relevant weightings. These are used in the trispectrum numerators.
        
        We split the flLX_arr into a real and imaginary part, which are handled separately (if specified).
        """
        if not (hasattr(self,'r_arr') or hasattr(self,'rtau_arr')):
            raise Exception("Radial arrays have not been computed!")

        # Check inputs
        assert n<=np.max([self.nmax,self.nmax_coll]), "Order can't be larger than n_max"
        assert mu<=n, "mu<=n is required!"
        assert mu>=0, "Negative mu can be obtained by conjugation properties"
                
        # Define arrays
        nmax = len(flLX_arr_re)//2
        
        if n==0:
            assert mu==0
            
            # Perform harmonic transforms
            if len(flLX_arr_im)!=0:
                # Sum over polarizations (only filling non-zero elements)
                summR = np.zeros((flLX_arr_re.shape[3],len(self.lminfilt)),order='C',dtype='complex128')
                summR[:,self.lminfilt] = self.utils.apply_fl_weights(flLX_arr_re[nmax], h_lm_filt, 1./np.sqrt(4.*np.pi))
                
                # Sum over polarizations (only filling non-zero elements)
                summI = np.zeros((flLX_arr_re.shape[3],len(self.lminfilt)),order='C',dtype='complex128')
                summI[:,self.lminfilt] = self.utils.apply_fl_weights(flLX_arr_im[nmax], h_lm_filt, 1./np.sqrt(4.*np.pi))
            
                # Perform harmonic transforms
                out = r2c(self.base.to_map_vec(summR, lmax=self.lmax), self.base.to_map_vec(summI,lmax=self.lmax), self.base.nthreads)  
            
            else:
                # Sum over polarizations (only filling non-zero elements)
                summR = np.zeros((flLX_arr_re.shape[3],len(self.lminfilt)),order='C',dtype='complex128')
                summR[:,self.lminfilt] = self.utils.apply_fl_weights(flLX_arr_re[nmax], h_lm_filt, 1./np.sqrt(4.*np.pi))
                
                out = self.base.to_map_vec(summR, lmax=self.lmax)
            
        else:
            
            # Create arrays to harmonic transform
            Larr_real = np.zeros((flLX_arr_re.shape[3],np.sum(self.base.l_arr<=self.lmax+n)),order='C',dtype='complex128')
            Larr_imag = np.zeros((flLX_arr_re.shape[3],np.sum(self.base.l_arr<=self.lmax+n)),order='C',dtype='complex128')
            
            # Compute sums over ell and polarization, giving ell+Delta, mu-m and ell-Delta, mu+m coefficients. 
            # We fill up a map of (L, M>=0) and (L, M<0) element by element (skipping m=0 in the second sum).
            # We do the heavy lifting in Cython 
            self.utils.shift_and_weight_map_all(flLX_arr_re, h_lm_filt, n, mu, 1.0, Larr_real, Larr_imag, nmax)
            if len(flLX_arr_im)!=0:
                self.utils.shift_and_weight_map_all(flLX_arr_im, h_lm_filt, n, mu, 1.0j, Larr_imag, Larr_real, nmax)
                
            # Perform harmonic transforms
            if (np.imag(Larr_imag)!=0.).any():
                out = r2cstar(self.base.to_map_vec(Larr_real,lmax=self.lmax+n), self.base.to_map_vec(1.0j*Larr_imag,lmax=self.lmax+n), self.base.nthreads)
            else:
                out = self.base.to_map_vec(Larr_real,lmax=self.lmax+n)
        return out
    
    @_timer_func('map_transforms')
    def _compute_lensing_U_map(self, h_lm_filt):
        """
        Compute lensing U map from a given data vector. These are used in the trispectrum numerators.
        
        The U^T map is also used in the point-source estimator. (If "lensing" is not in self.to_compute, we only compute U^T.)
        
        We return [U^T, U^E, U^B].
        """
        
        # Output array
        U = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
        
        # Compute X = T piece for point-source + lensing estimation
        inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
        inp_lm[self.lminfilt] = h_lm_filt[0]
        U[0] = self.base.to_map(inp_lm[None],lmax=self.lmax)[0]
        
        # Compute X = E, B if implementing lensing estimators
        if 'lensing' in self.templates:
            
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
    def _compute_lensing_V_map(self, h_lm_filt):
        """
        Compute lensing V map from a given data vector. These are used in the trispectrum numerators.
        
        We return [[V^T+, V^T-], [V^E+, V^E-], [V^B+, V^B-]].
        """
        
        ls = self.base.l_arr[self.lfilt]
        
        # Output array
        V = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
            
        if not self.pol:
            pref = np.sqrt(ls*(ls+1.))*self.C_lens_weight['TT'][ls]
            inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
            inp_lm[self.lminfilt] = pref*h_lm_filt[0]
            V[0] = self.base.to_map_spin(-inp_lm,inp_lm,spin=1,lmax=self.lmax)[1] # h_lm (-1)Y_lm
            del pref, inp_lm
        
        else:
        
            # Output array
            V = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
            
            # Spin-0, X = T
            pref = np.sqrt(ls*(ls+1.))
            wienerT = (self.C_lens_weight['TT'][ls]*h_lm_filt[0]+self.C_lens_weight['TE'][ls]*h_lm_filt[1])
            inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
            inp_lm[self.lminfilt] = pref*wienerT
            V[0] = self.base.to_map_spin(-inp_lm,inp_lm,spin=1,lmax=self.lmax)[1] # h_lm (-1)Y_lm
            del inp_lm
            
            # Spin-2
            pref_p = np.sqrt((ls+2.)*(ls-1.))
            pref_m = np.sqrt((ls-2.)*(ls+3.))
            wienerE = (self.C_lens_weight['TE'][ls]*h_lm_filt[0]+self.C_lens_weight['EE'][ls]*h_lm_filt[1])
            wienerB = self.C_lens_weight['BB'][ls]*h_lm_filt[2]
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
        
        if filtering=='A':
            return np.asarray([self._compute_weighted_maps(imap, self.alXs) for imap in input_maps], order='C')
               
        elif filtering=='B':
            return np.asarray([self._compute_weighted_maps(imap, self.blXs) for imap in input_maps], order='C')
            
        elif filtering=='C':
            return np.asarray([self._compute_weighted_maps(imap, self.clXs, spin=1) for imap in input_maps], order='C')
        
        elif filtering=='P':
            P_maps = {}
            for n in self.p_inds:
                if n==0:
                    Pn_maps = np.zeros((2,1,self.N_r,self.base.Npix),dtype=np.float64)
                    for i,imap in enumerate(input_maps):
                        Pn_maps[i,0] = self._compute_weighted_gaunt_maps(imap, self.plLXs, 0, 0)
                    P_maps[0] = Pn_maps
                else:
                    Pn_maps = np.zeros((2,n+1,self.N_r,self.base.Npix),dtype=np.complex128)
                    for i,imap in enumerate(input_maps):
                        for mu in range(n+1):
                            Pn_maps[i,mu] = self._compute_weighted_gaunt_maps(imap, self.plLXs, n, mu)
                    P_maps[n] = Pn_maps
            return P_maps
        
        elif filtering=='coll-P':    
            coll_P_maps = {}
            for beta in self.coll_params.keys():
                
                # Define plL filters, using conjugation symmetries
                if np.imag(beta)==0:
                    re_coll_plLX = np.asarray(self.coll_plLXs[beta].real,order='C')
                elif np.imag(beta)<0:
                    re_coll_plLX = np.asarray(self.coll_plLXs[beta].real,order='C')
                    im_coll_plLX = np.asarray(self.coll_plLXs[beta].imag,order='C')
                else:
                    re_coll_plLX = np.asarray(self.coll_plLXs[np.conj(beta)].real,order='C')
                    im_coll_plLX = np.asarray(-self.coll_plLXs[np.conj(beta)].imag,order='C')
                
                # Compute P[beta, s, lam] maps
                for s in np.unique(self.coll_params[beta][:,0]):
                    Ps_maps = np.zeros((2,s+1,self.N_r,self.base.Npix),dtype=np.complex128)
                    for i,imap in enumerate(input_maps):
                        for lam in range(s+1):
                            if np.imag(beta)==0:
                                Ps_maps[i,lam] = self._compute_weighted_gaunt_maps(imap, re_coll_plLX, s, lam)
                            else:
                                Ps_maps[i,lam] = self._compute_weighted_gaunt_maps(imap, re_coll_plLX, s, lam, im_coll_plLX)
                    coll_P_maps[beta,s] = Ps_maps
            return coll_P_maps
            
        elif filtering=='Q':
            return np.asarray([self._compute_weighted_maps(imap, self.qlXs) for imap in input_maps],order='C')     
        
        elif filtering=='R':
            return np.asarray([self._compute_weighted_maps(imap, self.rlXs) for imap in input_maps], order='C')
              
        elif filtering=='U':
            return np.asarray([self._compute_lensing_U_map(imap) for imap in input_maps], order='C')        
            
        elif filtering=='V':
            return np.asarray([self._compute_lensing_V_map(imap) for imap in input_maps], order='C')        
        
        else:
            raise Exception("Filtering %s is not implemented!"%filtering)

    def _apply_all_filters(self, input_map):
        """Compute the processed fields with all relevant filterings for a single input map."""
        
        # Output array
        output = {}
        
        # Compute EFTI maps
        if 'a' in self.to_compute:
            output['a'] = self._compute_weighted_maps(input_map, self.alXs)
        
        if 'b' in self.to_compute:
            output['b'] = self._compute_weighted_maps(input_map, self.blXs)
            
        if 'c' in self.to_compute:
            output['c'] = self._compute_weighted_maps(input_map, self.clXs, spin=1)
        
        # Compute P_nmu maps, if required
        for n in self.p_inds:
            ## Compute all P_nmu
            for mu in range(n+1):
                output['p%d%d'%(n,mu)] = self._compute_weighted_gaunt_maps(input_map, self.plLXs, n, mu)
        
        # Compute collider P^beta_sl maps, if required
        for beta in self.coll_params.keys():
            # Define p_l filter, using conjugation symmetries
            if np.imag(beta)!=0:
                if np.imag(beta)<=0:
                    re_coll_plLX = np.asarray(self.coll_plLXs[beta].real,order='C')
                    im_coll_plLX = np.asarray(self.coll_plLXs[beta].imag,order='C')
                else:
                    re_coll_plLX = np.asarray(self.coll_plLXs[np.conj(beta)].real,order='C')
                    im_coll_plLX = np.asarray(-self.coll_plLXs[np.conj(beta)].imag,order='C')
            else:   
                re_coll_plLX = np.asarray(self.coll_plLXs[beta].real,order='C')
            
            ## Compute all collider P^beta_smu
            this_maps = {}
            for s in np.unique(self.coll_params[beta][:,0]):
                for lam in range(s+1):
                    if np.imag(beta)!=0:
                        this_maps['p%d%d'%(s,lam)] = self._compute_weighted_gaunt_maps(input_map, re_coll_plLX, s, lam, im_coll_plLX)
                    else:   
                        this_maps['p%d%d'%(s,lam)] = self._compute_weighted_gaunt_maps(input_map, re_coll_plLX, s, lam)
            output['coll-%.8f,%.8fi'%(beta.real,beta.imag)] = this_maps
            
        # Compute local maps
        if 'q' in self.to_compute:
            output['q'] = self._compute_weighted_maps(input_map, self.qlXs)
            
        if 'r' in self.to_compute:
            output['r'] = self._compute_weighted_maps(input_map, self.rlXs)
              
        # Compute lensing maps
        if 'u' in self.to_compute:
            output['u'] = self._compute_lensing_U_map(input_map)        
            
        if 'v' in self.to_compute:
            output['v'] = self._compute_lensing_V_map(input_map)
        
        return output
    
    @_timer_func('lensing_products')
    def _compute_lensing_Phi(self, maps1, maps2, add_sym=False):
        """Compute the Phi_LM field used in the lensing estimators."""    
        
        # First sum over U and V maps to form the spin-1 maps (in Cython for speed)
        if add_sym:
            input_map = lens_phi_sum_sym(maps1['u'], maps2['u'], maps1['v'], maps2['v'], self.base.nthreads)
        else:
            input_map = lens_phi_sum(maps1['u'], maps2['v'], self.base.nthreads)
        
        # Now compute spin-1 transforms
        return -0.25*np.sum(np.array([1,-1])[:,None]*self.base.to_lm_spin(input_map,input_map.conj(),spin=1,lmax=self.Lmax_lens),axis=0)[self.Lminfilt_lens]
        
    def _process_sim(self, sim_pair, input_type='map'):
        """
        Process a single pair of input simulations. This is used for the 2- and 0-field term of the trispectrum estimator.
        
        We return a set of weighted maps for this simulation (filtered by e.g. p_l^X).
        """
        # Transform to Fourier space and normalize appropriately
        t_init = time.time()
        h_alpha_lm = np.asarray(self.applySinv(sim_pair[0], input_type=input_type, lmax=self.lmax)[:,self.lminfilt],order='C')
        h_beta_lm =  np.asarray(self.applySinv(sim_pair[1], input_type=input_type, lmax=self.lmax)[:,self.lminfilt],order='C')
        self.timers['Sinv'] += time.time()-t_init
        
        # Compute H_alpha and H_beta maps
        proc_a_maps = self._apply_all_filters(h_alpha_lm)
        proc_b_maps = self._apply_all_filters(h_beta_lm)
        
        return proc_a_maps, proc_b_maps
    
    def load_sims(self, load_sim_pair, N_pairs, verb=False, preload=True, input_type='map'):
        """
        Load in and preprocess N_sim pairs of Monte Carlo simulations used in the two- and zero-field terms of the trispectrum estimator.

        The input is a function which loads the pairs of simulations in map- or harmonic-space given an index (0 to N_pairs-1).

        If preload=False, the simulation products will not be stored in memory, but instead accessed when necessary. This greatly reduces memory usage, but is less CPU efficient if many datasets are analyzed together.
        
        These can alternatively be generated with a fiducial spectrum using the generate_sims script.
        """
        
        self.N_it = N_pairs
        print("Using %d pairs of Monte Carlo simulations"%self.N_it)

        if preload:
            self.preload = True
            
            # Check we have initialized correctly
            if self.ints_1d and (not hasattr(self,'r_arr')):
                raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
            if self.ints_2d and (not hasattr(self,'rtau_arr')):
                raise Exception("Need to supply radial/tau integration points or run optimize_radial_sampling_2d()!")

            #  Define lists of maps
            self.proc_a_maps, self.proc_b_maps = [],[]

            # Iterate over simulations and preprocess appropriately    
            for ii in range(self.N_it):
                if ii%5==0 and verb: print("Processing bias simulation pair %d of %d"%(ii+1,self.N_it))

                sim_pair = load_sim_pair(ii)
                proc_a_maps, proc_b_maps = self._process_sim(sim_pair, input_type=input_type)
                
                # Compute P and Q maps
                self.proc_a_maps.append(proc_a_maps)
                self.proc_b_maps.append(proc_b_maps)
                
        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")

            # Simply save iterator and continue (simulations will be processed in serial later) 
            self.load_sim_data = lambda ii: self._process_sim(load_sim_pair(ii), input_type=input_type)
            
    def generate_sims(self, N_pairs, Cl_input=[], preload=True, verb=False):
        """
        Generate Monte Carlo simulations used in the two- and zero-field terms of the trispectrum generator. 
        These are pure GRFs. By default, these are beam-convolved and generated with the input survey mask.
        
        If preload=True, we create N_pairs of simulations and store the relevant transformations into memory.
        If preload=False, we store only the function used to generate the sims, which will be processed later. This is cheaper on memory, but less CPU efficient if many datasets are analyzed together.
        
        We can alternatively load custom simulations using the load_sims script.
        """
        
        self.N_it = N_pairs
        print("Using %d pairs of Monte Carlo simulations"%self.N_it)
        
        # Define input power spectrum (with noise)
        if len(Cl_input)==0:
            Cl_input = self.base.Cl_tot

        if preload:
            self.preload = True
            
            # Check we have initialized correctly
            if self.ints_1d and (not hasattr(self,'r_arr')):
                raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
            if self.ints_2d and (not hasattr(self,'rtau_arr')):
                raise Exception("Need to supply radial/tau integration points or run optimize_radial_sampling_2d()!")

            # Define lists of maps
            self.proc_a_maps, self.proc_b_maps = [],[]
            
            # Iterate over simulations
            for ii in range(self.N_it):
                if ii%5==0 and verb: print("Generating bias simulation pair %d of %d"%(ii+1,self.N_it))
                
                # Generate (beam-convolved and masked) simulation and Fourier transform
                if self.ones_mask:
                    alpha_lm = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, output_type='harmonic', lmax=self.lmax, deconvolve_beam=False)
                    beta_lm  = self.base.generate_data(int(2e5)+ii, Cl_input=Cl_input, output_type='harmonic', lmax=self.lmax, deconvolve_beam=False)
                    proc_a_maps, proc_b_maps = self._process_sim([alpha_lm, beta_lm], input_type='harmonic')
                else:
                    alpha = self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False)
                    beta  = self.mask*self.base.generate_data(int(2e5)+ii, Cl_input=Cl_input, deconvolve_beam=False)
                    proc_a_maps, proc_b_maps = self._process_sim([alpha, beta])
            
                # Compute P and Q maps
                self.proc_a_maps.append(proc_a_maps)
                self.proc_b_maps.append(proc_b_maps)

        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            if self.ones_mask:
                self.load_sim_data = lambda ii: self._process_sim([self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, output_type='harmonic', deconvolve_beam=False),self.base.generate_data(int(2e5)+ii, Cl_input=Cl_input, output_type='harmonic', deconvolve_beam=False)], input_type='harmonic')
            else:
                self.load_sim_data = lambda ii: self._process_sim([self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False),self.mask*self.base.generate_data(int(2e5)+ii, Cl_input=Cl_input, deconvolve_beam=False)])
        
    @_timer_func('tauNL_summation')
    def _tau_sum(self, A, B, weights, n1, n3, n):
        """Compute sum over mu1,mu3,mu,M',M,L',L,r,r' for (direction-dependent / local) tauNL estimator."""
        
        # Compute tau from these maps
        if n==0:
            assert n1==n3
            tau = self.utils.tau_sum_n0(A[n1], B[n1], self.FLLs[self.nmax_F], weights, n1)
        else:
            tau = self.utils.tau_sum_general(A[n1], B[n3], self.FLLs, weights, n1, n3, n)
        return tau
    
    @_timer_func('tauNL_summation')
    def _tau_sum_collider(self, A, B, weights, s, beta):
        """Compute sum over lam1,lam3,Lam,S,L,M,L',M',r,r' for the collider tauNL estimator."""
        # Define inputs for +ve and -ve chi
        pos_str = 'coll-%d,%.8f,%.8fi'%(s,beta.real,beta.imag)
        if np.imag(beta)==0:
            neg_str = pos_str
        else:
            neg_str = 'coll-%d,%.8f,%.8fi'%(s,beta.real,-beta.imag)
        
        nu_s = beta+3./2.
        
        # Use the simplified form for s = 0
        if s==0:
            this_nmax_F = np.max(self.coll_params[beta][:,1])
            tau = self.utils.tau_sum_n0_collider(A[pos_str],B[neg_str],
                                            self.coll_FLLs[beta][this_nmax_F], weights,
                                            4.*np.pi*np.exp(1.0j*self._omega_prime(0,1.0j*nu_s)))
            
        elif nu_s==0:
            this_nmax_F = np.max(self.coll_params[beta][:,1])
            tau = self.utils.tau_sum_nu0_collider(A[pos_str].astype(np.complex128), B[neg_str].astype(np.complex128),
                                                    self.coll_FLLs[beta][this_nmax_F].astype(np.complex128), weights, 
                                                    4.*np.pi/(2.*s+1.)*(-1.)**s)
            
        else:
            this_nmax_F = np.max(self.coll_params[beta][:,1])
            coeffs = np.asarray([np.exp(1.0j*self._omega_prime(s,1.0j*nu_s))*self._calC(s,S,1.0j*nu_s) for S in range(0,2*s+1,2)])
            tau = self.utils.tau_sum_general_collider(A[pos_str], A[neg_str], B[pos_str], B[neg_str], 
                                                      self.coll_FLLs[beta], weights, this_nmax_F, s, beta, np.asarray(coeffs))
        return tau            
        
    @_timer_func('tauNL_products')
    def _A_maps(self, map1, map2, n, add_sym=False, beta=np.inf):
        """Compute [P_nmu Q]_LM for all mu, noting that P_nmu is complex (if n>0) and using conjugation symmetries.
        We optionally add the map1 <-> map2 term."""
        
        def _m(map,component):
            if beta==np.inf or component=='q':
                return map[component]
            else:
                return map['coll-%.8f,%.8fi'%(beta.real,beta.imag)][component]
        
        def _mc(map,component):
            if beta==np.inf or component=='q':
                return map[component]
            else:
                if beta.imag==0:
                    return map['coll-%.8f,%.8fi'%(beta.real,0)][component]
                else:
                    return map['coll-%.8f,%.8fi'%(beta.real,-beta.imag)][component]
        
        if n==0:
            if np.imag(beta)==0:
                # NB: P00 is real here!
                if add_sym:
                    prod_map = multiplyRR_sym(_m(map2,'q'), _m(map1,'p00'), _m(map1,'q'), _m(map2,'p00'), self.base.nthreads)
                else:
                    prod_map = multiplyRR(_m(map2,'q'), _m(map1,'p00'), self.base.nthreads)
                
                out = self.base.to_lm_vec(prod_map,lmax=self.nLmax).T[self.nLminfilt,:]
                return out[None,:]
            
            else:
                if add_sym:
                    prod_map = multiplyRC_sym(_m(map2,'q'), _m(map1,'p00'), _m(map1,'q'), _m(map2,'p00'), self.base.nthreads)
                else:
                    prod_map = multiplyRC(_m(map2,'q'), _m(map1,'p00'), self.base.nthreads)
                
                out = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]
                if (prod_map.imag[:,0]!=0).any():
                    out += 1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                return out[None,:]
    
        else:
            output = []
            
            for mu in range(-n,n+1):
                if mu<0:
                    if add_sym:
                        prod_map = multiplyRCstar_sym(_m(map2,'q'), _mc(map1,'p%d%d'%(n,-mu)), _m(map1,'q'), _mc(map2,'p%d%d'%(n,-mu)), (-1.)**(n+mu), self.base.nthreads)
                    else:
                        prod_map = multiplyRCstar(_m(map2,'q'), _mc(map1,'p%d%d'%(n,-mu)), (-1.)**(n+mu), self.base.nthreads)
                else:
                    if add_sym:
                        prod_map = multiplyRC_sym(_m(map2,'q'), _m(map1,'p%d%d'%(n,mu)), _m(map1,'q'), _m(map2,'p%d%d'%(n,mu)), self.base.nthreads)
                    else:
                        prod_map = multiplyRC(_m(map2,'q'), _m(map1,'p%d%d'%(n,mu)), self.base.nthreads)
                out = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]                    
                if (prod_map[:,0].imag!=0).any():
                    out += 1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                output.append(out)
        return np.asarray(output)
        
    ### OPTIMAL ESTIMATOR
    @_timer_func('numerator')
    def Tl_numerator(self, data, include_disconnected_term=True, verb=False, input_type='map'):
        """
        Compute the numerator of the unwindowed trispectrum estimator for all templates, given some data (either in map or harmonic space).

        We can also optionally switch off the disconnected terms, which affects only parity-conserving trispectra.
        """
        # Check we have initialized correctly
        if self.ints_1d and (not hasattr(self,'r_arr')):
            raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
        if self.ints_2d and (not hasattr(self,'rtau_arr')):
            raise Exception("Need to supply radial/tau integration points or run optimize_radial_sampling_2d()!")
        
        # Check if simulations have been supplied
        if not hasattr(self, 'preload') and include_disconnected_term:
            raise Exception("Need to generate or specify bias simulations!")
        
        # Check input data format
        if self.pol:
            assert len(data)==3, "Data must contain T, Q, U components!"
        else:
            assert (len(data)==1 and len(data[0])==self.base.Npix) or len(data)==self.base.Npix, "Data must contain T only!"

        # Decide whether to compute t0 term, if not already computed
        if hasattr(self, 't0_num') and include_disconnected_term:
            compute_t0 = False
            if verb: print("Using precomputed t0 term")
        else:
            compute_t0 = True

        # Apply S^-1 to data and transform to harmonic space
        t_init = time.time()
        h_data_lm = np.asarray(self.applySinv(data, input_type=input_type, lmax=self.lmax)[:,self.lminfilt], order='C')
        self.timers['Sinv'] += time.time()-t_init
                
        # Compute all relevant weighted maps
        proc_maps = self._apply_all_filters(h_data_lm)
        
        # Define 4-, 2- and 0-field arrays
        t4_num = np.zeros(len(self.templates))
        A_maps_dd = {}
        if not include_disconnected_term:
            print("## No subtraction of (parity-conserving) disconnected terms performed!")
        else:
            t2_num = np.zeros(len(self.templates))
            if compute_t0: t0_num = np.zeros(len(self.templates))
        
        if verb: print("# Assembling trispectrum numerator (4-field term)")
        for ii,t in enumerate(self.templates):
            
            if t=='gNL-loc':
                # gNL local template
                t_init = time.time()
                if verb: print("Computing gNL-loc template")
                t4_num[ii] = 9./100.*self.utils.gnl_loc_sum(self.r_weights[t], proc_maps['p00'], proc_maps['q'])*4.*self.base.A_pix*(4.*np.pi)**(1.5)
                self.timers['gNL_summation'] += time.time()-t_init

            if t=='gNL-con':
                # gNL constant template
                t_init = time.time()
                if verb: print("Computing gNL-con template")
                t4_num[ii] = 9./25.*self.utils.gnl_con_sum(self.r_weights[t], proc_maps['r'])*self.base.A_pix
                self.timers['gNL_summation'] += time.time()-t_init

            if t=='gNL-dotdot':
                # gNL-dot{pi}^4 EFTI shape
                t_init = time.time()
                if verb: print("Computing gNL-dotdot template")
                t4_num[ii] = 384./25.*self.utils.gnl_dotdot_sum(self.rtau_weights[t], self.rtau_arr[:,1], proc_maps['a'])*self.base.A_pix
                self.timers['gNL_summation'] += time.time()-t_init

            if t=='gNL-dotdel':
                # gNL-dot{pi}^2del{pi}^2 EFTI shape
                t_init = time.time()
                if verb: print("Computing gNL-dotdel template")
                t4_num[ii] = 288./325.*self.utils.gnl_dotdel_sum(self.rtau_weights[t], self.rtau_arr[:,1], proc_maps['a'], proc_maps['b'], proc_maps['c'])*12*self.base.A_pix
                self.timers['gNL_summation'] += time.time()-t_init

            if t=='gNL-deldel':
                # gNL-del{pi}^4 EFTI shape
                t_init = time.time()
                if verb: print("Computing gNL-deldel template")
                t4_num[ii] = 1728./2575.*self.utils.gnl_deldel_sum(self.rtau_weights[t],proc_maps['b'],proc_maps['c'])*6*self.base.A_pix
                self.timers['gNL_summation'] += time.time()-t_init

            if t=='tauNL-loc':
                # tauNL-loc template
                if verb: print("Computing tauNL-loc template")

                # Compute A maps (unless already computed)
                if 0 not in A_maps_dd.keys():
                    A_maps_dd[0] = self._A_maps(proc_maps, proc_maps, 0)
                
                # Compute 4-field term (adding factors of Sqrt[4pi] to correct for Y_00 factors]
                t4_num[ii] = 1./24.*self._tau_sum(A_maps_dd, A_maps_dd, self.r_weights[t], 0, 0, 0)*12.*(4.*np.pi)**(3./2.)
                
            if 'tauNL-direc' in t:
                # Direction-dependent tauNL
                n1,n3,n = np.asarray(t.split(':')[1].split(',')).astype(int)
                if verb: print("Computing tauNL-direc(%d,%d,%d) template"%(n1,n3,n))
                
                ## Compute 4-field term
                # Compute A maps (unless already computed)
                if n1 not in A_maps_dd.keys():
                    A_maps_dd[n1] = self._A_maps(proc_maps, proc_maps, n1)
                if n3 not in A_maps_dd.keys():
                    A_maps_dd[n3] = self._A_maps(proc_maps, proc_maps, n3)
                
                # Compute tau from these maps
                t4_num[ii] = self._tau_sum(A_maps_dd, A_maps_dd, self.r_weights[t], n1, n3, n)/48.*24.
                
            if 'tauNL-even' in t:
                # Direction-dependent tauNL (parity-even)
                n = int(t.split(':')[1])
                if verb: print("Computing tauNL-even(%d) template"%n)
                
                if n not in A_maps_dd.keys():
                    A_maps_dd[n] = self._A_maps(proc_maps, proc_maps, n)
                if 0 not in A_maps_dd.keys():
                    A_maps_dd[0] = self._A_maps(proc_maps, proc_maps, 0)
                pref = (-1.)**n*(4.*np.pi)**(3./2.)/np.sqrt(2.*n+1.)
                if n==0:
                    tau = 3*self._tau_sum(A_maps_dd, A_maps_dd, self.r_weights[t], 0, 0, 0)
                else:
                    tau = self._tau_sum(A_maps_dd, A_maps_dd, self.r_weights[t], n, n, 0)
                    tau += 2*self._tau_sum(A_maps_dd, A_maps_dd, self.r_weights[t], 0, n, n)
                
                t4_num[ii] = 1./48.*tau*pref*8.
                
            if 'tauNL-odd' in t:
                # Direction-dependent tauNL (parity-odd)
                n = int(t.split(':')[1])
                n1n3ns, weights = self._decompose_tauNL_odd(n)
                if verb: print("Computing tauNL-odd(%d) template"%n)
                
                # Iterate over n1,n3,n combinations
                tau = 0.
                for ind in range(len(n1n3ns)):
                    n1, n3, n = n1n3ns[ind]
                    
                    # Compute relevant A maps
                    if n1 not in A_maps_dd.keys():
                        A_maps_dd[n1] = self._A_maps(proc_maps, proc_maps, n1)
                    if n3 not in A_maps_dd.keys():
                        A_maps_dd[n3] = self._A_maps(proc_maps, proc_maps, n3)
                
                    # Compute sum of A pairs
                    tau += weights[ind]*self._tau_sum(A_maps_dd, A_maps_dd, self.r_weights[t], n1, n3, n)
                
                # Assemble 4-field term
                t4_num[ii] = 1./48.*tau*24.
                
            if 'tauNL-light' in t:
                # Light particle collider tauNL
                s = int(t.split(':')[1].split(',')[0])
                nu_s = float(t.split(':')[1].split(',')[1])
                if verb: print("Computing tauNL-light(%d,%.2f) template"%(s,nu_s))
                
                # Compute A maps (unless already computed)
                coll_str = 'coll-%d,%.8f,%.8fi'%(s,-1.5+nu_s,0)
                if coll_str not in A_maps_dd.keys():
                    A_maps_dd[coll_str] = self._A_maps(proc_maps, proc_maps, s, beta=-1.5+nu_s)
                
                # Compute 4-field term (internally taking real part)
                t4_num[ii] = self._tau_sum_collider(A_maps_dd, A_maps_dd, self.r_weights[t], s, -1.5+nu_s)/24.*12.
                
            if 'tauNL-heavy' in t:
                # Heavy particle collider tauNL
                s = int(t.split(':')[1].split(',')[0])
                mu_s = float(t.split(':')[1].split(',')[1])
                if verb: print("Computing tauNL-heavy(%d,%.2f) template"%(s,mu_s))
                
                # Compute A maps (unless already computed)
                pos_str = 'coll-%d,%.8f,%.8fi'%(s,-1.5,-1.0*mu_s)
                neg_str = 'coll-%d,%.8f,%.8fi'%(s,-1.5,+1.0*mu_s)
                if pos_str not in A_maps_dd.keys():
                    A_maps_dd[pos_str] = self._A_maps(proc_maps, proc_maps, s, False, beta=-1.5-1.0j*mu_s)
                if neg_str not in A_maps_dd.keys():
                    A_maps_dd[neg_str] = self._A_maps(proc_maps, proc_maps, s, False, beta=-1.5+1.0j*mu_s)
                
                # Compute 4-field term (internally taking real part)
                t4_num[ii] = self._tau_sum_collider(A_maps_dd, A_maps_dd, self.r_weights[t], s, -1.5-1.0j*mu_s)/24.*12.
                
            if t=='lensing':
                # Lensing template
                if verb: print("Computing lensing template")

                ## Compute estimator
                Phi_dd = self._compute_lensing_Phi(proc_maps,proc_maps)
                Ls = self.base.l_arr[(self.base.l_arr>=self.Lmin_lens)*(self.base.l_arr<=self.Lmax_lens)]
                Ms = self.base.m_arr[(self.base.l_arr>=self.Lmin_lens)*(self.base.l_arr<=self.Lmax_lens)]
                t_init = time.time()
                t4_num[ii] = 1./24.*np.sum(Ls*(Ls+1.)*Phi_dd*Phi_dd.conj()*(1.+(Ms>0))*self.C_phi[Ls]).real*12.
                self.timers['lensing_summation'] += time.time()-t_init
               
            if t=='point-source':
                # Point-source template
                t_init = time.time()
                if verb: print("Computing point-source template")
                t4_num[ii] = 1./24.*np.sum(proc_maps['u'][0].real**4.)*self.base.A_pix
                self.timers['gNL_summation'] += time.time()-t_init
            
        if include_disconnected_term:

            # Iterate over simulations
            for isim in range(self.N_it):
                if not compute_t0:
                    if verb: print("# Assembling 2-field trispectrum numerator for simulation pair %d of %d "%(isim+1,self.N_it))
                else:
                    if verb: print("# Assembling 2-field and 0-field trispectrum numerator for simulation pair %d of %d"%(isim+1,self.N_it))

                # Load processed bias simulations
                if self.preload:
                    this_proc_a_maps = self.proc_a_maps[isim]
                    this_proc_b_maps = self.proc_b_maps[isim]
                else:
                    this_proc_a_maps, this_proc_b_maps = self.load_sim_data(isim)
                
                # Define empty exchange dictionaries
                A_maps_aa = {}
                A_maps_ab_sym = {}
                A_maps_bb = {}
                A_maps_ad_sym = {}
                A_maps_bd_sym = {}

                # Compute templates
                for ii,t in enumerate(self.templates):
                    
                    if t=='gNL-loc':
                        t_init = time.time()
                        
                        def _return_perm(map12, map34):
                            return self.utils.gnl_loc_disc_sum(self.r_weights[t], map12['p00'], map34['p00'], map12['q'], map34['q'])
                        
                        # First set of fields
                        summ  = _return_perm(proc_maps, this_proc_a_maps)
                        # Second set of fields
                        summ += _return_perm(proc_maps, this_proc_b_maps)
                        t2_num[ii] += -54./100.*summ/self.N_it*self.base.A_pix*(4.*np.pi)**(1.5)

                        if compute_t0:
                            summ  = _return_perm(this_proc_a_maps, this_proc_b_maps)
                            t0_num[ii] += 54./100.*summ/self.N_it*self.base.A_pix*(4.*np.pi)**(1.5)
                        self.timers['gNL_summation'] += time.time()-t_init

                    if t=='gNL-con':
                        t_init = time.time()
                        
                        def _return_perm(map12, map34):
                            return self.utils.gnl_con_disc_sum(self.r_weights[t], map12['r'], map34['r'])
                        
                        # First set of fields
                        summ = _return_perm(proc_maps, this_proc_a_maps)
                        # Second set of fields
                        summ += _return_perm(proc_maps, this_proc_b_maps)
                        t2_num[ii] += -27./25.*summ/self.N_it*self.base.A_pix

                        if compute_t0:
                            summ = _return_perm(this_proc_a_maps, this_proc_b_maps)
                            t0_num[ii] += 27./25.*summ/self.N_it*self.base.A_pix
                        self.timers['gNL_summation'] += time.time()-t_init

                    if t=='gNL-dotdot':
                        t_init = time.time()
                        
                        def _return_perm(map12,map34):
                            return self.utils.gnl_dotdot_disc_sum(self.rtau_weights[t], self.rtau_arr[:,1], map12['a'], map34['a'])
                        
                        # First set of fields
                        summ  = _return_perm(proc_maps, this_proc_a_maps)
                        # Second set of fields
                        summ += _return_perm(proc_maps, this_proc_b_maps)
                        t2_num[ii] += -1152./25.*summ/self.N_it*self.base.A_pix
                        
                        if compute_t0:
                            summ  = _return_perm(this_proc_a_maps, this_proc_b_maps)
                            t0_num[ii] += 1152./25.*summ/self.N_it*self.base.A_pix
                        self.timers['gNL_summation'] += time.time()-t_init

                    if t=='gNL-dotdel':
                        t_init = time.time()
                        
                        def _return_perm(map1,map2,map3,map4):
                            return self.utils.gnl_dotdel_disc_sum(self.rtau_weights[t], self.rtau_arr[:,1], map1['a'],map2['a'],map3['b'],map4['b'],map3['c'],map4['c'])
                        
                        # First set of fields
                        summ  = 2*_return_perm(this_proc_a_maps,this_proc_a_maps,proc_maps,proc_maps)
                        summ += 8*_return_perm(this_proc_a_maps,proc_maps,this_proc_a_maps,proc_maps)
                        summ += 2*_return_perm(proc_maps,proc_maps,this_proc_a_maps,this_proc_a_maps)
                        # Second set of fields
                        summ += 2*_return_perm(this_proc_b_maps,this_proc_b_maps,proc_maps,proc_maps)
                        summ += 8*_return_perm(this_proc_b_maps,proc_maps,this_proc_b_maps,proc_maps)
                        summ += 2*_return_perm(proc_maps,proc_maps,this_proc_b_maps,this_proc_b_maps)
                        t2_num[ii] += -864./325.*summ/self.N_it*self.base.A_pix

                        if compute_t0:
                            summ  = 2*_return_perm(this_proc_a_maps,this_proc_a_maps,this_proc_b_maps,this_proc_b_maps)
                            summ += 8*_return_perm(this_proc_a_maps,this_proc_b_maps,this_proc_a_maps,this_proc_b_maps)
                            summ += 2*_return_perm(this_proc_b_maps,this_proc_b_maps,this_proc_a_maps,this_proc_a_maps)
                            t0_num[ii] += 864./325.*summ/self.N_it*self.base.A_pix
                        self.timers['gNL_summation'] += time.time()-t_init
                        
                    if t=='gNL-deldel':
                        t_init = time.time()
                        
                        ## Sum over 4 permutations
                        
                        def _return_perm(map1,map2,map3,map4):
                            return self.utils.gnl_deldel_disc_sum(self.rtau_weights[t], map1['b'], map2['b'], map3['b'], map4['b'], map1['c'], map2['c'], map3['c'], map4['c'])
                            
                        # First set of fields
                        summ =  2*_return_perm(this_proc_a_maps,this_proc_a_maps,proc_maps,proc_maps)
                        summ += 4*_return_perm(this_proc_a_maps,proc_maps,this_proc_a_maps,proc_maps)
                        # Second set of fields
                        summ += 2*_return_perm(this_proc_b_maps,this_proc_b_maps,proc_maps,proc_maps)
                        summ += 4*_return_perm(this_proc_b_maps,proc_maps,this_proc_b_maps,proc_maps)
                        t2_num[ii] += -5184./2575.*summ/self.N_it*self.base.A_pix

                        if compute_t0:
                            ## Sum over 4 permutations
                            summ =  2*_return_perm(this_proc_a_maps,this_proc_a_maps,this_proc_b_maps,this_proc_b_maps)
                            summ += 4*_return_perm(this_proc_a_maps,this_proc_b_maps,this_proc_a_maps,this_proc_b_maps)
                            t0_num[ii] += 5184./2575.*summ/self.N_it*self.base.A_pix
                        self.timers['gNL_summation'] += time.time()-t_init

                    if t=='tauNL-loc':
                        
                        # First set of fields
                        if 0 not in A_maps_aa.keys():
                            A_maps_aa[0] = self._A_maps(this_proc_a_maps,this_proc_a_maps, 0)
                            A_maps_ad_sym[0] = self._A_maps(this_proc_a_maps, proc_maps, 0, add_sym=True)
                        tau  = 4*self._tau_sum(A_maps_aa,A_maps_dd, self.r_weights[t], 0, 0, 0)
                        tau += 2*self._tau_sum(A_maps_ad_sym,A_maps_ad_sym, self.r_weights[t], 0, 0, 0)
                        
                        # Second set of fields
                        if 0 not in A_maps_bb.keys():
                            A_maps_bb[0] = self._A_maps(this_proc_b_maps,this_proc_b_maps, 0)
                            A_maps_bd_sym[0] = self._A_maps(this_proc_b_maps, proc_maps, 0, add_sym=True)
                        tau += 4*self._tau_sum(A_maps_bb,A_maps_dd, self.r_weights[t], 0, 0, 0)
                        tau += 2*self._tau_sum(A_maps_bd_sym,A_maps_bd_sym, self.r_weights[t], 0, 0, 0)
                        
                        t2_num[ii] += -1./24.*tau*6/self.N_it/2.*(4.*np.pi)**(3./2.)

                        if compute_t0:
                            # Compute additional (P*Q)_lm fields
                            if 0 not in A_maps_ab_sym.keys():
                                A_maps_ab_sym[0] = self._A_maps(this_proc_a_maps, this_proc_b_maps, 0, add_sym=True)
                            tau  = 4*self._tau_sum(A_maps_aa,A_maps_bb, self.r_weights[t], 0, 0, 0)
                            tau += 2*self._tau_sum(A_maps_ab_sym,A_maps_ab_sym, self.r_weights[t], 0, 0, 0)
                                
                            t0_num[ii] += 1./24.*tau*3/self.N_it*(4.*np.pi)**(3./2.)

                    if 'tauNL-direc' in t:
                        # Direction-dependent tauNL
                        n1,n3,n = np.asarray(t.split(':')[1].split(',')).astype(int)
                        AA_sum = 0.
                        
                        # Compute tau from these maps
                        tmp = 0.
                        for A_maps_xx, A_maps_xd_sym, this_proc_x_maps in zip([A_maps_aa, A_maps_bb],[A_maps_ad_sym, A_maps_bd_sym],[this_proc_a_maps,this_proc_b_maps]):
                            for nn in np.unique([n1,n3]):
                                if nn not in A_maps_xx.keys():
                                    A_maps_xx[nn] = self._A_maps(this_proc_x_maps, this_proc_x_maps, nn)
                                    A_maps_xd_sym[nn] = self._A_maps(this_proc_x_maps, proc_maps, nn, add_sym=True)
                            if n1==n3:
                                tmp += 2*self._tau_sum(A_maps_xx, A_maps_dd, self.r_weights[t], n1, n3, n)
                            else:
                                tmp += self._tau_sum(A_maps_xx,A_maps_dd, self.r_weights[t], n1, n3, n)+self._tau_sum(A_maps_dd, A_maps_xx, self.r_weights[t], n1, n3, n)
                            tmp += self._tau_sum(A_maps_xd_sym,A_maps_xd_sym, self.r_weights[t], n1, n3, n)
                                
                        t2_num[ii] += -1./48.*tmp*6./self.N_it/2.*4. # x4 from dropped perms
                        
                        if compute_t0:
                            # Compute additional (P_nmu*Q)_lm fields
                            tmp = 0.
                            for nn in np.unique([n1,n3]):
                                if nn not in A_maps_ab_sym.keys():
                                    A_maps_ab_sym[nn] = self._A_maps(this_proc_a_maps, this_proc_b_maps, nn, add_sym=True)
                            if n1==n3:
                                tmp += 2*self._tau_sum(A_maps_aa, A_maps_bb, self.r_weights[t], n1, n3, n)
                            else:
                                tmp += self._tau_sum(A_maps_aa, A_maps_bb, self.r_weights[t], n1, n3, n)+self._tau_sum(A_maps_bb, A_maps_aa, self.r_weights[t], n1, n3, n)
                            tmp += self._tau_sum(A_maps_ab_sym, A_maps_ab_sym, self.r_weights[t], n1, n3, n)
                            
                            t0_num[ii] += 1./48.*tmp*3/self.N_it*4. # x4 from dropped perms
                        
                    if 'tauNL-even' in t:
                        # Direction-dependent even tauNL
                        n = int(t.split(':')[1])
                        tau = 0.
                        
                        for A_maps_xx, A_maps_xd_sym, this_proc_x_maps in zip([A_maps_aa, A_maps_bb],[A_maps_ad_sym, A_maps_bd_sym],[this_proc_a_maps,this_proc_b_maps]):
                            for nn in np.unique([n,0]):
                                if nn not in A_maps_xx.keys():
                                    A_maps_xx[nn] = self._A_maps(this_proc_x_maps, this_proc_x_maps, nn)
                                    A_maps_xd_sym[nn] = self._A_maps(this_proc_x_maps, proc_maps, nn, add_sym=True)
                            
                            if n==0:
                                tau += 6*self._tau_sum(A_maps_xx, A_maps_dd, self.r_weights[t], 0, 0, 0)
                                tau += 3*self._tau_sum(A_maps_xd_sym, A_maps_xd_sym, self.r_weights[t], 0, 0, 0)
                            else:    
                                tau += 2*self._tau_sum(A_maps_xx, A_maps_dd, self.r_weights[t], n, n, 0)
                                tau += self._tau_sum(A_maps_xd_sym, A_maps_xd_sym, self.r_weights[t], n, n, 0)
                                tau += 2*(self._tau_sum(A_maps_xx, A_maps_dd, self.r_weights[t], 0, n, n)+self._tau_sum(A_maps_dd, A_maps_xx, self.r_weights[t], 0, n, n))
                                tau += 2*self._tau_sum(A_maps_xd_sym,A_maps_xd_sym, self.r_weights[t], 0, n, n)
                        
                        pref = (-1.)**n/np.sqrt(2.*n+1.)*np.sqrt(4.*np.pi)**3.
                        t2_num[ii] += -1./48.*tau*pref*6./self.N_it/2.*4./3. # x4 from dropped perms

                        if compute_t0:
                            # Compute additional (P_nmu*Q)_lm fields
                            for nn in np.unique([n,0]):
                                if nn not in A_maps_ab_sym.keys():
                                    A_maps_ab_sym[nn] = self._A_maps(this_proc_a_maps, this_proc_b_maps, nn, add_sym=True)
                            if n==0:
                                tau = 6*self._tau_sum(A_maps_aa, A_maps_bb, self.r_weights[t], 0, 0, 0)
                                tau += 3*self._tau_sum(A_maps_ab_sym, A_maps_ab_sym, self.r_weights[t], 0, 0, 0)
                            else:
                                tau = 2*self._tau_sum(A_maps_aa, A_maps_bb, self.r_weights[t], n, n, 0)
                                tau += self._tau_sum(A_maps_ab_sym, A_maps_ab_sym, self.r_weights[t], n, n, 0)
                                tau += 2*(self._tau_sum(A_maps_aa, A_maps_bb, self.r_weights[t], 0, n, n)+self._tau_sum(A_maps_bb, A_maps_aa, self.r_weights[t], 0, n, n))
                                tau += 2*self._tau_sum(A_maps_ab_sym, A_maps_ab_sym, self.r_weights[t], 0, n, n)
                            
                            t0_num[ii] += 1./48.*tau*pref*3./self.N_it*4./3. # x4 from dropped perms
                    
                    if 'tauNL-odd' in t:
                        # Direction-dependent odd tauNL
                        n = int(t.split(':')[1])
                        n1n3ns, weights = self._decompose_tauNL_odd(n)
                        
                        # Assemble sum for each n1,n3,n combination
                        tau_sum = 0.
                        for ind in range(len(n1n3ns)):
                        
                            n1, n3, n = n1n3ns[ind]
                            
                            for A_maps_xx, A_maps_xd_sym, this_proc_x_maps in zip([A_maps_aa, A_maps_bb],[A_maps_ad_sym, A_maps_bd_sym],[this_proc_a_maps,this_proc_b_maps]):
                                if n1 not in A_maps_xx.keys():
                                    A_maps_xx[n1] = self._A_maps(this_proc_x_maps, this_proc_x_maps, n1)
                                    A_maps_xd_sym[n1] = self._A_maps(this_proc_x_maps, proc_maps, n1, add_sym=True)
                                if n3 not in A_maps_xx.keys():
                                    A_maps_xx[n3] = self._A_maps(this_proc_x_maps, this_proc_x_maps, n3)
                                    A_maps_xd_sym[n3] = self._A_maps(this_proc_x_maps, proc_maps, n3, add_sym=True)

                                if n1==n3:
                                    tau = 2*self._tau_sum(A_maps_xx, A_maps_dd, self.r_weights[t], n1, n3, n)
                                else:
                                    tau = self._tau_sum(A_maps_xx, A_maps_dd, self.r_weights[t], n1, n3, n)+self._tau_sum(A_maps_dd, A_maps_xx, self.r_weights[t], n1, n3, n)
                                tau += self._tau_sum(A_maps_xd_sym, A_maps_xd_sym, self.r_weights[t], n1, n3, n)
                                                                    
                                # Add to output    
                                tau_sum += weights[ind]*tau
                        
                        t2_num[ii] += -1./48.*tau_sum*6./self.N_it/2.*4. # x4 from dropped perms

                        if compute_t0:
                                
                            # Assemble sum for each n1,n3,n combination
                            tau_sum = 0.
                            for ind in range(len(n1n3ns)):
                                n1, n3, n = n1n3ns[ind]
                            
                                # Compute additional (P_nmu*Q)_lm fields
                                if n1 not in A_maps_ab_sym.keys():
                                    A_maps_ab_sym[n1] = self._A_maps(this_proc_a_maps, this_proc_b_maps, n1, add_sym=True)
                                if n3 not in A_maps_ab_sym.keys():
                                    A_maps_ab_sym[n3] = self._A_maps(this_proc_a_maps, this_proc_b_maps, n3, add_sym=True)
                                
                                if n1==n3:
                                    tau = 2*self._tau_sum(A_maps_aa, A_maps_bb, self.r_weights[t], n1, n3, n)
                                else:
                                    tau = self._tau_sum(A_maps_aa, A_maps_bb, self.r_weights[t], n1, n3, n)+self._tau_sum(A_maps_bb, A_maps_aa, self.r_weights[t], n1, n3, n)
                                tau += self._tau_sum(A_maps_ab_sym, A_maps_ab_sym, self.r_weights[t], n1, n3, n)
                                                                    
                                # Add to output    
                                tau_sum += weights[ind]*tau
                            
                            # Add to t0                              
                            t0_num[ii] += 1./48.*tau_sum*3./self.N_it*4. # x4 from dropped perms

                    if 'tauNL-light' in t:
                        # Light particle collider tauNL
                        s = int(t.split(':')[1].split(',')[0])
                        nu_s = float(t.split(':')[1].split(',')[1])
                        
                        # Compute 2-field term
                        tau = 0.
                        for A_maps_xx, A_maps_xd_sym, this_proc_x_maps in zip([A_maps_aa, A_maps_bb],[A_maps_ad_sym, A_maps_bd_sym],[this_proc_a_maps,this_proc_b_maps]):
                                        
                            # Compute additional (P^beta_sl*Q)_LM fields
                            coll_str = 'coll-%d,%.8f,%.8fi'%(s,-1.5+nu_s,0)
                            if coll_str not in A_maps_xx.keys():
                                A_maps_xx[coll_str] = self._A_maps(this_proc_x_maps, this_proc_x_maps, s, beta=-1.5+nu_s)
                            if coll_str not in A_maps_xd_sym.keys():
                                A_maps_xd_sym[coll_str] = self._A_maps(this_proc_x_maps, proc_maps, s, add_sym=True, beta=-1.5+nu_s)
                            
                            # Compute 4-field term
                            tau += 2*self._tau_sum_collider(A_maps_xx, A_maps_dd, self.r_weights[t], s, -1.5+nu_s)
                            tau += self._tau_sum_collider(A_maps_xd_sym, A_maps_xd_sym, self.r_weights[t], s, -1.5+nu_s)
                        
                        # Integrate over r, r'
                        t2_num[ii] += -1./24.*tau*6./self.N_it/2.*2. # x2 from dropped perms

                        if compute_t0:
                            
                            # Compute additional (P^beta_sl*Q)_LM fields
                            if coll_str not in A_maps_ab_sym.keys():
                                A_maps_ab_sym[coll_str] = self._A_maps(this_proc_a_maps, this_proc_b_maps, s, add_sym=True, beta=-1.5+nu_s)
                            
                            # Compute 4-field term
                            tau = 2*self._tau_sum_collider(A_maps_aa, A_maps_bb, self.r_weights[t], s, -1.5+nu_s)
                            tau += self._tau_sum_collider(A_maps_ab_sym,A_maps_ab_sym, self.r_weights[t], s, -1.5+nu_s)
                            
                            # Integrate over r, r'
                            t0_num[ii] += 1./24.*tau*3/self.N_it*2. # x2 from dropped perms

                    if 'tauNL-heavy' in t:
                        # Heavy particle collider tauNL
                        s = int(t.split(':')[1].split(',')[0])
                        mu_s = float(t.split(':')[1].split(',')[1])
                        
                        pos_str = 'coll-%d,%.8f,%.8fi'%(s,-1.5,-1.0*mu_s)
                        neg_str = 'coll-%d,%.8f,%.8fi'%(s,-1.5,+1.0*mu_s)
                            
                        # Compute 2-field term
                        tau = 0.
                        for A_maps_xx, A_maps_xd_sym, this_proc_x_maps in zip([A_maps_aa, A_maps_bb],[A_maps_ad_sym, A_maps_bd_sym],[this_proc_a_maps,this_proc_b_maps]):
                            
                            # Compute additional (P^beta_sl*Q)_LM fields
                            if pos_str not in A_maps_xx.keys():
                                A_maps_xx[pos_str] = self._A_maps(this_proc_x_maps, this_proc_x_maps, s, False, beta=-1.5-1.0j*mu_s)
                                A_maps_xd_sym[pos_str] = self._A_maps(this_proc_x_maps, proc_maps, s, True, beta=-1.5-1.0j*mu_s)
                            if neg_str not in A_maps_xx.keys():
                                A_maps_xx[neg_str] = self._A_maps(this_proc_x_maps, this_proc_x_maps, s, False, beta=-1.5+1.0j*mu_s)
                                A_maps_xd_sym[neg_str] = self._A_maps(this_proc_x_maps, proc_maps, s, True, beta=-1.5+1.0j*mu_s)
                            
                            # Compute 4-field term
                            tau += 2*self._tau_sum_collider(A_maps_xx, A_maps_dd, self.r_weights[t], s, -1.5-1.0j*mu_s)
                            tau += self._tau_sum_collider(A_maps_xd_sym,A_maps_xd_sym, self.r_weights[t], s, -1.5-1.0j*mu_s)
                        
                        # Integrate over r, r'
                        t2_num[ii] += -1./24.*tau*6./self.N_it/2.*2. # x2 from dropped perms
                        
                        if compute_t0:
                            
                            # Compute additional (P^beta_sl*Q)_LM fields
                            if pos_str not in A_maps_ab_sym.keys():
                                A_maps_ab_sym[pos_str] = self._A_maps(this_proc_a_maps, this_proc_b_maps, s, True, beta=-1.5-1.0j*mu_s)
                            if neg_str not in A_maps_ab_sym.keys():
                                A_maps_ab_sym[neg_str] = self._A_maps(this_proc_a_maps, this_proc_b_maps, s, True, beta=-1.5+1.0j*mu_s)
                            
                            # Compute 4-field term
                            tau = 2*self._tau_sum_collider(A_maps_aa, A_maps_bb, self.r_weights[t], s, -1.5-1.0j*mu_s)
                            tau += self._tau_sum_collider(A_maps_ab_sym,A_maps_ab_sym, self.r_weights[t], s, -1.5-1.0j*mu_s)
                            
                            # Integrate over r, r'
                            t0_num[ii] += 1./24.*tau*3/self.N_it*2. # x2 from dropped perms
                        
                    if t=='lensing':
                        # Lensing template
                        Ls = self.base.l_arr[(self.base.l_arr>=self.Lmin_lens)*(self.base.l_arr<=self.Lmax_lens)]
                        Ms = self.base.m_arr[(self.base.l_arr>=self.Lmin_lens)*(self.base.l_arr<=self.Lmax_lens)]

                        # First set of fields
                        Phi_aa = self._compute_lensing_Phi(this_proc_a_maps,this_proc_a_maps)
                        Phi_ad = self._compute_lensing_Phi(this_proc_a_maps,proc_maps,add_sym=True)
                        Phi_sum = 4.*Phi_aa*Phi_dd.conj()+2.*Phi_ad*Phi_ad.conj()
                        del Phi_ad
                        
                        # Second set of fields
                        Phi_bb = self._compute_lensing_Phi(this_proc_b_maps,this_proc_b_maps)
                        Phi_bd = self._compute_lensing_Phi(this_proc_b_maps,proc_maps,add_sym=True)
                        Phi_sum += 4.*Phi_bb*Phi_dd.conj()+2.*Phi_bd*Phi_bd.conj()
                        del Phi_bd
                        
                        t_init = time.time()
                        t2_num[ii] += -1./24.*np.sum(Ls*(Ls+1.)*Phi_sum*(1.+(Ms>0))*self.C_phi[Ls]).real*3./self.N_it
                        self.timers['lensing_summation'] += time.time()-t_init
               
                        if compute_t0:
                            Phi_ab = self._compute_lensing_Phi(this_proc_a_maps,this_proc_b_maps,add_sym=True)
                            Phi_sum = 4.*Phi_aa*Phi_bb.conj()+2.*Phi_ab*Phi_ab.conj()
                            del Phi_ab, Phi_aa, Phi_bb
                            
                            t_init = time.time()
                            t0_num[ii] += 1./24.*np.sum(Ls*(Ls+1.)*Phi_sum*(1.+(Ms>0))*self.C_phi[Ls]).real*3./self.N_it
                            self.timers['lensing_summation'] += time.time()-t_init
                        del Phi_sum
                        
                    if t=='point-source':
                        t_init = time.time()
                        
                        def _return_perm(map12, map34):
                            return np.sum(map12['u'][0].real**2.*map34['u'][0].real**2.)
                        
                        # First set of fields
                        summ = _return_perm(proc_maps, this_proc_a_maps)
                        # Second set of fields
                        summ += _return_perm(proc_maps, this_proc_b_maps)
                        t2_num[ii] += -3./24.*summ/self.N_it*self.base.A_pix

                        if compute_t0:
                            summ = _return_perm(this_proc_a_maps, this_proc_b_maps)
                            t0_num[ii] += 3./24.*summ/self.N_it*self.base.A_pix
                        self.timers['gNL_summation'] += time.time()-t_init

            # Load t0 from memory, if already computed
            if not compute_t0:
                t0_num = self.t0_num
            else:
                self.t0_num = t0_num

        if include_disconnected_term:
            t_num = t4_num+t2_num+t0_num
        else:
            t_num = t4_num

        return t_num

    @_timer_func('fish_outer')
    def _assemble_fish(self, Q4_a, Q4_b, sym=False):
        """Compute Fisher matrix between two Q arrays as an outer product. This is parallelized across the l,m axis."""
        return outer_product(Q4_a, Q4_b, self.base.nthreads, sym)
    
    @_timer_func('fish_outer')
    def _assemble_fish_ideal(self, Qa, Qb, sym=False):
        """Compute Fisher matrix between two Q arrays as an outer product. This is parallelized across the template axis."""
        return outer_product_ideal(Qa, Qb, self.base.nthreads, sym)
    
    @_timer_func('fish_products')
    def _compute_PQ_vecs(self, Pn3_maps, Q_maps, i, j, n3=0, add_sym=False):
        """Compute all [P_{n3mu3} Q]_{LM}(r) maps for given input maps P_n3mu3, Q."""

        # Treat n3=0 separately (since this uses real maps)
        if n3==0: 
            
            # Compute product map
            if add_sym:
                prod_map = multiplyRR_sym(Q_maps[j], Pn3_maps[i,0], Q_maps[i], Pn3_maps[j,0], self.base.nthreads)
            else:
                prod_map = multiplyRR(Q_maps[j], Pn3_maps[i,0], self.base.nthreads)
                
            # Take SHT, being careful of real / imaginary parts
            PQ_vecs = self.base.to_lm_vec(prod_map,lmax=self.nLmax).T[None,self.nLminfilt,:]
            
        else:
            # Output vector
            PQ_vecs = np.zeros((2*n3+1,np.sum(self.nLminfilt),len(Q_maps[0])),dtype=np.complex128)

            # Iterate over mu3
            for mu3 in range(n3+1):
                
                # Compute product map
                if add_sym:
                    prod_map = multiplyRC_sym(Q_maps[j], Pn3_maps[i,mu3], Q_maps[i], Pn3_maps[j,mu3], self.base.nthreads)
                else:
                    prod_map = multiplyRC(Q_maps[j], Pn3_maps[i,mu3], self.base.nthreads)
                
                # Take SHT, being careful of real / imaginary parts
                if mu3==0 and n3%2==0:
                    # Real only!
                    PQ_vecs[n3] = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]
                elif mu3==0 and n3%2==1:
                    # Imaginary only!
                    PQ_vecs[n3] = 1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                else:
                    pq_real = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]
                    pq_imag = 1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                    PQ_vecs[n3+mu3] = pq_real+pq_imag
                    PQ_vecs[n3-mu3] = (-1.)**(n3-mu3)*(pq_real-pq_imag)
                    del pq_real, pq_imag
        
        return PQ_vecs

    @_timer_func('fish_products')
    def _compute_coll_PQ_vecs(self, Ps_maps, Q_maps, i, j, s, beta_coll, add_sym=False, conjPs_maps=[]):
        """Compute all [P_{s,lam}^beta Q]_{LM} maps for given input maps P_s,lam, Q. We also compute the conjugate maps if non-trivial."""
       
        ## Simplified form for s = 0
        if s==0:
            
            # Compute product map
            if add_sym:
                prod_map = multiplyRC_sym(Q_maps[j], Ps_maps[i,0], Q_maps[i], Ps_maps[j,0], self.base.nthreads)
            else:
                prod_map = multiplyRC(Q_maps[j], Ps_maps[i,0], self.base.nthreads)
            
            # Take SHT, being careful of real / imaginary parts
            if np.imag(beta_coll)==0:
                PQ_vecs = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]
                return PQ_vecs[None]
            
            else:
                pq_real = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]
                pq_imag = self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                PQ_vecs, conjPQ_vecs = complex_to_complex(pq_real, pq_imag, self.base.nthreads)
                return PQ_vecs.T[None], conjPQ_vecs.T[None]

        ## Full form for s > 0
        else:
        
            # Use symmetries if conjP = P
            if len(conjPs_maps)==0:
                assert np.imag(beta_coll)==0.
                
                # Define output vector
                PQ_vecs = np.zeros((2*s+1,np.sum(self.nLminfilt),len(Q_maps[0])),dtype=np.complex128)
                
                # Iterate over lam3
                for lam3 in range(s+1):
                    
                    # Compute product map
                    if add_sym:
                        prod_map = multiplyRC_sym(Q_maps[j], Ps_maps[i,lam3], Q_maps[i], Ps_maps[j,lam3], self.base.nthreads)
                    else:
                        prod_map = multiplyRC(Q_maps[j], Ps_maps[i,lam3], self.base.nthreads)
                    
                    # Take SHT, being careful of real / imaginary parts
                    if lam3==0 and s%2==0 and np.imag(beta_coll)==0:
                        # Purely real
                        PQ_vecs[s+lam3] = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]
                    elif lam3==0 and s%2==1 and np.imag(beta_coll)==0:
                        # Purely imaginary
                        PQ_vecs[s+lam3] = 1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                    else:
                        # Complex
                        pq_real = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]
                        pq_imag = 1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                        PQ_vecs[s+lam3] = pq_real+pq_imag
                        if lam3!=0:
                            PQ_vecs[s-lam3] = (-1.)**(s-lam3)*(pq_real-pq_imag)
                        del pq_real, pq_imag
                            
                return PQ_vecs
                
            else:
                
                # Define output vectors
                PQ_vecs = np.zeros((2*s+1,np.sum(self.nLminfilt),len(Q_maps[0])),dtype=np.complex128)
                conjPQ_vecs = np.zeros((2*s+1,np.sum(self.nLminfilt),len(Q_maps[0])),dtype=np.complex128)
                
                if s==0:
                    
                    # Compute product map
                    if add_sym:
                        prod_map = multiplyRC_sym(Q_maps[j], Ps_maps[i,0], Q_maps[i], Ps_maps[j,0], self.base.nthreads)
                    else:
                        prod_map = multiplyRC(Q_maps[j], Ps_maps[i,0], self.base.nthreads)
                                        
                    # Take SHT, being careful of real / imaginary parts
                    PQ_vecs[0] = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]+1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                    
                    ## Compute conjugates (beta -> - beta)
                    # Compute product map
                    if add_sym:
                        prod_map = multiplyRC_sym(Q_maps[j], conjPs_maps[i,0], Q_maps[i], conjPs_maps[j,0], self.base.nthreads)
                    else:
                        prod_map = multiplyRC(Q_maps[j], conjPs_maps[i,0], self.base.nthreads)
                    
                    # Take SHT (noting that imaginary part is non-trivial)
                    conjPQ_vecs[0] = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]+1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                    
                else:
                    # Iterate over lam3
                    for lam3 in range(-s,s+1):
                        
                        # Compute product map
                        if lam3<0:
                            if add_sym:
                                prod_map = multiplyRCstar_sym(Q_maps[j], conjPs_maps[i,-lam3], Q_maps[i], conjPs_maps[j,-lam3], (-1.)**(s+lam3), self.base.nthreads)
                            else:
                                prod_map = multiplyRCstar(Q_maps[j], conjPs_maps[i,-lam3], (-1.)**(s+lam3), self.base.nthreads)
                        else:
                            if add_sym:
                                prod_map = multiplyRC_sym(Q_maps[j], Ps_maps[i,lam3], Q_maps[i], Ps_maps[j,lam3], self.base.nthreads)
                            else:
                                prod_map = multiplyRC(Q_maps[j], Ps_maps[i,lam3], self.base.nthreads)
                                        
                        # Take SHT, being careful of real / imaginary parts
                        PQ_vecs[s+lam3] = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]+1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                        
                    # Compute conjugates (beta -> -beta) if needed
                    if np.imag(beta_coll)==0:
                        conjPQ_vecs = PQ_vecs
                        
                    else:
                        # Compute product map
                        for lam3 in range(-s,s+1):
                            if lam3<0:
                                if add_sym:
                                    prod_map = multiplyRCstar_sym(Q_maps[j], Ps_maps[i,-lam3], Q_maps[i], Ps_maps[j,-lam3], (-1.)**(s+lam3), self.base.nthreads)
                                else:
                                    prod_map = multiplyRCstar(Q_maps[j], Ps_maps[i,-lam3], (-1.)**(s+lam3), self.base.nthreads)
                            else:
                                if add_sym:
                                    prod_map = multiplyRC_sym(Q_maps[j], conjPs_maps[i,lam3], Q_maps[i], conjPs_maps[j,lam3], self.base.nthreads)
                                else:
                                    prod_map = multiplyRC(Q_maps[j], conjPs_maps[i,lam3], self.base.nthreads)
                                    
                            # Take SHT (noting that imaginary part is non-trivial)
                            conjPQ_vecs[s+lam3] = self.base.to_lm_vec(prod_map.real,lmax=self.nLmax).T[self.nLminfilt,:]+1.0j*self.base.to_lm_vec(prod_map.imag,lmax=self.nLmax).T[self.nLminfilt,:]
                    
                return PQ_vecs, conjPQ_vecs

    @_timer_func('fish_convolve')
    def _convolve_F(self, PQ_vecs, r_weights, n1=0, n3=0, n=0, inds=[]):
        """
        Compute Sum_{mu3 mu} ThreeJ[n1, n3, n, mu1, mu3, mu] Sum_LM (-1)^M Y_LM ∫r'^2 dr' Sum_{L'M'} i^{L-L'} G^{LL'n}_{(-M)M'mu} F_LL'(r, r') [P_{n3mu3} Q]_{L' M'}(r') for all mu1.
        
        For n1=n3=n=0 this simplifies to
        Sum_LM Y_LM^* ∫r'^2 dr' F_L(r, r') [P Q]_{L' M'}(r').
        
        We optionally restrict to a reduced set of r,r' indices of F_LL(r,r') [specified by "inds"].
        """
        if len(inds)==0:
            inds = np.arange(self.N_r, dtype=np.int32)
        
        # Special case for isotropic basis
        if n1==0 and n3==0:
            
            # Compute convolution
            out_real = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128)
            self.utils.convolve_F_n000(PQ_vecs[0], self.FLLs, r_weights, out_real, self.nmax_F, inds)            
            
            # Add to output
            out_vec = self.base.to_map_vec(out_real.T,lmax=self.nLmax)[None]
            
        # General case
        else:
            
            # Output array
            out_vec = np.zeros((n1+1,len(inds),self.base.Npix),dtype=np.complex128)

            # Iterate over mu1
            for mu1 in range(n1+1):
                
                # Define buffer vectors of each parity       
                out_real = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
                out_imag = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
    
                # Apply convolution
                if n==0:
                    self.utils.convolve_F_n0(PQ_vecs[n1-mu1], PQ_vecs[n1+mu1], self.FLLs, r_weights, out_real, out_imag, n1, mu1, self.nmax_F, inds)
                else:
                    self.utils.convolve_F_general(PQ_vecs, self.FLLs, r_weights, out_real, out_imag, n1, n3, n, mu1, inds)
                
                # Return to map-space
                if (out_imag!=0.).any():
                    r2cstar_inplace(self.base.to_map_vec(out_real.T,lmax=self.nLmax), self.base.to_map_vec(1.0j*out_imag.T,lmax=self.nLmax), out_vec, mu1, self.base.nthreads)
                else:
                    out_vec[mu1] = self.base.to_map_vec(out_real.T,lmax=self.nLmax)
            
        return out_vec

    @_timer_func('fish_convolve')
    def _convolve_coll_F(self, PQ_vecs, r_weights, s, beta_coll, inds=[], conjPQ_vecs=[]):
        """
        Compute Sum_{lam lam3} ThreeJ[s, s, S, lam1, lam3, lam] Sum_LM (-1)^M Y_LM ∫r'^2 dr' Sum_{L'M'} i^{L-L'} G^{LL'S}_{(-M)M'lam} F_LL'^beta(r, r') [P^beta_{slam3} Q]_{L' M'}(r') for all lam1.
        
        For s=n=0 this simplifies to
        Sum_LM Y_LM^* ∫r'^2 dr' F^beta_LL(r, r') [P^beta Q]_{L' M'}(r'). This is specifically for the collider templates.
        
        We optionally restrict to a reduced set of r,r' indices of F_LL'^beta(r,r') [specified by "inds"].
        """
        if len(inds)==0:
            inds = np.arange(self.N_r, dtype=np.int32)
        
        nu_s = beta_coll+1.5
            
        this_nmax_F = np.max(self.coll_params[beta_coll][:,1])
        
        ## Special case for spin-0
        if s==0:
            
            # Real maps
            if np.imag(beta_coll)==0:
                
                # Compute convolution
                out_real = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
                self.utils.convolve_F_n000_real(PQ_vecs[0], self.coll_FLLs[beta_coll], r_weights, out_real, this_nmax_F, inds)
                out_real *= (4.*np.pi)**(3./2.)
                    
                # Add to output
                return self.base.to_map_vec(out_real.T,lmax=self.nLmax)[None]
               
            else:
                
                # Compute spin weights
                S_weights = np.asarray([self._calC(0,0,1.0j*nu_s)*np.exp(1.0j*self._omega_prime(0,1.0j*nu_s))])
                
                # Define buffer vectors of each parity
                out_real = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
                out_imag = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
                
                # Compute convolution
                self.utils.convolve_F_general_collider(PQ_vecs, conjPQ_vecs, self.coll_FLLs[beta_coll], r_weights, out_real, out_imag, 0, 0, this_nmax_F, S_weights, inds)
                
                # Compute SHTs
                return r2cstar(self.base.to_map_vec(out_real.T,lmax=self.nLmax),self.base.to_map_vec(1.0j*out_imag.T,lmax=self.nLmax), self.base.nthreads)[None]
                
        ## General spin
        else:
                
            # Compute spin weights
            S_weights = np.asarray([self._calC(s,S,1.0j*nu_s)*np.exp(1.0j*self._omega_prime(s,1.0j*nu_s)) for S in range(0,2*s+1,2)])

            if np.imag(beta_coll)==0:
                
                # Output array
                out_vec = np.zeros((s+1,len(inds),self.base.Npix),dtype=np.complex128)
                
                # Iterate over lam1
                for lam1 in range(s+1):
    
                    # Define buffer vectors of each parity
                    out_real = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
                    out_imag = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
                
                    # Compute convolution, summed over S, lam3m lam, etc.
                    self.utils.convolve_F_general_collider(PQ_vecs, PQ_vecs, self.coll_FLLs[beta_coll], r_weights, out_real, out_imag, s, lam1, this_nmax_F, S_weights, inds)
                    
                    # Compute SHTs
                    if (out_imag!=0).any():
                        r2cstar_inplace(self.base.to_map_vec(out_real.T,lmax=self.nLmax),self.base.to_map_vec(1.0j*out_imag.T,lmax=self.nLmax), out_vec, lam1, self.base.nthreads)
                    else:
                        out_vec[lam1] = self.base.to_map_vec(out_real.T,lmax=self.nLmax)
                    
            else:
                
                # Output array
                out_vec = np.zeros((2*s+1,len(inds),self.base.Npix),dtype=np.complex128)

                # Iterate over lam1
                for lam1 in range(-s,s+1):
                    
                    # Define buffer vectors of each parity
                    out_real = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
                    out_imag = np.zeros((len(self.nLminfilt),len(inds)),dtype=np.complex128,order='C')
                        
                    # Compute convolution, summed over S, lam3m lam, etc.
                    self.utils.convolve_F_general_collider(PQ_vecs, conjPQ_vecs, self.coll_FLLs[beta_coll], r_weights, out_real, out_imag, s, lam1, this_nmax_F, S_weights, inds)
                    
                    # Compute SHTs
                    if (out_imag!=0).any():
                        r2cstar_inplace(self.base.to_map_vec(out_real.T,lmax=self.nLmax),self.base.to_map_vec(1.0j*out_imag.T,lmax=self.nLmax), out_vec, s+lam1, self.base.nthreads)
                    else:
                        out_vec[s+lam1] = self.base.to_map_vec(out_real.T,lmax=self.nLmax)
                    
        return out_vec

    def _F_PQ(self, Pn3_maps, Q_maps, r_weights, n1=0, n3=0, n=0, inds=[]):
        """Wrapper function to compute F_L * [P_{nmu} Q]_LM maps for all mu1. 
        
        We make use of the conjugate symmetry Conj[F_PQ[n1, mu1]] = (-1)^(n1+mu1)F_PQ[n1, -mu1]."""
        
        # Compute F_L * [P_{nmu} Q]_LM maps
        F_vecs = np.zeros((3,n1+1,self.N_r,self.base.Npix),dtype=np.complex128)
        
        for i,i1,i2 in zip([0,1,2], # index for output array,
                           [0,1,0], # which primary map
                           [0,1,1], # which secondary map 
                           ):
            
            # Compute PQ vectors
            PQ_vecs = self._compute_PQ_vecs(Pn3_maps, Q_maps, i1, i2, n3, add_sym=(i1!=i2))
            
            # Convolve with F
            F_vecs[i] = self._convolve_F(PQ_vecs, r_weights, n1, n3, n, inds)
            
        return F_vecs
    
    def _coll_F_PQ(self, Ps_maps, Q_maps, r_weights, s, beta_coll, conjPs_maps=[], inds=[]):
        """Wrapper function to compute F^beta_L * [P^beta_{s,lam} Q]_LM maps for all lam1.
        
        If beta is real, we make use of the conjugate symmetry Conj[F_PQ[beta, s, lam1]] = (-1)^(s+lam1)F_PQ[Conj[beta], s, -lam1]."""
        
        # Compute F_L * [P_{nmu} Q]_LM maps
        if np.imag(beta_coll)==0:
            F_vecs = np.zeros((3,s+1,self.N_r,self.base.Npix),dtype=np.complex128)
        else:
            F_vecs = np.zeros((3,2*s+1,self.N_r,self.base.Npix),dtype=np.complex128) # symmetry is broken for beta != beta*
            
        for i,i1,i2 in zip([0,1,2], # index for output array,
                           [0,1,0], # which primary map
                           [0,1,1], # which secondary map 
                           ):
            
            if np.imag(beta_coll)==0:
                # Compute PQ vectors
                PQ_vecs = self._compute_coll_PQ_vecs(Ps_maps, Q_maps, i1, i2, s, beta_coll, add_sym=(i1!=i2))
                
                # Convolve with F
                F_vecs[i] = self._convolve_coll_F(PQ_vecs, r_weights, s, beta_coll, inds)
                
            else:
                # Compute PQ vectors and conjugates
                PQ_vecs, conjPQ_vecs = self._compute_coll_PQ_vecs(Ps_maps, Q_maps, i1, i2, s, beta_coll, add_sym=(i1!=i2), conjPs_maps = conjPs_maps)
                
                # Convolve with F
                F_vecs[i] = self._convolve_coll_F(PQ_vecs, r_weights, s, beta_coll, inds, conjPQ_vecs=conjPQ_vecs)
                
        return F_vecs
        
    @_timer_func('fish_deriv')
    def _Q4_contributions(self, Pn1_maps, Q_maps, F_PQ, pq_inds, F_inds, r_weights, n1=0, beta_coll=np.inf, conjPn1_maps=[], weights=[1.], inds=[], with_deriv=False, del_weights=[]):
        """Compute 1/2 (q_l^X(r) ∫ Y_lm^*(r) P_{n1mu1}[r] + ∫r^2dr Sum_{L1M1} i^{L1-l} (-1)^mu1 Gaunt[l,L1,n1,m,M1,-mu1]p_{l L1}^X(r) ∫r^2dr Y_{L1M1}(r)Q[r])FPQ_{n1,n3,n,mu1}[r]).
        
        If n1=n3=n=0 this simplifies to 1/2 [(q_l^X(r) ∫ Y_lm^*(r) P[r] FPQ_{0,0,0,0}[r] + ∫r^2dr p_l^X(r) ∫r^2dr Y^*_{lm}(r)Q[r])FPQ_{0,0,0,0}[r])]/sqrt(4pi). 
        
        We optionally include collider terms if beta_coll is specified. If "with_deriv" is specified, we also compute the derivative with respect to the weights. 
        
        Note that we pass all possible P, Q, F maps and indices specifying which to use in computation.
        """
        
        # Write arrays in correct Cython types
        weights = np.asarray(weights,dtype=np.float64)
        pq_inds = np.asarray(pq_inds, dtype=np.int32)
        F_inds = np.asarray(F_inds, dtype=np.int32)
        
        # Define output vectors
        output = np.zeros((1+2*self.pol,np.sum(self.lfilt)), dtype=np.complex128, order='C')
        if with_deriv:
            output2 = np.zeros((len(inds),1+2*self.pol,np.sum(self.lfilt)), dtype=np.complex128, order='C')
        
        if n1==0:
            
            ## Step 1: compute q_l [P FPQ_{}]_lm piece (considering only even-parity piece)
            # Define product map
            # prod_map = np.zeros((len(inds),self.base.Npix),dtype=np.float64)
            # for i in range(len(weights)):
            #     compute_productP(Pn1_maps[pq_inds[i],0], F_PQ[F_inds[i],0], weights[i], prod_map, self.base.nthreads)
            
            ## Step 1: compute q_l [P FPQ_{}]_lm piece (considering only even-parity piece)
            # Define product map
            if beta_coll==np.inf:
                prod_map = compute_productP_real_all(Pn1_maps[:,0], F_PQ[:,0], weights, pq_inds, F_inds, self.base.nthreads)
            else:
                prod_map = compute_productP_all(Pn1_maps[:,0], F_PQ[:,0], weights, pq_inds, F_inds, self.base.nthreads)
            
            # Compute [P FPQ_{}]_lm
            P_FPQ = self.base.to_lm_vec(prod_map,lmax=self.lmax).T[self.lminfilt]
            del prod_map
            
            # Integrate over r, weighted by p or q, and sum
            integrate_pq(self.qlXs, P_FPQ, self.ls, r_weights, output, inds, self.base.nthreads)
            
            # Optionally add derivative
            if with_deriv:
                integrate_pq_deriv(self.qlXs, P_FPQ, self.ls, del_weights, output2, inds, self.base.nthreads)
            del P_FPQ
            
            ## Step 2: compute p_ll [Q FPQ_{}]_lm piece
            
            if np.imag(beta_coll)!=0:
                
                # Define product map
                prod_map = compute_productQ_complex_all(Q_maps, F_PQ, weights, pq_inds, F_inds, 0, self.base.nthreads)
                
                # Compute [Q FPQ_{}]_lm and (-1)^m [Q FPQ_{}]^*_{l-m}
                Q_FPQ_plus, Q_FPQ_minus = complex_to_complex_transpose(self.base.to_lm_vec(prod_map.real,lmax=self.lmax).T[self.lminfilt], self.base.to_lm_vec(prod_map.imag,lmax=self.lmax).T[self.lminfilt], self.base.nthreads)
                del prod_map
                
            else:
                
                # Define product map
                prod_map = compute_productQ_real_all(Q_maps, F_PQ, weights, pq_inds, F_inds, 0, self.base.nthreads)
                
                # Compute [Q FPQ_{}]_lm
                Q_FPQ_plus = self.base.to_lm_vec(prod_map,lmax=self.lmax).T[self.lminfilt]
                del prod_map
                
            # Compute p*Q_FPQ, dropping the imaginary piece
            if beta_coll!=np.inf:
                this_nmax = np.max(self.coll_params[beta_coll][:,0])
                
                if np.imag(beta_coll)!=0:
                    
                    # Define output (dropping the -M piece which cancels, and adding 1/sqrt(4pi) from Gaunt factor)
                    integrate_pq_complex(self.coll_plLXs[beta_coll][this_nmax], Q_FPQ_plus, Q_FPQ_minus, self.ls, r_weights/np.sqrt(4.*np.pi), output, inds, self.base.nthreads)
                    
                    if with_deriv:
                        integrate_pq_complex_deriv(self.coll_plLXs[beta_coll][this_nmax], Q_FPQ_plus, Q_FPQ_minus, self.ls, del_weights/np.sqrt(4.*np.pi), output2, inds, self.base.nthreads)
                    
                else:
                    
                    # Define output (dropping the -M piece which cancels, and adding 1/sqrt(4pi) from Gaunt factor)
                    integrate_pq(np.asarray(self.coll_plLXs[beta_coll][this_nmax].real,order='C'), Q_FPQ_plus, self.ls, r_weights/np.sqrt(4.*np.pi), output, inds, self.base.nthreads)
                    
                    # Repeat for derivative
                    if with_deriv:
                        integrate_pq_deriv(np.asarray(self.coll_plLXs[beta_coll][this_nmax].real,order='C'), Q_FPQ_plus, self.ls, del_weights/np.sqrt(4.*np.pi), output2, inds, self.base.nthreads)
            
            else:
                
                # Define output (dropping the -M piece which cancels, and adding 1/sqrt(4pi) from Gaunt factor)
                integrate_pq(self.plLXs[self.nmax], Q_FPQ_plus, self.ls, r_weights/np.sqrt(4.*np.pi), output, inds, self.base.nthreads)
            
                # Repeat for derivative
                if with_deriv:
                    integrate_pq_deriv(self.plLXs[self.nmax], Q_FPQ_plus, self.ls, r_weights/np.sqrt(4.*np.pi), output2, inds, self.base.nthreads)
                
        else:
            
            ## Step 1: compute q_l [P FPQ_{}]_lm piece (considering only even-parity piece)
            if np.imag(beta_coll)!=0:    
                prod_map = compute_productPnmu_all(Pn1_maps, conjPn1_maps, F_PQ, weights, n1, pq_inds, F_inds, self.base.nthreads)
            else:
                prod_map = compute_productPnmu_sym_all(Pn1_maps, F_PQ, weights, n1, pq_inds, F_inds, self.base.nthreads)
            
            # Compute [P FPQ_{}]_lm (considering only even-parity piece)
            P_FPQ = self.base.to_lm_vec(prod_map,lmax=self.lmax).T[self.lminfilt]
            del prod_map
            
            # Integrate over r, weighted by p or q, and sum
            integrate_pq(self.qlXs, P_FPQ, self.ls, r_weights, output, inds, self.base.nthreads)
            
            # Optionally add derivative
            if with_deriv:
                integrate_pq_deriv(self.qlXs, P_FPQ, self.ls, del_weights, output2, inds, self.base.nthreads)
            del P_FPQ
            
            ## Step 2: compute p_lL [Q FPQ_{}]_lm piece
            for mu1 in range(-n1,n1+1):
                
                if np.imag(beta_coll)!=0:
                    mu1_index = n1+mu1
                    #this_F_PQ = np.asarray(F_PQ[:,n1+mu1],order='C')
                else:    
                    if mu1<0: continue # computed from symmetry
                    mu1_index = mu1
                    
                if (F_PQ[F_inds[0],mu1_index,:,0].imag!=0).any():
                    
                    # Define product map
                    prod_map = compute_productQ_complex_all(Q_maps, F_PQ, weights, pq_inds, F_inds, mu1_index, self.base.nthreads)
                    
                    # Compute [Q FPQ_{}]_lm for m>=0 and m<0 (no conjugate symmetries needed!)
                    Q_FPQ_plus, Q_FPQ_minus = complex_to_complex(self.base.to_lm_vec(prod_map.real,lmax=min([self.lmax+n1,self.base.lmax])),
                                                                 self.base.to_lm_vec(prod_map.imag,lmax=min([self.lmax+n1,self.base.lmax])),
                                                                 self.base.nthreads)

                    del prod_map
                    
                else:
                
                    # Define product map
                    prod_map = compute_productQ_real_all(Q_maps, F_PQ, weights, pq_inds, F_inds, mu1_index, self.base.nthreads)
                    
                    # Compute [Q FPQ_{}]_lm
                    Q_FPQ_plus = np.asarray(self.base.to_lm_vec(prod_map,lmax=min([self.lmax+n1,self.base.lmax])).T,dtype=np.complex128,order='C')
                    Q_FPQ_minus = Q_FPQ_plus
                
                # Compute product with p, summed over l',r and add to output
                if np.imag(beta_coll)!=0:
                    this_nmax = np.max(self.coll_params[beta_coll][:,0])
                    self.utils.compute_pF_map_complex(Q_FPQ_plus, Q_FPQ_minus, self.coll_plLXs[beta_coll],
                                        r_weights, n1, mu1, this_nmax, min([self.lmax+n1,self.base.lmax]), output, inds)
                    
                    if with_deriv:
                        self.utils.compute_pF_map_complex_deriv(Q_FPQ_plus, Q_FPQ_minus, self.coll_plLXs[beta_coll],
                                        del_weights, n1, mu1, this_nmax, min([self.lmax+n1,self.base.lmax]), output2, inds)
                elif beta_coll!=np.inf:
                    this_nmax = np.max(self.coll_params[beta_coll][:,0])
                    re_plLX = np.asarray(self.coll_plLXs[beta_coll].real,order='C')
                    self.utils.compute_pF_map_real(Q_FPQ_plus, Q_FPQ_minus, re_plLX,
                                            r_weights, n1, mu1, this_nmax, min([self.lmax+n1,self.base.lmax]), output, inds)
                    
                    if with_deriv:
                        self.utils.compute_pF_map_real_deriv(Q_FPQ_plus, Q_FPQ_minus, re_plLX,
                                        del_weights, n1, mu1, this_nmax, min([self.lmax+n1,self.base.lmax]), output2, inds)
                    
                else:
                    self.utils.compute_pF_map_real(Q_FPQ_plus, Q_FPQ_minus, self.plLXs,r_weights, n1, mu1, self.nmax, min([self.lmax+n1,self.base.lmax]), output, inds)
                    
                    if with_deriv:
                        self.utils.compute_pF_map_real_deriv(Q_FPQ_plus, Q_FPQ_minus, self.plLXs,
                                                             del_weights, n1, mu1, self.nmax, min([self.lmax+n1,self.base.lmax]), output2, inds)
        
        # Return outputs
        if with_deriv:
            return output, output2
        else:
            return output
    
    def _get_Q4_perms(self, Pn1_maps, Q_maps, F_vecs, r_weights, n1=0, beta_coll=np.inf, conjPn1_maps=[], with_deriv=False, inds=[], del_weights=[]):
        """This is a wrapper function to compute all Q4 contributions, and, optionally the derivatives with respect to the weights."""
        
        # Optionally restrict to a subset of terms
        if len(inds)==0:
            inds = np.arange(self.N_r, dtype=np.int32)
        
        if with_deriv:
            assert len(del_weights)==len(r_weights), "Must specify derivative of weights!"
        
        # Compute Q4 terms and (optionally) derivatives
        if with_deriv:
            Qs = np.zeros((4,1+2*self.pol,np.sum(self.lfilt)), dtype=np.complex128, order='C')
            dQs = np.zeros((4,len(inds),1+2*self.pol,np.sum(self.lfilt)), dtype=np.complex128, order='C')
            Qs[0], dQs[0] = self._Q4_contributions(Pn1_maps, Q_maps, F_vecs, [0], [0], r_weights, weights=[12.], conjPn1_maps = conjPn1_maps, 
                                                   n1=n1, beta_coll=beta_coll, inds=inds, with_deriv=True, del_weights=del_weights)
            Qs[1], dQs[1] = self._Q4_contributions(Pn1_maps, Q_maps, F_vecs, [1], [1], r_weights, weights=[12.], conjPn1_maps = conjPn1_maps, 
                                                   n1=n1, beta_coll=beta_coll, inds=inds, with_deriv=True, del_weights=del_weights)
            Qs[2], dQs[2] = self._Q4_contributions(Pn1_maps, Q_maps, F_vecs, [1,0], [0,2], r_weights, weights=[4.,4.], conjPn1_maps = conjPn1_maps,
                                                   n1=n1, beta_coll=beta_coll, inds=inds, with_deriv=True, del_weights=del_weights)
            Qs[3], dQs[3] = self._Q4_contributions(Pn1_maps, Q_maps, F_vecs, [0,1], [1,2], r_weights, weights=[4.,4.], conjPn1_maps = conjPn1_maps,
                                                   n1=n1, beta_coll=beta_coll, inds=inds, with_deriv=True, del_weights=del_weights)
            return Qs, dQs
        else:
            Qs = np.zeros((4,1+2*self.pol,np.sum(self.lfilt)), dtype=np.complex128, order='C')
            Qs[0] = self._Q4_contributions(Pn1_maps, Q_maps, F_vecs, [0], [0], r_weights, weights=[12.], conjPn1_maps = conjPn1_maps,
                                           n1=n1, beta_coll=beta_coll, inds=inds, with_deriv=False)
            Qs[1] = self._Q4_contributions(Pn1_maps, Q_maps, F_vecs, [1], [1], r_weights, weights=[12.], conjPn1_maps = conjPn1_maps,
                                           n1=n1, beta_coll=beta_coll, inds=inds, with_deriv=False)
            Qs[2] = self._Q4_contributions(Pn1_maps, Q_maps, F_vecs, [1,0], [0,2], r_weights, weights=[4.,4.], conjPn1_maps = conjPn1_maps, 
                                           n1=n1, beta_coll=beta_coll, inds=inds, with_deriv=False)
            Qs[3] = self._Q4_contributions(Pn1_maps, Q_maps, F_vecs, [0,1], [1,2], r_weights, weights=[4.,4.], conjPn1_maps = conjPn1_maps, 
                                           n1=n1, beta_coll=beta_coll, inds=inds, with_deriv=False)
            return Qs
    
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
    def _transform_maps(self, map123, flXs, weights, spin=0):
        """Compute Sum_i w_i M_LM f^X_L(i) for real-space map M(n). We optionally average over spins."""
        output = np.zeros((1+2*self.pol,np.sum(self.lfilt)),dtype='complex')
        if spin==0:
            lm_map = np.asarray(self.base.to_lm_vec(map123,lmax=self.lmax)[:,self.lminfilt],order='C')
            return self.utils.radial_sum(lm_map, weights, flXs)
            # return np.sum(self.base.to_lm_vec(map123,lmax=self.lmax).T[self.lminfilt,None,:]*flXs*weights,axis=2).T
        elif spin==1:
            lm_map = np.asarray(self.base.to_lm_vec([map123,map123.conj()],spin=1,lmax=self.lmax)[:,:,self.lminfilt],order='C')
            return self.utils.radial_sum_spin1(lm_map, weights, flXs)
            # return 0.5*np.sum((np.array([1,-1])[:,None,None]*self.base.to_lm_vec([map123,map123.conj()],spin=1,lmax=self.lmax)).sum(axis=0).T[self.lminfilt,None,:]*flXs*weights,axis=2).T
        else:
            raise Exception(f"Wrong spin s = {spin}!")
        
    @_timer_func('fisher')
    def compute_fisher_contribution(self, seed, verb=False):
        """
        This computes the contribution to the Fisher matrix from a single pair of GRF simulations, created internally.
        """
        # Check we have initialized correctly
        if self.ints_1d and (not hasattr(self,'r_arr')):
            raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
        if self.ints_2d and (not hasattr(self,'rtau_arr')):
            raise Exception("Need to supply radial/tau integration points or run optimize_radial_sampling_2d()!")

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
        def compute_Q4(weighting):
            """
            Assemble and return an array of Q4 maps in real- or harmonic-space, for S^-1 or A^-1 weighting. 

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
            if 'r' in self.to_compute:
                if verb: print("Creating R maps")
                R_maps = self._filter_pair(Uinv_a_lms, 'R')
            if 'a' in self.to_compute:
                if verb: print("Creating A maps")
                A_maps = self._filter_pair(Uinv_a_lms, 'A')
            if 'b' in self.to_compute:
                if verb: print("Creating B maps")
                B_maps = self._filter_pair(Uinv_a_lms, 'B')
            if 'c' in self.to_compute:
                if verb: print("Creating C maps")
                C_maps = self._filter_pair(Uinv_a_lms, 'C')
            if 'u' in self.to_compute:
                if verb: print("Creating U maps")
                U_maps = self._filter_pair(Uinv_a_lms, 'U')
            if 'v' in self.to_compute:
                if verb: print("Creating V maps")
                V_maps = self._filter_pair(Uinv_a_lms, 'V')
            
            # Compute all P maps
            if verb and len(self.p_inds)>0: print("Creating P_{n,mu} maps")
            P_maps = self._filter_pair(Uinv_a_lms, 'P')
            
            # Compute all collider P maps
            if verb and len(self.coll_params.keys())>0: print("Creating collider P^beta_{s,lam} maps")
            coll_P_maps = self._filter_pair(Uinv_a_lms, 'coll-P')
            
            # Define output arrays (Q111, Q222, Q112, Q122)
            Qs = np.zeros((4,len(self.templates),1+2*self.pol,np.sum(self.lfilt)),dtype=np.complex128,order='C')
            
            # Helper functions
            def _add_to_Q(n1,n3,n,w):
                """Process a given (n1,n3,n) triplet and add it to the tauNL Q-derivative. This takes also a weight w."""
                
                # Compute F_L * [P_{nmu} Q]_LM maps
                F_vecs = self._F_PQ(P_maps[n3], Q_maps, self.r_weights[t], n1, n3, n)
                
                # Compute Q4 maps
                Qs[:,ii] += w*self._get_Q4_perms(P_maps[n1], Q_maps, F_vecs, self.r_weights[t], n1)
                
            def _add_to_Q_coll(beta, s):
                """Process a given collider field and add it to the tauNL Q-derivative."""
                
                # Compute Sum_S C_s(S, i nu_s)F^{-2beta}_L * [P^beta_{sl} Q]_LM maps
                if np.imag(beta)!=0:
                    
                    # Compute F_L^beta * [P^beta_{slam} Q]_LM maps
                    F_vecs = self._coll_F_PQ(coll_P_maps[beta,s], Q_maps, self.r_weights[t], s, beta, conjPs_maps=coll_P_maps[np.conj(beta),s]) 
                    
                    # Compute Q4 maps
                    Qs[:,ii] += self._get_Q4_perms(coll_P_maps[beta,s], Q_maps, F_vecs, self.r_weights[t], s, beta, conjPn1_maps=coll_P_maps[np.conj(beta),s])
                    
                else:
                    # Compute F_L^beta * [P^beta_{slam} Q]_LM maps
                    F_vecs = self._coll_F_PQ(coll_P_maps[beta,s], Q_maps, self.r_weights[t], s, beta) 
                    
                    # Compute Q4 maps
                    Qs[:,ii] += self._get_Q4_perms(coll_P_maps[beta,s], Q_maps, F_vecs, self.r_weights[t], s, beta)
                    
            # Compute products (with symmetries)
            for ii,t in enumerate(self.templates):
                
                if t=='gNL-loc':
                    
                    if verb: print("Computing Q-derivative for gNL-loc")
                    P0_maps = P_maps[0][:,0]*np.sqrt(4.*np.pi)
                    
                    # 111 
                    Qs[0,ii]  = 162./25.*self._transform_maps(self.utils.multiply(P0_maps[0],P0_maps[0],Q_maps[0]),
                                                              self.plLXs[self.nmax],self.r_weights[t])
                    Qs[0,ii] += 54./25.*self._transform_maps(self.utils.multiply(P0_maps[0],P0_maps[0],P0_maps[0]),
                                                             self.qlXs,self.r_weights[t])

                    # 222
                    Qs[1,ii]  = 162./25.*self._transform_maps(self.utils.multiply(P0_maps[1],P0_maps[1],Q_maps[1]),
                                                              self.plLXs[self.nmax],self.r_weights[t])
                    Qs[1,ii] += 54./25.*self._transform_maps(self.utils.multiply(P0_maps[1],P0_maps[1],P0_maps[1]),
                                                             self.qlXs,self.r_weights[t])

                    # 112
                    Qs[2,ii]  = 54./25.*self._transform_maps(self.utils.multiply_asym(P0_maps[0],P0_maps[1],Q_maps[0],Q_maps[1]),
                                                             self.plLXs[self.nmax],self.r_weights[t])
                    Qs[2,ii] += 54./25.*self._transform_maps(self.utils.multiply(P0_maps[0],P0_maps[0],P0_maps[1]),
                                                             self.qlXs,self.r_weights[t])

                    # 122
                    Qs[3,ii]  = 54./25.*self._transform_maps(self.utils.multiply_asym(P0_maps[1],P0_maps[0],Q_maps[1], Q_maps[0]),
                                                             self.plLXs[self.nmax],self.r_weights[t])
                    Qs[3,ii] += 54./25.*self._transform_maps(self.utils.multiply(P0_maps[0],P0_maps[1],P0_maps[1]),
                                                             self.qlXs,self.r_weights[t])

                if t=='gNL-con':
                    if verb: print("Computing Q-derivative for gNL-con")
                    
                    # Compute all fields 
                    Qs[0,ii]  = 216./25.*self._transform_maps(self.utils.multiply(R_maps[0],R_maps[0],R_maps[0]),
                                                              self.rlXs,self.r_weights[t])
                    Qs[1,ii]  = 216./25.*self._transform_maps(self.utils.multiply(R_maps[1],R_maps[1],R_maps[1]),
                                                              self.rlXs,self.r_weights[t])
                    Qs[2,ii]  = 216./25.*self._transform_maps(self.utils.multiply(R_maps[0],R_maps[0],R_maps[1]),
                                                              self.rlXs,self.r_weights[t])
                    Qs[3,ii]  = 216./25.*self._transform_maps(self.utils.multiply(R_maps[0],R_maps[1],R_maps[1]),
                                                              self.rlXs,self.r_weights[t])
                
                if t=='gNL-dotdot':
                    if verb: print("Computing Q-derivative for gNL-dotdot")
                    
                    # Compute all fields 
                    Qs[0,ii]  = 9216./25.*self._transform_maps(self.utils.multiply(A_maps[0],A_maps[0],A_maps[0]),
                                                               self.alXs*self.rtau_arr[:,1][None,None,:]**4,self.rtau_weights[t])
                    Qs[1,ii]  = 9216./25.*self._transform_maps(self.utils.multiply(A_maps[1],A_maps[1],A_maps[1]),
                                                               self.alXs*self.rtau_arr[:,1][None,None,:]**4,self.rtau_weights[t])
                    Qs[2,ii]  = 9216./25.*self._transform_maps(self.utils.multiply(A_maps[0],A_maps[0],A_maps[1]),
                                                               self.alXs*self.rtau_arr[:,1][None,None,:]**4,self.rtau_weights[t])
                    Qs[3,ii]  = 9216./25.*self._transform_maps(self.utils.multiply(A_maps[0],A_maps[1],A_maps[1]),
                                                               self.alXs*self.rtau_arr[:,1][None,None,:]**4,self.rtau_weights[t])
                
                if t=='gNL-dotdel':
                    if verb: print("Computing Q-derivative for gNL-dotdel")
                    
                    # 111 
                    Qs[0,ii]  = 41472./325.*self._transform_maps(self.utils.multiplyCC(A_maps[0],B_maps[0],C_maps[0]),
                                                                   self.alXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t])
                    Qs[0,ii] += 41472./325.*self._transform_maps(self.utils.multiply(A_maps[0],A_maps[0],B_maps[0]),
                                                                   self.blXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t])
                    Qs[0,ii] += 41472./325.*self._transform_maps(self.utils.multiplyC(A_maps[0],A_maps[0],C_maps[0]),
                                                                   self.clXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t],spin=1)
                    
                    # 222
                    Qs[1,ii]  = 41472./325.*self._transform_maps(self.utils.multiplyCC(A_maps[1],B_maps[1],C_maps[1]),
                                                                   self.alXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t])
                    Qs[1,ii] += 41472./325.*self._transform_maps(self.utils.multiply(A_maps[1],A_maps[1],B_maps[1]),
                                                                   self.blXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t])
                    Qs[1,ii] += 41472./325.*self._transform_maps(self.utils.multiplyC(A_maps[1],A_maps[1],C_maps[1]),
                                                                   self.clXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t],spin=1)
                    
                    # 112
                    Qs[2,ii]  = 13824./325.*self._transform_maps(self.utils.multiplyCC_asym(A_maps[0],A_maps[1],B_maps[0],B_maps[1],C_maps[0],C_maps[1]),
                                                                   self.alXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t])
                    Qs[2,ii] += 13824./325.*self._transform_maps(self.utils.multiply_asym(A_maps[0],A_maps[1],B_maps[0], B_maps[1]),
                                                                   self.blXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t])
                    Qs[2,ii] += 13824./325.*self._transform_maps(self.utils.multiplyC_asym(A_maps[0],A_maps[1],C_maps[0],C_maps[1]),
                                                                   self.clXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t],spin=1)
                    
                    # 122
                    Qs[3,ii]  = 13824./325.*self._transform_maps(self.utils.multiplyCC_asym(A_maps[1],A_maps[0],B_maps[1],B_maps[0],C_maps[1],C_maps[0]),
                                                                   self.alXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t])
                    Qs[3,ii] += 13824./325.*self._transform_maps(self.utils.multiply_asym(A_maps[1],A_maps[0],B_maps[1],B_maps[0]),
                                                                   self.blXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t])
                    Qs[3,ii] += 13824./325.*self._transform_maps(self.utils.multiplyC_asym(A_maps[1],A_maps[0],C_maps[1],C_maps[0]),
                                                                   self.clXs*self.rtau_arr[:,1][None,None,:]**2.,self.rtau_weights[t],spin=1)
                    
                if t=='gNL-deldel':
                    if verb: print("Computing Q-derivative for gNL-deldel")
                    
                    # Precompute useful quantities
                    quad_11 = self.utils.multiply2(B_maps[0],B_maps[0],C_maps[0],C_maps[0])
                    quad_12 = self.utils.multiply2(B_maps[0],B_maps[1],C_maps[0],C_maps[1])
                    quad_22 = self.utils.multiply2(B_maps[1],B_maps[1],C_maps[1],C_maps[1])
                    
                    # 111
                    Qs[0,ii]  = 248772./2575.*self._transform_maps(self.utils.multiply2R(B_maps[0],quad_11),
                                                                   self.blXs,self.rtau_weights[t])
                    Qs[0,ii] += 248772./2575.*self._transform_maps(self.utils.multiply2C(C_maps[0],quad_11),
                                                                   self.clXs,self.rtau_weights[t],spin=1)
                    
                    # 222
                    Qs[1,ii]  = 248772./2575.*self._transform_maps(self.utils.multiply2R(B_maps[1],quad_22),
                                                                   self.blXs,self.rtau_weights[t])
                    Qs[1,ii] += 248772./2575.*self._transform_maps(self.utils.multiply2C(C_maps[1],quad_22),
                                                                   self.clXs,self.rtau_weights[t],spin=1)
                    
                    # 112 
                    Qs[2,ii]  = 82944./2575.*self._transform_maps(self.utils.multiply2R(B_maps[1],quad_11),
                                                                  self.blXs,self.rtau_weights[t])
                    Qs[2,ii] += 82944./2575.*self._transform_maps(self.utils.multiply2C(C_maps[1],quad_11),
                                                                  self.clXs,self.rtau_weights[t],spin=1)
                    Qs[2,ii] += 165888./2575.*self._transform_maps(self.utils.multiply2R(B_maps[0],quad_12),
                                                                   self.blXs,self.rtau_weights[t])
                    Qs[2,ii] += 165888./2575.*self._transform_maps(self.utils.multiply2C(C_maps[0],quad_12),
                                                                   self.clXs,self.rtau_weights[t],spin=1)
                    
                    # 122
                    Qs[3,ii]  = 82944./2575.*self._transform_maps(self.utils.multiply2R(B_maps[0],quad_22),
                                                                  self.blXs,self.rtau_weights[t])
                    Qs[3,ii] += 82944./2575.*self._transform_maps(self.utils.multiply2C(C_maps[0],quad_22),
                                                                  self.clXs,self.rtau_weights[t],spin=1)
                    Qs[3,ii] += 165888./2575.*self._transform_maps(self.utils.multiply2R(B_maps[1],quad_12),
                                                                   self.blXs,self.rtau_weights[t])
                    Qs[3,ii] += 165888./2575.*self._transform_maps(self.utils.multiply2C(C_maps[1],quad_12),
                                                                   self.clXs,self.rtau_weights[t],spin=1)
                    
                if t=='tauNL-loc':
                    # Local tauNL
                    if verb: print("Computing Q-derivative for tauNL-loc")
                    
                    # Add to output array, including all permutations and weighting by (4pi)^{3/2} to correct for different conventions
                    _add_to_Q(0,0,0,(4.*np.pi)**1.5)
                    
                if 'tauNL-direc' in t:
                    # Direction-dependent tauNL
                    n1,n3,n = np.asarray(t.split(':')[1].split(',')).astype(int)
                    if verb: print("Computing Q-derivative for tauNL-direc(%d,%d,%d)"%(n1,n3,n))
                    
                    # Add to output array, incorporating symmetries
                    if n1==n3:
                        _add_to_Q(n1, n3, n, 1.)
                    else:
                        _add_to_Q(n1, n3, n, 0.5)
                        _add_to_Q(n3, n1, n, 0.5*(-1.)**(n1+n3))
                        
                if 'tauNL-even' in t:
                    # Direction-dependent even tauNL
                    n = int(t.split(':')[1])
                    if verb: print("Computing Q-derivative for tauNL-even(%d)"%n)
                    
                    # Compute the decomposition into n1, n3, n pieces
                    uniq_n1n3ns, uniq_weights = self._decompose_tauNL_even(n)
                    
                    # Add to output array for each choice of (n1,n3,n), including symmetries
                    for n_it in range(len(uniq_n1n3ns)):
                        n1,n3,n = uniq_n1n3ns[n_it]
                        if n1==n3:
                            _add_to_Q(n1, n3, n, uniq_weights[n_it])
                        else:
                            _add_to_Q(n1, n3, n, 0.5*uniq_weights[n_it])
                            _add_to_Q(n3, n1, n, 0.5*(-1.)**(n1+n3)*uniq_weights[n_it])
                    
                if 'tauNL-odd' in t:
                    # Direction-dependent odd tauNL
                    n = int(t.split(':')[1])
                    if verb: print("Computing Q-derivative for tauNL-odd(%d)"%n)
                    
                    # Compute the decomposition into n1, n3, n pieces
                    uniq_n1n3ns, uniq_weights = self._decompose_tauNL_odd(n)
                    
                    # Add to output array for each choice of (n1,n3,n)
                    for n_it in range(len(uniq_n1n3ns)):
                        n1,n3,n = uniq_n1n3ns[n_it]
                        if n1==n3:
                            _add_to_Q(n1, n3, n, uniq_weights[n_it])
                        else:
                            _add_to_Q(n1, n3, n, 0.5*uniq_weights[n_it])
                            _add_to_Q(n3, n1, n, 0.5*(-1.)**(n1+n3)*uniq_weights[n_it])
                
                if 'tauNL-light' in t:
                    # Light particle collider tauNL
                    s = int(t.split(':')[1].split(',')[0])
                    nu_s = float(t.split(':')[1].split(',')[1])
                    if verb: print("Computing Q-derivative for tauNL-light(%d,%.2f)"%(s,nu_s))
                    
                    # Add to output array
                    _add_to_Q_coll(-1.5+nu_s,s)
                    
                if 'tauNL-heavy' in t:
                    # Heavy particle collider tauNL
                    s = int(t.split(':')[1].split(',')[0])
                    mu_s = float(t.split(':')[1].split(',')[1])
                    if verb: print("Computing Q-derivative for tauNL-heavy(%d,%.2f)"%(s,mu_s))
                    
                    # Add to output array (only one conjugate needed here!)
                    _add_to_Q_coll(-1.5-1.0j*mu_s,s)
                    
                if t=='lensing':
                    # Lensing A_lens estimator
                    fields1 = {'u':U_maps[0],'v':V_maps[0]}
                    fields2 = {'u':U_maps[1],'v':V_maps[1]}
                    if verb: print("Computing Q-derivative for lensing")
                    
                    def _compute_lensing_W(maps1,maps2,add_sym=False):
                        tmp_lm = np.zeros(len(self.Lminfilt_lens),dtype=np.complex128)
                        tmp_lm[self.Lminfilt_lens] = Ls*(Ls+1.)*self._compute_lensing_Phi(maps1,maps2,add_sym=add_sym)*self.C_phi[Ls]
                        return self.base.to_map_spin(tmp_lm,-tmp_lm,spin=1,lmax=self.Lmax_lens)[0]
                        
                    # Compute all W maps
                    Ls = self.base.l_arr[(self.base.l_arr>=self.Lmin_lens)*(self.base.l_arr<=self.Lmax_lens)]
                    W_11 = _compute_lensing_W(fields1,fields1)
                    W_12sym = _compute_lensing_W(fields1,fields2,add_sym=True)
                    W_22 = _compute_lensing_W(fields2,fields2)
                    
                    def _get_Q(fields,W_maps):
                        
                        Qlm = np.zeros((1+2*self.pol,np.sum(self.lfilt)),dtype=np.complex128)
                        ls = self.ls
                        lpref0 = 0.5*np.sqrt(ls*(ls+1.))
                        Cl_TTs = self.C_lens_weight['TT'][ls]
                        if self.pol:
                            lprefp1 = 0.25*np.sqrt((ls+2.)*(ls-1.))
                            lprefm1 = 0.25*np.sqrt((ls-2.)*(ls+3.))
                            Cl_TEs = self.C_lens_weight['TE'][ls]
                            Cl_EEs = self.C_lens_weight['EE'][ls]
                            Cl_BBs = self.C_lens_weight['BB'][ls]
                            
                        ## Compute first term
                        # Spin-0
                        input_map = np.real(fields['v'][0]*W_maps)
                        Qlm[0] = -self.base.to_lm([input_map],lmax=self.lmax)[0][self.lminfilt]
                        
                        if self.pol:
                            # Spin>0
                            input_map = fields['v'][1]*W_maps-fields['v'][2]*W_maps.conj()
                            lm_map_plus, lm_map_minus = self.base.to_lm_spin(input_map,input_map.conj(),spin=2,lmax=self.lmax)[:,self.lminfilt]
                            Qlm[1] = -0.25*(lm_map_plus+lm_map_minus)
                            Qlm[2] = 0.25j*(lm_map_plus-lm_map_minus)
                                   
                        ## Compute second term
                        # Z = T
                        input_map = fields['u'][0]*W_maps
                        tmp_lm_plus = np.sum(np.array([1,-1])[:,None]*self.base.to_lm_spin(input_map, input_map.conj(),spin=1,lmax=self.lmax),axis=0)[self.lminfilt]
                        # add to X = T
                        Qlm[0] += lpref0*Cl_TTs*tmp_lm_plus
                        if self.pol:
                            # also add to X = E
                            Qlm[1] += lpref0*Cl_TEs*tmp_lm_plus
                        # Z = E
                        if self.pol:
                            # lam=+1
                            input_map = -fields['u'][1]*W_maps.conj()
                            tmp_lm_plus, tmp_lm_minus = self.base.to_lm_spin(input_map,input_map.conj(),spin=1,lmax=self.lmax)[:,self.lminfilt]
                            base = lprefp1*(tmp_lm_plus-tmp_lm_minus)
                            Qlm[0] += Cl_TEs*base
                            Qlm[1] += Cl_EEs*base
                            Qlm[2] += lprefp1*(-1.0j)*Cl_BBs*(tmp_lm_plus+tmp_lm_minus)
                            # lam=-1
                            input_map = fields['u'][1]*W_maps
                            tmp_lm_plus, tmp_lm_minus = self.base.to_lm_spin(input_map,input_map.conj(),spin=3,lmax=self.lmax)[:,self.lminfilt]
                            base = lprefm1*(tmp_lm_plus-tmp_lm_minus)
                            Qlm[0] += Cl_TEs*base
                            Qlm[1] += Cl_EEs*base
                            Qlm[2] += lprefm1*(-1.0j)*Cl_BBs*(tmp_lm_plus+tmp_lm_minus)    
                        # Z = B
                        if self.pol:
                            # lam=+1
                            input_map = -fields['u'][2]*W_maps.conj()
                            tmp_lm_plus, tmp_lm_minus = self.base.to_lm_spin(input_map,input_map.conj(),spin=1,lmax=self.lmax)[:,self.lminfilt]
                            base = lprefp1*1.0j*(tmp_lm_plus+tmp_lm_minus)
                            Qlm[0] += Cl_TEs*base
                            Qlm[1] += Cl_EEs*base
                            Qlm[2] += lprefp1*Cl_BBs*(tmp_lm_plus-tmp_lm_minus)
                            # lam=-1
                            input_map = fields['u'][2]*W_maps
                            tmp_lm_plus, tmp_lm_minus = self.base.to_lm_spin(input_map,input_map.conj(),spin=3,lmax=self.lmax)[:,self.lminfilt]
                            base = lprefm1*1.0j*(tmp_lm_plus+tmp_lm_minus)
                            Qlm[0] += Cl_TEs*base
                            Qlm[1] += Cl_EEs*base
                            Qlm[2] += lprefm1*Cl_BBs*(tmp_lm_plus-tmp_lm_minus)
                        
                        return Qlm/2. # halving to get correct symmetries later
                    
                    ### Assemble outputs
                    # 111
                    Qs[0,ii] += 12*_get_Q(fields1, W_11)
                    
                    # 222
                    Qs[1,ii] += 12*_get_Q(fields2, W_22)
                    
                    # 112
                    Qs[2,ii] += 4*_get_Q(fields1, W_12sym)
                    Qs[2,ii] += 4*_get_Q(fields2, W_11)

                    # 122
                    Qs[3,ii] += 4*_get_Q(fields2, W_12sym)
                    Qs[3,ii] += 4*_get_Q(fields1, W_22)
                    
                    del fields1, fields2, W_11, W_22, W_12sym
                
                if t=='point-source':
                    if verb: print("Computing Q-derivative for point sources")
                    
                    def point_source_Q(product_map):
                        # Output array
                        q = np.zeros((1+2*self.pol,np.sum(self.lminfilt)),dtype=np.complex128,order='C')
                        # Fill X=T element only!
                        q[0] = self.base.to_lm(product_map[None].real,lmax=self.lmax)[0,self.lminfilt]
                        return q

                    # Compute all fields 
                    Qs[0,ii]  = point_source_Q(U_maps[0][0]**3)
                    Qs[1,ii]  = point_source_Q(U_maps[1][0]**3)
                    Qs[2,ii]  = point_source_Q(U_maps[0][0]**2*U_maps[1][0])
                    Qs[3,ii]  = point_source_Q(U_maps[1][0]**2*U_maps[0][0])
                    
            if weighting=='Ainv' and verb: print("Applying S^-1 weighting to output")
            for qindex in range(4):
                self._weight_Q_maps(Qs[qindex], weighting)
            
            return Qs.reshape(4,len(self.templates),-1)

        # Compute Q4 maps
        if verb: print("\n# Computing Q4 map for S^-1 weighting")
        Q4_Sinv = compute_Q4('Sinv')
        if verb: print("\n# Computing Q4 map for A^-1 weighting")
        Q4_Ainv = compute_Q4('Ainv')
        
        # Assemble Fisher matrix
        if verb: print("Assembling Fisher matrix")
        
        # Compute Fisher matrix as an outer product
        fish = self._assemble_fish(Q4_Sinv, Q4_Ainv, sym=False)
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
    
    def Tl_unwindowed(self, data, fish=[], include_disconnected_term=True, verb=False, input_type='map'):
        """
        Compute the unwindowed trispectrum estimator for all combinations of fields.
        
        The code either uses pre-computed Fisher matrices or reads them in on input. 
        
        Note that we return the imaginary part of odd-parity trispectra.

        We can also optionally switch off the disconnected terms.
        """
        if verb: print("")

        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)

        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix first!")
        
        # Compute numerator
        Tl_num = self.Tl_numerator(data, verb=verb, include_disconnected_term=include_disconnected_term)

        # Apply normalization
        Tl_out = np.matmul(self.inv_fish,Tl_num)

        # Create output dictionary
        Tl_dict = {}
        index = 0
        # Iterate over fields
        for t in self.templates:
            Tl_dict[t] = Tl_out[index]
            index += 1
            
        return Tl_dict

    ### OPTIMIZATION
    @_timer_func('optimization')
    def optimize_radial_sampling_1d(self, reduce_r=1, tolerance=1e-3, N_fish_optim=1, stalled_iterations=np.inf, N_split=None, split_index=None, initial_r_points=None, verb=False, optimize_weights = False, optimize_skip=0, repeat_trials=True, r_guess = [], weight_guess = {}):
        """
        Compute the 1D radial sampling points and weights via optimization (as in Smith & Zaldarriaga 06), up to some tolerance in the Fisher distance.
        Optimization will be done for each template, starting with exchange forms. For contact shapes we will analytically compute the 'distance' between template approximations, whilst for exchange trispectra this will be computed using N_fish_optim Monte Carlo iterations.
        
        Main Inputs:
            - reduce_r: Downsample the number of points in the starting radial integral grid (default: 1)
            - tolerance: Convergence threshold for the optimization (default 1e-3). This indicates the approximate error in the Fisher matrix induced by the optimization.
            - N_fish_optim: Number of Monte Carlo iterations to use when computing the tauNL Fisher matrices (default: 1). Typically, 1-5 is sufficient.
            - stalled_iterations (optional): If step, exit the optimization if the Fisher distance has failed to improve by 5% over this many iterations.
        
        For large problems, it is too expensive to optimize the whole matrix at once. Instead, we can split the optimization into N_split pieces, each of which is optimized separately.
        Following this, we perform a final optimization of all N_split components, using the union of all previously obtained radial points. 
        
        Additional Inputs (for chunked computations):
            - N_split (optional): Number of chunks to split the optimization into. If None, no splitting is performed.
            - split_index (optional): Index of the chunk to optimize. 
            - initial_r_points (optional): Starting set of radial points (used for the final optimization step).
        
        The other input parameters are deprecated and may be removed in future releases.
        """
        assert self.ints_1d, "Only 2D sampling is required for these templates!"
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
        self._prepare_templates(ints_1d=True,ints_2d=False,ints_coll=self.ints_coll)
        
        # Reorder templates such that tauNL pieces are processed first (these are most expensive)
        ordered_templates = [tem for tem in self.templates if 'tauNL' in tem]
        ordered_templates += [tem for tem in self.templates if ('tauNL' not in tem) and (tem in self.all_templates_1d)]
        
        # Create list of radial indices in the optimized representation
        inds = []
        for i in range(len(r_guess)):
            index = np.where(r_guess[i]==r_init)[0]
            if len(index)==0:
                raise Exception("r_guess was computed with different settings! All points in r_guess must be in r_init.")
            inds.append(index[0])
        
        inds_init = np.arange(self.N_r)
        
        if len(weight_guess.keys())>0:
            assert len(inds)>0, "Must supply r_guess!"
            if verb: print("Starting from a set of %d radial points"%(len(inds)))
            for t in ordered_templates:
                if 'tauNL' in t: assert t in weight_guess.keys(), "weight_guess does not contain template %d"%template
            inds_guess = inds
        
        # Compute all Fisher matrix derivatives of interest
        if verb: print("Computing all Fisher matrix derivatives")
        derivs = self._compute_fisher_derivatives(ordered_templates, N_fish_optim=N_fish_optim, verb=verb)
        
        # Save ideal Fisher matrices
        if not hasattr(self, 'ideal_fisher'):
            self.ideal_fisher = {}
        for t in ordered_templates:
            self.ideal_fisher[t] = derivs[t][0]
        
        for template in ordered_templates:
            
            if 'tauNL' in template:
                if optimize_weights:
                    if verb: print("\nRunning optimization for template %s (numerically maximizing each step)"%(template))
                else:
                    if verb: print("\nRunning optimization for template %s"%(template))                    
            else:
                if verb: print("\nRunning optimization for template %s"%template)
            
            # Compute Fisher matrix derivative
            init_score, deriv_matrix = derivs[template][0], derivs[template][1]
            if verb: print("Initial score: %.2e"%init_score)
            
            # Load pre-computed functions, if necessary
            if 'tauNL' in template:
                all_Uinv_lms, all_Q4 = derivs[template][2], derivs[template][3]
                all_PQ_vecs = [{} for _ in range(N_fish_optim)]
                all_conjPQ_vecs = [{} for _ in range(N_fish_optim)]
                all_maps = [{} for _ in range(N_fish_optim)]
            
            def _compute_score(w_vals, scaling=1, gradient=True, full_score=False, update=False, replace=False, initialize=False):
                """Compute the Fisher distance between templates given weights w_vals. This optionally computes the gradients.
                
                We note that the gradient computation assumes that the Monte Carlo sum has converged (to absorb expensive terms by symmetry)."""
                
                if 'gNL' in template:
                    if full_score:
                        return np.sum(G_mat), np.sum(np.outer(w_vals,w_vals)*deriv_matrix[inds][:,inds])
                    else:
                        return np.sum(G_mat)
                
                elif 'tauNL' in template:
            
                    # Score functions
                    def _tau_direc_score(n1n3ns, weights):
                        """Compute the score and its derivative for a direction-dependent tau-NL template."""
                        tscore = 0.
                        del_score = 0.
                        fscore = 0.
                        
                        # Define indices as an arrray
                        ainds = np.asarray(inds,dtype=np.int32)
                        
                        def _update_maps(next_ind, seed, n1s, n3s):
                            """Compute weighted maps for new radial index and add to lists."""
                            
                            def _add(field, data, axis=1):
                                """Add new maps to list of maps"""
                                if field not in all_maps[seed].keys():
                                    all_maps[seed][field] = np.asarray(data,order='C')
                                elif replace:
                                    if axis==1:
                                        all_maps[seed][field][:,-1] = np.asarray(data,order='C')[:,0]
                                    elif axis==2:
                                        all_maps[seed][field][:,:,-1] = np.asarray(data,order='C')[:,:,0]
                                    else: 
                                        raise Exception()
                                else:
                                    all_maps[seed][field] = np.asarray(np.concatenate([all_maps[seed][field], data], axis=axis),order='C')
                                
                            # Compute weighted maps
                            next_Q_map = np.asarray([self._compute_weighted_maps(np.asarray(Uinv_a_lm,order='C'), self.qlXs[:,:,[next_ind]]) for Uinv_a_lm in all_Uinv_lms[seed]],dtype=np.float64,order='C')
                            _add('q',next_Q_map)
                            
                            next_P_maps = {}
                            for n in np.unique(np.asarray(list(n1s)+list(n3s))):
                                if n==0:
                                    # Real maps for n = 0
                                    next_Pn_maps = np.zeros((2,n+1,1,self.base.Npix),dtype=np.float64)
                                    for i,Uinv_a_lm in enumerate(all_Uinv_lms[seed]):
                                        next_Pn_maps[i,0] = self._compute_weighted_gaunt_maps(Uinv_a_lm, self.plLXs[:,:,:,[next_ind]], 0, 0) 
                                    _add('p0', next_Pn_maps, axis=2)
                                    next_P_maps[0] = next_Pn_maps
                                else:
                                    next_Pn_maps = np.zeros((2,n+1,1,self.base.Npix),dtype=np.complex128)
                                    for i,Uinv_a_lm in enumerate(all_Uinv_lms[seed]):
                                        for mu in range(n+1):
                                            next_Pn_maps[i,mu] = self._compute_weighted_gaunt_maps(Uinv_a_lm, self.plLXs[:,:,:,[next_ind]], n, mu) 
                                    _add('p%d'%n, next_Pn_maps, axis=2)
                                    next_P_maps[n] = next_Pn_maps
                            
                            # Compute updated product maps
                            next_PQ_vecs = {n3: [self._compute_PQ_vecs(next_P_maps[n3], next_Q_map, 0, 0, n3),
                                                 self._compute_PQ_vecs(next_P_maps[n3], next_Q_map, 1, 1, n3),
                                                 self._compute_PQ_vecs(next_P_maps[n3], next_Q_map, 0, 1, n3, add_sym=True)] for n3 in np.unique(n3s)}
                            
                            # Add to arrays
                            if len(all_PQ_vecs[seed].keys())==0:
                                all_PQ_vecs[seed] = next_PQ_vecs
                            else:
                                for n3 in np.unique(n3s):
                                    for i in range(3):
                                        if replace:
                                            all_PQ_vecs[seed][n3][i][:,:,-1] = next_PQ_vecs[n3][i][:,:,0]
                                        else:
                                            all_PQ_vecs[seed][n3][i] = np.concatenate([all_PQ_vecs[seed][n3][i],next_PQ_vecs[n3][i]],axis=2)
                            
                        # Iterate over realizations
                        for seed in range(N_fish_optim):
                            this_weight = self.quad_weights_1d[inds]*w_vals
                            
                            # If this is the first iteration with this template, compute all maps of interest
                            if initialize:
                                for prev_ind in inds:
                                    _update_maps(prev_ind, seed, np.asarray(n1n3ns)[:,0], np.asarray(n1n3ns)[:,1])
                            
                            # Compute new P, Q maps for this radial index
                            if update:
                                _update_maps(next_ind, seed, np.asarray(n1n3ns)[:,0], np.asarray(n1n3ns)[:,1])           
                                continue
                            
                            Q4a_ww, Q4b_ww, Q4a_wdel = 0.,0.,0.
                            for n_it in range(len(n1n3ns)):
                                n1, n3, n = n1n3ns[n_it]
                                
                                # Compute F_L * [P_{nmu} Q]_LM maps
                                F_vec_w = np.zeros((3,n1+1,len(inds),self.base.Npix),dtype=np.complex128)
                                
                                # Compute F_L * [P Q]_LM maps
                                F_vec_w[0] = self._convolve_F(all_PQ_vecs[seed][n3][0], this_weight, n1, n3, n, inds=ainds)
                                F_vec_w[1] = self._convolve_F(all_PQ_vecs[seed][n3][1], this_weight, n1, n3, n, inds=ainds)
                                F_vec_w[2] = self._convolve_F(all_PQ_vecs[seed][n3][2], this_weight, n1, n3, n, inds=ainds)
                                
                                # Compute full Q maps
                                if gradient:
                                    Qs = self._compute_ideal_Q4(all_maps[seed]['p%d'%n1], all_maps[seed]['q'], F_vec_w, this_weight, n1=n1, inds=ainds, del_weights=self.quad_weights_1d[inds], apply_weighting=True, with_deriv=True)
                                    Q4a_ww += Qs[0]*weights[n_it]
                                    Q4b_ww += Qs[1]*weights[n_it]
                                    Q4a_wdel += Qs[2]*weights[n_it] 
                                else:
                                    Qs = self._compute_ideal_Q4(all_maps[seed]['p%d'%n1], all_maps[seed]['q'], F_vec_w, this_weight, n1=n1, inds=ainds, apply_weighting=True)
                                    Q4a_ww += Qs[0]*weights[n_it]
                                    Q4b_ww += Qs[1]*weights[n_it]
                            
                            # Compute score
                            tscore += self._assemble_fish(all_Q4[seed][0]-Q4a_ww, all_Q4[seed][1]-Q4b_ww, sym=True).ravel()
                            if full_score:
                                fscore += self._assemble_fish(Q4a_ww, Q4b_ww, sym=True).ravel()
                            if gradient: 
                                del_score += 4*self._assemble_fish(Q4a_wdel, Q4b_ww-all_Q4[seed][1], sym=False).ravel()

                        if gradient:
                            return tscore/scaling/N_fish_optim, del_score/scaling/N_fish_optim
                        elif full_score:
                            return tscore/scaling/N_fish_optim, fscore/scaling/N_fish_optim
                        else:
                            return tscore/scaling/N_fish_optim

                    def _tau_coll_score(beta, s):
                        """Compute the score and its derivative for a collider tau-NL template."""
                        tscore = 0.
                        del_score = 0.
                        fscore = 0.
                        
                        # Define indices as an arrray
                        ainds = np.asarray(inds,dtype=np.int32)
                        
                        def _update_maps(next_ind, seed, beta, s):
                            """Compute weighted maps for new radial index and add to lists."""
                            
                            def _add(field, data, axis=1):
                                """Add new maps to list of maps"""
                                if field not in all_maps[seed].keys():
                                    all_maps[seed][field] = np.asarray(data,order='C')
                                elif replace:
                                    if axis==1:
                                        all_maps[seed][field][:,-1] = np.asarray(data,order='C')[:,0]
                                    elif axis==2:
                                        all_maps[seed][field][:,:,-1] = np.asarray(data,order='C')[:,:,0]
                                    else:
                                        raise Exception()
                                else:
                                    all_maps[seed][field] = np.asarray(np.concatenate([all_maps[seed][field], data], axis=axis),order='C')
                                
                            # Compute weighted maps
                            next_Q_map = np.asarray([self._compute_weighted_maps(np.asarray(Uinv_a_lm,order='C'), self.qlXs[:,:,[next_ind]]) for Uinv_a_lm in all_Uinv_lms[seed]],dtype=np.float64,order='C')
                            _add('q',next_Q_map)
                            
                            next_coll_P_maps = {}
                            for tbeta in np.unique([beta,np.conj(beta)]):
                                
                                # Define plL filters, using conjugation symmetries
                                if np.imag(tbeta)==0:
                                    re_coll_plLX = np.asarray(self.coll_plLXs[tbeta][:,:,:,[next_ind]].real,order='C')
                                elif np.imag(tbeta)<0:
                                    re_coll_plLX = np.asarray(self.coll_plLXs[tbeta][:,:,:,[next_ind]].real,order='C')
                                    im_coll_plLX = np.asarray(self.coll_plLXs[tbeta][:,:,:,[next_ind]].imag,order='C')
                                else:
                                    re_coll_plLX = np.asarray(self.coll_plLXs[np.conj(tbeta)][:,:,:,[next_ind]].real,order='C')
                                    im_coll_plLX = np.asarray(-self.coll_plLXs[np.conj(tbeta)][:,:,:,[next_ind]].imag,order='C')   
                                
                                # Compute P[beta, s, lam] maps   
                                next_Ps_maps = np.zeros((2,s+1,1,self.base.Npix),dtype=np.complex128)
                                for i,Uinv_a_lm in enumerate(all_Uinv_lms[seed]):
                                    for lam in range(s+1):
                                        if np.imag(tbeta)==0:
                                            next_Ps_maps[i,lam] = self._compute_weighted_gaunt_maps(Uinv_a_lm, re_coll_plLX, s, lam)
                                        else:
                                            next_Ps_maps[i,lam] = self._compute_weighted_gaunt_maps(Uinv_a_lm, re_coll_plLX, s, lam, im_coll_plLX)
                                _add('coll-%d,%.8f,%.8fi'%(s,tbeta.real,tbeta.imag),next_Ps_maps, axis=2)
                                next_coll_P_maps[tbeta, s] = next_Ps_maps
                                
                            # Compute updated product maps
                            if np.imag(beta)!=0:
                                next_PQ_vecs = {beta: [0,0,0]}
                                next_conjPQ_vecs = {beta: [0,0,0]}
                                next_PQ_vecs[beta][0], next_conjPQ_vecs[beta][0] = self._compute_coll_PQ_vecs(next_coll_P_maps[beta, s], next_Q_map, 0, 0, s, beta, conjPs_maps=next_coll_P_maps[np.conj(beta), s])
                                next_PQ_vecs[beta][1], next_conjPQ_vecs[beta][1] = self._compute_coll_PQ_vecs(next_coll_P_maps[beta, s], next_Q_map, 1, 1, s, beta, conjPs_maps=next_coll_P_maps[np.conj(beta), s])
                                next_PQ_vecs[beta][2], next_conjPQ_vecs[beta][2] = self._compute_coll_PQ_vecs(next_coll_P_maps[beta, s], next_Q_map, 0, 1, s, beta, add_sym=True, conjPs_maps=next_coll_P_maps[np.conj(beta), s])
                            else:
                                next_PQ_vecs = {beta: [self._compute_coll_PQ_vecs(next_coll_P_maps[beta, s], next_Q_map, 0, 0, s, beta),
                                                       self._compute_coll_PQ_vecs(next_coll_P_maps[beta, s], next_Q_map, 1, 1, s, beta),
                                                       self._compute_coll_PQ_vecs(next_coll_P_maps[beta, s], next_Q_map, 0, 1, s, beta, add_sym=True)]}
                            
                            # Add to arrays
                            if len(all_PQ_vecs[seed].keys())==0:
                                all_PQ_vecs[seed] = next_PQ_vecs
                                if np.imag(beta)!=0:
                                    all_conjPQ_vecs[seed] = next_conjPQ_vecs
                            else:
                                for i in range(3):
                                    if replace:
                                        all_PQ_vecs[seed][beta][i][:,:,-1] = next_PQ_vecs[beta][i][:,:,0]
                                        if np.imag(beta)!=0:
                                            all_conjPQ_vecs[seed][beta][i][:,:,-1] = next_conjPQ_vecs[beta][i][:,:,0]
                                    else:
                                        all_PQ_vecs[seed][beta][i] = np.concatenate([all_PQ_vecs[seed][beta][i],next_PQ_vecs[beta][i]],axis=2)
                                        if np.imag(beta)!=0:
                                            all_conjPQ_vecs[seed][beta][i] = np.concatenate([all_conjPQ_vecs[seed][beta][i],next_conjPQ_vecs[beta][i]],axis=2)
                            
                        # Iterate over realizations
                        for seed in range(N_fish_optim):
                            this_weight = self.quad_weights_1d[inds]*w_vals
                            
                            # If this is the first iteration with this template, compute all maps of interest
                            if initialize:
                                for prev_ind in inds:
                                    _update_maps(prev_ind, seed, beta, s)
                            
                            # Compute new P, Q maps for this radial index
                            if update:
                                _update_maps(next_ind, seed, beta, s)     
                                continue      
                            
                            if np.imag(beta)==0:
                                F_vec_w = np.zeros((3,s+1,len(inds),self.base.Npix),dtype=np.complex128)             
                                # Convolve with F
                                F_vec_w[0] = self._convolve_coll_F(all_PQ_vecs[seed][beta][0], this_weight, s, beta, inds=ainds)
                                F_vec_w[1] = self._convolve_coll_F(all_PQ_vecs[seed][beta][1], this_weight, s, beta, inds=ainds)
                                F_vec_w[2] = self._convolve_coll_F(all_PQ_vecs[seed][beta][2], this_weight, s, beta, inds=ainds)
                            
                            else:
                                F_vec_w = np.zeros((3,2*s+1,len(inds),self.base.Npix),dtype=np.complex128)                       
                                # Convolve with F
                                F_vec_w[0] = self._convolve_coll_F(all_PQ_vecs[seed][beta][0], this_weight, s, beta, inds=ainds, conjPQ_vecs=all_conjPQ_vecs[seed][beta][0])
                                F_vec_w[1] = self._convolve_coll_F(all_PQ_vecs[seed][beta][1], this_weight, s, beta, inds=ainds, conjPQ_vecs=all_conjPQ_vecs[seed][beta][1])
                                F_vec_w[2] = self._convolve_coll_F(all_PQ_vecs[seed][beta][2], this_weight, s, beta, inds=ainds, conjPQ_vecs=all_conjPQ_vecs[seed][beta][2])
                            
                            # Compute full Q maps
                            if np.imag(beta)!=0:
                                if gradient:
                                    Q4a_ww, Q4b_ww, Q4a_wdel = self._compute_ideal_Q4(all_maps[seed]['coll-%d,%.8f,%.8fi'%(s,beta.real,beta.imag)], all_maps[seed]['q'], F_vec_w, this_weight, s, beta, all_maps[seed]['coll-%d,%.8f,%.8fi'%(s,beta.real,-beta.imag)], inds=ainds, del_weights=self.quad_weights_1d[inds], apply_weighting=True, with_deriv=True)
                                else:
                                    Q4a_ww, Q4b_ww = self._compute_ideal_Q4(all_maps[seed]['coll-%d,%.8f,%.8fi'%(s,beta.real,beta.imag)], all_maps[seed]['q'], F_vec_w, this_weight, s, beta, all_maps[seed]['coll-%d,%.8f,%.8fi'%(s,beta.real,-beta.imag)], inds=ainds, apply_weighting=True)
                            else:
                                if gradient:
                                    Q4a_ww, Q4b_ww, Q4a_wdel = self._compute_ideal_Q4(all_maps[seed]['coll-%d,%.8f,%.8fi'%(s,beta.real,beta.imag)], all_maps[seed]['q'], F_vec_w, this_weight, s, beta, inds=ainds, del_weights=self.quad_weights_1d[inds], apply_weighting=True, with_deriv=True)
                                else:
                                    Q4a_ww, Q4b_ww = self._compute_ideal_Q4(all_maps[seed]['coll-%d,%.8f,%.8fi'%(s,beta.real,beta.imag)], all_maps[seed]['q'], F_vec_w, this_weight, s, beta, inds=ainds, apply_weighting=True)

                            # Compute score
                            tscore += self._assemble_fish(all_Q4[seed][0]-Q4a_ww, all_Q4[seed][1]-Q4b_ww, sym=True).ravel()
                            if full_score:
                                fscore += self._assemble_fish(Q4a_ww, Q4b_ww, sym=True).ravel()
                            if gradient: 
                                del_score += 4*self._assemble_fish(Q4a_wdel, Q4b_ww-all_Q4[seed][1], sym=False).ravel()
                        
                        if gradient:
                            return tscore/scaling/N_fish_optim, del_score/scaling/N_fish_optim
                        elif full_score:
                            return tscore/scaling/N_fish_optim, fscore/scaling/N_fish_optim
                        else:
                            return tscore/scaling/N_fish_optim
                    
                    # Main code
                    if template=='tauNL-loc':
                        return _tau_direc_score([[0, 0, 0]], [(4.*np.pi)**1.5])
                    
                    elif 'tauNL-direc' in template:
                        n1,n3,n = np.asarray(template.split(':')[1].split(',')).astype(int)
                        return _tau_direc_score([[n1, n3, n]], [1.])    
                    
                    elif 'tauNL-even' in template:
                        n = int(template.split(':')[1])
                        uniq_n1n3ns, uniq_weights = self._decompose_tauNL_even(n)
                        return _tau_direc_score(uniq_n1n3ns, uniq_weights)
                    
                    elif 'tauNL-odd' in template:
                        n = int(template.split(':')[1])
                        uniq_n1n3ns, uniq_weights = self._decompose_tauNL_odd(n)
                        return _tau_direc_score(uniq_n1n3ns, uniq_weights)
                        
                    elif 'tauNL-light' in template:
                        s = int(template.split(':')[1].split(',')[0])
                        nu_s = float(template.split(':')[1].split(',')[1])
                        return _tau_coll_score(-1.5+nu_s, s)
                    
                    elif 'tauNL-heavy' in template:
                        s = int(template.split(':')[1].split(',')[0])
                        mu_s = float(template.split(':')[1].split(',')[1])
                        return _tau_coll_score(-1.5-1.0j*mu_s, s)
            
            def _test_inds(inds, score_old, w_vals):
                """Test the current set of indices"""
                if optimize_weights and ('tauNL' in template):
                    # Run optimization
                    if score_old==np.inf:
                        scale = init_score
                    else:
                        scale = score_old
                    output = minimize(_compute_score, w_vals, method='BFGS',jac=True, options={"xrtol":0.000001,"maxiter":10}, args=scale)
                    w_vals = output.x
                    if output.fun<1:
                        score = scale*output.fun
                    else:
                        score = scale
                else:
                    score = _compute_score(w_vals, 1, gradient=False, update=False)
                    if (optimize_skip!=0) and (iit%optimize_skip==0) and ('tauNL' in template) and (score/init_score > tolerance) and (iit>0) and (score<score_old):
                        # Run an extra optimization step
                        output = minimize(_compute_score, w_vals, method='BFGS',jac=True, options={"xrtol":0.000001,"maxiter":25}, args=score)
                        w_vals = output.x
                        if output.fun<1:
                            score = score*output.fun
                        if verb: print("Optimization reduced score by %.2f"%output.fun)
                return score, w_vals
            
            # Check zeroth iteration
            if len(inds)!=0:
                # Compute quadratic weights
                notinds = [i for i in np.arange(self.N_r) if i not in inds]
                inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
                G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]
                w_quad = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))
                
                # Add previously optimized weights, if present
                if template in weight_guess.keys():
                    n_guess = np.sum(weight_guess[template]!=0)
                    notinds_guess = [i for i in np.arange(self.N_r) if i not in inds_guess[:n_guess]]
                    inv_deriv_guess = np.linalg.inv(deriv_matrix[inds_guess[:n_guess]][:,inds_guess[:n_guess]])
                    w_quad_guess = (1+np.sum(inv_deriv_guess@deriv_matrix[inds_guess[:n_guess]][:,notinds_guess],axis=1))
                    w_vals = w_quad + np.concatenate([weight_guess[template][:n_guess]/self.quad_weights_1d[inds_guess[:n_guess]]-w_quad_guess,np.zeros(len(inds)-n_guess)])
                else: 
                    w_vals = w_quad
                
                # Update tauNL arrays
                if 'tauNL' in template:
                    _compute_score(w_vals, 1, gradient=False, update=False, replace=False, initialize=True)
                        
                # Compute score
                score, w_vals = _test_inds(inds, init_score, w_vals)
                if verb: print("Unoptimized relative score: %.2e"%(score/init_score))
            else:
                score = init_score
                w_quad = []
                w_vals = []
                
            # Set-up stalled memory
            stalled_score = score.copy()
            
            # Set up iteration
            if score/init_score >= tolerance:
                
                # Define starting indices
                if len(inds)==0:
                    next_inds = np.argsort(np.sum(deriv_matrix,axis=1)**2/np.diag(deriv_matrix))[::-1]
                else:
                    next_inds = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[::-1]]
                next_ind = next_inds[0]
                inds.append(next_ind)
                
                # Set-up memory
                w_vals_old = w_vals
                w_quad_old = w_quad
                score_old = score
                    
                # Iterate until convergence
                for iit,iteration in enumerate(range(len(inds),self.N_r)):
                    
                    # Trial a new index
                    for trial in range(len(next_inds)):
                        
                        # Define indices  
                        next_ind = next_inds[trial]
                        inds[-1] = next_ind
                        notinds = [i for i in np.arange(self.N_r) if i not in inds]
                        
                        # Set-up weights
                        inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
                        G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]
                        
                        # Compute optimal quadratic weights and add to current values            
                        w_quad = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))
                        w_vals = np.append(w_vals_old, 0.)+w_quad-np.append(w_quad_old,0.)
                           
                        # Update tauNL arrays
                        if 'tauNL' in template:
                            _compute_score(w_vals, 1, gradient=False, update=True, replace=(trial!=0))
                            
                        # Compute score
                        score, w_vals = _test_inds(inds, score_old, w_vals)
                        if verb: print("Iteration %d, trial %d, relative score: %.2e, old score: %.2e"%(iteration, trial, score/init_score, score_old/init_score))
                              
                        # Accept new indices
                        if score<score_old or (not repeat_trials):
                            break
                    
                    # Check for numerical errors
                    if score<0:
                        print("## Score is negative; this indicates a numerical error!")
                        break
                    
                    # Finish if converged
                    if score/init_score < tolerance:
                        break
                    
                    # Check if stalled and finish if so
                    if iit%stalled_iterations==0 and iit>0:
                        if np.abs(score-stalled_score)/stalled_score < 0.05: 
                            print("Stalled at iteration %d"%iteration)
                            break
                        else:
                            # Update memory
                            stalled_score = score.copy()
                            
                    # Update memory when score is accepted
                    w_vals_old = w_vals
                    w_quad_old = w_quad
                    score_old = score
                    
                    # Compute indices for next iteration
                    next_inds = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[::-1]]
                    inds.append(next_inds[0])
                    
            if len(G_mat)==0:
                raise Exception("Failed to converge after %d iterations; this indicates a bug!"%N_fish_optim)
                
            if verb: print("\nScore threshold met with %d indices"%len(inds))
            w_opt = np.asarray(w_vals.copy())
            
            # Check final Fisher matrix
            score, fish = _compute_score(w_opt, 1, gradient=False, full_score=True, update=False)
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
        self._prepare_templates(ints_1d=True,ints_2d=False,ints_coll=self.ints_coll)

        print("\nOptimization complete after %.2f seconds"%(time.time()-t_init))
        
        return self.r_arr, self.r_weights
    
    @_timer_func('optimization')
    def optimize_radial_sampling_2d(self, reduce_r=1, reduce_tau=1, tolerance=1e-3, verb=False, N_split=None, split_index=None, initial_rtau_points=None):
        """
        Compute the 2D radial sampling points and weights via optimization (as in Smith & Zaldarriaga 06), up to some tolerance in the Fisher distance.
        Optimization will be done for each template, analytically compute the 'distance' between template approximations. Note that this is required only for the EFTI templates (which involve 2D integrals over r and tau).
        
        Main Inputs:
            - reduce_r: Downsample the number of points in the starting radial integral grid (default: 1)
            - reduce_tau: Downsample the number of points in the starting conformal time integral grid (default: 1)
            - tolerance: Convergence threshold for the optimization (default 1e-3). This indicates the approximate error in the Fisher matrix induced by the optimization.
            
        For large problems, it is too expensive to optimize the whole matrix at once. Instead, we can split the optimization into N_split pieces, each of which is optimized separately.
        Following this, we perform a final optimization of all N_split components, using the union of all previously obtained r/tau points. 
        
        Additional Inputs (for chunked computations):
            - N_split (optional): Number of chunks to split the optimization into. If None, no splitting is performed.
            - split_index (optional): Index of the chunk to optimize. 
            - initial_rtau_points (optional): Starting set of r/tau points (used for the final optimization step).
        """

        assert self.ints_2d, "Only 1D sampling is required for these templates!"
        t_init = time.time()
        
        # Check precision parameters
        assert reduce_r>0, "reduce_r parameter must be positive"
        assert reduce_tau>0, "reduce_tau parameter must be positive"
        
        if reduce_r<0.5:
            print("## Caution: very dense r-sampling requested; computation may be very slow") 
        if reduce_r>3:
            print("## Caution: very sparse r-sampling requested; computation may be inaccurate")
        if reduce_tau<0.5:
            print("## Caution: very dense tau-sampling requested; computation may be very slow") 
        if reduce_tau>3:
            print("## Caution: very sparse tau-sampling requested; computation may be inaccurate")

        # Create r/tau array
        tau_init = -np.exp(np.arange(np.log(10./self.lmax),np.log(1e6),np.log(10.)/5.*reduce_tau))[::-1]
        dtau = np.diff(np.log(-tau_init))[0]*tau_init
        rtau_arr = []
        rtau_w_arr = []
        for t,tau in enumerate(tau_init):
            delta_r1 = max(50.,-0.1*tau)*reduce_r
            delta_r2 = max(5.,-0.1*tau)*reduce_r
            r_raw = np.asarray(list(np.arange(1,self.r_star*0.95,delta_r1))+list(np.arange(self.r_star*0.95,self.r_hor*1.05,delta_r2))+list(np.arange(self.r_hor*1.05,self.r_hor+5000-5*tau,delta_r1)))
            this_r = 0.5*(r_raw[1:]+r_raw[:-1])
            rtau_arr.append([this_r,tau*np.ones(len(this_r))])
            rtau_w_arr.append(this_r**2*np.diff(r_raw)*dtau[t])
        rtau_init = np.concatenate(rtau_arr,axis=1).T
        self.quad_weights_2d = np.concatenate(rtau_w_arr)
        
        # Partition the radial/tau indices if required or read in precomputed points
        if initial_rtau_points is not None:
            assert split_index is None, "Cannot specify both initial_rtau_points and index_split"
            inds = np.asarray([np.argmin(np.sum((rtau-rtau_init)**2,1)) for rtau in initial_rtau_points])
            rtau_init = rtau_init[inds]
            self.quad_weights_2d = self.quad_weights_2d[inds]
        else:
            if N_split is not None:
                print("Partitioning sampling grid into %d pieces"%(N_split))                
                rtau_init = rtau_init[split_index::N_split]
                self.quad_weights_2d = self.quad_weights_2d[split_index::N_split]

        rtau_weights = {}
        
        # Precompute k-integrals with initial r grid
        self.rtau_arr = rtau_init
        self.N_rtau = len(rtau_init)
        print("Computing k integrals with fiducial r/tau grid")
        self._prepare_templates(ints_1d=False,ints_2d=True,ints_coll=False)
        
        # Reorder templates if necessary (no exchange trispectra here, so less of a problem!)
        ordered_templates = []
        for template in self.templates:
            if template in self.all_templates_2d: ordered_templates.append(template)
            if (template in self.all_templates_2d) and ('tauNL' in template): raise Exception("2D exchange templates not implemented!")
            
        # Create list of r/tau indices in the optimized representation
        inds = []
        inds_init = np.arange(self.N_rtau)
            
        # Compute Fisher matrix derivative
        if verb: print("Computing all Fisher matrix derivatives")
        derivs = self._compute_fisher_derivatives(ordered_templates, verb=verb)
        
        # Save ideal Fisher matrices
        if not hasattr(self, 'ideal_fisher'):
            self.ideal_fisher = {}
        for t in ordered_templates:
            self.ideal_fisher[t] = derivs[t][0]

        for template in ordered_templates:
            
            # Load initial score and normalize to avoid conditioning problems
            init_score, deriv_matrix = derivs[template][0], derivs[template][1]
            deriv_matrix /= init_score
            
            # Initialize optimization (from scratch, if this is the first template)
            print("Init score: %.2e"%init_score)
            if len(inds)==0:
                next_ind = np.argmax(np.sum(deriv_matrix,axis=1)**2/np.diag(deriv_matrix))
                inds = [next_ind]

            # Iterate over radial points using a greedy algorithm
            for iteration in range(len(inds)-1,self.N_rtau):

                notinds = [i for i in np.arange(self.N_rtau) if i not in inds]
                inv_deriv = pinv(deriv_matrix[inds][:,inds],rtol=1e-20)
                G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]

                def _compute_score(w_vals, scaling=1, full_score=False):
                    """Compute the Fisher distance between templates given weights w_vals. This optionally computes the gradients."""
                    
                    if full_score:
                        return scaling*np.sum(G_mat), scaling*np.sum(np.outer(w_vals,w_vals)*deriv_matrix[inds][:,inds])
                    else:
                        return scaling*np.sum(G_mat)

                # Compute quadratic weights
                w_vals = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))

                # Compute score with fiducial weights
                score = _compute_score(w_vals, init_score)
                if verb: print("Iteration %d, score: %.2e"%(iteration, score))

                # Check for numerical errors
                if score<0:
                    print("## Score is negative; this indicates a numerical error!")
                    
                    # Define optimal weights and exit
                    w_opt = w_vals.copy()
                    break
                
                if score/init_score<tolerance:
                    print("Terminating at step %d"%iteration)

                    # Compute optimal weights
                    w_opt = w_vals.copy()
                    break

                # Update indices
                next_ind = inds_init[notinds][np.argmax(np.sum(G_mat,axis=1)**2/np.diag(G_mat))]
                inds.append(next_ind)

            # Check final Fisher matrix
            score, fish = _compute_score(w_opt, init_score, full_score=True)
            print("Ideal Fisher: %.4e (initial), %.4e (optimized). Fractional score: %.2e\n"%(init_score, fish, score/fish))

            # Store attributes
            rtau_weights[template] = w_opt*self.quad_weights_2d[inds]
            
        # Store attributes
        self.rtau_arr = rtau_init[inds]
        self.N_rtau = len(self.rtau_arr)
        self.rtau_weights = {}
        for template in ordered_templates:
            # add weights, padding with zeros
            self.rtau_weights[template] = np.zeros(len(w_opt)) 
            self.rtau_weights[template][:len(rtau_weights[template])] = rtau_weights[template]

        # Precompute k-space integrals with new radial integration
        print("Computing k integrals with optimized radial grid")
        self._prepare_templates(ints_1d=False,ints_2d=True,ints_coll=False)

        print("\nOptimization complete after %.2f seconds"%(time.time()-t_init))

        return self.rtau_arr, self.rtau_weights
    
    def _compute_fisher_derivatives(self, templates, N_fish_optim=None, verb=False):
        """Compute the derivative of the ideal Fisher matrix with respect to the weights for each template of interest."""

        # Output array
        output = {}
        
        # Separate out tauNL templates
        contact_templates = []
        exchange_templates = []
        for template in templates:
            if 'gNL' in template: 
                contact_templates.append(template)
            elif 'tauNL' in template:
                exchange_templates.append(template)
            else:
                raise Exception("Template %s not implemented!"%template)
                
        # Start with contact templates
        if len(contact_templates)!=0:
            # Compute arrays
            # NB: using exact Gauss-Legendre integration in mu
            [mus, w_mus] = p_roots(2*self.lmax+1)
            ls = np.arange(self.lmin,self.lmax+1)
            legs = np.asarray([lpmn(0,self.lmax,mus[i])[0][0,self.lmin:] for i in range(len(mus))])
                
            t_init = time.time()
            for template in contact_templates:
                    
                if template=='gNL-loc':
                    if verb: print("\tComputing gNL-loc Fisher matrix derivative exactly")
                    deriv_matrix = np.asarray(fisher_deriv_gNL_loc(self.plLXs[self.nmax], self.qlXs, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                        legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                    
                elif template=='gNL-con':

                    if verb: print("\tComputing gNL-con Fisher matrix derivative exactly")
                    deriv_matrix = np.asarray(fisher_deriv_gNL_con(self.rlXs, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'),
                                        legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                
                elif template=='gNL-dotdot':

                    if verb: print("\tComputing gNL-dotdot Fisher matrix derivative exactly")
                    deriv_matrix = np.asarray(fisher_deriv_gNL_dotdot(self.alXs, self.rtau_arr[:,1], self.quad_weights_2d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                            legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                
                elif template=='gNL-dotdel':
                    
                    if verb: print("\tComputing gNL-dotdel Fisher matrix derivative exactly")
                    deriv_matrix = np.asarray(fisher_deriv_gNL_dotdel(self.alXs, self.blXs, self.clXs, self.rtau_arr[:,1], self.quad_weights_2d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                            legs, mus, w_mus, self.lmin, self.lmax, self.base.nthreads))
                    
                elif template=='gNL-deldel':
                    
                    if verb: print("\tComputing gNL-deldel Fisher matrix derivative exactly")
                    deriv_matrix = np.asarray(fisher_deriv_gNL_deldel(self.blXs, self.clXs, self.quad_weights_2d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                            legs, mus, w_mus, self.lmin, self.lmax, self.base.nthreads))
                   
                else:
                    raise Exception("Template %s not implemented!"%template)
                    
                output[template] = np.sum(deriv_matrix), deriv_matrix

            self.timers['analytic_fisher'] += time.time()-t_init

        # Move to tauNL estimators
        if len(exchange_templates)!=0:
            
            # Precompute common terms
            all_Uinv_lms = []
            Q_maps = []
            
            all_Q4 = {t: [] for t in exchange_templates}
            init_score = {t: 0 for t in exchange_templates}
            deriv_matrix = {t: 0 for t in exchange_templates}
            
            # Define weight indices
            inds = np.arange(self.N_r,dtype=np.int32)
            i_weights = np.ones(self.N_r)*self.quad_weights_1d[inds]
            d_weights = self.quad_weights_1d[inds]
        
            for seed in range(N_fish_optim):
                
                print("\nComputing Fisher matrix derivatives from realization %d of %d"%(seed+1, N_fish_optim))
                
                # Compute A^-1.a GRF maps 
                Uinv_a_lms = []
                for ii in range(2):
                    t_init = time.time()
                    a_lm = self.base.generate_data(seed=seed+int((1+ii)*1e9), output_type='harmonic', deconvolve_beam=True)
                    self.timers['fish_grfs'] += time.time()-t_init
                    t_init = time.time()
                    Uinv_a_lms.append(np.asarray(self.base.applyAinv(a_lm, input_type='harmonic')[:,self.lfilt],order='C'))
                    self.timers['Ainv'] += time.time()-t_init
                all_Uinv_lms.append(Uinv_a_lms)
                
                # Compute Q maps
                if verb: print("Creating Q maps")
                Q_maps = np.asarray(self._filter_pair(Uinv_a_lms, 'Q'),dtype=np.float64,order='C')    

                # Compute P maps
                if verb and len(self.p_inds)>0: print("Creating P_{n,mu} maps")
                P_maps = self._filter_pair(Uinv_a_lms, 'P')
                
                # Compute collider P maps     
                if verb and len(self.coll_params.keys())>0: print("Creating collider P^beta_{s,lam} maps")
                coll_P_maps = self._filter_pair(Uinv_a_lms, 'coll-P')
                
                def _compute_direc_deriv_single(n1n3ns, weights):
                    """Compute the idealized directional Fisher matrix derivative for a given random seed."""
                    
                    # Compute F_L * [P_{nmu} Q]_LM maps
                    Q4a_11, Q4b_11, Q4a_1del, Q4b_1del = 0.,0.,0.,0.
                    for n_it in range(len(n1n3ns)):
                        n1, n3, n = n1n3ns[n_it]
                        
                        F_vecs_all = self._F_PQ(P_maps[n3], Q_maps, i_weights, n1, n3, n, inds=inds)
                        
                        # Compute Q4 maps
                        Qs = self._compute_ideal_Q4(P_maps[n1], Q_maps, F_vecs_all, i_weights, n1=n1, del_weights=d_weights, apply_weighting=True, with_deriv=True, with_deriv_weight=True)
                        Q4a_11 += weights[n_it]*Qs[0]
                        Q4b_11 += weights[n_it]*Qs[1]
                        Q4a_1del += weights[n_it]*Qs[2]
                        Q4b_1del += weights[n_it]*Qs[3] 
                        
                    this_Q4 = [Q4a_11, Q4b_11]
                    
                    # Assemble score and derivative
                    this_score = self._assemble_fish(Q4a_11, Q4b_11, sym=True).ravel()
                    this_matrix = self._assemble_fish_ideal(Q4a_1del, Q4b_1del, sym=True)
                    return this_Q4, this_score, this_matrix

                def _compute_coll_deriv_single(beta, s):
                    """Compute the idealized collider Fisher matrix derivative for a term specified by (complex) beta and spin s for a given random seed."""
            
                    # Compute Sum_S C_s(S, i nu_s)F^{-2beta}_L * [P^beta_{sl} Q]_LM maps
                    Q4a_11, Q4b_11, Q4a_1del, Q4b_1del = 0.,0.,0.,0.
                    
                    if np.imag(beta)!=0:
                        # Compute F_L^beta * [P^beta_{slam} Q]_LM maps
                        F_vecs_all = self._coll_F_PQ(coll_P_maps[beta,s], Q_maps, i_weights, s, beta, coll_P_maps[np.conj(beta),s], inds=inds) 
                        
                        # Compute Q4 maps
                        Q4a_11, Q4b_11, Q4a_1del, Q4b_1del = self._compute_ideal_Q4(coll_P_maps[beta,s], Q_maps, F_vecs_all, i_weights, s, beta, coll_P_maps[np.conj(beta),s], del_weights=d_weights, apply_weighting=True, with_deriv=True, with_deriv_weight=True)
                        
                    else:
                        # Compute F_L^beta * [P^beta_{slam} Q]_LM maps
                        F_vecs_all = self._coll_F_PQ(coll_P_maps[beta,s], Q_maps, i_weights, s, beta, inds=inds) 
                        
                        # Compute Q4 maps
                        Q4a_11, Q4b_11, Q4a_1del, Q4b_1del = self._compute_ideal_Q4(coll_P_maps[beta,s], Q_maps, F_vecs_all, i_weights, s, beta, del_weights=d_weights, apply_weighting=True, with_deriv=True, with_deriv_weight=True)
                        
                    this_Q4 = [Q4a_11, Q4b_11]
                    
                    # Assemble score and derivative
                    this_score = self._assemble_fish(Q4a_11, Q4b_11, sym=True).ravel()
                    this_matrix = self._assemble_fish_ideal(Q4a_1del, Q4b_1del, sym=True)
                    return this_Q4, this_score, this_matrix
            
                # Now compute estimator
                for template in exchange_templates:
                
                    if template=='tauNL-loc':
                        # Local tauNL
                        if verb: print("Computing tauNL-loc Fisher matrix derivative")

                        # Compute derivative
                        outs = _compute_direc_deriv_single([[0, 0, 0]], [(4.*np.pi)**1.5]) # add (4pi)^3
                        
                    elif 'tauNL-direc' in template:
                        # Direction-dependent tauNL
                        n1,n3,n = np.asarray(template.split(':')[1].split(','),dtype=int)
                        if verb: print("Computing tauNL-direc:(%d,%d,%d) Fisher matrix derivative"%(n1,n3,n))
               
                        # Compute derivative
                        outs = _compute_direc_deriv_single([[n1, n3, n]], [1.])
                        
                    elif 'tauNL-even' in template:
                        # Direction-dependent even tauNL
                        n = int(template.split(':')[1])
                        if verb: print("Computing tauNL-even:(%d) Fisher matrix derivative"%n)
                        
                        # Compute the decomposition into n1, n3, n pieces
                        uniq_n1n3ns, uniq_weights = self._decompose_tauNL_even(n)
                        
                        # Compute derivative
                        outs = _compute_direc_deriv_single(uniq_n1n3ns, uniq_weights)
                        
                    elif 'tauNL-odd' in template:
                        # Direction-dependent odd tauNL
                        n = int(template.split(':')[1])
                        if verb: print("Computing tauNL-odd:(%d) Fisher matrix derivative"%n)

                        # Compute the decomposition into n1, n3, n pieces                 
                        uniq_n1n3ns, uniq_weights = self._decompose_tauNL_odd(n)
                        
                        # Compute derivative
                        outs = _compute_direc_deriv_single(uniq_n1n3ns, uniq_weights)                    
                    
                    elif 'tauNL-light' in template:
                        # Light particle collider tauNL
                        s = int(template.split(':')[1].split(',')[0])
                        nu_s = float(template.split(':')[1].split(',')[1])
                        if verb: print("Computing tauNL-light:(%d,%.2f) Fisher matrix derivative"%(s,nu_s))
                        
                        # Compute derivative
                        outs = _compute_coll_deriv_single(-1.5+nu_s, s)
                        
                    elif 'tauNL-heavy' in template:
                        # Heavy particle collider tauNL
                        s = int(template.split(':')[1].split(',')[0])
                        mu_s = float(template.split(':')[1].split(',')[1])
                        if verb: print("Computing tauNL-heavy(%d,%.2f) Fisher matrix derivative"%(s,mu_s))
                        
                        # Compute derivative
                        outs = _compute_coll_deriv_single(-1.5-1.0j*mu_s, s)
                    
                    else:
                        raise Exception("Template %s not implemented!"%template)
                        
                    all_Q4[template].append(outs[0])
                    init_score[template] += outs[1]/N_fish_optim
                    deriv_matrix[template] += outs[2]/N_fish_optim
                    
            # Add to output dictionary
            for template in exchange_templates:
                output[template] = init_score[template], deriv_matrix[template], all_Uinv_lms, all_Q4[template]
            
        return output

    def _compute_ideal_Q4(self, Pn1_maps, Q_maps, F_vecs, r_weights, n1=0, beta_coll=np.inf, conjPn1_maps=[], del_weights=[], apply_weighting=False, with_deriv=False, with_deriv_weight=False, inds=[]):
        """
        Assemble and return an array of ideal Q4 maps and derivatives. This is specific to the optimization routines and applies only to exchange trispectra.


        The outputs are either Q(b) or S^-1Q(b), or the derivatives.
        """
        if len(inds)==0:
            inds = np.arange(self.N_r, dtype=np.int32)
        
        ### Assemble Q4 maps, with relevant symmetries
        if with_deriv:
            Q4, dQ4 = self._get_Q4_perms(Pn1_maps, Q_maps, F_vecs, r_weights, n1, beta_coll, conjPn1_maps, with_deriv=True, inds=inds, del_weights=del_weights)
        else:
            Q4 = self._get_Q4_perms(Pn1_maps, Q_maps, F_vecs, r_weights, n1, beta_coll, conjPn1_maps, with_deriv=False, inds=inds)
        Q4 = Q4[:,None,...]
        
        # Assemble output
        output = [Q4.reshape(4,1,-1)]
        
        if apply_weighting: 
            output.append(apply_ideal_weight(Q4, self.Cinv, self.m_weight, self.base.nthreads))
        
        if with_deriv:
            output.append(dQ4.reshape(4,len(inds),-1))

            if with_deriv_weight:
                output.append(apply_ideal_weight(dQ4, self.Cinv, self.m_weight, self.base.nthreads))
        
        return output
