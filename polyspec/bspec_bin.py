### Code for ideal and unwindowed binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2025)
## This module contains the bispectrum estimation code

import numpy as np
import multiprocessing as mp, os, tqdm, time
from scipy.interpolate import InterpolatedUnivariateSpline
import pywigxjpf as wig
import sys, uuid

def globalize(func):
    """Decorator to make a multiprocessing function globally accessible."""
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result

class BSpecBin():
    """Binned bispectrum estimation class. This takes the binning strategy as input and a base class. 
    We also feed in a function that applies the S^-1 operator (which is ideally beam.mask.C_l^{tot,-1}, where C_l^tot includes the beam and noise). 
    
    Note that we can additionally compute squeezed triangles by setting l_bins_squeeze, giving a different lmax for short and long sides.
    
    Inputs:
    - base: PolySpec class
    - mask: HEALPix mask to deconvolve
    - applySinv: function which returns S^-1 applied to a given input map
    - l_bins: array of bin edges
    - l_bins_squeeze: array of squeezed bin edges (optional)
    - include_partial_triangles: whether to include triangles whose centers don't satisfy the triangle conditions. (Default: False)
    - fields: which T/E/B bispectra to compute
    - parity: whether to include contributions from parity 'even' , 'odd' physics or 'both'.
    """
    def __init__(self, base, mask, applySinv, l_bins, l_bins_squeeze=[], include_partial_triangles=False, fields=['TTT','TTE','TEE','EEE','TBB','EBB','TTB','TEB','EEB','BBB'], parity='both'):
        # Read in attributes
        self.base = base
        self.mask = mask
        self.applySinv = applySinv
        self.l_bins = l_bins
        self.pol = self.base.pol
        self.parity = parity
        self.fields = fields
        self.min_l = np.min(l_bins)
        self.Nl = len(l_bins)-1
        
        if len(l_bins_squeeze)>0:
            self.l_bins_squeeze = l_bins_squeeze
            self.Nl_squeeze = len(l_bins_squeeze)-1
            for l_edge in self.l_bins:
                assert l_edge in self.l_bins_squeeze, "Squeezed bins must contain all the unsqueezed bins!"
        else:
            self.l_bins_squeeze = self.l_bins.copy()
            self.Nl_squeeze = self.Nl
        
        self.include_partial_triangles = include_partial_triangles
        self.beam = self.base.beam
        self.beam_lm = self.base.beam_lm
        
        # Check correct fields are being used
        for f in fields:
            assert f in ['TTT','TTE','TEE','EEE','TBB','EBB','TTB','TEB','EEB','BBB'], "Unknown field '%s' supplied!"%f 
        assert len(fields)==len(np.unique(fields)), "Duplicate fields supplied!"

        if not self.pol and fields!=['TTT']:
            print("## Polarization mode not turned on; setting fields to TT only!")
            self.fields = ['TTT']
        
        # Check correct parities being used
        assert parity in ['even','odd','both'], "Parity must be one of 'even', 'odd' or 'both'!"
        if not self.pol and parity!='even':
            print("Caution: scalar 3-point functions can't probe parity-odd primordial physics!")
        if parity=='even':
            self.chi_arr = [1]
        elif parity=='odd':
            self.chi_arr = [-1]
        else:
            self.chi_arr = [1, -1]

        if np.max(self.l_bins_squeeze)>base.lmax:
            raise Exception("Maximum l is larger than HEALPix resolution!")
        if np.max(self.l_bins)>base.lmax//2:
            print("## Caution: Maximum l is greater than HEALPix-lmax/2; this might cause boundary effects.")
        print("Binning: %d bins in [%d, %d]"%(self.Nl,self.min_l,np.max(self.l_bins)))
        if self.Nl_squeeze!=self.Nl:
            print("Squeezed binning: %d bins in [%d, %d]"%(self.Nl_squeeze,self.min_l,np.max(self.l_bins_squeeze)))
        
        # Define l filters
        self.ell_bins = [(self.base.l_arr>=self.l_bins_squeeze[bin1])&(self.base.l_arr<self.l_bins_squeeze[bin1+1]) for bin1 in range(self.Nl_squeeze)]

        # Check if window is uniform
        if np.std(self.mask)<1e-12 and np.abs(np.mean(self.mask)-1)<1e-12:
            print("Mask: ones")
            self.ones_mask = True
        else:
            print("Mask: spatially varying")
            self.ones_mask = False
            
        # Create a list of all bins
        self.all_bins = []
        self.good_bins = []
        for u in self.fields:
            u1, u2, u3 = [self.base.indices[u[i]] for i in range(3)]
            p_u = np.product([self.base.parities[u[i]] for i in range(3)])
            
            for bin1 in range(self.Nl_squeeze):
                for bin2 in range(self.Nl_squeeze):
                    if u1==u2 and bin2<bin1: continue
                    for bin3 in range(self.Nl_squeeze):
                        if u2==u3 and bin3<bin2: continue

                        # skip bins outside the triangle conditions
                        if not self._check_bin(bin1,bin2,bin3): continue
                        
                        # Remove any unused squeezed triangles
                        if min([bin1,bin2,bin3])>=self.Nl: continue
                        self.all_bins.append([u1,u2,u3,bin1,bin2,bin3,p_u])
                        self.good_bins.append(np.asarray([bin1,bin2,bin3]))
        self.all_bins = np.asarray(self.all_bins)
        self.is_good = lambda bins: len(np.where((np.asarray(self.good_bins)[:,0]==bins[0])&(np.asarray(self.good_bins)[:,1]==bins[1])&(np.asarray(self.good_bins)[:,2]==bins[2]))[0])>0
        
        # Define 3j calculation
        wig.wig_table_init(self.l_bins_squeeze[-1]*2,9)
        wig.wig_temp_init(self.l_bins_squeeze[-1]*2)
        self.tj_sym = lambda l1,l2,l3: (wig.wig3jj(2*l1,2*l2,2*l3,-2,-2,4)+wig.wig3jj(2*l1,2*l2,2*l3,4,-2,-2)+wig.wig3jj(2*l1,2*l2,2*l3,-2,4,-2))/3.
               
    def _check_bin(self, bin1, bin2, bin3):
        """Return one if modes in the bin satisfy the triangle conditions, or zero else.

        This is used either for all triangles in the bin, or just the center of the bin.
        """
        if self.include_partial_triangles:
            l1s = np.arange(self.l_bins_squeeze[bin1],self.l_bins_squeeze[bin1+1])[:,None,None]
            l2s = np.arange(self.l_bins_squeeze[bin2],self.l_bins_squeeze[bin2+1])[None,:,None]
            l3s = np.arange(self.l_bins_squeeze[bin3],self.l_bins_squeeze[bin3+1])[None,None,:]
            filt = (l3s>=abs(l1s-l2s))*(l3s<=l1s+l2s)
            if np.sum(filt)>0: 
                return 1
            else: 
                return 0
        else:
            l1 = 0.5*(self.l_bins_squeeze[bin1]+self.l_bins_squeeze[bin1+1])
            l2 = 0.5*(self.l_bins_squeeze[bin2]+self.l_bins_squeeze[bin2+1])
            l3 = 0.5*(self.l_bins_squeeze[bin3]+self.l_bins_squeeze[bin3+1])
            if l3<abs(l1-l2) or l3>l1+l2:
                return 0
            else:
                return 1
         
    def _compute_symmetry_factor(self):
        """
        Compute symmetry factor giving the degeneracy of each bin. This is performed for each choice of parity and field and stacked.
        """
        
        # Iterate over parities
        self.sym_factor_all = []
        for chi in self.chi_arr: 
            sym_factor = []       
            for i in range(len(self.all_bins)):
                u1,u2,u3,bin1,bin2,bin3,p_u = self.all_bins[i]

                # compute symmetry factor
                if bin1==bin2 and u1==u2 and bin2==bin3 and u2==u3:
                    sym = 6
                elif bin1==bin2 and u1==u2:
                    sym = 2
                elif bin2==bin3 and u2==u3:
                    sym = 2
                else:
                    sym = 1
                sym_factor.append(sym)
            self.sym_factor_all.append(np.asarray(sym_factor))
        
        self.sym_factor = np.concatenate(self.sym_factor_all)
        self.N_b = len(self.sym_factor)
        print("Using a total of %d bins"%self.N_b)
        
    def _compute_H_pm(self, a_lm, spin):
        """
        Compute (+-s)H maps for an input a_lm field. This is defined as Sum_lm (+-s)Y_lm(n) a_lm.
        """
        assert spin>0
        H_plus = [self.base.compute_spin_transform_map(a_lm*self.ell_bins[bin1], spin) for bin1 in range(self.Nl_squeeze)]
        H_minus = [np.asarray([(-1)**spin*H.conj() for H in this_H_plus]) for this_H_plus in H_plus]
        return H_plus, H_minus
    
    def get_ells(self, field='TTT'):
        """
        Return an array with the central l1, l2, l3 values for each bispectrum bin, given a set of fields.
        """
        assert field in self.fields, "Specified field '%s' not in inputs!"%field
        u1, u2, u3 = [self.base.indices[field[i]] for i in range(3)]

        l1s, l2s, l3s = [],[],[]
        for bin1 in range(self.Nl_squeeze):
            l1 = 0.5*(self.l_bins_squeeze[bin1]+self.l_bins_squeeze[bin1+1])
            for bin2 in range(self.Nl_squeeze):
                if u1==u2 and bin2<bin1: continue
                l2 = 0.5*(self.l_bins_squeeze[bin2]+self.l_bins_squeeze[bin2+1])
                for bin3 in range(self.Nl_squeeze):
                    if u2==u3 and bin3<bin2: continue
                    l3 = 0.5*(self.l_bins_squeeze[bin3]+self.l_bins_squeeze[bin3+1])
            
                    # skip bins outside the triangle conditions
                    if not self.is_good([bin1,bin2,bin3]): continue
                    
                    # Remove any unused squeezed triangles
                    if min([bin1,bin2,bin3])>=self.Nl: continue
                    
                    # Add to output array
                    l1s.append(l1)
                    l2s.append(l2)
                    l3s.append(l3)
        
        return l1s,l2s,l3s
    
    def _process_sim(self, sim, input_type='map'):
        """
        Process a single input simulation. This is used for the linear term of the bispectrum estimator.
        
        We return (+1)H and (-2)H maps for this simulation.
        """
        # Transform to Fourier space and normalize appropriately
        h_alpha_lm = self.applySinv(sim, input_type=input_type)
        
        # Compute (+1)H and (-2)H maps
        p1_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*h_alpha_lm.conj(),+1) for bin1 in range(self.Nl_squeeze)]
        m2_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*h_alpha_lm.conj(),-2) for bin1 in range(self.Nl_squeeze)]
        return p1_H_maps, m2_H_maps

    def load_sims(self, load_sim, N_sims, verb=False, input_type='map', preload=True):
        """
        Load in and preprocess N_sim Monte Carlo simulations used in the linear term of the bispectrum estimator. 
        These can alternatively be generated with a fiducial spectrum using the generate_sims script.

        The input is a function which loads the simulations in map- or harmonic-space given an index (0 to N_sims-1).

        If preload=False, the simulation products will not be stored in memory, but instead accessed when necessary. This greatly reduces memory usage, but is less CPU efficient if many datasets are analyzed together.
        """
        
        self.N_it = N_sims
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        if preload:
            self.preload = True

            # Define lists of maps
            self.p1_H_maps, self.m2_H_maps = [], []
        
            # Iterate over simulations and preprocess appropriately
            for ii in range(self.N_it):
                if verb: print("Loading bias simulation %d of %d"%(ii+1,self.N_it))    
                this_sim = load_sim(ii)

                # Process simulation
                p1_H_maps, m2_H_maps = self._process_sim(this_sim, input_type=input_type)

                self.p1_H_maps.append(p1_H_maps)
                self.m2_H_maps.append(m2_H_maps)

        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            self.load_sim_data = lambda ii: self._process_sim(load_sim(ii), input_type=input_type)
           
    def generate_sims(self, N_sim, Cl_input=[], b_input=None, add_B=False, remove_mean=True, verb=False, preload=True):
        """
        Generate N_sim Monte Carlo simulations used in the linear term of the bispectrum generator. 
        These are pure GRFs, optionally with a bispectrum added. By default, they are generated with the input survey mask.
        
        If preload=True, we create N_it such simulations and store the relevant transformations into memory.
        If preload=False, we store only the function used to generate the sims, which will be processed later. This is cheaper on memory, but less CPU efficient if many datasets are analyzed together.
        
        We can alternatively load custom simulations using the load_sims script.
        """
        
        self.N_it = N_sim
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        # Define input power spectrum (with noise)
        if len(Cl_input)==0:
            Cl_input = self.base.Cl_tot

        if preload:
            self.preload = True

            # Define lists of maps
            self.p1_H_maps, self.m2_H_maps = [],[]

            # Iterate over simulations and preprocess appropriately
            for ii in range(self.N_it):
                if verb: print("Generating bias simulation %d of %d"%(ii+1,self.N_it))
                
                # Generate beam-convolved simulation and Fourier transform
                if self.ones_mask:
                    alpha_lm = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False, b_input=b_input, add_B=add_B, remove_mean=remove_mean, output_type='harmonic')
                    p1_H_maps, m2_H_maps = self._process_sim(alpha_lm, input_type='harmonic')
                else:
                    alpha = self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False, b_input=b_input, add_B=add_B, remove_mean=remove_mean)
                    p1_H_maps, m2_H_maps = self._process_sim(alpha)
                
                self.p1_H_maps.append(p1_H_maps)
                self.m2_H_maps.append(m2_H_maps)
                
        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            if self.ones_mask:
                self.load_sim_data = lambda ii: self._process_sim(self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False, b_input=b_input, add_B=add_B, remove_mean=remove_mean, output_type='harmonic'), input_type='harmonic')
            else:
                self.load_sim_data = lambda ii: self._process_sim(self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False, b_input=b_input, add_B=add_B, remove_mean=remove_mean))

    ### OPTIMAL ESTIMATOR
    def Bl_numerator(self, data, include_linear_term=True, verb=False):
        """
        Compute the numerator of the unwindowed bispectrum estimator for all fields of interest. We can optionally drop the linear term (usually for testing).
        """
        
        # Compute symmetry factor if necessary
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
            
        if (not hasattr(self, 'preload')) and include_linear_term:
            raise Exception("Need to generate or specify bias simulations!")
        
        # Apply S^-1 to data and transform to harmonic space
        h_data_lm = self.applySinv(data, input_type='map')
        
        # Compute (+1)H and (-2)H maps
        if verb: print("Computing H maps")
        
        global p1_H_maps, m2_H_maps
        p1_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*h_data_lm.conj(),+1) for bin1 in range(self.Nl_squeeze)]
        m2_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*h_data_lm.conj(),-2) for bin1 in range(self.Nl_squeeze)]
        
        @globalize
        def _analyze3(full_index):
            """Compute the cubic piece of the bispectrum numerator"""
            index = full_index%len(self.all_bins)
            chi_index = full_index//len(self.all_bins)
            
            u1,u2,u3,bin1,bin2,bin3,p_u = self.all_bins[index]
            
            # Compute combination of fields
            tmp_sum  = p1_H_maps[bin1][u1]*p1_H_maps[bin2][u2]*m2_H_maps[bin3][u3]
            tmp_sum += p1_H_maps[bin2][u2]*p1_H_maps[bin3][u3]*m2_H_maps[bin1][u1]
            tmp_sum += p1_H_maps[bin3][u3]*p1_H_maps[bin1][u1]*m2_H_maps[bin2][u2]
                
            # Perform map-level summation and take real/im part
            chi = self.chi_arr[chi_index]
            if p_u*chi==-1:
                tmp_sum2 = tmp_sum*1.0j
            else:
                tmp_sum2 = tmp_sum
            return self.base.A_pix*np.real(np.sum(tmp_sum2))/3./self.sym_factor[index]
        
        with mp.Pool(self.base.nthreads) as p:
            if verb:
                b3_num = np.asarray(list(tqdm.tqdm(p.imap(_analyze3,range(self.N_b)),total=self.N_b)))
            else:
                b3_num = np.asarray(list(p.imap(_analyze3,range(self.N_b))))
        
        # Compute b_1 part of cubic estimator, averaging over simulations
        b1_num = np.zeros(self.N_b)
        
        if not include_linear_term:
            print("No linear correction applied!")
        else:
            for ii in range(self.N_it):
                if (ii+1)%5==0 and verb: print("Computing b_1 piece from simulation %d"%(ii+1))

                # Load processed bias simulations 
                global this_p1_H_maps, this_m2_H_maps
                if self.preload:
                    this_p1_H_maps, this_m2_H_maps = self.p1_H_maps[ii], self.m2_H_maps[ii]
                else:
                    this_p1_H_maps, this_m2_H_maps = self.load_sim_data(ii)

                @globalize
                def _analyze1(full_index):
                    """Compute the cubic piece of the bispectrum numerator"""
                    index = full_index%len(self.all_bins)
                    chi_index = full_index//len(self.all_bins)
                    
                    u1,u2,u3,bin1,bin2,bin3,p_u = self.all_bins[index]
                    
                    # Compute combination of fields
                    tmp_sum  = p1_H_maps[bin1][u1]*this_p1_H_maps[bin2][u2]*this_m2_H_maps[bin3][u3]
                    tmp_sum += p1_H_maps[bin2][u2]*this_p1_H_maps[bin3][u3]*this_m2_H_maps[bin1][u1]
                    tmp_sum += p1_H_maps[bin3][u3]*this_p1_H_maps[bin1][u1]*this_m2_H_maps[bin2][u2]
                    
                    tmp_sum += this_p1_H_maps[bin1][u1]*p1_H_maps[bin2][u2]*this_m2_H_maps[bin3][u3]
                    tmp_sum += this_p1_H_maps[bin2][u2]*p1_H_maps[bin3][u3]*this_m2_H_maps[bin1][u1]
                    tmp_sum += this_p1_H_maps[bin3][u3]*p1_H_maps[bin1][u1]*this_m2_H_maps[bin2][u2]
                    
                    tmp_sum += this_p1_H_maps[bin1][u1]*this_p1_H_maps[bin2][u2]*m2_H_maps[bin3][u3]
                    tmp_sum += this_p1_H_maps[bin2][u2]*this_p1_H_maps[bin3][u3]*m2_H_maps[bin1][u1]
                    tmp_sum += this_p1_H_maps[bin3][u3]*this_p1_H_maps[bin1][u1]*m2_H_maps[bin2][u2]
    
                    # Perform map-level summation and take real/im part
                    chi = self.chi_arr[chi_index]
                    if p_u*chi==-1:
                        tmp_sum2 = tmp_sum*1.0j
                    else:
                        tmp_sum2 = tmp_sum
                    return -self.base.A_pix*np.real(np.sum(tmp_sum2))/3./self.sym_factor[index]/self.N_it
    
                with mp.Pool(self.base.nthreads) as p:
                    if verb:
                        b1_num += np.asarray(list(tqdm.tqdm(p.imap(_analyze1,range(self.N_b)),total=self.N_b)))
                    else:
                        b1_num += np.asarray(list(p.imap(_analyze1,range(self.N_b))))
    
        # Assemble numerator
        b_num = b3_num + b1_num
        del p1_H_maps, m2_H_maps
        return b_num
        
    def compute_fisher_contribution(self, seed, verb=False):
        """
        This computes the contribution to the Fisher matrix from a single pair of GRF simulations, created internally.
        """

        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()

        # Initialize output
        fish = np.zeros((self.N_b,self.N_b),dtype='complex')

        if verb: print("# Generating GRFs")
        a_maps = []
        for ii in range(2):
            # Compute two random realizations with known power spectrum
            a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e8), deconvolve_beam=True, output_type='harmonic'))
            
        # Define Q map code
        def compute_Q3(a_lm, weighting):
            """
            Assemble and return the Q3 maps in real- or harmonic-space, given a random field a_lm. 

            This computes maps with chi = +1 and -1 if parity='both'.

            The outputs are either Q(b) or S^-1.P.Q(b).
            """
            
            # Compute S^-1.P.a or A^-1.a maps
            if weighting=='Sinv':
                if self.ones_mask:
                    Uinv_a_lm = self.applySinv(self.beam_lm*a_lm, input_type='harmonic')
                else:
                    Uinv_a_lm = self.applySinv(self.mask*self.base.to_map(self.beam_lm*a_lm))
            elif weighting=='Ainv':
                Uinv_a_lm = self.base.applyAinv(a_lm, input_type='harmonic')
                
            # Compute (+-1)H maps and (+-2)H maps
            if verb: print("Creating H maps")
            global H_pm2_Uinv_maps, H_pm1_Uinv_maps 
            H_pm2_Uinv_maps = self._compute_H_pm(Uinv_a_lm, 2)
            H_pm1_Uinv_maps = self._compute_H_pm(Uinv_a_lm, 1)

            # Now assemble and return Q3 maps
            # Define arrays
            global tmp_Q
            tmp_Q = np.zeros((self.N_b,1+2*self.pol,len(Uinv_a_lm[0])),dtype='complex')
            if verb: print("Allocating %.2f GB of memory"%(tmp_Q.nbytes/1024./1024./1024.))
            
            # Iterate over fields and bins
            us = np.unique(self.all_bins[:,:3])
            task_vals = []
            for u1 in us:
                for u2 in us:
                    for bin1 in range(self.Nl_squeeze):
                        for bin2 in range(self.Nl_squeeze):
                            # Find which elements of the Q3 matrix this pair is used for
                            these_ind1 = np.where((self.all_bins[:,0]==u1)&(self.all_bins[:,1]==u2)&(self.all_bins[:,3]==bin1)&(self.all_bins[:,4]==bin2))[0]
                            these_ind2 = np.where((self.all_bins[:,1]==u1)&(self.all_bins[:,2]==u2)&(self.all_bins[:,4]==bin1)&(self.all_bins[:,5]==bin2))[0]
                            these_ind3 = np.where((self.all_bins[:,2]==u2)&(self.all_bins[:,0]==u1)&(self.all_bins[:,5]==bin2)&(self.all_bins[:,3]==bin1))[0] # note u2 > u1 ordering!
                            if len(these_ind1)+len(these_ind2)+len(these_ind3)==0: continue
                            
                            task_vals.append([u1,u2,bin1,bin2])
                   
            # Iterate over fields and bins
            us = np.unique(self.all_bins[:,:3])
            for u1 in us:
                for u2 in us:
                    for bin1 in range(self.Nl_squeeze):
                        for bin2 in range(self.Nl_squeeze):
                            
                            # Find which elements of the Q3 matrix this pair is used for
                            these_ind1 = np.where((self.all_bins[:,0]==u1)&(self.all_bins[:,1]==u2)&(self.all_bins[:,3]==bin1)&(self.all_bins[:,4]==bin2))[0]
                            these_ind2 = np.where((self.all_bins[:,1]==u1)&(self.all_bins[:,2]==u2)&(self.all_bins[:,4]==bin1)&(self.all_bins[:,5]==bin2))[0]
                            these_ind3 = np.where((self.all_bins[:,2]==u2)&(self.all_bins[:,0]==u1)&(self.all_bins[:,5]==bin2)&(self.all_bins[:,3]==bin1))[0] # note u2 > u1 ordering!
                            if len(these_ind1)+len(these_ind2)+len(these_ind3)==0: continue
                            
                            # Define H maps from precomputed fields
                            H_p1_1 = H_pm1_Uinv_maps[0][bin1][u1]
                            H_m1_1 = H_pm1_Uinv_maps[1][bin1][u1]

                            H_p1_2 = H_pm1_Uinv_maps[0][bin2][u2]
                            H_m1_2 = H_pm1_Uinv_maps[1][bin2][u2]

                            H_p2_1 = H_pm2_Uinv_maps[0][bin1][u1]
                            H_m2_1 = H_pm2_Uinv_maps[1][bin1][u1]

                            H_p2_2 = H_pm2_Uinv_maps[0][bin2][u2]
                            H_m2_2 = H_pm2_Uinv_maps[1][bin2][u2]

                            # Compute forward harmonic transforms (for both odd and even pieces simultaneously)
                            HH12 = self.base.to_lm_spin(H_p1_1*H_p1_2,H_m1_1*H_m1_2,2)[::-1].conj()
                            if bin1==bin2 and u1==u2:
                                HH12 -= 2*np.asarray([[1],[-1]])*self.base.to_lm_spin(H_m1_1*H_p2_2,-H_p1_1*H_m2_2,1).conj()
                            else:
                                HH12 -= np.asarray([[1],[-1]])*self.base.to_lm_spin(H_m1_1*H_p2_2,-H_p1_1*H_m2_2,1).conj()
                                HH12 -= np.asarray([[1],[-1]])*self.base.to_lm_spin(H_m1_2*H_p2_1,-H_p1_2*H_m2_1,1).conj()
                            del H_p1_1, H_m1_1, H_p1_2, H_m1_2, H_p2_1, H_m2_1, H_p2_2, H_m2_2
                            
                            def add_Q3_element(u3_index, bin3_index, these_ind):
                                # Iterate over these elements and add to the output arrays
                                for ii in these_ind:
                                    u3 = self.all_bins[ii,u3_index]
                                    bin3 = self.all_bins[ii,bin3_index]
                                    p_u = self.all_bins[ii,-1]

                                    this_Q = np.zeros((1+2*self.pol,2,len(Uinv_a_lm[0])),dtype='complex')
                                    this_Q[u3] = 1./3./self.sym_factor[ii]*self.ell_bins[bin3]*HH12

                                    # Compute parity +/-
                                    if p_u==1:
                                        tmp_Q_p = this_Q[:,0]+p_u*this_Q[:,1] # chi = 1
                                        tmp_Q_m = 1.0j*(this_Q[:,0]-p_u*this_Q[:,1]) # chi = -1
                                    else:
                                        tmp_Q_p = 1.0j*(this_Q[:,0]+p_u*this_Q[:,1]) # chi = 1
                                        tmp_Q_m = this_Q[:,0]-p_u*this_Q[:,1] # chi = -1

                                    if self.parity in ['even','both']:
                                        tmp_Q[ii] += tmp_Q_p
                                    if self.parity=='both':
                                        tmp_Q[ii+self.N_b//2] += tmp_Q_m
                                    if self.parity=='odd':
                                        tmp_Q[ii] += tmp_Q_m
                                    del tmp_Q_p, tmp_Q_m
                                        
                            add_Q3_element(2, 5, these_ind1)
                            add_Q3_element(0, 3, these_ind2)
                            add_Q3_element(1, 4, these_ind3)

            # Compute output Q3 maps
            Q_maps = np.zeros((self.N_b,len(np.asarray(self.beam_lm).ravel())),dtype='complex')
            if weighting=='Ainv' and verb: print("Applying S^-1 weighting to output")
            
            # Compute S^-1.P.Q ro Q
            for index in range(self.N_b):
                if weighting=='Ainv':
                    if self.ones_mask:
                        Q_maps[index] = self.applySinv(self.beam_lm*tmp_Q[index],input_type='harmonic').ravel()
                    else:
                        Q_maps[index] = self.applySinv(self.mask*self.base.to_map(self.beam_lm*tmp_Q[index])).ravel()
                elif weighting=='Sinv':
                    Q_maps[index] = (self.base.m_weight*tmp_Q[index]).ravel()
                        
            return Q_maps            

        # Compute Q maps
        if verb: print("\n# Computing Q3 map for S^-1 weighting")
        Q3_Sinv12 = [compute_Q3(a_map, 'Sinv') for a_map in a_maps]
        if verb: print("\n# Computing Q3 map for A^-1 weighting")
        Q3_Ainv12 = [compute_Q3(a_map, 'Ainv') for a_map in a_maps]

        # Assemble Fisher matrix
        if verb: print("\n# Assembling Fisher matrix\n")

        # Compute Fisher matrix as an outer product
        fish += (Q3_Sinv12[0].conj())@(Q3_Ainv12[0].T)
        fish += (Q3_Sinv12[1].conj())@(Q3_Ainv12[1].T)
        fish -= (Q3_Sinv12[0].conj())@(Q3_Ainv12[1].T)
        fish -= (Q3_Sinv12[1].conj())@(Q3_Ainv12[0].T)

        fish = fish.conj()/24.

        return fish.real
        
    def compute_fisher(self, N_it, verb=False):
        """
        Compute the Fisher matrix using N_it realizations. These are run in serial (since the code is already parallelized).
        
        For high-dimensional problems, it is usually preferred to split the computation across a cluster with MPI, calling compute_fisher_contribution for each instead of this function.
        """

        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Initialize output
        fish = np.zeros((self.N_b,self.N_b))
        
        # Iterate over N_it seeds
        for seed in range(N_it):
            print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
            fish += self.compute_fisher_contribution(seed, verb=verb*(seed==0))/N_it
        
        self.fish = fish
        self.inv_fish = np.linalg.inv(fish)
        return fish
    
    def Bl_unwindowed(self, data, fish=[], include_linear_term=True, verb=False):
        """
        Compute the unwindowed bispectrum estimator, using a precomputed or supplied Fisher matrix.
        
        Note that we return the imaginary part of odd-parity bispectra.

        We can optionally switch off the linear term.
        """

        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)
        
        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix first!")
        
        # Compute numerator
        Bl_num = self.Bl_numerator(data, include_linear_term=include_linear_term, verb=verb)
        
        # Apply normalization
        Bl_out = np.matmul(self.inv_fish,Bl_num)
        
        # Create output dictionary
        Bl_dict = {}
        index, config_index = 0,0
        chi_name = {1:'+',-1:'-'}
        for chi in self.chi_arr:

            # Iterate over fields
            for u in self.fields:
                Bl_dict['%s'%u+'%s'%chi_name[chi]] = Bl_out[index:index+len(self.get_ells(u)[0])]
                index += len(self.get_ells(u)[0])
            config_index += 1

        return Bl_dict
    
    ### IDEAL ESTIMATOR
    def Bl_numerator_ideal(self, data, verb=False):
        """
        Compute the numerator of the idealized bispectrum estimator for all fields of interest. We normalize by < mask^3 >. Note that this does not include one-field terms.
        """

        # Compute symmetry factor if necessary
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Transform to harmonic space and normalize by b.C^-1_th
        Cinv_data_lm = self.beam_lm*np.einsum('ijk,jk->ik',self.base.inv_Cl_tot_lm_mat,self.base.to_lm(data),order='C')

        # Compute (+1)H and (-2)H maps
        if verb: print("Computing H maps")
        global p1_H_maps, m2_H_maps
        p1_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*Cinv_data_lm.conj(),1) for bin1 in range(self.Nl_squeeze)]
        m2_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*Cinv_data_lm.conj(),-2) for bin1 in range(self.Nl_squeeze)]

        # Define output array
        b_num_ideal = np.zeros(self.N_b)
    
        @globalize
        def _analyze_num(full_index):
            """Compute the cubic piece of the bispectrum numerator"""
            index = full_index%len(self.all_bins)
            chi_index = index//len(self.all_bins)
            
            u1,u2,u3,bin1,bin2,bin3,p_u = self.all_bins[index]
            
            # Compute combination of fields
            tmp_sum  = p1_H_maps[bin1][u1]*p1_H_maps[bin2][u2]*m2_H_maps[bin3][u3]
            tmp_sum += p1_H_maps[bin2][u2]*p1_H_maps[bin3][u3]*m2_H_maps[bin1][u1]
            tmp_sum += p1_H_maps[bin3][u3]*p1_H_maps[bin1][u1]*m2_H_maps[bin2][u2]
                
            # Perform map-level summation and take real/im part
            chi = self.chi_arr[chi_index]
            if p_u*chi==-1:
                tmp_sum2 = tmp_sum*1.0j
            else:
                tmp_sum2 = tmp_sum
            return self.base.A_pix*np.real(np.sum(tmp_sum2))/3./self.sym_factor[index]
        
        if verb: print("Analyzing bispectrum numerator")
        with mp.Pool(self.base.nthreads) as p:
            if verb:
                b_num_ideal = np.asarray(list(tqdm.tqdm(p.imap(_analyze_num,range(self.N_b)),total=self.N_b)))
            else:
                b_num_ideal = np.asarray(list(p.imap(_analyze_num,range(self.N_b))))
        
        # Normalize
        b_num_ideal *= 1./np.mean(self.mask**3)
        return b_num_ideal
    
    def compute_fisher_ideal(self, verb=False, N_cpus=1):
        """This computes the idealized Fisher matrix for the bispectrum, including cross-correlations between different fields."""

        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()

        global _iterator
        def _iterator(index_input, verb=False):
            """Create an iterator for multiprocessing. This iterates over the first index."""
            
            # Compute full matrix
            fish = np.zeros((self.N_b,self.N_b))

            if self.parity=='both':
                if verb and (index_input%10)==0: print("Computing Fisher matrix row %d of %d"%(index_input+1,self.N_b//2))
            else:
                if verb and (index_input%10)==0: print("Computing Fisher matrix row %d of %d"%(index_input+1,self.N_b))
            
            # Iterate over first set of fields, parities, and bins
            index = -1
            for u in self.fields:
                u1, u2, u3 = [self.base.indices[u[i]] for i in range(3)]
                p_u = np.product([self.base.parities[u[i]] for i in range(3)])

                for bin1 in range(self.Nl_squeeze):
                    for bin2 in range(self.Nl_squeeze):
                        if u1==u2 and bin2<bin1: continue
                        for bin3 in range(self.Nl_squeeze):
                            if u2==u3 and bin3<bin2: continue

                            # skip bins outside the triangle conditions
                            if not self.is_good([bin1,bin2,bin3]): continue
                    
                            # Remove any unused squeezed triangles
                            if min([bin1,bin2,bin3])>=self.Nl: continue

                            # Update indices
                            index += 1

                            # Specialize to only the desired index
                            if index!=index_input: continue

                            # Iterate over second set of fields, parities, and bins
                            index_p = -1
                            for u_p in self.fields:
                                u1_p, u2_p, u3_p = [self.base.indices[u_p[i]] for i in range(3)]
                                p_u_p = np.product([self.base.parities[u_p[i]] for i in range(3)])

                                for bin1_p in range(self.Nl_squeeze):
                                    for bin2_p in range(self.Nl_squeeze):
                                        if u1_p==u2_p and bin2_p<bin1_p: continue
                                        for bin3_p in range(self.Nl_squeeze):
                                            if u2_p==u3_p and bin3_p<bin2_p: continue

                                            # skip bins outside the triangle conditions
                                            if not self.is_good([bin1_p,bin2_p,bin3_p]): continue
                    
                                            # Remove any unused squeezed triangles
                                            if min([bin1_p,bin2_p,bin3_p])>=self.Nl: continue

                                            # Update indices
                                            index_p += 1

                                            # fill in this part by symmetry!
                                            if index_p<index: continue

                                            if (np.sort([bin1,bin2,bin3])==np.sort([bin1_p,bin2_p,bin3_p])).all(): 

                                                # Now iterate over l values in bin
                                                value = 0.
                                                for l1 in range(self.l_bins_squeeze[bin1],self.l_bins_squeeze[bin1+1]):
                                                    for l2 in range(self.l_bins_squeeze[bin2],self.l_bins_squeeze[bin2+1]):
                                                        for l3 in range(max([abs(l1-l2),self.l_bins_squeeze[bin3]]),min([l1+l2+1,self.l_bins_squeeze[bin3+1]])):

                                                            # Define a factor of 1 or i to parity to separate
                                                            if (-1)**(l1+l2+l3)==1:
                                                                fac = 1.0
                                                            else:
                                                                fac = 1.0j

                                                            # Compute product of three inverse covariances with permutations
                                                            Cinv_bin = lambda i,j,l: self.base.inv_Cl_tot_mat[[u1,u2,u3][i],[u1_p,u2_p,u3_p][j]][l]*([bin1,bin2,bin3][i]==[bin1_p,bin2_p,bin3_p][j])
                                                            inv_cov3  = Cinv_bin(0,0,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,2,l3)
                                                            inv_cov3 += Cinv_bin(0,1,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,0,l3)
                                                            inv_cov3 += Cinv_bin(0,2,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,1,l3)
                                                            inv_cov3 += Cinv_bin(0,0,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,1,l3)
                                                            inv_cov3 += Cinv_bin(0,1,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,2,l3)
                                                            inv_cov3 += Cinv_bin(0,2,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,0,l3)
                                                            if inv_cov3==0: continue

                                                            tj = self.tj_sym(l1,l2,l3)
                                                            if tj==0: continue

                                                            # note absorbing factor of chi*p_u here 
                                                            value += fac*tj**2*self.beam[u1][l1]*self.beam[u1_p][l1]*self.beam[u2][l2]*self.beam[u2_p][l2]*self.beam[u3][l3]*self.beam[u3_p][l3]*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)/(4.*np.pi)*inv_cov3

                                                # Reconstruct output for even / odd ell and note symmetric matrix!
                                                for chi_index,chi in enumerate(self.chi_arr):
                                                    for chi_index_p, chi_p in enumerate(self.chi_arr):
                                                        if p_u*chi!=p_u_p*chi_p: continue
                                                        if p_u*chi==1:
                                                            out_value = value.real
                                                        else:
                                                            out_value = value.imag
                                                        fish[chi_index*self.N_b//2+index, chi_index_p*self.N_b//2+index_p] = out_value/self.sym_factor[index]/self.sym_factor[index_p]
                                                        fish[chi_index_p*self.N_b//2+index_p, chi_index*self.N_b//2+index] = out_value/self.sym_factor[index]/self.sym_factor[index_p]
            return fish
        
        
        # Assemble matrix, multiprocessing if necessary
        degeneracy = 1+(self.parity=='both')
        
        if N_cpus==1:
            fish = np.zeros((self.N_b, self.N_b))
            for i in range(self.N_b//degeneracy):
                fish += _iterator(i,verb=verb)
        else:
            p = mp.Pool(N_cpus)
            if verb: print("Multiprocessing computation on %d cores"%N_cpus)

            result = list(tqdm.tqdm(p.imap_unordered(_iterator,range(self.N_b//degeneracy)),total=self.N_b//degeneracy))
            fish = np.sum(result,axis=0)

        if verb: print("Fisher matrix computation complete\n")
        self.fish_ideal = fish
        self.inv_fish_ideal = np.linalg.inv(self.fish_ideal)

        return fish

    def Bl_ideal(self, data, fish_ideal=[], verb=False, N_cpus=1):
        """
        Compute the idealized bispectrum estimator, including normalization, if not supplied or already computed. Note that this normalizes by < mask^3 >.
        
        Note that we return the imaginary part of odd-parity bispectra. We can optionally multi-process computation of the ideal Fisher matrix.
        """

        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)

        if not hasattr(self,'inv_fish_ideal'):
            print("Computing ideal Fisher matrix")
            self.compute_fisher_ideal(verb=verb, N_cpus=N_cpus)

        # Compute numerator
        Bl_num_ideal = self.Bl_numerator_ideal(data, verb=verb)

        # Apply normalization
        Bl_out = np.matmul(self.inv_fish_ideal,Bl_num_ideal)

        # Create output dictionary
        Bl_dict = {}
        index, config_index = 0,0
        chi_name = {1:'+',-1:'-'}
        for chi in self.chi_arr:

            # Iterate over fields
            for u in self.fields:
                Bl_dict['%s'%u+'%s'%chi_name[chi]] = Bl_out[index:index+len(self.get_ells(u)[0])]
                index += len(self.get_ells(u)[0])
            config_index += 1

        return Bl_dict