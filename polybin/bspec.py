### Code for ideal and unwindowed binned polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022)
## This module contains the bispectrum estimation code

import numpy as np
import multiprocessing as mp
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
import pywigxjpf as wig

class BSpec():
    """Bispectrum estimation class. This takes the binning strategy as input and a base class. 
    We also feed in a function that applies the S^-1 operator.

    Note that we can additionally compute squeezed triangles by setting l_bins_squeeze, giving a different lmax for short and long sides.
    
    Inputs:
    - base: PolyBin class
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
        
    def _check_bin(self, bin1, bin2, bin3):
        """Return one if modes in the bin satisfy the triangle conditions, or zero else.

        This is used either for all triangles in the bin, or just the center of the bin.
        """
        if self.include_partial_triangles:
            good = 0
            for l1 in range(self.l_bins[bin1],self.l_bins[bin1+1]):
                for l2 in range(self.l_bins_squeeze[bin2],self.l_bins_squeeze[bin2+1]):
                    for l3 in range(self.l_bins_squeeze[bin3],self.l_bins_squeeze[bin3+1]):
                        if l1>=abs(l1-l2) and l3<=l1+l2:
                            good = 1
                        if good==1: break
                    if good==1: break
                if good==1: break
            if good==1: return 1
            else:
                return 0
        else:
            l1 = 0.5*(self.l_bins[bin1]+self.l_bins[bin1+1])
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
        self.sym_factor_all = []

        # Iterate over parities
        for chi in self.chi_arr:
            
            # Iterate over fields
            for u in self.fields:
                u1, u2, u3 = [self.base.indices[u[i]] for i in range(3)]

                sym_factor = []
                for bin1 in range(self.Nl):
                    for bin2 in range(self.Nl_squeeze):
                        if u1==u2 and bin2<bin1: continue
                        for bin3 in range(self.Nl_squeeze):
                            if u2==u3 and bin3<bin2: continue

                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,bin3): continue

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
        print("Using %d combination(s) of fields/parities"%len(self.sym_factor_all))
        print("Using a maximum of %d bins per field/parity"%np.max([len(s) for s in self.sym_factor_all]))
        print("Using a total of %d bins"%self.N_b)
        
    def _compute_H_pm(self, a_lm, spin):
        """
        Compute (+-s)H maps for an input a_lm field. This is defined as Sum_lm (+-s)Y_lm(n) a_lm.
        
        This function is used in computation of the bispectrum Fisher matrix.
        """
        assert spin>0
        if self.ones_mask:
            H_plus = [self.base.compute_spin_transform_map(a_lm*self.ell_bins[bin1]*self.beam_lm, spin) for bin1 in range(self.Nl_squeeze)]
        else:
            H_plus = [self.mask*self.base.compute_spin_transform_map(a_lm*self.ell_bins[bin1]*self.beam_lm, spin) for bin1 in range(self.Nl_squeeze)]
        H_minus = [np.asarray([(-1)**spin*H.conj() for H in this_H_plus]) for this_H_plus in H_plus]
        return H_plus, H_minus
    
    def get_ells(self, field='TTT'):
        """
        Return an array with the central l1, l2, l3 values for each bispectrum bin, given a set of fields.
        """
        assert field in self.fields, "Specified field '%s' not in inputs!"%field
        u1, u2, u3 = [self.base.indices[field[i]] for i in range(3)]

        l1s, l2s, l3s = [],[],[]
        for bin1 in range(self.Nl):
            l1 = 0.5*(self.l_bins[bin1]+self.l_bins[bin1+1])
            for bin2 in range(self.Nl_squeeze):
                if u1==u2 and bin2<bin1: continue
                l2 = 0.5*(self.l_bins_squeeze[bin2]+self.l_bins_squeeze[bin2+1])
                for bin3 in range(self.Nl_squeeze):
                    if u2==u3 and bin3<bin2: continue
                    l3 = 0.5*(self.l_bins_squeeze[bin3]+self.l_bins_squeeze[bin3+1])
            
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue
                    
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
        if self.ones_mask:
            Wh_alpha_lm = self.applySinv(sim, input_type=input_type, output_type='harmonic')
        else:
            Wh_alpha_lm = self.base.to_lm(self.mask*self.applySinv(sim,input_type=input_type))

        # Compute (+1)H and (-2)H maps
        p1_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*self.beam_lm*Wh_alpha_lm.conj(),1) for bin1 in range(self.Nl_squeeze)]
        m2_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*self.beam_lm*Wh_alpha_lm.conj(),-2) for bin1 in range(self.Nl_squeeze)]
        
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
            Cl_input = self.base.Cl

        if preload:
            self.preload = True

            # Define lists of maps
            self.p1_H_maps, self.m2_H_maps = [],[]

            # Iterate over simulations and preprocess appropriately
            for ii in range(self.N_it):
                if verb: print("Generating bias simulation %d of %d"%(ii+1,self.N_it))
                
                # Generate simulation and Fourier transform
                if self.ones_mask:
                    alpha_lm = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, b_input=b_input, add_B=add_B, remove_mean=remove_mean, output_type='harmonic')
                    p1_H_maps, m2_H_maps = self._process_sim(alpha_lm, input_type='harmonic')
                else:
                    alpha = self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, b_input=b_input, add_B=add_B, remove_mean=remove_mean)
                    p1_H_maps, m2_H_maps = self._process_sim(alpha)
                    
                self.p1_H_maps.append(p1_H_maps)
                self.m2_H_maps.append(m2_H_maps)
                
        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            if self.ones_mask:
                self.load_sim_data = lambda ii: self._process_sim(self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, b_input=b_input, add_B=add_B, remove_mean=remove_mean, output_type='harmonic'), input_type='harmonic')
            else:
                self.load_sim_data = lambda ii: self._process_sim(self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, b_input=b_input, add_B=add_B, remove_mean=remove_mean))

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
        
        # Apply W * S^-1 to data and transform to harmonic space
        if self.ones_mask:
            Wh_data_lm = self.applySinv(data, input_type='map', output_type='harmonic')
        else:
            Wh_data_lm = self.base.to_lm(self.mask*self.applySinv(data))
        
        # Compute (+1)H and (-2)H maps
        if verb: print("Computing H maps")
        p1_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*self.beam_lm*Wh_data_lm.conj(),1) for bin1 in range(self.Nl_squeeze)]
        m2_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*self.beam_lm*Wh_data_lm.conj(),-2) for bin1 in range(self.Nl_squeeze)]

        # Compute b_3 part of cubic estimator
        b3_num = np.zeros(self.N_b)
        index = 0
        if verb: print("Computing b_3 piece")
        for u in self.fields:
            u1, u2, u3 = [self.base.indices[u[i]] for i in range(3)]
            p_u = np.product([self.base.parities[u[i]] for i in range(3)])
            
            for bin1 in range(self.Nl):
                for bin2 in range(self.Nl_squeeze):
                    if u1==u2 and bin2<bin1: continue
                    for bin3 in range(self.Nl_squeeze):
                        if u2==u3 and bin3<bin2: continue

                        # skip bins outside the triangle conditions
                        if not self._check_bin(bin1,bin2,bin3): continue

                        # Compute combination of fields
                        tmp_sum  = p1_H_maps[bin1][u1]*p1_H_maps[bin2][u2]*m2_H_maps[bin3][u3]
                        tmp_sum += p1_H_maps[bin2][u2]*p1_H_maps[bin3][u3]*m2_H_maps[bin1][u1]
                        tmp_sum += p1_H_maps[bin3][u3]*p1_H_maps[bin1][u1]*m2_H_maps[bin2][u2]
                            
                        # Perform map-level summation and take real/im part
                        chi_index = 0
                        for chi in self.chi_arr:
                            if p_u*chi==-1:
                                tmp_sum2 = tmp_sum*1.0j
                            else:
                                tmp_sum2 = tmp_sum
                            b3_num[chi_index*self.N_b//2+index] = self.base.A_pix*np.real(np.sum(tmp_sum2))/3./self.sym_factor[index]
                            chi_index += 1
                        index += 1
    
        # Compute b_1 part of cubic estimator, averaging over simulations
        b1_num = np.zeros(self.N_b)
        
        if not include_linear_term:
            print("No linear correction applied!")
        else:
            for ii in range(self.N_it):
                if (ii+1)%5==0 and verb: print("Computing b_1 piece from simulation %d"%(ii+1))

                # Load processed bias simulations 
                if self.preload:
                    this_p1_H_maps, this_m2_H_maps = self.p1_H_maps[ii], self.m2_H_maps[ii]
                else:
                    this_p1_H_maps, this_m2_H_maps = self.load_sim_data(ii)

                index = 0
                for u in self.fields:
                    u1, u2, u3 = [self.base.indices[u[i]] for i in range(3)]
                    p_u = np.product([self.base.parities[u[i]] for i in range(3)])
                    
                    for bin1 in range(self.Nl):
                        for bin2 in range(self.Nl_squeeze):
                            if u1==u2 and bin2<bin1: continue
                            for bin3 in range(self.Nl_squeeze):
                                if u2==u3 and bin3<bin2: continue

                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,bin3): continue

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
                        
                                # Perform map-level summation and take real/im part, summing over permutations
                                chi_index = 0
                                for chi in self.chi_arr:
                                    if p_u*chi==-1:
                                        tmp_sum2 = tmp_sum*1.0j
                                    else:
                                        tmp_sum2 = tmp_sum
                                    b1_num[chi_index*self.N_b//2+index] -= self.base.A_pix*np.real(np.sum(tmp_sum2))/3./self.sym_factor[index]/self.N_it
                                    chi_index += 1
                                index += 1
    
        # Assemble numerator
        b_num = b3_num + b1_num
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
            # Compute two random realizations with known power spectrum and weight appropriately
            if self.ones_mask:
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e8), output_type='harmonic'))
            else:
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e8)))

        # Define Q map code
        def compute_Q3(a_map, weighting):
            """
            Assemble and return the Q3 maps in real- or harmonic-space, given a random field a_lm. 

            This computes maps with chi = +1 and -1 if parity='both'.

            The outputs are either Q(b) or WS^-1WQ(b).
            """
            if weighting=='Sinv':
                weighting_function = self.applySinv
            elif weighting=='Ainv':
                weighting_function = self.base.applyAinv

            # Weight maps appropriately
            if self.ones_mask:
                WUinv_a_lm = weighting_function(a_map, input_type='harmonic', output_type='harmonic')
            else:
                WUinv_a_lm = self.base.to_lm(self.mask*weighting_function(a_map))

            # Compute (+-1)H maps and (+-2)H maps
            if verb: print("Creating H maps")
            H_pm2_Uinv_maps = self._compute_H_pm(WUinv_a_lm, 2)
            H_pm1_Uinv_maps = self._compute_H_pm(WUinv_a_lm, 1)

            # Now assemble and return Q3 maps
            # Define arrays
            tmp_Q = np.zeros((self.N_b,1+2*self.pol,len(WUinv_a_lm[0])),dtype='complex')
            if verb: print("Allocating %.2f GB of memory"%(tmp_Q.nbytes/1024./1024./1024.))
            
            # Compute indices for each triplet of fields
            indices = []
            for u in self.fields:
                u1, u2, u3 = [self.base.indices[u[i]] for i in range(3)]
                p_u = np.product([self.base.parities[u[i]] for i in range(3)])

                for bin1 in range(self.Nl):
                    for bin2 in range(self.Nl_squeeze):
                        if u1==u2 and bin2<bin1: continue
                        
                        for bin3 in range(self.Nl_squeeze):
                            if u2==u3 and bin3<bin2: continue

                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,bin3): continue
                            indices.append([u1,u2,u3,bin1,bin2,bin3,p_u])
            indices = np.asarray(indices)
                            
            # Iterate over fields and bins
            us = np.unique(indices[:,:3])
            for u1 in us:
                for u2 in us:
                    for bin1 in range(self.Nl_squeeze):
                        for bin2 in range(self.Nl_squeeze):
                            
                            # Find which elements of the Q3 matrix this pair is used for
                            these_ind1 = np.where((indices[:,0]==u1)&(indices[:,1]==u2)&(indices[:,3]==bin1)&(indices[:,4]==bin2))[0]
                            these_ind2 = np.where((indices[:,1]==u1)&(indices[:,2]==u2)&(indices[:,4]==bin1)&(indices[:,5]==bin2))[0]
                            these_ind3 = np.where((indices[:,2]==u2)&(indices[:,0]==u1)&(indices[:,5]==bin2)&(indices[:,3]==bin1))[0] # note u2 > u1 ordering!
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
                            
                            def add_Q3_element(u3_index, bin3_index, these_ind):
                                # Iterate over these elements and add to the output arrays
                                for ii in these_ind:
                                    u3 = indices[ii,u3_index]
                                    bin3 = indices[ii,bin3_index]
                                    p_u = indices[ii,-1]

                                    this_Q = np.zeros((1+2*self.pol,2,len(WUinv_a_lm[0])),dtype='complex')
                                    this_Q[u3] = 1./3./self.sym_factor[ii]*self.ell_bins[bin3]*self.beam_lm[u3]*HH12

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
                                        
                            add_Q3_element(2, 5, these_ind1)
                            add_Q3_element(0, 3, these_ind2)
                            add_Q3_element(1, 4, these_ind3)

            # Compute output Q3 maps
            Q_maps = np.zeros((self.N_b,len(a_maps[0].ravel())),dtype='complex')
            if weighting=='Ainv' and verb: print("Applying S^-1 weighting to output")
            for index in range(self.N_b):
                if weighting=='Ainv':
                    if self.ones_mask:
                        Q_maps[index] = self.applySinv(tmp_Q[index],input_type='harmonic',output_type='harmonic').ravel()
                    else:
                        Q_maps[index] = (self.mask*self.applySinv(self.mask*self.base.to_map(tmp_Q[index]))).ravel()
                elif weighting=='Sinv':
                    if self.ones_mask:
                        Q_maps[index] = (self.base.m_weight*tmp_Q[index]).ravel()
                    else:
                        Q_maps[index] = self.base.A_pix*self.base.to_map(tmp_Q[index]).ravel()
                        
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
        
    def compute_fisher(self, N_it, N_cpus=1, verb=False):
        """
        Compute the Fisher matrix using N_it pairs of realizations. If N_cpus > 1, this parallelizes the operations (though HEALPix is already parallelized so the speed-up is not particularly significant).
        
        For high-dimensional problems, it is usually preferred to split the computation across a cluster with MPI, calling compute_fisher_contribution for each instead of this function.
        
        """

        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Initialize output
        fish = np.zeros((self.N_b,self.N_b))

        global _iterable
        def _iterable(seed):
            return self.compute_fisher_contribution(seed, verb=verb*(seed==0))
        
        if N_cpus==1:
            for seed in range(N_it):
                if seed%5==0: print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
                fish += self.compute_fisher_contribution(seed, verb=verb*(seed==0))/N_it
        else:
            p = mp.Pool(N_cpus)
            print("Computing Fisher contribution from %d pairs of Monte Carlo simulations on %d threads"%(N_it, N_cpus))
            all_fish = list(tqdm.tqdm(p.imap_unordered(_iterable,np.arange(N_it)),total=N_it))
            fish = np.sum(all_fish,axis=0)/N_it
        
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
                Bl_dict['%s'%u+'%s'%chi_name[chi]] = Bl_out[index:index+len(self.sym_factor_all[config_index])]
                index += len(self.sym_factor_all[config_index])
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
        
        # Transform to harmonic space and normalize by 1/C_th
        Cinv_data_lm = np.einsum('ijk,jk->ik',self.base.inv_Cl_lm_mat,self.base.to_lm(data),order='C')

        # Compute (+1)H and (-2)H maps
        if verb: print("Computing H maps")
        p1_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*self.beam_lm*Cinv_data_lm.conj(),1) for bin1 in range(self.Nl_squeeze)]
        m2_H_maps = [self.base.compute_spin_transform_map(self.ell_bins[bin1]*self.beam_lm*Cinv_data_lm.conj(),-2) for bin1 in range(self.Nl_squeeze)]

        # Define output array
        b_num_ideal = np.zeros(self.N_b)

        # Iterate over fields and bins
        index = 0
        for u in self.fields:
            u1, u2, u3 = [self.base.indices[u[i]] for i in range(3)]
            p_u = np.product([self.base.parities[u[i]] for i in range(3)])
            if verb: print("Analyzing bispectrum numerator for field %s"%u)

            for bin1 in range(self.Nl):
                for bin2 in range(self.Nl_squeeze):
                    if u1==u2 and bin2<bin1: continue
                    for bin3 in range(self.Nl_squeeze):
                        if u2==u3 and bin3<bin2: continue

                        # skip bins outside the triangle conditions
                        if not self._check_bin(bin1,bin2,bin3): continue

                        # Compute combination of fields
                        tmp_sum  = p1_H_maps[bin1][u1]*p1_H_maps[bin2][u2]*m2_H_maps[bin3][u3]
                        tmp_sum += p1_H_maps[bin2][u2]*p1_H_maps[bin3][u3]*m2_H_maps[bin1][u1]
                        tmp_sum += p1_H_maps[bin3][u3]*p1_H_maps[bin1][u1]*m2_H_maps[bin2][u2]
                            
                        # Perform map-level summation and take real/im part for each parity
                        chi_index = 0
                        for chi in self.chi_arr:
                            if p_u*chi==-1:
                                tmp_sum2 = tmp_sum*1.0j
                            else:
                                tmp_sum2 = tmp_sum
                            b_num_ideal[chi_index*self.N_b//2+index] = self.base.A_pix*np.real(np.sum(tmp_sum2))/3./self.sym_factor[index]
                            chi_index += 1
                        index += 1

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

                for bin1 in range(self.Nl):
                    for bin2 in range(self.Nl_squeeze):
                        if u1==u2 and bin2<bin1: continue
                        for bin3 in range(self.Nl_squeeze):
                            if u2==u3 and bin3<bin2: continue

                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,bin3): continue

                            # Update indices
                            index += 1

                            # Specialize to only the desired index
                            if index!=index_input: continue

                            # Iterate over second set of fields, parities, and bins
                            index_p = -1
                            for u_p in self.fields:
                                u1_p, u2_p, u3_p = [self.base.indices[u_p[i]] for i in range(3)]
                                p_u_p = np.product([self.base.parities[u_p[i]] for i in range(3)])

                                for bin1_p in range(self.Nl):
                                    for bin2_p in range(self.Nl_squeeze):
                                        if u1_p==u2_p and bin2_p<bin1_p: continue
                                        for bin3_p in range(self.Nl_squeeze):
                                            if u2_p==u3_p and bin3_p<bin2_p: continue

                                            # skip bins outside the triangle conditions
                                            if not self._check_bin(bin1_p,bin2_p,bin3_p): continue

                                            # Update indices
                                            index_p += 1

                                            # fill in this part by symmetry!
                                            if index_p<index: continue

                                            if (np.sort([bin1,bin2,bin3])==np.sort([bin1_p,bin2_p,bin3_p])).all(): 

                                                # Now iterate over l values in bin
                                                value = 0.
                                                for l1 in range(self.l_bins[bin1],self.l_bins[bin1+1]):
                                                    for l2 in range(self.l_bins_squeeze[bin2],self.l_bins_squeeze[bin2+1]):
                                                        for l3 in range(max([abs(l1-l2),self.l_bins_squeeze[bin3]]),min([l1+l2+1,self.l_bins_squeeze[bin3+1]])):

                                                            # Define a factor of 1 or i to parity to separate
                                                            if (-1)**(l1+l2+l3)==1:
                                                                fac = 1.0
                                                            else:
                                                                fac = 1.0j

                                                            # Compute product of three inverse covariances with permutations
                                                            Cinv_bin = lambda i,j,l: self.base.inv_Cl_mat[[u1,u2,u3][i],[u1_p,u2_p,u3_p][j]][l]*([bin1,bin2,bin3][i]==[bin1_p,bin2_p,bin3_p][j])
                                                            inv_cov3  = Cinv_bin(0,0,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,2,l3)
                                                            inv_cov3 += Cinv_bin(0,1,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,0,l3)
                                                            inv_cov3 += Cinv_bin(0,2,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,1,l3)
                                                            inv_cov3 += Cinv_bin(0,0,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,1,l3)
                                                            inv_cov3 += Cinv_bin(0,1,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,2,l3)
                                                            inv_cov3 += Cinv_bin(0,2,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,0,l3)
                                                            if inv_cov3==0: continue

                                                            tj = self.base.tj_sym(l1,l2,l3)
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
                Bl_dict['%s'%u+'%s'%chi_name[chi]] = Bl_out[index:index+len(self.sym_factor_all[config_index])]
                index += len(self.sym_factor_all[config_index])
                config_index += 1

        return Bl_dict

        
