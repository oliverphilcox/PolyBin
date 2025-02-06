### Code for ideal and unwindowed binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2025)
## This module contains the power spectrum estimation code

import numpy as np
import multiprocessing as mp
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

class PSpecBin():
    """Binned power spectrum estimation class. This takes the binning strategy as input and a base class. 
    We also feed in a function that applies the S^-1 operator (which is ideally beam.mask.C_l^{tot,-1}, where C_l^tot includes the beam and noise). 
    
    Inputs:
    - base: PolySpec class
    - mask: HEALPix mask to deconvolve
    - applySinv: function which returns S^-1 applied to a given input map
    - l_bins: array of bin edges
    - fields: which T/E/B power spectra to compute
    """
    def __init__(self, base, mask, applySinv, l_bins, fields=['TT','TE','TB','EE','EB','EE']):
        # Read in attributes
        self.base = base
        self.mask = mask
        self.applySinv = applySinv
        self.l_bins = l_bins
        self.min_l = np.min(l_bins)
        self.Nl = len(l_bins)-1
        self.pol = self.base.pol
        self.beam_lm = self.base.beam_lm
        self.fields = fields
        
        # Check correct fields are being used
        for f in fields:
            assert f in ['TT','TE','TB','EE','EB','BB'], "Unknown field '%s' supplied!"%f 
        assert len(fields)==len(np.unique(fields)), "Duplicate fields supplied!"
        self.N_p = len(self.fields)*self.Nl
        
        if not self.pol and fields!=['TT']:
            print("## Polarization mode not turned on; setting fields to TT only!")
            self.fields = ['TT']

        if np.max(self.l_bins)>base.lmax:
            raise Exception("Maximum l is larger than HEALPix resolution!")
        if np.max(self.l_bins)>base.lmax//2:
            print("## Caution: Maximum l is greater than HEALPix-lmax/2; this might cause boundary effects.")
        print("Binning: %d bins in [%d, %d]"%(self.Nl,self.min_l,np.max(self.l_bins)))
        print("Fields: %s"%fields)
        
        # Define l filters
        self.ell_bins = [(self.base.l_arr>=self.l_bins[bin1])&(self.base.l_arr<self.l_bins[bin1+1]) for bin1 in range(self.Nl)]
        self.all_ell_bins = np.vstack(self.ell_bins)
        
        # Check if window is uniform
        if np.std(self.mask)<1e-12 and np.abs(np.mean(self.mask)-1)<1e-12:
            print("Mask: ones")
            self.ones_mask = True
        else:
            print("Mask: spatially varying")
            self.ones_mask = False
 
    def get_ells(self):
        """
        Return a list of the central ell values for each power spectrum bin.
        """
        # Iterate over bins
        ls = [0.5*(self.l_bins[bin1]+self.l_bins[bin1+1]) for bin1 in range(self.Nl)]
        return ls

    ### OPTIMAL ESTIMATOR
    def Cl_numerator(self, data):
        """Compute the numerator of the unwindowed power spectrum estimator.
        """

        # Apply S^-1 to data and transform to harmonic space
        h_data_lm = self.applySinv(data, input_type='map')
        
        # Compute numerator
        Cl_num = []
        for u in self.fields:
            u1, u2 = self.base.indices[u[0]], self.base.indices[u[1]]
            
            Delta2_u = (u1==u2)+1.
            
            # Compute quadratic product of data matrices
            spec_squared = np.conj(h_data_lm[u1])*h_data_lm[u2]
            
            # Compute full numerator
            Cl_u1u2 = 1./Delta2_u*np.sum(self.base.m_weight*spec_squared*self.all_ell_bins,axis=1)
            
            Cl_num.append(np.real(Cl_u1u2))

        return np.asarray(Cl_num)
    
    def compute_fisher_contribution(self, seed, verb=False):
        """This computes the contribution to the Fisher matrix from a single GRF simulation, created internally."""

        # Initialize output
        fish = np.zeros((self.N_p,self.N_p))

        # Compute a random realization with known power spectrum and weight appropriately
        if verb: print("Generating GRF")
        a_lm = self.base.generate_data(seed=seed+int(1e7), output_type='harmonic', deconvolve_beam=True)
        
        # Define Q map code
        def compute_Q2(weighting):
            """
            Assemble and return the Q2 maps in real- or harmonic-space, given a weighting scheme.

            The outputs are either Q(b) or S^-1.P.Q(b).
            """
            
            # Compute S^-1.P or A^-1 maps
            if weighting=='Sinv':
                if self.ones_mask:
                    Uinv_a_lm = self.applySinv(self.beam_lm*a_lm, input_type='harmonic')
                else:
                    Uinv_a_lm = self.applySinv(self.mask*self.base.to_map(self.beam_lm*a_lm))
            elif weighting=='Ainv':
                Uinv_a_lm = self.base.applyAinv(a_lm, input_type='harmonic')
                
            # Now assemble and return Q2 maps
            # Define arrays
            Q_maps = np.zeros((self.N_p,len(np.asarray(self.beam_lm).ravel())),dtype='complex')

            # Iterate over fields and bins
            index = -1
            for u in self.fields:
                u1, u2 = self.base.indices[u[0]], self.base.indices[u[1]]

                Delta2_u = (u1==u2)+1.

                # Compute T/E/B field
                Uinv_a_lm_u = np.zeros_like(Uinv_a_lm)
                Uinv_a_lm_u[u1] += Uinv_a_lm[u2]
                Uinv_a_lm_u[u2] += Uinv_a_lm[u1]

                for bin1 in range(self.Nl):
                    index += 1

                    # Define summand
                    summand = Uinv_a_lm_u*self.ell_bins[bin1]/Delta2_u

                    # Optionally apply weighting and add to output arrays in real or harmonic space
                    if weighting=='Ainv':
                        if self.ones_mask:
                            Q_maps[index] = self.applySinv(self.beam_lm*summand,input_type='harmonic').ravel()
                        else:
                            Q_maps[index] = self.applySinv(self.mask*self.base.to_map(self.beam_lm*summand)).ravel()
                    elif weighting=='Sinv':
                        Q_maps[index] = (self.base.m_weight*summand).ravel()
            return Q_maps
        
        if verb: print("\nComputing Q2 map for S^-1 weighting")
        Q2_Sinv = compute_Q2('Sinv')
        if verb: print("\nComputing Q2 map for A^-1 weighting")
        Q2_Ainv = compute_Q2('Ainv')

        # Assemble Fisher matrix
        if verb: print("Assembling Fisher matrix\n")

        # Compute Fisher matrix as an outer product
        fish += 0.5*np.real(Q2_Sinv.conj()@(Q2_Ainv.T))

        return fish

    def compute_fisher(self, N_it, verb=False):
        """Compute the Fisher matrix using N_it realizations. These are run in serial (since the code is already parallelized)."""

        # Initialize output
        fish = np.zeros((self.N_p,self.N_p))

        # Iterate over N_it seeds
        for seed in range(N_it):
            print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
            fish += self.compute_fisher_contribution(seed, verb=verb)/N_it
        
        self.fish = fish
        self.inv_fish = np.linalg.inv(fish)
        return fish
    
    def Cl_unwindowed(self, data, fish=[]):
        """Compute the unwindowed power spectrum estimator. Note that the fisher matrix must be computed before this is run, or it can be supplied separately."""
        
        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)
        
        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix first!")
        
        # Compute numerator
        Cl_num = np.concatenate(self.Cl_numerator(data))

        # Apply normalization and restructure
        Cl_out = np.matmul(self.inv_fish,Cl_num)
        
        # Create output dictionary of spectra
        Cl_dict = {}
        index = 0
        for u in self.fields:
            Cl_dict['%s'%u] = Cl_out[index:index+self.Nl]
            index += self.Nl

        return Cl_dict
       
    ### IDEAL ESTIMATOR
    def Cl_numerator_ideal(self, data):
        """Compute the numerator of the idealized power spectrum estimator for all fields of interest. We normalize by < mask^2 >.
        """
        # Transform to harmonic space and normalize by b.C_th^{-1}
        Cinv_data_lm = self.beam_lm*np.einsum('ijk,jk->ik',self.base.inv_Cl_tot_lm_mat,self.base.to_lm(data),order='C')
        
        # Compute numerator
        Cl_num = []
        for u in self.fields:
            u1, u2 = self.base.indices[u[0]], self.base.indices[u[1]]
            
            Delta2_u = (u1==u2)+1.
            
            # Compute quadratic product of data matrices
            spec_squared = np.conj(Cinv_data_lm[u1])*Cinv_data_lm[u2]
            
            # Compute full numerator
            Cl_u1u2 = 1./Delta2_u*np.sum(self.base.m_weight*spec_squared*self.all_ell_bins,axis=1)/np.mean(self.mask**2)
            
            Cl_num.append(np.real(Cl_u1u2))

        return np.asarray(Cl_num)
                
    def compute_fisher_ideal(self):
        """This computes the idealized Fisher matrix for the power spectrum."""
        
        # Define output array
        fish = np.zeros((len(self.fields)*self.Nl,len(self.fields)*self.Nl))
        
        # Iterate over fields
        for i,u in enumerate(self.fields):
            
            u1, u2 = self.base.indices[u[0]], self.base.indices[u[1]]
            Delta2_u = (u1==u2)+1.
            
            for j,u_p in enumerate(self.fields):
            
                u1_p, u2_p = self.base.indices[u_p[0]], self.base.indices[u_p[1]]
                Delta2_u_p = (u1_p==u2_p)+1.
            
                # Compute product of two inverse covariances with permutations
                inv_cov_sq = self.base.inv_Cl_tot_lm_mat[u2_p,u1]*self.base.inv_Cl_tot_lm_mat[u2,u1_p]
                inv_cov_sq += self.base.inv_Cl_tot_lm_mat[u2_p,u2]*self.base.inv_Cl_tot_lm_mat[u1,u1_p]
                
                # Assemble fisher matrix
                fish_diag = 1./(Delta2_u*Delta2_u_p)*np.sum(self.base.m_weight*inv_cov_sq*self.all_ell_bins*self.beam_lm[u1]*self.beam_lm[u2]*self.beam_lm[u1_p]*self.beam_lm[u2_p],axis=1)
                
                # Add to output array
                fish[i*self.Nl:(i+1)*self.Nl,j*self.Nl:(j+1)*self.Nl] = np.diag(np.real(fish_diag))
        
        self.fish_ideal = fish
        self.inv_fish_ideal = np.linalg.inv(self.fish_ideal)
        
        return fish
    
    def Cl_ideal(self, data, fish_ideal=[]):
        """Compute the idealized power spectrum estimator, including normalization, if not supplied or already computed. Note that this normalizes by < mask^2 >.
        """

        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)

        if not hasattr(self,'inv_fish_ideal'):
            print("Computing ideal Fisher matrix")
            self.compute_fisher_ideal()

        # Compute numerator
        Cl_num_ideal = np.concatenate(self.Cl_numerator_ideal(data))

        # Apply normalization and restructure
        Cl_out = np.matmul(self.inv_fish_ideal,Cl_num_ideal)
        
        # Create output dictionary of spectra
        Cl_dict = {}
        index = 0
        for u in self.fields:
            Cl_dict['%s'%u] = Cl_out[index:index+self.Nl]
            index += self.Nl

        return Cl_dict
       