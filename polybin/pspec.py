### Code for ideal and unwindowed binned polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022)
## This module contains the power spectrum estimation code

import healpy
import numpy as np
import multiprocessing as mp
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
import pywigxjpf as wig

class PSpec():
    """Power spectrum estimation class. This takes the binning strategy as input and a base class. 
    We also feed in a function that applies the S^-1 operator in real space.
    
    Inputs:
    - base: PolyBin class
    - mask: HEALPix mask to deconvolve
    - applySinv: function which returns S^-1 applied to a given input map
    - min_l, dl, Nl: binning parameters
    """
    def __init__(self, base, mask, applySinv, min_l, dl, Nl):
        # Read in attributes
        self.base = base
        self.mask = mask
        self.applySinv = applySinv
        self.min_l = min_l
        self.dl = dl
        self.Nl = Nl
        self.beam_lm = self.base.beam_lm
        
        if min_l+Nl*dl>base.lmax:
            raise Exception("Maximum l is larger than HEALPix resolution!")
        print("Binning: %d bins in [%d, %d]"%(Nl,min_l,min_l+Nl*dl))
        
        # Define l filters
        self.ell_bins = [(self.base.l_arr>=self.min_l+self.dl*bin1)&(self.base.l_arr<self.min_l+self.dl*(bin1+1)) for bin1 in range(self.Nl)]
        self.all_ell_bins = np.vstack(self.ell_bins)
        
        # Define m weights (for complex conjugates)
        self.m_weight = (1.+1.*(self.base.m_arr>0.))
 
    def get_ells(self):
        """
        Return a list of the central ell values for each power spectrum bin.
        """
        # Iterate over bins
        ls = [self.min_l+(bin1+0.5)*self.dl-0.5 for bin1 in range(self.Nl)]
        return ls

    ### OPTIMAL ESTIMATOR
    def Cl_numerator(self, data):
        """Compute the numerator of the unwindowed power spectrum estimator.
        """
        # Apply W * S^-1 to data and transform to harmonic space
        Wh_data_lm = self.base.to_lm(self.mask*self.applySinv(data))

        # Compute numerator (including beam)
        Cl_num = 0.5*np.real(np.sum(self.m_weight*Wh_data_lm*np.conj(Wh_data_lm)*self.all_ell_bins*self.beam_lm**2,axis=1))

        return Cl_num
    
    def compute_fisher_contribution(self, seed):
        """This computes the contribution to the Fisher matrix from a single GRF simulation, created internally."""
        
        # Initialize output
        fish = np.zeros((self.Nl,self.Nl))
        
        # Compute random realization with known power spectrum
        u = self.base.generate_data(seed=seed+int(1e7))

        # Compute weighted fields
        Sinv_u = self.applySinv(u)
        Uinv_u = self.base.applyUinv(u)

        # Compute Q_b fields, including S^-1 weighting
        WSinv_u_lm = self.base.to_lm(self.mask*Sinv_u)
        WUinv_u_lm = self.base.to_lm(self.mask*Uinv_u)
        Sinv_Q_b_Uinv_u = [self.applySinv(self.base.to_map(WUinv_u_lm*self.ell_bins[bin1]*self.beam_lm**2)*self.mask) for bin1 in range(self.Nl)]
        Q_b_Sinv_u = [self.base.to_map(WSinv_u_lm*self.ell_bins[bin1]*self.beam_lm**2)*self.mask for bin1 in range(self.Nl)]

        # Compute the Fisher matrix
        for bin1 in range(self.Nl):
            for bin2 in range(self.Nl):
                fish[bin1,bin2] = 0.5*np.real(self.base.A_pix*np.sum(Q_b_Sinv_u[bin1]*Sinv_Q_b_Uinv_u[bin2]))
        
        # Return matrix
        return fish
    
    def compute_fisher(self, N_it, N_cpus=1):
        """Compute the Fisher matrix using N_it realizations. If N_cpus > 1, this parallelizes the operations."""

        # Initialize output
        fish = np.zeros((self.Nl,self.Nl))

        global _iterable
        def _iterable(seed):
            return self.compute_fisher_contribution(seed)
        
        if N_cpus==1:
            for seed in range(N_it):
                if seed%5==0: print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
                fish += self.compute_fisher_contribution(seed)/N_it
        else:
            p = mp.Pool(N_cpus)
            print("Computing Fisher contribution from %d Monte Carlo simulations on %d threads"%(N_it, N_cpus))
            all_fish = list(tqdm.tqdm(p.imap_unordered(_iterable,np.arange(N_it)),total=N_it))
            fish = np.sum(all_fish,axis=0)/N_it
        
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
        Cl_num = self.Cl_numerator(data)
        
        # Apply normalization
        Cl_out = np.matmul(self.inv_fish,Cl_num)
        
        return Cl_out
       
    ### IDEAL ESTIMATOR
    def Cl_numerator_ideal(self, data):
        """Compute the numerator of the idealized power spectrum estimator. We normalize by < mask^2 >.
        """
        # Normalize data by C_th and transform to harmonic space
        Cinv_data_lm = self.base.safe_divide(self.base.to_lm(data),self.base.Cl_lm)

        # Compute numerator
        Cl_num = 0.5*np.real(np.sum(self.m_weight*Cinv_data_lm*np.conj(Cinv_data_lm)*self.all_ell_bins*self.beam_lm**2,axis=1))/np.mean(self.mask**2)

        return Cl_num
                
    def compute_fisher_ideal(self):
        """This computes the idealized Fisher matrix for the power spectrum."""
        
        # Compute normalization
        fish_diag = 0.5*np.sum(self.base.safe_divide(self.m_weight,self.base.Cl_lm**2)*self.all_ell_bins*self.beam_lm**4,axis=1)
        fish = np.diag(fish_diag)
        self.fish_ideal = fish
        self.inv_fish_ideal = np.diag(1./fish_diag)
        
        return fish
    
    def Cl_ideal(self, data, fish_ideal=[]):
        """Compute the idealized power spectrum estimator, including normalization, if not supplied or already computed. Note that this normalizes by < mask^2 >."""
        
        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)
        
        if not hasattr(self,'inv_fish_ideal'):
            print("Computing ideal Fisher matrix")
            self.compute_fisher_ideal()
            
        # Compute numerator
        Cl_num_ideal = self.Cl_numerator_ideal(data)
        
        # Apply normalization
        Cl_out = np.matmul(self.inv_fish_ideal,Cl_num_ideal)
        
        return Cl_out