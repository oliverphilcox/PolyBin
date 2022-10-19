### Code for ideal and unwindowed binned polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022)
## This module contains the basic code

import healpy
import numpy as np
import multiprocessing as mp
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
import pywigxjpf as wig

class PolyBin():
    """Base class for PolyBin.
    
    Inputs:
    - Nside: HEALPix Nside
    - Cl: Fiducial power spectrum. This is used for creating synthetic maps for the optimal estimators, and, optionally, generating GRFs to test on."""
    def __init__(self, Nside, Cl):
        
        # Load attributes
        self.Nside = Nside
        self.Cl = Cl
        
        # Derived parameters
        self.Npix = 12*Nside**2
        self.A_pix = 4.*np.pi/self.Npix
        self.lmax = 3*self.Nside-1
        self.l_arr,self.m_arr = healpy.Alm.getlm(self.lmax)
        
        # Apply Cl to grid
        ls = np.arange(self.lmax+1)
        self.Cl_lm = InterpolatedUnivariateSpline(ls, self.Cl)(self.l_arr)
        for i in range(self.lmax+1):
            if self.Cl[i]==0:
                self.Cl_lm[self.l_arr==i] = 0.
                
        # Define 3j calculation
        wig.wig_table_init(self.lmax*2,9)
        wig.wig_temp_init(self.lmax*2)
        self.tj0 = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)

    # Basic HEALPix utilities
    def to_lm(self, input_map):
        """Convert from map-space to harmonic-space"""
        return healpy.map2alm(input_map,pol=False)

    def to_map(self, input_lm):
        """Convert from harmonic-space to map-space"""
        return healpy.alm2map(input_lm,self.Nside,pol=False)

    def to_lm_spin(self, input_map_plus, input_map_minus, spin):
        """Convert (+-s)A from map-space to harmonic-space, weighting by (+-s)Y_{lm}. Our convention definitions follow HEALPix.
        
        The inputs are [(+s)M(n), (-s)M(n)] and the outputs are [(+s)M_lm, (-s)M_lm]
        """
        assert spin>=1, "Spin must be positive!"
        assert type(spin)==int, "Spin must be an integer!"
        
        # Define inputs
        map_inputs = [np.real((input_map_plus+input_map_minus)/2.), np.real((input_map_plus-input_map_minus)/(2.0j))]
        
        # Perform transformation
        lm_outputs = healpy.map2alm_spin(map_inputs,spin,self.lmax)
        
        # Reconstruct output
        lm_plus = -(lm_outputs[0]+1.0j*lm_outputs[1])
        lm_minus = -1*(-1)**spin*(lm_outputs[0]-1.0j*lm_outputs[1])
        
        return lm_plus, lm_minus

    def to_map_spin(self, input_lm_plus, input_lm_minus, spin):
        """Convert (+-s)A_lm from harmonic-space to map-space, weighting by (+-s)Y_{lm}. Our convention definitions follow HEALPix.
        
        The inputs are [(+s)M_lm, (-s)M_lm] and the outputs are [(+s)M(n), (-s)M(n)]
        """
        assert spin>=1, "Spin must be positive!"
        assert type(spin)==int, "Spin must be an integer!"
        
        # Define inputs
        lm_inputs = [-(input_lm_plus+(-1)**spin*input_lm_minus)/2.,-(input_lm_plus-(-1)**spin*input_lm_minus)/(2.0j)]
        
        # Perform transformation
        map_outputs = healpy.alm2map_spin(lm_inputs,self.Nside,spin,self.lmax)
        
        # Reconstruct output
        map_plus = map_outputs[0]+1.0j*map_outputs[1]
        map_minus = map_outputs[0]-1.0j*map_outputs[1]
        
        return map_plus, map_minus

    def safe_divide(self, x, y):
        """Function to divide maps without zero errors."""
        out = np.zeros_like(x)
        out[y!=0] = x[y!=0]/y[y!=0]
        return out

    def generate_data(self, seed=None, Cl_input=[], b_input=None, add_B=False, remove_mean=True):
        """Generate a cmb map with a given C_ell and (optionally) b_l1l2l3. 

        We use the method of Smith & Zaldarriaga 2006, and assume that b_l1l2l3 is separable into three identical pieces.

        We optionally subtract off the mean of the map (numerically, but could be done analytically), since it is not guaranteed to be zero if we include a synthetic bispectrum.

        No mask is added at this stage."""
        
        # Define seed
        if seed!=None:
            np.random.seed(seed)
            
        # Define input power spectrum
        if len(Cl_input)==0:
            Cl_input = self.Cl

        # Generate a_lm
        initial_lm = healpy.synalm(Cl_input,self.lmax)

        if not add_B:
            return self.to_map(initial_lm)
        
        # Interpolate Cl to all l values
        ls = np.arange(self.lmax+1)
        Cl_input_lm = InterpolatedUnivariateSpline(ls, Cl_input)(self.l_arr)
        for i in range(self.lmax+1):
            if Cl_input[i]==0:
                Cl_input_lm[self.l_arr==i] = 0.
        
        # Compute gradient map
        Cinv_lm = self.safe_divide(initial_lm,Cl_input_lm)
        bCinv_map = self.to_map(b_input(self.l_arr)*Cinv_lm)
        grad_lm = 0.5*b_input(self.l_arr)*self.to_lm(bCinv_map*bCinv_map)

        # Apply correction to a_lm
        output_lm = initial_lm + 1./3.*grad_lm
        output_map = self.to_map(output_lm)

        # Optionally remove mean of map
        if remove_mean:
            
            # Compute mean of synthetic maps numerically
            if not hasattr(self, 'map_offset') or (hasattr(self,'map_offset') and self.saved_spec!=[Cl_input,b_input]):
                print("Computing offset for synthetic maps")
                map_offset = 0.
                for ii in range(100):
                    map_offset += self.generate_data(Cl_input=Cl_input, b_input=b_input, seed=int(1e6)+ii,add_B=True,remove_mean=False)/100.
                self.map_offset = map_offset
                self.saved_spec = [Cl_input, b_input]
            # Remove mean
            output_map -= self.map_offset
        
        return output_map
    
    def applyUinv(self, input_map):
        """Apply the exact inverse weighting U^{-1} to a map. This assumes a diagonal C_l weighting, as produced by generate_data."""
        
        # Transform to harmonic space
        input_map_lm = self.to_lm(input_map)
        
        # Divide by covariance and return to map-space
        output = self.to_map(self.safe_divide(input_map_lm,self.Cl_lm))
        
        return output
 