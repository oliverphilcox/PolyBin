### Code for ideal and unwindowed binned polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022)
## This module contains the parity-odd and parity-even trispectrum estimation code

import numpy as np
import multiprocessing as mp
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
import pywigxjpf as wig

class TSpec():
    """
    Trispectrum estimation class. This takes the binning strategy as input and a base class. 
    We also feed in a function that applies the S^-1 operator in real space.

    Note that we can additionally compute squeezed tetrahedra by setting l_bins_squeeze, giving a different lmax for short and long sides.
    We allow sides l2, l4 and L to have higher ell in this set-up (noting that l2 > l1, l4 > l3, and that {l1,l2,L} and {l3,l4,L} obey triangle conditions.)
    
    We can also specialize to collapsed tetrahedra, by restricting the range of L using L_bins.

    Inputs:
    - base: PolyBin class
    - mask: HEALPix mask to deconvolve
    - applySinv: function which returns S^-1 applied to a given input map
    - l_bins: array of bin edges
    - l_bins_squeeze: array of squeezed bin edges (optional)
    - L_bins: array of diagonal bin edges (optional)
    - include_partial_triangles: whether to include triangles (in l1,l2,L or l3,l4,L) whose centers don't satisfy the triangle conditions. (Default: False)
    """
    def __init__(self, base, mask, applySinv, l_bins, l_bins_squeeze=[], L_bins=[], include_partial_triangles=False):
        # Read in attributes
        self.base = base
        self.mask = mask
        self.applySinv = applySinv
        self.l_bins = l_bins
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

        if len(L_bins)>0:
            self.L_bins = L_bins
            self.NL = len(L_bins)-1
            for L_edge in self.L_bins:
                assert L_edge in self.l_bins_squeeze, "l-bins must contain all L-bins!"
        else:
            self.L_bins = self.l_bins.copy()
            self.NL = self.Nl

        self.beam = self.base.beam
        self.beam_lm = self.base.beam_lm
        self.include_partial_triangles = include_partial_triangles
        
        if np.max(self.l_bins_squeeze)>base.lmax:
            raise Exception("Maximum l is larger than HEALPix resolution!")
        print("Binning: %d bins in [%d, %d]"%(self.Nl,self.min_l,np.max(self.l_bins)))
        if self.Nl_squeeze!=self.Nl:
            print("Squeezed binning: %d bins in [%d, %d]"%(self.Nl_squeeze,self.min_l,np.max(self.l_bins_squeeze)))
        if self.NL!=self.Nl_squeeze:
            print("L binning: %d bins in [%d, %d]"%(self.NL,self.min_l,np.max(self.L_bins))) 
        
        # Define l filters
        self.ell_bins = [(self.base.l_arr>=self.l_bins_squeeze[bin1])&(self.base.l_arr<self.l_bins_squeeze[bin1+1]) for bin1 in range(self.Nl_squeeze)]
        self.phase_factor = (-1.)**self.base.l_arr

        # Define m weights (for complex conjugates)
        self.m_weight = (1.+1.*(self.base.m_arr>0.))

        # Define wigner calculator
        wig.wig_table_init(self.base.lmax*2,9)
        wig.wig_temp_init(self.base.lmax*2)

        # Define 3j with specified spins, and 6j
        self.threej = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,2*-1,2*-1,2*2)
        self.sixj = lambda l1,l2,l3,l4,l5,l6: wig.wig6jj(2*l1,2*l2,2*l3,2*l4,2*l5,2*l6)

    def _check_bin(self, bin1, bin2, bin3, even=False):
        """
        Return one if modes in the bin satisfy the triangle conditions, or zero else.

        If even=true, we enforce that the sum of the three ells must be even.

        This is used either for all triangles in the bin, or just the center of the bin.
        """
        if self.include_partial_triangles:
            good = 0
            for l1 in range(self.l_bins_squeeze[bin1],self.l_bins_squeeze[bin1+1]):
                for l2 in range(self.l_bins_squeeze[bin2],self.l_bins_squeeze[bin2+1]):
                    for l3 in range(self.l_bins_squeeze[bin3],self.l_bins_squeeze[bin3+1]):
                        # skip any odd bins
                        if ((-1)**(l1+l2+l3)==-1) and even: continue 
                        if l1>=abs(l1-l2) and l3<=l1+l2:
                            good = 1
                        if good==1: break
                    if good==1: break
                if good==1: break
            if good==1: return 1
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
    
    def _compute_even_symmetry_factor(self):
        """
        Compute symmetry factor giving the degeneracy of each bin, for the parity-even trispectrum.
        """
        sym_factor_even = []

        # iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>=b2 if b1=b3
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl_squeeze):
                for bin3 in range(bin1,self.Nl):
                    for bin4 in range(bin3,self.Nl_squeeze):
                        if bin1==bin3 and bin4<bin2: continue # note different condition to odd estimator!
                        
                        # Iterate over L bins
                        for binL in range(self.NL):
                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,binL,even=False): continue
                            if not self._check_bin(bin3,bin4,binL,even=False): continue

                            # compute symmetry factor
                            if bin1==bin2 and bin3==bin4 and bin1==bin3:
                                sym = 8
                            elif bin1==bin2 and bin3==bin4:
                                sym = 4
                            elif bin1==bin2:
                                sym = 2
                            elif bin3==bin4:
                                sym = 2
                            elif bin1==bin3 and bin2==bin4:
                                sym = 2
                            else:
                                sym = 1
                            sym_factor_even.append(sym)        
        
        self.sym_factor_even = np.asarray(sym_factor_even)

        # Count number of bins
        self.N_t_even = len(self.sym_factor_even)
        print("Using %d even-parity trispectrum bins"%self.N_t_even)
            
    def _compute_odd_symmetry_factor(self):
        """
        Compute symmetry factor giving the degeneracy of each bin, for the parity-odd trispectrum.
        """
        sym_factor_odd = []

        # iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>b2 if b1=b3
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl_squeeze):
                for bin3 in range(bin1,self.Nl):
                    for bin4 in range(bin3,self.Nl_squeeze):
                        if bin1==bin3 and bin4<=bin2: continue
                        
                        # Iterate over L bins
                        for binL in range(self.NL):
                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,binL,even=False): continue
                            if not self._check_bin(bin3,bin4,binL,even=False): continue

                            # compute symmetry factor
                            if bin1==bin2 and bin3==bin4 and bin1==bin3:
                                sym = 8
                            elif bin1==bin2 and bin3==bin4:
                                sym = 4
                            elif bin1==bin2:
                                sym = 2
                            elif bin3==bin4:
                                sym = 2
                            elif bin1==bin3 and bin2==bin4:
                                sym = 2
                            else:
                                sym = 1
                            sym_factor_odd.append(sym)        
        
        self.sym_factor_odd = np.asarray(sym_factor_odd)

        # Count number of bins
        self.N_t_odd = len(self.sym_factor_odd)
        print("Using %d odd-parity trispectrum bins"%self.N_t_odd)
    
    def get_ells(self):
        """
        Return an array with the central l1, l2, l3, l4, L values for each trispectrum bin. 
        We also give which parity trispectra each bin corresponds to.
        """

        # Define arrays
        l1s, l2s, l3s, l4s, Ls = [],[],[],[],[]
        parities  = []

        # Iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>b2 if b1=b3
        for bin1 in range(self.Nl):
            l1 = 0.5*(self.l_bins[bin1]+self.l_bins[bin1+1])
            for bin2 in range(bin1,self.Nl_squeeze):
                l2 = 0.5*(self.l_bins_squeeze[bin2]+self.l_bins_squeeze[bin2+1])
                for bin3 in range(bin1,self.Nl):
                    l3 =  0.5*(self.l_bins[bin3]+self.l_bins[bin3+1])
                    for bin4 in range(bin3,self.Nl_squeeze):
                        l4 = 0.5*(self.l_bins_squeeze[bin4]+self.l_bins_squeeze[bin4+1])
                        if bin1==bin3 and bin4<bin2: continue
                        
                        # Iterate over L bins
                        for binL in range(self.NL):
                            L = 0.5*(self.L_bins[binL]+self.L_bins[binL+1])

                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,binL,even=False): continue
                            if not self._check_bin(bin3,bin4,binL,even=False): continue

                            if bin1==bin3 and bin4==bin2:
                                parities.append('even')
                            else:
                                parities.append('both')

                            # Add to output array
                            l1s.append(l1)
                            l2s.append(l2)
                            l3s.append(l3)
                            l4s.append(l4)
                            Ls.append(L)
        return l1s,l2s,l3s,l4s,Ls,parities

    def _compute_H(self, h_lm):
        """
        Compute the H^+-(n) map given input field h_lm. This calls the to_map_spin routine with the correct inputs.
        """
        H_plus, H_minus = self.base.to_map_spin(h_lm,-1.0*h_lm,1)
        return [H_plus, -H_minus]

    def _compute_Alm(self, H_maps, bin1, bin2, H2_maps=[]):
        """
        Compute the A_{b1b2}[x,y](L,M) map given H^+-[x], H^+-[y] and bins. This calls the to_lm_spin routine with the correct inputs.
        
        Note, we can optionally use two different H fields here - this corresponds to allowing for x!=y.
        """

        if len(H2_maps)==0:
            H2_maps = H_maps
        
        A_plus = H_maps[bin1][0]*H2_maps[bin2][0]
        A_minus = H_maps[bin1][1]*H2_maps[bin2][1]
        
        A_plus_lm, A_minus_lm = self.base.to_lm_spin(A_plus, A_minus, 2)
        
        return A_minus_lm.conj()
    
    def _compute_t0_numerator(self, parity='even'):
        """
        Compute the zero-field contribution to the parity-odd or parity-even trispectrum. This is a sum over Monte Carlo simulations but does not involve data. 

        The output is *not* normalized by the symmetry factor.
       
        Note that this requires processed simulations, computed either from generate_sims() or load_sims().
        """

        # First check that simulations have been loaded
        if not hasattr(self, 'A_ab_lms'):
                    raise Exception("Need to generate or specify bias simulations!")
        
        # Even parity estimator
        if parity=='even' or parity=='both':
            
            # Define arrays
            self.t0_even_num = np.zeros(self.N_t_even)
                
            # Iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>=b2 if b1=b3
            index = 0
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl_squeeze):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl_squeeze):
                            if bin1==bin3 and bin4<bin2: continue
             
                            # Compute summands, summing over permutations and MC fields
                            summand_t0 = 0.
                            for ii in range(self.N_it):
                                summand_t0 += self.Abar_bb_lms[ii][bin2][bin1].conj()*self.A_aa_lms[ii][bin4][bin3] + self.A_bb_lms[ii][bin2][bin1].conj()*self.Abar_aa_lms[ii][bin4][bin3]
                                summand_t0 += self.Abar_ab_lms[ii][bin1][bin2].conj()*self.A_ab_lms[ii][bin3][bin4] + self.A_ab_lms[ii][bin1][bin2].conj()*self.Abar_ab_lms[ii][bin3][bin4]
                                summand_t0 += self.Abar_ab_lms[ii][bin1][bin2].conj()*self.A_ab_lms[ii][bin4][bin3] + self.A_ab_lms[ii][bin1][bin2].conj()*self.Abar_ab_lms[ii][bin4][bin3]
                                summand_t0 += self.Abar_ab_lms[ii][bin2][bin1].conj()*self.A_ab_lms[ii][bin3][bin4] + self.A_ab_lms[ii][bin2][bin1].conj()*self.Abar_ab_lms[ii][bin3][bin4]
                                summand_t0 += self.Abar_ab_lms[ii][bin2][bin1].conj()*self.A_ab_lms[ii][bin4][bin3] + self.A_ab_lms[ii][bin2][bin1].conj()*self.Abar_ab_lms[ii][bin4][bin3]
                                summand_t0 += self.Abar_aa_lms[ii][bin2][bin1].conj()*self.A_bb_lms[ii][bin4][bin3] + self.A_aa_lms[ii][bin2][bin1].conj()*self.Abar_bb_lms[ii][bin4][bin3]
                            summand_t0 = self.m_weight*np.real(summand_t0)/self.N_it
                            
                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue
                                
                                # Compute estimator numerator
                                self.t0_even_num[index] = 1./4.*np.sum(summand_t0*self.ell_bins[binL])
                                index += 1
            
        if parity=='odd' or parity=='both':

            # Define arrays 
            self.t0_odd_num = np.zeros(self.N_t_odd)

            # Iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>b2 if b1=b3
            index = 0
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl_squeeze):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl_squeeze):
                            if bin1==bin3 and bin4<=bin2: continue
             
                            # Compute summands, summing over permutations and MC fields
                            summand_t0 = 0.
                            for ii in range(self.N_it):
                                summand_t0 += self.Abar_bb_lms[ii][bin2][bin1].conj()*self.A_aa_lms[ii][bin4][bin3] - self.A_bb_lms[ii][bin2][bin1].conj()*self.Abar_aa_lms[ii][bin4][bin3]
                                summand_t0 += self.Abar_ab_lms[ii][bin1][bin2].conj()*self.A_ab_lms[ii][bin3][bin4] - self.A_ab_lms[ii][bin1][bin2].conj()*self.Abar_ab_lms[ii][bin3][bin4]
                                summand_t0 += self.Abar_ab_lms[ii][bin1][bin2].conj()*self.A_ab_lms[ii][bin4][bin3] - self.A_ab_lms[ii][bin1][bin2].conj()*self.Abar_ab_lms[ii][bin4][bin3]
                                summand_t0 += self.Abar_ab_lms[ii][bin2][bin1].conj()*self.A_ab_lms[ii][bin3][bin4] - self.A_ab_lms[ii][bin2][bin1].conj()*self.Abar_ab_lms[ii][bin3][bin4]
                                summand_t0 += self.Abar_ab_lms[ii][bin2][bin1].conj()*self.A_ab_lms[ii][bin4][bin3] - self.A_ab_lms[ii][bin2][bin1].conj()*self.Abar_ab_lms[ii][bin4][bin3]
                                summand_t0 += self.Abar_aa_lms[ii][bin2][bin1].conj()*self.A_bb_lms[ii][bin4][bin3] - self.A_aa_lms[ii][bin2][bin1].conj()*self.Abar_bb_lms[ii][bin4][bin3]
                            summand_t0 = self.m_weight*np.imag(summand_t0)/self.N_it
                            
                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue

                                # Compute estimator numerator
                                self.t0_odd_num[index] = -1./4.*np.sum(summand_t0*self.ell_bins[binL])
                                index += 1

    def _compute_A_map_pm2(self, H_maps, Hbar_maps, H_maps2=[], Hbar_maps2=[]):
        """
        Compute (+-2)A[y,z] maps for all possible bins, optionally for two different fields.

        This is used in computation of the Fisher matrix.
        """
        
        ## Cross-spectra
        if len(H_maps2)!=0 and len(Hbar_maps2)!=0:
            # Compute A[u,v](L,M) maps (note that these are evaluated for bin1<bin2 and bin1>bin2!)
            A_uv_lm = [[self._compute_Alm(H_maps,bin1,bin2,H_maps2) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)]
            Abar_uv_lm = [[self.phase_factor*self._compute_Alm(Hbar_maps,bin1,bin2,Hbar_maps2) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)]

            # Convert to map-space
            all_A_pm2 = [[[self.base.to_map_spin(A_uv_lm[bin3][bin4].conj()*self.ell_bins[binL],Abar_uv_lm[bin3][bin4].conj()*self.ell_bins[binL],2) for binL in range(self.Nl_squeeze)] for bin4 in range(self.Nl_squeeze)] for bin3 in range(self.Nl_squeeze)]

        ## Auto-spectra
        else:
            # Compute A[a,b](L,M) maps
            A_uu_lm = [[self._compute_Alm(H_maps,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            Abar_uu_lm = [[self.phase_factor*self._compute_Alm(Hbar_maps,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]

            # Convert to map-space
            all_A_pm2 = [[[self.base.to_map_spin(A_uu_lm[bin4][bin3].conj()*self.ell_bins[binL],Abar_uu_lm[bin4][bin3].conj()*self.ell_bins[binL],2) for binL in range(self.NL)] for bin4 in range(bin3,self.Nl_squeeze)] for bin3 in range(self.Nl_squeeze)]

        return all_A_pm2

    def load_sims(self, sims, verb=False):
        """
        Load in Monte Carlo simulations used in the two- and zero-field terms of the trispectrum estimator. 

        This should be a list of *pairs* of simulations, [[simA-1, simB-1], [simA-2, simB-2], ...]

        These can alternatively be generated with a fiducial spectrum using the generate_sims script.
        """

        self.N_it = len(sims)
        print("Using %d pairs of Monte Carlo simulations"%self.N_it)
        
        # Define lists
        self.A_aa_lms, self.Abar_aa_lms = [],[]
        self.A_bb_lms, self.Abar_bb_lms = [],[]
        self.A_ab_lms, self.Abar_ab_lms = [],[]
        self.H_a_maps, self.H_b_maps = [],[]
        self.Hbar_a_maps, self.Hbar_b_maps = [],[]

        for ii in range(self.N_it):
            if ii%5==0 and verb: print("Processing bias simulation %d of %d"%(ii+1,self.N_it))
            
            # Transform to Fourier space and normalize appropriately
            Wh_alpha_lm = self.base.to_lm(self.mask*self.applySinv(sims[ii][0]))
            Wh_beta_lm = self.base.to_lm(self.mask*self.applySinv(sims[ii][1]))
            
            # Compute H_alpha maps
            H_alpha_map = [self._compute_H(self.ell_bins[bin1]*self.beam_lm*Wh_alpha_lm) for bin1 in range(self.Nl_squeeze)]
            Hbar_alpha_map = [self._compute_H(self.phase_factor*self.ell_bins[bin1]*self.beam_lm*Wh_alpha_lm) for bin1 in range(self.Nl_squeeze)]
            H_beta_map = [self._compute_H(self.ell_bins[bin1]*self.beam_lm*Wh_beta_lm) for bin1 in range(self.Nl_squeeze)]
            Hbar_beta_map = [self._compute_H(self.phase_factor*self.ell_bins[bin1]*self.beam_lm*Wh_beta_lm) for bin1 in range(self.Nl_squeeze)]
            
            # Compute A[alpha,alpha](b1,b2) maps with bin2>=bin1
            A_aa_lm = [[self._compute_Alm(H_alpha_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            Abar_aa_lm = [[self.phase_factor*self._compute_Alm(Hbar_alpha_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            
            # Compute A[beta,beta](b1,b2) maps with bin2>=bin1
            A_bb_lm = [[self._compute_Alm(H_beta_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            Abar_bb_lm = [[self.phase_factor*self._compute_Alm(Hbar_beta_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            
            # Compute A[alpha,beta](b1,b2) maps for all possible bins
            A_ab_lm = [[self._compute_Alm(H_alpha_map,bin1,bin2,H_beta_map) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)]
            Abar_ab_lm = [[self.phase_factor*self._compute_Alm(Hbar_alpha_map,bin1,bin2,Hbar_beta_map) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)]

            # Add to arrays
            self.H_a_maps.append(H_alpha_map)
            self.H_b_maps.append(H_beta_map)
            self.Hbar_a_maps.append(Hbar_alpha_map)
            self.Hbar_b_maps.append(Hbar_beta_map)
            self.A_aa_lms.append(A_aa_lm)
            self.Abar_aa_lms.append(Abar_aa_lm)
            self.A_bb_lms.append(A_bb_lm)
            self.Abar_bb_lms.append(Abar_bb_lm)
            self.A_ab_lms.append(A_ab_lm)
            self.Abar_ab_lms.append(Abar_ab_lm) 
        
    def generate_sims(self, N_it, Cl_input=[], verb=False):
        """
        Generate Monte Carlo simulations used in the two- and zero-field terms of the trispectrum generator. 
        These are pure GRFs. By default, they are generated with the input survey mask.
        We create N_it such simulations and store the relevant transformations into memory.
        
        We can alternatively load custom simulations using the load_sims script.
        """

        self.N_it = N_it
        print("Using %d pairs of Monte Carlo simulations"%self.N_it)
        
        # Define input power spectrum (with noise)
        if len(Cl_input)==0:
            Cl_input = self.base.Cl
        
        # Define lists
        self.A_aa_lms, self.Abar_aa_lms = [],[]
        self.A_bb_lms, self.Abar_bb_lms = [],[]
        self.A_ab_lms, self.Abar_ab_lms = [],[]
        self.H_a_maps, self.H_b_maps = [],[]
        self.Hbar_a_maps, self.Hbar_b_maps = [],[]

        # Iterate over simulations
        for ii in range(self.N_it):
            if ii%5==0 and verb: print("Generating bias simulation %d of %d"%(ii+1,N_it))
            
            # Generate simulations (including mask)
            raw_alpha = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input)
            raw_beta = self.base.generate_data(int(2e5)+ii, Cl_input=Cl_input)

            # Transform to Fourier space and normalize appropriately
            Wh_alpha_lm = self.base.to_lm(self.mask*self.applySinv(raw_alpha*self.mask))
            Wh_beta_lm = self.base.to_lm(self.mask*self.applySinv(raw_beta*self.mask))
    
            # Compute H_alpha maps
            H_alpha_map = [self._compute_H(self.ell_bins[bin1]*self.beam_lm*Wh_alpha_lm) for bin1 in range(self.Nl_squeeze)]
            Hbar_alpha_map = [self._compute_H(self.phase_factor*self.ell_bins[bin1]*self.beam_lm*Wh_alpha_lm) for bin1 in range(self.Nl_squeeze)]
            H_beta_map = [self._compute_H(self.ell_bins[bin1]*self.beam_lm*Wh_beta_lm) for bin1 in range(self.Nl_squeeze)]
            Hbar_beta_map = [self._compute_H(self.phase_factor*self.ell_bins[bin1]*self.beam_lm*Wh_beta_lm) for bin1 in range(self.Nl_squeeze)]
            
            # Compute A[alpha,alpha](b1,b2) maps with bin2>=bin1
            A_aa_lm = [[self._compute_Alm(H_alpha_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            Abar_aa_lm = [[self.phase_factor*self._compute_Alm(Hbar_alpha_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            
            # Compute A[beta,beta](b1,b2) maps with bin2>=bin1
            A_bb_lm = [[self._compute_Alm(H_beta_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            Abar_bb_lm = [[self.phase_factor*self._compute_Alm(Hbar_beta_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
            
            # Compute A[alpha,beta](b1,b2) maps for all possible bins
            A_ab_lm = [[self._compute_Alm(H_alpha_map,bin1,bin2,H_beta_map) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)]
            Abar_ab_lm = [[self.phase_factor*self._compute_Alm(Hbar_alpha_map,bin1,bin2,Hbar_beta_map) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)]

            # Add to arrays
            self.H_a_maps.append(H_alpha_map)
            self.H_b_maps.append(H_beta_map)
            self.Hbar_a_maps.append(Hbar_alpha_map)
            self.Hbar_b_maps.append(Hbar_beta_map)
            self.A_aa_lms.append(A_aa_lm)
            self.Abar_aa_lms.append(Abar_aa_lm)
            self.A_bb_lms.append(A_bb_lm)
            self.Abar_bb_lms.append(Abar_bb_lm)
            self.A_ab_lms.append(A_ab_lm)
            self.Abar_ab_lms.append(Abar_ab_lm) 

    ### OPTIMAL ESTIMATOR
    def Tl_numerator(self, data, parity='even', include_disconnected_term=True, verb=False):
        """
        Compute the numerator of the unwindowed trispectrum estimator.

        The `parity' parameter can be 'even', 'odd' or 'both'. This specifies what parity trispectra to compute.

        Note that we return the imaginary part of the odd-parity trispectrum.

        We can also optionally switch off the disconnected terms.
        """
        
        # Check type
        if parity not in ['even','odd','both']:
            raise Exception("Parity parameter not set correctly!")
        
        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor_even') and parity!='odd':
            self._compute_even_symmetry_factor()
        if not hasattr(self, 'sym_factor_odd') and parity!='even':
            self._compute_odd_symmetry_factor()

        # Check if simulations have been supplied
        if not hasattr(self, 'A_ab_lms') and include_disconnected_term:
            raise Exception("Need to generate or specify bias simulations!")
        
        # Compute t0 term, if not already computed
        if not hasattr(self, 't0_even_num') and parity!='odd' and include_disconnected_term:
            if verb: print("Computing t0 term")
            self._compute_t0_numerator(parity='even')
        if not hasattr(self, 't0_odd_num') and parity!='even' and include_disconnected_term:
            if verb: print("Computing t0 term")
            self._compute_t0_numerator(parity='odd')
        
        # Normalize data by S^-1 and transform to harmonic space
        Wh_data_lm = self.base.to_lm(self.mask*self.applySinv(data))
        
        # Compute H and H-bar maps
        if verb: print("Computing H^+- maps")
        H_map = [self._compute_H(self.ell_bins[bin1]*self.beam_lm*Wh_data_lm) for bin1 in range(self.Nl_squeeze)]
        Hbar_map = [self._compute_H(self.phase_factor*self.ell_bins[bin1]*self.beam_lm*Wh_data_lm) for bin1 in range(self.Nl_squeeze)]

        # Define array of A maps (restricting to bin2 <= bin1, by symmetry)
        if verb: print("Computing A maps")
        Alm = [[self._compute_Alm(H_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
        Abar_lm = [[self.phase_factor*self._compute_Alm(Hbar_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
        
        # Compute cross-spectra of MC simulations and data
        if include_disconnected_term:
            if verb: print("Computing A maps for cross-spectra")
            
            # Compute all A[alpha, d] and A[beta, d] for all bins
            A_ad_lms = [[[self._compute_Alm(self.H_a_maps[ii],bin1,bin2,H_map) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)] for ii in range(self.N_it)]
            Abar_ad_lms = [[[self.phase_factor*self._compute_Alm(self.Hbar_a_maps[ii],bin1,bin2,Hbar_map) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)] for ii in range(self.N_it)]
            A_bd_lms = [[[self._compute_Alm(self.H_b_maps[ii],bin1,bin2,H_map) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)] for ii in range(self.N_it)]
            Abar_bd_lms = [[[self.phase_factor*self._compute_Alm(self.Hbar_b_maps[ii],bin1,bin2,Hbar_map) for bin2 in range(self.Nl_squeeze)] for bin1 in range(self.Nl_squeeze)] for ii in range(self.N_it)]
    
        # Even parity estimator
        if parity=='even' or parity=='both':
            
            # Define arrays
            t4_even_num = np.zeros(self.N_t_even)
            if not include_disconnected_term:
                print("No subtraction of even-parity disconnected terms performed!")
            else:
                t2_even_num = np.zeros(self.N_t_even)
               
            # Iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>=b2 if b1=b3
            if verb: print("Assembling parity-even trispectrum numerator")
            index = 0
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl_squeeze):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl_squeeze):
                            if bin1==bin3 and bin4<bin2: continue
             
                            # Compute summands
                            summand_t4 = self.m_weight*np.real(Abar_lm[bin2][bin1].conj()*Alm[bin4][bin3] + Alm[bin2][bin1].conj()*Abar_lm[bin4][bin3])
                
                            # Compute 2-field term summand
                            if include_disconnected_term: 
                                # Sum over 6 permutations and all MC fields
                                summand_t2 = 0.
                                for ii in range(self.N_it):
                                    # first set of fields
                                    summand_t2 += Abar_lm[bin2][bin1].conj()*self.A_aa_lms[ii][bin4][bin3] + Alm[bin2][bin1].conj()*self.Abar_aa_lms[ii][bin4][bin3]
                                    summand_t2 += Abar_ad_lms[ii][bin1][bin2].conj()*A_ad_lms[ii][bin3][bin4] + A_ad_lms[ii][bin1][bin2].conj()*Abar_ad_lms[ii][bin3][bin4]
                                    summand_t2 += Abar_ad_lms[ii][bin1][bin2].conj()*A_ad_lms[ii][bin4][bin3] + A_ad_lms[ii][bin1][bin2].conj()*Abar_ad_lms[ii][bin4][bin3]
                                    summand_t2 += Abar_ad_lms[ii][bin2][bin1].conj()*A_ad_lms[ii][bin3][bin4] + A_ad_lms[ii][bin2][bin1].conj()*Abar_ad_lms[ii][bin3][bin4]
                                    summand_t2 += Abar_ad_lms[ii][bin2][bin1].conj()*A_ad_lms[ii][bin4][bin3] + A_ad_lms[ii][bin2][bin1].conj()*Abar_ad_lms[ii][bin4][bin3]
                                    summand_t2 += self.Abar_aa_lms[ii][bin2][bin1].conj()*Alm[bin4][bin3] + self.A_aa_lms[ii][bin2][bin1].conj()*Abar_lm[bin4][bin3]
                                    # second set of fields
                                    summand_t2 += Abar_lm[bin2][bin1].conj()*self.A_bb_lms[ii][bin4][bin3] + Alm[bin2][bin1].conj()*self.Abar_bb_lms[ii][bin4][bin3]
                                    summand_t2 += Abar_bd_lms[ii][bin1][bin2].conj()*A_bd_lms[ii][bin3][bin4] + A_bd_lms[ii][bin1][bin2].conj()*Abar_bd_lms[ii][bin3][bin4]
                                    summand_t2 += Abar_bd_lms[ii][bin1][bin2].conj()*A_bd_lms[ii][bin4][bin3] + A_bd_lms[ii][bin1][bin2].conj()*Abar_bd_lms[ii][bin4][bin3]
                                    summand_t2 += Abar_bd_lms[ii][bin2][bin1].conj()*A_bd_lms[ii][bin3][bin4] + A_bd_lms[ii][bin2][bin1].conj()*Abar_bd_lms[ii][bin3][bin4]
                                    summand_t2 += Abar_bd_lms[ii][bin2][bin1].conj()*A_bd_lms[ii][bin4][bin3] + A_bd_lms[ii][bin2][bin1].conj()*Abar_bd_lms[ii][bin4][bin3]
                                    summand_t2 += self.Abar_bb_lms[ii][bin2][bin1].conj()*Alm[bin4][bin3] + self.A_bb_lms[ii][bin2][bin1].conj()*Abar_lm[bin4][bin3]
                                summand_t2 = self.m_weight*np.real(summand_t2)/self.N_it/2.

                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue

                                # Compute estimator numerator
                                t4_even_num[index] = 1./2.*np.sum(summand_t4*self.ell_bins[binL])
                                if include_disconnected_term:
                                    t2_even_num[index] = -1./2.*np.sum(summand_t2*self.ell_bins[binL])

                                index += 1

            if include_disconnected_term:
                t_even_num = (t4_even_num+t2_even_num+self.t0_even_num)/self.sym_factor_even
            else:
                t_even_num = t4_even_num/self.sym_factor_even

        # Odd parity estimator
        if parity=='odd' or parity=='both':
            
            # Define arrays
            t4_odd_num = np.zeros(self.N_t_odd)
            if not include_disconnected_term:
                print("No subtraction of odd-parity disconnected terms performed!")
            else:
                t2_odd_num = np.zeros(self.N_t_odd)
                        
            # Iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>b2 if b1=b3
            if verb: print("Assembling parity-odd trispectrum numerator")
            index = 0
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl_squeeze):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl_squeeze):
                            if bin1==bin3 and bin4<=bin2: continue
                            
                            # Compute 4-field term summand
                            summand_t4 = self.m_weight*np.imag(Abar_lm[bin2][bin1].conj()*Alm[bin4][bin3] - Alm[bin2][bin1].conj()*Abar_lm[bin4][bin3])

                            # Compute 2-field term summand
                            if include_disconnected_term: 
                                # Sum over 6 permutations and all MC fields
                                summand_t2 = 0.
                                for ii in range(self.N_it):
                                    # first set of fields
                                    summand_t2 += Abar_lm[bin2][bin1].conj()*self.A_aa_lms[ii][bin4][bin3] - Alm[bin2][bin1].conj()*self.Abar_aa_lms[ii][bin4][bin3]
                                    summand_t2 += Abar_ad_lms[ii][bin1][bin2].conj()*A_ad_lms[ii][bin3][bin4] - A_ad_lms[ii][bin1][bin2].conj()*Abar_ad_lms[ii][bin3][bin4]
                                    summand_t2 += Abar_ad_lms[ii][bin1][bin2].conj()*A_ad_lms[ii][bin4][bin3] - A_ad_lms[ii][bin1][bin2].conj()*Abar_ad_lms[ii][bin4][bin3]
                                    summand_t2 += Abar_ad_lms[ii][bin2][bin1].conj()*A_ad_lms[ii][bin3][bin4] - A_ad_lms[ii][bin2][bin1].conj()*Abar_ad_lms[ii][bin3][bin4]
                                    summand_t2 += Abar_ad_lms[ii][bin2][bin1].conj()*A_ad_lms[ii][bin4][bin3] - A_ad_lms[ii][bin2][bin1].conj()*Abar_ad_lms[ii][bin4][bin3]
                                    summand_t2 += self.Abar_aa_lms[ii][bin2][bin1].conj()*Alm[bin4][bin3] - self.A_aa_lms[ii][bin2][bin1].conj()*Abar_lm[bin4][bin3]
                                    # second set of fields
                                    summand_t2 += Abar_lm[bin2][bin1].conj()*self.A_bb_lms[ii][bin4][bin3] - Alm[bin2][bin1].conj()*self.Abar_bb_lms[ii][bin4][bin3]
                                    summand_t2 += Abar_bd_lms[ii][bin1][bin2].conj()*A_bd_lms[ii][bin3][bin4] - A_bd_lms[ii][bin1][bin2].conj()*Abar_bd_lms[ii][bin3][bin4]
                                    summand_t2 += Abar_bd_lms[ii][bin1][bin2].conj()*A_bd_lms[ii][bin4][bin3] - A_bd_lms[ii][bin1][bin2].conj()*Abar_bd_lms[ii][bin4][bin3]
                                    summand_t2 += Abar_bd_lms[ii][bin2][bin1].conj()*A_bd_lms[ii][bin3][bin4] - A_bd_lms[ii][bin2][bin1].conj()*Abar_bd_lms[ii][bin3][bin4]
                                    summand_t2 += Abar_bd_lms[ii][bin2][bin1].conj()*A_bd_lms[ii][bin4][bin3] - A_bd_lms[ii][bin2][bin1].conj()*Abar_bd_lms[ii][bin4][bin3]
                                    summand_t2 += self.Abar_bb_lms[ii][bin2][bin1].conj()*Alm[bin4][bin3] - self.A_bb_lms[ii][bin2][bin1].conj()*Abar_lm[bin4][bin3]
                                summand_t2 = self.m_weight*np.imag(summand_t2)/self.N_it/2.

                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue
                                
                                # Compute estimator numerator
                                t4_odd_num[index]=-1./2.*np.sum(summand_t4*self.ell_bins[binL])

                                if include_disconnected_term:
                                    t2_odd_num[index] = 1./2.*np.sum(summand_t2*self.ell_bins[binL])

                                index += 1

            if include_disconnected_term:
                t_odd_num = (t4_odd_num+t2_odd_num+self.t0_odd_num)/self.sym_factor_odd
            else:
                t_odd_num = t4_odd_num/self.sym_factor_odd

        if parity=='even':
            return t_even_num
        elif parity=='odd':
            return t_odd_num
        else:
            return t_even_num, t_odd_num
    
    def compute_fisher_contribution(self, seed, parity='even', verb=False):
        """
        This computes the contribution to the Fisher matrix from a single GRF simulation, created internally.
        
        The `parity' parameter can be 'even', 'odd' or 'both'. This specifies what parity trispectra to compute.
        """
        
        # Check type
        if parity not in ['even','odd','both']:
            raise Exception("Parity parameter not set correctly!")
        
        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor_even') and parity!='odd':
            self._compute_even_symmetry_factor()
        if not hasattr(self, 'sym_factor_odd') and parity!='even':
            self._compute_odd_symmetry_factor()
        
        # Initialize output
        if parity=='even':
            fish_even = np.zeros((self.N_t_even,self.N_t_even))
        if parity=='odd':
            fish_odd = np.zeros((self.N_t_odd,self.N_t_odd))
        if parity=='both':
            fish_both = np.zeros((self.N_t_even+self.N_t_odd,self.N_t_even+self.N_t_odd))
        
        # Compute two random realizations with known power spectrum
        if verb: print("\nGenerating data")
        u1 = self.base.generate_data(seed=seed+int(1e7))
        u2 = self.base.generate_data(seed=seed+int(2e7))

        def compute_all_Q(weighting='U'):
            """
            Compute all possible W Q[x,y,z] maps for a given simulation pair (u1, u2). This applies some weighting: either U^-1 or S^-1.

            For U^-1, this returns S^-1 W Q[x,y,z], as required below.

            The 'parity' parameter can be 'even', 'odd' or 'both'.
            """

            # Compute weighted fields
            if weighting=='U':
                Xinv_u1, Xinv_u2 = self.base.applyUinv(u1), self.base.applyUinv(u2)
            elif weighting=='S':
                Xinv_u1, Xinv_u2 = self.applySinv(u1), self.applySinv(u2)
            WXinv_u1_lm, WXinv_u2_lm = self.base.to_lm(self.mask*Xinv_u1), self.base.to_lm(self.mask*Xinv_u2)
            
            # Define H fields
            if verb: print("\nCreating H maps for %s-inverse-weighted fields"%weighting)
            H_1 = [self._compute_H(WXinv_u1_lm*self.ell_bins[bin1]*self.beam_lm) for bin1 in range(self.Nl_squeeze)]
            H_2 = [self._compute_H(WXinv_u2_lm*self.ell_bins[bin1]*self.beam_lm) for bin1 in range(self.Nl_squeeze)]
            Hbar_1 = [self._compute_H(self.phase_factor*WXinv_u1_lm*self.ell_bins[bin1]*self.beam_lm) for bin1 in range(self.Nl_squeeze)]
            Hbar_2 = [self._compute_H(self.phase_factor*WXinv_u2_lm*self.ell_bins[bin1]*self.beam_lm) for bin1 in range(self.Nl_squeeze)]
            
            # Compute A maps
            if verb: print("Computing A[u1,u2] maps for %s-inverse-weighted fields"%weighting)
            A_11_pm2 = self._compute_A_map_pm2(H_1,Hbar_1)
            A_12_pm2 = self._compute_A_map_pm2(H_1,Hbar_1,H_2,Hbar_2)
            A_22_pm2 = self._compute_A_map_pm2(H_2,Hbar_2)

            def compute_Qraw(bin2,bin3,bin4,binL,fields='111'):
                """
                Compute raw Q[x,y,z] maps (without Theta_l).
                Note that this includes the (x<->y) and (x<->z) symmetries.

                The 'fields' parameter gives the relevant symmetries.

                This returns an array containing both even- and odd-parity maps.
                
                If using the U^-1 weighting, this also premultiplies by S^-1.
                """

                if fields=='111':
                    # NB: 3x symmetry here!
                    mapMinus = 3*H_1[bin2][0]*A_11_pm2[bin3][bin4-bin3][binL][1]
                    mapPlus =  3*H_1[bin2][1]*A_11_pm2[bin3][bin4-bin3][binL][0]

                elif fields=='222':
                    # NB: 3x symmetry here!
                    mapMinus = 3*H_2[bin2][0]*A_22_pm2[bin3][bin4-bin3][binL][1]
                    mapPlus =  3*H_2[bin2][1]*A_22_pm2[bin3][bin4-bin3][binL][0]

                elif fields=='112':
                    # x <-> y symmetry here!
                    binMax, binMin = max(bin3,bin4), min(bin3,bin4)
                    mapMinus  = 2*H_1[bin2][0]*A_12_pm2[bin3][bin4][binL][1]
                    mapMinus += H_2[bin2][0]*A_11_pm2[binMin][binMax-binMin][binL][1]
                    mapPlus  =  2*H_1[bin2][1]*A_12_pm2[bin3][bin4][binL][0]
                    mapPlus +=  H_2[bin2][1]*A_11_pm2[binMin][binMax-binMin][binL][0]

                elif fields=='122':
                    # No symmetry here!
                    binMax, binMin = max(bin3,bin4), min(bin3,bin4)
                    mapMinus  = H_1[bin2][0]*A_22_pm2[binMin][binMax-binMin][binL][1]
                    mapMinus += H_2[bin2][0]*A_12_pm2[bin3][bin4][binL][1]
                    mapMinus += H_2[bin2][0]*A_12_pm2[bin4][bin3][binL][1]
                    mapPlus  =  H_1[bin2][1]*A_22_pm2[binMin][binMax-binMin][binL][0]
                    mapPlus +=  H_2[bin2][1]*A_12_pm2[bin3][bin4][binL][0]
                    mapPlus +=  H_2[bin2][1]*A_12_pm2[bin4][bin3][binL][0]
                else:
                    raise Exception("Incorrect field type")

                tmp_lm = self.base.to_lm_spin(-mapPlus, mapMinus, 1)

                return np.asarray([-tmp_lm[1]+tmp_lm[0], tmp_lm[1]+tmp_lm[0]])
            
            # Compute raw Q[x,y,z] maps
            if verb: print("Computing raw Q maps for %s-inverse-weighted fields"%weighting)
            Qraw_111 = [[[[compute_Qraw(bin2,bin3,bin4,binL,'111') for binL in range(self.NL)] for bin4 in range(bin3,self.Nl_squeeze)] for bin3 in range(self.Nl_squeeze)] for bin2 in range(self.Nl_squeeze)]
            Qraw_112 = [[[[compute_Qraw(bin2,bin3,bin4,binL,'112') for binL in range(self.NL)] for bin4 in range(self.Nl_squeeze)] for bin3 in range(self.Nl_squeeze)] for bin2 in range(self.Nl_squeeze)]
            Qraw_122 = [[[[compute_Qraw(bin2,bin3,bin4,binL,'122') for binL in range(self.NL)] for bin4 in range(self.Nl_squeeze)] for bin3 in range(self.Nl_squeeze)] for bin2 in range(self.Nl_squeeze)]
            Qraw_222 = [[[[compute_Qraw(bin2,bin3,bin4,binL,'222') for binL in range(self.NL)] for bin4 in range(bin3,self.Nl_squeeze)] for bin3 in range(self.Nl_squeeze)] for bin2 in range(self.Nl_squeeze)]

            # Compute all Q filters
            if verb: print("Assembling Q filters for %s-inverse-weighted fields"%weighting)
            WQ_111_maps = []
            WQ_112_maps = []
            WQ_122_maps = []
            WQ_222_maps = []
            index_odd = 0
            index_even = 0
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl_squeeze):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl_squeeze):
                            # always include all bins here for counting!
                            if bin1==bin3 and bin4<bin2: continue
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue
                                
                                # Compute Q[1,1,1] map, including symmetries (noting that 2 alphas are equivalent)
                                Q111_lm =  2*self.ell_bins[bin1]*self.beam_lm*Qraw_111[bin2][bin3][bin4-bin3][binL]
                                Q111_lm += 2*self.ell_bins[bin2]*self.beam_lm*Qraw_111[bin1][bin3][bin4-bin3][binL]
                                Q111_lm += 2*self.ell_bins[bin3]*self.beam_lm*Qraw_111[bin4][bin1][bin2-bin1][binL]
                                Q111_lm += 2*self.ell_bins[bin4]*self.beam_lm*Qraw_111[bin3][bin1][bin2-bin1][binL]
                                
                                # Compute Q[1,1,2] map
                                Q112_lm =  self.ell_bins[bin1]*self.beam_lm*Qraw_112[bin2][bin3][bin4][binL]
                                Q112_lm += self.ell_bins[bin2]*self.beam_lm*Qraw_112[bin1][bin3][bin4][binL]
                                Q112_lm += self.ell_bins[bin1]*self.beam_lm*Qraw_112[bin2][bin4][bin3][binL]
                                Q112_lm += self.ell_bins[bin2]*self.beam_lm*Qraw_112[bin1][bin4][bin3][binL]
                                Q112_lm += self.ell_bins[bin3]*self.beam_lm*Qraw_112[bin4][bin1][bin2][binL]
                                Q112_lm += self.ell_bins[bin4]*self.beam_lm*Qraw_112[bin3][bin1][bin2][binL]
                                Q112_lm += self.ell_bins[bin3]*self.beam_lm*Qraw_112[bin4][bin2][bin1][binL]
                                Q112_lm += self.ell_bins[bin4]*self.beam_lm*Qraw_112[bin3][bin2][bin1][binL]
                                
                                # Compute Q[1,2,2] map
                                Q122_lm =  self.ell_bins[bin1]*self.beam_lm*Qraw_122[bin2][bin3][bin4][binL]
                                Q122_lm += self.ell_bins[bin2]*self.beam_lm*Qraw_122[bin1][bin3][bin4][binL]
                                Q122_lm += self.ell_bins[bin1]*self.beam_lm*Qraw_122[bin2][bin4][bin3][binL]
                                Q122_lm += self.ell_bins[bin2]*self.beam_lm*Qraw_122[bin1][bin4][bin3][binL]
                                Q122_lm += self.ell_bins[bin3]*self.beam_lm*Qraw_122[bin4][bin1][bin2][binL]
                                Q122_lm += self.ell_bins[bin4]*self.beam_lm*Qraw_122[bin3][bin1][bin2][binL]
                                Q122_lm += self.ell_bins[bin3]*self.beam_lm*Qraw_122[bin4][bin2][bin1][binL]
                                Q122_lm += self.ell_bins[bin4]*self.beam_lm*Qraw_122[bin3][bin2][bin1][binL]
                                
                                # Compute Q[2,2,2] map
                                Q222_lm =  2*self.ell_bins[bin1]*self.beam_lm*Qraw_222[bin2][bin3][bin4-bin3][binL]
                                Q222_lm += 2*self.ell_bins[bin2]*self.beam_lm*Qraw_222[bin1][bin3][bin4-bin3][binL]
                                Q222_lm += 2*self.ell_bins[bin3]*self.beam_lm*Qraw_222[bin4][bin1][bin2-bin1][binL]
                                Q222_lm += 2*self.ell_bins[bin4]*self.beam_lm*Qraw_222[bin3][bin1][bin2-bin1][binL]
                                
                                # Apply mask and convert to real-space
                                WQ_111_map, WQ_112_map, WQ_122_map, WQ_222_map = [],[],[],[]

                                if weighting=='U':
                                    # Additionally premultiply by S^-1 here
                                    if parity!='odd':
                                        WQ_111_map.append(self.applySinv(self.mask*self.base.to_map(Q111_lm[0])/(2.*self.sym_factor_even[index_even])))
                                        WQ_112_map.append(self.applySinv(self.mask*self.base.to_map(Q112_lm[0])/(2.*self.sym_factor_even[index_even])))
                                        WQ_122_map.append(self.applySinv(self.mask*self.base.to_map(Q122_lm[0])/(2.*self.sym_factor_even[index_even])))
                                        WQ_222_map.append(self.applySinv(self.mask*self.base.to_map(Q222_lm[0])/(2.*self.sym_factor_even[index_even])))
                                        index_even += 1
                                    
                                    if parity!='even' and ((bin1==bin3)*(bin2==bin4))!=1:
                                        # Note the extra i factors here, as Q is an imaginary map
                                        # We absorb one additional i factor into a -1 multiplying the whole matrix, which is then removed by the two imaginary parts
                                        WQ_111_map.append(self.applySinv(self.mask*self.base.to_map(-1.0j*Q111_lm[1])/(2.*self.sym_factor_odd[index_odd])))
                                        WQ_112_map.append(self.applySinv(self.mask*self.base.to_map(-1.0j*Q112_lm[1])/(2.*self.sym_factor_odd[index_odd])))
                                        WQ_122_map.append(self.applySinv(self.mask*self.base.to_map(-1.0j*Q122_lm[1])/(2.*self.sym_factor_odd[index_odd])))
                                        WQ_222_map.append(self.applySinv(self.mask*self.base.to_map(-1.0j*Q222_lm[1])/(2.*self.sym_factor_odd[index_odd])))
                                        index_odd += 1

                                else:
                                    # No premultiplication needed!
                                    if parity!='odd':
                                        WQ_111_map.append(self.mask*self.base.to_map(Q111_lm[0])/(2.*self.sym_factor_even[index_even]))
                                        WQ_112_map.append(self.mask*self.base.to_map(Q112_lm[0])/(2.*self.sym_factor_even[index_even]))
                                        WQ_122_map.append(self.mask*self.base.to_map(Q122_lm[0])/(2.*self.sym_factor_even[index_even]))
                                        WQ_222_map.append(self.mask*self.base.to_map(Q222_lm[0])/(2.*self.sym_factor_even[index_even]))
                                        index_even += 1
                                    
                                    if parity!='even' and ((bin1==bin3)*(bin2==bin4))!=1:
                                        # Note the extra i factors here, as Q is an imaginary map
                                        # We absorb one additional i factor into a -1 multiplying the whole matrix, which is then removed by the two imaginary parts
                                        WQ_111_map.append(self.mask*self.base.to_map(-1.0j*Q111_lm[1])/(2.*self.sym_factor_odd[index_odd]))
                                        WQ_112_map.append(self.mask*self.base.to_map(-1.0j*Q112_lm[1])/(2.*self.sym_factor_odd[index_odd]))
                                        WQ_122_map.append(self.mask*self.base.to_map(-1.0j*Q122_lm[1])/(2.*self.sym_factor_odd[index_odd]))
                                        WQ_222_map.append(self.mask*self.base.to_map(-1.0j*Q222_lm[1])/(2.*self.sym_factor_odd[index_odd]))
                                        index_odd += 1
                                    
                                # Add to output arrays
                                WQ_111_maps.append(WQ_111_map)
                                WQ_112_maps.append(WQ_112_map)
                                WQ_122_maps.append(WQ_122_map)
                                WQ_222_maps.append(WQ_222_map)
                                
            return WQ_111_maps, WQ_112_maps, WQ_122_maps, WQ_222_maps

        # First consider S-weighted leg
        WQ_111_maps_S, WQ_112_maps_S, WQ_122_maps_S, WQ_222_maps_S = compute_all_Q('S')
        
        # Next consider U-weighted leg
        SiWQ_111_maps_U, SiWQ_112_maps_U, SiWQ_122_maps_U, SiWQ_222_maps_U = compute_all_Q('U')

        # Now compute Fisher matrix contribution
        if verb: print("\nComputing Fisher matrix contribution")
        index1o = 0
        index1e = 0
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl_squeeze):
                for bin3 in range(bin1,self.Nl):
                    for bin4 in range(bin3,self.Nl_squeeze):
                        if bin1==bin3 and bin4<bin2: continue
                        for binL in range(self.NL):
                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,binL,even=False): continue
                            if not self._check_bin(bin3,bin4,binL,even=False): continue

                            # Skip bin
                            if parity=='odd' and bin1==bin3 and bin2==bin4: 
                                index1e += 1
                                continue
                            
                            if parity!='odd':
                                if (index1e)%5==0 and verb: print("On bin %d of %d"%(index1e+1,self.N_t_even))
                            else:
                                if (index1o)%5==0 and verb: print("On bin %d of %d"%(index1o+1,self.N_t_odd))
                            
                            # Define relevant S^-1 W Q[U] maps
                            WQ_111_S = WQ_111_maps_S[index1e]
                            WQ_112_S = WQ_112_maps_S[index1e]
                            WQ_122_S = WQ_122_maps_S[index1e]
                            WQ_222_S = WQ_222_maps_S[index1e]

                            # Iterate over second set of bins
                            index2o, index2e = 0, 0
                            for bin1p in range(self.Nl):
                                for bin2p in range(bin1p,self.Nl_squeeze):
                                    for bin3p in range(bin1p,self.Nl):
                                        for bin4p in range(bin3p,self.Nl_squeeze):
                                            if bin1p==bin3p and bin4p<bin2p: continue
                                            for binLp in range(self.NL):
                                                # skip bins outside the triangle conditions
                                                if not self._check_bin(bin1p,bin2p,binLp,even=False): continue
                                                if not self._check_bin(bin3p,bin4p,binLp,even=False): continue

                                                # Skip bin
                                                if parity=='odd' and bin1p==bin3p and bin2p==bin4p:
                                                    index2e += 1
                                                    continue
                                                
                                                ## Compute Fisher matrix contributions
                                                
                                                # Even-Even
                                                if parity!='odd':
                                                    fish_summand_ee  =     (WQ_111_S[0]*SiWQ_111_maps_U[index2e][0]+WQ_222_S[0]*SiWQ_222_maps_U[index2e][0])
                                                    fish_summand_ee +=  9.*(WQ_112_S[0]*SiWQ_112_maps_U[index2e][0]+WQ_122_S[0]*SiWQ_122_maps_U[index2e][0])
                                                    fish_summand_ee += -6.*(WQ_111_S[0]*SiWQ_122_maps_U[index2e][0]+WQ_222_S[0]*SiWQ_112_maps_U[index2e][0])
                                                    
                                                # Odd-Odd
                                                if parity!='even' and ((bin1==bin3)*(bin2==bin4))!=1 and ((bin1p==bin3p)*(bin2p==bin4p))!=1:
                                                    fish_summand_oo  =     (WQ_111_S[-1]*SiWQ_111_maps_U[index2e][-1]+WQ_222_S[-1]*SiWQ_222_maps_U[index2e][-1]) # note we always index with even indices here
                                                    fish_summand_oo +=  9.*(WQ_112_S[-1]*SiWQ_112_maps_U[index2e][-1]+WQ_122_S[-1]*SiWQ_122_maps_U[index2e][-1])
                                                    fish_summand_oo += -6.*(WQ_111_S[-1]*SiWQ_122_maps_U[index2e][-1]+WQ_222_S[-1]*SiWQ_112_maps_U[index2e][-1])
                                                    
                                                # Even-Odd 
                                                if parity=='both' and ((bin1p==bin3p)*(bin2p==bin4p))!=1:
                                                    fish_summand_eo =      (WQ_111_S[0]*SiWQ_111_maps_U[index2e][1]+WQ_222_S[0]*SiWQ_222_maps_U[index2e][1])
                                                    fish_summand_eo +=  9.*(WQ_112_S[0]*SiWQ_112_maps_U[index2e][1]+WQ_122_S[0]*SiWQ_122_maps_U[index2e][1])
                                                    fish_summand_eo += -6.*(WQ_111_S[0]*SiWQ_122_maps_U[index2e][1]+WQ_222_S[0]*SiWQ_112_maps_U[index2e][1])
                                                    
                                                # Odd-Even
                                                if parity=='both' and ((bin1==bin3)*(bin2==bin4))!=1:
                                                    fish_summand_oe =      (WQ_111_S[1]*SiWQ_111_maps_U[index2e][0]+WQ_222_S[1]*SiWQ_222_maps_U[index2e][0])
                                                    fish_summand_oe +=  9.*(WQ_112_S[1]*SiWQ_112_maps_U[index2e][0]+WQ_122_S[1]*SiWQ_122_maps_U[index2e][0])
                                                    fish_summand_oe += -6.*(WQ_111_S[1]*SiWQ_122_maps_U[index2e][0]+WQ_222_S[1]*SiWQ_112_maps_U[index2e][0])
                                                    
                                                # Assemble relevant matrices
                                                if parity=='even':
                                                    fish_even[index1e, index2e] = 1./144.*self.base.A_pix*np.sum(fish_summand_ee/8.)
                                                
                                                if parity=='odd' and ((bin1==bin3)*(bin2==bin4))!=1 and ((bin1p==bin3p)*(bin2p==bin4p))!=1:
                                                    # note negative sign since Q was originally imaginary
                                                    fish_odd[index1o, index2o] = 1./144.*self.base.A_pix*np.sum(fish_summand_oo/8.)
                                                
                                                if parity=='both':
                                                    fish_both[index1e, index2e] = 1./144.*self.base.A_pix*np.sum(fish_summand_ee/8.)
                                                    if ((bin1==bin3)*(bin2==bin4))!=1 and ((bin1p==bin3p)*(bin2p==bin4p))!=1:
                                                        fish_both[self.N_t_even+index1o, self.N_t_even+index2o] =  1./144.*self.base.A_pix*np.sum(fish_summand_oo/8.)
                                                    if ((bin1==bin3)*(bin2==bin4))!=1:
                                                        fish_both[self.N_t_even+index1o, index2e] = 1./144.*self.base.A_pix*np.sum(fish_summand_oe/8.)
                                                    if ((bin1p==bin3p)*(bin2p==bin4p))!=1:
                                                        fish_both[index1e, self.N_t_even+index2o] = 1./144.*self.base.A_pix*np.sum(fish_summand_eo/8.)                                             
                                                
                                                # Update indexing
                                                index2e+=1
                                                if ((bin1p==bin3p)*(bin2p==bin4p))!=1:
                                                    index2o += 1
                            
                            # Update indexing
                            index1e += 1
                            if ((bin1==bin3)*(bin2==bin4))!=1:
                                index1o += 1

        # Return Fisher contributions
        if parity=='even':
            return fish_even
        elif parity=='odd':
            return fish_odd
        else:
            return fish_both
        
    def compute_fisher(self, N_it, parity='even', N_cpus=1, verb=False):
        """
        Compute the Fisher matrix using N_it realizations. If N_cpus > 1, this parallelizes the operations (though HEALPix is already parallelized so the speed-up is not particularly significant).

        The `parity' parameter can be 'even', 'odd' or 'both'. This specifies what parity trispectra to compute.
        
        For high-dimensional problems, it is usually preferred to split the computation across a cluster with MPI, calling compute_fisher_contribution for each instead of this function.
        """

        # Check type
        if parity not in ['even','odd','both']:
            raise Exception("Parity parameter not set correctly!")
        
        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor_even') and parity!='odd':
            self._compute_even_symmetry_factor()
        if not hasattr(self, 'sym_factor_odd') and parity!='even':
            self._compute_odd_symmetry_factor()
        
        # Initialize output
        if parity=='even':
            fish = np.zeros((self.N_t_even,self.N_t_even))
        elif parity=='odd':
            fish = np.zeros((self.N_t_odd,self.N_t_odd))
        elif parity=='both':
            fish = np.zeros((self.N_t_even+self.N_t_odd,self.N_t_even+self.N_t_odd))
        else:
            raise Exception("Parity parameter not set correctly!")

        global _iterable
        def _iterable(seed):
            return self.compute_fisher_contribution(seed, parity=parity, verb=verb)
        
        if N_cpus==1:
            for seed in range(N_it):
                if seed%5==0: print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
                fish += self.compute_fisher_contribution(seed, parity=parity, verb=verb*(seed==0))/N_it
        else:
            p = mp.Pool(N_cpus)
            print("Computing Fisher contribution from %d pairs of Monte Carlo simulations on %d threads"%(N_it, N_cpus))
            all_fish = list(tqdm.tqdm(p.imap_unordered(_iterable,np.arange(N_it)),total=N_it))
            fish = np.sum(all_fish,axis=0)/N_it
        
        if parity=='even':
            self.fish_even = fish
            self.inv_fish_even = np.linalg.inv(fish)
        elif parity=='odd':
            self.fish_odd = fish
            self.inv_fish_odd = np.linalg.inv(fish)
        elif parity=='both':
            self.fish_both = fish
            self.inv_fish_both = np.linalg.inv(fish)
            # Also store even + odd matrices
            self.fish_even = fish[:self.N_t_even,:self.N_t_even]
            self.inv_fish_even = np.linalg.inv(self.fish_even)
            self.fish_odd = fish[self.N_t_even:,self.N_t_even:]
            self.inv_fish_odd = np.linalg.inv(self.fish_odd)
        
        return fish

    def Tl_unwindowed(self, data, fish=[], parity='even', include_disconnected_term=True, verb=False):
        """
        Compute the unwindowed trispectrum estimator.
        
        The `parity' parameter can be 'even', 'odd' or 'both'. This specifies what parity trispectra to compute.

        The code either uses pre-computed Fisher matrices or reads them in on input. 
        Note that for parity='both', the Fisher matrix should contain *both* even and odd trispectra, stacked (unlike for ideal estimators).
        
        Note that we return the imaginary part of the odd-parity trispectrum.

        We can also optionally switch off the disconnected terms.
        """
        
        # Check type
        if parity not in ['even','odd','both']:
            raise Exception("Parity parameter not set correctly!")
        
        # Read in Fisher matrices, if supplied
        if len(fish)!=0:
            if parity=='even':
                self.fish_even = fish
                self.inv_fish_even = np.linalg.inv(fish)
            elif parity=='odd':
                self.fish_odd = fish
                self.inv_fish_odd = np.linalg.inv(fish)
            elif parity=='both':
                self.fish_both = fish
                self.inv_fish_both = np.linalg.inv(fish)
        
        if parity=='even' and not hasattr(self,'inv_fish_even'):
            raise Exception("Need to compute even-parity Fisher matrix first!")
        if parity=='odd' and not hasattr(self,'inv_fish_odd'):
            raise Exception("Need to compute odd-parity Fisher matrix first!")
        if parity=='both' and not hasattr(self,'inv_fish_both'):
            raise Exception("Need to compute both-parity Fisher matrix first!")
        
        # Compute numerator
        Tl_num = self.Tl_numerator(data, parity=parity, include_disconnected_term=include_disconnected_term, verb=verb)
        
        # Apply Fisher matrix and output
        if parity=='even':
            Tl_even = np.matmul(self.inv_fish_even,Tl_num)
            return Tl_even
        if parity=='odd':
            Tl_odd = np.matmul(self.inv_fish_odd,Tl_num)
            return Tl_odd
        if parity=='both':
            Tl_both = np.matmul(self.inv_fish_both,np.concatenate([Tl_num[0],Tl_num[1]]))
            Tl_even = Tl_both[:self.N_t_even]
            Tl_odd = Tl_both[self.N_t_even:]
            return Tl_even, Tl_odd

    ### IDEAL ESTIMATOR
    def Tl_numerator_ideal(self, data, parity='even', verb=False, include_disconnected_term=True):
        """
        Compute the numerator of the idealized trispectrum estimator. We normalize by < mask^4 >.

        The `parity' parameter can be 'even', 'odd' or 'both'. This specifies what parity trispectra to compute.

        Note that we return the imaginary part of the odd-parity trispectrum.
        
        We can also optionally switch off the disconnected terms. This only affects the parity-even trispectrum.
        """
        # Check type
        if parity not in ['even','odd','both']:
            raise Exception("Parity parameter not set correctly!")
        
        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor_even') and parity!='odd':
            self._compute_even_symmetry_factor()
        if not hasattr(self, 'sym_factor_odd') and parity!='even':
            self._compute_odd_symmetry_factor()
        
        # Normalize data by C_th and transform to harmonic space
        Cinv_data_lm = self.base.safe_divide(self.base.to_lm(data),self.base.Cl_lm)
        
        # Compute H and H-bar maps
        if verb: print("Computing H^+- maps")
        H_map = [self._compute_H(self.ell_bins[bin1]*self.beam_lm*Cinv_data_lm) for bin1 in range(self.Nl_squeeze)]
        Hbar_map = [self._compute_H(self.phase_factor*self.ell_bins[bin1]*self.beam_lm*Cinv_data_lm) for bin1 in range(self.Nl_squeeze)]

        # Define array of A maps (restricting to bin2 <= bin1, by symmetry)
        if verb: print("Computing A maps")
        Alm = [[self._compute_Alm(H_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
        Abar_lm = [[self.phase_factor*self._compute_Alm(Hbar_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
        
        # Even parity estimator
        if parity=='even' or parity=='both':

            # Define 4-, 2- and 0-field arrays
            t4_even_num_ideal = np.zeros(self.N_t_even)
            if not include_disconnected_term:
                print("No subtraction of even-parity disconnected terms performed!")
            else:
                t2_even_num_ideal = np.zeros(self.N_t_even)
                t0_even_num_ideal = np.zeros(self.N_t_even)

            if verb: print("Assembling parity-even trispectrum numerator")
            
            # Compute squared field    
            Cinv_data_lm_sq = np.real(Cinv_data_lm*np.conj(Cinv_data_lm)*self.m_weight)

            index = 0
            # iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>=b2 if b1=b3
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl_squeeze):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl_squeeze):
                            if bin1==bin3 and bin4<bin2: continue # note different condition to odd estimator!
                            
                            # Compute summands
                            summand = self.m_weight*np.real(Abar_lm[bin2][bin1].conj()*Alm[bin4][bin3] + Alm[bin2][bin1].conj()*Abar_lm[bin4][bin3])
                                        
                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue

                                # Compute four-field term
                                t4_even_num_ideal[index] = 1./2.*np.sum(summand*self.ell_bins[binL])

                                if include_disconnected_term:
                                    # Check if two external bins are equal (if not, no contribution to 2- and 0-field terms)
                                    kroneckers = (bin1==bin3)*(bin2==bin4)+(bin1==bin4)*(bin2==bin3)
                                    if kroneckers==0:
                                        index += 1
                                        continue

                                    # Sum over ells for two- and zero-point terms
                                    value2, value0 = 0., 0.
                                    for l1 in range(self.l_bins[bin1],self.l_bins[bin1+1]):

                                        # Compute sum over l1
                                        Cinvsq_l1 = np.sum(Cinv_data_lm_sq[self.base.l_arr==l1]*self.beam[l1]**2)

                                        for l2 in range(self.l_bins_squeeze[bin2],self.l_bins_squeeze[bin2+1]):

                                            # Compute sum over l2
                                            Cinvsq_l2 = np.sum(Cinv_data_lm_sq[self.base.l_arr==l2]*self.beam[l2]**2)
                                            
                                            for L in range(self.L_bins[binL],self.L_bins[binL+1]):
                                                if L<abs(l1-l2) or L>l1+l2: continue

                                                # define 3j symbols with spin (-1, -1, 2)
                                                tjs = self.threej(l1,l2,L)**2.

                                                # 2-field: (l1, l2) contribution
                                                value2 += -(2.*L+1.)/(4.*np.pi)*tjs*(-1.)**(l1+l2+L)*((2.*l1+1.)*Cinvsq_l2*self.beam[l1]**2/self.base.Cl[l1]+(2.*l2+1.)*Cinvsq_l1*self.beam[l2]**2/self.base.Cl[l2])*kroneckers
                                                
                                                # 0-field contribution
                                                value0 += (2.*l1+1.)*(2.*l2+1.)*(2.*L+1.)/(4.*np.pi)*tjs*(-1.)**(l1+l2+L)*self.beam[l1]**2*self.beam[l2]**2/self.base.Cl[l1]/self.base.Cl[l2]*kroneckers
                                                
                                    t2_even_num_ideal[index] = value2
                                    t0_even_num_ideal[index] = value0

                                index += 1

            if include_disconnected_term:
                t_even_num_ideal = (t4_even_num_ideal/np.mean(self.mask**4.)+t2_even_num_ideal/np.mean(self.mask**2.)+t0_even_num_ideal)/self.sym_factor_even
            else:
                t_even_num_ideal = t4_even_num_ideal/np.mean(self.mask**4.)/self.sym_factor_even

        # Odd parity estimator
        if parity=='odd' or parity=='both':
            
            # Define arrays
            t_odd_num_ideal = np.zeros(self.N_t_odd)
        
            # iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>b2 if b1=b3
            if verb: print("Assembling parity-odd trispectrum numerator")
            index = 0
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl_squeeze):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl_squeeze):
                            if bin1==bin3 and bin4<=bin2: continue
                                        
                            # Compute summands
                            summand = self.m_weight*np.imag(Abar_lm[bin2][bin1].conj()*Alm[bin4][bin3] - Alm[bin2][bin1].conj()*Abar_lm[bin4][bin3])
                            
                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue

                                # Compute estimator numerator
                                t_odd_num_ideal[index] = -1./2.*np.sum(summand*self.ell_bins[binL])
                                index += 1

            # Normalize
            t_odd_num_ideal *= 1./self.sym_factor_odd/np.mean(self.mask**4.)

        if parity=='even':
            return t_even_num_ideal
        elif parity=='odd':
            return t_odd_num_ideal
        else:
            return t_even_num_ideal, t_odd_num_ideal

    def compute_fisher_ideal(self, parity='even', verb=False, N_cpus=1, diagonal=False):
        """
        This computes the idealized Fisher matrix for the trispectrum. If N_cpus > 1, this parallelizes the operation.

        The `parity' parameter can be 'even', 'odd' or 'both'. This specifies what parity trispectra to compute.

        We can optionally drop any off-diagonal terms in the ideal Fisher matrix and restrict the internal L values. This is not recommended in practice!
        """
        # Check type
        if parity not in ['even','odd','both']:
            raise Exception("Parity parameter not set correctly!")
        
        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor_even'): # always need this!
            self._compute_even_symmetry_factor()
        if not hasattr(self, 'sym_factor_odd') and parity!='even':
            self._compute_odd_symmetry_factor()

        if diagonal:
            print("\n## Caution: dropping off-diagonal terms in the Fisher matrix!\n")
                
        # Define arrays
        if parity!='even':
            fish_odd = np.zeros((self.N_t_odd, self.N_t_odd))
        if parity!='odd':
            fish_even = np.zeros((self.N_t_even, self.N_t_even))

        if N_cpus==1:
            # Iterate over first set of bins
            # Note that we use two sets of indices here, since there are a different number of odd and even bins
            index1e = -1
            index1o = -1
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl_squeeze):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl_squeeze):
                            if bin1==bin3 and bin4<bin2: continue
                            if bin1==bin3 and bin2==bin4 and parity=='odd': continue
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue
                                
                                # Update indices
                                index1e += 1
                                if ((bin1==bin3)*(bin2==bin4))!=1:
                                    index1o += 1 # no equal bins!
                                
                                if verb and parity!='odd':
                                    if (index1e+1)%5==0: print("Computing bin %d of %d"%(index1e+1,self.N_t_even))
                                if verb and parity=='odd':
                                    if (index1o+1)%5==0: print("Computing bin %d of %d"%(index1o+1,self.N_t_odd))
                                        
                                # Iterate over second set of bins
                                index2e = -1
                                index2o = -1
                                for bin1p in range(self.Nl):
                                    for bin2p in range(bin1p,self.Nl_squeeze):
                                        for bin3p in range(bin1p,self.Nl):
                                            for bin4p in range(bin3p,self.Nl_squeeze):
                                                if bin1p==bin3p and bin4p<bin2p: continue
                                                if bin1p==bin3p and bin2p==bin4p and parity=='odd': continue
                                                for binLp in range(self.NL):
                                                    # skip bins outside the triangle conditions
                                                    if not self._check_bin(bin1p,bin2p,binLp,even=False): continue
                                                    if not self._check_bin(bin3p,bin4p,binLp,even=False): continue
                                                    
                                                    # Update indices
                                                    index2e += 1
                                                    if ((bin1p==bin3p)*(bin2p==bin4p))!=1:
                                                        index2o += 1 # no equal bins!
                                                    
                                                    # fill in this part by symmetry!
                                                    if index2e<index1e or index2o<index1o: continue
                                                        
                                                    ## Compute permutation factors
                                                    pref1  = (bin1==bin1p)*(bin2==bin2p)*(bin3==bin3p)*(bin4==bin4p)*(binL==binLp)
                                                    pref1 += (bin1==bin2p)*(bin2==bin1p)*(bin3==bin3p)*(bin4==bin4p)*(binL==binLp)
                                                    pref1 += (bin1==bin1p)*(bin2==bin2p)*(bin3==bin4p)*(bin4==bin3p)*(binL==binLp)
                                                    pref1 += (bin1==bin2p)*(bin2==bin1p)*(bin3==bin4p)*(bin4==bin3p)*(binL==binLp)
                                                    pref1 += (bin1==bin3p)*(bin2==bin4p)*(bin3==bin1p)*(bin4==bin2p)*(binL==binLp)
                                                    pref1 += (bin1==bin4p)*(bin2==bin3p)*(bin3==bin1p)*(bin4==bin2p)*(binL==binLp)
                                                    pref1 += (bin1==bin3p)*(bin2==bin4p)*(bin3==bin2p)*(bin4==bin1p)*(binL==binLp)
                                                    pref1 += (bin1==bin4p)*(bin2==bin3p)*(bin3==bin2p)*(bin4==bin1p)*(binL==binLp)

                                                    pref2  = (bin1==bin1p)*(bin2==bin3p)*(bin3==bin2p)*(bin4==bin4p)
                                                    pref2 += (bin1==bin2p)*(bin2==bin3p)*(bin3==bin1p)*(bin4==bin4p)
                                                    pref2 += (bin1==bin1p)*(bin2==bin4p)*(bin3==bin2p)*(bin4==bin3p)
                                                    pref2 += (bin1==bin2p)*(bin2==bin4p)*(bin3==bin1p)*(bin4==bin3p)
                                                    pref2 += (bin1==bin3p)*(bin2==bin1p)*(bin3==bin4p)*(bin4==bin2p)
                                                    pref2 += (bin1==bin3p)*(bin2==bin2p)*(bin3==bin4p)*(bin4==bin1p)
                                                    pref2 += (bin1==bin4p)*(bin2==bin1p)*(bin3==bin3p)*(bin4==bin2p)
                                                    pref2 += (bin1==bin4p)*(bin2==bin2p)*(bin3==bin3p)*(bin4==bin1p)
                                                    
                                                    pref3  = (bin1==bin1p)*(bin2==bin4p)*(bin3==bin3p)*(bin4==bin2p)
                                                    pref3 += (bin1==bin2p)*(bin2==bin4p)*(bin3==bin3p)*(bin4==bin1p)
                                                    pref3 += (bin1==bin1p)*(bin2==bin3p)*(bin3==bin4p)*(bin4==bin2p)
                                                    pref3 += (bin1==bin2p)*(bin2==bin3p)*(bin3==bin4p)*(bin4==bin1p)
                                                    pref3 += (bin1==bin3p)*(bin2==bin2p)*(bin3==bin1p)*(bin4==bin4p)
                                                    pref3 += (bin1==bin3p)*(bin2==bin1p)*(bin3==bin2p)*(bin4==bin4p)
                                                    pref3 += (bin1==bin4p)*(bin2==bin2p)*(bin3==bin1p)*(bin4==bin3p)
                                                    pref3 += (bin1==bin4p)*(bin2==bin1p)*(bin3==bin2p)*(bin4==bin3p)
                                                            
                                                    if pref1+pref2+pref3==0: continue
                                                        
                                                    # Check if two sets of bins are permutations of each other (no contribution else!)
                                                    b1234 = np.sort([bin1,bin2,bin3,bin4])
                                                    b1234p = np.sort([bin1p,bin2p,bin3p,bin4p])
                                                    if not (b1234==b1234p).all():
                                                        continue
                                            
                                                    value_even = 0.
                                                    value_odd = 0.

                                                    # Now iterate over l bins
                                                    for l1 in range(self.l_bins[bin1],self.l_bins[bin1+1]):
                                                        for l2 in range(self.l_bins_squeeze[bin2],self.l_bins_squeeze[bin2+1]):
                                                            for L in range(self.L_bins[binL],self.L_bins[binL+1]):

                                                                # Check triangle conditions
                                                                if L<abs(l1-l2) or L>l1+l2: continue

                                                                # first 3j symbols with spin (-1, -1, 2)
                                                                tj12 = self.threej(l1,l2,L)
                                                                
                                                                for l3 in range(self.l_bins[bin3],self.l_bins[bin3+1]):
                                                                    for l4 in range(self.l_bins_squeeze[bin4],self.l_bins_squeeze[bin4+1]):
                                                                        if L<abs(l3-l4) or L>l3+l4: continue
                                                                        
                                                                        # Continue if wrong-parity, or in [b1=b3, b2=b4] bin and odd
                                                                        if (-1)**(l1+l2+l3+l4)==-1 and ((bin1==bin3)*(bin2==bin4))==1: continue
                                                                        if (-1)**(l1+l2+l3+l4)==-1 and ((bin1p==bin3p)*(bin2p==bin4p))==1: continue
                                                                        if (-1)**(l1+l2+l3+l4)==1 and parity=='odd': continue
                                                                        if (-1)**(l1+l2+l3+l4)==-1 and parity=='even': continue
                                                                        
                                                                        # second 3j symbols with spin (-1, -1, 2)
                                                                        tj34 = self.threej(l3,l4,L)
                                                                        
                                                                        ## add first permutation
                                                                        if pref1!=0 and tj12*tj34!=0:

                                                                            # assemble relevant contribution
                                                                            if (-1)**(l1+l2+l3+l4)==-1:
                                                                                value_odd += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref1*tj12**2*tj34**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)/(4.*np.pi)**2/self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                            else:
                                                                                value_even += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref1*tj12**2*tj34**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)/(4.*np.pi)**2/self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]

                                                                        if diagonal: continue

                                                                        # Iterate over L' for off-diagonal terms
                                                                        for Lp in range(self.L_bins[binLp],self.L_bins[binLp+1]):

                                                                            # Impose 6j symmetries
                                                                            if Lp<abs(l3-l4) or Lp>l3+l4: continue
                                                                            if Lp<abs(l1-l2) or Lp>l1+l2: continue
                                                                            
                                                                            # Compute 3j symbols if non-zero
                                                                            if Lp>=abs(l1-l3) and Lp<=l1+l3 and Lp>=abs(l2-l4) and Lp<=l2+l4: 
                                                                                tj1324 = self.threej(l1,l3,Lp)*self.threej(l2,l4,Lp)
                                                                            else:
                                                                                tj1324 = 0
                                                                            if Lp>=abs(l1-l4) and Lp<=l1+l4 and Lp>=abs(l2-l3) and Lp<=l2+l3:
                                                                                tj1432 = self.threej(l1,l4,Lp)*self.threej(l3,l2,Lp)
                                                                            else:
                                                                                tj1432 = 0

                                                                            ## add second permutation
                                                                            if pref2!=0 and tj1324!=0: 
                                                                                if (-1)**(l1+l2+l3+l4)==-1:
                                                                                    value_odd += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref2*(-1.)**(l2+l3)*tj12*tj34*tj1324*self.sixj(L,l1,l2,Lp,l4,l3)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2.*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                                else:
                                                                                    value_even += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref2*(-1.)**(l2+l3)*tj12*tj34*tj1324*self.sixj(L,l1,l2,Lp,l4,l3)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2.*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]

                                                                            ## add third permutation
                                                                            if pref3!=0 and tj1432!=0:
                                                                                if (-1)**(l1+l2+l3+l4)==-1:
                                                                                    value_odd += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref3*(-1.)**(L+Lp)*tj12*tj34*tj1432*self.sixj(L,l1,l2,Lp,l3,l4)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                                else:
                                                                                    value_even += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref3*(-1.)**(L+Lp)*tj12*tj34*tj1432*self.sixj(L,l1,l2,Lp,l3,l4)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                        
                                                    # Note that matrix is symmetric if ideal!
                                                    if parity!='even' and ((bin1==bin3)*(bin2==bin4)!=1) and ((bin1p==bin3p)*(bin2p==bin4p)!=1):
                                                        fish_odd[index1o, index2o] = value_odd
                                                        fish_odd[index2o, index1o] = value_odd
                                                    if parity!='odd':
                                                        fish_even[index1e, index2e] = value_even
                                                        fish_even[index2e, index1e] = value_even
        elif N_cpus>1:

            global _iterator
            def _iterator(index1e_input):
                """Create an iterator for multiprocessing. This iterates over the first index."""
                # Iterate over first set of bins
                # Note that we use two sets of indices here, since there are a different number of odd and even bins
                
                if parity!='even':
                    fish_odd_internal = np.zeros((self.N_t_odd, self.N_t_odd))
                if parity!='odd':
                    fish_even_internal = np.zeros((self.N_t_even, self.N_t_even))
                
                index1e = -1
                index1o = -1
                for bin1 in range(self.Nl):
                    for bin2 in range(bin1,self.Nl_squeeze):
                        for bin3 in range(bin1,self.Nl):
                            for bin4 in range(bin3,self.Nl_squeeze):
                                if bin1==bin3 and bin4<bin2: continue
                                if bin1==bin3 and bin2==bin4 and parity=='odd': continue
                                for binL in range(self.NL):
                                    # skip bins outside the triangle conditions
                                    if not self._check_bin(bin1,bin2,binL,even=False): continue
                                    if not self._check_bin(bin3,bin4,binL,even=False): continue
                                    
                                    # Update indices
                                    index1e += 1

                                    if ((bin1==bin3)*(bin2==bin4))!=1:
                                        index1o += 1 # no equal bins!

                                    # Specialize to only the desired index
                                    if index1e!=index1e_input:
                                        continue
                                        
                                    # Iterate over second set of bins
                                    index2e = -1
                                    index2o = -1
                                    for bin1p in range(self.Nl):
                                        for bin2p in range(bin1p,self.Nl_squeeze):
                                            for bin3p in range(bin1p,self.Nl):
                                                for bin4p in range(bin3p,self.Nl_squeeze):
                                                    if bin1p==bin3p and bin4p<bin2p: continue
                                                    if bin1p==bin3p and bin2p==bin4p and parity=='odd': continue
                                                    for binLp in range(self.NL):
                                                        # skip bins outside the triangle conditions
                                                        if not self._check_bin(bin1p,bin2p,binLp,even=False): continue
                                                        if not self._check_bin(bin3p,bin4p,binLp,even=False): continue
                                                        
                                                        # Update indices
                                                        index2e += 1
                                                        if ((bin1p==bin3p)*(bin2p==bin4p))!=1:
                                                            index2o += 1 # no equal bins!
                                                            
                                                        # fill in this part by symmetry!
                                                        if index2e<index1e or index2o<index1o: continue
                                                        
                                                        ## Compute permutation factors
                                                        pref1  = (bin1==bin1p)*(bin2==bin2p)*(bin3==bin3p)*(bin4==bin4p)*(binL==binLp)
                                                        pref1 += (bin1==bin2p)*(bin2==bin1p)*(bin3==bin3p)*(bin4==bin4p)*(binL==binLp)
                                                        pref1 += (bin1==bin1p)*(bin2==bin2p)*(bin3==bin4p)*(bin4==bin3p)*(binL==binLp)
                                                        pref1 += (bin1==bin2p)*(bin2==bin1p)*(bin3==bin4p)*(bin4==bin3p)*(binL==binLp)
                                                        pref1 += (bin1==bin3p)*(bin2==bin4p)*(bin3==bin1p)*(bin4==bin2p)*(binL==binLp)
                                                        pref1 += (bin1==bin4p)*(bin2==bin3p)*(bin3==bin1p)*(bin4==bin2p)*(binL==binLp)
                                                        pref1 += (bin1==bin3p)*(bin2==bin4p)*(bin3==bin2p)*(bin4==bin1p)*(binL==binLp)
                                                        pref1 += (bin1==bin4p)*(bin2==bin3p)*(bin3==bin2p)*(bin4==bin1p)*(binL==binLp)

                                                        pref2  = (bin1==bin1p)*(bin2==bin3p)*(bin3==bin2p)*(bin4==bin4p)
                                                        pref2 += (bin1==bin2p)*(bin2==bin3p)*(bin3==bin1p)*(bin4==bin4p)
                                                        pref2 += (bin1==bin1p)*(bin2==bin4p)*(bin3==bin2p)*(bin4==bin3p)
                                                        pref2 += (bin1==bin2p)*(bin2==bin4p)*(bin3==bin1p)*(bin4==bin3p)
                                                        pref2 += (bin1==bin3p)*(bin2==bin1p)*(bin3==bin4p)*(bin4==bin2p)
                                                        pref2 += (bin1==bin3p)*(bin2==bin2p)*(bin3==bin4p)*(bin4==bin1p)
                                                        pref2 += (bin1==bin4p)*(bin2==bin1p)*(bin3==bin3p)*(bin4==bin2p)
                                                        pref2 += (bin1==bin4p)*(bin2==bin2p)*(bin3==bin3p)*(bin4==bin1p)
                                                        
                                                        pref3  = (bin1==bin1p)*(bin2==bin4p)*(bin3==bin3p)*(bin4==bin2p)
                                                        pref3 += (bin1==bin2p)*(bin2==bin4p)*(bin3==bin3p)*(bin4==bin1p)
                                                        pref3 += (bin1==bin1p)*(bin2==bin3p)*(bin3==bin4p)*(bin4==bin2p)
                                                        pref3 += (bin1==bin2p)*(bin2==bin3p)*(bin3==bin4p)*(bin4==bin1p)
                                                        pref3 += (bin1==bin3p)*(bin2==bin2p)*(bin3==bin1p)*(bin4==bin4p)
                                                        pref3 += (bin1==bin3p)*(bin2==bin1p)*(bin3==bin2p)*(bin4==bin4p)
                                                        pref3 += (bin1==bin4p)*(bin2==bin2p)*(bin3==bin1p)*(bin4==bin3p)
                                                        pref3 += (bin1==bin4p)*(bin2==bin1p)*(bin3==bin2p)*(bin4==bin3p)
                                                                
                                                        if pref1+pref2+pref3==0: continue
                                                            
                                                        value_even = 0.
                                                        value_odd = 0.

                                                        # Now iterate over l bins
                                                        for l1 in range(self.l_bins[bin1],self.l_bins[bin1+1]):
                                                            for l2 in range(self.l_bins_squeeze[bin2],self.l_bins_squeeze[bin2+1]):
                                                                for L in range(self.L_bins[binL],self.L_bins[binL+1]):
                                                                    # Check triangle conditions
                                                                    if L<abs(l1-l2) or L>l1+l2: continue
                                                                    # first 3j symbols with spin (-1, -1, 2)
                                                                    tj12 = self.threej(l1,l2,L)
                                                                    for l3 in range(self.l_bins[bin3],self.l_bins[bin3+1]):
                                                                        for l4 in range(self.l_bins_squeeze[bin4],self.l_bins_squeeze[bin4+1]):
                                                                            if L<abs(l3-l4) or L>l3+l4: continue
                                                                            
                                                                            # Continue if wrong-parity, or in [b1=b3, b2=b4] bin and odd
                                                                            if (-1)**(l1+l2+l3+l4)==-1 and ((bin1==bin3)*(bin2==bin4))==1: continue
                                                                            if (-1)**(l1+l2+l3+l4)==-1 and ((bin1p==bin3p)*(bin2p==bin4p))==1: continue
                                                                            if (-1)**(l1+l2+l3+l4)==1 and parity=='odd': continue
                                                                            if (-1)**(l1+l2+l3+l4)==-1 and parity=='even': continue
                                                                            
                                                                            # second 3j symbols with spin (-1, -1, 2)
                                                                            tj34 = self.threej(l3,l4,L)
                                                                            
                                                                            ## add first permutation
                                                                            if pref1!=0 and tj12*tj34!=0:

                                                                                # assemble relevant contribution
                                                                                if (-1)**(l1+l2+l3+l4)==-1:
                                                                                    value_odd += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref1*tj12**2*tj34**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)/(4.*np.pi)**2/self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                                else:
                                                                                    value_even += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref1*tj12**2*tj34**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)/(4.*np.pi)**2/self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]

                                                                            if diagonal: continue

                                                                            # Iterate over L' for off-diagonal terms
                                                                            for Lp in range(self.L_bins[binLp],self.L_bins[binLp+1]):
                                                                                # Impose 6j symmetries
                                                                                if Lp<abs(l3-l4) or Lp>l3+l4: continue
                                                                                if Lp<abs(l1-l2) or Lp>l1+l2: continue
                                                                                
                                                                                # Compute 3j symbols if non-zero
                                                                                if Lp>=abs(l1-l3) and Lp<=l1+l3 and Lp>=abs(l2-l4) and Lp<=l2+l4: 
                                                                                    tj1324 = self.threej(l1,l3,Lp)*self.threej(l2,l4,Lp)
                                                                                else:
                                                                                    tj1324 = 0
                                                                                if Lp>=abs(l1-l4) and Lp<=l1+l4 and Lp>=abs(l2-l3) and Lp<=l2+l3:
                                                                                    tj1432 = self.threej(l1,l4,Lp)*self.threej(l3,l2,Lp)
                                                                                else:
                                                                                    tj1432 = 0

                                                                                ## add second permutation
                                                                                if pref2!=0 and tj1324!=0: 
                                                                                    if (-1)**(l1+l2+l3+l4)==-1:
                                                                                        value_odd += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref2*(-1.)**(l2+l3)*tj12*tj34*tj1324*self.sixj(L,l1,l2,Lp,l4,l3)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2.*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                                    else:
                                                                                        value_even += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref2*(-1.)**(l2+l3)*tj12*tj34*tj1324*self.sixj(L,l1,l2,Lp,l4,l3)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2.*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]

                                                                                ## add third permutation
                                                                                if pref3!=0 and tj1432!=0:
                                                                                    if (-1)**(l1+l2+l3+l4)==-1:
                                                                                        value_odd += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref3*(-1.)**(L+Lp)*tj12*tj34*tj1432*self.sixj(L,l1,l2,Lp,l3,l4)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                                    else:
                                                                                        value_even += self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2*pref3*(-1.)**(L+Lp)*tj12*tj34*tj1432*self.sixj(L,l1,l2,Lp,l3,l4)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                            
                                                        # Note that matrix is symmetric if ideal!
                                                        if parity!='even' and ((bin1==bin3)*(bin2==bin4)!=1) and ((bin1p==bin3p)*(bin2p==bin4p)!=1):
                                                            fish_odd_internal[index1o, index2o] = value_odd
                                                            fish_odd_internal[index2o, index1o] = value_odd
                                                        if parity!='odd':
                                                            fish_even_internal[index1e, index2e] = value_even
                                                            fish_even_internal[index2e, index1e] = value_even

                if parity=='odd':
                    return fish_odd_internal
                elif parity=='even':
                    return fish_even_internal
                else:
                    return fish_even_internal, fish_odd_internal

            # Now run multiprocessing
            p = mp.Pool(N_cpus)
            print("Multiprocessing computation on %d cores"%N_cpus)

            result = list(tqdm.tqdm(p.imap_unordered(_iterator,range(self.N_t_even)),total=self.N_t_even))
            if parity=='odd':
                fish_odd = np.sum(result,axis=0)
            elif parity=='even':
                fish_even = np.sum(result,axis=0)
            else:
                fish_even = np.sum([r[0] for r in result],axis=0)
                fish_odd  = np.sum([r[1] for r in result],axis=0)

        else:
            raise Exception("Need at least one CPU!")

        # Add symmetry factors and save attributes
        if parity!='even':
            fish_odd *= 1./np.outer(self.sym_factor_odd,self.sym_factor_odd)
            self.fish_ideal_odd = fish_odd
            self.inv_fish_ideal_odd = np.linalg.inv(fish_odd)
        if parity!='odd':
            fish_even *= 1./np.outer(self.sym_factor_even,self.sym_factor_even)
            self.fish_ideal_even = fish_even
            self.inv_fish_ideal_even = np.linalg.inv(fish_even)
        
        # Return matrices
        if parity=='even':
            return fish_even
        elif parity=='odd':
            return fish_odd
        else:
            return fish_even, fish_odd
    
    def Tl_ideal(self, data, fish_ideal=[], parity='even', verb=False, include_disconnected_term=True, N_cpus=1):
        """
        Compute the idealized trispectrum estimator, including normalization, if not supplied or already computed. Note that this normalizes by < mask^4 >.
        
        The `parity' parameter can be 'even', 'odd' or 'both'. This specifies what parity trispectra to compute.

        Note that we return the imaginary part of the odd-parity trispectrum.

        We can also optionally switch off the disconnected terms. This only affects the parity-even trispectrum.
        
        The N_cpus parameter specifies how many CPUs to use in computation of the ideal Fisher matrix.
        """
        # Check type
        if parity not in ['even','odd','both']:
            raise Exception("Parity parameter not set correctly!")
        
        # Read in Fisher matrices, if supplied
        if len(fish_ideal)!=0:
            if parity=='even':
                self.fish_ideal_even = fish_ideal
                self.inv_fish_ideal_even = np.linalg.inv(fish_ideal)
            elif parity=='odd':
                self.fish_ideal_odd = fish_ideal
                self.inv_fish_ideal_odd = np.linalg.inv(fish_ideal)
            elif parity=='both':
                if len(fish_ideal)!=2:
                    raise Exception("Must supply two Fisher matrices: even and odd!")
                self.fish_ideal_even = fish_ideal[0]
                self.fish_ideal_odd = fish_ideal[1]
                self.inv_fish_ideal_even = np.linalg.inv(fish_ideal[0])
                self.inv_fish_ideal_odd = np.linalg.inv(fish_ideal[1])
        
        # Compute Fisher matrices, if not supplied
        if (parity!='odd' and not hasattr(self,'inv_fish_ideal_even')) or (parity!='even' and not hasattr(self,'inv_fish_ideal_odd')):
            print("Computing ideal Fisher matrix")
            self.compute_fisher_ideal(parity=parity, verb=verb, N_cpus=N_cpus)
        else:
            print("Using precomputed Fisher matrix")
            
        # Compute numerator
        if verb: print("Computing numerator")
        Tl_num_ideal = self.Tl_numerator_ideal(data, parity=parity, include_disconnected_term=include_disconnected_term, verb=verb)
        
        # Compute full estimator
        if parity=='even':
            Tl_even = np.matmul(self.inv_fish_ideal_even,Tl_num_ideal)
            return Tl_even
        elif parity=='odd':
            Tl_odd = np.matmul(self.inv_fish_ideal_odd,Tl_num_ideal)
            return Tl_odd
        else:
            Tl_even = np.matmul(self.inv_fish_ideal_even,Tl_num_ideal[0])
            Tl_odd = np.matmul(self.inv_fish_ideal_odd,Tl_num_ideal[1])
            return Tl_even, Tl_odd
    