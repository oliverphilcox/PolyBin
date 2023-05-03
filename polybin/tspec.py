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
    We also feed in a function that applies the S^-1 operator.

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
    - fields: which T/E/B trispectra to compute
    - parity: whether to include contributions from parity 'even' , 'odd' physics or 'both'.
    """
    def __init__(self, base, mask, applySinv, l_bins, l_bins_squeeze=[], L_bins=[], include_partial_triangles=False, fields=['TTTT','TTTE','TTEE','TTBB','TETE','TEEE','TEBB','TBTB','TBEB','EEEE','EEBB','EBEB','BBBB','TTTB','TTEB','TETB','TEEB','TBEE','TBBB','EEEB','EBBB'], parity='both'):
        # Read in attributes
        self.base = base
        self.mask = mask
        self.applySinv = applySinv
        self.l_bins = l_bins
        self.pol = self.base.pol
        self.fields = fields
        self.parity = parity
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

        self.include_partial_triangles = include_partial_triangles
        self.beam = self.base.beam
        self.beam_lm = self.base.beam_lm
        
        # Check correct fields are being used
        for f in fields:
            assert f in ['TTTT','TTTE','TTEE','TTBB','TETE','TEEE','TEBB','TBTB','TBEB','EEEE','EEBB','EBEB','BBBB','TTTB','TTEB','TETB','TEEB','TBEE','TBBB','EEEB','EBBB'], "Unknown field '%s' supplied!"%f 
        assert len(fields)==len(np.unique(fields)), "Duplicate fields supplied!"
        
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
        print("Binning: %d bins in [%d, %d]"%(self.Nl,self.min_l,np.max(self.l_bins)))
        if self.Nl_squeeze!=self.Nl:
            print("Squeezed binning: %d bins in [%d, %d]"%(self.Nl_squeeze,self.min_l,np.max(self.l_bins_squeeze)))
        if self.NL!=self.Nl_squeeze:
            print("L binning: %d bins in [%d, %d]"%(self.NL,self.min_l,np.max(self.L_bins))) 
        
        # Check if window is uniform
        if np.std(self.mask)<1e-12 and np.abs(np.mean(self.mask)-1)<1e-12:
            print("Mask: ones")
            self.ones_mask = True
        else:
            print("Mask: spatially varying")
            self.ones_mask = False
        
        # Define l filters
        self.ell_bins = [(self.base.l_arr>=self.l_bins_squeeze[bin1])&(self.base.l_arr<self.l_bins_squeeze[bin1+1]) for bin1 in range(self.Nl_squeeze)]

        # Define wigner calculator
        wig.wig_table_init(self.base.lmax*2,9)
        wig.wig_temp_init(self.base.lmax*2)

        # Define 3j with specified spins, and 6j
        self.threej = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,2*-1,2*-1,2*2)
        self.sixj = lambda l1,l2,l3,l4,l5,l6: wig.wig6jj(2*l1,2*l2,2*l3,2*l4,2*l5,2*l6)

    def _check_bin(self, bin1, bin2, bin3):
        """
        Return one if modes in the bin satisfy the triangle conditions, or zero else.

        This is used either for all triangles in the bin, or just the center of the bin.
        """
        if self.include_partial_triangles:
            good = 0
            for l1 in range(self.l_bins_squeeze[bin1],self.l_bins_squeeze[bin1+1]):
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
        self.sym_factor_all = []

        # Iterate over parities
        for chi in self.chi_arr:
            
            # Iterate over fields
            for u in self.fields:
                u1, u2, u3, u4 = [self.base.indices[u[i]] for i in range(4)]

                sym_factor = []
                # iterate over bins satisfying relevant conditions
                for bin1 in range(self.Nl):
                    for bin2 in range(self.Nl_squeeze):
                        if u1==u2 and bin2<bin1: continue
                        for bin3 in range(self.Nl):
                            if u1==u3 and u2==u4 and bin3<bin1: continue
                            for bin4 in range(self.Nl_squeeze):
                                if u3==u4 and bin4<bin3: continue
                                if u1==u3 and u2==u4 and bin1==bin3 and bin4<bin2: continue
                                
                                # Iterate over L bins
                                for binL in range(self.NL):
                                    # skip bins outside the triangle conditions
                                    if not self._check_bin(bin1,bin2,binL): continue
                                    if not self._check_bin(bin3,bin4,binL): continue

                                    # compute symmetry factor
                                    if bin1==bin2 and bin3==bin4 and bin1==bin3 and u1==u2 and u3==u4 and u1==u3:
                                        sym = 8
                                    elif bin1==bin2 and bin3==bin4 and u1==u2 and u3==u4:
                                        sym = 4
                                    elif bin1==bin2 and u1==u2:
                                        sym = 2
                                    elif bin3==bin4 and u3==u4:
                                        sym = 2
                                    elif bin1==bin3 and bin2==bin4 and u1==u3 and u2==u4:
                                        sym = 2
                                    else:
                                        sym = 1
                                    sym_factor.append(sym)   
                self.sym_factor_all.append(np.asarray(sym_factor))     
        
        self.sym_factor = np.concatenate(self.sym_factor_all)
        self.N_t = len(self.sym_factor)
        print("Using %d combination(s) of fields/parities"%len(self.sym_factor_all))
        print("Using a maximum of %d bins per field/parity"%np.max([len(s) for s in self.sym_factor_all]))
        print("Using a total of %d bins"%self.N_t)
    
    def get_ells(self, field='TTTT'):
        """
        Return an array with the central l1, l2, l3, l4, L values for each trispectrum bin, given a set of fields.
        """

        assert field in self.fields, "Specified field '%s' not in inputs!"%field
        u1, u2, u3, u4 = [self.base.indices[field[i]] for i in range(4)]

        # Define arrays
        l1s, l2s, l3s, l4s, Ls = [],[],[],[],[]

        # Iterate over bins satisfying relevant conditions
        for bin1 in range(self.Nl):
            l1 = 0.5*(self.l_bins[bin1]+self.l_bins[bin1+1])
            for bin2 in range(self.Nl_squeeze):
                if u1==u2 and bin2<bin1: continue
                l2 = 0.5*(self.l_bins_squeeze[bin2]+self.l_bins_squeeze[bin2+1])
                for bin3 in range(self.Nl):
                    if u1==u3 and u2==u4 and bin3<bin1: continue
                    l3 =  0.5*(self.l_bins[bin3]+self.l_bins[bin3+1])
                    for bin4 in range(self.Nl_squeeze):
                        if u3==u4 and bin4<bin3: continue
                        if u1==u3 and u2==u4 and bin1==bin3 and bin4<bin2: continue
                        l4 = 0.5*(self.l_bins_squeeze[bin4]+self.l_bins_squeeze[bin4+1])
                        
                        # Iterate over L bins
                        for binL in range(self.NL):
                            L = 0.5*(self.L_bins[binL]+self.L_bins[binL+1])

                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,binL): continue
                            if not self._check_bin(bin3,bin4,binL): continue

                            # Add to output array
                            l1s.append(l1)
                            l2s.append(l2)
                            l3s.append(l3)
                            l4s.append(l4)
                            Ls.append(L)
        
        return l1s,l2s,l3s,l4s,Ls
    
    def _compute_t0_numerator(self, verb=False):
        """
        Compute the zero-field contribution to the parity-odd or parity-even trispectrum. This is a sum over Monte Carlo simulations but does not involve data. 

        The output is *not* normalized by the symmetry factor.

        Note that this requires processed simulations, computed either from generate_sims() or load_sims().
        """

        # First check that simulations have been loaded
        if not hasattr(self, 'A_ab_lms'):
            raise Exception("Need to generate or specify bias simulations!")

        # Define arrays
        self.t0_num = np.zeros(self.N_t)
        
        # Iterate over fields
        index = 0
        for u in self.fields:
            u1, u2, u3, u4 = [self.base.indices[u[i]] for i in range(4)]
            p_u = np.product([self.base.parities[u[i]] for i in range(4)])
            if verb: print("Analyzing trispectrum numerator for field %s"%u)

            # Iterate over bins satisfying relevant conditions
            for bin1 in range(self.Nl):
                for bin2 in range(self.Nl_squeeze):
                    if u1==u2 and bin2<bin1: continue
                    for bin3 in range(self.Nl):
                        if u1==u3 and u2==u4 and bin3<bin1: continue
                        for bin4 in range(self.Nl_squeeze):
                            if u3==u4 and bin4<bin3: continue
                            if u1==u3 and u2==u4 and bin1==bin3 and bin4<bin2: continue

                            # Compute summands, summing over permutations and MC fields
                            # Note we have already symmetrized over A_xy vs A_yx
                            summand_t0 = 0.
                            for ii in range(self.N_it):
                                summand_t0 += self.A_bb_lms[ii][u2][u1][bin2][bin1][1].conj()*self.A_aa_lms[ii][u4][u3][bin4][bin3][0]+self.A_bb_lms[ii][u2][u1][bin2][bin1][0]*self.A_aa_lms[ii][u4][u3][bin4][bin3][1].conj()
                                summand_t0 += 4*(self.A_ab_lms[ii][u2][u1][bin2][bin1][1].conj()*self.A_ab_lms[ii][u4][u3][bin4][bin3][0]+self.A_ab_lms[ii][u2][u1][bin2][bin1][0]*self.A_ab_lms[ii][u4][u3][bin4][bin3][1].conj())
                                summand_t0 += self.A_aa_lms[ii][u2][u1][bin2][bin1][1].conj()*self.A_bb_lms[ii][u4][u3][bin4][bin3][0]+self.A_aa_lms[ii][u2][u1][bin2][bin1][0]*self.A_bb_lms[ii][u4][u3][bin4][bin3][1].conj()
                            summand_t0 = self.base.m_weight*summand_t0/self.N_it/4.

                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL): continue
                                if not self._check_bin(bin3,bin4,binL): continue

                                # Assemble numerators for both parities
                                chi_index = 0
                                for chi in self.chi_arr:
                                    if p_u*chi==-1:
                                        summand_t0_real = 1.0j*summand_t0
                                    else:
                                        summand_t0_real = summand_t0
                                    self.t0_num[chi_index*self.N_t//2+index] = np.real(np.sum(summand_t0_real*self.ell_bins[binL]))
                                    chi_index += 1
                                index += 1
                
    def _compute_H_maps(self, h_lm):
        """
        Compute (+-1)H(n) maps for each bin. These are used in the trispectrum numerators.
        """
        H_pm1_maps = []
        for bin1 in range(self.Nl_squeeze):
            H_pm1_maps.append([np.asarray([[1],[-1]])*self.base.to_map_spin(self.ell_bins[bin1]*self.beam_lm*h_lm[i].conj(),-1.*self.ell_bins[bin1]*self.beam_lm*h_lm[i].conj(),1) for i in range(len(h_lm))])
        return H_pm1_maps

    def _compute_A_lms(self, H_maps1, H_maps2 = []):
        """
        Compute A(L,M) and \bar{A}(L,M) fields for each bin pair given (+-1)H(n) maps. These are used in the trispectrum numerators. Note that we restrict to bin2 <= bin1 if u1=u2, by symmetry, and each element contains [A, \bar{A}].
        
        Note, we can optionally use two different H fields here (i.e. defined from different maps). These will be symmetrized over such that A_lm is symmetric.
        """
        if len(H_maps2)==0:
            H_maps2 = H_maps1
            one_field=True
        else:
            one_field=False

        A_lms = []
        for u1 in range(1+2*self.pol):
            A_lms1 = []
            for u2 in range(u1+1):
                A_lms2 = []
                for b1 in range(self.Nl_squeeze):
                    A_lms3 = []
                    for b2 in range(self.Nl):
                        if u1==u2 and b2>b1: continue
                        if one_field:
                            this_lm_plus = H_maps1[b1][u1][0]*H_maps2[b2][u2][0]
                            this_lm_minus = H_maps1[b1][u1][1]*H_maps2[b2][u2][1]
                        else:
                            this_lm_plus = (H_maps1[b1][u1][0]*H_maps2[b2][u2][0]+H_maps2[b1][u1][0]*H_maps1[b2][u2][0])/2.
                            this_lm_minus = (H_maps1[b1][u1][1]*H_maps2[b2][u2][1]+H_maps2[b1][u1][1]*H_maps1[b2][u2][1])/2.
                        A_maps = self.base.to_lm_spin(this_lm_plus,this_lm_minus,2)[::-1].conj()
                        A_lms3.append(A_maps)
                    A_lms2.append(A_lms3)
                A_lms1.append(A_lms2)
            A_lms.append(A_lms1)
        return A_lms

    def _compute_A_maps(self, A_lms):
        """
        Compute (+-2)A(n) maps for each pair of bins and L-bin, given A(L,M) and \bar{A}(L,M) fields. These are used in the trispectrum Fisher matrix.

        Note that we sum over exchange of the two fields in A_lm, since this quantity is needed in the Fisher matrix.
        """
        A_maps = []
        for u1 in range(1+2*self.pol):
            A_maps1 = []
            for u2 in range(u1+1):
                A_maps2 = []
                for b1 in range(self.Nl_squeeze):
                    A_maps3 = []
                    for b2 in range(self.Nl):
                        if u1==u2 and b2>b1: continue
                        A_maps4 = []
                        for B in range(self.NL):
                            if not self._check_bin(b1,b2,B):
                                A_maps4.append([])
                            else:
                                A_maps4.append(self.base.to_map_spin(A_lms[u1][u2][b1][b2][0].conj()*self.ell_bins[B],A_lms[u1][u2][b1][b2][1].conj()*self.ell_bins[B],2))
                        A_maps3.append(A_maps4)
                    A_maps2.append(A_maps3)
                A_maps1.append(A_maps2)
            A_maps.append(A_maps1)
        return A_maps

    def _compute_HA_lms(self, H_maps, A_maps):
        """
        Compute the harmonic space maps Int[(-1)H(n)(+2)A(n)(+1)Y^*_lm] and Int[(+1)H(n)(-2)A(n)(-1)Y^*_lm] for each choice of bin 2, 3, 4, and L-bin and fields u2,u3,u4.
        
        This is used in the trispectrum Fisher matrix.

        """    
        HA_lms = []
        for u2 in range(1+2*self.pol):
            HA_lms1 = []
            for u3 in range(1+2*self.pol):
                HA_lms2 = []
                for u4 in range(u3+1):
                    HA_lms3 = []
                    for b2 in range(self.Nl_squeeze):
                        this_H = H_maps[b2][u2]
                        HA_lms4 = []
                        for b3 in range(self.Nl_squeeze):
                            HA_lms5 = []
                            for b4 in range(self.Nl):
                                if u3==u4 and b4>b3: continue
                                HA_lms6 = []
                                for B in range(self.NL):
                                    if not self._check_bin(b3,b4,B): 
                                        HA_lms6.append([])
                                    else:
                                        this_A = A_maps[u3][u4][b3][b4][B]
                                        HA_lms6.append(np.asarray([[1],[-1]])*self.base.to_lm_spin(this_H[1]*this_A[0],-this_H[0]*this_A[1],1))
                                HA_lms5.append(HA_lms6)
                            HA_lms4.append(HA_lms5)
                        HA_lms3.append(HA_lms4)
                    HA_lms2.append(HA_lms3)
                HA_lms1.append(HA_lms2)
            HA_lms.append(HA_lms1)
        return HA_lms
    
    def load_sims(self, sims, verb=False, input_type='map'):
        """
        Load in Monte Carlo simulations used in the two- and zero-field terms of the trispectrum estimator. This should be a list of *pairs* of simulations, [[simA-1, simB-1], [simA-2, simB-2], ...]

        These can alternatively be generated with a fiducial spectrum using the generate_sims script.

        We can read in input simulations either in map- or harmonic-space.
        """
        
        self.N_it = len(sims)
        print("Using %d pairs of Monte Carlo simulations"%self.N_it)

        # Define lists
        self.A_aa_lms, self.A_bb_lms, self.A_ab_lms = [],[],[]
        self.H_a_maps, self.H_b_maps = [],[]
        
        for ii in range(self.N_it):
            if ii%5==0 and verb: print("Processing bias simulation %d of %d"%(ii+1,self.N_it))

            # Transform to Fourier space and normalize appropriately
            if self.ones_mask:
                Wh_alpha_lm = self.applySinv(sims[ii][0], input_type=input_type, output_type='harmonic')
                Wh_beta_lm = self.applySinv(sims[ii][1], input_type=input_type, output_type='harmonic')
            else:
                Wh_alpha_lm = self.base.to_lm(self.mask*self.applySinv(sims[ii][0],input_type=input_type))
                Wh_beta_lm = self.base.to_lm(self.mask*self.applySinv(sims[ii][1],input_type=input_type))
            
            # Compute H_alpha maps
            self.H_a_maps.append(self._compute_H_maps(Wh_alpha_lm))
            self.H_b_maps.append(self._compute_H_maps(Wh_beta_lm))
            
            # Compute A[alpha,alpha](b1,b2) maps
            self.A_aa_lms.append(self._compute_A_lms(self.H_a_maps[-1]))
            self.A_bb_lms.append(self._compute_A_lms(self.H_b_maps[-1]))

            # Compute A[alpha,beta](b1,b2) maps (symmetrized over alpha and beta)
            self.A_ab_lms.append(self._compute_A_lms(self.H_a_maps[-1], self.H_b_maps[-1]))
            
    def generate_sims(self, N_it, Cl_input=[], verb=False):
        """
        Generate Monte Carlo simulations used in the two- and zero-field terms of the trispectrum generator. 
        These are pure GRFs. By default, they are generated with the input survey mask.
        We create N_it such simulations and store the relevant transformations into memory.
        
        We can alternatively load custom simulations using the load_sims script.
        
        We can read in input simulations either in map- or harmonic-space.
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
            
            # Generate simulation and Fourier transform
            if self.ones_mask:
                raw_alpha_lm = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, output_type='harmonic')
                Wh_alpha_lm = self.applySinv(raw_alpha_lm, input_type='harmonic', output_type='harmonic')
                
                raw_beta_lm = self.base.generate_data(int(2e5)+ii, Cl_input=Cl_input, output_type='harmonic')
                Wh_beta_lm = self.applySinv(raw_beta_lm, input_type='harmonic', output_type='harmonic')
            else:
                raw_alpha = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input)
                Wh_alpha_lm = self.base.to_lm(self.mask*self.applySinv(raw_alpha*self.mask))
            
                raw_beta = self.base.generate_data(int(2e5)+ii, Cl_input=Cl_input)
                Wh_beta_lm = self.base.to_lm(self.mask*self.applySinv(raw_beta*self.mask))
        
            # Compute H_alpha maps
            self.H_a_maps.append(self._compute_H_maps(Wh_alpha_lm))
            self.H_b_maps.append(self._compute_H_maps(Wh_beta_lm))

            # Compute A[alpha,alpha](b1,b2) maps
            self.A_aa_lms.append(self._compute_A_lms(self.H_a_maps[-1]))
            self.A_bb_lms.append(self._compute_A_lms(self.H_b_maps[-1]))

            # Compute A[alpha,beta](b1,b2) maps (symmetrized over alpha and beta)
            self.A_ab_lms.append(self._compute_A_lms(self.H_a_maps[-1], self.H_b_maps[-1]))
        
    ### OPTIMAL ESTIMATOR
    def Tl_numerator(self, data, include_disconnected_term=True, verb=False):
        """
        Compute the numerator of the unwindowed trispectrum estimator for all combinations of fields.

        Note that we return the imaginary part of odd-parity trispectra.

        We can also optionally switch off the disconnected terms, which affects only parity-conserving trispectra.
        """
        
        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Check if simulations have been supplied
        if not hasattr(self, 'A_ab_lms') and include_disconnected_term:
            raise Exception("Need to generate or specify bias simulations!")

        # Compute t0 term, if not already computed
        if not hasattr(self, 't0_num') and include_disconnected_term:
            if verb: print("# Computing t0 term")
            self._compute_t0_numerator(verb=verb)
        
        # Apply W * S^-1 to data and transform to harmonic space
        if self.ones_mask:
            Wh_data_lm = self.applySinv(data, input_type='map', output_type='harmonic')
        else:
            Wh_data_lm = self.base.to_lm(self.mask*self.applySinv(data))

        # Compute H and H-bar maps
        if verb: print("Computing H maps")
        H_maps = self._compute_H_maps(Wh_data_lm)

        # Define array of A[d,d] maps
        if verb: print("Computing A_lm fields")
        A_dd_lms = self._compute_A_lms(H_maps)
        
        # Compute cross-spectra of MC simulations and data
        if include_disconnected_term:
            if verb: print("Computing A_lm fields for cross-spectra")

            # Compute all A[alpha, d] and A[beta, d] for all bins
            A_ad_lms = [self._compute_A_lms(self.H_a_maps[ii],H_maps) for ii in range(self.N_it)]
            A_bd_lms = [self._compute_A_lms(self.H_b_maps[ii],H_maps) for ii in range(self.N_it)]
                        
        # Define 4-, 2- and 0-field arrays
        t4_num = np.zeros(self.N_t)
        if not include_disconnected_term:
            print("## No subtraction of (parity-conserving) disconnected terms performed!")
        else:
            t2_num = np.zeros(self.N_t)
            t0_num = np.zeros(self.N_t)
        if verb: print("# Assembling trispectrum numerator")

        # Iterate over fields
        index = 0
        for u in self.fields:
            u1, u2, u3, u4 = [self.base.indices[u[i]] for i in range(4)]
            p_u = np.product([self.base.parities[u[i]] for i in range(4)])
            if verb: print("Analyzing trispectrum numerator for field %s"%u)

            # Iterate over bins satisfying relevant conditions
            for bin1 in range(self.Nl):
                for bin2 in range(self.Nl_squeeze):
                    if u1==u2 and bin2<bin1: continue
                    for bin3 in range(self.Nl):
                        if u1==u3 and u2==u4 and bin3<bin1: continue
                        for bin4 in range(self.Nl_squeeze):
                            if u3==u4 and bin4<bin3: continue
                            if u1==u3 and u2==u4 and bin1==bin3 and bin4<bin2: continue

                            # Compute summands
                            summand_t4 = self.base.m_weight*(A_dd_lms[u2][u1][bin2][bin1][1].conj()*A_dd_lms[u4][u3][bin4][bin3][0]+A_dd_lms[u2][u1][bin2][bin1][0]*A_dd_lms[u4][u3][bin4][bin3][1].conj())/2.

                            # Compute 2-field term summand
                            if include_disconnected_term: 
                                # Sum over 6 permutations and all MC fields
                                summand_t2 = 0.
                                for ii in range(self.N_it):
                                    # first set of fields (note we have already symmetrized over A_xy vs A_yx)
                                    summand_t2 += A_dd_lms[u2][u1][bin2][bin1][1].conj()*self.A_aa_lms[ii][u4][u3][bin4][bin3][0]+A_dd_lms[u2][u1][bin2][bin1][0]*self.A_aa_lms[ii][u4][u3][bin4][bin3][1].conj()
                                    summand_t2 += 4*(A_ad_lms[ii][u2][u1][bin2][bin1][1].conj()*A_ad_lms[ii][u4][u3][bin4][bin3][0]+A_ad_lms[ii][u2][u1][bin2][bin1][0]*A_ad_lms[ii][u4][u3][bin4][bin3][1].conj())
                                    summand_t2 += self.A_aa_lms[ii][u2][u1][bin2][bin1][1].conj()*A_dd_lms[u4][u3][bin4][bin3][0]+self.A_aa_lms[ii][u2][u1][bin2][bin1][0]*A_dd_lms[u4][u3][bin4][bin3][1].conj()
                                    # second set of fields
                                    summand_t2 += A_dd_lms[u2][u1][bin2][bin1][1].conj()*self.A_bb_lms[ii][u4][u3][bin4][bin3][0]+A_dd_lms[u2][u1][bin2][bin1][0]*self.A_bb_lms[ii][u4][u3][bin4][bin3][1].conj()
                                    summand_t2 += 4*(A_bd_lms[ii][u2][u1][bin2][bin1][1].conj()*A_bd_lms[ii][u4][u3][bin4][bin3][0]+A_bd_lms[ii][u2][u1][bin2][bin1][0]*A_bd_lms[ii][u4][u3][bin4][bin3][1].conj())
                                    summand_t2 += self.A_bb_lms[ii][u2][u1][bin2][bin1][1].conj()*A_dd_lms[u4][u3][bin4][bin3][0]+self.A_bb_lms[ii][u2][u1][bin2][bin1][0]*A_dd_lms[u4][u3][bin4][bin3][1].conj()
                                summand_t2 = self.base.m_weight*summand_t2/self.N_it/4.

                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL): continue
                                if not self._check_bin(bin3,bin4,binL): continue

                                # Assemble numerators for both parities
                                chi_index = 0
                                for chi in self.chi_arr:
                                    if p_u*chi==-1:
                                        summand_t4_real = 1.0j*summand_t4
                                        if include_disconnected_term: summand_t2_real = 1.0j*summand_t2
                                    else:
                                        summand_t4_real = summand_t4
                                        if include_disconnected_term: summand_t2_real = summand_t2
                                    t4_num[chi_index*self.N_t//2+index] = np.real(np.sum(summand_t4_real*self.ell_bins[binL]))
                                    if include_disconnected_term:
                                        t2_num[chi_index*self.N_t//2+index] = -np.real(np.sum(summand_t2_real*self.ell_bins[binL]))
                                    chi_index += 1
                                index += 1
        if include_disconnected_term:
            t_num = (t4_num+t2_num+self.t0_num)/self.sym_factor
        else:
            t_num = t4_num/self.sym_factor

        return t_num
    
    def compute_fisher_contribution(self, seed, verb=False):
        """
        This computes the contribution to the Fisher matrix from a single pair of GRF simulations, created internally.
        """
        
        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()

        # Initialize output
        fish = np.zeros((self.N_t,self.N_t),dtype='complex')

        if verb: print("# Generating GRFs")
        a_maps = []
        for ii in range(2):
            # Compute two random realizations with known power spectrum and weight appropriately
            if self.ones_mask:
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e9), output_type='harmonic'))
            else:
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e9)))
                
        # Define Q map code
        def compute_Q4(weighting):
            """Compute Q4 map for a given choice of weighting (S or A)."""

            if weighting=='Sinv':
                weighting_function = self.applySinv
            elif weighting=='Ainv':
                weighting_function = self.base.applyAinv

            # Weight maps appropriately
            if verb: print("Weighting maps")
            if self.ones_mask:
                WUinv_a_lms = [weighting_function(a_lm, input_type='harmonic', output_type='harmonic') for a_lm in a_maps]
            else:
                WUinv_a_lms = [self.base.to_lm(self.mask*weighting_function(a)) for a in a_maps]

            # Compute (+-1)H maps
            if verb: print("Creating H maps")
            H_maps = [self._compute_H_maps(WUinv_a_lm) for WUinv_a_lm in WUinv_a_lms]
            
            # Compute A_lms for each pair of fields
            if verb: print("Computing A-lm fields")
            A_lms11 = self._compute_A_lms(H_maps[0])
            A_lms22 = self._compute_A_lms(H_maps[1])
            A_lms12 = self._compute_A_lms(H_maps[0],H_maps[1])
            
            # Compute A_maps for each pair of fields and L-bins
            if verb: print("Computing A-maps")
            A_maps11 = self._compute_A_maps(A_lms11)
            A_maps22 = self._compute_A_maps(A_lms22)
            A_maps12 = self._compute_A_maps(A_lms12)
            
            # Compute (H[x]A[y,z])_lm for each triplet of fields and bins
            if verb: print("Computing (H A)_lm fields")
            HA_lms111 = self._compute_HA_lms(H_maps[0], A_maps11)
            HA_lms122 = self._compute_HA_lms(H_maps[0], A_maps22)
            HA_lms112 = self._compute_HA_lms(H_maps[0], A_maps12)
            HA_lms211 = self._compute_HA_lms(H_maps[1], A_maps11)
            HA_lms222 = self._compute_HA_lms(H_maps[1], A_maps22)
            HA_lms212 = self._compute_HA_lms(H_maps[1], A_maps12)
            
            def _assemble_Q_maps(HxAyz_lms, HyAxz_lms, HzAxy_lms, output_weighting=None):
                """
                Assemble and return the Q4 maps in real- or harmonic-space, given H and A maps. This computes maps with chi = +1 and -1 if parity='both'.

                Schematically Q ~ (H[x]A[y,z]_lm + perms., and we input each permutation of (H A)_lm.

                We optionally assert symmetry in the A map under field interchange.

                The outputs are either Q(b) or WS^-1WQ(b).
                """
                # Define array
                Q_maps = np.zeros((self.N_t,len(a_maps[0].ravel())),dtype='complex')

                # Iterate over fields and bins
                index = -1
                for u in self.fields:
                    u1, u2, u3, u4 = [self.base.indices[u[i]] for i in range(4)]
                    p_u = np.product([self.base.parities[u[i]] for i in range(4)])

                    # Iterate over bins satisfying relevant conditions
                    for bin1 in range(self.Nl):
                        for bin2 in range(self.Nl_squeeze):
                            if u1==u2 and bin2<bin1: continue
                            for bin3 in range(self.Nl):
                                if u1==u3 and u2==u4 and bin3<bin1: continue
                                for bin4 in range(self.Nl_squeeze):
                                    if u3==u4 and bin4<bin3: continue
                                    if u1==u3 and u2==u4 and bin1==bin3 and bin4<bin2: continue
                                    for binL in range(self.NL):
                                        # skip bins outside the triangle conditions
                                        if not self._check_bin(bin1,bin2,binL): continue
                                        if not self._check_bin(bin3,bin4,binL): continue
                                        index += 1

                                        # Create harmonic-space Q^X_lm maps
                                        tmp_Q = np.zeros((1+2*self.pol,2,len(WUinv_a_lms[0].ravel())),dtype='complex')

                                        ## Add all permutations (noting that we have already symmetrized over the two indices of A)
                                        for HA_lms in [HxAyz_lms, HyAxz_lms, HzAxy_lms]:
                                            tmp_Q[u1] -= 1./self.sym_factor[index]*self.ell_bins[bin1]*self.beam_lm*HA_lms[u2][u4][u3][bin2][bin4][bin3][binL].conj()
                                            tmp_Q[u2] -= 1./self.sym_factor[index]*self.ell_bins[bin2]*self.beam_lm*HA_lms[u1][u4][u3][bin1][bin4][bin3][binL].conj()
                                            tmp_Q[u3] -= 1./self.sym_factor[index]*self.ell_bins[bin3]*self.beam_lm*HA_lms[u4][u2][u1][bin4][bin2][bin1][binL].conj()
                                            tmp_Q[u4] -= 1./self.sym_factor[index]*self.ell_bins[bin4]*self.beam_lm*HA_lms[u3][u2][u1][bin3][bin2][bin1][binL].conj()
                                            
                                        # Define chi = +- 1 pieces
                                        tmp_Q_p = tmp_Q[:,0]+p_u*tmp_Q[:,1] # chi = 1
                                        tmp_Q_m = tmp_Q[:,0]-p_u*tmp_Q[:,1] # chi = -1

                                        # Add imaginary parts if necessary
                                        if p_u==1:
                                            tmp_Q_m *= 1.0j
                                        else:
                                            tmp_Q_p *= 1.0j

                                        # Optionally apply weighting and add to output arrays
                                        if weighting=='Ainv':
                                            if self.ones_mask:
                                                if self.parity=='even' or 'both': Q_maps[index] = self.applySinv(tmp_Q_p,input_type='harmonic',output_type='harmonic').ravel()
                                                if self.parity=='both': Q_maps[index+self.N_t//2] = self.applySinv(tmp_Q_m,input_type='harmonic',output_type='harmonic').ravel()
                                                if self.parity=='odd': Q_maps[index] = self.applySinv(tmp_Q_m,input_type='harmonic',output_type='harmonic').ravel()
                                            else:
                                                if self.parity=='even' or 'both': Q_maps[index] = (self.mask*self.applySinv(self.mask*self.base.to_map(tmp_Q_p))).ravel()
                                                if self.parity=='both': Q_maps[index+self.N_t//2] = (self.mask*self.applySinv(self.mask*self.base.to_map(tmp_Q_m))).ravel()
                                                if self.parity=='odd': Q_maps[index] = (self.mask*self.applySinv(self.mask*self.base.to_map(tmp_Q_m))).ravel()
                                        elif weighting=='Sinv':
                                            if self.ones_mask:
                                                if self.parity=='even' or 'both': Q_maps[index] = (self.base.m_weight*tmp_Q_p).ravel()
                                                if self.parity=='both': Q_maps[index+self.N_t//2] = (self.base.m_weight*tmp_Q_m).ravel()
                                                if self.parity=='odd': Q_maps[index] = (self.base.m_weight*tmp_Q_m).ravel()
                                            else:
                                                if self.parity=='even' or 'both': Q_maps[index] = self.base.A_pix*self.base.to_map(tmp_Q_p).ravel()
                                                if self.parity=='both': Q_maps[index+self.N_t//2] = self.base.A_pix*self.base.to_map(tmp_Q_m).ravel()
                                                if self.parity=='odd': Q_maps[index] = self.base.A_pix*self.base.to_map(tmp_Q_m).ravel()
                return Q_maps

            # Compute pairs of H-maps and A-maps
            if verb: print("Computing Q(b) maps")
            Q_maps111 = _assemble_Q_maps(HA_lms111, HA_lms111, HA_lms111)
            Q_maps222 = _assemble_Q_maps(HA_lms222, HA_lms222, HA_lms222)
            Q_maps112 = _assemble_Q_maps(HA_lms112, HA_lms112, HA_lms211)
            Q_maps122 = _assemble_Q_maps(HA_lms122, HA_lms212, HA_lms212)
            
            return Q_maps111, Q_maps222, Q_maps112, Q_maps122

        # Compute Q4 maps
        if verb: print("\n# Computing Q4 map for S^-1 weighting")
        Q4_Sinv = compute_Q4('Sinv')
        if verb: print("\n# Computing Q4 map for A^-1 weighting")
        Q4_Ainv = compute_Q4('Ainv')
        
        # Assemble Fisher matrix
        if verb: print("\n# Assembling Fisher matrix\n")
        
        # Compute Fisher matrix as an outer product
        fish += (Q4_Sinv[0].conj())@(Q4_Ainv[0].T)
        fish += (Q4_Sinv[1].conj())@(Q4_Ainv[1].T)
        fish += 9.*(Q4_Sinv[2].conj())@(Q4_Ainv[2].T)
        fish += 9.*(Q4_Sinv[3].conj())@(Q4_Ainv[3].T)
        fish -= 6.*(Q4_Sinv[0].conj())@(Q4_Ainv[3].T)
        fish -= 6.*(Q4_Sinv[1].conj())@(Q4_Ainv[2].T)
        
        fish = fish.conj()/24./48.
        
        return fish.real


    def compute_fisher(self, N_it, N_cpus=1, verb=False):
        """
        Compute the Fisher matrix using N_it realizations. If N_cpus > 1, this parallelizes the operations (though HEALPix is already parallelized so the speed-up is not particularly significant).

        For high-dimensional problems, it is usually preferred to split the computation across a cluster with MPI, calling compute_fisher_contribution for each instead of this function.
        """

        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Initialize output
        fish = np.zeros((self.N_t,self.N_t))
        
        global _iterable
        def _iterable(seed):
            return self.compute_fisher_contribution(seed, verb=verb)
        
        if N_cpus==1:
            for seed in range(N_it):
                if seed%5==0: print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
                fish += self.compute_fisher_contribution(seed, verb=verb*(seed==0))/N_it
        else:
            p = mp.Pool(N_cpus)
            print("Computing Fisher contribution from %d pairs of Monte Carlo simulations on %d threads"%(N_it, N_cpus))
            all_fish = list(tqdm.tqdm(p.imap_unordered(_iterable,np.arange(N_it)),total=N_it))
            fish = np.sum(all_fish,axis=0)/N_it
        
        # Store matrix in class attributes
        self.fish = fish
        self.inv_fish = np.linalg.inv(fish)
        
        return fish

    def Tl_unwindowed(self, data, fish=[], include_disconnected_term=True, verb=False):
        """
        Compute the unwindowed trispectrum estimator for all combinations of fields.
        
        The code either uses pre-computed Fisher matrices or reads them in on input. 
        
        Note that we return the imaginary part of odd-parity trispectra.

        We can also optionally switch off the disconnected terms.
        """
        if verb: print("\n")

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
        index, config_index = 0,0
        chi_name = {1:'+',-1:'-'}
        for chi in self.chi_arr:

            # Iterate over fields
            for u in self.fields:
                Tl_dict['%s'%u+'%s'%chi_name[chi]] = Tl_out[index:index+len(self.sym_factor_all[config_index])]
                index += len(self.sym_factor_all[config_index])
                config_index += 1

        return Tl_dict

    ### IDEAL ESTIMATOR
    def Tl_numerator_ideal(self, data, verb=False, include_disconnected_term=True):
        """
        Compute the numerator of the idealized trispectrum estimator. We normalize by < mask^4 >.

        We can also optionally switch off the disconnected terms, which affects only parity-conserving trispectra.
        """
        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Transform to harmonic space and normalize data by 1/C_th
        Cinv_data_lm = np.einsum('ijk,jk->ik',self.base.inv_Cl_lm_mat,self.base.to_lm(data),order='C')

        # Compute (+-1)H maps
        if verb: print("\nComputing H maps")
        H_pm1_maps = self._compute_H_maps(Cinv_data_lm)

        # Define array of A maps
        # Each element contains [A, \bar{A}]
        if verb: print("Computing A_lm fields")
        A_lms = self._compute_A_lms(H_pm1_maps)
        
        # Compute empirical power spectrum estimates  
        if include_disconnected_term:
            if verb: print("Computing empirical power spectra")
            Cinv_l_empirical = []
            for u1 in range(1+2*self.pol):
                Cinv_l_empirical1 = []
                for u2 in range(u1+1):
                    Cinv_data_lm_sq = 0.5*(Cinv_data_lm[u1]*np.conj(Cinv_data_lm[u2])+Cinv_data_lm[u2]*np.conj(Cinv_data_lm[u1]))*self.base.m_weight
                    Cinv_l_empirical1.append([np.real(np.sum(Cinv_data_lm_sq[self.base.l_arr==l]*self.beam[l]**2)/(2*l+1)) for l in range(np.max(self.l_bins_squeeze))])
                Cinv_l_empirical.append(Cinv_l_empirical1)            
        
        # Define 4-, 2- and 0-field arrays
        t4_num_ideal = np.zeros(self.N_t)
        if not include_disconnected_term:
            print("## No subtraction of (parity-conserving) disconnected terms performed!")
        else:
            t2_num_ideal = np.zeros(self.N_t)
            t0_num_ideal = np.zeros(self.N_t)
        if verb: print("Assembling trispectrum numerator")
        
        # Iterate over fields
        index = 0
        for u in self.fields:
            u1, u2, u3, u4 = [self.base.indices[u[i]] for i in range(4)]
            p_u = np.product([self.base.parities[u[i]] for i in range(4)])
            if verb: print("Analyzing trispectrum numerator for field %s"%u)

            # Iterate over bins satisfying relevant conditions
            for bin1 in range(self.Nl):
                for bin2 in range(self.Nl_squeeze):
                    if u1==u2 and bin2<bin1: continue
                    for bin3 in range(self.Nl):
                        if u1==u3 and u2==u4 and bin3<bin1: continue
                        for bin4 in range(self.Nl_squeeze):
                            if u3==u4 and bin4<bin3: continue
                            if u1==u3 and u2==u4 and bin1==bin3 and bin4<bin2: continue

                            # Compute summands
                            summand = self.base.m_weight*(A_lms[u2][u1][bin2][bin1][1].conj()*A_lms[u4][u3][bin4][bin3][0]+A_lms[u2][u1][bin2][bin1][0]*A_lms[u4][u3][bin4][bin3][1].conj())/2.
                            
                            # Iterate over L bins
                            for binL in range(self.NL):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL): continue
                                if not self._check_bin(bin3,bin4,binL): continue
                                
                                # Compute four-field term
                                chi_index = 0
                                for chi in self.chi_arr:
                                    if p_u*chi==-1:
                                        summand2 = 1.0j*summand
                                    else:
                                        summand2 = summand     
                                    t4_num_ideal[chi_index*self.N_t//2+index] = np.real(np.sum(summand2*self.ell_bins[binL]))
                                    chi_index += 1
                                    
                                # Compute disconnected terms
                                if include_disconnected_term:
                                    # Check if two external bins are equal (if not, no contribution to 2- and 0-field terms)
                                    if ((bin1==bin3)*(bin2==bin4)+(bin1==bin4)*(bin2==bin3))==0:
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
                                                if tjs==0: continue
                                                pref = (2*L+1.)*(2*l1+1.)*(2*l2+1.)/(4.*np.pi)*tjs*(-1.)**(l1+l2+L)
                                                
                                                # 2-field contribution
                                                value2 -= pref*Cinv_l_empirical[max([u2,u4])][min([u2,u4])][l2]*self.base.inv_Cl_mat[u3,u1][l1]*self.beam[l1]**2*(bin1==bin3)*(bin2==bin4)
                                                value2 -= pref*Cinv_l_empirical[max([u1,u3])][min([u1,u3])][l1]*self.base.inv_Cl_mat[u4,u2][l2]*self.beam[l2]**2*(bin1==bin3)*(bin2==bin4)
                                                value2 -= pref*Cinv_l_empirical[max([u2,u3])][min([u2,u3])][l2]*self.base.inv_Cl_mat[u4,u1][l1]*self.beam[l1]**2*(bin1==bin4)*(bin2==bin3)
                                                value2 -= pref*Cinv_l_empirical[max([u1,u4])][min([u1,u4])][l1]*self.base.inv_Cl_mat[u3,u2][l2]*self.beam[l2]**2*(bin1==bin4)*(bin2==bin3)
                                                # 0-field contribution
                                                value0 += pref*self.base.inv_Cl_mat[u3,u1][l1]*self.base.inv_Cl_mat[u4,u2][l2]*self.beam[l1]**2*self.beam[l2]**2*(bin1==bin3)*(bin2==bin4)
                                                value0 += pref*self.base.inv_Cl_mat[u4,u1][l1]*self.base.inv_Cl_mat[u3,u2][l2]*self.beam[l1]**2*self.beam[l2]**2*(bin1==bin4)*(bin2==bin3)
                                        t2_num_ideal[index] = value2
                                        t0_num_ideal[index] = value0

                                index += 1
        
        if include_disconnected_term:
            t_num_ideal = (t4_num_ideal/np.mean(self.mask**4.)+t2_num_ideal/np.mean(self.mask**2.)+t0_num_ideal)/self.sym_factor
        else:
            t_num_ideal = t4_num_ideal/np.mean(self.mask**4.)/self.sym_factor
        
        # Save t0 array for posterity
        if include_disconnected_term: self.t0_num_ideal = t0_num_ideal
            
        return t_num_ideal

    def compute_fisher_ideal(self, verb=False, N_cpus=1, diagonal=False):
        """
        This computes the idealized Fisher matrix for the trispectrum, including cross-correlations between fields. If N_cpus > 1, this parallelizes the operation.

        We can optionally drop any off-diagonal terms in the ideal Fisher matrix and restrict the internal L values. This is not recommended for general usage, but may be useful for forecasting.
        """
        
        # Compute symmetry factors, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        if diagonal:
            print("\n## Caution: dropping off-diagonal terms in the Fisher matrix!\n")

        global _iterator
        def _iterator(index_input, verb=False):
            """Create an iterator for multiprocessing. This iterates over the first index."""
            
            fish = np.zeros((self.N_t, self.N_t))
            
            if verb and (index_input%10)==0: print("Computing Fisher matrix row %d of %d"%(index_input+1,self.N_t))
        
            # Iterate over first set of fields, parities, and bins
            index = -1
            for u in self.fields:
                u1, u2, u3, u4 = [self.base.indices[u[i]] for i in range(4)]
                p_u = np.product([self.base.parities[u[i]] for i in range(4)])
                
                # Iterate over bins satisfying relevant conditions
                for bin1 in range(self.Nl):
                    for bin2 in range(self.Nl_squeeze):
                        if u1==u2 and bin2<bin1: continue
                        for bin3 in range(self.Nl):
                            if u1==u3 and u2==u4 and bin3<bin1: continue
                            for bin4 in range(self.Nl_squeeze):
                                if u3==u4 and bin4<bin3: continue
                                if u1==u3 and u2==u4 and bin1==bin3 and bin4<bin2: continue
                                for binL in range(self.NL):
                                    # skip bins outside the triangle conditions
                                    if not self._check_bin(bin1,bin2,binL): continue
                                    if not self._check_bin(bin3,bin4,binL): continue

                                    # Update indices 
                                    index += 1
                                    
                                    # Specialize to only the desired index
                                    if index!=index_input: continue

                                    # Iterate over second set of fields, parities, and bins
                                    index_p = -1
                                    for u_p in self.fields:
                                        u1_p, u2_p, u3_p, u4_p = [self.base.indices[u_p[i]] for i in range(4)]
                                        p_u_p = np.product([self.base.parities[u_p[i]] for i in range(4)])

                                        # Iterate over bins satisfying relevant conditions
                                        for bin1_p in range(self.Nl):
                                            for bin2_p in range(self.Nl_squeeze):
                                                if u1_p==u2_p and bin2_p<bin1_p: continue
                                                for bin3_p in range(self.Nl):
                                                    if u1_p==u3_p and u2_p==u4_p and bin3_p<bin1_p: continue
                                                    for bin4_p in range(self.Nl_squeeze):
                                                        if u3_p==u4_p and bin4_p<bin3_p: continue
                                                        if u1_p==u3_p and u2_p==u4_p and bin1_p==bin3_p and bin4_p<bin2_p: continue
                                                        for binL_p in range(self.NL):
                                                            # skip bins outside the triangle conditions
                                                            if not self._check_bin(bin1_p,bin2_p,binL_p): continue
                                                            if not self._check_bin(bin3_p,bin4_p,binL_p): continue

                                                            # Update indices
                                                            index_p += 1
                                                            
                                                            # fill in this part by symmetry!
                                                            if index_p<index: continue

                                                            if index_p!=index and diagonal: continue

                                                            # Check if two sets of bins are permutations of each other (no contribution else!)
                                                            b1234 = np.sort([bin1,bin2,bin3,bin4])
                                                            b1234_p = np.sort([bin1_p,bin2_p,bin3_p,bin4_p])
                                                            if not (b1234==b1234_p).all():
                                                                continue

                                                            ## Check that at least one term contributes!
                                                            perm  = (bin1==bin1_p)*(bin2==bin2_p)*(bin3==bin3_p)*(bin4==bin4_p)*(binL==binL_p)
                                                            perm += (bin1==bin2_p)*(bin2==bin1_p)*(bin3==bin3_p)*(bin4==bin4_p)*(binL==binL_p)
                                                            perm += (bin1==bin1_p)*(bin2==bin2_p)*(bin3==bin4_p)*(bin4==bin3_p)*(binL==binL_p)
                                                            perm += (bin1==bin2_p)*(bin2==bin1_p)*(bin3==bin4_p)*(bin4==bin3_p)*(binL==binL_p)
                                                            perm += (bin1==bin3_p)*(bin2==bin4_p)*(bin3==bin1_p)*(bin4==bin2_p)*(binL==binL_p)
                                                            perm += (bin1==bin4_p)*(bin2==bin3_p)*(bin3==bin1_p)*(bin4==bin2_p)*(binL==binL_p)
                                                            perm += (bin1==bin3_p)*(bin2==bin4_p)*(bin3==bin2_p)*(bin4==bin1_p)*(binL==binL_p)
                                                            perm += (bin1==bin4_p)*(bin2==bin3_p)*(bin3==bin2_p)*(bin4==bin1_p)*(binL==binL_p)
                                                            perm += (bin1==bin1_p)*(bin2==bin3_p)*(bin3==bin2_p)*(bin4==bin4_p)
                                                            perm += (bin1==bin2_p)*(bin2==bin3_p)*(bin3==bin1_p)*(bin4==bin4_p)
                                                            perm += (bin1==bin1_p)*(bin2==bin4_p)*(bin3==bin2_p)*(bin4==bin3_p)
                                                            perm += (bin1==bin2_p)*(bin2==bin4_p)*(bin3==bin1_p)*(bin4==bin3_p)
                                                            perm += (bin1==bin3_p)*(bin2==bin1_p)*(bin3==bin4_p)*(bin4==bin2_p)
                                                            perm += (bin1==bin3_p)*(bin2==bin2_p)*(bin3==bin4_p)*(bin4==bin1_p)
                                                            perm += (bin1==bin4_p)*(bin2==bin1_p)*(bin3==bin3_p)*(bin4==bin2_p)
                                                            perm += (bin1==bin4_p)*(bin2==bin2_p)*(bin3==bin3_p)*(bin4==bin1_p)
                                                            perm += (bin1==bin1_p)*(bin2==bin4_p)*(bin3==bin3_p)*(bin4==bin2_p)
                                                            perm += (bin1==bin2_p)*(bin2==bin4_p)*(bin3==bin3_p)*(bin4==bin1_p)
                                                            perm += (bin1==bin1_p)*(bin2==bin3_p)*(bin3==bin4_p)*(bin4==bin2_p)
                                                            perm += (bin1==bin2_p)*(bin2==bin3_p)*(bin3==bin4_p)*(bin4==bin1_p)
                                                            perm += (bin1==bin3_p)*(bin2==bin2_p)*(bin3==bin1_p)*(bin4==bin4_p)
                                                            perm += (bin1==bin3_p)*(bin2==bin1_p)*(bin3==bin2_p)*(bin4==bin4_p)
                                                            perm += (bin1==bin4_p)*(bin2==bin2_p)*(bin3==bin1_p)*(bin4==bin3_p)
                                                            perm += (bin1==bin4_p)*(bin2==bin1_p)*(bin3==bin2_p)*(bin4==bin3_p)
                                                            if perm==0: continue
                                                            
                                                            value = 0.+0.j

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

                                                                                # Define a factor of 1 or i to parity to separate
                                                                                if (-1)**(l1+l2+l3+l4)==1: fac = 1.0
                                                                                else: fac = 1.0j

                                                                                # second 3j symbols with spin (-1, -1, 2)
                                                                                tj1234 = tj12*self.threej(l3,l4,L)
                                                                                if tj1234==0: continue

                                                                                norm = (2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)/(4.*np.pi)**2*self.beam[l1]**2*self.beam[l2]**2*self.beam[l3]**2*self.beam[l4]**2

                                                                                Cinv_bin = lambda i,j,l: self.base.inv_Cl_mat[[u1,u2,u3,u4][i],[u1_p,u2_p,u3_p,u4_p][j]][l]*([bin1,bin2,bin3,bin4][i]==[bin1_p,bin2_p,bin3_p,bin4_p][j])

                                                                                ## Add first permutation
                                                                                # Compute combinations of covariances
                                                                                inv_cov4  = Cinv_bin(0,0,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,2,l3)*Cinv_bin(3,3,l4)
                                                                                inv_cov4 += Cinv_bin(0,1,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,2,l3)*Cinv_bin(3,3,l4)
                                                                                inv_cov4 += Cinv_bin(0,0,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,3,l3)*Cinv_bin(3,2,l4)
                                                                                inv_cov4 += Cinv_bin(0,1,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,3,l3)*Cinv_bin(3,2,l4)
                                                                                inv_cov4 += Cinv_bin(0,2,l1)*Cinv_bin(1,3,l2)*Cinv_bin(2,0,l3)*Cinv_bin(3,1,l4)
                                                                                inv_cov4 += Cinv_bin(0,3,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,0,l3)*Cinv_bin(3,1,l4)
                                                                                inv_cov4 += Cinv_bin(0,2,l1)*Cinv_bin(1,3,l2)*Cinv_bin(2,1,l3)*Cinv_bin(3,0,l4)
                                                                                inv_cov4 += Cinv_bin(0,3,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,1,l3)*Cinv_bin(3,0,l4)
                                                                                if inv_cov4!=0: value += fac*norm*tj1234**2*inv_cov4*(binL==binL_p)

                                                                                # Finish if we don't want off-diagonal contributions
                                                                                if diagonal: continue

                                                                                # Iterate over L' for off-diagonal terms
                                                                                for L_p in range(self.L_bins[binL_p],self.L_bins[binL_p+1]):

                                                                                    # Impose 6j symmetries
                                                                                    if L_p<abs(l3-l4) or L_p>l3+l4: continue
                                                                                    if L_p<abs(l1-l2) or L_p>l1+l2: continue

                                                                                    # Compute 3j symbols if non-zero
                                                                                    if L_p>=abs(l1-l3) and L_p<=l1+l3 and L_p>=abs(l2-l4) and L_p<=l2+l4: 
                                                                                        tj1324 = self.threej(l1,l3,L_p)*self.threej(l2,l4,L_p)
                                                                                    else:
                                                                                        tj1324 = 0
                                                                                    if L_p>=abs(l1-l4) and L_p<=l1+l4 and L_p>=abs(l2-l3) and L_p<=l2+l3:
                                                                                        tj1432 = self.threej(l1,l4,L_p)*self.threej(l3,l2,L_p)
                                                                                    else:
                                                                                        tj1432 = 0

                                                                                    ## Add second permutation
                                                                                    if tj1324!=0: 
                                                                                        inv_cov4  = Cinv_bin(0,0,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,1,l3)*Cinv_bin(3,3,l4)
                                                                                        inv_cov4 += Cinv_bin(0,1,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,0,l3)*Cinv_bin(3,3,l4)
                                                                                        inv_cov4 += Cinv_bin(0,0,l1)*Cinv_bin(1,3,l2)*Cinv_bin(2,1,l3)*Cinv_bin(3,2,l4)
                                                                                        inv_cov4 += Cinv_bin(0,1,l1)*Cinv_bin(1,3,l2)*Cinv_bin(2,0,l3)*Cinv_bin(3,2,l4)
                                                                                        inv_cov4 += Cinv_bin(0,2,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,3,l3)*Cinv_bin(3,1,l4)
                                                                                        inv_cov4 += Cinv_bin(0,2,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,3,l3)*Cinv_bin(3,0,l4)
                                                                                        inv_cov4 += Cinv_bin(0,3,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,2,l3)*Cinv_bin(3,1,l4)
                                                                                        inv_cov4 += Cinv_bin(0,3,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,2,l3)*Cinv_bin(3,0,l4)
                                                                                        if inv_cov4!=0: value += fac*norm*(2*L_p+1)*(-1.)**(l2+l3)*tj1234*tj1324*self.sixj(L,l1,l2,L_p,l4,l3)*inv_cov4

                                                                                    ## Add third permutation
                                                                                    if tj1432!=0:
                                                                                        inv_cov4  = Cinv_bin(0,0,l1)*Cinv_bin(1,3,l2)*Cinv_bin(2,2,l3)*Cinv_bin(3,1,l4)
                                                                                        inv_cov4 += Cinv_bin(0,1,l1)*Cinv_bin(1,3,l2)*Cinv_bin(2,2,l3)*Cinv_bin(3,0,l4)
                                                                                        inv_cov4 += Cinv_bin(0,0,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,3,l3)*Cinv_bin(3,1,l4)
                                                                                        inv_cov4 += Cinv_bin(0,1,l1)*Cinv_bin(1,2,l2)*Cinv_bin(2,3,l3)*Cinv_bin(3,0,l4)
                                                                                        inv_cov4 += Cinv_bin(0,2,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,0,l3)*Cinv_bin(3,3,l4)
                                                                                        inv_cov4 += Cinv_bin(0,2,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,1,l3)*Cinv_bin(3,3,l4)
                                                                                        inv_cov4 += Cinv_bin(0,3,l1)*Cinv_bin(1,1,l2)*Cinv_bin(2,0,l3)*Cinv_bin(3,2,l4)
                                                                                        inv_cov4 += Cinv_bin(0,3,l1)*Cinv_bin(1,0,l2)*Cinv_bin(2,1,l3)*Cinv_bin(3,2,l4)
                                                                                        if inv_cov4!=0: value += fac*norm*(2*L_p+1)*(-1.)**(L+L_p)*tj1234*tj1432*self.sixj(L,l1,l2,L_p,l3,l4)*inv_cov4

                                                            # Reconstruct output for even / odd ell and note symmetric matrix!
                                                            for chi_index,chi in enumerate(self.chi_arr):
                                                                for chi_index_p, chi_p in enumerate(self.chi_arr):
                                                                    if p_u*chi!=p_u_p*chi_p: continue
                                                                    if p_u*chi==1:
                                                                        out_value = value.real
                                                                    else:
                                                                        out_value = value.imag
                                                                    fish[chi_index*self.N_t//2+index, chi_index_p*self.N_t//2+index_p] = out_value
                                                                    fish[chi_index_p*self.N_t//2+index_p, chi_index*self.N_t//2+index] = out_value
                                                                    
            return fish
    
        # Assemble matrix, multiprocessing if necessary
        if N_cpus==1:
            fish = np.zeros((self.N_t, self.N_t))
            for i in range(self.N_t):
                fish += _iterator(i,verb=verb)
        else:
            p = mp.Pool(N_cpus)
            if verb: print("Multiprocessing computation on %d cores"%N_cpus)

            result = list(tqdm.tqdm(p.imap_unordered(_iterator,range(self.N_t)),total=self.N_t))
            fish = np.sum(result,axis=0)
        
        if verb: print("Fisher matrix computation complete\n")

        # Add symmetry factors and save attributes
        fish *= 1./np.outer(self.sym_factor,self.sym_factor)
        self.fish_ideal = fish
        self.inv_fish_ideal = np.linalg.inv(fish)

        return fish

    def Tl_ideal(self, data, fish_ideal=[], verb=False, include_disconnected_term=True, N_cpus=1):
        """
        Compute the idealized trispectrum estimator, including normalization, if not supplied or already computed. Note that this normalizes by < mask^4 >.
        
        We can also optionally switch off the disconnected terms. This only affects the parity-even trispectrum.
        
        The N_cpus parameter specifies how many CPUs to use in computation of the ideal Fisher matrix.
        """
        if verb: print("\n")

        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)

        if not hasattr(self,'inv_fish_ideal'):
            print("Computing ideal Fisher matrix")
            self.compute_fisher_ideal(verb=verb, N_cpus=N_cpus)

        # Compute numerator
        Tl_num_ideal = self.Tl_numerator_ideal(data, verb=verb, include_disconnected_term=include_disconnected_term)

        # Apply normalization
        Tl_out = np.matmul(self.inv_fish_ideal,Tl_num_ideal)

        # Create output dictionary
        Tl_dict = {}
        index, config_index = 0,0
        chi_name = {1:'+',-1:'-'}
        for chi in self.chi_arr:
            # Iterate over fields
            for u in self.fields:
                Tl_dict['%s'%u+'%s'%chi_name[chi]] = Tl_out[index:index+len(self.sym_factor_all[config_index])]
                index += len(self.sym_factor_all[config_index])
                config_index += 1

        return Tl_dict