### Code for ideal and unwindowed binned polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022)
## This module contains the parity-odd and parity-even trispectrum estimation code

import healpy
import numpy as np
import multiprocessing as mp
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
import pywigxjpf as wig

class TSpec():
    """Trispectrum estimation class. This takes the binning strategy as input and a base class. 
    We also feed in a function that applies the S^-1 operator in real space.
    
    Inputs:
    - base: PolyBin class
    - mask: HEALPix mask to deconvolve
    - applySinv: function which returns S^-1 applied to a given input map
    - min_l, dl, Nl: binning parameters
    - include_partial_triangles: whether to include triangles (in l1,l2,L or l3,l4,L) whose centers don't satisfy the triangle conditions. (Default: False)
    """
    def __init__(self, base, mask, applySinv, min_l, dl, Nl, include_partial_triangles=False):
            # Read in attributes
            self.base = base
            self.mask = mask
            self.applySinv = applySinv
            self.min_l = min_l
            self.dl = dl
            self.Nl = Nl
            self.include_partial_triangles = include_partial_triangles
            
            if min_l+Nl*dl>base.lmax:
                raise Exception("Maximum l is larger than HEALPix resolution!")
            print("Binning: %d bins in [%d, %d]"%(Nl,min_l,min_l+Nl*dl))
            
            # Define l filters
            self.ell_bins = [(self.base.l_arr>=self.min_l+self.dl*bin1)&(self.base.l_arr<self.min_l+self.dl*(bin1+1)) for bin1 in range(self.Nl)]
            self.phase_factor = (-1.)**self.base.l_arr

            # Define m weights (for complex conjugates)
            self.m_weight = (1.+1.*(self.base.m_arr>0.))

            # Define wigner calculator
            wig.wig_table_init(self.base.lmax*2,9)
            wig.wig_temp_init(self.base.lmax*2)

            # Define 3j with specified spins, and 6j
            self.threej = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,-1,-1,2)
            self.sixj = lambda l1,l2,l3,l4,l5,l6: wig.wig6jj(2*l1,2*l2,2*l3,2*l4,2*l5,2*l6)

    def _check_bin(self, bin1, bin2, bin3, even=False):
        """Return one if modes in the bin satisfy the triangle conditions, or zero else.

        If even=true, we enforce that the sum of the three ells must be even.

        This is used either for all triangles in the bin, or just the center of the bin.
        """
        if self.include_partial_triangles:
            good = 0
            for l1 in range(self.min_l+bin1*self.dl,self.min_l+(bin1+1)*self.dl):
                for l2 in range(self.min_l+bin2*self.dl,self.min_l+(bin2+1)*self.dl):
                    for l3 in range(self.min_l+bin3*self.dl,self.min_l+(bin3+1)*self.dl):
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
            l1 = self.min_l+(bin1+0.5)*self.dl
            l2 = self.min_l+(bin2+0.5)*self.dl
            l3 = self.min_l+(bin3+0.5)*self.dl
            if even and ((-1)**(l1+l2+l3)==-1): return 0 
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
            for bin2 in range(bin1,self.Nl):
                for bin3 in range(bin1,self.Nl):
                    for bin4 in range(bin3,self.Nl):
                        if bin1==bin3 and bin4<bin2: continue # note different condition to odd estimator!
                        
                        # Iterate over L bins
                        for binL in range(self.Nl):
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
            for bin2 in range(bin1,self.Nl):
                for bin3 in range(bin1,self.Nl):
                    for bin4 in range(bin3,self.Nl):
                        if bin1==bin3 and bin4<=bin2: continue
                        
                        # Iterate over L bins
                        for binL in range(self.Nl):
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

    def _compute_H(self, h_lm):
        """
        Compute the H^+-(n) map given input field h_lm. This calls the to_map_spin routine with the correct inputs.
        """
        H_plus, H_minus = self.base.to_map_spin(h_lm,-1.0*h_lm,1)
        return [H_plus, -H_minus]

    def _compute_Alm(self, H_maps, bin1, bin2, H2_maps=[]):
        """Compute the A_{b1b2}[x,y](L,M) map given H^+-[x], H^+-[y] and bins. This calls the to_lm_spin routine with the correct inputs.
        
        Note, we can optionally use two different H fields here - this corresponds to allowing for x!=y.
        """

        if len(H2_maps)==0:
            H2_maps = H_maps
        
        A_plus = H_maps[bin1][0]*H2_maps[bin2][0]
        A_minus = H_maps[bin1][1]*H2_maps[bin2][1]
        
        A_plus_lm, A_minus_lm = self.base.to_lm_spin(A_plus, A_minus, 2)
        
        return A_minus_lm.conj()

    ### IDEAL ESTIMATOR
    def Tl_numerator_ideal(self, data, parity='even', verb=False):
        """
        Compute the numerator of the idealized trispectrum estimator. We normalize by < mask^4 >.

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
        
        # Normalize data by C_th and transform to harmonic space
        Cinv_data_lm = self.base.safe_divide(self.base.to_lm(data),self.base.Cl_lm)
        
        # Compute H and H-bar maps
        if verb: print("Computing H^+- maps")
        H_map = [self._compute_H(self.ell_bins[bin1]*Cinv_data_lm) for bin1 in range(self.Nl)]
        Hbar_map = [self._compute_H(self.phase_factor*self.ell_bins[bin1]*Cinv_data_lm) for bin1 in range(self.Nl)]

        # Define array of A maps (restricting to bin2 <= bin1, by symmetry)
        if verb: print("Computing A maps")
        Alm = [[self._compute_Alm(H_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl)]
        Abar_lm = [[self.phase_factor*self._compute_Alm(Hbar_map,bin1,bin2) for bin2 in range(bin1+1)] for bin1 in range(self.Nl)]
        
        # Even parity estimator
        if parity=='even' or parity=='both':

            # Define 4-, 2- and 0-field arrays
            t4_even_num_ideal = np.zeros(self.N_t_even)
            t2_even_num_ideal = np.zeros(self.N_t_even)
            t0_even_num_ideal = np.zeros(self.N_t_even)

            if verb: print("Assembling parity-even trispectrum numerator")
            
            # Compute squared field    
            Cinv_data_lm_sq = np.real(Cinv_data_lm*np.conj(Cinv_data_lm)*self.m_weight)

            index = 0
            # iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>=b2 if b1=b3
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl):
                            if bin1==bin3 and bin4<bin2: continue # note different condition to odd estimator!
                            
                            # Compute summands
                            summand = self.m_weight*np.real(Abar_lm[bin2][bin1].conj()*Alm[bin4][bin3] + Alm[bin2][bin1].conj()*Abar_lm[bin4][bin3])
                                        
                            # Iterate over L bins
                            for binL in range(self.Nl):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue

                                # Compute four-field term
                                t4_even_num_ideal[index] = 1./2.*np.sum(summand*self.ell_bins[binL])

                                # Check if two external bins are equal (if not, no contribution to 2- and 0-field terms)
                                kroneckers = (bin1==bin3)*(bin2==bin4)+(bin1==bin4)*(bin2==bin3)
                                if kroneckers==0:
                                    index += 1
                                    continue

                                # Sum over ells for two- and zero-point terms
                                value2, value0 = 0., 0.
                                for l1 in range(self.min_l+bin1*self.dl,self.min_l+(bin1+1)*self.dl):

                                    # Compute sum over l1
                                    Cinvsq_l1 = np.sum(Cinv_data_lm_sq[self.base.l_arr==l1])

                                    for l2 in range(self.min_l+bin2*self.dl,self.min_l+(bin2+1)*self.dl):

                                        # Compute sum over l2
                                        Cinvsq_l2 = np.sum(Cinv_data_lm_sq[self.base.l_arr==l2])
                                        
                                        for L in range(self.min_l+binL*self.dl,self.min_l+(binL+1)*self.dl):
                                            if L<abs(l1-l2) or L>l1+l2: continue

                                            # define 3j symbols with spin (-1, -1, 2)
                                            tjs = self.threej(l1,l2,L)**2.

                                            # 2-field: (l1, l2) contribution
                                            value2 += -(2.*L+1.)/(4.*np.pi)*tjs*(-1.)**(l1+l2+L)*((2.*l1+1.)*Cinvsq_l2/self.base.Cl[l1]+(2.*l2+1.)*Cinvsq_l1/self.base.Cl[l2])*kroneckers
                                            
                                            # 0-field contribution
                                            value0 += (2.*l1+1.)*(2.*l2+1.)*(2.*L+1.)/(4.*np.pi)*tjs*(-1.)**(l1+l2+L)/self.base.Cl[l1]/self.base.Cl[l2]*kroneckers
                                            
                                t2_even_num_ideal[index] = value2
                                t0_even_num_ideal[index] = value0

                                index += 1

            t_even_num_ideal = (t4_even_num_ideal/np.mean(self.mask**4.)+t2_even_num_ideal/np.mean(self.mask**2.)+t0_even_num_ideal)/self.sym_factor_even
        
        # Odd parity estimator
        if parity=='odd' or parity=='both':
            
            # Define arrays
            t_odd_num_ideal = np.zeros(self.N_t_odd, dtype='complex')
        
            # iterate over bins with b2>=b1, b4>=b3, b3>=b1 and b4>b2 if b1=b3
            if verb: print("Assembling parity-odd trispectrum numerator")
            index = 0
            for bin1 in range(self.Nl):
                for bin2 in range(bin1,self.Nl):
                    for bin3 in range(bin1,self.Nl):
                        for bin4 in range(bin3,self.Nl):
                            if bin1==bin3 and bin4<=bin2: continue
                                        
                            # Compute summands
                            summand = self.m_weight*np.imag(Abar_lm[bin2][bin1].conj()*Alm[bin4][bin3] - Alm[bin2][bin1].conj()*Abar_lm[bin4][bin3])
                            
                            # Iterate over L bins
                            for binL in range(self.Nl):
                                # skip bins outside the triangle conditions
                                if not self._check_bin(bin1,bin2,binL,even=False): continue
                                if not self._check_bin(bin3,bin4,binL,even=False): continue

                                # Compute estimator numerator
                                t_odd_num_ideal[index] = -1.0j/2.*np.sum(summand*self.ell_bins[binL])
                                index += 1

            # Normalize
            t_odd_num_ideal *= 1./self.sym_factor_odd/np.mean(self.mask**4.)

        if parity=='even':
            return t_even_num_ideal
        elif parity=='odd':
            return t_odd_num_ideal
        else:
            return t_even_num_ideal, t_odd_num_ideal

    def fisher_ideal(self, parity='even', verb=False):
        """
        This computes the idealized Fisher matrix for the trispectrum.

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
                
        # Define arrays
        if parity!='even':
            fish_odd = np.zeros((self.N_t_odd, self.N_t_odd))
        if parity!='odd':
            fish_even = np.zeros((self.N_t_even, self.N_t_even))

        # Iterate over first set of bins
        # Note that we use two sets of indices here, since there are a different number of odd and even bins
        index1e = -1
        index1o = -1
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl):
                for bin3 in range(bin1,self.Nl):
                    for bin4 in range(bin3,self.Nl):
                        if bin1==bin3 and bin4<bin2: continue
                        for binL in range(self.Nl):
                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,binL,even=False): continue
                            if not self._check_bin(bin3,bin4,binL,even=False): continue
                            
                            # Update indices
                            index1e += 1
                            if bin2!=bin4: index1o += 1 # no equal bins!
                    
                            if verb and parity!='odd':
                                if (index1e+1)%5==0: print("Computing bin %d of %d"%(index1e+1,self.N_t_even))
                            if verb and parity=='odd':
                                if (index1o+1)%5==0: print("Computing bin %d of %d"%(index1o+1,self.N_t_odd))
                            
                            # Iterate over second set of bins
                            index2e = -1
                            index2o = -1
                            for bin1p in range(self.Nl):
                                for bin2p in range(bin1p,self.Nl):
                                    for bin3p in range(bin1p,self.Nl):
                                        for bin4p in range(bin3p,self.Nl):
                                            if bin1p==bin3p and bin4p<bin2p: continue
                                            for binLp in range(self.Nl):
                                                # skip bins outside the triangle conditions
                                                if not self._check_bin(bin1p,bin2p,binLp,even=False): continue
                                                if not self._check_bin(bin3p,bin4p,binLp,even=False): continue
                                                
                                                # Update indices
                                                index2e += 1
                                                if bin2p!=bin4p: index2o += 1 # no equal bins!

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
                                                for l1 in range(self.min_l+bin1*self.dl,self.min_l+(bin1+1)*self.dl):
                                                    for l2 in range(self.min_l+bin2*self.dl,self.min_l+(bin2+1)*self.dl):
                                                        for L in range(self.min_l+binL*self.dl,self.min_l+(binL+1)*self.dl):
                                                            # first 3j symbols with spin (-1, -1, 2)
                                                            tj12 = self.threej(l1,l2,L)
                                                            if L<abs(l1-l2) or L>l1+l2: continue
                                                            for l3 in range(self.min_l+bin3*self.dl,self.min_l+(bin3+1)*self.dl):
                                                                for l4 in range(self.min_l+bin4*self.dl,self.min_l+(bin4+1)*self.dl):
                                                                    if L<abs(l3-l4) or L>l3+l4: continue
                                                                    
                                                                    print(l1,l2,l3,l4,L)

                                                                    # Continue if wrong-parity, or in b2 = b4 bin and odd
                                                                    if (-1)**(l1+l2+l3+l4)==-1 and bin2==bin4: continue
                                                                    if (-1)**(l1+l2+l3+l4)==1 and parity=='odd': continue
                                                                    if (-1)**(l1+l2+l3+l4)==-1 and parity=='even': continue
                                                                    
                                                                    # second 3j symbols with spin (-1, -1, 2)
                                                                    tj34 = self.threej(l3,l4,L)

                                                                    ## add first permutation
                                                                    if pref1!=0 and tj12*tj34!=0:
                                                                        # assemble relevant contribution
                                                                        if (-1)**(l1+l2+l3+l4)==-1:
                                                                            value_odd += -pref1*tj12**2*tj34**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)/(4.*np.pi)**2/self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                        else:
                                                                            value_even += pref1*tj12**2*tj34**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)/(4.*np.pi)**2/self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]

                                                                        # Iterate over L' for off-diagonal terms
                                                                        for Lp in range(self.min_l+binLp*self.dl,self.min_l+(binLp+1)*self.dl):
                                                                            tj1324 = self.threej(l1,l3,Lp)*self.threej(l2,l4,Lp)
                                                                            tj1432 = self.threej(l1,l4,Lp)*self.threej(l3,l2,Lp)

                                                                            ## add second permutation
                                                                            if pref2!=0 and tj1324!=0: 
                                                                                if (-1)**(l1+l2+l3+l4)==-1:
                                                                                    value_odd += -pref2*(-1.)**(l2+l3)*tj12*tj34*tj1324*self.sixj(L,l1,l2,Lp,l4,l3)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2.*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                                else:
                                                                                    value_even += pref2*(-1.)**(l2+l3)*tj12*tj34*tj1324*self.sixj(L,l1,l2,Lp,l4,l3)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2.*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]

                                                                            ## add third permutation
                                                                            if pref3!=0 and tj1432!=0:
                                                                                if (-1)**(l1+l2+l3+l4)==-1:
                                                                                    value_odd += -pref3*(-1.)**(L+Lp)*tj12*tj34*tj1432*self.sixj(L,l1,l2,Lp,l3,l4)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                                else:
                                                                                    value_even += pref3*(-1.)**(L+Lp)*tj12*tj34*tj1432*self.sixj(L,l1,l2,Lp,l3,l4)*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)*(2.*l4+1.)*(2.*L+1.)*(2*Lp+1.)/(4.*np.pi)**2./self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]/self.base.Cl[l4]
                                                                    
                                                if parity!='even':
                                                    fish_odd[index1o, index2o] = value_odd
                                                if parity!='odd':
                                                    fish_even[index1e, index2e] = value_even

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
    
    def Tl_ideal(self, data, fish_ideal=[], parity='even', verb=False):
        """
        Compute the idealized trispectrum estimator, including normalization, if not supplied or already computed. Note that this normalizes by < mask^4 >.
        
        The `parity' parameter can be 'even', 'odd' or 'both'. This specifies what parity trispectra to compute.
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
        if parity=='even' and not hasattr(self,'inv_fish_ideal'):
            print("Computing ideal Fisher matrix")
            self.fisher_ideal(parity=parity, verb=verb)
            
        # Compute numerator
        if verb: print("Computing numerator")
        Bl_num_ideal = self.Tl_numerator_ideal(data, parity=parity, verb=False)
        
        # Compute full estimator
        if parity=='even':
            Bl_even = np.matmul(self.inv_fish_ideal_even,Bl_num_ideal)
            return Bl_even
        elif parity=='odd':
            Bl_odd = np.matmul(self.inv_fish_ideal_odd,Bl_num_ideal)
            return Bl_odd
        else:
            Bl_even = np.matmul(self.inv_fish_ideal_even,Bl_num_ideal[0])
            Bl_odd = np.matmul(self.inv_fish_ideal_odd,Bl_num_ideal[1])
            return Bl_even, Bl_odd
    