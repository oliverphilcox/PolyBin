### Code for ideal and unwindowed binned polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022)
## This module contains the bispectrum estimation code

import numpy as np
import multiprocessing as mp
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
import pywigxjpf as wig

class BSpec():
    """Bispectrum estimation class. This takes the binning strategy as input and a base class. 
    We also feed in a function that applies the S^-1 operator in real space.

    Note that we can additionally compute squeezed triangles by setting Nl_squeeze > Nl, giving a different lmax for short and long sides.
    
    Inputs:
    - base: PolyBin class
    - mask: HEALPix mask to deconvolve
    - applySinv: function which returns S^-1 applied to a given input map
    - min_l, dl, Nl: binning parameters
    - Nl_squeeze: number of squeezed ell bins 
    - include_partial_triangles: whether to include triangles whose centers don't satisfy the triangle conditions. (Default: False)
    """
    def __init__(self, base, mask, applySinv, min_l, dl, Nl, Nl_squeeze=None, include_partial_triangles=False):
        # Read in attributes
        self.base = base
        self.mask = mask
        self.applySinv = applySinv
        self.min_l = min_l
        self.dl = dl
        self.Nl = Nl
        if Nl_squeeze!=None:
            assert Nl_squeeze>=Nl, "Squeezed Nl must be at least as large as Nl!"
            self.Nl_squeeze = Nl_squeeze
        else:
            self.Nl_squeeze = Nl
        self.include_partial_triangles = include_partial_triangles
        self.beam = self.base.beam
        self.beam_lm = self.base.beam_lm
        
        if min_l+Nl*dl>base.lmax:
            raise Exception("Maximum l is larger than HEALPix resolution!")
        print("Binning: %d bins in [%d, %d]"%(Nl,min_l,min_l+Nl*dl))
        if self.Nl_squeeze!=self.Nl:
            print("Squeezed binning: %d bins in [%d, %d]"%(Nl_squeeze,min_l,min_l+Nl_squeeze*dl))
        
        # Define l filters
        self.ell_bins = [(self.base.l_arr>=self.min_l+self.dl*bin1)&(self.base.l_arr<self.min_l+self.dl*(bin1+1)) for bin1 in range(self.Nl_squeeze)]
        
    def _check_bin(self, bin1, bin2, bin3):
        """Return one if modes in the bin satisfy the even-parity triangle conditions, or zero else.

        This is used either for all triangles in the bin, or just the center of the bin.
        """
        if self.include_partial_triangles:
            good = 0
            for l1 in range(self.min_l+bin1*self.dl,self.min_l+(bin1+1)*self.dl):
                for l2 in range(self.min_l+bin2*self.dl,self.min_l+(bin2+1)*self.dl):
                    for l3 in range(self.min_l+bin3*self.dl,self.min_l+(bin3+1)*self.dl):
                        # skip any odd bins
                        if (-1)**(l1+l2+l3)==-1: continue 
                        if l1>=abs(l1-l2) and l3<=l1+l2:
                            good = 1
                        if good==1: break
                    if good==1: break
                if good==1: break
            if good==1: return 1
            else:
                return 0
        else:
            l1 = self.min_l+(bin1+0.5)*self.dl-0.5
            l2 = self.min_l+(bin2+0.5)*self.dl-0.5
            l3 = self.min_l+(bin3+0.5)*self.dl-0.5
            if l3<abs(l1-l2) or l3>l1+l2:
                return 0
            else:
                return 1
         
    def _compute_symmetry_factor(self):
        """
        Compute symmetry factor giving the degeneracy of each bin.
        """
        sym_factor = []
        
        # Iterate over bins
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl_squeeze):
                for bin3 in range(bin2,self.Nl_squeeze):
                    
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue

                    # compute symmetry factor
                    if bin1==bin2 and bin2==bin3:
                        sym = 6
                    elif bin1==bin2 or bin2==bin3:
                        sym = 2
                    else:
                        sym = 1
                    sym_factor.append(sym)
        self.sym_factor = np.asarray(sym_factor)

        # Count number of bins
        self.N_b = len(self.sym_factor)
        print("Using %d bispectrum bins"%self.N_b)
        
    def get_ells(self):
        """
        Return an array with the central l1, l2, l3 values for each bispectrum bin.
        """
        # Iterate over bins
        l1s, l2s, l3s = [],[],[]
        for bin1 in range(self.Nl):
            l1 = self.min_l+(bin1+0.5)*self.dl-0.5
            for bin2 in range(bin1,self.Nl_squeeze):
                l2 = self.min_l+(bin2+0.5)*self.dl-0.5
                for bin3 in range(bin2,self.Nl_squeeze):
                    l3 = self.min_l+(bin3+0.5)*self.dl-0.5
                
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue

                    # Add to output array
                    l1s.append(l1)
                    l2s.append(l2)
                    l3s.append(l3)
        return l1s,l2s,l3s

    def load_sims(self, sims, verb=False):
        """
        Load in Monte Carlo simulations used in the linear term of the bispectrum estimator. 
        These can alternatively be generated with a fiducial spectrum using the generate_sims script.
        """
        
        self.N_it = len(sims)
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        # Iterate over simulations
        self.Q_b_alpha_maps = []
        
        for ii in range(self.N_it):
            if ii%5==0 and verb: print("Processing bias simulation %d of %d"%(ii+1,self.N_it))
            
            # Transform to Fourier space and normalize appropriately
            Wh_alpha_lm = self.base.to_lm(self.mask*self.applySinv(sims[ii]))
            
            Q_b_alpha_map = [self.base.to_map(Wh_alpha_lm*self.ell_bins[bin1]*self.beam_lm) for bin1 in range(self.Nl_squeeze)]
            
            self.Q_b_alpha_maps.append(Q_b_alpha_map)
        
    def generate_sims(self, N_it, Cl_input=[], b_input=None, add_B=False, remove_mean=True, verb=False):
        """
        Generate Monte Carlo simulations used in the linear term of the bispectrum generator. 
        These are pure GRFs, optionally with a bispectrum added. By default, they are generated with the input survey mask.
        We create N_it such simulations and store the relevant transformations into memory.
        
        We can alternatively load custom simulations using the load_sims script.
        """
        
        self.N_it = N_it
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        # Define input power spectrum (with noise)
        if len(Cl_input)==0:
            Cl_input = self.base.Cl
        
        # Iterate over simulations
        self.Q_b_alpha_maps = []
        for ii in range(N_it):
            if ii%5==0 and verb: print("Generating bias simulation %d of %d"%(ii+1,N_it))
            
            # Generate simulation
            raw_alpha = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, b_input=b_input, add_B=add_B, remove_mean=remove_mean)
            
            # Transform to Fourier space and normalize appropriately
            Wh_alpha_lm = self.base.to_lm(self.mask*self.applySinv(raw_alpha*self.mask))
            
            Q_b_alpha_map = [self.base.to_map(Wh_alpha_lm*self.ell_bins[bin1]*self.beam_lm) for bin1 in range(self.Nl_squeeze)]
            
            self.Q_b_alpha_maps.append(Q_b_alpha_map)
    
    ### OPTIMAL ESTIMATOR
    def Bl_numerator(self, data, include_linear_term=True, verb=False):
        """
        Compute the numerator of the unwindowed bispectrum estimator. We can optionally drop the linear term
        """
        
        # Compute symmetry factor if necessary
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
            
        if not hasattr(self, 'Q_b_alpha_maps') and include_linear_term:
            raise Exception("Need to generate or specify bias simulations!")
        
        # Apply W * S^-1 to data and transform to harmonic space, then compute Q map
        Wh_data_lm = self.base.to_lm(self.mask*self.applySinv(data))
        Q_b_map = [self.base.to_map(Wh_data_lm*self.ell_bins[bin1]*self.beam_lm) for bin1 in range(self.Nl_squeeze)]

        # Compute b_3 part of cubic estimator
        b3_num = np.zeros(self.N_b)
        index = 0
        if verb: print("Computing b_3 piece")
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl_squeeze):
                for bin3 in range(bin2,self.Nl_squeeze):
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue

                    # compute numerators
                    b3_num[index] = self.base.A_pix*np.sum(Q_b_map[bin1]*Q_b_map[bin2]*Q_b_map[bin3])/self.sym_factor[index]
                    index += 1
        
        # Compute b_1 part of cubic estimator, averaging over simulations
        b1_num = np.zeros(self.N_b)
        
        if not include_linear_term:
            print("No linear correction applied!")
        else:
            for ii in range(self.N_it):
                if (ii+1)%5==0 and verb: print("Computing b_1 piece from simulation %d"%(ii+1))

                # Iterate over bins
                index = 0
                for bin1 in range(self.Nl):
                    for bin2 in range(bin1,self.Nl_squeeze):
                        for bin3 in range(bin2,self.Nl_squeeze):
                            # skip bins outside the triangle conditions
                            if not self._check_bin(bin1,bin2,bin3): continue

                            # compute numerators, summing over permutations
                            b1_num[index] += -self.base.A_pix*np.sum(self.Q_b_alpha_maps[ii][bin1]*self.Q_b_alpha_maps[ii][bin2]*Q_b_map[bin3])/self.sym_factor[index]/self.N_it
                            b1_num[index] += -self.base.A_pix*np.sum(self.Q_b_alpha_maps[ii][bin1]*Q_b_map[bin2]*self.Q_b_alpha_maps[ii][bin3])/self.sym_factor[index]/self.N_it
                            b1_num[index] += -self.base.A_pix*np.sum(Q_b_map[bin1]*self.Q_b_alpha_maps[ii][bin2]*self.Q_b_alpha_maps[ii][bin3])/self.sym_factor[index]/self.N_it

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
        fish = np.zeros((self.N_b,self.N_b))
        
        # Compute two random realizations with known power spectrum
        if verb: print("Generating pair of GRFs")
        u1 = self.base.generate_data(seed=seed+int(1e7))
        u2 = self.base.generate_data(seed=seed+int(2e7))

        # Compute weighted fields
        Sinv_u1, Sinv_u2 = self.applySinv(u1), self.applySinv(u2)
        Uinv_u1, Uinv_u2 = self.base.applyUinv(u1), self.base.applyUinv(u2)

        # Define H maps
        if verb: print("Creating H_b maps")
        WSinv_u1_lm, WSinv_u2_lm = self.base.to_lm(self.mask*Sinv_u1), self.base.to_lm(self.mask*Sinv_u2)
        WUinv_u1_lm, WUinv_u2_lm = self.base.to_lm(self.mask*Uinv_u1), self.base.to_lm(self.mask*Uinv_u2)
        H_b_Sinv_u1 = [self.base.to_map(WSinv_u1_lm*self.ell_bins[bin1]*self.beam_lm)*self.mask for bin1 in range(self.Nl_squeeze)]
        H_b_Uinv_u1 = [self.base.to_map(WUinv_u1_lm*self.ell_bins[bin1]*self.beam_lm)*self.mask for bin1 in range(self.Nl_squeeze)]
        H_b_Sinv_u2 = [self.base.to_map(WSinv_u2_lm*self.ell_bins[bin1]*self.beam_lm)*self.mask for bin1 in range(self.Nl_squeeze)]
        H_b_Uinv_u2 = [self.base.to_map(WUinv_u2_lm*self.ell_bins[bin1]*self.beam_lm)*self.mask for bin1 in range(self.Nl_squeeze)]

        # Compute pairs of H fields
        # NB: ordering is such that largest index is first
        if verb: print("Computing (H_b H_b')_{lm}")
        HH_Sinv_lms11 = [[self.base.to_lm(H_b_Sinv_u1[bin1]*H_b_Sinv_u1[bin2]) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
        HH_Uinv_lms11 = [[self.base.to_lm(H_b_Uinv_u1[bin1]*H_b_Uinv_u1[bin2]) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
        HH_Sinv_lms22 = [[self.base.to_lm(H_b_Sinv_u2[bin1]*H_b_Sinv_u2[bin2]) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]
        HH_Uinv_lms22 = [[self.base.to_lm(H_b_Uinv_u2[bin1]*H_b_Uinv_u2[bin2]) for bin2 in range(bin1+1)] for bin1 in range(self.Nl_squeeze)]

        # Iterate over bins
        WSinvW_Q_Uinv_maps11 = []
        WSinvW_Q_Uinv_maps22 = []
        Q_Sinv_maps11 = []
        Q_Sinv_maps22 = []
        index = 0
        if verb: print("Computing Q(b) maps")
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl_squeeze):
                for bin3 in range(bin2,self.Nl_squeeze):
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue

                    # Define Q with U^-1 weights
                    this_Q_Uinv_lm11 =  2./self.sym_factor[index]*self.ell_bins[bin1]*self.beam_lm*HH_Uinv_lms11[bin3][bin2]
                    this_Q_Uinv_lm11 += 2./self.sym_factor[index]*self.ell_bins[bin2]*self.beam_lm*HH_Uinv_lms11[bin3][bin1]
                    this_Q_Uinv_lm11 += 2./self.sym_factor[index]*self.ell_bins[bin3]*self.beam_lm*HH_Uinv_lms11[bin2][bin1]
                    this_Q_Uinv_lm22 =  2./self.sym_factor[index]*self.ell_bins[bin1]*self.beam_lm*HH_Uinv_lms22[bin3][bin2]
                    this_Q_Uinv_lm22 += 2./self.sym_factor[index]*self.ell_bins[bin2]*self.beam_lm*HH_Uinv_lms22[bin3][bin1]
                    this_Q_Uinv_lm22 += 2./self.sym_factor[index]*self.ell_bins[bin3]*self.beam_lm*HH_Uinv_lms22[bin2][bin1]

                    # Apply weighting
                    WSinvW_Q_Uinv_maps11.append(self.mask*self.applySinv(self.mask*self.base.to_map(this_Q_Uinv_lm11)))
                    WSinvW_Q_Uinv_maps22.append(self.mask*self.applySinv(self.mask*self.base.to_map(this_Q_Uinv_lm22)))

                    # Define Q with S^-1 weights
                    this_Q_Sinv_lm11 =  2./self.sym_factor[index]*self.ell_bins[bin1]*self.beam_lm*HH_Sinv_lms11[bin3][bin2]
                    this_Q_Sinv_lm11 += 2./self.sym_factor[index]*self.ell_bins[bin2]*self.beam_lm*HH_Sinv_lms11[bin3][bin1]
                    this_Q_Sinv_lm11 += 2./self.sym_factor[index]*self.ell_bins[bin3]*self.beam_lm*HH_Sinv_lms11[bin2][bin1]
                    this_Q_Sinv_lm22 =  2./self.sym_factor[index]*self.ell_bins[bin1]*self.beam_lm*HH_Sinv_lms22[bin3][bin2]
                    this_Q_Sinv_lm22 += 2./self.sym_factor[index]*self.ell_bins[bin2]*self.beam_lm*HH_Sinv_lms22[bin3][bin1]
                    this_Q_Sinv_lm22 += 2./self.sym_factor[index]*self.ell_bins[bin3]*self.beam_lm*HH_Sinv_lms22[bin2][bin1]

                    Q_Sinv_maps11.append(self.base.to_map(this_Q_Sinv_lm11))
                    Q_Sinv_maps22.append(self.base.to_map(this_Q_Sinv_lm22))
                    index += 1

        # Assemble Fisher matrix
        if verb: print("Computing Fisher matrix")
        for index1 in range(self.N_b):

            # Define relevant maps
            this_Q11 = Q_Sinv_maps11[index1]
            this_Q22 = Q_Sinv_maps22[index1]

            for index2 in range(self.N_b):
                # Form summand
                summand  = 0.5*(this_Q11*WSinvW_Q_Uinv_maps11[index2]+this_Q22*WSinvW_Q_Uinv_maps22[index2])
                summand  -= 0.5*(this_Q11*WSinvW_Q_Uinv_maps22[index2]+this_Q22*WSinvW_Q_Uinv_maps11[index2])
                fish[index1,index2] += 1./12.*self.base.A_pix*np.sum(summand)

        # Return matrix
        return fish
    
    def compute_fisher(self, N_it, N_cpus=1):
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
            return self.compute_fisher_contribution(seed)
        
        if N_cpus==1:
            for seed in range(N_it):
                if seed%5==0: print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
                fish += self.compute_fisher_contribution(seed)/N_it
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
        
        return Bl_out
    
    ### IDEAL ESTIMATOR
    def Bl_numerator_ideal(self, data):
        """
        Compute the numerator of the idealized bispectrum estimator. We normalize by < mask^3 >. Note that this does not include one-field terms.
        """
        # Normalize data by C_th and transform to harmonic space
        Cinv_data_lm = self.base.safe_divide(self.base.to_lm(data),self.base.Cl_lm)
        
        # Compute I maps
        I_map = [self.base.to_map(self.ell_bins[bin1]/self.beam_lm*Cinv_data_lm) for bin1 in range(self.Nl_squeeze)]

        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Combine to find numerator
        b_num_ideal = np.zeros(self.N_b)
        
        # Iterate over bins
        index = 0
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl_squeeze):
                for bin3 in range(bin2,self.Nl_squeeze):
                    
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue
                    
                    # compute numerators
                    b_num_ideal[index] = self.base.A_pix*np.sum(I_map[bin1]*I_map[bin2]*I_map[bin3])
                    index += 1
                    
        # Normalize
        b_num_ideal *= 1./self.sym_factor/np.mean(self.mask**3)
        return b_num_ideal

    def compute_fisher_ideal(self, verb=False):
        """This computes the idealized Fisher matrix for the bispectrum."""
        
        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Compute diagonal of matrix
        fish_diag = np.zeros(self.N_b)
        
        # Iterate over bins
        index = 0
        for bin1 in range(self.Nl):
            for bin2 in range(bin1,self.Nl_squeeze):
                for bin3 in range(bin2,self.Nl_squeeze):
                    
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue

                    if (index+1)%25==0 and verb: print("Computing bin %d of %d"%(index+1,self.N_b))
                    value = 0.

                    # Now iterate over l values in bin
                    for l1 in range(self.min_l+bin1*self.dl,self.min_l+(bin1+1)*self.dl):
                        for l2 in range(self.min_l+bin2*self.dl,self.min_l+(bin2+1)*self.dl):
                            for l3 in range(max([abs(l1-l2),self.min_l+bin3*self.dl]),min([l1+l2,self.min_l+(bin3+1)*self.dl])):
                                
                                if (-1)**(l1+l2+l3)==-1: continue # 3j = 0 here
                                tj = self.base.tj0(l1,l2,l3)
                                value += tj**2*(2.*l1+1.)*(2.*l2+1.)*(2.*l3+1.)/(4.*np.pi)/self.base.Cl[l1]/self.base.Cl[l2]/self.base.Cl[l3]
                                            
                    fish_diag[index] = value/self.sym_factor[index]
                    index += 1
        fish = np.diag(fish_diag)
        
        # Save attributes
        self.fish_ideal = fish
        self.inv_fish_ideal = np.diag(1./fish_diag)
        
        return fish
    
    def Bl_ideal(self, data, fish_ideal=[]):
        """Compute the idealized bispectrum estimator, including normalization, if not supplied or already computed. Note that this normalizes by < mask^3 >."""
        
        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)
        
        if not hasattr(self,'inv_fish_ideal'):
            print("Computing ideal Fisher matrix")
            self.compute_fisher_ideal()
            
        # Compute numerator
        Bl_num_ideal = self.Bl_numerator_ideal(data)
        
        # Apply normalization
        Bl_out = np.matmul(self.inv_fish_ideal,Bl_num_ideal)
        
        return Bl_out
    