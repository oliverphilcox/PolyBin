### Compute the odd-parity CMB Fisher matrix
# This works in a diagonal approximation and assumes ideal mask properties
# It should be used for Fisher forecasting only

import sys, numpy as np, os
sys.path.append('../')
import polybin as pb
import healpy, time
import fitsio
from scipy.interpolate import InterpolatedUnivariateSpline

if __name__=='__main__':

    if len(sys.argv)!=4: 
        raise Exception("binfile, binfile_squeeze, binfile_L not specified!")

# PARAMETERS
binfile = str(sys.argv[1])
binfile_squeeze = str(sys.argv[2])
binfile_L = str(sys.argv[3])

# Load binning files
if not os.path.exists(binfile):
    raise Exception("Binning file doesn't exist!")
if not os.path.exists(binfile_squeeze):
    raise Exception("Squeezed binning file doesn't exist!")
if not os.path.exists(binfile_L):
    raise Exception("L binning file doesn't exist!")
l_bins = np.load(binfile)
l_bins_squeeze = np.load(binfile_squeeze)
L_bins = np.load(binfile_L)
Nl = len(l_bins)-1
Nl_squeeze = len(l_bins_squeeze)-1
NL = len(L_bins)-1
assert np.min(l_bins)>=2, "No physical modes below ell = 2!"

for L_edge in L_bins:
    assert L_edge in l_bins_squeeze, "l-bins must contain all L-bins!"
for l_edge in l_bins:
    assert l_edge in l_bins_squeeze, "Squeezed bins must contain all the unsqueezed bins!"

# HEALPix settings
Nside = 256
lmax = 3*Nside-1
print("binned lmax: %d, HEALPix lmax: %d"%(np.max(l_bins_squeeze),lmax))

# Whether to include bins only partially satisfying triangle conditions
include_partial_triangles = False

# whether to add a separable reduced bispectrum to the input maps
include_synthetic_b = False

root = '/projects/QUIJOTE/Oliver/planck_maps/'

from classy import Class
cosmo = Class()

# Define ell arrays and weighting
l = np.arange(lmax+1)
Sl = np.load('Sl_weighting.npy')
beam = 1.+0.*l

# Load class with fiducial Cl and Nside (ensuring no zeros in Cl_th)
# Note that Cl_th includes the beam here!
base = pb.PolyBin(Nside, Sl, beam=beam, include_pixel_weights=False)

## compute mask 
mask = healpy.synfast(Sl,Nside)*0.+1.

# Initialize trispectrum class
tspec = pb.TSpec(base, 1.+0.*mask, lambda x: 1, l_bins, l_bins_squeeze=l_bins_squeeze, L_bins=L_bins)

def compute_fish_diag(self, parity='odd',verb=False, N_cpus = 4):
    
    # Compute symmetry factors
    self._compute_even_symmetry_factor()
    self._compute_odd_symmetry_factor()

    # Define arrays
    if parity!='even':
        fish_odd_diag = np.zeros((self.N_t_odd))
    if parity!='odd':
        fish_even_diag = np.zeros((self.N_t_even))

    # Define list of bins
    bins = []
    index1e,index1o=-1,-1
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
                        
                        # Iterate over second set of bins
                        bin1p, bin2p, bin3p, bin4p = bin1,bin2,bin3,bin4
                        ## Compute permutation factors
                        pref1  = (bin1==bin1p)*(bin2==bin2p)*(bin3==bin3p)*(bin4==bin4p)
                        pref1 += (bin1==bin2p)*(bin2==bin1p)*(bin3==bin3p)*(bin4==bin4p)
                        pref1 += (bin1==bin1p)*(bin2==bin2p)*(bin3==bin4p)*(bin4==bin3p)
                        pref1 += (bin1==bin2p)*(bin2==bin1p)*(bin3==bin4p)*(bin4==bin3p)
                        pref1 += (bin1==bin3p)*(bin2==bin4p)*(bin3==bin1p)*(bin4==bin2p)
                        pref1 += (bin1==bin4p)*(bin2==bin3p)*(bin3==bin1p)*(bin4==bin2p)
                        pref1 += (bin1==bin3p)*(bin2==bin4p)*(bin3==bin2p)*(bin4==bin1p)
                        pref1 += (bin1==bin4p)*(bin2==bin3p)*(bin3==bin2p)*(bin4==bin1p)

                        bins.append([bin1,bin2,bin3,bin4,binL,pref1,index1e,index1o])
    
    global _compute_bin
    def _compute_bin(i):
        """Compute a single bin of the diagonal Fisher matrix"""
        bin1,bin2,bin3,bin4,binL,pref1,index1e,index1o = bins[i]

        value_even = 0.
        value_odd = 0.

        # Now iterate over l bins
        for L in range(L_bins[binL],L_bins[binL+1]):

            # Perform sum over l1, l2
            val12_a, val12_b = 0.,0.
            for l1 in range(l_bins[bin1],l_bins[bin1+1]):
                for l2 in range(l_bins_squeeze[bin2],l_bins_squeeze[bin2+1]):

                    # Check triangle conditions
                    if L<abs(l1-l2) or L>l1+l2: continue

                    # first 3j symbols with spin (-1, -1, 2)
                    tj12 = self.threej(l1,l2,L)

                    # Assemble contribution depending only on l1,l2
                    val12_contrib = tj12**2.*self.beam[l1]**2*self.beam[l2]**2*(2.*l1+1.)*(2.*l2+1.)/self.base.Cl[l1]/self.base.Cl[l2]
                    val12_a += val12_contrib
                    val12_b += val12_contrib*(-1.)**(l1+l2)

            # Perform sum over l3, l4
            val34_a, val34_b = 0.,0.
            for l3 in range(l_bins[bin3],l_bins[bin3+1]):
                for l4 in range(l_bins_squeeze[bin4],l_bins_squeeze[bin4+1]):
                    if L<abs(l3-l4) or L>l3+l4: continue

                    # second 3j symbols with spin (-1, -1, 2)
                    tj34 = self.threej(l3,l4,L)

                    # Assemble contribution depending only on l1,l2
                    val34_contrib = tj34**2.*self.beam[l3]**2*self.beam[l4]**2*(2.*l3+1.)*(2.*l4+1.)/self.base.Cl[l3]/self.base.Cl[l4]
                    val34_a += val34_contrib
                    val34_b += val34_contrib*(-1.)**(l3+l4)

            ## add first permutation
            # assemble relevant contribution, finding signs appropriately
            value_odd += pref1*(2.*L+1.)/(4.*np.pi)**2*(val12_a*val34_a-val12_b*val34_b)/2.
            value_even += pref1*(2.*L+1.)/(4.*np.pi)**2*(val12_a*val34_a+val12_b*val34_b)/2.

        return value_even, value_odd
             
    # Iterate over bins
    if N_cpus==1:
        output = [_compute_bin(i) for i in range(len(bins))]
    else:
        import multiprocessing as mp, tqdm 
        p = mp.Pool(N_cpus)
        t1 = time.time()
        output = list(tqdm.tqdm(p.imap(_compute_bin,np.arange(len(bins))),total=len(bins)))
        p.close()
        p.join()
        t2 = time.time()
        print("Runtime: %.2fs"%(t2-t1))
    
    for i in range(len(bins)):
    
        bin1,bin2,bin3,bin4,binL,pref1,index1e,index1o = bins[i]
        value_even, value_odd = output[i]
        
        # Note that matrix is symmetric if ideal!
        if parity!='even' and ((bin1==bin3)*(bin2==bin4)!=1):
            fish_odd_diag[index1o] = value_odd
        if parity!='odd':
            fish_even_diag[index1e] = value_even
    if parity=='even':
        return fish_even_diag/self.sym_factor_even/self.sym_factor_even
    elif parity=='odd':
        return fish_odd_diag/self.sym_factor_odd/self.sym_factor_odd
    else:
        return fish_even_diag/self.sym_factor_even/self.sym_factor_even, fish_odd_diag/self.sym_factor_odd/self.sym_factor_odd

fish = compute_fish_diag(tspec, N_cpus=os.cpu_count())

fish_out = '../fish_l(%d,%d,%d).txt'%(Nl,Nl_squeeze,NL)
np.savetxt(fish_out,fish)
print("Fisher matrix saved to %s; exiting"%fish_out)