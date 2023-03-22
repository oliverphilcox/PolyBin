### Compute the odd-parity CMB trispectrum arising from the gauge field model of Shiraishi 16 as in Philcox 22
# The output contains the trispectrum, but without a prefactor of A_CS

import sys, os
sys.path.append('/home/ophilcox/wigxjpf-1.11/')
import numpy as np, time, tqdm, multiprocessing as mp, itertools
import camb
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn, hyp2f1, gamma, legendre
import pywigxjpf as wig
from scipy.interpolate import interp1d
from classy import Class
import multiprocessing as mp, tqdm

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
assert Nl_squeeze >= Nl

for L_edge in L_bins:
    assert L_edge in l_bins_squeeze, "l-bins must contain all L-bins!"
for l_edge in l_bins:
    assert l_edge in l_bins_squeeze, "Squeezed bins must contain all the unsqueezed bins!"

Ncpus = os.cpu_count()
transfer = 'full'
print("lmax: %d, lmax-squeeze: %d, Lmax: %d"%(np.max(l_bins),np.max(l_bins_squeeze),np.max(L_bins)))

init_time = time.time()
lmax = np.max(l_bins_squeeze)

### Cosmology

# Run CAMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.32, ombh2=0.022383, omch2=0.12011,tau=0.0543,mnu=0.06,omk=0,standard_neutrino_neff=True)
As = 1e-10*np.exp(3.0448)
pars.InitPower.set_params(ns=0.96605,r=0.,pivot_scalar=0.05,As=As)

DeltaSq_zeta = As*2*np.pi**2 # primordial amplitude (from Planck)

# Compute distance to last scattering (or peak thereof)
back = camb.get_background(pars)
r_star = (back.tau0-back.tau_maxvis)

## Precision parameters on k, s, x
# NB: k integrals only appear in pre-processing so large N_k is fine!

# Compute k, s, x arrays
k0_arr = np.linspace(0.1,21,501)/r_star # 1001
x_arr = np.linspace(0.01,3.00,100).reshape(-1,1)*r_star # 500
s_arr = np.linspace(0.01,lmax*3,101)/r_star # 501
dx = np.diff(x_arr.ravel())[0]
dk = np.diff(k0_arr)[0]
ds = np.diff(s_arr)[0]
xx_arr = x_arr.ravel()

print("N_k: %d"%len(k0_arr.ravel()))
print("N_s: %d"%len(s_arr.ravel()))
print("N_x: %d"%len(x_arr.ravel()))

if transfer=='SW':
    print("Assuming Sachs-Wolfe transfer function")
    Tl_k_func = lambda k0: np.asarray([-(l>=2)*(1./5.)*spherical_jn(l,l*k0*r_star) for l in range(lmax+3)])
    Tl_k_arr = Tl_k_func(k0_arr)
elif transfer=='full':
    print("Using full transfer functions")
    transfer_inp = np.load('planck/camb_transfer.npz')
    Tl_k_func = interp1d(transfer_inp['k'],transfer_inp['transfer'][:lmax+1],bounds_error=False,fill_value=0.)
    Tl_k_arr = np.vstack([[np.zeros_like(k0_arr) for _ in range(2)],[Tl_k_func(k0_arr*l)[l] for l in range(2,lmax+1)]])
else:
    raise Exception("Wrong transfer function!")

# Compute integrals over Tl(k)
k = np.diff(k0_arr).mean()
k2_sq = k0_arr**2./(2.*np.pi**2.) # adding k from dlog(k) = dk/k!
xx_arr = x_arr.ravel()
jls_sx = np.asarray([spherical_jn(l,s_arr*x_arr) for l in range(lmax+3)])

# Compute Tl integrals for required ranges of l,L
print("Computing h integrals")
Tl_ints = []
Tl_k3_ints = []
for l in range(lmax+1):
    pref = Tl_k_arr[l]*k2_sq*dk/k0_arr**3.

    # Compute integral h^{ll}_0
    Tl_ints.append(np.sum(Tl_k_arr[l]*l**3*k2_sq*dk*spherical_jn(l,l*k0_arr*x_arr),axis=1))
    
    # Compute integral h^{lL}_{-3}
    tmp_arr = []
    for L in range(abs(l-2),l+3,1):
        tmp_arr.append(np.sum(pref*spherical_jn(L,l*k0_arr*x_arr),axis=1))
    Tl_k3_ints.append(tmp_arr)

# 3j/9j set-up
wig.wig_table_init(4*lmax+4,9)
wig.wig_temp_init(4*lmax+4)

# Wigner 3j + 9j
tj0 = lambda l1, l2, l3: wig.wig3jj(2*l1 , 2*l2, 2*l3, 0, 0, 0)
tjSpin = lambda l1, l2, l3: wig.wig3jj(2*l1, 2*l2, 2*l3, 2*-1, 2*-1, 2*2)
sixj = lambda l1,l2,l3,l4,l5,l6: wig.wig6jj(2*l1,2*l2,2*l3,2*l4,2*l5,2*l6)
ninej = lambda l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3: wig.wig9jj(2*l1,2*l2,2*l3,2*lp1,2*lp2,2*lp3,2*lpp1,2*lpp2,2*lpp3)

########################### COUPLING MATRICES ###########################

def check(l1,l2,l3,even=True):
    """Check triangle conditions for a triplet of momenta. Returns True if conditions are *not* obeyed"""
    if l1<abs(l2-l3): return True
    if l1>l2+l3: return True
    if even:
        if (-1)**(l1+l2+l3)==-1: return True
    return False

def _check_bin(bin1, bin2, bin3, even=False):
    """
    Return one if modes in the bin satisfy the triangle conditions, or zero else.

    If even=true, we enforce that the sum of the three ells must be even.

    This is used either for all triangles in the bin, or just the center of the bin.
    """
    l1 = 0.5*(l_bins_squeeze[bin1]+l_bins_squeeze[bin1+1])
    l2 = 0.5*(l_bins_squeeze[bin2]+l_bins_squeeze[bin2+1])
    l3 = 0.5*(l_bins_squeeze[bin3]+l_bins_squeeze[bin3+1])
    if l3<abs(l1-l2) or l3>l1+l2:
        return 0
    else:
        return 1

# Define ell arrays and custom harmonic-space weighting
l = np.arange(lmax+1)
Sl = np.load('Sl_weighting.npy')

########################### COMPUTE TRISPECTRA ###########################

C_ell = lambda Ls: np.sqrt(np.product(2.*np.asarray(Ls)+1.))

def _compute_trispectrum(index):
    bin1,bin2,bin3,bin4,binL = bins[index]
    #print("On index %d of %d"%(index+1,len(bins)))

    # Compute weighting (weighted number of ell modes in bin)
    weighting = 0.
    for L in range(max(2,L_bins[binL]),L_bins[binL+1]):
        
        tjspin1e,tjspin1o = 0.,0.
        for l1 in range(max(1,l_bins[bin1]),l_bins[bin1+1]):
            for l2 in range(max(1,l_bins_squeeze[bin2]),l_bins_squeeze[bin2+1]):
                
                # Check triangle conditions
                if check(l1,l2,L,False): continue
                
                tjspin1 = tjSpin(l1,l2,L)**2.*(2*l1+1)*(2*l2+1)/Sl[l1]/Sl[l2]/(4*np.pi)
                tjspin1e += tjspin1
                tjspin1o += tjspin1*(-1)**(l1+l2)
                
        tjspin2e,tjspin2o = 0.,0.
        for l3 in range(max(1,l_bins[bin3]),l_bins[bin3+1]):
            for l4 in range(max(1,l_bins_squeeze[bin4]),l_bins_squeeze[bin4+1]):

                # Check L conditions
                if check(l3,l4,L,False): continue

                tjspin2 = tjSpin(l3,l4,L)**2.*(2*l3+1)*(2*l4+1)/Sl[l3]/Sl[l4]/(4*np.pi)
                tjspin2e += tjspin2
                tjspin2o += tjspin2*(-1)**(l3+l4)
                
        weighting += 0.5*(tjspin1e*tjspin2e-tjspin1o*tjspin1o)*(2*L+1)
    
    if weighting==0: return 0.
    
    # Compute prefactor
    pref = -1.*np.sqrt(2.)*DeltaSq_zeta**3.*(4.*np.pi)**5.

    # Assemble theory prediction
    output = 0.    
    for L in range(max(2,L_bins[binL]),L_bins[binL+1]):  

        term = 0.
        for lam1,lam3,lam,A in zip([1,2,2,1],[1,2,1,2],[1,1,2,2],[1.,1./np.sqrt(5.),-1./np.sqrt(5.),1./np.sqrt(5.)]):

            pref2 = A*C_ell([lam1,lam3,lam])

            # Compute sums over L1,l1,l2 for each Lp
            first_terms1, first_terms2 = [],[]
            for Lp in range(abs(L-lam1),L+lam1+1,1):
                
                first_term1, first_term2 = 0.,0.
                for l1 in range(max(1,l_bins[bin1]),l_bins[bin1+1]):
                    for l2 in range(max(1,l_bins_squeeze[bin2]),l_bins_squeeze[bin2+1]):

                        # Check triangle conditions
                        if check(l1,l2,L,False): continue

                        tjspin1 = tjSpin(l1,l2,L)
                        if tjspin1==0: continue

                        for L1 in range(abs(l1-lam1),l1+lam1+1,2):
                            sixj1 = sixj(L,l1,l2,L1,Lp,lam1)
                            if sixj1==0: continue
                            first_term = (1.0j)**(-l1-L1+Lp)*(2*L1+1.)*tj0(L1,l2,Lp)*tj0(L1,lam1,l1)*sixj1
                            first_term *= np.sum((xx_arr**2.*dx*Tl_k3_ints[l1][L1-abs(l1-2)]*Tl_ints[l2]).reshape(-1,1)*jls_sx[Lp],axis=0)
                            # Add ell weighting and 1/ThreeJ 
                            first_term *= tjspin1*(2*l1+1)*(2*l2+1)/Sl[l1]/Sl[l2]/(4*np.pi)
                            # Compute parity terms
                            first_term1 += first_term
                            first_term2 += (-1.)**(l1+l2)*first_term
                first_terms1.append(first_term1)
                first_terms2.append(first_term2)

            # Compute sums over L2,l3,l4 for each Lpp
            second_terms1, second_terms2 = [],[]
            for Lpp in range(abs(lam3-L),lam3+L+1,1):

                second_term1, second_term2 = 0.,0.
                for l3 in range(max(1,l_bins[bin3]),l_bins[bin3+1]):
                    for l4 in range(max(1,l_bins_squeeze[bin4]),l_bins_squeeze[bin4+1]):

                        # Check triangle conditions
                        if check(l3,l4,L,False): continue

                        tjspin2 = tjSpin(l3,l4,L)
                        if tjspin2==0: continue

                        for L3 in range(abs(l3-lam3),l3+lam3+1,2):
                            
                            sixj2 = sixj(l3,l4,L,Lpp,lam3,L3)
                            if sixj2==0: continue
                            second_term = (1.0j)**(l3+L3+Lpp)*(2.*L3+1.)*tj0(L3,l4,Lpp)*tj0(lam3,L3,l3)*sixj2
                            second_term *= np.sum((xx_arr**2.*dx*Tl_k3_ints[l3][L3-abs(l3-2)]*Tl_ints[l4]).reshape(-1,1)*jls_sx[Lpp],axis=0)
                            # Add ell weighting and 1/ThreeJ 
                            second_term *= tjspin2*(2*l3+1)*(2*l4+1)/Sl[l3]/Sl[l4]/(4*np.pi)
                            second_term1 += second_term
                            second_term2 += (-1.)**(l3+l4)*second_term
                second_terms1.append(second_term1)
                second_terms2.append(second_term2)

            # Assemble trispectrum
            for Lp in range(abs(L-lam1),L+lam1+1,1):
                for Lpp in range(abs(L-lam3),L+lam3+1,1):
                    if Lpp<abs(lam-Lp) or Lpp>lam+Lp or (-1.)**(lam+Lp+Lpp)==-1.: continue            

                    # Combine, including (2L+1) weighting
                    this_term = pref*pref2*(2.*Lp+1.)*(2.*Lpp+1.)*tj0(Lp,lam,Lpp)*sixj(lam,lam1,lam3,L,Lpp,Lp)
                    integ = 0.5*(first_terms1[Lp-abs(L-lam1)]*second_terms1[Lpp-abs(L-lam3)]-first_terms2[Lp-abs(L-lam1)]*second_terms2[Lpp-abs(L-lam3)])/s_arr
                    this_term *= np.sum(ds*integ)/(2.*np.pi**2.)
                    term += this_term

        output += term.imag
                        
    if weighting!=0:
        output = output/weighting
    else:
        raise Exception()
    return output

# Create output list
bins = []
# Iterate over bins
for bin1 in range(Nl):
    for bin2 in range(bin1,Nl_squeeze):
        for bin3 in range(bin1,Nl):
            for bin4 in range(bin3,Nl_squeeze):
                if bin1==bin3 and bin4<=bin2: continue

                # Iterate over L bins
                for binL in range(NL):
                    # skip bins outside the triangle conditions
                    if not _check_bin(bin1,bin2,binL,even=False): continue
                    if not _check_bin(bin3,bin4,binL,even=False): continue
                    bins.append([bin1,bin2,bin3,bin4,binL])
print("Using %d trispectrum bins"%len(bins))

print("Beginning multiprocessing on %d CPUs"%Ncpus)

t_start = time.time()
p = mp.Pool(Ncpus)
tl_out = list(tqdm.tqdm(p.imap(_compute_trispectrum,np.arange(len(bins))),total=len(bins)))
p.close()
p.join()
print("Time: %.2f s"%(time.time()-t_start))
print("Multiprocessing complete after %.2f s"%(time.time()-t_start))

outfile = 'gauge_trispectrum_%s_weighted_l(%d,%d,%d).txt'%(transfer,Nl,Nl_squeeze,NL)
np.savetxt(outfile,tl_out)

print("Saved output to %s after %.1f s; exiting."%(outfile,time.time()-init_time))
