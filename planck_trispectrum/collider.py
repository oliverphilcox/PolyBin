### Compute the odd-parity CMB trispectrum arising from the cosmological collider as in Philcox 23
# The output contains the trispectrum, but without a prefactor of cs^4 lambda_1 lambda_3 sin pi(nu+1/2)/(H)

import sys, os
sys.path.append('/home/ophilcox/wigxjpf-1.11/')
import numpy as np, time, tqdm, multiprocessing as mp, itertools
from classy import Class
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn, hyp2f1, gamma, legendre
import pywigxjpf as wig
import camb

if __name__=='__main__':

    if len(sys.argv)!=6: 
        raise Exception("binfile, binfile_squeeze, binfile_L, cs, nu not specified!")

# PARAMETERS
binfile = str(sys.argv[1])
binfile_squeeze = str(sys.argv[2])
binfile_L = str(sys.argv[3])
cs = float(sys.argv[4])
nu = float(sys.argv[5])

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

print("cs = %.2f, nu = %.2f"%(cs,nu))
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

## Precision parameters on K, s, x
# NB: K_i = k_i / s
u_arr = np.arange(-0.99,1.00,0.04)[np.newaxis,np.newaxis,:] 
v_arr = np.arange(1.05,11.,0.05)[np.newaxis,:,np.newaxis]
K1_arr = (u_arr+v_arr)/2.
K2_arr = (v_arr-u_arr)/2.
dv = np.diff(v_arr.ravel())[0]
du = np.diff(u_arr.ravel())[0]

s_arr = np.logspace(-5,-1,50)
dlogs = np.diff(np.log(s_arr))[0]

print("N_k: %d x %d"%(len(K1_arr[0]),len(K1_arr[0,0])))
print("N_s: %d"%len(s_arr))
K1s_mat = K1_arr*s_arr[:,np.newaxis,np.newaxis]
K2s_mat = K2_arr*s_arr[:,np.newaxis,np.newaxis]
    
# Compute transfer function for each ell
if transfer=='SW':
    print("Assuming Sachs-Wolfe transfer function")
    Tl_k_func = lambda k: np.asarray([-(l>=2)*(1./5.)*spherical_jn(l,k*r_star) for l in range(lmax+3)])
    Tl_K1_arr = Tl_k_func(K1s_mat)
    Tl_K2_arr = Tl_k_func(K2s_mat)
    
# # Load pre-computed transfer functions
elif transfer=='full':
    print("Using full transfer functions")
    # Read in transfer functions computed from CAMB
    transfer_inp = np.load('planck/camb_transfer.npz')
    Tl_k_func = interp1d(transfer_inp['k'],transfer_inp['transfer'][:lmax+1],bounds_error=False,fill_value=0.)
    Tl_K1_arr = np.vstack([[np.zeros_like(K1s_mat) for _ in range(2)],Tl_k_func(K1s_mat)])
    Tl_K2_arr = np.vstack([[np.zeros_like(K2s_mat) for _ in range(2)],Tl_k_func(K2s_mat)])

# Define ell arrays and custom harmonic-space weighting
l = np.arange(lmax+1)
Sl = np.load('planck/Sl_weighting.npy')

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

########################### Q INTEGRALS ##################################

integ_time = time.time()

# Define J functions
def JJn(n,a_b):
    alpha = 0.5+n-nu
    beta = 0.5+n+nu
    pref = (2.)**(-n-1./2.)*gamma(alpha)*gamma(beta)/gamma(1.+n)
    return pref*hyp2f1(alpha,beta,1.+n,0.5-a_b/2.)

# Compute useful functions
K12_arr = K1_arr+K2_arr
K12sqdK_arr = K1_arr**2./(2.*np.pi**2)*K2_arr**2./(2.*np.pi**2)*du*dv/2.

# Assemble Jn functions
print("Computing Jn functions")
JJ3_arr = JJn(3,cs*K12_arr)
JJ4_arr = JJn(4,cs*K12_arr)
JJ5_arr = JJn(5,cs*K12_arr)

# Assemble t^A and t^B integrals
ttA_integ = 1./K1_arr**2*1./K2_arr*(K1_arr-K2_arr)*(K12_arr*JJ3_arr+cs*K1_arr*K2_arr*JJ4_arr)
ttB_integ = 1./K1_arr*1./K2_arr*(K1_arr-K2_arr)*(K12_arr*JJ4_arr+cs*K1_arr*K2_arr*JJ5_arr)

# Compute Delta matrix and Legendre multipoles
DDelta_mat = (K1_arr**2+K2_arr**2-1.)/(2.*K1_arr*K2_arr)
leg_DDelta = np.empty((2*lmax+5,len(K1_arr[0,:,0]),len(K2_arr[0,0,:])))
for l in range(2*lmax+5):
    leg_DDelta[l] = legendre(l)(DDelta_mat)

ff_pref = (np.abs(DDelta_mat)<1.)/(4.*K1_arr*K2_arr)
K21_ratio = (K2_arr/K1_arr)

from scipy.special import binom, legendre

lmax_reg = np.max(l_bins)
lmax_sq = np.max(l_bins_squeeze)
Lmax = np.max(L_bins)

# Arrays for (approximate) explicit integration
x_arr = np.arange(0.,100,0.5)[:,np.newaxis,np.newaxis]
dx = np.diff(x_arr.ravel())[0]
K1x = K1_arr*x_arr
K2x = K2_arr*x_arr
print("Loading Bessel functions")
jl1s = [spherical_jn(L1,K1x) for L1 in range(lmax_reg+3)]
jl2s = [spherical_jn(L2,K2x) for L2 in range(lmax_sq+3)]
jlxs = [spherical_jn(Lp,x_arr) for Lp in range(Lmax+2)]

def compute_f(L1):
    f1 = []
    for L2 in range(lmax_sq+3):
        f2 = []
        for Lp in range(abs(L1-L2),min(L1+L2+1,Lmax+2),2):
            
            tj_init = tj0(L1,L2,Lp)
            if tj_init == 0.: 
                f2.append(0.*K21_ratio)
                continue

            logprod1 = Lp*np.log(K1_arr)
            pref = 1./tj_init*np.pi*(-1.)**((L1+L2-Lp)/2.)*np.sqrt(2.*Lp+1.)*ff_pref
            summ = 0.
            for lam in range(0,Lp+1):
                pref2 = binom(2*Lp,2*lam)**0.5
                logprod = logprod1+lam*np.log(K21_ratio)
                tmp_sum = 0.
                for l in range(abs(L2-lam),L2+lam+1,2):
                    tt=tj0(L1,Lp-lam,l)
                    if tt==0: continue
                    tt*=tj0(L2,lam,l)
                    if tt==0: continue
                    tt *= sixj(L1,L2,Lp,lam,Lp-lam,l)
                    if tt==0: continue
                    tmp_sum += (2*l+1.)*tt*leg_DDelta[l]
                summ += tmp_sum*pref2*np.exp(logprod)
            summ2 = summ*pref
            summ2[summ2==0] = 1e-16
            
            # Add explicit integration to catch and remove overflow errors at high v
            summ2_explicit = np.sum(x_arr**2*jl1s[L1]*jl2s[L2]*jlxs[Lp]*dx,axis=0)[np.newaxis,:,:]
            
            # Filter out bad values
            filt = ((np.abs(summ2_explicit)/np.abs(summ2[0])>5)|(np.abs(summ2_explicit)/np.abs(summ2[0])<0.2))
            summ2_corrected = summ2*(filt==0)+summ2_explicit*(filt!=0)
            
            f2.append(summ2_corrected)
        f1.append(f2)
    return f1

print("Multiprocessing Bessel function integrals")
p = mp.Pool(Ncpus)
f_ints = list(tqdm.tqdm(p.imap(compute_f,np.arange(lmax_reg+3)),total=lmax_reg+3))
p.close()
p.join()

integ_time = time.time()-integ_time
print("Q integral set-up took %.3f seconds"%integ_time)

# Compute f functions, i.e. analytic integrals over x.
def computeQ(L1,L2,Lp,l1,l2,Qtype='A'):
    """Compute the Q functions, integrating over k1, k2. This computes either Q^A or Q^B depending on `Qtype'.
    """
    tj_init = tj0(L1,L2,Lp)
    if tj_init == 0.: return 0.*s_arr
    
    summ = f_ints[L1][L2][(Lp-abs(L1-L2))//2]
       
    # Perform integrals
    if Qtype=='A':
        integrand = ttA_integ*summ
    elif Qtype=='B':
        integrand = ttB_integ*summ
    else:
        raise Exception("Q-type must be 'A' or 'B'")
    
    # Integrate over K1, K2
    f_s = np.sum(integrand*Tl_K1_arr[l1]*Tl_K2_arr[l2]*K12sqdK_arr,axis=(1,2))
    
    # Add dimension factor
    Q_out = f_s/s_arr**(3./2.)
    
    return Q_out

def coupling_matrix1(l1,l2,L1,L2,L,Lp):
    """Compute the M coupling matrix"""

    # Assemble matrix prefactor
    pref = (2*L1+1)*(2*L2+1)*tj0(L1,L2,Lp)*tj0(L1,1,l1)
    if pref==0: return 0.
    
    # term1
    term1 =2.*np.sqrt(5.)*tj0(L2,2,l2)*ninej(L1,L2,Lp,1,2,1,l1,l2,L)
    
    # term2
    term2 = np.sqrt(2./3.)/(2*l2+1)*(l2==L2)*(-1.)**(l1+Lp)*sixj(L,l1,l2,L1,Lp,1)
    
    return pref*(term1+term2)

def coupling_matrix2(l3,l4,L3,L4,L,Lp):
    """Compute the N coupling matrix"""

    # Assemble matrix
    pref = (2*L3+1)*(2*L4+1)*(2*Lp+1)*tj0(L3,L4,Lp)*tj0(L3,2,l3)*tj0(L4,2,l4)
    if pref==0: return 0.
    
    term = ninej(1,2,2,Lp,L3,L4,L,l3,l4)
    
    return pref*term

########################### COMPUTE TRISPECTRUM ###########################

C_ell = lambda Ls: np.sqrt(np.product(2.*np.asarray(Ls)+1.))

def _compute_trispectrum(index):
    bin1,bin2,bin3,bin4,binL = bins[index]
    
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
    pref = np.sqrt(5.)/6.*(4.*np.pi)**(5.)*DeltaSq_zeta**4.

    # Assemble theory prediction
    output = 0.    
    for L in range(max(2,L_bins[binL]),L_bins[binL+1]):    
        
        # 2L+1 factor from bin weighting
        pref2 = (-1.)**L*(2*L+1)
            
        for Lp in range(abs(L-2),L+3,1):

            # Perform sums over l1, l2
            first_term1,first_term2 = [np.zeros(len(s_arr.ravel()),dtype='complex128') for _ in range(2)]
            for l1 in range(max(1,l_bins[bin1]),l_bins[bin1+1]):
                for l2 in range(max(1,l_bins_squeeze[bin2]),l_bins_squeeze[bin2+1]):
        
                    # Check triangle conditions
                    if check(l1,l2,L,False): continue

                    # Compute bin weighting (dividing by external 3j symbols)
                    bin_weight = tjSpin(l1,l2,L)*(2*l1+1)*(2*l2+1)/Sl[l1]/Sl[l2]/(4.*np.pi)
                    if bin_weight==0: continue
                        
                    for L1 in range(abs(l1-1),l1+2,1):
                        for L2 in range(abs(l2-2),l2+3,2):
                            
                            # Compute first coupling
                            
                            M_mat = coupling_matrix1(l1,l2,L1,L2,L,Lp)
                            if M_mat==0: continue
                            QA = computeQ(L1,L2,Lp,l1,l2,'A')
                            
                            first_term = bin_weight*(1.0j)**(l1+l2-L1-L2)*QA*M_mat
                            first_term1 += first_term
                            first_term2 += (-1.)**(l1+l2)*first_term
            
            # Perform sums over l3,l4
            second_term1, second_term2 = [np.zeros((len(s_arr.ravel())),dtype='complex128') for _ in range(2)]
            for l3 in range(max(1,l_bins[bin3]),l_bins[bin3+1]):
                for l4 in range(max(1,l_bins_squeeze[bin4]),l_bins_squeeze[bin4+1]):

                    # Check triangle conditions
                    if check(l3,l4,L,False): continue

                    # Compute bin weighting (dividing by external 3j symbols)
                    bin_weight = tjSpin(l3,l4,L)*(2*l3+1)*(2*l4+1)/Sl[l3]/Sl[l4]/(4.*np.pi)
                    if bin_weight==0: continue
                    
                    for L3 in range(abs(l3-2),l3+3,2):
                        for L4 in range(abs(l4-2),l4+3,2):
                            
                            # Compute second coupling
                            N_mat = coupling_matrix2(l3,l4,L3,L4,L,Lp)
                            if N_mat==0: continue
                            QB = computeQ(L3,L4,Lp,l3,l4,'B')
                            
                            second_term = bin_weight*(1.0j)**(l3+l4-L3-L4)*QB*N_mat
                            second_term1 += second_term
                            second_term2 += (-1.)**(l3+l4)*second_term
            
            # Assemble integral
            this_term = pref*pref2
            integ = 0.5*(first_term1*second_term1-first_term2*second_term2)
            this_term *= np.sum(dlogs*s_arr**3*integ)/(2.*np.pi**2.)
            output += this_term.imag
                        
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
print("Multiprocessing complete after %.2f s"%(time.time()-t_start))

outfile = 'collider_%.2f,%.2f_trispectrum_%s_l(%d,%d,%d).txt'%(cs,nu,transfer,Nl,Nl_squeeze,NL)
np.savetxt(outfile,tl_out)

print("Saved output to %s after %.1f s; exiting."%(outfile,time.time()-init_time))
