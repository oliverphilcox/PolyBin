### Compute the odd-parity CMB trispectrum arising from the ghost inflation model as in Cabass+22
# The output contains the M_PO and Lambda^2_PO trispectra, but without the relevant prefactors

import sys, os
sys.path.append('/home/ophilcox/wigxjpf-1.11/')
import numpy as np, time, tqdm, multiprocessing as mp, itertools
import camb
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn, hyp2f1, gamma, legendre, hankel1
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

dq = 0.001
qmax = 5.
logy_min=-2
logy_max = 3.5
Ny = 100
loglam_min=-6
loglam_max=0.5
Nlam = 100

q_arr = np.arange(dq/100,qmax,dq)[:,None,None]
y_arr = np.logspace(logy_min,logy_max,Ny)[None,:,None]
lam_arr = np.logspace(loglam_min,loglam_max,Nlam)[None,None,:]
print("N_q: %d"%len(q_arr.ravel()))
dlogy = np.diff(np.log(y_arr.ravel()))[0]
dloglam = np.diff(np.log(lam_arr.ravel()))[0]

print("N_y: %d"%len(y_arr.ravel()))
print("N_lambda: %d"%len(lam_arr.ravel()))

# Compute distance to last scattering (or peak thereof)
back = camb.get_background(pars)
r_star = (back.tau0-back.tau_maxvis)

# Load transfer function
q_lam_arr = (q_arr*lam_arr)[:,0,:]
if transfer=='SW':
    print("Assuming Sachs-Wolfe transfer function")
    Tl_k_func = lambda k: np.asarray([-(l>=2)*(1./5.)*spherical_jn(l,k*r_star) for l in range(lmax+3)])
    Tl_qlam_arr = Tl_k_func(q_lam_arr)
elif transfer=='full':
    print("Using full transfer functions")
    # Read in transfer functions computed from CAMB
    transfer_inp = np.load('planck/camb_transfer.npz')
    Tl_k_func = interp1d(transfer_inp['k'],transfer_inp['transfer'][:lmax+1],bounds_error=False,fill_value=0.)
    Tl_qlam_arr = np.vstack([[np.zeros_like(q_lam_arr) for _ in range(2)],Tl_k_func(q_lam_arr)])

# Compute H_{alpha}(2iq^2) possibilities
print("Computing Hankel functions")
Hankel34=hankel1(3./4.,2.*1.0j*q_arr**2.)
Hankelm14=hankel1(-1./4.,2.*1.0j*q_arr**2.)

### Compute the various g integrals
print("Computing g integrals")
q2_norm = q_arr**2./(2.*np.pi**2.)
qrav = q_arr.ravel()

J_12_34_pref = (q_arr**0.5*q2_norm*Hankel34)[:,0,:]*Tl_qlam_arr*dq
J_32_34_pref = (q_arr**1.5*q2_norm*Hankel34)[:,0,:]*Tl_qlam_arr*dq
J_52_34_pref = (q_arr**2.5*q2_norm*Hankel34)[:,0,:]*Tl_qlam_arr*dq
J_12_m14_pref = (q_arr**0.5*q2_norm*Hankelm14)[:,0,:]*Tl_qlam_arr*dq

# Compute all spherical Bessel functions
qy_arr = (q_arr*y_arr)[:,:,0]

jls_qy = np.asarray([spherical_jn(L,qy_arr) for L in range(lmax+4)])

g_0_12,g_0_32,g_0_52,g_1_12 = [],[],[],[]
for l in range(lmax+1):
    gsA,gsB,gsC,gsD = [],[],[],[]
    for L in range(abs(l-3),l+4):
        this_jl = jls_qy[L]
        gsA.append((J_12_34_pref[l].T@this_jl).T)
        gsB.append((J_32_34_pref[l].T@this_jl).T)
        gsC.append((J_52_34_pref[l].T@this_jl).T)
        gsD.append((J_12_m14_pref[l].T@this_jl).T)
    g_0_12.append(gsA)
    g_0_32.append(gsB)
    g_0_52.append(gsC)
    g_1_12.append(gsD)

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
    """Compute ghost inflation trispectra (both types)."""
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
    
    if weighting==0: return 0.,0.
    
    # Compute prefactors
    prefA = 2.*(4.*np.pi)**7.*DeltaSq_zeta**3.
    prefB = -40.*np.sqrt(2./15.)*(4.*np.pi)**7.*DeltaSq_zeta**3.

    # Assemble theory prediction
    outputA, outputB = 0.,0.    
    for L in range(max(2,L_bins[binL]),L_bins[binL+1]):    
    
        # M-PO contribution
        termA = 0.

        # Basis function expansion coefficients
        coeffs = [[0,1,1,2,2,1.],[0,3,3,2,2,-2.],[2,1,1,2,2,4.*np.sqrt(2.)/5.],[2,1,2,2,0,1.],[2,1,2,2,2,-np.sqrt(14./5.)],
    [2,1,3,2,2,2.*np.sqrt(7.)/5.],[2,3,1,2,2,np.sqrt(3.)/5.],[2,3,2,2,0,-2.],[2,3,2,2,2,-np.sqrt(2./35.)],
    [2,3,3,2,2,-2.*np.sqrt(3.)/5.],[2,3,4,2,2,6./np.sqrt(7.)]]
        for i in range(len(coeffs)):
            ll1,ll2,llp,ll3,ll4,c_ll = coeffs[i]
    
            c_ll *= np.sqrt(10.)/225.

            pref2 = c_ll*C_ell([ll1,ll2,ll3,ll4,llp])

            for Lp in range(abs(L-llp),L+llp+1):

                # Compute sums over L1,L2,l1,l2
                first_term1, first_term2 = 0.,0.

                for l1 in range(max(1,l_bins[bin1]),l_bins[bin1+1]):
                    for l2 in range(max(1,l_bins_squeeze[bin2]),l_bins_squeeze[bin2+1]):
        
                        # Check triangle conditions
                        if check(l1,l2,L,False): continue

                        tjspin1 = tjSpin(l1,l2,L)
                        if tjspin1==0: continue
                        first_term_tmp = (1.0j)**(l1+l2)*tjspin1*(2*l1+1)*(2*l2+1)/Sl[l1]/Sl[l2]/(4*np.pi)

                        for L1 in range(abs(l1-ll1),l1+ll1+1,2):
                            for L2 in range(abs(l2-ll2),l2+ll2+1,2):
                                first_term = (1.0j)**(-L1-L2)*(2*L1+1)*(2*L2+1)*tj0(L1,L2,Lp)*tj0(L1,ll1,l1)*tj0(L2,ll2,l2)
                                if first_term==0: continue
                                first_term *= ninej(L1,L2,Lp,ll1,ll2,llp,l1,l2,L)
                                if first_term==0: continue
                                first_term *= g_0_12[l1][L1-abs(l1-3)]*g_0_32[l2][L2-abs(l2-3)]
                                # Add ell weighting and 1/ThreeJ
                                first_term *= first_term_tmp
                                # Compute parity terms
                                first_term1 += first_term
                                first_term2 += (-1.)**(l1+l2)*first_term
                
                # Compute sums over L3,L4,l3,l4
                second_term1, second_term2 = 0.,0.

                for l3 in range(max(1,l_bins[bin3]),l_bins[bin3+1]):
                    for l4 in range(max(1,l_bins_squeeze[bin4]),l_bins_squeeze[bin4+1]):

                        # Check triangle conditions
                        if check(l3,l4,L,False): continue

                        tjspin2 = tjSpin(l3,l4,L)
                        if tjspin2==0: continue
                        second_term_tmp = (1.0j)**(l3+l4)*tjspin2*(2*l3+1)*(2*l4+1)/Sl[l3]/Sl[l4]/(4*np.pi)

                        for L3 in range(abs(l3-ll3),l3+ll3+1,2):
                            for L4 in range(abs(l4-ll4),l4+ll4+1,2):
                                second_term = (1.0j)**(-L3-L4)*(2.*L3+1.)*(2*L4+1.)*tj0(L3,L4,Lp)*tj0(L3,ll3,l3)*tj0(L4,ll4,l4)
                                if second_term==0: continue
                                second_term *= ninej(Lp,L3,L4,llp,ll3,ll4,L,l3,l4)
                                if second_term==0: continue
                                second_term *= g_0_12[l3][L3-abs(l3-3)]*g_0_12[l4][L4-abs(l4-3)]
                                # Add ell weighting and 1/ThreeJ 
                                second_term *= second_term_tmp
                                second_term1 += second_term
                                second_term2 += (-1.)**(l3+l4)*second_term

                # Combine, including (2L+1) weighting
                this_term = prefA*pref2*(-1.)**L*(2.*Lp+1.)*(2*L+1.)
                integ = 0.5*(first_term1*second_term1-first_term2*second_term2)
                this_term *= np.sum(integ*y_arr**3.*dlogy*dloglam)
                termA += this_term

        # Lambda-PO contribution
        termB = 0.

        for Lp in range(abs(L-1),L+2):

            # Compute sums over L1,L2,l1,l2
            first_term1, first_term2 = 0.,0.

            for l1 in range(max(1,l_bins[bin1]),l_bins[bin1+1]):
                for l2 in range(max(1,l_bins_squeeze[bin2]),l_bins_squeeze[bin2+1]):
        
                    # Check triangle conditions
                    if check(l1,l2,L,False): continue

                    tjspin1 = tjSpin(l1,l2,L)
                    if tjspin1==0: continue
                    first_term_tmp = (1.0j)**(l1+l2)*tjspin1*(2*l1+1)*(2*l2+1)/Sl[l1]/Sl[l2]/(4*np.pi)

                    for L1 in range(abs(l1-2),l1+3,2):
                        for L2 in range(abs(l2-2),l2+3,2):
                            first_term = (1.0j)**(-L1-L2)*(2*L1+1)*(2*L2+1)*tj0(L1,L2,Lp)*tj0(L1,2,l1)*tj0(L2,2,l2)
                            if first_term==0: continue
                            first_term *= ninej(L1,L2,Lp,2,2,1,l1,l2,L)
                            if first_term==0: continue
                            first_term *= g_0_12[l1][L1-abs(l1-3)]*g_0_52[l2][L2-abs(l2-3)]
                            # Add ell weighting and 1/ThreeJ
                            first_term *= first_term_tmp
                            # Compute parity terms
                            first_term1 += first_term
                            first_term2 += (-1.)**(l1+l2)*first_term
            
            # Compute sums over L3,L4,l3,l4
            second_term1, second_term2 = 0.,0.
            
            for l3 in range(max(1,l_bins[bin3]),l_bins[bin3+1]):
                for l4 in range(max(1,l_bins_squeeze[bin4]),l_bins_squeeze[bin4+1]):

                    # Check triangle conditions
                    if check(l3,l4,L,False): continue

                    tjspin2 = tjSpin(l3,l4,L)
                    if tjspin2==0: continue
                    second_term_tmp = (1.0j)**(-l3)*g_1_12[l4][l4-abs(l4-3)]*tjspin2*(2*l3+1)*(2*l4+1)/Sl[l3]/Sl[l4]/(4*np.pi)

                    for L3 in range(abs(l3-1),l3+2,2):
                        second_term = (1.0j)**(-L3)*(2.*L3+1.)*tj0(L3,l4,Lp)*tj0(L3,1,l3)
                        if second_term==0: continue
                        second_term *= sixj(Lp,L3,l4,l3,L,1)
                        if second_term==0: continue
                        second_term *= g_0_32[l3][L3-abs(l3-3)]
                        # Add ell weighting and 1/ThreeJ 
                        second_term *= second_term_tmp
                        second_term1 += second_term
                        second_term2 += (-1.)**(l3+l4)*second_term
            
            # Combine, including (2L+1) weighting
            this_term = prefB*(-1.)**(L+Lp)*(2.*Lp+1.)*(2*L+1.)
            integ = 0.5*(first_term1*second_term1-first_term2*second_term2)
            this_term *= np.sum(integ*y_arr**3.*dlogy*dloglam)
            termB += this_term

        # NB: taking (-)real part of A due to Im in definition
        outputA += -termA.real
        outputB += termB.imag
                        
    if weighting!=0:
        outputA = outputA/weighting
        outputB = outputB/weighting
    else:
        raise Exception()
    return outputA,outputB

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

outfile = 'ghost_trispectrum_%s_weighted_l(%d,%d,%d).txt'%(transfer,Nl,Nl_squeeze,NL)
np.savetxt(outfile,tl_out)

print("Saved output to %s after %.1f s; exiting."%(outfile,time.time()-init_time))
