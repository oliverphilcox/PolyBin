### Compute the Odd-Parity CMB trispectrum arising from the cosmological collider as in Philcox 22
# The output contains the trispectrum, but without a prefactor of cs^4 lambda_1 lambda_3 sin pi(nu+1/2)/(H)

import sys
sys.path.append('/home/ophilcox/wigxjpf-1.11/')
import numpy as np, time, tqdm, multiprocessing as mp, itertools
from classy import Class
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import spherical_jn, hyp2f1, gamma, legendre
import pywigxjpf as wig

if __name__=='__main__':

    if len(sys.argv)!=4: 
        raise Exception("LMAX and nu not specified!")

    # Maximum ell
    LMAX = int(sys.argv[1])
    nu = float(sys.argv[2])
    cs = float(sys.argv[3])

    init_time = time.time()
    
    ### Cosmology
    cosmo = Class()

    # Run CLASS
    cosmo.set({'output':'tCl,lCl,mPk','l_max_scalars':1001,'lensing':'yes',
            'omega_b':0.022383,
            'non linear':'no',
            'omega_cdm':0.12011,
            'h':0.6732,
            'm_ncdm':0.06,
            'N_ncdm':1,
            'N_ur':2.0328,
            'T_ncdm':0.71611,
            'tau_reio':0.0543,
            'A_s':1e-10*np.exp(3.0448),
            'k_pivot':0.05,
            'YHe':'BBN',
            'n_s':0.96605})
    cosmo.compute()
    DeltaSq_zeta = cosmo.A_s()*2*np.pi**2 # primordial amplitude (from Planck)

    # CPU-cores
    cores = 24

    print("ell_max: %d"%LMAX)
    print("nu parameter: %.2f"%nu)
    print("Sound speed: %.2f"%cs)

    ## Precision parameters on k, s
    dk = 0.005
    kmax = 0.5
    ds = 0.01
    smax = 2*kmax

    # Compute k, s arrays
    k_arr = np.arange(dk/100,kmax,dk)
    s0_arr = np.arange(ds/100,smax,ds)

    k1_arr = k_arr[:,None,None]
    k2_arr = k_arr[None,:,None]
    s_arr = s0_arr[None,None,:]
    print("N_k: %d"%len(k_arr.ravel()))
    print("N_s: %d"%len(s_arr.ravel()))

    # Compute transfer function for each ell
    print("Assuming Sachs-Wolfe transfer function")
    r_star = cosmo.rs_drag()
    Tl_k_arr = [-(1./5.)*spherical_jn(l,k_arr*r_star) for l in range(lmax+1)]

    # 3j/9j set-up
    wig.wig_table_init(2*(2*LMAX),9)
    wig.wig_temp_init(2*(2*LMAX))

    # Wigner 3j + 9j
    tj0 = lambda l1, l2, l3: wig.wig3jj(2*l1 , 2*l2, 2*l3, 0, 0, 0)
    tjSpin = lambda l1, l2, l3: wig.wig3jj(2*l1, 2*l2, 2*l3, 2*-1, 2*-1, 2*2)
    sixj = lambda l1,l2,l3,l4,l5,l6: wig.wig6jj(2*l1,2*l2,2*l3,2*l4,2*l5,2*l6)
    ninej = lambda l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3: wig.wig9jj(2*l1,2*l2,2*l3,2*lp1,2*lp2,2*lp3,2*lpp1,2*lpp2,2*lpp3)

    ########################### Q INTEGRALS ##################################

    integ_time = time.time()

    # Define J functions
    def Jn(n,a,b):
        alpha = 0.5+n-nu
        beta = 0.5+n+nu
        pref = (2.*b)**(-n-1./2.)*gamma(alpha)*gamma(beta)/gamma(1.+n)
        return pref*hyp2f1(alpha,beta,1.+n,0.5-a/(2.*b))

    # Compute useful functions
    k12_arr = k1_arr+k2_arr
    ksq_arr = k_arr**2./(2.*np.pi**2)
    
    # Assemble Jn functions
    print("Computing Jn functions")
    J3_arr = Jn(3,cs*k12_arr,s_arr)
    J4_arr = Jn(4,cs*k12_arr,s_arr)
    J5_arr = Jn(5,cs*k12_arr,s_arr)

    # Assemble t^A and t^B integrals
    tA_integ = 1./k1_arr**2*1./k2_arr*(k1_arr-k2_arr)*(k12_arr*J3_arr+cs*k1_arr*k2_arr*J4_arr)
    tB_integ = 1./k1_arr*1./k2_arr*(k1_arr-k2_arr)*(k12_arr*J4_arr+cs*k1_arr*k2_arr*J5_arr)

    # Compute Delta matrix and Legendre multipoles
    Delta_mat = (k1_arr**2+k2_arr**2-s_arr**2.)/(2.*k1_arr*k2_arr)
    leg_Delta = np.empty((2*LMAX+1,len(k_arr),len(k_arr),len(s_arr.ravel())))
    for l in range(2*LMAX+1):
        leg_Delta[l] = legendre(l)(Delta_mat)

    # Compute j_ell(kr) possibilities
    # add one extra row for the r = 0 case!
    jell_kr = []
    print("Computing bin-averaged j_ell(kr)")
    for ell in range(LMAX_data+1):
        tmp_jell_kr = []
        for rbin in range(n_r):
            tmp_jell_kr.append(integ_bessel(ell,rbin,k_arr))
        tmp_jell_kr.append(np.ones_like(integ_bessel(ell,rbin,k_arr)))
        jell_kr.append(tmp_jell_kr)

    # Compute f functions, i.e. analytic integrals over x.
    from scipy.special import binom, legendre

    def f_integral(L1,L2,Lp,acc=None):
        """Compute the integral over x for an array of k1, k2, s for given {L1, L2, L'} inputs.

        This is a (finite) sum over Legendre polynomials in the (k1.k2) angle."""
        tj_init = tj0(L1,L2,Lp)
        if tj_init==0:
            raise Exception("L1+L2+L' should be even and obey triangle conditions!")

        pref = 1./tj_init*np.pi*(np.abs(Delta_mat)<1.)/(4.*k1_arr*k2_arr*s_arr)*(-1.)**((L1+L2-Lp)/2.)*np.sqrt(2.*Lp+1.)*(k1_arr/s_arr)**Lp
        summ = 0.
        for lam in range(0,Lp+1):
            pref2 = binom(2*Lp,2*lam)**0.5*(k2_arr/k1_arr)**lam
            summ2 = 0.
            for l in range(abs(L2-lam),L2+lam+1,2):
                tt=acc.tj0(L1,Lp-lam,l)
                if tt==0: continue
                tt*=acc.tj0(L2,lam,l)
                if tt==0: continue
                tt *= acc.sixj(L1,L2,Lp,lam,Lp-lam,l)
                if tt==0: continue
                summ2 += (2.*l+1.)*tt*leg_Delta[l]
            summ += pref2*summ2

        return pref*summ

    def computeQ(L1,L2,Lp,l1,l2,rBin1,rBin2,Qtype='A',acc=None):
        """Compute the Q functions, integrating over k1, k2. This computes either Q^A or Q^B depending on `Qtype'.

        We also take a 3j/6j/9j accelerator function as input."""
        ## Assemble Q integrand
        # x integral
        Q_integrand = f_integral(L1,L2,Lp,acc=acc)
        # k1 integral
        Q_integrand *= (kM_arr*jell_kr[l1][rBin1]).reshape(-1,1,1)
        # k2 integral
        Q_integrand *= (kM_arr*jell_kr[l2][rBin2]).reshape(1,-1,1)

        # Perform integrals
        if Qtype=='A':
            Q_out = dk**2*np.sum(tA_integ*Q_integrand,axis=(0,1))
        elif Qtype=='B':
            Q_out = dk**2*np.sum(tB_integ*Q_integrand,axis=(0,1))
        else:
            raise Exception("Q-type must be 'A' or 'B'")

        return Q_out

    integ_time = time.time()-integ_time
    print("Q integral set-up took %.3f seconds"%integ_time)

    ########################### COMPUTE RADIAL INTEGRALS ###########################

    C_ell = lambda Ls: np.sqrt(np.product(2.*np.asarray(Ls)+1.))

    def Z_ell(ell):
        """Compute the Z_ell coefficients, which are the spherical harmonic expansion of the Kaiser kernel"""
        if ell==0:
            return (b+fz/3.)
        elif ell==2:
            return (2./15.*fz)
        else:
            raise Exception('Wrong ell!')

    def load_integrals(lH1,lH2,lH3,lH4,bH1,bH2,bH3,bH4,L1,L2,L3,L4,Lp,phiH,tjprod=np.inf,acc=None):
        """Load contribution to 4PCF involving radial bins (but not the coupling matrix), and perform the s integral.

        We keep only the imaginary part (the real part is ~ 0)"""

        # Compute 3j symbols if needed
        if tjprod!=np.inf:
            pref = tjprod
        else:
            pref = acc.tj0(L1,L2,Lp)*acc.tj0(Lp,L3,L4)
            if pref==0: return 0.

        pref *= (4.*np.pi)**(7./2.)*(-1.0j)**(lH1+lH2+lH3+lH4)/(18.*np.sqrt(5.))*phiH*(1.0j)**(L1+L2+L3+L4)
        pref *= C_ell([L1,L2,L3,L4,Lp])

        integ = s_arr**2./(2.*np.pi**2.)*computeQ(L1,L2,Lp,lH1,lH2,bH1,bH2,'A',acc=acc)*computeQ(L3,L4,Lp,lH3,lH4,bH3,bH4,'B',acc=acc)
        out = pref*simps(integ,s_arr.ravel())
        return out.imag

    ########################### COUPLING MATRICES ###########################

    def check(l1,l2,l3,even=True):
        """Check triangle conditions for a triplet of momenta"""
        if l1<abs(l2-l3): return True
        if l1>l2+l3: return True
        if even:
            if (-1)**(l1+l2+l3)==-1: return True
        return False

    class wigner_acc():
        def __init__(self,nmax):
            """Accelerator for 3j, 6j and 9j computation. This only computes a symbol explicitly if it has not been computed before."""
            self.recent_inputs9 = []
            self.recent_outputs9 = []
            self.recent_inputs6 = []
            self.recent_outputs6 = []
            self.recent_inputs3 = []
            self.recent_outputs3 = []
            self.nmax = nmax
            self.calls9 = 0
            self.computes9 = 0
            self.calls6 = 0
            self.computes6 = 0
            self.calls3 = 0
            self.computes3 = 0

        def ninej(self,l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3):
            """9j accelerator"""
            self.calls9 += 1
            if [l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3] in self.recent_inputs9:
                return self.recent_outputs9[self.recent_inputs9.index([l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3])]
            else:
                val = ninej(l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3)
                self.computes9+=1
                self.recent_inputs9.append([l1,l2,l3,lp1,lp2,lp3,lpp1,lpp2,lpp3])
                self.recent_outputs9.append(val)
            if len(self.recent_inputs9)>self.nmax:
                self.recent_inputs9 = self.recent_inputs9[1:]
                self.recent_outputs9 = self.recent_outputs9[1:]
            return val

        def sixj(self,l1,l2,l3,l4,l5,l6):
            """6j accelerator"""
            self.calls6 += 1
            if [l1,l2,l3,l4,l5,l6] in self.recent_inputs6:
                return self.recent_outputs6[self.recent_inputs6.index([l1,l2,l3,l4,l5,l6])]
            else:
                val = sixj(l1,l2,l3,l4,l5,l6)
                self.computes6+=1
                self.recent_inputs6.append([l1,l2,l3,l4,l5,l6])
                self.recent_outputs6.append(val)
            if len(self.recent_inputs6)>self.nmax:
                self.recent_inputs6 = self.recent_inputs6[1:]
                self.recent_outputs6 = self.recent_outputs6[1:]
            return val

        def tj0(self,l1,l2,l3):
            """3j (m1=m2=m3=0) accelerator"""
            self.calls3 += 1
            if [l1,l2,l3] in self.recent_inputs3:
                return self.recent_outputs3[self.recent_inputs3.index([l1,l2,l3])]
            else:
                val = tj0(l1,l2,l3)
                self.computes3+=1
                self.recent_inputs3.append([l1,l2,l3])
                self.recent_outputs3.append(val)
            if len(self.recent_inputs3)>self.nmax:
                self.recent_inputs3 = self.recent_inputs3[1:]
                self.recent_outputs3 = self.recent_outputs3[1:]
            return val

    def coupling_matrix(lH1,lH2,lH12,lH3,lH4,L1,L2,L3,L4,Lp,acc=None):
        """Compute the M coupling matrix"""

        # Assemble matrix prefactor
        pref = 15.*C_ell([lH1,lH2,lH12,lH3,lH4])*C_ell([L1,L2,Lp,L3,L4])
        output = 0.

        # Sum over lambda
        for lam1 in range(abs(L1-1),L1+2,2):
            tj1 = acc.tj0(1,L1,lam1)

            for lam2 in range(abs(L2-2),L2+3,2):
                tj12a = tj1*acc.tj0(2,L2,lam2)
                tj12b = tj1*acc.tj0(0,L2,lam2)

                for lam12 in range(abs(lam1-lam2),lam1+lam2+1):
                    if check(1,Lp,lam12,False): continue

                    for lam3 in range(abs(L3-2),L3+3,2):
                        tj123a = tj12a*acc.tj0(2,L3,lam3)
                        tj123b = tj12b*acc.tj0(2,L3,lam3)

                        for lam4 in range(abs(L4-2),L4+3,2):
                            if check(lam12,lam3,lam4,False): continue
                            tj1234a = tj123a*acc.tj0(2,L4,lam4)
                            tj1234b = tj123b*acc.tj0(2,L4,lam4)

                            # lambda factor
                            lam_piece = (-1)**(lam1+lam2+lam3+lam4)*C_ell([lam1,lam2,lam12,lam3,lam4])**2.

                            # nine-js (j-independent)
                            nj1 = acc.ninej(1,2,2,Lp,L3,L4,lam12,lam3,lam4)
                            if nj1==0: continue
                            tjnj1a = tj1234a*nj1*acc.ninej(1,2,1,L1,L2,Lp,lam1,lam2,lam12)
                            tjnj1b = tj1234b*nj1*acc.ninej(1,0,1,L1,L2,Lp,lam1,lam2,lam12)
                            # Combine the two terms
                            tjnj1 = 2.*np.sqrt(5.)*tjnj1a-np.sqrt(2.)*tjnj1b
                            if tjnj1==0: continue

                            # Sum over j
                            for j1 in [0,2]:
                                if check(lH1,lam1,j1): continue
                                ttj1 = acc.tj0(j1,lH1,lam1)

                                for j2 in [0,2]:
                                    if check(lH2,lam2,j2): continue
                                    ttj12 = ttj1*acc.tj0(j2,lH2,lam2)

                                    for j12 in [0,2,4]:
                                        if check(j1,j2,j12): continue
                                        if check(lH12,lam12,j12,0): continue

                                        # Assemble j pieces
                                        j_piece1 = acc.tj0(j1,j2,j12)

                                        for j3 in [0,2]:
                                            if check(lH3,lam3,j3): continue
                                            ttj123 = ttj12*acc.tj0(j3,lH3,lam3)

                                            for j4 in [0,2]:
                                                if check(j12,j3,j4): continue
                                                if check(lH4,lam4,j4): continue
                                                ttj1234 = ttj123*acc.tj0(j4,lH4,lam4)

                                                j_piece12 = j_piece1*acc.tj0(j12,j3,j4)
                                                j_piece12 *= Z_ell(j1)*Z_ell(j2)*Z_ell(j3)*Z_ell(j4)*C_ell([j1,j2,j12,j3,j4])**2.

                                                # nine-js (j-dependent)
                                                nj2 = acc.ninej(j1,j2,j12,lH1,lH2,lH12,lam1,lam2,lam12)
                                                if nj2==0: continue
                                                nj2 *= acc.ninej(j12,j3,j4,lH12,lH3,lH4,lam12,lam3,L4)
                                                if nj2==0: continue

                                                output += pref*j_piece12*lam_piece*ttj1234*tjnj1*nj2

        return output

    ########################### COMPUTE ZETA FOR SINGLE PERMUTATION ###########################

    # Permutations of [0,1,2,3]
    indices=set(itertools.permutations([0,1,2,3]))

    def load_perm(index_id):
        """Load 4PCF contributions from a single permutation."""

        sum_output = np.zeros((n_l,int(n_r*(n_r-1)*(n_r-2)/6)))
        t_coupling,t_matrix = 0.,0.

        # Create 3j/6j/9j accelerator (to be threadsafe)
        acc = wigner_acc(100000)

        # Sum over Ls
        for L1 in range(LMAX+1):
            for L2 in range(LMAX+1):
                for Lp in range(abs(L1-L2),L1+L2+1,2):
                    # Compute first 3j symbol
                    tj1 = acc.tj0(L1,L2,Lp)
                    if tj1==0: continue

                    for L3 in range(LMAX+1):
                        for L4 in range(LMAX+1):
                            if (-1)**(L1+L2+L3+L4)==-1: continue
                            if Lp<abs(L3-L4): continue
                            if Lp>L3+L4: continue

                            # Compute second 3j symbol
                            tj2 = acc.tj0(Lp,L3,L4)
                            if tj2==0: continue

                            # Iterate over odd-parity {l1,l2,l3}
                            l_index = 0
                            for l1 in range(LMAX_data+1):
                                for l2 in range(LMAX_data+1):
                                    for l3 in range(abs(l1-l2),min([l1+l2,LMAX_data])+1):
                                        if (-1.)**(l1+l2+l3)==1: continue

                                        # Define permutated ells
                                        ells = np.asarray([l1,l2,l3,0])
                                        ells_perm = ells[list(list(indices)[index_id])]
                                        lH1,lH2,lH3,lH4 = ells_perm

                                        # Determine intermediate ell
                                        if lH1==0: lH12=lH2
                                        elif lH2==0: lH12 = lH1
                                        elif lH3==0: lH12 = lH4
                                        elif lH4==0: lH12 = lH3
                                        else: raise Exception("wrong ells!")

                                        # Determine permutation factor
                                        cnt = 0
                                        ells_perm = np.delete(ells_perm,np.where(ells_perm==0)[0][0])
                                        for i in range(3):
                                            for j in range(i+1,3):
                                                if (ells_perm[i]>ells_perm[j]):
                                                    cnt+=1
                                        if cnt%2==0: phiH = 1
                                        else: phiH = -1

                                        # Load coupling
                                        ta = time.time()
                                        coupling_kernel = coupling_matrix(lH1,lH2,lH12,lH3,lH4,L1,L2,L3,L4,Lp,acc=acc)

                                        t_coupling += time.time()-ta
                                        if coupling_kernel==0:
                                            l_index += 1
                                            continue

                                        # Load integrals    
                                        tb = time.time()
                                        bin_index = 0
                                        for b1 in range(n_r):
                                            for b2 in range(b1+1,n_r):
                                                for b3 in range(b2+1,n_r):
                                                    # Define bin quadruplet
                                                    bH1, bH2, bH3, bH4 = np.asarray([b1,b2,b3,-1])[list(list(indices)[index_id])]

                                                    # compute integrals
                                                    integ = load_integrals(lH1,lH2,lH3,lH4,bH1,bH2,bH3,bH4,L1,L2,L3,L4,Lp,phiH,tj1*tj2,acc=acc)
                                                    if integ!=0: 
                                                        sum_output[l_index,bin_index] += integ*coupling_kernel

                                                    bin_index += 1
                                        t_matrix += time.time()-tb
                                        l_index += 1
        if index_id==0:
            print("Coupling time: %.2f s"%t_coupling)
            print("Matrix time: %.2f s"%t_matrix)
            print("3j calls: %d, 3j computations: %d"%(acc.calls3,acc.computes3))
            print("6j calls: %d, 6j computations: %d"%(acc.calls6,acc.computes6))
            print("9j calls: %d, 9j computations: %d"%(acc.calls9,acc.computes9))

        return sum_output

    ########################### COMPUTE ALL PERMUTATIONS ###########################

    all_output = np.zeros((n_l,int(n_r*(n_r-1)*(n_r-2)/6)))
    t_perms = time.time()
    p = mp.Pool(cores)
    out = list(tqdm.tqdm(p.imap_unordered(load_perm,np.arange(len(indices))),total=len(indices)))
    p.close()
    p.join()

    for i in range(len(indices)):
        all_output += out[i]
    t_perms = time.time()-t_perms

    print("Computed %d permutations in %.2f s on %d cores"%(len(indices),t_perms,cores))

    outfile = 'collider_4pcf_L%d_nu%.2f_cs%.2f.txt'%(LMAX,nu,cs)
    np.savetxt(outfile,all_output)

    print("Saved output to %s after %.1f s; exiting."%(outfile,time.time()-init_time))
