#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs, M_PI, sqrt
from cython.parallel import prange

cdef extern from "<complex.h>" namespace "std" nogil:
    double complex pow(double complex, double complex)
    double real(double complex)

cdef extern from "wignerSymbols-cpp.h" namespace "WignerSymbols" nogil:
    double wigner3j(double, double, double, double, double, double)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class tauNL_utils:

    # Define local memviews
    cdef int [:] ls, ms, l_arr, m_arr, Ls, Ms, L_indices, lmin_indices
    cdef int base_lmax, lmin, lmax, nl, nthreads, Lmin, Lmax, nLmin, nLmax, nmax_F, nr

    def __init__(self, int nthreads, int nr, np.ndarray[np.int32_t,ndim=1] l_arr, np.ndarray[np.int32_t,ndim=1] m_arr,
                 int Lmin, int Lmax, np.ndarray[np.int32_t,ndim=1] ls, np.ndarray[np.int32_t,ndim=1] ms, 
                 np.ndarray[np.int32_t,ndim=1] Ls, np.ndarray[np.int32_t,ndim=1] Ms, int nmax_F):
        """Initialize the class with various l and L quantities."""

        self.l_arr = l_arr
        self.m_arr = m_arr
        self.ls = ls
        self.ms = ms
        self.Ls = Ls
        self.Ms = Ms
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.lmin = min(ls)
        self.lmax = max(ls)
        self.nmax_F = nmax_F
        self.nLmin = min(Ls)
        self.nLmax = max(Ls)
        self.nl = len(ls)
        self.nr = nr
        self.base_lmax = max(l_arr)
        self.nthreads = nthreads

        # Define indices for L array
        cdef int i, ip, jp
        ip = -1
        jp = -1
        self.L_indices = np.ones(len(self.Ls),dtype=np.int32)*-1
        for i in xrange(len(self.l_arr)):
            if self.l_arr[i]>self.nLmax: continue
            jp += 1
            if self.l_arr[i]<self.nLmin: continue
            ip += 1
            if self.Ls[ip]<self.Lmin: continue
            if self.Ls[ip]>self.Lmax: continue
            self.L_indices[ip] = jp

        cdef int ct=0
        for i in xrange(len(self.l_arr)):
            if self.l_arr[i]<=self.lmax: ct += 1
        self.lmin_indices = np.ones(ct,dtype=np.int32)*-1
        ip = -1
        jp = -1
        for i in xrange(len(self.l_arr)):
            if self.l_arr[i]>self.lmax: continue
            ip += 1
            if self.l_arr[i]<self.lmin: continue
            jp += 1
            self.lmin_indices[ip] = jp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=2] apply_fl_weights(self, double[:,:,::1] flX, complex[:,::1] h_lm, double weight):
        """Compute w*f_l^X(r)*h^X_lm, summing over polarizations X (assuming only T and/or E contribute)."""

        cdef int nlm = h_lm.shape[1], npol = flX.shape[1], nr = flX.shape[2]
        cdef int ir, ilm, l
        cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((nr,nlm),dtype=np.complex128)

        # Iterate over l, r and sum over polarizations
        if npol==1:
            for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
                for ilm in xrange(nlm):
                    l = self.ls[ilm]
                    out[ir, ilm] = flX[l,0,ir]*h_lm[0,ilm]*weight
        else:
            for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
                for ilm in xrange(nlm):
                    l = self.ls[ilm]
                    out[ir, ilm] = (flX[l,0,ir]*h_lm[0,ilm]+flX[l,1,ir]*h_lm[1,ilm])*weight  
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=2] multiply(self, double[:,::1] map1, double[:,::1] map2, double[:,::1] map3):
        """Multiply three maps together in parallel"""
        cdef int n1 = map1.shape[0], n2 = map1.shape[1], i1, i2
        cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = map1[i1,i2]*map2[i1,i2]*map3[i1,i2]
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=2] multiply_asym(self, double[:,::1] map1A, double[:,::1] map1B, double[:,::1] map3A, double[:,::1] map3B):
        """Multiply three maps together in parallel"""
        cdef int n1 = map1A.shape[0], n2 = map1A.shape[1], i1, i2
        cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = map1A[i1,i2]*(2*map1B[i1,i2]*map3A[i1,i2]+map1A[i1,i2]*map3B[i1,i2])
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=2] multiplyC(self, double[:,::1] map1, double[:,::1] map2, complex[:,::1] map3c):
        """Multiply three maps together in parallel. The third is complex."""
        cdef int n1 = map1.shape[0], n2 = map1.shape[1], i1, i2
        cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1,n2),dtype=np.complex128)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = map1[i1,i2]*map2[i1,i2]*map3c[i1,i2]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=2] multiplyC_asym(self, double[:,::1] map1A, double[:,::1] map1B, complex[:,::1] map3cA, complex[:,::1] map3cB):
        """Multiply three maps together in parallel. The third is complex."""
        cdef int n1 = map1A.shape[0], n2 = map1A.shape[1], i1, i2
        cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1,n2),dtype=np.complex128)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = map1A[i1,i2]*(2*map1B[i1,i2]*map3cA[i1,i2]+map1A[i1,i2]*map3cB[i1,i2])
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=2] multiplyCC(self, double[:,::1] map1, double[:,::1] map2, complex[:,::1] map3c):
        """Multiply three maps together in parallel. The third is complex, and we compute A(B^2+|C|^2)."""
        cdef int n1 = map1.shape[0], n2 = map1.shape[1], i1, i2
        cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = map1[i1,i2]*(map2[i1,i2]**2+real(map3c[i1,i2]*map3c[i1,i2].conjugate()))
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=2] multiplyCC_asym(self, double[:,::1] map1A, double[:,::1] map1B, double[:,::1] map2A, double[:,::1] map2B, complex[:,::1] map3cA, complex[:,::1] map3cB):
        """Multiply three maps together in parallel. The third is complex, and we compute A(B^2+|C|^2)."""
        cdef int n1 = map1A.shape[0], n2 = map1A.shape[1], i1, i2
        cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = 2*map1A[i1,i2]*(map2A[i1,i2]*map2B[i1,i2]+real(map3cA[i1,i2].conjugate()*map3cB[i1,i2])) + map1B[i1,i2]*(map2A[i1,i2]*map2A[i1,i2]+real(map3cA[i1,i2]*map3cA[i1,i2].conjugate()))
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=2] multiply2(self, double[:,::1] B1, double[:,::1] B2, complex[:,::1] C1, complex[:,::1] C2):
        """Multiply two maps together to compute B1*B2 + |C1*C2|^2 in parallel. """
        cdef int n1 = B1.shape[0], n2 = B1.shape[1], i1, i2
        cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = B1[i1,i2]*B2[i1,i2]+real(C1[i1,i2]*C2[i1,i2].conjugate())
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=2] multiply2R(self, double[:,::1] A, double[:,::1] B):
        """Multiply two maps together to compute A*B in parallel. """
        cdef int n1 = A.shape[0], n2 = A.shape[1], i1, i2
        cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = A[i1,i2]*B[i1,i2]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=2] multiply2C(self, complex[:,::1] Ac, double[:,::1] B):
        """Multiply two maps together to compute A*B in parallel, where A is complex. """
        cdef int n1 = Ac.shape[0], n2 = Ac.shape[1], i1, i2
        cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1,n2),dtype=np.complex128)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = Ac[i1,i2]*B[i1,i2]
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=2] radial_sum(self, complex[:,::1] lm_map, double[:] r_weights, double[:,:,::1] flXs):
        """Compute [Sum_r weight(r) f^X_l(r) A^X_lm(r)], where A is complex. """
        cdef int nlm = lm_map.shape[1], nr = lm_map.shape[0], npol = flXs.shape[1]
        cdef int ilm, ir, ipol, l
        cdef complex tmp_out
        cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((npol,nlm),dtype=np.complex128)

        with nogil:
            for ipol in xrange(npol):
                for ilm in prange(nlm, schedule='static', num_threads=self.nthreads):
                    l = self.ls[ilm]
                    tmp_out = 0.
                    for ir in xrange(nr):
                        tmp_out = tmp_out + lm_map[ir,ilm]*flXs[l,ipol,ir]*r_weights[ir]
                    out[ipol,ilm] = tmp_out
        return out    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=2] radial_sum_spin1(self, complex[:,:,::1] lm_map, double[:] r_weights, double[:,:,::1] flXs):
        """Compute [Sum_r weight(r) f^X_l(r) (A^X_lm(r)-B^X_lm(r)], where A has two complex components. """
        cdef int nlm = lm_map.shape[2], nr = lm_map.shape[1], npol = flXs.shape[1]
        cdef int ilm, ir, ipol, l
        cdef complex tmp_out
        cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((npol,nlm),dtype=np.complex128)

        with nogil:
            for ipol in xrange(npol):
                for ilm in prange(nlm, schedule='static', num_threads=self.nthreads):
                    l = self.ls[ilm]
                    tmp_out = 0.
                    for ir in xrange(nr):
                        tmp_out = tmp_out + (lm_map[0,ir,ilm]-lm_map[1,ir,ilm])*flXs[l,ipol,ir]*r_weights[ir]
                    out[ipol,ilm] = 0.5*tmp_out
        return out    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_loc_sum(self, double[:] r_weights, double[:,::1] P_maps, double[:,::1] Q_maps):
        """Compute Sum_i w_i P_i(r)^3Q_i(r) for maps P, Q."""
        cdef int ir, ipix, nr = P_maps.shape[0], npix = P_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + P_maps[ir,ipix]**3*Q_maps[ir,ipix]
            out += r_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_loc_disc_sum(self, double[:] r_weights, double[:,::1] P_maps1, double[:,::1] P_maps2, double[:,::1] Q_maps1, double[:,::1] Q_maps2):
        """Compute Sum_i w_i P1_i(r)^2 P2_i(r)Q2_i(r) + [1 <-> 2] for maps P, Q."""
        cdef int ir, ipix, nr = P_maps1.shape[0], npix = P_maps1.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + P_maps1[ir,ipix]**2*P_maps2[ir,ipix]*Q_maps2[ir,ipix] + P_maps2[ir,ipix]**2*P_maps1[ir,ipix]*Q_maps1[ir,ipix]
            out += r_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_con_sum(self, double[:] r_weights, double[:,::1] R_maps):
        """Compute Sum_i w_i R_i(r)^4 for maps R."""
        cdef int ir, ipix, nr = R_maps.shape[0], npix = R_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + R_maps[ir,ipix]**4
            out += r_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_con_disc_sum(self, double[:] r_weights, double[:,::1] R_maps1, double[:,::1] R_maps2):
        """Compute Sum_i w_i R1_i(r)^2 R2_i(r)^2 for maps R."""
        cdef int ir, ipix, nr = R_maps1.shape[0], npix = R_maps1.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + R_maps1[ir,ipix]**2*R_maps2[ir,ipix]**2
            out += r_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_dotdot_sum(self, double[:] rtau_weights, double[:] tau_arr, double[:,::1] A_maps):
        """Compute Sum_i w_i tau_i^4 A_i(r)^4 for maps A."""
        cdef int ir, ipix, nr = A_maps.shape[0], npix = A_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + A_maps[ir,ipix]**4
            out += tau_arr[ir]**4*rtau_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_dotdot_disc_sum(self, double[:] rtau_weights, double[:] tau_arr, double[:,::1] A_maps1, double[:,::1] A_maps2):
        """Compute Sum_i w_i tau_i^4 A1_i(r)^2 A2_i(r)^2 for maps A."""
        cdef int ir, ipix, nr = A_maps1.shape[0], npix = A_maps1.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + A_maps1[ir,ipix]**2*A_maps2[ir,ipix]**2
            out += tau_arr[ir]**4*rtau_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_dotdel_sum(self, double[:] rtau_weights, double[:] tau_arr, double[:,::1] A_maps, double[:,::1] B_maps, complex[:,::1] C_maps):
        """Compute Sum_i w_i tau_i^2 A_i(r)^2[B_i(r)^2 + |C_i(r)|^2] for maps A,B,C."""
        cdef int ir, ipix, nr = A_maps.shape[0], npix = A_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + A_maps[ir,ipix]**2*(B_maps[ir,ipix]**2+real(C_maps[ir,ipix]*C_maps[ir,ipix].conjugate()))
            out += tau_arr[ir]**2*rtau_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_dotdel_disc_sum(self, double[:] rtau_weights, double[:] tau_arr, double[:,::1] A_maps1, double[:,::1] A_maps2, double[:,::1] B_maps1, double[:,::1] B_maps2, complex[:,::1] C_maps1, complex[:,::1] C_maps2):
        """Compute Sum_i w_i tau_i^2 A1_i(r)A2_i(r)Re[B1_i(r)B2_i(r) + C1_i(r)C2_i*(r)] for maps A,B,C."""
        cdef int ir, ipix, nr = A_maps1.shape[0], npix = A_maps2.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + A_maps1[ir,ipix]*A_maps2[ir,ipix]*(B_maps1[ir,ipix]*B_maps2[ir,ipix]+real(C_maps1[ir,ipix]*C_maps2[ir,ipix].conjugate()))
            out += tau_arr[ir]**2*rtau_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_deldel_sum(self, double[:] rtau_weights, double[:,::1] B_maps, complex[:,::1] C_maps):
        """Compute Sum_i w_i [B_i(r)^2 + |C_i(r)|^2]^2 for maps B,C."""
        cdef int ir, ipix, nr = B_maps.shape[0], npix = B_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + (B_maps[ir,ipix]**2+real(C_maps[ir,ipix]*C_maps[ir,ipix].conjugate()))**2
            out += rtau_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double gnl_deldel_disc_sum(self, double[:] rtau_weights, double[:,::1] B1_maps, double[:,::1] B2_maps, double[:,::1] B3_maps, double[:,::1] B4_maps, complex[:,::1] C1_maps, complex[:,::1] C2_maps, complex[:,::1] C3_maps, complex[:,::1] C4_maps):
        """Compute Sum_i w_i Re[B1_i(r)B2_i(r) + |C1_i(r)C2_i*(r)|]x[(1,2) <-> (3,4)] for maps B,C."""
        cdef int ir, ipix, nr = B1_maps.shape[0], npix = B1_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + (B1_maps[ir,ipix]*B2_maps[ir,ipix]+real(C1_maps[ir,ipix]*C2_maps[ir,ipix].conjugate()))*(B3_maps[ir,ipix]*B4_maps[ir,ipix]+real(C3_maps[ir,ipix]*C4_maps[ir,ipix].conjugate()))
            out += rtau_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void shift_and_weight_map_all(self, double[:,:,:,::1] flX, complex[:,::1] h_lm, int n, int mu, complex weight, 
                                        complex[:,::1] Larr_real, complex[:,::1] Larr_imag, int nmax):
                        
        """
        Weight a map by transfer functions, then shift by a_lm -> a_{l+Delta,mu-m} and a_{l+Delta, mu+m} coefficients and multiply by Gaunt factors.
        We fill up a map of (L, M>=0) and (L, M<0) element by element (skipping m=0 in the second sum).
        These are shifted to real and imaginary part maps for SHTs. We sum over all Delta.
        """
        # Define variables and outputs
        cdef int nlm = h_lm.shape[1], npol = flX.shape[2], n_r = flX.shape[3], nl = len(self.ls)
        cdef int i, j, l, L, M, LM_ind, ir, ilm, Delta
        cdef complex tmp, pref, pref2
        cdef complex[:,::1] input_lm = np.zeros((n_r,nlm),dtype=np.complex128)
        cdef complex[:] gaunt1 = np.zeros(nl,dtype=np.complex128)
        cdef complex[:] gaunt2 = np.zeros(nl,dtype=np.complex128)
        
        # Iterate over all Delta
        for Delta in xrange(-n,n+1):
            if (Delta+n)%2==1: continue

            pref = weight*pow(1.0j,Delta+0.j)*pow(-1.,mu+0.j)
            
            # Iterate over l, r and sum over polarizations
            if npol==1:
                for ilm in prange(nlm, nogil=True, schedule='static', num_threads=self.nthreads):
                    l = self.ls[ilm]
                    for ir in xrange(n_r):
                        input_lm[ir, ilm] = flX[nmax+Delta,l,0,ir]*h_lm[0,ilm].conjugate()
            else:
                for ilm in prange(nlm, nogil=True, schedule='static', num_threads=self.nthreads):
                    l = self.ls[ilm]
                    for ir in xrange(n_r):
                        input_lm[ir, ilm] = (flX[nmax+Delta,l,0,ir]*h_lm[0,ilm].conjugate()+flX[nmax+Delta,l,1,ir]*h_lm[1,ilm].conjugate())
            
            # Compute all Gaunt symbols
            for i in prange(nl,nogil=True,num_threads=self.nthreads):
                L = self.ls[i]+Delta
                if L>self.lmax+n: continue 
                if L<0: continue
                
                # Compute prefactor for m >= 0
                M = mu-self.ms[i]
                if M==0:
                    pref2 = pref
                elif M>0:
                    pref2 = pref/2.
                else:
                    pref2 = pow(-1.,M+Delta)*pref/2.

                # Compute Gaunt symbol
                if abs(M)<=L:
                    gaunt1[i] = pref2*gaunt_symbol(self.ls[i],L,n,self.ms[i],M,-mu)
                
                if self.ms[i]==0: continue
                
                # Compute prefactor for m < 0
                if mu==0:
                    gaunt2[i] = pow(-1.,Delta)*gaunt1[i]
                else:
                    M = mu+self.ms[i]
                
                    if M==0:
                        pref2 = pref*pow(-1.,self.ms[i])
                    elif M>0:
                        pref2 = pref*pow(-1.,self.ms[i])/2.
                    else:
                        pref2 = pref*pow(-1.,mu+Delta)/2.
                    
                    if abs(M)<=L:
                        gaunt2[i] = pref2*gaunt_symbol(self.ls[i],L,n,-self.ms[i],M,-mu)
                
            # Compute shifted map
            for j in prange(n_r,nogil=True,schedule='static',num_threads=self.nthreads):
                for i in xrange(nl):
                    L = self.ls[i]+Delta

                    # m>=0
                    if gaunt1[i]!=0:
                        M = mu-self.ms[i]
                        LM_ind = abs(M)*(self.lmax+n+1)+L-abs(M)*(abs(M)+1)//2
                        
                        if M>=0:
                            tmp = input_lm[j,i]*gaunt1[i]
                            Larr_real[j,LM_ind] += tmp
                            Larr_imag[j,LM_ind] += tmp
                        else:
                            tmp = input_lm[j,i].conjugate()*gaunt1[i]
                            Larr_real[j,LM_ind] += tmp
                            Larr_imag[j,LM_ind] -= tmp

                    # m < 0
                    if gaunt2[i]!=0:
                        M = mu+self.ms[i]
                        LM_ind = abs(M)*(self.lmax+n+1)+L-abs(M)*(abs(M)+1)//2

                        if M>=0:
                            tmp = input_lm[j,i].conjugate()*gaunt2[i]
                            Larr_real[j,LM_ind] += tmp
                            Larr_imag[j,LM_ind] += tmp
                        else:
                            tmp = input_lm[j,i]*gaunt2[i]
                            Larr_real[j,LM_ind] += tmp
                            Larr_imag[j,LM_ind] -= tmp
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void shift_and_weight_map(self, double[:,:,::1] flX, complex[:,::1] h_lm, int Delta, int n, int mu, complex weight, 
                                    complex[:,::1] Larr_real, complex[:,::1] Larr_imag):
                        
        """
        Weight a map by transfer functions, then shift by a_lm -> a_{l+Delta,mu-m} and a_{l+Delta, mu+m} coefficients and multiply by Gaunt factors.
        We fill up a map of (L, M>=0) and (L, M<0) element by element (skipping m=0 in the second sum).
        These are shifted to real and imaginary part maps for SHTs.
        """
        # Define variables and outputs
        cdef int nlm = h_lm.shape[1], npol = flX.shape[1], n_r = flX.shape[2]
        cdef int i, j, l, L, M, LM_ind, ir, ilm
        cdef complex tmp
        cdef complex pref = weight*pow(1.0j,Delta+0.j)*pow(-1.,mu+0.j)
        cdef complex conj_pref = weight*pow(-1.0j,Delta+0.j)*pow(-1.,mu+0.j)
        cdef complex[:,::1] input_lm = np.zeros((n_r,nlm),dtype=np.complex128)
        cdef double[:] gaunt1 = np.zeros(len(self.ls),np.float64)
        cdef double[:] gaunt2 = np.zeros(len(self.ls),np.float64)
        
        # Iterate over l, r and sum over polarizations
        if npol==1:
            for ilm in prange(nlm, nogil=True, schedule='static', num_threads=self.nthreads):
                l = self.ls[ilm]
                for ir in xrange(n_r):
                    input_lm[ir, ilm] = flX[l,0,ir]*h_lm[0,ilm]
        else:
            for ilm in prange(nlm, nogil=True, schedule='static', num_threads=self.nthreads):
                l = self.ls[ilm]
                for ir in xrange(n_r):
                    input_lm[ir, ilm] = (flX[l,0,ir]*h_lm[0,ilm]+flX[l,1,ir]*h_lm[1,ilm])
        
        # Compute all Gaunt symbols
        for i in xrange(len(self.ls)):
            gaunt1[i] = gaunt_symbol(self.ls[i],self.ls[i]+Delta,n,self.ms[i],mu-self.ms[i],-mu)
            gaunt2[i] = gaunt_symbol(self.ls[i],self.ls[i]+Delta,n,-self.ms[i],mu+self.ms[i],-mu)
        
        for j in prange(n_r,schedule='static',nogil=True,num_threads=self.nthreads):
            
            # m>=0
            for i in xrange(self.nl):
                if gaunt1[i]==0: continue
                L = self.ls[i]+Delta
                M = mu-self.ms[i]
                if abs(M)>L: continue
                if L>self.lmax+n: continue 
                if L<0: continue
                #LM_ind = abs(M)*(self.base_lmax+1)+L-abs(M)*(abs(M)+1)//2
                LM_ind = abs(M)*(self.lmax+n+1)+L-abs(M)*(abs(M)+1)//2
                if M==0:
                    tmp = input_lm[j,i]*pref*gaunt1[i]
                    Larr_real[j,LM_ind] += tmp
                    Larr_imag[j,LM_ind] += tmp
                elif M>0:
                    tmp = input_lm[j,i]*pref*gaunt1[i]/2.
                    Larr_real[j,LM_ind] += tmp
                    Larr_imag[j,LM_ind] += tmp
                else:
                    tmp = pow(-1.,M)*input_lm[j,i].conjugate()*conj_pref*gaunt1[i]/2.
                    Larr_real[j,LM_ind] += tmp
                    Larr_imag[j,LM_ind] -= tmp
            
            # m < 0
            for i in xrange(self.nl):
                if self.ms[i]==0: continue
                if gaunt2[i]==0: continue
                L = self.ls[i]+Delta
                M = mu+self.ms[i]
                if abs(M)>L: continue
                if L>self.lmax+n: continue 
                if L<0: continue
                #LM_ind = abs(M)*(self.base_lmax+1)+L-abs(M)*(abs(M)+1)//2
                LM_ind = abs(M)*(self.lmax+n+1)+L-abs(M)*(abs(M)+1)//2
                if M==0:
                    tmp = input_lm[j,i].conjugate()*pref*pow(-1.,self.ms[i])*gaunt2[i]
                    Larr_real[j,LM_ind] += tmp
                    Larr_imag[j,LM_ind] += tmp
                elif M>0:
                    tmp = input_lm[j,i].conjugate()*pref*pow(-1.,self.ms[i])*gaunt2[i]/2.
                    Larr_real[j,LM_ind] += tmp
                    Larr_imag[j,LM_ind] += tmp
                else:
                    tmp = input_lm[j,i]*conj_pref*pow(-1.,self.ms[i]+M)*gaunt2[i]/2.
                    Larr_real[j,LM_ind] += tmp
                    Larr_imag[j,LM_ind] -= tmp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double tau_sum_n0(self, np.ndarray[np.complex128_t,ndim=3] _arr1,np.ndarray[np.complex128_t,ndim=3]  _arr2,
                        np.ndarray[np.float64_t,ndim=3] _FLLs, np.ndarray[np.float64_t,ndim=1] _r_weights, int n1):
        """
        Compute sum over mu1,mu3,mu,M',M,L',L,r,r' for (direction-dependent / local) tauNL estimator.
        This is the simplified version for n = 0.
        """
        cdef int i,j,k,index,mu_index,L,M
        cdef int nmu = _arr1.shape[0]
        cdef complex m_weight
        cdef complex tmp
        cdef double out=0.
        
        # Memviews
        cdef complex[:,:,::1] arr1 = _arr1
        cdef complex[:,:,::1] arr2 = _arr2
        cdef double[:,:,::1] FLLs = _FLLs
        cdef double[:] r_weights = _r_weights
        
        # Sum over radial components (collapsed)
        for index in prange(self.nr*self.nr,schedule='static',nogil=True,num_threads=self.nthreads):
            j = index//self.nr
            k = index%self.nr

            # Skip if trivial!
            if r_weights[j]==0: continue
            if r_weights[k]==0: continue

            # Sum over L,M
            for i in xrange(len(self.Ls)):
                L = self.Ls[i]
                M = self.Ms[i]
                if L<self.Lmin: continue
                if L>self.Lmax: continue
                if M==0:
                    m_weight = 1. 
                else:
                    m_weight = 2.
                
                # Add to output array, summing over mu axis
                tmp = m_weight*FLLs[L,j,k]/sqrt(4*M_PI)*r_weights[j]*r_weights[k]/sqrt(2*n1+1.)
                for mu_index in xrange(nmu):
                    out += real(tmp*arr1[mu_index,i,j]*arr2[mu_index,i,k].conjugate())
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double tau_sum_general(self, np.ndarray[np.complex128_t,ndim=3] _arr1,np.ndarray[np.complex128_t,ndim=3]  _arr2,
                        np.ndarray[np.float64_t,ndim=4] _FLLs,
                        np.ndarray[np.float64_t,ndim=1] _r_weights,
                        int n1, int n3, int n):
        """
        Compute sum over mu1,mu3,mu,M',M,L',L,r,r' for (direction-dependent / local) tauNL estimator. 
        This is the version for general (non-zero) n.
        """
        cdef int nmu1 = _arr1.shape[0], nmu2 = _arr2.shape[0]
        cdef int L, M, Lp, Mp, LMp_ind=0, mu, mu1, mu3, i, j, k, Delta, Deltamu
        cdef complex pref1, pref, ksum
        cdef double out=0.

        # Memviews
        cdef complex[:,:,::1] arr1 = _arr1
        cdef complex[:,:,::1] arr2 = _arr2
        cdef double[:,:,:,::1] FLLs = _FLLs
        cdef double[:] r_weights = _r_weights
        
        for Deltamu in prange((2*n+1)*(2*n+1),nogil=True,schedule='dynamic',num_threads=self.nthreads):
            Delta = Deltamu//(2*n+1)-n
            mu = Deltamu%(2*n+1)-n
            # Iterate over L, M
            for i in xrange(len(self.Ls)):
                L = self.Ls[i]
                M = self.Ms[i]
                # Restrict to valid L
                if L<self.Lmin: continue
                if L>self.Lmax: continue
                # Restrict to valid L',M'
                Lp = L+Delta
                if Lp<self.Lmin: continue
                if Lp>self.Lmax: continue
                Mp = -self.Ms[i]-mu
                if abs(Mp)>Lp: continue

                if self.Ms[i]!=0:
                    pref1 = 2*pow(1.0j,Delta)*gaunt_symbol(L,L+Delta,n,self.Ms[i],-self.Ms[i]-mu,mu) # for M -> -M pieces
                else:
                    pref1 = pow(1.0j,Delta)*gaunt_symbol(L,L+Delta,n,self.Ms[i],-self.Ms[i]-mu,mu)
                
                # Compute (L+Delta, -M-mu) index
                LMp_ind = self.LM_index(Lp,Mp)

                # Sum over mu1 array
                for mu1 in xrange(-n1,n1+1):
                    mu3 = -mu1-mu
                    if abs(mu3)>n3: continue

                    # Compute 3j symbol
                    if Mp<0:
                        pref = (-1.)**(Mp+n3+mu3)*pref1*threej(n1,n3,n,mu1,mu3,mu)
                    else:
                        pref = pref1*threej(n1,n3,n,mu1,mu3,mu)
                    if pref==0: continue
                    
                    # Sum over r, r'
                    for k in xrange(self.nr):
                        if r_weights[k]==0: continue
                        if Mp>=0: 
                            ksum = pref*arr2[n3+mu3][LMp_ind,k]*r_weights[k]
                        else:
                            ksum = pref*arr2[n3-mu3][LMp_ind,k].conjugate()*r_weights[k]
                        for j in xrange(self.nr):
                            if r_weights[j]==0: continue
                            out += real(ksum*arr1[n1+mu1,i,j])*FLLs[self.nmax_F+Delta,L,j,k]*r_weights[j]

        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double tau_sum_general_collider(self, 
                                        np.ndarray[np.complex128_t,ndim=3] _arr1, 
                                        np.ndarray[np.complex128_t,ndim=3] _arr1c, 
                                        np.ndarray[np.complex128_t,ndim=3] _arr2, 
                                        np.ndarray[np.complex128_t,ndim=3] _arr2c, 
                                        np.ndarray[np.complex128_t,ndim=4] _FLLs,
                                        np.ndarray[np.float64_t,ndim=1] _r_weights,
                                        int nmax_F, int s, complex beta,
                                        np.ndarray[np.complex128_t,ndim=1] _coeffs
                        ):
        """
        Compute sum over S,lam1,lam3,lam,M',M,L',L,r,r' for collider tauNL estimator.
        """
        cdef int i, j, k, Mp, L, M, Lp, LMp_ind, lam1, lam3, Lam, Delta, S
        cdef complex ksum, coeff, pref, pref1
        cdef double tj, out=0.

        # Memviews
        cdef complex[:,:,::1] arr1 = _arr1
        cdef complex[:,:,::1] arr1c = _arr1c
        cdef complex[:,:,::1] arr2 = _arr2
        cdef complex[:,:,::1] arr2c = _arr2c
        cdef complex[:,:,:,::1] FLLs = _FLLs
        cdef complex[:] coeffs = _coeffs
        cdef double[:] r_weights = _r_weights
        
        # Create a lookup table to facilitate multiprocessing operations
        cdef int index=0, size = 0
        cdef int[:,:] indices
        size = 0
        for S in xrange(0,2*s+1,2):
            for Delta in xrange(-S,S+1):
                for Lam in range(-S,S+1):
                    size += 1
        indices = np.zeros((size,3),dtype=np.int32)
        for S in xrange(0,2*s+1,2):
            for Delta in xrange(-S,S+1):
                for Lam in range(-S,S+1):
                    indices[index,0] = S
                    indices[index,1] = Delta
                    indices[index,2] = Lam
                    index += 1
        
        # Iterate over S, Delta, Lam variables
        for index in prange(size,schedule='dynamic',nogil=True,num_threads=self.nthreads):
            S = indices[index,0]
            Delta = indices[index,1]
            Lam = indices[index,2]
            coeff = coeffs[S//2]
            if coeff==0: continue

            # Iterate over L, M
            for i in xrange(len(self.Ls)):
                L = self.Ls[i]
                M = self.Ms[i]
                # Restrict to valid L
                if L<self.Lmin: continue
                if L>self.Lmax: continue
                Lp = L+Delta
                # Restrict to valid L',M'
                if Lp<self.Lmin: continue
                if Lp>self.Lmax: continue
                Mp = -self.Ms[i]-Lam
                if abs(Mp)>Lp: continue

                # Compute r,r' independent pieces (adding weight for M -> -M switch)
                if self.Ms[i]!=0 and beta==beta.conjugate():
                    pref1 = 2.*coeff*pow(1.0j,Delta)*gaunt_symbol(L,L+Delta,S,self.Ms[i],-self.Ms[i]-Lam,Lam)
                else:
                    pref1 = coeff*pow(1.0j,Delta)*gaunt_symbol(L,L+Delta,S,self.Ms[i],-self.Ms[i]-Lam,Lam)
                if pref1==0: continue

                # Compute (L+Delta, -M-mu) index
                LMp_ind = self.LM_index(Lp,Mp)
                
                # Sum over lam1 array
                for lam1 in xrange(-s,s+1):
                    lam3 = -lam1-Lam
                    if abs(lam3)>s: continue

                    # Compute 3j symbol
                    if Mp<0:
                        pref = pref1*threej(s,s,S,lam1,lam3,Lam)*pow(-1.,Mp+s+lam3)
                    else:
                        pref = pref1*threej(s,s,S,lam1,lam3,Lam)
                    if pref==0: continue

                    # Sum over r, r'
                    for k in xrange(self.nr):
                        if r_weights[k]==0: continue
                        if Mp>=0:
                            ksum = pref*arr2[s+lam3,LMp_ind,k]*r_weights[k]
                            for j in xrange(self.nr):
                                if r_weights[j]==0: continue
                                out += real(ksum*arr1[s+lam1,i,j]*FLLs[nmax_F+Delta,L,j,k])*r_weights[j]
                        else:
                            ksum = pref*arr2c[s-lam3,LMp_ind,k].conjugate()*r_weights[k]
                            for j in xrange(self.nr):
                                if r_weights[j]==0: continue
                                out += real(ksum*arr1[s+lam1,i,j]*FLLs[nmax_F+Delta,L,j,k])*r_weights[j]

            # Add m -> -m piece if needed
            if beta!=beta.conjugate():
                for i in xrange(len(self.Ls)):
                    L = self.Ls[i]
                    M = self.Ms[i]
                    # Restrict to valid L,M
                    if L<self.Lmin: continue
                    if L>self.Lmax: continue
                    if self.Ms[i]==0: continue
                    # Restrict to valid L',M'
                    Lp = L+Delta
                    if Lp<self.Lmin: continue
                    if Lp>self.Lmax: continue
                    Mp = self.Ms[i]-Lam
                    if abs(Mp)>Lp: continue

                    # Compute r,r' independent pieces
                    pref1 = coeff*pow(1.0j,Delta)*gaunt_symbol(L,L+Delta,S,-self.Ms[i],self.Ms[i]-Lam,Lam)
                    if pref1==0: continue
                    
                    # Compute (L+Delta, -M-mu) index
                    LMp_ind = self.LM_index(Lp,Mp)

                    # Sum over lam1 array
                    for lam1 in xrange(-s,s+1):
                        lam3 = -lam1-Lam
                        if abs(lam3)>s: continue

                        # Compute 3j symbol
                        if Mp<0:
                            pref = pow(-1.,self.Ms[i]+Mp+lam1+lam3)*pref1*threej(s,s,S,lam1,lam3,Lam)
                        else:
                            pref = pow(-1,s+lam1+self.Ms[i])*pref1*threej(s,s,S,lam1,lam3,Lam)
                        if pref==0: continue    

                        # Sum over r, r'
                        for k in xrange(self.nr):
                            if r_weights[k]==0: continue
                            if Mp>=0:
                                ksum = pref*arr2[s+lam3,LMp_ind,k]*r_weights[k]
                                for j in xrange(self.nr):
                                    if r_weights[j]==0: continue
                                    out += real(ksum*arr1c[s-lam1,i,j].conjugate()*FLLs[nmax_F+Delta,L,j,k])*r_weights[j]
                            else:
                                ksum = pref*arr2c[s-lam3,LMp_ind,k].conjugate()*r_weights[k]
                                for j in xrange(self.nr):
                                    if r_weights[j]==0: continue
                                    out += real(ksum*arr1c[s-lam1,i,j].conjugate()*FLLs[nmax_F+Delta,L,j,k])*r_weights[j]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double tau_sum_nu0_collider(self, np.ndarray[np.complex128_t,ndim=3] _arr1,np.ndarray[np.complex128_t,ndim=3]  _arr2,
                        np.ndarray[np.complex128_t,ndim=3] _FLLs, np.ndarray[np.float64_t,ndim=1] _r_weights, complex pref):
        """
        Compute sum over S,lam1,lam3,lam,M',M,L',L,r,r' for collider tauNL estimator.
        This is the simplified version for nu_s = 0.
        """
        cdef int i,j,k,index,lam_index, L, M
        cdef int nlam = _arr1.shape[0]
        cdef complex m_weight, tmp
        cdef double out = 0. 
        
        # Memviews
        cdef complex[:,:,::1] arr1 = _arr1
        cdef complex[:,:,::1] arr2 = _arr2
        cdef complex[:,:,::1] FLLs = _FLLs
        cdef double[:] r_weights = _r_weights
        
        # Sum over radial components (collapsed)
        for index in prange(self.nr*self.nr,schedule='static',nogil=True,num_threads=self.nthreads):
            j = index//self.nr
            k = index%self.nr
            
            # Check if trivial
            if r_weights[j]==0: continue
            if r_weights[k]==0: continue

            # Sum over L,M
            for i in xrange(len(self.Ls)):
                L = self.Ls[i]
                M = self.Ms[i]
                if L<self.Lmin: continue
                if L>self.Lmax: continue
                if self.Ms[i]==0:
                    m_weight = 1. 
                else:
                    m_weight = 2.
                
                # Add to output array, summing over mu axis
                tmp = m_weight*FLLs[L,j,k]*pref*r_weights[j]*r_weights[k]
                for lam_index in xrange(nlam):
                    out += real(tmp*arr1[lam_index,i,j]*arr2[lam_index,i,k].conjugate())
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double tau_sum_n0_collider(self, np.ndarray[np.complex128_t,ndim=3] _arr1,np.ndarray[np.complex128_t,ndim=3]  _arr2,
                        np.ndarray[np.complex128_t,ndim=3] _FLLs, np.ndarray[np.float64_t,ndim=1] _r_weights, complex pref):
        """
        Compute sum over S,lam1,lam3,lam,M',M,L',L,r,r' for collider tauNL estimator.
        This is the simplified version for s = 0.
        """
        cdef int i,j,k,index, L, M
        cdef complex m_weight, tmp, rr
        cdef double out = 0.
        
        # Memviews
        cdef complex[:,:,::1] arr1 = _arr1
        cdef complex[:,:,::1] arr2 = _arr2
        cdef complex[:,:,::1] FLLs = _FLLs
        cdef double[::1] r_weights = _r_weights
        
        # Sum over radial components (collapsed)
        for index in prange(self.nr*self.nr,schedule='static',nogil=True,num_threads=self.nthreads):
            j = index//self.nr
            k = index%self.nr
            rr = pref*r_weights[j]*r_weights[k]
            
            # Check if trivial
            if r_weights[j]==0: continue
            if r_weights[k]==0: continue

            # Sum over L,M
            for i in xrange(len(self.Ls)):
                L = self.Ls[i]
                M = self.Ms[i]
                if L<self.Lmin: continue
                if L>self.Lmax: continue

                # Add to output array
                if self.Ms[i]==0:
                    out += real(rr*FLLs[L,j,k]*arr1[0,i,j]*arr2[0,i,k].conjugate())
                else:
                    out += 2*real(rr*FLLs[L,j,k]*arr1[0,i,j]*arr2[0,i,k].conjugate())
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void convolve_F_n000(self, complex[:,::1] PQ, double[:,:,:,::1] FLLs, double[:] r_weights, 
                                     complex[:,::1] out, int nmax_F, int[:] inds):
        """Compute the convolution of a product map with F (optionally restricting to the "inds" indices). We output a map, filling only non-trivial elements.
        
        This is a simplification for n = n1 = 0 and real maps."""
        cdef int i, j, k, Lp, ip, nr = len(inds), indj
        cdef complex tmp_sum
        assert PQ.shape[1]==nr, 'Wrong PQ shape: %d,%d | %d'%(PQ.shape[0],PQ.shape[1],nr)
        assert out.shape[1]==nr, 'Wrong output shape'
        
        # Iterate over r
        for j in prange(nr,nogil=True,num_threads=self.nthreads):
            indj = inds[j]
        
            # Iterate over L
            for ip in xrange(len(self.Ls)):

                i = self.L_indices[ip]
                if i==-1: continue
                Lp = self.Ls[ip]
                
                # Sum over r'
                tmp_sum = 0.
                for k in xrange(nr):
                    if r_weights[k]==0: continue
                    tmp_sum = tmp_sum + FLLs[nmax_F,Lp,indj,inds[k]]*r_weights[k]*PQ[ip,k]/sqrt(4*M_PI)
                out[i,j] = tmp_sum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void convolve_F_n000_real(self, complex[:,::1] PQ, complex[:,:,:,::1] FLLs, double[:] r_weights, 
                                     complex[:,::1] out, int nmax_F, int[:] inds):
        """Compute the convolution of a product map with F (optionally restricting to the "inds" indices). We output a map, filling only non-trivial elements.
        
        This is a simplification for n = n1 = 0 and real maps."""
        cdef int i, j, k, Lp, ip, nr = len(inds), indj
        cdef complex tmp_sum
        assert PQ.shape[1]==nr, 'Wrong PQ shape: %d,%d | %d'%(PQ.shape[0],PQ.shape[1],nr)
        assert out.shape[1]==nr, 'Wrong output shape'
        
        # Iterate over r
        for j in prange(nr,nogil=True,num_threads=self.nthreads):
            indj = inds[j]
        
            # Iterate over L
            for ip in xrange(len(self.Ls)):

                i = self.L_indices[ip]
                if i==-1: continue
                Lp = self.Ls[ip]

                # Sum over r'
                tmp_sum = 0.
                for k in xrange(nr):
                    if r_weights[k]==0: continue
                    tmp_sum = tmp_sum + real(FLLs[nmax_F,Lp,indj,inds[k]])*r_weights[k]*PQ[ip,k]/sqrt(4*M_PI)
                out[i,j] = tmp_sum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void convolve_F_n0(self, np.ndarray[np.complex128_t,ndim=2] _PQ, np.ndarray[np.complex128_t,ndim=2] _cPQ, 
                                        np.ndarray[np.float64_t,ndim=4] _FLLs, 
                                        np.ndarray[np.float64_t,ndim=1] _r_weights, 
                                        np.ndarray[np.complex128_t,ndim=2] _out_real, np.ndarray[np.complex128_t,ndim=2] _out_imag, 
                                        int n1, int mu1, int nmax_F, int[:] inds):
        """Compute the convolution of a product map with F. We output a harmonic-space map, filling only non-trivial elements.
        
        This is a simplification for n = 0 and real FLLs."""
        cdef int ipj, i, j, k, Lp, ip, indj, nr = len(inds)
        cdef complex out_p, out_m, pref_p, pref_m
        assert _PQ.shape[1]==nr, 'Wrong PQ shape: %d,%d | %d'%(_PQ.shape[0],_PQ.shape[1],nr)
        assert _cPQ.shape[1]==nr, 'Wrong cPQ shape: %d,%d | %d'%(_cPQ.shape[0],_cPQ.shape[1],nr)
        assert _out_real.shape[1]==nr, 'Wrong output shape'
        assert _out_imag.shape[1]==nr, 'Wrong output shape'
        
        pref_p = pow(-1,n1-mu1)/np.sqrt(2.*n1+1.)/sqrt(4*M_PI)/2.
        pref_m = 1./np.sqrt(2.*n1+1.)/sqrt(4*M_PI)/2.

        # Define memviews
        cdef complex[:,::1] PQ = _PQ, cPQ = _cPQ
        cdef complex[:,::1] out_real = _out_real, out_imag = _out_imag
        cdef double[:,:,:,::1] FLLs = _FLLs
        cdef double[:] r_weights = _r_weights
        
        # Iterate over r
        for ipj in prange(nr*len(self.Ls),nogil=True,num_threads=self.nthreads,schedule='dynamic'):
            ip = ipj//nr
            j = ipj%nr
            indj = inds[j]

            i = self.L_indices[ip]
            if i==-1: continue
            Lp = self.Ls[ip]

            # Sum over r'
            out_p = 0.
            out_m = 0.
            for k in xrange(nr):
                if r_weights[k]==0: continue
                out_p = out_p+pref_p*FLLs[nmax_F,Lp,indj,inds[k]]*r_weights[k]*PQ[ip,k]
                out_m = out_m+pref_m*FLLs[nmax_F,Lp,indj,inds[k]]*r_weights[k]*cPQ[ip,k]
            out_real[i,j] = out_p+out_m
            out_imag[i,j] = out_p-out_m
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void convolve_F_general(self, np.ndarray[np.complex128_t,ndim=3] _PQ,
                                        np.ndarray[np.float64_t,ndim=4] _FLLs,
                                        np.ndarray[np.float64_t,ndim=1] _r_weights,
                                        np.ndarray[np.complex128_t,ndim=2] out_real, np.ndarray[np.complex128_t,ndim=2] out_imag,
                                        int n1, int n3, int n, int mu1, int[:] inds): 
        """Compute the convolution of a product map with F. We output a map, filling only non-trivial elements.
        
        This is a simplification for real FLLs, but arbitrary n."""
        
        cdef int i, j, k, Deltamu, Delta, ip, L, M, Lp, Mp, LMp_ind, mu, mu3, indj, nr = len(inds), nL = len(self.Ls)
        cdef complex[:,::1] out_plus = np.zeros((len(self.Ls),nr),dtype=np.complex128)
        cdef complex[:,::1] out_minus = np.zeros((len(self.Ls),nr),dtype=np.complex128)
        
        # Define memviews
        cdef complex[:,:,::1] PQ = _PQ
        cdef double[:,:,:,::1] FLLs = _FLLs
        cdef double[:] r_weights = _r_weights
        cdef complex gaunt, pref, ksum

        # Sum over Delta, mu
        for Deltamu in xrange((2*n+1)*(2*n+1)):
            Delta = Deltamu//(2*n+1)-n
            mu = Deltamu%(2*n+1)-n
            if (Delta+n)%2==1: continue # killed by Gaunt symbol
                    
            # Compute 3j weighting
            mu3 = -mu1-mu
            if abs(mu3)>n3: continue
            pref = pow(1.0j,Delta)*threej(n1,n3,n,mu1,mu3,mu)
            if pref==0: continue       
            
            # Compute (L+Delta, M-mu) map
            for i in prange(nL,nogil=True,schedule='dynamic',num_threads=self.nthreads):
                L = self.Ls[i]
                M = self.Ms[i]
                # Restrict to valid L
                if L<self.Lmin: continue
                if L>self.Lmax: continue
                Lp = L+Delta
                # Restrict to valid L',M'
                if Lp<self.Lmin: continue
                if Lp>self.Lmax: continue
                Mp = M-mu
                if abs(Mp)>Lp: continue
            
                # Define Gaunt factor
                if Mp>=0:
                    gaunt = pref*pow(-1.,M)*gaunt_symbol(L,L+Delta,n,-M,M-mu,mu)
                else:
                    gaunt = pref*pow(-1.,M+Mp+n3+mu3)*gaunt_symbol(L,L+Delta,n,-M,M-mu,mu)
                if gaunt==0: continue

                if M!=0:
                    gaunt = gaunt/2.
                
                # Compute (L+Delta, -M-mu) index
                LMp_ind = self.LM_index(Lp,Mp)

                # Iterate over r axis
                for j in xrange(nr):
                    indj = inds[j]
                
                    # Sum over r'
                    ksum = 0.
                    for k in xrange(nr):
                        if r_weights[k]==0: continue
                        if Mp>=0:
                            ksum = ksum+PQ[n3+mu3,LMp_ind,k]*FLLs[self.nmax_F+Delta,L,indj,inds[k]]*r_weights[k]
                        else:
                            ksum = ksum+PQ[n3-mu3,LMp_ind,k].conjugate()*FLLs[self.nmax_F+Delta,L,indj,inds[k]]*r_weights[k]
                    out_plus[i,j] += ksum*gaunt

            # Compute (L+Delta, -M-mu) map
            for i in prange(nL,nogil=True,schedule='dynamic',num_threads=self.nthreads):
                L = self.Ls[i]
                M = self.Ms[i]
                # Restrict to valid L
                if L<self.Lmin: continue
                if L>self.Lmax: continue
                if M==0: continue
                Lp = L+Delta
                # Restrict to valid L',M'
                if Lp<self.Lmin: continue
                if Lp>self.Lmax: continue
                Mp = -M-mu
                if abs(Mp)>Lp: continue

                # Define Gaunt factor
                if Mp>=0:
                    gaunt = pref.conjugate()*gaunt_symbol(L,L+Delta,n,M,-M-mu,mu)/2.
                else:
                    gaunt = pref.conjugate()*pow(-1.,Mp+n3+mu3)*gaunt_symbol(L,L+Delta,n,M,-M-mu,mu)/2.
                if gaunt==0: continue           
                
                # Compute (L+Delta, -M-mu) indexs
                LMp_ind = self.LM_index(Lp,Mp)

                # Iterate over r axis in parallel
                for j in xrange(nr):
                    indj = inds[j]
                
                    # Sum over r'
                    ksum = 0.
                    for k in xrange(nr):
                        if r_weights[k]==0: continue
                        if Mp>=0:
                            ksum = ksum+PQ[n3+mu3,LMp_ind,k].conjugate()*FLLs[self.nmax_F+Delta,L,indj,inds[k]]*r_weights[k]
                        else:
                            ksum = ksum+PQ[n3-mu3,LMp_ind,k]*FLLs[self.nmax_F+Delta,L,indj,inds[k]]*r_weights[k]
                    out_minus[i,j] += ksum*gaunt

        # Now cast to the full l,m-range
        for ip in xrange(len(self.Ls)):
            i = self.L_indices[ip]
            if i==-1: continue
            for j in xrange(nr):
                out_real[i,j] += out_plus[ip,j]+out_minus[ip,j]
                out_imag[i,j] += out_plus[ip,j]-out_minus[ip,j]
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void convolve_F_general_collider(self, np.ndarray[np.complex128_t,ndim=3] PQ, np.ndarray[np.complex128_t,ndim=3] cPQ,
                                        np.ndarray[np.complex128_t,ndim=4] FLLs,
                                        np.ndarray[np.float64_t,ndim=1] r_weights,
                                        np.ndarray[np.complex128_t,ndim=2] out_real, np.ndarray[np.complex128_t,ndim=2] out_imag,
                                        int s, int lam1, int nmax_F, np.ndarray[np.complex128_t,ndim=1] S_weights, int[:] inds): 
        """Compute the convolution of a product map with F. We output a map, filling only non-trivial elements.
        
        This is the full collider version for arbitrary n."""
        
        cdef int i, j, k, S, Deltalam, Delta, ip, L, M, Lp, Mp, LMp_ind, lam, lam3, indj, indk, nr = len(inds), nL=len(self.Ls)
        cdef complex[:,::1] out_plus = np.zeros((len(self.Ls),nr),dtype=np.complex128)
        cdef complex[:,::1] out_minus = np.zeros((len(self.Ls),nr),dtype=np.complex128)
        
        # Define memviews
        cdef complex gaunt, pref, ksum, pref_m, tmpP, tmpM
    
        with nogil:

            ## S = 0 contributions
            # Compute 3j weighting
            pref = S_weights[0]*pow(-1.,s-lam1)/sqrt(2.*s+1.)/sqrt(4*M_PI)/2.
            if pref!=0:       
                pref_m = S_weights[0].conjugate()/sqrt(2.*s+1.)/sqrt(4*M_PI)/2.
                
                # Sum over L,M
                for i in prange(nL,schedule='dynamic',num_threads=self.nthreads):
                    L = self.Ls[i]
                    M = self.Ms[i]
                    # Restrict to valid L
                    if L<self.Lmin: continue
                    if L>self.Lmax: continue

                    if M==0:
                        # Iterate over r,r' axes
                        for j in xrange(nr):
                            indj = inds[j]
                            tmpP = 0.
                            for k in xrange(nr):
                                indk = inds[k]
                                if r_weights[k]==0: continue
                                tmpP = tmpP + 2.*PQ[s-lam1,i,k]*FLLs[nmax_F,L,indj,indk]*r_weights[k]*pref
                            out_plus[i,j] = tmpP
                    else:
                        # Iterate over r,r' axes
                        for j in xrange(nr):
                            indj = inds[j]
                            tmpP, tmpM = 0., 0.
                            for k in xrange(nr):
                                indk = inds[k]
                                if r_weights[k]==0: continue
                                tmpP = tmpP + PQ[s-lam1,i,k]*FLLs[nmax_F,L,indj,indk]*r_weights[k]*pref
                                tmpM = tmpM + cPQ[s+lam1,i,k]*FLLs[nmax_F,L,indj,indk].conjugate()*r_weights[k]*pref_m
                            out_plus[i,j] = tmpP
                            out_minus[i,j] = tmpM

            ## S > 0 pieces
            for S in xrange(2,2*s+1,2):
                # Sum over Delta, lam
                for Deltalam in xrange((2*S+1)*(2*S+1)):
                    Delta = Deltalam//(2*S+1)-S
                    lam = Deltalam%(2*S+1)-S
                    if (Delta+S)%2==1: continue # killed by Gaunt symbol
                            
                    # Compute 3j weighting
                    lam3 = -lam1-lam
                    if abs(lam3)>s: continue
                    pref = S_weights[S//2]*pow(1.0j,Delta)*threej(s,s,S,lam1,lam3,lam)
                    if pref==0: continue       
                    
                    # Compute (L+Delta, M-lam) map
                    for i in prange(nL,schedule='dynamic',num_threads=self.nthreads):
                        L = self.Ls[i]
                        M = self.Ms[i]
                        # Restrict to valid L
                        if L<self.Lmin: continue
                        if L>self.Lmax: continue
                        Lp = L+Delta
                        # Restrict to valid L',M'
                        if Lp<self.Lmin: continue
                        if Lp>self.Lmax: continue
                        Mp = M-lam
                        if abs(Mp)>Lp: continue
                        
                        # Define Gaunt factor
                        if Mp>=0:
                            gaunt = pref*pow(-1.,M)*gaunt_symbol(L,L+Delta,S,-M,M-lam,lam)
                        else:
                            gaunt = pref*pow(-1.,M+Mp+s+lam3)*gaunt_symbol(L,L+Delta,S,-M,M-lam,lam)
                        if gaunt==0: continue

                        if M!=0:
                            gaunt = gaunt/2.
                        
                        # Compute (L+Delta, -M-lam) index
                        LMp_ind = self.LM_index(Lp,Mp)
                        
                        # Iterate over r axis in parallel
                        for j in xrange(nr):
                            indj = inds[j]

                            # Sum over r'
                            ksum = 0.
                            for k in xrange(nr):
                                if r_weights[k]==0: continue
                                if Mp>=0:
                                    ksum = ksum+PQ[s+lam3,LMp_ind,k]*FLLs[nmax_F+Delta,L,indj,inds[k]]*r_weights[k]
                                else:
                                    ksum = ksum+cPQ[s-lam3,LMp_ind,k].conjugate()*FLLs[nmax_F+Delta,L,indj,inds[k]]*r_weights[k]
                            out_plus[i,j] += ksum*gaunt

                    # Compute (L+Delta, -M-lam) map
                    for i in prange(nL,schedule='dynamic',num_threads=self.nthreads):
                        L = self.Ls[i]
                        M = self.Ms[i]
                        # Restrict to valid L
                        if L<self.Lmin: continue
                        if L>self.Lmax: continue
                        if M==0: continue
                        Lp = L+Delta
                        # Restrict to valid L',M'
                        if Lp<self.Lmin: continue
                        if Lp>self.Lmax: continue
                        Mp = -M-lam
                        if abs(Mp)>Lp: continue

                        # Define Gaunt factor
                        if Mp>=0:
                            gaunt = pref.conjugate()*gaunt_symbol(L,L+Delta,S,M,-M-lam,lam)/2.
                        else:
                            gaunt = pref.conjugate()*pow(-1.,Mp+s+lam3)*gaunt_symbol(L,L+Delta,S,M,-M-lam,lam)/2.
                        if gaunt==0: continue           
                        
                        # Compute (L+Delta, -M-lam) indexs
                        LMp_ind = self.LM_index(Lp,Mp)

                        # Iterate over r axis in parallel
                        for j in xrange(nr):
                            indj = inds[j]

                            # Sum over r'
                            ksum = 0.
                            for k in xrange(nr):
                                if r_weights[k]==0: continue
                                if Mp>=0:
                                    ksum = ksum+PQ[s+lam3,LMp_ind,k].conjugate()*FLLs[nmax_F+Delta,L,indj,inds[k]].conjugate()*r_weights[k]
                                else:
                                    ksum = ksum+cPQ[s-lam3,LMp_ind,k]*FLLs[nmax_F+Delta,L,indj,inds[k]].conjugate()*r_weights[k]
                            out_minus[i,j] += ksum*gaunt

        # Now cast to the full l,m-range
        for ip in xrange(nL):
            i = self.L_indices[ip]
            if i==-1: continue
            for j in xrange(nr):
                out_real[i,j] += out_plus[ip,j]+out_minus[ip,j]
                out_imag[i,j] += out_plus[ip,j]-out_minus[ip,j]
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void compute_pF_map_complex(self, np.ndarray[np.complex128_t,ndim=2] _input_plus, np.ndarray[np.complex128_t,ndim=2] _input_minus, 
                            np.ndarray[np.complex128_t,ndim=4] _plLXs,
                            np.ndarray[np.float64_t,ndim=1] _r_weights,
                            int n1, int mu1, int nmax, int lmax, 
                            np.ndarray[np.complex128_t,ndim=2] _output, int[:] inds):
        """Shift an input a_lm map by Delta,-mu, multiply it by p_lL^X(r), and sum over r."""

        cdef int i, j, u, l, m, lp, mp, lmp_ind, Delta, npol=len(_plLXs[0][0]), nr = len(inds)
        cdef complex gaunt_p, gaunt_m, tmp_sum

        # Define memviews
        cdef complex[:,::1] input_plus = _input_plus, input_minus = _input_minus
        cdef complex[:,:,:,::1] plLXs = _plLXs
        cdef double[:] r_weights = _r_weights
        cdef complex[:,::1] output = _output

        # Sum over Delta
        for Delta in xrange(-n1,n1+1):
            if (Delta+n1)%2==1: continue # must be even

            # Compute shifted map for m >= 0
            for i in prange(self.nl, nogil=True, schedule='dynamic', num_threads=self.nthreads):
                l = self.ls[i]
                lp = l+Delta
                if lp>lmax: continue
                if lp<0: continue
                m = self.ms[i]
                mp = m-mu1
                if abs(mp)>lp: continue

                # Define l+Delta,m-mu1 index
                lmp_ind = abs(mp)*(lmax+1)+lp-abs(mp)*(abs(mp)+1)//2
                
                # Gaunt factor
                gaunt_p = 0.25*pow(-1.,m)*pow(1.0j,Delta)*gaunt_symbol(l,lp,n1,m,-mp,-mu1)
                if gaunt_p==0: continue
                gaunt_m = gaunt_p*pow(-1.,mp)
                
                # Sum over r and u
                if mp>=0:
                    for u in xrange(npol):
                        tmp_sum = 0.
                        for j in xrange(nr):   
                            if r_weights[j]==0: continue 
                            tmp_sum = tmp_sum + gaunt_p*input_plus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                        output[u,i] += tmp_sum
                else:
                    for u in xrange(npol):
                        tmp_sum = 0.
                        for j in xrange(nr):    
                            if r_weights[j]==0: continue 
                            tmp_sum = tmp_sum + gaunt_m*input_minus[lmp_ind,j].conjugate()*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                        output[u,i] += tmp_sum
            
            # Compute shifted map for m <= 0
            for i in prange(self.nl, nogil=True, schedule='dynamic', num_threads=self.nthreads):
                l = self.ls[i]
                lp = l+Delta
                if lp>lmax: continue
                if lp<0: continue
                m = self.ms[i]
                mp = -m-mu1
                if abs(mp)>lp: continue

                # Define l+Delta,-m-mu1 index
                lmp_ind = abs(mp)*(lmax+1)+lp-abs(mp)*(abs(mp)+1)//2
                
                # Gaunt factor
                gaunt_p = 0.25*pow(-1.0j,Delta)*gaunt_symbol(l,lp,n1,-m,-mp,-mu1)
                if gaunt_p==0: continue
                gaunt_m = gaunt_p*pow(-1.,mp)
                
                # Sum over r
                if mp>=0:
                    for u in xrange(npol):
                        tmp_sum = 0
                        for j in xrange(nr):
                            if r_weights[j]==0: continue
                            tmp_sum = tmp_sum + gaunt_p*(input_plus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]]).conjugate()*r_weights[j]
                        output[u,i] += tmp_sum
                else:
                    for u in xrange(npol):
                        tmp_sum = 0
                        for j in xrange(nr):
                            if r_weights[j]==0: continue
                            tmp_sum = tmp_sum + gaunt_m*pow(-1.,mp)*input_minus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]].conjugate()*r_weights[j]
                        output[u,i] += tmp_sum
                        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void compute_pF_map_complex_deriv(self, np.ndarray[np.complex128_t,ndim=2] _input_plus, np.ndarray[np.complex128_t,ndim=2] _input_minus, 
                            np.ndarray[np.complex128_t,ndim=4] _plLXs,
                            np.ndarray[np.float64_t,ndim=1] _r_weights,
                            int n1, int mu1, int nmax, int lmax, 
                            np.ndarray[np.complex128_t,ndim=3] _output, int[:] inds):
        """Shift an input a_lm map by Delta,-mu and multiply it by p_lL^X(r)."""

        cdef int i, j, u, l, m, lp, mp, lmp_ind, Delta, npol=len(_plLXs[0][0]), nr = len(inds)
        cdef complex gaunt_p, gaunt_m

        # Define memviews
        cdef complex[:,::1] input_plus = _input_plus, input_minus = _input_minus
        cdef complex[:,:,:,::1] plLXs = _plLXs
        cdef double[:] r_weights = _r_weights
        cdef complex[:,:,::1] output = _output

        # Sum over Delta
        for Delta in xrange(-n1,n1+1):
            if (Delta+n1)%2==1: continue # must be even

            # Compute shifted map for m >= 0
            for i in prange(self.nl, nogil=True, schedule='dynamic', num_threads=self.nthreads):
                l = self.ls[i]
                lp = l+Delta
                if lp>lmax: continue
                if lp<0: continue
                m = self.ms[i]
                mp = m-mu1
                if abs(mp)>lp: continue

                # Define l+Delta,m-mu1 index
                lmp_ind = abs(mp)*(lmax+1)+lp-abs(mp)*(abs(mp)+1)//2
                
                # Gaunt factor
                gaunt_p = 0.25*pow(-1.,m)*pow(1.0j,Delta)*gaunt_symbol(l,lp,n1,m,-mp,-mu1)
                if gaunt_p==0: continue
                gaunt_m = gaunt_p*pow(-1.,mp)
                
                # Sum over r and u
                if mp>=0:
                    for u in xrange(npol):
                        for j in xrange(nr):  
                            if r_weights[j]==0: continue  
                            output[j,u,i] += gaunt_p*input_plus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                else:
                    for u in xrange(npol):
                        for j in xrange(nr):    
                            if r_weights[j]==0: continue  
                            output[j,u,i] += gaunt_m*input_minus[lmp_ind,j].conjugate()*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]

            # Compute shifted map for m <= 0 if necessary
            for i in prange(self.nl, nogil=True, schedule='dynamic', num_threads=self.nthreads):
                l = self.ls[i]
                lp = l+Delta
                if lp>lmax: continue
                if lp<0: continue
                m = self.ms[i]
                mp = -m-mu1
                if abs(mp)>lp: continue

                # Define l+Delta,-m-mu1 index
                lmp_ind = abs(mp)*(lmax+1)+lp-abs(mp)*(abs(mp)+1)//2
                
                # Gaunt factor
                gaunt_p = 0.25*pow(-1.0j,Delta)*gaunt_symbol(l,lp,n1,-m,-mp,-mu1)
                if gaunt_p==0: continue
                gaunt_m = gaunt_p*pow(-1.,mp)
                
                # Sum over r
                if mp>=0:
                    for u in xrange(npol):
                        for j in xrange(nr):
                            if r_weights[j]==0: continue  
                            output[j,u,i] += gaunt_p*(input_plus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]]).conjugate()*r_weights[j]
                else:
                    for u in xrange(npol):
                        for j in xrange(nr):
                            if r_weights[j]==0: continue  
                            output[j,u,i] += gaunt_p*input_minus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]].conjugate()*r_weights[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void compute_pF_map_real(self, np.ndarray[np.complex128_t,ndim=2] _input_plus, np.ndarray[np.complex128_t,ndim=2] _input_minus, 
                            np.ndarray[np.float64_t,ndim=4] _plLXs,
                            np.ndarray[np.float64_t,ndim=1] _r_weights,
                            int n1, int mu1, int nmax, int lmax, np.ndarray[np.complex128_t,ndim=2] _output, int[:] inds):
        """Shift an input a_lm map by Delta,-mu, multiply it by p_lL^X(r), and sum over r.
        
        This is the simplified form for real p_lL^X"""

        cdef int i, j, u, l, m, lp, mp, lmp_ind, Delta, npol=len(_plLXs[0][0]), nr = len(inds)
        cdef complex gaunt_p, gaunt_m, tmp=0, tmp_sum

        # Define memviews
        cdef complex[:,::1] input_plus = _input_plus, input_minus = _input_minus
        cdef double[:,:,:,::1] plLXs = _plLXs
        cdef double[:] r_weights = _r_weights
        cdef complex[:,::1] output = _output
        
        # Sum over Delta
        for Delta in xrange(-n1,n1+1):
            if (Delta+n1)%2==1: continue # must be even
            
            # Compute shifted map for m >= 0
            for i in prange(self.nl, nogil=True, schedule='dynamic', num_threads=self.nthreads):
                l = self.ls[i]
                lp = l+Delta
                if lp>lmax: continue
                if lp<0: continue
                m = self.ms[i]
                mp = m-mu1
                if abs(mp)>lp: continue

                # Define l+Delta,m-mu1 index
                lmp_ind = abs(mp)*(lmax+1)+lp-abs(mp)*(abs(mp)+1)//2
                
                # Gaunt factor
                gaunt_p = 0.5*pow(-1.,m)*pow(1.0j,Delta)*gaunt_symbol(l,lp,n1,m,-mp,-mu1)
                if gaunt_p==0: continue
                gaunt_m = gaunt_p*pow(-1.,mp)
                
                # Sum over r and u
                if mp>=0:
                    for u in xrange(npol):
                        tmp_sum = 0.
                        for j in xrange(nr):    
                            if r_weights[j]==0: continue  
                            tmp_sum = tmp_sum + gaunt_p*input_plus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                        output[u,i] += tmp_sum
                else:
                    for u in xrange(npol):
                        tmp_sum = 0.
                        for j in xrange(nr):    
                            if r_weights[j]==0: continue  
                            tmp_sum = tmp_sum + gaunt_m*input_minus[lmp_ind,j].conjugate()*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                        output[u,i] += tmp_sum

            # Add -mu1 contribution
            if mu1==0: continue
            for i in prange(self.nl, nogil=True, schedule='dynamic', num_threads=self.nthreads):
                l = self.ls[i]
                lp = l+Delta
                if lp>lmax: continue
                if lp<0: continue
                m = self.ms[i]
                mp = m+mu1
                if abs(mp)>lp: continue

                # Define l+Delta,m-mu1 index
                lmp_ind = abs(mp)*(lmax+1)+lp-abs(mp)*(abs(mp)+1)//2
                
                # Gaunt factor
                gaunt_p = 0.5*pow(-1.,n1+mu1+m)*pow(1.0j,Delta)*gaunt_symbol(l,lp,n1,m,-mp,mu1)
                if gaunt_p==0: continue
                gaunt_m = gaunt_p*pow(-1.,mp)
                
                # Sum over r and u
                if mp>=0:
                    for u in xrange(npol):
                        tmp_sum = 0.
                        for j in xrange(nr):    
                            if r_weights[j]==0: continue  
                            tmp_sum = tmp_sum + gaunt_p*input_minus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                        output[u,i] += tmp_sum
                else:
                    for u in xrange(npol):
                        tmp_sum = 0.
                        for j in xrange(nr):    
                            if r_weights[j]==0: continue  
                            tmp_sum = tmp_sum + gaunt_m*input_plus[lmp_ind,j].conjugate()*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                        output[u,i] += tmp_sum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void compute_pF_map_real_deriv(self, np.ndarray[np.complex128_t,ndim=2] _input_plus, np.ndarray[np.complex128_t,ndim=2] _input_minus, 
                            np.ndarray[np.float64_t,ndim=4] _plLXs,
                            np.ndarray[np.float64_t,ndim=1] _r_weights,
                            int n1, int mu1, int nmax, int lmax, np.ndarray[np.complex128_t,ndim=3] _output, int[:] inds):
        """Shift an input a_lm map by Delta,-mu, multiply it by p_lL^X(r), and sum over r.
        
        This is the simplified form for real p_lL^X"""

        cdef int i, j, u, l, m, lp, mp, lmp_ind, Delta, npol=len(_plLXs[0][0]), nr = len(inds)
        cdef complex gaunt_p, gaunt_m

        # Define memviews
        cdef complex[:,::1] input_plus = _input_plus, input_minus = _input_minus
        cdef double[:,:,:,::1] plLXs = _plLXs
        cdef double[:] r_weights = _r_weights
        cdef complex[:,:,::1] output = _output

        # Sum over Delta
        for Delta in xrange(-n1,n1+1):
            if (Delta+n1)%2==1: continue # must be even

            # Compute shifted map for m >= 0
            for i in prange(self.nl, nogil=True, schedule='dynamic', num_threads=self.nthreads):
                l = self.ls[i]
                lp = l+Delta
                if lp>lmax: continue
                if lp<0: continue
                m = self.ms[i]
                mp = m-mu1
                if abs(mp)>lp: continue

                # Define l+Delta,m-mu1 index
                lmp_ind = abs(mp)*(lmax+1)+lp-abs(mp)*(abs(mp)+1)//2
                
                # Gaunt factor
                gaunt_p = 0.5*pow(-1.,m)*pow(1.0j,Delta)*gaunt_symbol(l,lp,n1,m,-mp,-mu1)
                if gaunt_p==0: continue
                gaunt_m = gaunt_p*pow(-1.,mp)
                
                # Sum over r and u
                for u in xrange(npol):
                    for j in xrange(nr):    
                        if r_weights[j]==0: continue  
                        if mp>=0:
                            output[j,u,i] += gaunt_p*input_plus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                        else:
                            output[j,u,i] += gaunt_m*input_minus[lmp_ind,j].conjugate()*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]

            # Repeat for mu1 < 0
            if mu1==0: continue
            for i in prange(self.nl, nogil=True, schedule='static', num_threads=self.nthreads):
                l = self.ls[i]
                lp = l+Delta
                if lp>lmax: continue
                if lp<0: continue
                m = self.ms[i]
                mp = m+mu1
                if abs(mp)>lp: continue

                # Define l+Delta,m-mu1 index
                lmp_ind = abs(mp)*(lmax+1)+lp-abs(mp)*(abs(mp)+1)//2
                
                # Gaunt factor
                gaunt_p = 0.5*pow(-1.,m+n1+mu1)*pow(1.0j,Delta)*gaunt_symbol(l,lp,n1,m,-mp,mu1)
                if gaunt_p==0: continue
                gaunt_m = gaunt_p*pow(-1.,mp)
                
                # Sum over r and u
                for u in xrange(npol):
                    for j in xrange(nr):    
                        if r_weights[j]==0: continue  
                        if mp>=0:
                            output[j,u,i] += gaunt_p*input_minus[lmp_ind,j]*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]
                        else:
                            output[j,u,i] += gaunt_m*input_plus[lmp_ind,j].conjugate()*plLXs[nmax+Delta,l,u,inds[j]]*r_weights[j]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline int LM_index(self, int Lp, int Mp) noexcept nogil:
        """Compute the index of (L',M') in the Ls, Ms arrays."""
        if abs(Mp)>self.nLmin:
            return abs(Mp)*(self.nLmax-self.nLmin+1)+(Lp-self.nLmin)-(abs(Mp)-self.nLmin)*(abs(Mp)-self.nLmin+1)//2
        else:
            return abs(Mp)*(self.nLmax-self.nLmin+1)+(Lp-self.nLmin)

## Wigner symbols
cdef inline double threej(int l1, int l2, int l3, int m1, int m2, int m3) noexcept nogil:
    """ThreeJ symbol (separately defined to be compatible with any wigner pipeline.)"""
    return wigner3j(l1,l2,l3,m1,m2,m3)

cdef inline double gaunt_symbol(int l1, int l2, int l3, int m1, int m2, int m3) noexcept nogil:
    """Gaunt symbol"""
    return threej(l1,l2,l3,m1,m2,m3)*threej(l1,l2,l3,0,0,0)*sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*M_PI))