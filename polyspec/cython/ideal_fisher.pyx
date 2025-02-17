#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs, M_PI, sqrt, exp, pow as dpow
from cython.parallel import prange

cdef extern from "complex.h" nogil:
    double creal(double complex)
    double cimag(double complex)

## General Utilities
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] r2c(double[:,::1] re_arr, double[:,::1] im_arr, int nthreads):
    """Utility function to transform real/im parts of a 2D array to a complex array"""
    cdef int i, j, si = re_arr.shape[0], sj = re_arr.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out_arr = np.zeros((si, sj), dtype=np.complex128)
    for i in prange(si,nogil=True,schedule='static',num_threads=nthreads):
        for j in xrange(sj):
            out_arr[i,j] = re_arr[i,j]+1.0j*im_arr[i,j]
    return out_arr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] r2cstar(double[:,::1] re_arr, double[:,::1] im_arr, int nthreads):
    """Utility function to transform real/im parts of a 2D array to a complex array"""
    cdef int i, j, si = re_arr.shape[0], sj = re_arr.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out_arr = np.zeros((si, sj), dtype=np.complex128)
    for i in prange(si,nogil=True,schedule='static',num_threads=nthreads):
        for j in xrange(sj):
            out_arr[i,j] = re_arr[i,j]-1.0j*im_arr[i,j]
    return out_arr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void r2cstar_inplace(double[:,::1] re_arr, double[:,::1] im_arr, complex[:,:,::1] out_arr, int mu_index, int nthreads):
    """Utility function to transform real/im parts of a 2D array to a complex array"""
    cdef int i, j, si = re_arr.shape[0], sj = re_arr.shape[1]
    for i in prange(si,nogil=True,schedule='static',num_threads=nthreads):
        for j in xrange(sj):
            out_arr[mu_index,i,j] = re_arr[i,j]-1.0j*im_arr[i,j]
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] multiplyRC(double[:,::1] arrR, complex[:,::1] arrC, int nthreads):
    """Multiply a real and a complex 2D map together."""
    cdef int i1, i2, n1 = arrR.shape[0], n2 = arrR.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1, n2), dtype=np.complex128)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = arrR[i1,i2]*arrC[i1,i2]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] multiplyRCstar(double[:,::1] arrR, complex[:,::1] arrC, double fac, int nthreads):
    """Multiply a real and a complex 2D map together, adding a conjugate."""
    cdef int i1, i2, n1 = arrR.shape[0], n2 = arrR.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1, n2), dtype=np.complex128)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = fac*arrR[i1,i2]*arrC[i1,i2].conjugate()
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] multiplyRC_sym(double[:,::1] arrR, complex[:,::1] arrC, double[:,::1] arrR2, complex[:,::1] arrC2, int nthreads):
    """Multiply a real and a complex 2D map together and add to an output array."""
    cdef int i1, i2, n1 = arrR.shape[0], n2 = arrR.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1, n2), dtype=np.complex128)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = arrR[i1,i2]*arrC[i1,i2]+arrR2[i1,i2]*arrC2[i1,i2]
    return out
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] multiplyRCstar_sym(double[:,::1] arrR, complex[:,::1] arrC, double[:,::1] arrR2, complex[:,::1] arrC2, double fac, int nthreads):
    """Multiply a real and a complex 2D map together and add to an output array, adding a conjugate."""
    cdef int i1, i2, n1 = arrR.shape[0], n2 = arrR.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1, n2), dtype=np.complex128)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = fac*(arrR[i1,i2]*arrC[i1,i2].conjugate()+arrR2[i1,i2]*arrC2[i1,i2].conjugate())
    return out
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] multiplyRR(double[:,::1] arr1, double[:,::1] arr2, int nthreads):
    """Multiply two real 2D maps together."""
    cdef int i1, i2, n1 = arr1.shape[0], n2 = arr2.shape[1]
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1, n2), dtype=np.float64)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = arr1[i1,i2]*arr2[i1,i2]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] multiplyRR_sym(double[:,::1] arr1, double[:,::1] arr2, double[:,::1] arr1b, double[:,::1] arr2b, int nthreads):
    """Multiply a real and a complex 2D map together and add to an output array."""
    cdef int i1, i2, n1 = arr1.shape[0], n2 = arr2.shape[1]
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1, n2), dtype=np.float64)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = arr1[i1,i2]*arr2[i1,i2]+arr1b[i1,i2]*arr2b[i1,i2]
    return out
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=1] lens_phi_sum(complex[:,::1] umap, complex[:,::1] vmap, int nthreads):
    """Compute the sum over U and V maps required for the lensing Phi estimator""" 
    cdef int i, ipol, npol = umap.shape[0], npix = umap.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=1] out = np.zeros(npix, dtype=np.complex128)

    for i in prange(npix, nogil=True, schedule='static', num_threads=nthreads):
        if npol==1:
            # Spin-0
            out[i] = 2*umap[0,i]*vmap[0,i].conjugate()
        else:
            # All spins
            out[i] = 2*umap[0,i]*vmap[0,i].conjugate()+(umap[1,i]*vmap[1,i].conjugate()-umap[1,i].conjugate()*vmap[2,i])+1.0j*(umap[2,i]*vmap[1,i].conjugate()+umap[2,i].conjugate()*vmap[2,i])

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=1] lens_phi_sum_sym(complex[:,::1] umap1, complex[:,::1] umap2, complex[:,::1] vmap1, complex[:,::1] vmap2, int nthreads):
    """Compute the sum over U and V maps required for the lensing Phi estimator, symmetrizing over two sets of fields.""" 
    cdef int i, ipol, npol = umap1.shape[0], npix = umap1.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=1] out = np.zeros(npix, dtype=np.complex128)
    
    for i in prange(npix, nogil=True, schedule='static', num_threads=nthreads):
        if npol==1:
            # Spin-0
            out[i] = 2*(umap1[0,i]*vmap2[0,i].conjugate()+umap2[0,i]*vmap1[0,i].conjugate())
        else:   
            # All spins
            out[i] =  2*umap1[0,i]*vmap2[0,i].conjugate()+(umap1[1,i]*vmap2[1,i].conjugate()-umap1[1,i].conjugate()*vmap2[2,i])+1.0j*(umap1[2,i]*vmap2[1,i].conjugate()+umap1[2,i].conjugate()*vmap2[2,i])+2*umap2[0,i]*vmap1[0,i].conjugate()+(umap2[1,i]*vmap1[1,i].conjugate()-umap2[1,i].conjugate()*vmap1[2,i])+1.0j*(umap2[2,i]*vmap1[1,i].conjugate()+umap2[2,i].conjugate()*vmap1[2,i])
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productP_all(complex[:,:,::1] P_map, complex[:,:,::1] F_PQ, double[:] weights, int[:] pq_inds, int[:] F_inds, int nthreads):
    """Utility function to take the real product of two maps and a scalar."""
    cdef int n1 = P_map.shape[1], n2 = P_map.shape[2], nw = len(weights)
    cdef int i1, i2, iw
    cdef double tmp_out
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)
    for i1 in prange(n1, nogil=True, schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            tmp_out = 0.
            for iw in xrange(nw):
                tmp_out = tmp_out + weights[iw]*creal(P_map[pq_inds[iw],i1,i2]*F_PQ[F_inds[iw],i1,i2])
            out[i1,i2] = tmp_out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productP_real_all(double[:,:,::1] P_map, complex[:,:,::1] F_PQ, double[:] weights, int[:] pq_inds, int[:] F_inds, int nthreads):
    """Utility function to take the real product of two maps and a scalar."""
    cdef int n1 = P_map.shape[1], n2 = P_map.shape[2], nw = len(weights)
    cdef int i1, i2, iw
    cdef double tmp_out
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)
    # Parallelize if there's a lot of terms
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            tmp_out = 0.
            for iw in xrange(nw):
                tmp_out = tmp_out + weights[iw]*P_map[pq_inds[iw],i1,i2]*creal(F_PQ[F_inds[iw],i1,i2])
            out[i1,i2] = tmp_out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productPnmu_sym_all(complex[:,:,:,::1] Pnmu_maps, complex[:,:,:,::1] F_PQ, double[:] weights, int n1, int[:] pq_inds, int[:] F_inds, int nthreads):
    """Utility function to take the real product of two maps and a scalar. We sum over the mu axis, assuming symmetries."""
    cdef int na = Pnmu_maps.shape[2], nb = Pnmu_maps.shape[3], nw = len(weights)
    cdef int i1, i2, mu1, iw, ip, iF
    cdef double tmp_sum
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((na,nb),dtype=np.float64)
    for i1 in prange(na, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(nb):
            tmp_sum = 0.
            for iw in xrange(nw):
                ip = pq_inds[iw]
                iF = F_inds[iw]
                tmp_sum = tmp_sum + weights[iw]*creal(Pnmu_maps[ip,0,i1,i2]*F_PQ[iF,0,i1,i2])
                for mu1 in xrange(1,n1+1):
                    tmp_sum = tmp_sum + 2.* weights[iw]*creal(Pnmu_maps[ip,mu1,i1,i2]*F_PQ[iF,mu1,i1,i2])
            out[i1,i2] = tmp_sum
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productPnmu_all(complex[:,:,:,::1] Pnmu_maps, complex[:,:,:,::1] conjPnmu_maps, complex[:,:,:,::1] F_PQ, double[:] weights, int n1, int[:] pq_inds, int[:] F_inds, int nthreads):
    """Utility function to take the real product of two maps and a scalar. We sum over the mu axis, assuming symmetries."""
    cdef int na = Pnmu_maps.shape[2], nb = Pnmu_maps.shape[3]
    cdef int i1, i2, ip, iF, mu1, iw, nw = len(weights)
    cdef double tmp_sum
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((na,nb),dtype=np.float64)
    for i1 in prange(na, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(nb):
            tmp_sum = 0.
            for iw in xrange(nw):
                ip = pq_inds[iw]
                iF = F_inds[iw]
                tmp_sum = tmp_sum + weights[iw]*creal(Pnmu_maps[ip,0,i1,i2]*F_PQ[iF,n1,i1,i2])
                for mu1 in xrange(1,n1+1):
                    tmp_sum = tmp_sum + weights[iw]*creal(Pnmu_maps[ip,mu1,i1,i2]*F_PQ[iF,n1+mu1,i1,i2]+dpow(-1.,n1-mu1)*conjPnmu_maps[ip,mu1,i1,i2].conjugate()*F_PQ[iF,n1-mu1,i1,i2])
            out[i1,i2] = tmp_sum
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] to_plus_minus(double[:,:,::1] map, int nthreads):
    """Utility function to turn the Re/Im parts of a map to the + and - maps."""
    cdef int n1 = map.shape[0], n2 = map.shape[2]
    cdef int i1, i2
    cdef double outR, outI
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((2,n1,n2),dtype=np.complex128)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            outR = map[i1,0,i2]
            outI = map[i1,1,i2]
            out[0,i1,i2] = outR+1.0j*outI
            out[1,i1,i2] = outR-1.0j*outI
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] to_plus_minus_complex(complex[:,:,::1] map, int spin, int nthreads):
    """Utility function to turn the Re/Im parts of a map to the + and - maps."""
    cdef int n1 = map.shape[0], n2 = map.shape[2]
    cdef int i1, i2
    cdef complex outR, outI
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((2,n1,n2),dtype=np.complex128)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            outR = map[i1,0,i2]
            outI = map[i1,1,i2]
            out[0,i1,i2] = -(outR+1.0j*outI)
            if spin%2==0:
                out[1,i1,i2] = -(outR-1.0j*outI)
            else:
                out[1,i1,i2] = (outR-1.0j*outI)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] complex_to_complex(complex[:,::1] map1, complex[:,::1] map2, int nthreads):
    """Utility function to turn the Re/Im parts of a map to the + and - maps."""
    cdef int n1 = map1.shape[1], n2 = map1.shape[0]
    cdef int i1, i2
    cdef complex plus, minus
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((2,n1,n2),dtype=np.complex128)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            plus = map1[i2,i1]
            minus = map2[i2,i1]
            out[0,i1,i2] = plus+1.0j*minus
            out[1,i1,i2] = plus-1.0j*minus
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] complex_to_complex_transpose(complex[:,::1] map1, complex[:,::1] map2, int nthreads):
    """Utility function to turn the Re/Im parts of a map to the + and - maps."""
    cdef int n1 = map1.shape[0], n2 = map1.shape[1]
    cdef int i1, i2
    cdef complex plus, minus
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((2,n1,n2),dtype=np.complex128)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            plus = map1[i1,i2]
            minus = map2[i1,i2]
            out[0,i1,i2] = plus+1.0j*minus
            out[1,i1,i2] = plus-1.0j*minus
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] to_real_imag(complex[:,::1] cmapP, complex[:,::1] cmapM, int nthreads):
    """Utility function to turn the +/- parts of a map to the Re and Im maps."""
    cdef int n1 = cmapP.shape[0], n2 = cmapP.shape[1]
    cdef int i1, i2
    cdef complex outP, outM
    cdef np.ndarray[np.float64_t,ndim=3] out = np.zeros((n1,2,n2),dtype=np.float64)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            outP = cmapP[i1,i2]
            outM = cmapM[i1,i2]
            out[i1,0,i2] = creal(outP+outM)/2.
            out[i1,1,i2] = cimag(outP-outM)/2.
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productQ_real_all(double[:,:,::1] Q_map, complex[:,:,:,::1] F_PQ, double[:] weights, int[:] pq_inds, int[:] F_inds, int mu1_index, int nthreads):
    """Utility function to take the real product of two maps and a scalar. We assume the maps are real."""
    cdef int n1 = Q_map.shape[1], n2 = Q_map.shape[2], nw = len(weights)
    cdef int i1, i2, iw
    cdef double tmp_out
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)
    # Parallelize if large matrix
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            tmp_out = 0.
            for iw in xrange(nw):
                tmp_out = tmp_out+weights[iw]*Q_map[pq_inds[iw],i1,i2]*creal(F_PQ[F_inds[iw],mu1_index,i1,i2])
            out[i1,i2] = tmp_out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] compute_productQ_complex_all(double[:,:,::1] Q_map, complex[:,:,:,::1] F_PQ, double[:] weights, int[:] pq_inds, int[:] F_inds, int mu1_index, int nthreads):
    """Utility function to take the complex product of two maps and a scalar. We assume the first map is real."""
    cdef int n1 = Q_map.shape[1], n2 = Q_map.shape[2], nw = len(weights)
    cdef int i1, i2, iw
    cdef complex tmp_out
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1,n2),dtype=np.complex128)
        
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            tmp_out = 0.
            for iw in xrange(nw):
                tmp_out = tmp_out + weights[iw]*Q_map[pq_inds[iw],i1,i2]*F_PQ[F_inds[iw],mu1_index,i1,i2]
            out[i1,i2] = tmp_out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void integrate_pq(double[:,:,::1] flXs, complex[:,::1] P_FPQ, long[:] ls, double[:] r_weights, complex[:,::1] out, int[:] inds, int nthreads):
    """Utility function to sum over the radial axis of an array, weighted by f_l^X(r)."""
    cdef int nlm = P_FPQ.shape[0], npol = flXs.shape[1], nr = len(inds)
    cdef int ipol, l, ilm, ir
    cdef complex tmp_out
    with nogil:
        for ipol in xrange(npol):
            for ilm in prange(nlm,schedule='static',num_threads=nthreads):
                l = ls[ilm]
                tmp_out = 0.
                for ir in xrange(nr):
                    tmp_out = tmp_out + 0.5*r_weights[ir]*flXs[l,ipol,inds[ir]]*P_FPQ[ilm,ir]
                out[ipol,ilm] += tmp_out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void integrate_pq_complex(complex[:,:,::1] flXs, complex[:,::1] P_FPQ_plus, complex[:,::1] P_FPQ_minus, long[:] ls, double[:] r_weights, complex[:,::1] out, int[:] inds, int nthreads):
    """Utility function to sum over the radial axis of an array, weighted by f_l^X(r). We allow for two complex input maps."""
    cdef int nlm = P_FPQ_plus.shape[0], npol = flXs.shape[1], nr = len(inds)
    cdef int ipol, l, ilm, ir
    cdef complex tmp_out
    with nogil:
        for ipol in xrange(npol):
            for ilm in prange(nlm,schedule='static',num_threads=nthreads):
                l = ls[ilm]
                tmp_out = 0.
                for ir in xrange(nr):
                    tmp_out = tmp_out + 0.25*r_weights[ir]*(flXs[l,ipol,inds[ir]]*P_FPQ_plus[ilm,ir]+flXs[l,ipol,inds[ir]].conjugate()*P_FPQ_minus[ilm,ir])
                out[ipol,ilm] += tmp_out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void integrate_pq_deriv(double[:,:,::1] flXs, complex[:,::1] P_FPQ, long[:] ls, double[:] r_weights, complex[:,:,::1] out, int[:] inds, int nthreads):
    """Utility function to sum over the radial axis of an array, weighted by f_l^X(r)."""
    cdef int nlm = P_FPQ.shape[0], npol = flXs.shape[1], nr = len(inds)
    cdef int ipol, ilm, l, ir
    with nogil:
        for ipol in xrange(npol):
            for ilm in prange(nlm,schedule='static',num_threads=nthreads):
                l = ls[ilm]
                for ir in xrange(nr):
                    out[ir,ipol,ilm] += 0.5*r_weights[ir]*flXs[l,ipol,inds[ir]]*P_FPQ[ilm,ir]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void integrate_pq_complex_deriv(complex[:,:,::1] flXs, complex[:,::1] P_FPQ_plus, complex[:,::1] P_FPQ_minus, long[:] ls, double[:] r_weights, complex[:,:,::1] out, int[:] inds, int nthreads):
    """Utility function to sum over the radial axis of an array, weighted by f_l^X(r)."""
    cdef int nlm = P_FPQ_plus.shape[0], npol = flXs.shape[1], nr = len(inds)
    cdef int ipol, ilm, l, ir
    with nogil:
        for ipol in xrange(npol):
            for ilm in prange(nlm,schedule='static',num_threads=nthreads):
                l = ls[ilm]
                for ir in xrange(nr):
                    out[ir,ipol,ilm] += 0.5*r_weights[ir]*(flXs[l,ipol,inds[ir]]*P_FPQ_plus[ilm,ir]+flXs[l,ipol,inds[ir]].conjugate()*P_FPQ_minus[ilm,ir])/2.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] outer_product(complex[:,:,::1] Q4_a, complex[:,:,::1] Q4_b, int nthreads, bint sym):
    """Compute Fisher matrix between two large arrays as an outer product.
    
    If "sym" is specified, we assume a square symmetric matrix."""
    
    cdef int n_out1 = Q4_a.shape[1], n_out2 = Q4_b.shape[1], n_in = Q4_a.shape[2]
    cdef int iab, ia, ib, j
    cdef complex tmp
    cdef np.ndarray[np.float64_t,ndim=2] fish = np.zeros((n_out1,n_out2),dtype=np.float64)
    if sym:
        assert n_out1==n_out2, "Matrix must be square!"

    if sym:
        with nogil:
            for iab in xrange(n_out1*n_out2):
                ia = iab//n_out2
                ib = iab%n_out2
                if ia > ib: continue
                tmp = 0.
                for j in prange(n_in,schedule='static',num_threads=nthreads):
                    tmp += (Q4_a[0,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[1,ib,j])+9*(Q4_a[2,ia,j].conjugate()*Q4_b[2,ib,j]+Q4_a[3,ia,j].conjugate()*Q4_b[3,ib,j])-3*(Q4_a[0,ia,j].conjugate()*Q4_b[3,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[2,ib,j]+Q4_a[3,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[2,ia,j].conjugate()*Q4_b[1,ib,j])
                fish[ia,ib] += creal(tmp)/24./48.
                if ia!=ib:
                    fish[ib,ia] += creal(tmp)/24./48.
    else:
        with nogil:
            for iab in xrange(n_out1*n_out2):
                ia = iab//n_out2
                ib = iab%n_out2
                tmp = 0.
                for j in prange(n_in,schedule='static',num_threads=nthreads):
                    tmp += (Q4_a[0,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[1,ib,j])+9*(Q4_a[2,ia,j].conjugate()*Q4_b[2,ib,j]+Q4_a[3,ia,j].conjugate()*Q4_b[3,ib,j])-3*(Q4_a[0,ia,j].conjugate()*Q4_b[3,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[2,ib,j]+Q4_a[3,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[2,ia,j].conjugate()*Q4_b[1,ib,j])
                fish[ia,ib] += creal(tmp)/24./48.
    
    return fish  

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] outer_product_ideal(complex[:,:,::1] Q4_a, complex[:,:,::1] Q4_b, int nthreads, bint sym):
    """Compute Fisher matrix between two large arrays as an outer product. This is parallelized across templates, not l,m.
    
    We assume a square matrix."""
    
    cdef int n_out1 = Q4_a.shape[1], n_out2 = Q4_b.shape[1], n_in = Q4_a.shape[2]
    cdef int iab, ia, ib, j
    cdef complex tmp
    cdef np.ndarray[np.float64_t,ndim=2] fish = np.zeros((n_out1,n_out2),dtype=np.float64)
        
    if sym:
        assert n_out1==n_out2, "Matrix must be square!"
        with nogil:
            for iab in prange(n_out1*n_out2,schedule='dynamic',num_threads=nthreads):
                ia = iab//n_out2
                ib = iab%n_out2
                if ia > ib: continue
                tmp = 0.
                for j in xrange(n_in):
                    tmp = tmp+Q4_a[0,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[1,ib,j]+9*Q4_a[2,ia,j].conjugate()*Q4_b[2,ib,j]+9*Q4_a[3,ia,j].conjugate()*Q4_b[3,ib,j]-3*Q4_a[0,ia,j].conjugate()*Q4_b[3,ib,j]-3*Q4_a[1,ia,j].conjugate()*Q4_b[2,ib,j]-3*Q4_a[3,ia,j].conjugate()*Q4_b[0,ib,j]-3*Q4_a[2,ia,j].conjugate()*Q4_b[1,ib,j]
                fish[ia,ib] = creal(tmp)/24./48.
                if ia!=ib:
                    fish[ib,ia] = creal(tmp)/24./48.
    else:
        with nogil:
            for iab in prange(n_out1*n_out2,schedule='static',num_threads=nthreads):
                ia = iab//n_out2
                ib = iab%n_out2
                tmp = 0.
                for j in xrange(n_in):
                    tmp = tmp+Q4_a[0,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[1,ib,j]+9*Q4_a[2,ia,j].conjugate()*Q4_b[2,ib,j]+9*Q4_a[3,ia,j].conjugate()*Q4_b[3,ib,j]-3*Q4_a[0,ia,j].conjugate()*Q4_b[3,ib,j]-3*Q4_a[1,ia,j].conjugate()*Q4_b[2,ib,j]-3*Q4_a[3,ia,j].conjugate()*Q4_b[0,ib,j]-3*Q4_a[2,ia,j].conjugate()*Q4_b[1,ib,j]
                fish[ia,ib] = creal(tmp)/24./48.
    return fish  

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] apply_ideal_weight(complex[:,:,:,::1] a_map, double[:,:,::1] Ainv, double[:] m_weight, int nthreads):
    """Compute A^-1 x for a list of harmonic-space map x. We additionally add a weight of (1+l>0)."""
    cdef int npol = Ainv.shape[0], nm = a_map.shape[0], nt = a_map.shape[1], nl = a_map.shape[3]
    cdef int ipol, jpol, im, it, il
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((nm,nt,npol*nl),dtype=np.complex128)

    # Code polarizations explicitly for speed
    if npol==1:
        with nogil:
            for im in xrange(nm):
                for it in prange(nt,schedule='static',num_threads=nthreads):
                    for il in xrange(nl):
                        out[im,it,il] = m_weight[il]*Ainv[0,0,il]*a_map[im,it,0,il]
    else:
        with nogil:
            for im in xrange(nm):
                for it in prange(nt,schedule='static',num_threads=nthreads):
                    for ipol in xrange(npol):
                        for il in xrange(nl):
                            out[im,it,ipol*nl+il] = m_weight[il]*(Ainv[ipol,0,il]*a_map[im,it,0,il]+Ainv[ipol,1,il]*a_map[im,it,1,il]+Ainv[ipol,2,il]*a_map[im,it,2,il])

    return out
    
## Ideal Fisher
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_dotdot(double[:,:,::1] alXs, double[:] tau_arr, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{dot,dot} template."""

    cdef int nl = lmax+1-lmin, nr = len(alXs[0,0]), npol = len(alXs[0]), nmu = len(w_mus)
    cdef int il, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum, partial_sum
    cdef double[:] rfactor = np.zeros(nr,dtype=np.float64)
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaAA_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 24./dpow(4.*M_PI,2.)*dpow(384./25.,2.)/2.
    
    # Precompute r-dependent and l-dependent factors
    with nogil:
        for ir in xrange(nr):
            rfactor[ir] = weights[ir]*dpow(tau_arr[ir],4.)
        for il in xrange(nl):
            twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                partial_sum = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        partial_sum = partial_sum + twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*alXs[il+lmin,ipol,ir]*alXs[il+lmin,jpol,jr]
                zetaAA_l[il,ir,jr] = partial_sum

    # Compute sum over l, mu for each r, r'
    for ir in prange(nr, nogil=True,schedule='dynamic',num_threads=nthreads):
        for jr in xrange(ir+1):
            partial_sum = pref*rfactor[ir]*rfactor[jr]*_zeta_sum(zetaAA_l[:,ir,jr], legs, w_mus, nmu, nl)
            deriv_matrix[ir,jr] = partial_sum
            # Add upper triangle by symmetry
            if jr!=ir:
                deriv_matrix[jr,ir] = partial_sum

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_dotdel(double[:,:,::1] alXs, double[:,:,::1] blXs, double[:,:,::1] clXs,
                                   double[:] tau_arr, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] mu_arr, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{dot,del} template."""

    cdef int nl = lmax+1-lmin, nr = len(alXs[0,0]), npol = len(alXs[0]), nmu = len(w_mus)
    cdef int il, l, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum, sins, partial_sum
    cdef double[:] rfactor = np.zeros(nr,dtype=np.float64)
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaAA_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaBB_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaCC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaAB_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaAC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaBC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 4./dpow(4.*M_PI,2.)*dpow(3456./325.,2.)/2.
    
    # Compute Wigner d^l_ss'(theta) functions
    cdef double[:,::1] wig10s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig01s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig11s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig1m1s = np.zeros((nmu,lmax),dtype=np.float64)
    
    for imu in prange(nmu, nogil=True, schedule='dynamic',num_threads=nthreads):
        sins = sqrt(1.-mu_arr[imu]*mu_arr[imu]) # since 0<theta<pi
        wig10s[imu,0] = 1./sqrt(2.)*sins
        wig11s[imu,0] = 0.5*(1.+mu_arr[imu])
        wig1m1s[imu,0] = 0.5*(1.-mu_arr[imu])
        for l in xrange(1,lmax):
            if l==1:
                wig10s[imu,l] =  ((2.*l+1.)*mu_arr[imu]*wig10s[imu,l-1])/_alpha(l+1,1,0)
                wig11s[imu,l] =  ((2.*l+1.)*(mu_arr[imu]-1./(l*(l+1.)))*wig11s[imu,l-1])/_alpha(l+1,1,1)
                wig1m1s[imu,l] = ((2.*l+1.)*(mu_arr[imu]+1./(l*(l+1.)))*wig1m1s[imu,l-1])/_alpha(l+1,1,-1)
            else:
                wig10s[imu,l] =  ((2.*l+1.)*mu_arr[imu]*wig10s[imu,l-1]-_alpha(l,1,0)*wig10s[imu,l-2])/_alpha(l+1,1,0)
                wig11s[imu,l] =  ((2.*l+1.)*(mu_arr[imu]-1./(l*(l+1.)))*wig11s[imu,l-1]-_alpha(l,1,1)*wig11s[imu,l-2])/_alpha(l+1,1,1)
                wig1m1s[imu,l] = ((2.*l+1.)*(mu_arr[imu]+1./(l*(l+1.)))*wig1m1s[imu,l-1]-_alpha(l,1,-1)*wig1m1s[imu,l-2])/_alpha(l+1,1,-1)
    
    # Precompute r-dependent and l-dependent factors
    with nogil:
        for ir in xrange(nr):
            rfactor[ir] = weights[ir]*dpow(tau_arr[ir],2.)
        for il in xrange(nl):
            twol_arr[il] = (2.*il+2*lmin+1.)
    
    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
        for ipol in xrange(npol):
            for jpol in xrange(npol):
                partial_sum = twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]
                for ir in xrange(nr):
                    for jr in xrange(nr):
                        zetaAA_l[il,ir,jr] += partial_sum*alXs[il+lmin,ipol,ir]*alXs[il+lmin,jpol,jr]
                        zetaBB_l[il,ir,jr] += partial_sum*blXs[il+lmin,ipol,ir]*blXs[il+lmin,jpol,jr]
                        zetaCC_l[il,ir,jr] += partial_sum*clXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]
                        zetaAB_l[il,ir,jr] += partial_sum*alXs[il+lmin,ipol,ir]*blXs[il+lmin,jpol,jr]
                        zetaAC_l[il,ir,jr] += partial_sum*alXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]
                        zetaBC_l[il,ir,jr] += partial_sum*blXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]
    
    # Compute sum over l, mu for each r, r'
    for ir in prange(nr,nogil=True,schedule='dynamic',num_threads=nthreads):
        for jr in xrange(ir+1):
            partial_sum = pref*rfactor[ir]*rfactor[jr]*_zeta_sum_sym2(zetaAA_l[:,ir,jr], zetaBB_l[:,ir,jr], zetaCC_l[:,ir,jr], 
                                                                            zetaAB_l[:,ir,jr], zetaAC_l[:,ir,jr], zetaBC_l[:,ir,jr], 
                                                                            zetaAB_l[:,jr,ir], zetaAC_l[:,jr,ir], zetaBC_l[:,jr,ir], 
                                                                            legs, wig10s, wig11s, wig1m1s, w_mus, nmu, nl, lmin)
            deriv_matrix[ir,jr] = partial_sum
            # Add upper triangle by symmetry
            if jr!=ir:
                deriv_matrix[jr,ir] = partial_sum
    
    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_deldel(double[:,:,::1] blXs, double[:,:,::1] clXs,
                                            double[:] weights, double[:,:,::1] inv_Cl_mat,
                                            double[:,::1] legs, double[:] mu_arr, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{del,del} template."""

    cdef int nl = lmax+1-lmin, nr = len(blXs[0,0]), npol = len(blXs[0]), nmu = len(w_mus)
    cdef int il, l, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum, sins, partial_sum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaBB_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaCC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaBC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 4./dpow(4.*M_PI,2.)*dpow(10368./2575.,2.)/2.
    
    # Compute Wigner d^l_ss'(theta) functions
    cdef double[:,::1] wig10s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig01s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig11s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig1m1s = np.zeros((nmu,lmax),dtype=np.float64)
    for imu in prange(nmu, nogil=True, schedule='static',num_threads=nthreads):
        sins = sqrt(1.-mu_arr[imu]*mu_arr[imu]) # since 0<theta<pi
        wig10s[imu,0] = 1./sqrt(2.)*sins
        wig11s[imu,0] = 0.5*(1.+mu_arr[imu])
        wig1m1s[imu,0] = 0.5*(1.-mu_arr[imu])
        for l in xrange(1,lmax):
            if l==1:
                wig10s[imu,l] =  ((2.*l+1.)*mu_arr[imu]*wig10s[imu,l-1])/_alpha(l+1,1,0)
                wig11s[imu,l] =  ((2.*l+1.)*(mu_arr[imu]-1./(l*(l+1.)))*wig11s[imu,l-1])/_alpha(l+1,1,1)
                wig1m1s[imu,l] = ((2.*l+1.)*(mu_arr[imu]+1./(l*(l+1.)))*wig1m1s[imu,l-1])/_alpha(l+1,1,-1)
            else:
                wig10s[imu,l] =  ((2.*l+1.)*mu_arr[imu]*wig10s[imu,l-1]-_alpha(l,1,0)*wig10s[imu,l-2])/_alpha(l+1,1,0)
                wig11s[imu,l] =  ((2.*l+1.)*(mu_arr[imu]-1./(l*(l+1.)))*wig11s[imu,l-1]-_alpha(l,1,1)*wig11s[imu,l-2])/_alpha(l+1,1,1)
                wig1m1s[imu,l] = ((2.*l+1.)*(mu_arr[imu]+1./(l*(l+1.)))*wig1m1s[imu,l-1]-_alpha(l,1,-1)*wig1m1s[imu,l-2])/_alpha(l+1,1,-1)
    
    # Precompute l-dependent factors
    with nogil:
        for il in xrange(nl):
            twol_arr[il] = (2.*il+2*lmin+1.)
    
    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True, schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaBB_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*blXs[il+lmin,ipol,ir]*blXs[il+lmin,jpol,jr]
                        zetaCC_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*clXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]
                        zetaBC_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*blXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]

    # Compute sum over l, mu for each r, r'
    for ir in prange(nr, nogil=True, schedule='dynamic',num_threads=nthreads):
        for jr in xrange(ir+1):
            partial_sum = pref*weights[ir]*weights[jr]*_zeta_sum_sym3(zetaBB_l[:,ir,jr], zetaCC_l[:,ir,jr], 
                                                                                zetaBC_l[:,ir,jr], zetaBC_l[:,jr,ir], 
                                                                                legs, wig10s, wig11s, wig1m1s, w_mus, nmu, nl, lmin)
            deriv_matrix[ir,jr] = partial_sum
            # Add upper triangle by symmetry
            if jr!=ir:
                deriv_matrix[jr,ir] = partial_sum
    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_con(double[:,:,::1] rlXs, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{con} template."""

    cdef int nl = lmax+1-lmin, nr = len(rlXs[0,0]), npol = len(rlXs[0]), nmu = len(w_mus)
    cdef int il, ir, ijr, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum, partial_sum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaRR_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 24./dpow(4.*M_PI,2.)*dpow(9./25.,2.)/2.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) a^Y S^-1 a^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                partial_sum = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        partial_sum = partial_sum + twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*rlXs[il+lmin,ipol,ir]*rlXs[il+lmin,jpol,jr]
                zetaRR_l[il,ir,jr] += partial_sum

    # Compute sum over l, mu for each r, r'
    for ijr in prange(nr*nr, nogil=True,schedule='static',num_threads=nthreads):
        ir = ijr//nr
        jr = ijr%nr
        deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum(zetaRR_l[:,ir,jr], legs, w_mus, nmu, nl)

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_fNL_loc(double[:,:,::1] plXs, double[:,:,::1] qlXs, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the fNL^{loc} template."""

    cdef int nl = lmax+1-lmin, nr = len(plXs[0,0]), npol = len(plXs[0]), nmu = len(w_mus)
    cdef int il, ir, ijr, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaPP_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zeta_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,2.)*18./25.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaPP_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*plXs[il+lmin,jpol,jr]
                        zetaPQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]
                        zetaQQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]

    # Compute sum over l, mu for each r, r'
    for ijr in prange(nr*nr, nogil=True,schedule='static',num_threads=nthreads):
        ir = ijr//nr
        jr = ijr%nr
        deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum_symB(zetaPP_l[:,ir,jr], zetaPQ_l[:,ir,jr], zetaPQ_l[:,jr,ir], zetaQQ_l[:,ir,jr], legs, w_mus, nmu, nl)

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_loc(double[:,:,::1] plXs, double[:,:,::1] qlXs, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{loc} template."""

    cdef int nl = lmax+1-lmin, nr = len(plXs[0,0]), npol = len(plXs[0]), nmu = len(w_mus)
    cdef int il, ir, ijr, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaPP_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zeta_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 6./dpow(4.*M_PI,2.)*dpow(9./25.,2.)/2.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaPP_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*plXs[il+lmin,jpol,jr]
                        zetaPQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]
                        zetaQQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]

    # Compute sum over l, mu for each r, r'
    for ijr in prange(nr*nr, nogil=True,schedule='static',num_threads=nthreads):
        ir = ijr//nr
        jr = ijr%nr
        deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum_sym(zetaPP_l[:,ir,jr], zetaPQ_l[:,ir,jr], zetaPQ_l[:,jr,ir], zetaQQ_l[:,ir,jr], legs, w_mus, nmu, nl)

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _alpha(int l, int s, int sp) noexcept nogil:
    return sqrt((l**2-s**2)*(l**2-sp**2))/l

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum(double[:] zetaAA_l, double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators."""
    cdef int il,imu
    cdef double musum, lsum
    musum = 0.
    for imu in xrange(nmu):
        lsum = 0.
        for il in xrange(nl):
            lsum += zetaAA_l[il]*legs[imu,il]
        musum += dpow(lsum,4.)*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_symB(double[:] zetaAA_l, double[:] zetaAB_l, double[:] zetaBA_l, double[:] zetaBB_l, 
                          double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators. This is a specialized version for the bispectrum."""
    cdef int il,imu
    cdef double musum, AAsum, ABsum, BAsum, BBsum
    musum = 0.
    for imu in xrange(nmu):
        AAsum = 0.
        ABsum = 0.
        BAsum = 0.
        BBsum = 0.
        for il in xrange(nl):
            AAsum += zetaAA_l[il]*legs[imu,il]
            ABsum += zetaAB_l[il]*legs[imu,il]
            BAsum += zetaBA_l[il]*legs[imu,il]
            BBsum += zetaBB_l[il]*legs[imu,il]
        musum += (AAsum*AAsum*BBsum+2*ABsum*BAsum*AAsum)*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_sym(double[:] zetaAA_l, double[:] zetaAB_l, double[:] zetaBA_l, double[:] zetaBB_l, 
                          double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators."""
    cdef int il,imu
    cdef double musum, AAsum, ABsum, BAsum, BBsum
    musum = 0.
    for imu in xrange(nmu):
        AAsum = 0.
        ABsum = 0.
        BAsum = 0.
        BBsum = 0.
        for il in xrange(nl):
            AAsum += zetaAA_l[il]*legs[imu,il]
            ABsum += zetaAB_l[il]*legs[imu,il]
            BAsum += zetaBA_l[il]*legs[imu,il]
            BBsum += zetaBB_l[il]*legs[imu,il]
        musum += (AAsum*AAsum*AAsum*BBsum+3*ABsum*BAsum*AAsum*AAsum)*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_sym2(double[:] zetaAA_l, double[:] zetaBB_l, double[:] zetaCC_l,
                          double[:] zetaAB_l, double[:] zetaAC_l, double[:] zetaBC_l,
                          double[:] zetaBA_l, double[:] zetaCA_l, double[:] zetaCB_l,
                          double[:,::1] legs, double[:,::1] wig10s, double[:,::1] wig11s, double[:,::1] wig1m1s, 
                          double[:] w_mus, int nmu, int nl, int lmin) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators."""
    cdef int il,imu
    cdef double musum, tmp, AAsum, BBsum, CCsum, CsCsum, ABsum, BAsum, ACsum, CAsum, BCsum, CBsum
    musum = 0.
    for imu in xrange(nmu):
        AAsum = 0.
        BBsum = 0.
        CCsum = 0.
        CsCsum = 0.
        ABsum = 0.
        BAsum = 0.
        ACsum = 0.
        CAsum = 0.
        BCsum = 0.
        CBsum = 0.
        for il in xrange(nl):
            AAsum += zetaAA_l[il]*legs[imu,il]
            BBsum += zetaBB_l[il]*legs[imu,il]
            CCsum += zetaCC_l[il]*wig1m1s[imu,lmin+il-1]
            CsCsum -= zetaCC_l[il]*wig11s[imu,lmin+il-1]
            ABsum += zetaAB_l[il]*legs[imu,il]
            BAsum += zetaBA_l[il]*legs[imu,il]
            ACsum -= zetaAC_l[il]*wig10s[imu,lmin+il-1]
            CAsum -= zetaCA_l[il]*wig10s[imu,lmin+il-1]
            BCsum -= zetaBC_l[il]*wig10s[imu,lmin+il-1]
            CBsum -= zetaCB_l[il]*wig10s[imu,lmin+il-1]
        tmp = AAsum*AAsum*(BBsum*BBsum+BCsum*BCsum+CBsum*CBsum+CsCsum*CsCsum/2.+CCsum*CCsum/2.)
        tmp += 4.*AAsum*(ABsum*BAsum*BBsum+ACsum*BAsum*BCsum+ABsum*CAsum*CBsum+ACsum*CAsum*(CCsum+CsCsum)/2.)
        tmp += ABsum*ABsum*(BAsum*BAsum+CAsum*CAsum)+ACsum*ACsum*(BAsum*BAsum+CAsum*CAsum)
        musum += tmp*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_sym3(double[:] zetaBB_l, double[:] zetaCC_l,
                           double[:] zetaBC_l, double[:] zetaCB_l,
                           double[:,::1] legs, double[:,::1] wig10s, double[:,::1] wig11s, double[:,::1] wig1m1s, 
                           double[:] w_mus, int nmu, int nl, int lmin) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators."""
    cdef int il,imu
    cdef double musum, tmp, BBsum, CCsum, CsCsum, BCsum, CBsum
    musum = 0.
    for imu in xrange(nmu):
        BBsum = 0.
        CCsum = 0.
        CsCsum = 0.
        BCsum = 0.
        CBsum = 0.
        for il in xrange(nl):
            BBsum += zetaBB_l[il]*legs[imu,il]
            CCsum += zetaCC_l[il]*wig1m1s[imu,lmin+il-1]
            CsCsum -= zetaCC_l[il]*wig11s[imu,lmin+il-1]
            BCsum -= zetaBC_l[il]*wig10s[imu,lmin+il-1]
            CBsum -= zetaCB_l[il]*wig10s[imu,lmin+il-1]
        tmp = 6*(BBsum*BBsum*BBsum*BBsum+BCsum*BCsum*BCsum*BCsum+CBsum*CBsum*CBsum*CBsum)+12*BBsum*BBsum*(BCsum*BCsum+CBsum*CBsum)+4*(BCsum*BCsum+CBsum*CBsum)*(CsCsum*CCsum+CCsum*CCsum+CsCsum*CsCsum)
        tmp += 2*BBsum*BBsum*(CsCsum*CsCsum+CCsum*CCsum)+8*BBsum*BCsum*CBsum*(CsCsum+CCsum)+4*BCsum*BCsum*CBsum*CBsum
        tmp += CCsum*CCsum*CCsum*CCsum+CsCsum*CsCsum*CsCsum*CsCsum+4*CsCsum*CsCsum*CCsum*CCsum
        musum += tmp*w_mus[imu]
    return musum