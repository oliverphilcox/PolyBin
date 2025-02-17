#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs, M_PI, M_E, sqrt, exp, pow as dpow, sin
from cython.parallel import prange

cdef extern from "gsl/gsl_errno.h" nogil:
    void gsl_set_error_handler_off()
    
cdef extern from "gsl/gsl_sf_bessel.h" nogil:
    int gsl_sf_bessel_jl_steed_array(int, double, double*)

cdef extern from "gsl/gsl_sf_gamma.h" nogil:
    double gsl_sf_lngamma(double)
    
cdef extern from "gsl/gsl_sf_hyperg.h" nogil:
    double gsl_sf_hyperg_2F1(double, double, double, double)

cdef extern from "complex.h" nogil:
    double complex cpow(double complex, double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void compute_bessel(double[:] x_arr, int lmin, int lmax, double[:,::1] jlx_arr, int nthreads):
    """Compute j_ell(x) for all ell in [lmin, lmax] and an array of x values. We use Steed's method, filling in out-of-bounds areas with low-x or high-x approximations. Values below 1e-150 are set to zero."""
    cdef long ix, l, nx=len(x_arr)
    cdef double[:] tmp = np.zeros(nx*(lmax+1),dtype=np.float64)
    cdef double small = 1e-150 # set to zero below this

    # Iterate over x values    
    for ix in prange(nx,schedule='dynamic',nogil=True,num_threads=nthreads):
        
        # Special case for x = 0
        if x_arr[ix] == 0.:
            tmp[ix*(lmax+1)] = 1. # this is only non-zero value

        # Use large-x limit if needed 
        elif x_arr[ix]>18000+lmax:
            for l in xrange(lmax+1):
                tmp[ix*(lmax+1)+l] = bessel_largex(l,x_arr[ix])
                
        else:
            # Should be safe!
            if x_arr[ix]>lmax:
                
                # Compute Bessel functions using Steed's approximation
                gsl_sf_bessel_jl_steed_array(lmax,x_arr[ix],&tmp[ix*(lmax+1)])
                
            # Check for underflow
            if abs(bessel_smallx(lmax, x_arr[ix])) > small:
                
                # Compute Bessel functions using Steed's approximation
                gsl_sf_bessel_jl_steed_array(lmax,x_arr[ix],&tmp[ix*(lmax+1)])
            
            else:
                # Check where underflow occurs
                for l in xrange(lmax,0,-1):
                    if abs(bessel_smallx(l,x_arr[ix])) > small:
                        break
                
                # Compute Bessel functions up to this value with Steed's approximation (others are set to zero)
                gsl_sf_bessel_jl_steed_array(l,x_arr[ix],&tmp[ix*(lmax+1)])

    # Reshape output
    for ix in prange(nx,schedule='static',nogil=True,num_threads=nthreads):
        for l in xrange(lmin,lmax+1):
            jlx_arr[l-lmin,ix] = tmp[ix*(lmax+1)+l]

cdef inline double bessel_smallx(int l, double x) noexcept nogil:
    """x << l limit of spherical Bessel function"""
    return dpow(M_E*x/(2.*l+1.),l+0.5)/sqrt(4*x*(l+0.5))

cdef inline double bessel_largex(int l, double x) noexcept nogil:
    """x >> l limit of spherical Bessel function"""
    return sin(x-l*M_PI/2.)/x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] interpolate_jlkr(double[:] x_arr, np.ndarray[np.float64_t,ndim=1] _k_arr, np.ndarray[np.float64_t,ndim=1] _r_arr, double[:,::1] jlx_arr, int nthreads):
    """Interpolate a 2D jl(x) grid to a 3D jl(k*r) grid."""
    cdef int nr = _r_arr.shape[0], nk = _k_arr.shape[0], nl = jlx_arr.shape[0]
    cdef int ik, ir, irk, ix, il
    cdef double xmin, xmax, kr, dx, xp, xm
    cdef np.ndarray[np.float64_t,ndim=3] out = np.zeros((nl,nr,nk),dtype=np.float64)
    cdef double[:] r_arr = _r_arr, k_arr = _k_arr

    # Define search indices (with GIL)
    cdef int[:,::1] indices = np.asarray(np.searchsorted(x_arr, _k_arr[np.newaxis,:]*_r_arr[:,np.newaxis]) - 1, dtype=np.int32)
    
    # Interpolate array (rate-limiting)
    for irk in prange(nr*nk,nogil=True,schedule='static',num_threads=nthreads):
        ir = irk//nk
        ik = irk%nk
        kr = k_arr[ik]*r_arr[ir]
        ix = indices[ir,ik]
        dx = x_arr[ix+1]-x_arr[ix]
        xp = (x_arr[ix+1]-kr)/dx
        xm = (kr-x_arr[ix])/dx
        for il in xrange(nl):
            out[il,ir,ik] = jlx_arr[il,ix]*xp+jlx_arr[il,ix+1]*xm

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] compute_jl_prime(int lmin, int lmax, int lmin_base, int lmax_base, double[:,:,::1] jls, int nthreads):
    """Compute j_l'(kr) from a 3D array of j_l(kr)."""
    cdef int l, ir, ik
    cdef int nr = jls.shape[1], nk = jls.shape[2]
    cdef np.ndarray[np.float64_t,ndim=3] out = np.zeros((lmax_base-lmin_base+1,nr,nk),dtype=np.float64)
    
    for l in prange(lmin_base,lmax_base+1,nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for ik in xrange(nk):
                out[l-lmin_base,ir,ik] = (l*jls[l-lmin-1,ir,ik]-(l+1.)*jls[l-lmin+1,ir,ik])/(2.*l+1.)
    return out       
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void a_integral(double[:] k_arr, double[:] tau_arr, double[:] Pzeta_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkrtau, 
                     int lmin, int lmax, int nthreads, np.ndarray[np.float64_t,ndim=3] _integs):
    """Compute the a_l^X(r, tau) integral with the trapezium rule."""
    
    cdef int il, ik, itau, nk = len(k_arr), ntau = len(tau_arr), nl = lmax+1-lmin, npol = len(Tl_arr)
    cdef double[:,::1] ktauprod = np.zeros((ntau,nk),dtype=np.float64)
    cdef double lpref, kpref, f_low, f_high, ksum
    cdef double[:,:,::1] integs = _integs

    # Compute k,tau-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kpref = 2./M_PI*dpow(k_arr[ik],13./4.)*dpow(Pzeta_arr[ik],3./4.)/2.
        for itau in xrange(ntau):
            ktauprod[itau,ik] += kpref*exp(k_arr[ik]*tau_arr[itau])
    
    # Perform sum for each polarization
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        lpref = dpow(-1.,lmin+il)
        
        # Iterate over tau
        for itau in xrange(ntau):
            
            # Compute trapezium rule
            f_low = ktauprod[itau,0]*Tl_arr[0,lmin+il,0]*jlkrtau[il,itau,0]
            for ik in xrange(1,nk):
                f_high = ktauprod[itau,ik]*Tl_arr[0,lmin+il,ik]*jlkrtau[il,itau,ik]
                integs[lmin+il,0,itau] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                f_low = f_high
    if npol>1:
        for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
            lpref = dpow(-1.,lmin+il)
            
            # Iterate over tau
            for itau in xrange(ntau):
                
                # Compute trapezium rule
                f_low = ktauprod[itau,0]*Tl_arr[1,lmin+il,0]*jlkrtau[il,itau,0]
                for ik in xrange(1,nk):
                    f_high = ktauprod[itau,ik]*Tl_arr[1,lmin+il,ik]*jlkrtau[il,itau,ik]
                    integs[lmin+il,1,itau] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                    f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void b_integral(double[:] k_arr, double[:] tau_arr, double[:] Pzeta_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkrtau_prime, 
                     int lmin, int lmax, int nthreads, np.ndarray[np.float64_t,ndim=3] _integs):
    """Compute the b_l^X(r, tau) integral with the trapezium rule."""
    
    cdef int il, ik, itau, nk = len(k_arr), ntau = len(tau_arr), nl = lmax+1-lmin, npol = len(Tl_arr)
    cdef double[:,::1] ktauprod = np.zeros((ntau,nk),dtype=np.float64)
    cdef double lpref, kpref, f_low, f_high, ksum
    cdef double[:,:,::1] integs = _integs

    # Compute k,tau-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kpref = 2./M_PI*dpow(k_arr[ik],9./4.)*dpow(Pzeta_arr[ik],3./4.)/2.
        for itau in xrange(ntau):
            ktauprod[itau,ik] += kpref*(1.-k_arr[ik]*tau_arr[itau])*exp(k_arr[ik]*tau_arr[itau])
    
    # Perform sum for each polarization
    for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
        lpref = dpow(-1.,lmin+il)
        
        # Iterate over tau
        for itau in xrange(ntau):
            
            # Compute trapezium rule
            f_low = ktauprod[itau,0]*Tl_arr[0,lmin+il,0]*jlkrtau_prime[il,itau,0]
            for ik in xrange(1,nk):
                f_high = ktauprod[itau,ik]*Tl_arr[0,lmin+il,ik]*jlkrtau_prime[il,itau,ik]
                integs[lmin+il,0,itau] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                f_low = f_high
    if npol>1:
        for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
            lpref = dpow(-1.,lmin+il)
            
            # Iterate over tau
            for itau in xrange(ntau):
                
                # Compute trapezium rule
                f_low = ktauprod[itau,0]*Tl_arr[1,lmin+il,0]*jlkrtau_prime[il,itau,0]
                for ik in xrange(1,nk):
                    f_high = ktauprod[itau,ik]*Tl_arr[1,lmin+il,ik]*jlkrtau_prime[il,itau,ik]
                    integs[lmin+il,1,itau] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                    f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void c_integral(double[:] k_arr, double[:] r_arr, double[:] tau_arr, double[:] Pzeta_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkrtau, 
                     int lmin, int lmax, int nthreads, np.ndarray[np.float64_t,ndim=3] _integs):
    """Compute the c_l^X(r, tau) integral with the trapezium rule."""
    
    cdef int il, ik, itau, nk = len(k_arr), ntau = len(tau_arr), nl = lmax+1-lmin, npol = len(Tl_arr)
    cdef double[:,::1] ktauprod = np.zeros((ntau,nk),dtype=np.float64)
    cdef double lpref, kpref, f_low, f_high, ksum
    cdef double[:,:,::1] integs = _integs

    # Compute k,tau-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kpref = 2./M_PI*dpow(k_arr[ik],5./4.)*dpow(Pzeta_arr[ik],3./4.)/2.
        for itau in xrange(ntau):
            ktauprod[itau,ik] += kpref*(1.-k_arr[ik]*tau_arr[itau])*exp(k_arr[ik]*tau_arr[itau])/r_arr[itau]    
    
    # Perform sum for each polarization
    for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
        lpref = dpow(-1.,lmin+il)*sqrt((lmin+il)*(lmin+il+1))
        
        # Iterate over tau
        for itau in xrange(ntau):
            
            # Compute trapezium rule
            f_low = ktauprod[itau,0]*Tl_arr[0,lmin+il,0]*jlkrtau[il,itau,0]
            for ik in xrange(1,nk):
                f_high = ktauprod[itau,ik]*Tl_arr[0,lmin+il,ik]*jlkrtau[il,itau,ik]
                integs[lmin+il,0,itau] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                f_low = f_high
    if npol>1:
        for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
            lpref = dpow(-1.,lmin+il)*sqrt((lmin+il)*(lmin+il+1))
            
            # Iterate over tau
            for itau in xrange(ntau):
                
                # Compute trapezium rule
                f_low = ktauprod[itau,0]*Tl_arr[1,lmin+il,0]*jlkrtau[il,itau,0]
                for ik in xrange(1,nk):
                    f_high = ktauprod[itau,ik]*Tl_arr[1,lmin+il,ik]*jlkrtau[il,itau,ik]
                    integs[lmin+il,1,itau] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                    f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void p_integral_all(double[:] k_arr, double[:] Pzeta_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkr_all, 
                     int lmin, int lmax, int arr_lmin, int arr_lmax, int this_nmax, int nthreads, np.ndarray[np.float64_t,ndim=4] _integs):
    """Compute the p_lL^X(r) integral with the trapezium rule."""
    
    cdef int il, ik, ir, n, lshift1, lshift2
    cdef int nk = len(k_arr), nr = _integs.shape[3], npol = len(Tl_arr), nmax = len(_integs)//2
    cdef double[:] kprod = np.zeros((nk),dtype=np.float64)
    cdef double lpref, f_low, f_high, ksum
    cdef double[:,:,:,::1] integs = _integs
    
    # Compute k-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kprod[ik] = 2./M_PI*k_arr[ik]*k_arr[ik]*Pzeta_arr[ik]/2.
    
    # Perform sum for each polarization
    with nogil:
        for n in xrange(-this_nmax,this_nmax+1):
        
            lshift1 = min(0,lmin-arr_lmin+n)
            lshift2 = max(0,lmin-arr_lmin+n)

            for il in prange(lmax-lmin+lshift1+1,schedule='static',num_threads=nthreads):
                lpref = dpow(-1.,lmin-lshift1+il)
                
                # Iterate over r
                for ir in xrange(nr):
                    
                    # Compute trapezium rule
                    f_low = kprod[0]*Tl_arr[0,lmin-lshift1+il,0]*jlkr_all[lshift2+il,ir,0]
                    for ik in xrange(1,nk):
                        f_high = kprod[ik]*Tl_arr[0,lmin-lshift1+il,ik]*jlkr_all[lshift2+il,ir,ik]
                        integs[nmax+n,lmin-lshift1+il,0,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                        f_low = f_high
            if npol>1:
                for il in prange(lmax-lmin+lshift1+1,schedule='static',num_threads=nthreads):
                    lpref = dpow(-1.,lmin-lshift1+il)
                    
                    # Iterate over r
                    for ir in xrange(nr):
                        
                        # Compute trapezium rule
                        f_low = kprod[0]*Tl_arr[1,lmin-lshift1+il,0]*jlkr_all[lshift2+il,ir,0]
                        for ik in xrange(1,nk):
                            f_high = kprod[ik]*Tl_arr[1,lmin-lshift1+il,ik]*jlkr_all[lshift2+il,ir,ik]
                            integs[nmax+n,lmin-lshift1+il,1,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                            f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void q_integral(double[:] k_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkr, 
                     int lmin, int lmax, int nthreads, np.ndarray[np.float64_t,ndim=3] _integs):
    """Compute the q_l^X(r) integral with the trapezium rule."""
    
    cdef int il, ik, ir, nk = len(k_arr), nr = _integs.shape[2], nl = lmax+1-lmin, npol = len(Tl_arr)
    cdef double[:] kprod = np.zeros((nk),dtype=np.float64)
    cdef double lpref, f_low, f_high, ksum
    cdef double[:,:,::1] integs = _integs

    # Compute k-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kprod[ik] = 2./M_PI*k_arr[ik]*k_arr[ik]/2.
    
    # Perform sum for each polarization
    for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
        lpref = dpow(-1.,lmin+il)
        
        # Iterate over r
        for ir in xrange(nr):
            
            # Compute trapezium rule
            f_low = kprod[0]*Tl_arr[0,lmin+il,0]*jlkr[il,ir,0]
            for ik in xrange(1,nk):
                f_high = kprod[ik]*Tl_arr[0,lmin+il,ik]*jlkr[il,ir,ik]
                integs[lmin+il,0,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                f_low = f_high
    if npol>1:
        for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
            lpref = dpow(-1.,lmin+il)
            
            # Iterate over r
            for ir in xrange(nr):
                
                # Compute trapezium rule
                f_low = kprod[0]*Tl_arr[1,lmin+il,0]*jlkr[il,ir,0]
                for ik in xrange(1,nk):
                    f_high = kprod[ik]*Tl_arr[1,lmin+il,ik]*jlkr[il,ir,ik]
                    integs[lmin+il,1,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                    f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void p_integral(double[:] k_arr, double[:] Pzeta_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkr, 
                     int lmin, int lmax, int nthreads, np.ndarray[np.float64_t,ndim=3] _integs):
    """Compute the p_l^X(r) integral with the trapezium rule."""
    
    cdef int il, ik, ir, nk = len(k_arr), nr = _integs.shape[2], nl = lmax+1-lmin, npol = len(Tl_arr)
    cdef double[:] kprod = np.zeros((nk),dtype=np.float64)
    cdef double lpref, f_low, f_high, ksum
    cdef double[:,:,::1] integs = _integs

    # Compute k-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kprod[ik] = 2./M_PI*k_arr[ik]*k_arr[ik]/2.*Pzeta_arr[ik]
    
    # Perform sum for each polarization
    for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
        lpref = dpow(-1.,lmin+il)
        
        # Iterate over r
        for ir in xrange(nr):
            
            # Compute trapezium rule
            f_low = kprod[0]*Tl_arr[0,lmin+il,0]*jlkr[il,ir,0]
            for ik in xrange(1,nk):
                f_high = kprod[ik]*Tl_arr[0,lmin+il,ik]*jlkr[il,ir,ik]
                integs[lmin+il,0,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                f_low = f_high
    if npol>1:
        for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
            lpref = dpow(-1.,lmin+il)
            
            # Iterate over r
            for ir in xrange(nr):
                
                # Compute trapezium rule
                f_low = kprod[0]*Tl_arr[1,lmin+il,0]*jlkr[il,ir,0]
                for ik in xrange(1,nk):
                    f_high = kprod[ik]*Tl_arr[1,lmin+il,ik]*jlkr[il,ir,ik]
                    integs[lmin+il,1,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                    f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void r_integral(double[:] k_arr, double[:] Pzeta_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkr, 
                     int lmin, int lmax, int nthreads, np.ndarray[np.float64_t,ndim=3] _integs):
    """Compute the r_l^X(r) integral with the trapezium rule."""
    
    cdef int il, ik, ir, nk = len(k_arr), nr = _integs.shape[2], nl = lmax+1-lmin, npol = len(Tl_arr)
    cdef double[:] kprod = np.zeros((nk),dtype=np.float64)
    cdef double lpref, f_low, f_high, ksum
    cdef double[:,:,::1] integs = _integs

    # Compute k-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kprod[ik] = 2./M_PI*k_arr[ik]*k_arr[ik]/2.*dpow(Pzeta_arr[ik],3./4.)
    
    # Perform sum for each polarization
    for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
        lpref = dpow(-1.,lmin+il)
        
        # Iterate over r
        for ir in xrange(nr):
            
            # Compute trapezium rule
            f_low = kprod[0]*Tl_arr[0,lmin+il,0]*jlkr[il,ir,0]
            for ik in xrange(1,nk):
                f_high = kprod[ik]*Tl_arr[0,lmin+il,ik]*jlkr[il,ir,ik]
                integs[lmin+il,0,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                f_low = f_high
    if npol>1:
        # Sum over ls
        for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
            lpref = dpow(-1.,lmin+il)
            
            # Iterate over r
            for ir in xrange(nr):
                
                # Compute trapezium rule
                f_low = kprod[0]*Tl_arr[1,lmin+il,0]*jlkr[il,ir,0]
                for ik in xrange(1,nk):
                    f_high = kprod[ik]*Tl_arr[1,lmin+il,ik]*jlkr[il,ir,ik]
                    integs[lmin+il,1,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                    f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void collider_p_integral_all(double[:] k_arr, double[:] Pzeta_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkr_all, 
                     complex beta, double k_coll, int lmin, int lmax, int arr_lmin, int arr_lmax, int this_nmax, int nthreads, np.ndarray[np.complex128_t,ndim=4] _integs):
    """Compute the p_lL^X(r) integral with the trapezium rule."""
    
    cdef int il, ik, ir, n, lshift1, lshift2
    cdef int nk = len(k_arr), nr = _integs.shape[3], npol = len(Tl_arr)
    cdef complex[:] kprod = np.zeros(nk,dtype=np.complex128)
    cdef double lpref, ksum
    cdef complex f_low, f_high
    cdef complex[:,:,:,::1] integs = _integs

    # Compute k-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        if k_arr[ik]>=k_coll:
            kprod[ik] = 2./M_PI*cpow(k_arr[ik],2.+beta)*Pzeta_arr[ik]/2.
        else:
            kprod[ik] = 0.
    
    # Perform sum for each polarization
    with nogil:
        for n in xrange(-this_nmax,this_nmax+1):
        
            lshift1 = min(0,lmin-arr_lmin+n)
            lshift2 = max(0,lmin-arr_lmin+n)
            
            for il in prange(lmax-lmin+lshift1+1,schedule='static',num_threads=nthreads):
                lpref = dpow(-1.,lmin-lshift1+il)
                
                # Iterate over r
                for ir in xrange(nr):
                    
                    # Compute trapezium rule
                    f_low = kprod[0]*Tl_arr[0,lmin-lshift1+il,0]*jlkr_all[lshift2+il,ir,0]
                    for ik in xrange(1,nk):
                        f_high = kprod[ik]*Tl_arr[0,lmin-lshift1+il,ik]*jlkr_all[lshift2+il,ir,ik]
                        integs[this_nmax+n,lmin-lshift1+il,0,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                        f_low = f_high
            if npol>1:
                for il in prange(lmax-lmin+lshift1+1,schedule='static',num_threads=nthreads):
                    lpref = dpow(-1.,lmin-lshift1+il)
                    
                    # Iterate over r
                    for ir in xrange(nr):
                        
                        # Compute trapezium rule
                        f_low = kprod[0]*Tl_arr[1,lmin-lshift1+il,0]*jlkr_all[lshift2+il,ir,0]
                        for ik in xrange(1,nk):
                            f_high = kprod[ik]*Tl_arr[1,lmin-lshift1+il,ik]*jlkr_all[lshift2+il,ir,ik]
                            integs[this_nmax+n,lmin-lshift1+il,1,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                            f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void compute_F_integral(double[:] r_arr, double[:,:,:,::1] FLLs, int Lmin, int Lmax, int nmax_F, double ns, double _pref, int nthreads):
    """Compute the F_LL'(r,r') integral for each L. This uses the exact result, implemented using GSL special functions."""
    cdef int ir, jr, L, n, min_L, max_L, nr = FLLs.shape[2]
    cdef double i1,i2a,i2b,i3a,i3b,prefa,prefb,lgam1,lgam2a,lgam2b, pref = dpow(2.,ns-3)*_pref
    
    # Turn off error handling, since it gives unnecessary errors
    gsl_set_error_handler_off()
            
    # Compute F integral for each L in the required range, using exact result
    with nogil:
        for n in xrange(-nmax_F,nmax_F+1):
            i2a = (ns-2-n)/2.
            i2b = (ns-2+n)/2.
            lgam2a = gsl_sf_lngamma(1.-i2a)
            lgam2b = gsl_sf_lngamma(1.-i2b)
            min_L = max([Lmin-nmax_F,0])
            max_L = Lmax+nmax_F+1

            for L in prange(min_L, max_L, num_threads=nthreads, schedule='static'):
                
                # Filter to Ls of interest
                if L+n<1 or L<1: continue
                if L+n>Lmax+nmax_F: continue

                # Define prefactors
                i1 = L+(n-1.+ns)/2.
                i3a = L+1.5
                i3b = L+n+1.5
                lgam1 = gsl_sf_lngamma(i1)
                prefa = pref*exp(lgam1-lgam2a-gsl_sf_lngamma(i3a))
                prefb = pref*exp(lgam1-lgam2b-gsl_sf_lngamma(i3b))
                
                # Compute exact solution for each r, r'
                for ir in xrange(nr):
                    for jr in xrange(nr):
                        if r_arr[ir]<=r_arr[jr]:
                            FLLs[nmax_F+n,L,ir,jr] = prefa*gsl_sf_hyperg_2F1(i1, i2a, i3a, r_arr[ir]*r_arr[ir]/(r_arr[jr]*r_arr[jr]))*dpow(r_arr[ir]/r_arr[jr],L)*dpow(r_arr[jr],1.-ns)
                        else:
                            FLLs[nmax_F+n,L,ir,jr] = prefb*gsl_sf_hyperg_2F1(i1, i2b, i3b, r_arr[jr]*r_arr[jr]/(r_arr[ir]*r_arr[ir]))*dpow(r_arr[jr]/r_arr[ir],L+n)*dpow(r_arr[ir],1.-ns)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void compute_collider_F_integral(double[:] K_arr, double[:] Pzeta_arr, double[:,:,::1] jlK, 
                                  complex betaF, double K_coll, int Lmin, int Lmax, int this_nmax_F, int coll_nmax_F, int nthreads,
                                  complex[:,:,:,::1] integs):
    """Compute the F^{-2beta}_LL'(r,r') integral for each L. The integral is computed numerically using the trapezium rule."""
    
    cdef int ik, ijr, ir, jr, n, L, nk = len(K_arr), nr = jlK.shape[1], this_Lmin, this_Lmax, ind1, ind2
    cdef complex[:] Kprod = np.zeros(nk,dtype=np.complex128)
    cdef complex f_low, f_high, tmp
    cdef int global_Lmin = max([Lmin-coll_nmax_F,0])

    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        if K_arr[ik]<=K_coll:
            Kprod[ik] = 2./M_PI*cpow(K_arr[ik],2+betaF)*Pzeta_arr[ik]/2.
        else:
            Kprod[ik] = 0.

    with nogil:
        for n in xrange(-this_nmax_F,this_nmax_F+1):
            this_Lmin = max([Lmin-this_nmax_F,0])
            this_Lmax = Lmax+this_nmax_F
            
            # Iterate over Ls in parallel
            for L in prange(this_Lmin, this_Lmax+1, schedule='dynamic', num_threads=nthreads):

                # Filter to Ls of interest
                if L+n<1 or L<1: continue
                if L+n>Lmax+this_nmax_F: continue
                ind1 = L-global_Lmin
                ind2 = L-global_Lmin+n
                
                # Iterate over r ,r'
                for ir in xrange(nr):
                    for jr in xrange(nr):
                        if n==0 and jr>ir: continue
                        
                        # Compute trapezium rule
                        f_low = Kprod[0]*jlK[ind1,ir,0]*jlK[ind2,jr,0]
                        tmp = 0.
                        for ik in xrange(1,nk):
                            f_high = Kprod[ik]*jlK[ind1,ir,ik]*jlK[ind2,jr,ik]
                            tmp = tmp+(K_arr[ik]-K_arr[ik-1])*(f_low+f_high)
                            f_low = f_high
                        integs[this_nmax_F+n,L,ir,jr] = tmp
                        if n==0 and jr!=ir:
                            integs[this_nmax_F+n,L,jr,ir] = tmp