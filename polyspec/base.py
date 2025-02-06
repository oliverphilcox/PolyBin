### Code for ideal and unwindowed binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2025)
## This module contains the basic code

import numpy as np, os, time
import multiprocessing as mp
import healpy
from scipy.interpolate import InterpolatedUnivariateSpline

class PolySpec():
    """Base class for PolySpec.
    
    Inputs:
    - Nside: HEALPix Nside
    - Cl_tot: Fiducial power spectra (including beam, pixellation, and noise). This is used for creating synthetic maps for the optimal estimators, and, optionally, generating GRFs to test on.
    - beam: Beam present in the signal maps (including any pixel window). This will be set to unity if unspecified, else deconvolved out the signal. If polarization is set, this contains two components: (Temperature-Beam, Polarization-Beam)
    - pol: Whether to include spin-2 fields in the computations. If true, Cl_tot should be a list of six spectra: [ClTT, ClTE, ClTB, ClEE, ClEB, ClBB]. If false, Cl_tot contains only ClTT.
    - backend: Which backend to use to compute spherical harmonic transforms. Options: "healpix" [requires healpy] or "ducc" [requires ducc0].   
    - nthreads: How many CPU threads to use to parallelize computations. This defaults to the maximum number available.
    """
    def __init__(self, Nside, Cl_tot, beam=[], pol=True, backend='ducc', nthreads=0):
        
        # Load attributes
        self.Nside = Nside
        self.pol = pol
        self.backend = backend
        
        # Derived parameters
        self.Npix = 12*Nside**2
        self.A_pix = 4.*np.pi/self.Npix
        self.lmax = 3*self.Nside-1
        self.l_arr,self.m_arr = healpy.Alm.getlm(self.lmax)
        if nthreads<=0:
            self.nthreads = len(os.sched_getaffinity(0))
        else:
            self.nthreads = nthreads
        
        # Define indices for T, E, B and their parities
        self.indices = {'T':0,'E':1,'B':2}
        self.parities = {'T':1,'E':1,'B':-1}
        
        # Check and load fiducial spectra
        ls = np.arange(self.lmax+1)
        if self.pol:
            assert len(Cl_tot.keys())==6 or len(Cl_tot.keys())==4, "Must specify four or six input spectra: {ClTT, ClTE, ClEE, ClBB} or {ClTT, ClTE, ClTB, ClEE, ClEB, ClBB}"
            if len(Cl_tot.keys())==4:
                Cl_tot['TB'] = Cl_tot['EB'] = 0.*Cl_tot['TT']
        for k in Cl_tot.keys():
            assert len(Cl_tot[k])>=len(ls), "Cl_tot must contain every ell mode from l=0 to l=lmax!"
        self.Cl_tot = Cl_tot.copy()
        
        if len(beam)==0:
            self.beam = np.asarray([1.+0.*self.Cl_tot['TT'] for _ in range(1+2*self.pol)])
        else:
            if self.pol:
                assert len(beam)==2 or len(beam)==3, "Beam must contain temperature and polarization or T/E/B components"
                if len(beam)==2:
                    self.beam = np.asarray([beam[0].copy(),beam[1].copy(),beam[1].copy()])
                else:
                    self.beam = np.asarray([beam[0].copy(),beam[1].copy(),beam[2].copy()])
            else:
                assert (len(beam)==1 or len(beam)==len(Cl_tot['TT'])), "Beam must contain the same ells as Cl_tot"
                self.beam = np.asarray(beam).copy().reshape(1,-1)
        for i in range(len(self.beam)):
            assert len(self.beam[i])>=len(ls), "Beam must contain every ell mode from l=0 to l=lmax!"
                
        # Define m weights (for complex conjugates)
        self.m_weight = (1.+1.*(self.m_arr>0.))
            
        # Set up relevant SHT modules
        if self.backend=='ducc':
            import ducc0
            map_info = ducc0.healpix.Healpix_Base(self.Nside, "RING")
            self.ducc_geom = map_info.sht_info()
            self.sht_lib = ducc0.sht
        elif self.backend=='healpix':
            pass
        else:
            raise Exception("Only 'healpix' and 'ducc' backends are currently implemented!")
        
        # Add a Cython utility function, if compiled
        try:
            from ideal_fisher import to_plus_minus, to_plus_minus_complex, to_real_imag
            self.to_plus_minus = lambda map: to_plus_minus(map, self.nthreads)
            self.to_plus_minus_complex = lambda map, spin: to_plus_minus_complex(map, spin, self.nthreads)
            self.to_real_imag = lambda mapP,mapM: to_real_imag(mapP,mapM, self.nthreads)
        except ImportError:
            self.to_plus_minus = lambda map: np.asarray([map[:,0]+1.0j*map[:,1],map[:,0]-1.0j*map[:,1]])
            self.to_plus_minus_complex = lambda map, spin: np.asarray([-(map[:,0]+1.0j*map[:,1]),-(-1.)**spin*(map[:,0]-1.0j*map[:,1])])
            self.to_real_imag = lambda mapP,mapM: np.swapaxes(np.asarray([np.real((mapP+mapM)/2.),np.real((mapP-mapM)/1.0j)]),0,1)
        
        # Compute C^-1 matrix
        if pol:
            Cl_tot_mat = np.moveaxis(np.asarray([[self.Cl_tot['TT'],self.Cl_tot['TE'],self.Cl_tot['TB']],
                                                 [self.Cl_tot['TE'],self.Cl_tot['EE'],self.Cl_tot['EB']],
                                                 [self.Cl_tot['TB'],self.Cl_tot['EB'],self.Cl_tot['BB']]]),[2,1,0],[0,2,1])
            
            # Find entries without any pathologies (i.e. avoiding ell=0,1)
            good_l = np.linalg.det(Cl_tot_mat)>0
            self.inv_Cl_tot_mat = np.zeros((3,3,len(ls)), dtype=np.float64, order='C')
            self.inv_Cl_tot_mat[:,:,good_l] = np.moveaxis(np.linalg.inv(Cl_tot_mat[good_l]), [0,1,2], [2,1,0])

        else:
            # Find entries without any pathologies (i.e. avoiding ell=0,1)
            good_l = self.Cl_tot['TT']>0
            self.inv_Cl_tot_mat = np.zeros((1,1,len(ls)), dtype=np.float64, order='C')
            self.inv_Cl_tot_mat[:,:,good_l] = 1./self.Cl_tot['TT'][None,None,good_l]
        
        # Cast to all l,m
        self.inv_Cl_tot_lm_mat = np.asarray(self.inv_Cl_tot_mat[:,:,self.l_arr], order='C')
        self.beam_lm = np.asarray(self.beam[:,self.l_arr], order='C')
        
        # Counter for SHTs
        self.n_SHTs_forward = 0
        self.n_SHTs_reverse = 0
        
        # Timer for SHTs
        self.time_sht = 0.
        
    def timer_func(func): 
        # Compute the execution time of a function
        def wrap_func(self,*args, **kwargs): 
            t1 = time.time() 
            result = func(self,*args, **kwargs) 
            t2 = time.time() 
            self.time_sht += t2-t1
            return result 
        return wrap_func 
        
    # Basic harmonic transform functions
    @timer_func
    def to_lm(self, input_map, lmax=None):
        """Convert from map-space to harmonic-space. If three fields are supplied, this transforms TQU -> TEB.
        
        This uses either HEALPix or DUCC, depending on the backend chosen."""
        self.n_SHTs_forward += 1+(len(input_map)==3)
        assert len(input_map) in [1,3], "Wrong input shape supplied!"
        if lmax==None:
            lmax = self.lmax
            
        if self.backend=='healpix':
            if len(input_map)==3:
                return healpy.map2alm(input_map, pol=True, iter=0, lmax=lmax) # no iteration, to match ducc
            elif len(input_map)==1:
                return healpy.map2alm(input_map[0], pol=False, iter=0, lmax=lmax).reshape(1,-1)
        elif self.backend=='ducc':
            if len(input_map)==3:
                T_lm = self.sht_lib.adjoint_synthesis(lmax=lmax, spin=0, map=input_map[:1], nthreads=self.nthreads, mmax=lmax, theta_interpol=False, **self.ducc_geom)*np.pi/(3.*self.Nside**2)
                E_lm,B_lm = self.sht_lib.adjoint_synthesis(lmax=lmax, spin=2, map=input_map[1:], nthreads=self.nthreads, mmax=lmax, theta_interpol=False, **self.ducc_geom)*np.pi/(3.*self.Nside**2)
                return np.asarray([T_lm[0],E_lm,B_lm])
            elif len(input_map)==1:
                return self.sht_lib.adjoint_synthesis(lmax=lmax, spin=0, map=np.asarray(input_map).reshape(1,-1), nthreads=self.nthreads, mmax=lmax, theta_interpol=False,**self.ducc_geom)*np.pi/(3.*self.Nside**2)
      
    @timer_func
    def to_map(self, input_lm, lmax=None):
        """Convert from harmonic-space to map-space. If three fields are supplied, this transforms TEB -> TQU.
        
        This uses either HEALPix or DUCC, depending on the backend chosen."""
        self.n_SHTs_reverse += 1+(len(input_lm)==3)
        assert len(input_lm) in [1,3], "Wrong input shape supplied!"
        if lmax==None:
            lmax = self.lmax
        assert len(input_lm[0])==(lmax+1)*(lmax+2)//2, "Wrong number of alm coefficients!"
        
        if self.backend=='healpix':
            if len(input_lm)==3:
                return healpy.alm2map(input_lm, self.Nside, pol=True, lmax=lmax)
            elif len(input_lm)==1:
                return healpy.alm2map(input_lm[0], self.Nside, pol=False, lmax=lmax)[None,:]
        elif self.backend=='ducc':
            if len(input_lm)==3:
                out = np.zeros((3,self.Npix),dtype=np.float64)
                out[0] = self.sht_lib.synthesis(alm=input_lm[:1], lmax=lmax, spin=0, nthreads=self.nthreads, mmax=lmax, theta_interpol=False, **self.ducc_geom)
                out[1:] = self.sht_lib.synthesis(alm=input_lm[1:], lmax=lmax, spin=2, nthreads=self.nthreads, mmax=lmax, theta_interpol=False, **self.ducc_geom)
                return out
            elif len(input_lm)==1:
                return self.sht_lib.synthesis(alm=input_lm, lmax=lmax, spin=0, nthreads=self.nthreads, mmax=lmax, theta_interpol=False, **self.ducc_geom)
             
    @timer_func
    def to_lm_vec(self, input_vec, spin=0, lmax=None):
        """Convert a vector of spin-s maps to harmonic-space. 

        If spin!=0 this will return both +s and -s harmonics, and input_vec is expected to be [(+s)input(n), (-s)input(n)].

        This uses either HEALPix or DUCC, depending on the backend chosen."""
        self.n_SHTs_forward += len(input_vec)
        if lmax==None:
            lmax = self.lmax
        
        if spin==0:
            if self.backend=='healpix':
                return healpy.map2alm(input_vec, pol=False, iter=0, lmax=lmax) # no iteration
            elif self.backend=='ducc':
                return self.sht_lib.adjoint_synthesis(lmax=lmax, spin=0, map=input_vec[:,None,:], nthreads=self.nthreads, mmax=lmax, theta_interpol=False,**self.ducc_geom)[:,0,:]*np.pi/(3.*self.Nside**2)
        else:
            assert spin>0, "Spin must be positive!"

            assert len(input_vec)==2, "Must input both +s and -s maps"
            
            # Define inputs
            map_inputs = self.to_real_imag(input_vec[0],input_vec[1])
            
            # Perform transformation
            if self.backend=='healpix':
                lm_outputs = np.asarray([healpy.map2alm_spin([map_inputs[r,0],map_inputs[r,1]],spin,lmax) for r in range(len(input_vec))])
            elif self.backend=='ducc':
                lm_outputs = self.sht_lib.adjoint_synthesis(lmax=lmax, spin=spin, map=map_inputs, nthreads=self.nthreads, mmax=lmax, theta_interpol=False,**self.ducc_geom)*np.pi/(3.*self.Nside**2)
            
            # Reconstruct output and return
            return self.to_plus_minus_complex(lm_outputs, spin)

    @timer_func
    def to_map_vec(self, input_lm_vec, output_spin=0, lmax=None):
        """Convert a vector of spin-0 harmonic coefficients to map-space. If spin!=0 this will return both +s and -s maps.

        This uses either HEALPix or DUCC, depending on the backend chosen."""
        self.n_SHTs_reverse += len(input_lm_vec)
        if lmax==None:
            lmax = self.lmax
        assert len(input_lm_vec[0])==(lmax+1)*(lmax+2)//2, "Wrong number of alm coefficients!"
        
        if output_spin==0:
            if self.backend=='healpix':
                return healpy.alm2map(input_lm_vec, self.Nside, pol=False, lmax=lmax)
            elif self.backend=='ducc':
                return self.sht_lib.synthesis(alm=input_lm_vec[:,None,:], lmax=lmax, spin=0, nthreads=self.nthreads, mmax=lmax, theta_interpol=False, **self.ducc_geom)[:,0,:]
        else:
            if self.backend=='healpix':
                map_outputs = np.asarray([healpy.alm2map_spin([-input_lm_vec[r],0.*input_lm_vec[r]],self.Nside,output_spin,lmax) for r in range(len(input_lm_vec))])
            elif self.backend=='ducc':
                map_outputs = self.sht_lib.synthesis(alm=np.stack([-input_lm_vec,np.zeros_like(input_lm_vec)],axis=1),
                                                     lmax=lmax, spin=output_spin, nthreads=self.nthreads, mmax=lmax, theta_interpol=False, **self.ducc_geom)
            return self.to_plus_minus(map_outputs)
            
    @timer_func
    def to_lm_spin(self, input_map_plus, input_map_minus, spin, lmax=None):
        """Convert (+-s)A from map-space to harmonic-space, weighting by (+-s)Y_{lm}. Our convention definitions follow HEALPix.

        The inputs are [(+s)M(n), (-s)M(n)] and the outputs are [(+s)M_lm, (-s)M_lm]
        """
        assert spin>0, "Spin must be positive!"
        assert type(spin)==int, "Spin must be an integer!"
        self.n_SHTs_forward += 1
        if lmax==None:
            lmax = self.lmax

        # Define inputs
        map_inputs = [np.real((input_map_plus+input_map_minus)/2.), np.real((input_map_plus-input_map_minus)/(2.0j))]

        # Perform transformation
        if self.backend=='healpix':
            lm_outputs = healpy.map2alm_spin(map_inputs,spin,lmax)
        elif self.backend=='ducc':
            lm_outputs = self.sht_lib.adjoint_synthesis(lmax=lmax, spin=spin, map=np.asarray(map_inputs).reshape(1,2,-1), nthreads=self.nthreads, mmax=lmax, theta_interpol=False,**self.ducc_geom)[0]*np.pi/(3.*self.Nside**2)
        # Reconstruct output
        lm_plus = -(lm_outputs[0]+1.0j*lm_outputs[1])
        lm_minus = -1*(-1)**spin*(lm_outputs[0]-1.0j*lm_outputs[1])

        return np.asarray([lm_plus, lm_minus])

    @timer_func
    def to_map_spin(self, _input_lm_plus, _input_lm_minus, spin, lmax=None):
        """Convert (+-s)A_lm from harmonic-space to map-space, weighting by (+-s)Y_{lm}. Our convention definitions follow HEALPix.

        The inputs are [(+s)M_lm, (-s)M_lm] and the outputs are [(+s)M(n), (-s)M(n)]
        """
        input_lm_plus = np.asarray(_input_lm_plus).copy()
        input_lm_minus = np.asarray(_input_lm_minus).copy()
        assert spin>0, "Spin must be positive!"
        assert type(spin)==int, "Spin must be an integer!"
        self.n_SHTs_reverse += 1
        if lmax==None:
            lmax = self.lmax
        assert len(input_lm_plus)==(lmax+1)*(lmax+2)//2, "Wrong number of alm coefficients!"
        
        # Define inputs
        lm_inputs = [-(input_lm_plus+(-1)**spin*input_lm_minus)/2.,-(input_lm_plus-(-1)**spin*input_lm_minus)/(2.0j)]

        # Perform transformation
        if self.backend=='healpix':
            map_outputs = healpy.alm2map_spin(lm_inputs,self.Nside,spin,lmax=lmax)
        elif self.backend=='ducc':
            map_outputs = self.sht_lib.synthesis(alm=np.asarray(lm_inputs), lmax=lmax, spin=spin, nthreads=self.nthreads, mmax=lmax, theta_interpol=False, **self.ducc_geom)

        # Reconstruct output
        map_plus = map_outputs[0]+1.0j*map_outputs[1]
        map_minus = map_outputs[0]-1.0j*map_outputs[1]

        return map_plus, map_minus

    def compute_spin_transform_map(self, a_lms, spin, lmax=None):
        """
        Compute Sum_lm {}_sY_lm a_lm for a given spin s and a set of *scalar* maps a_lm. This calls the relevant SHT functions.
        """
        if spin>0:
            return [self.to_map_spin(a_lm,(-1.)**spin*a_lm,int(abs(spin)),lmax=lmax)[0] for a_lm in a_lms]
        if spin<0:
            return [self.to_map_spin((-1.)**spin*a_lm,a_lm,int(abs(spin)),lmax=lmax)[1] for a_lm in a_lms]
        
    def safe_divide(self, x, y):
        """Function to divide maps without zero errors."""
        out = np.zeros_like(x)
        out[y!=0] = x[y!=0]/y[y!=0]
        return out

    def generate_data(self, seed=None, Cl_input={}, output_type='map', deconvolve_beam=False, b_input=None, add_B=False, remove_mean=True, sum_ells='even', lmax=None):
        """
        Generate a full-sky map with a given set of C_ell^{XY} and (optionally) b_l1l2l3. 
        The input Cl dictionary is expected to contain 'TT' and, if polarized, 'EE', 'BB', 'TE' and optionally 'TE', 'TB'.

        When adding a bispectrum, we use the method of Smith & Zaldarriaga 2006, and assume that b_l1l2l3 is separable into three identical pieces. We optionally subtract off the mean of the map (numerically, but could be done analytically), since it is not guaranteed to be zero if we include a synthetic bispectrum. We can additionally choose if l1+l2+l3 is restricted to be "even" or "odd" (or "both") when adding a primordial bispectrum.

        No mask is added at this stage, and the output can be in map- or harmonic-space. If "deconvolve_beam", we divide by the CMB beam in harmonic-space.
        """
        assert output_type in ['harmonic','map'], "Valid output types are 'harmonic' and 'map' only!"
        
        # Define seed and lmax
        if seed!=None:
            np.random.seed(seed)
        if lmax==None:
            lmax = self.lmax
            
        # Define input power spectrum
        if len(Cl_input.keys())==0:
            Cl_input = self.Cl_tot
        if self.pol:
            assert len(Cl_input.keys())==6 or len(Cl_input.keys())==4, "Need to specify {ClTT, ClEE, ClBB, ClTE} and optionally {ClTB, ClEB}."
            if len(Cl_input.keys())==4:
                Cl_input['TB'] = Cl_input['EB'] = 0.*Cl_input['TT']
        
        # Generate a_lm maps (either T or T,E,B)
        if self.pol:
            initial_lm = healpy.synalm([Cl_input['TT'], Cl_input['TE'], Cl_input['TB'], Cl_input['EE'], Cl_input['EB'], Cl_input['BB']], self.lmax, new=False)
        else:
            initial_lm = healpy.synalm([Cl_input['TT']], self.lmax, new=False)
        initial_lm = initial_lm[:,healpy.Alm.getlm(self.lmax)[0]<=lmax]
        
        if not add_B:
            if deconvolve_beam:
                this_beam = self.beam_lm[:,healpy.Alm.getlm(self.lmax)[0]<=lmax]
                initial_lm[this_beam!=0] = initial_lm[this_beam!=0]/this_beam[this_beam!=0]
                initial_lm[this_beam==0] = 0.
            if output_type=='map': return self.to_map(initial_lm, lmax=lmax)
            else: return initial_lm
        
        ls = np.arange(lmax+1)
        
        # Compute C^-1 matrix
        if len(Cl_input)==0:
            inv_Cl_lm_mat = self.inv_Cl_tot_lm_mat
        else:
            if self.pol:
                # Create two-dimensional Cl matrix
                Cl_mat = np.moveaxis(np.asarray([[Cl_input['TT'],Cl_input['TE'],Cl_input['TB']],
                                                [Cl_input['TE'],Cl_input['EE'],Cl_input['EB']],
                                                [Cl_input['TB'],Cl_input['EB'],Cl_input['BB']]]),[2,1,0],[0,2,1])
                
                # Find entries without any pathologies (i.e. avoiding ell=0,1)
                good_l = np.linalg.det(Cl_mat)>0
                inv_Cl_mat = np.zeros((3,3,len(ls)), dtype=np.float64, order='C')
                inv_Cl_mat[:,:,good_l] = np.moveaxis(np.linalg.inv(Cl_mat[good_l]), [2,1,0], [0,1,2])
            
            else:
                # Find entries without any pathologies (i.e. avoiding ell=0,1)
                good_l = Cl_input['TT']>0
                inv_Cl_mat = np.zeros((len(ls)), dtype=np.float64, order='C')
                inv_Cl_mat[good_l] = 1./Cl_input['TT'][good_l]
            
            # Cast to all ell
            inv_Cl_lm_mat = np.asarray(self.inv_Cl_tot_mat[:,:,self.l_arr], order='C')
        
        # Compute gradient map
        Cinv_lm = np.einsum('ijk,jk->ik',self.inv_Cl_tot_lm_mat,initial_lm,order='C')
        bCinv_lm = np.sum([b_input(self.l_arr)[i]*Cinv_lm[i] for i in range(len(Cinv_lm))],axis=0)
        
        # Transform to map space
        bCinv_map_pm1 = self.to_map_spin(bCinv_lm,-bCinv_lm,1)*np.asarray([[1],[-1]])
        bCinv_map_pm2 = self.to_map_spin(bCinv_lm,bCinv_lm,2)#
        
        tmp_lm  = np.asarray(self.to_lm_spin(bCinv_map_pm1[0]*bCinv_map_pm1[0],bCinv_map_pm1[1]*bCinv_map_pm1[1],2))[::-1]
        tmp_lm -= 2*np.asarray([[1],[-1]])*np.asarray(self.to_lm_spin(bCinv_map_pm1[1]*bCinv_map_pm2[0],-bCinv_map_pm1[0]*bCinv_map_pm2[1],1))
        
        # Restrict to even/odd/all l1+l2+l3
        if sum_ells=='even':
            tmp_lm = (tmp_lm[0]+tmp_lm[1])/2.
        elif sum_ells=='odd':
            tmp_lm = 1.0j*(tmp_lm[0]-tmp_lm[1])/2.
        elif sum_ells=='both':
            tmp_lm = (1.+1.0j)*tmp_lm[0]/2.+(1-1.0j)/2.*tmp_lm[1]
        else:
            raise Exception("Unknown value of sum_ells supplied")
        
        grad_lm = [1./6.*tmp_lm*b_input(self.l_arr)[i] for i in range(len(Cinv_lm))]
        
        # Apply correction to a_lm
        output_lm = np.asarray([initial_lm[i] + 1./3.*grad_lm[i] for i in range(len(Cinv_lm))])
        
        # Remove beam, if desired
        if deconvolve_beam:
            this_beam = self.beam_lm[:,healpy.Alm.getlm(self.lmax)[0]<=lmax]
            output_lm[this_beam!=0] = output_lm[this_beam!=0]/this_beam[this_beam!=0]
            output_lm[this_beam==0] = 0.

        output_map = self.to_map(output_lm)
        
        # Optionally remove mean of map
        if remove_mean:
            
            # Compute mean of synthetic maps numerically
            if not hasattr(self, 'map_offset') or (hasattr(self,'map_offset') and self.saved_spec!=[Cl_input,b_input]):
                print("Computing offset for synthetic maps")
                map_offset = 0.
                for ii in range(100):
                    map_offset += self.generate_data(Cl_input=Cl_input, b_input=b_input, seed=int(1e6)+ii, add_B=True, remove_mean=False, deconvolve_beam=deconvolve_beam)/100.
                self.map_offset = map_offset
                self.saved_spec = [Cl_input, b_input]
            # Remove mean
            output_map -= self.map_offset
            
        if output_type=='harmonic': 
            return self.to_lm(output_map)
        else:
            return output_map

    def applyAinv(self, input_map, input_type='map', lmax=-1):
        """Apply the exact inverse weighting A^{-1} to a map. This assumes a diagonal-in-ell C_l^{XY} weighting, as produced by generate_data.
        
        This is the inverse of the *beam-deconvolved* generate_data maps (which gives fastest convergence).
        
        Note that the code has two input options: "harmonic" or "map", to avoid unnecessary transforms.
        
        The output is always returned in harmonic-space.
        """
        if lmax==-1:
            lmax = self.lmax
    
        assert input_type in ['harmonic','map'], "Valid input types are 'harmonic' and 'map' only!"
        
        # Transform to harmonic space, if necessary
        if input_type=='map': input_map_lm = self.to_lm(input_map,lmax=lmax)
        else: input_map_lm = input_map.copy()
            
        # Divide by covariance and return
        lfilt = self.l_arr<=lmax
        output = self.beam_lm[:,lfilt]*np.einsum('ijk,jk->ik', self.inv_Cl_tot_lm_mat[:,:,lfilt], self.beam_lm[:,lfilt]*input_map_lm, order='C')
        
        return output