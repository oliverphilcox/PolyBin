### Code for ideal and unwindowed binned polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022)
## This module contains the basic code

import numpy as np
import multiprocessing as mp
import tqdm, healpy
from scipy.interpolate import InterpolatedUnivariateSpline
import pywigxjpf as wig

class PolyBin():
    """Base class for PolyBin.
    
    Inputs:
    - Nside: HEALPix Nside
    - Cl: Fiducial power spectra (including beam and noise). This is used for creating synthetic maps for the optimal estimators, and, optionally, generating GRFs to test on.
    - beam: Beam present in the signal maps. This will be set to unity if unspecified, else deconvolved out the signal. If polarization is set, this contains two components: (Temperature-Beam, Polarization-Beam)
    - include_pixel_window: Whether to account for the HEALPix pixel window function (usually true, unless data is generated at low Nside).
    - pol: Whether to include spin-2 fields in the computations. If true, Cl should be a list of six spectra: [ClTT, ClTE, ClTB, ClEE, ClEB, ClBB]. If false, Cl contains only ClTT.
    - backend: Which backend to use to compute spherical harmonic transforms. Options: "healpix" [requires healpy] or "libsharp" [requires pixell].   
    """
    def __init__(self, Nside, Cl, beam=[], include_pixel_window=True, pol=True, backend='libsharp'):
        
        # Load attributes
        self.Nside = Nside
        self.pol = pol
        self.backend = backend

        # Load in fiducial spectra
        if self.pol:
            assert len(Cl.keys())==6, "Must specify six input spectra: {ClTT, ClTE, ClTB, ClEE, ClEB, ClBB}"
            self.Cl = [Cl['TT'],Cl['TE'],Cl['TB'],Cl['EE'],Cl['EB'],Cl['BB']]
            self.n_Cl = len(self.Cl)
        else:
            self.Cl = [Cl['TT']]
            self.n_Cl = 1
        if len(beam)==0:
            self.beam = [1.+0.*self.Cl[0] for _ in range(1+2*self.pol)]
        else:
            if self.pol:
                assert len(beam)==2, "Beam must contain temperature and polarization components"
                self.beam = [beam[0],beam[1],beam[1]]
            else:
                assert (len(beam)==1 or len(beam)==len(Cl['TT'])), "Beam must contain the same ells as Cl"
                self.beam = np.asarray(beam).reshape(1,-1)
                
        # Define indices for T, E, B and their parities
        self.indices = {'T':0,'E':1,'B':2}
        self.parities = {'T':1,'E':1,'B':-1}

        # Account for pixel window if necessary
        if include_pixel_window:
            if not self.pol:
                pixwin = healpy.pixwin(self.Nside, pol=False)
                self.beam[0] *= pixwin
                self.Cl[0] *= pixwin**2
            else:
                pixwinT, pixwinP = healpy.pixwin(self.Nside, pol=True)
                pixwinP[:2] = 1. # avoid zero errors
                self.beam[0] *= pixwinT
                self.beam[1] *= pixwinP
                self.beam[2] *= pixwinP
                self.Cl[0] *= pixwinT**2 # TT
                self.Cl[1] *= pixwinT*pixwinP # TE
                self.Cl[2] *= pixwinT*pixwinP # TB
                self.Cl[3] *= pixwinP**2 # EE
                self.Cl[4] *= pixwinP**2 # EB
                self.Cl[5] *= pixwinP**2 # BB
        else:
            print("## Caution: not accounting for pixel window function")
        
        # Derived parameters
        self.Npix = 12*Nside**2
        self.A_pix = 4.*np.pi/self.Npix
        self.lmax = 3*self.Nside-1
        self.l_arr,self.m_arr = healpy.Alm.getlm(self.lmax)

        # Define m weights (for complex conjugates)
        self.m_weight = (1.+1.*(self.m_arr>0.))
        
        # Set up relevant SHT modules
        if self.backend=='libsharp':
            from pixell import sharp
            map_info = sharp.map_info_healpix(self.Nside)
            alm_info = sharp.alm_info(self.lmax)
            self.sht_func = sharp.sht(map_info,alm_info)
        elif self.backend=='healpix':
            pass
        else:
            raise Exception("Only 'healpix' and 'libsharp' backends are currently implemented!")
        
        # Apply Cl and beam to grid (including beam and noise)
        ls = np.arange(self.lmax+1)
        self.Cl_lm = [InterpolatedUnivariateSpline(ls, self.Cl[i])(self.l_arr) for i in range(self.n_Cl)]
        self.beam_lm = [InterpolatedUnivariateSpline(ls, self.beam[i])(self.l_arr) for i in range(1+2*self.pol)]

        for i in [0,3,5]:
            if not self.pol and i>0: continue
            if (self.Cl[i]==0).sum()>0:
                print("## Caution: Zeros detected in (auto) input Cl - this may cause problems for inversion!")
        if (self.beam[0]==0).sum()>0:
            print("## Caution: Zeros detected in input beam - this may cause problems for inversion!")
        if self.pol:
            if (self.beam[1]==0).sum()>0:
                print("## Caution: Zeros detected in input beam - this may cause problems for inversion!")
        for i in range(self.lmax+1):
            for f in range(self.n_Cl):
                if self.Cl[f][i]==0: self.Cl_lm[f][self.l_arr==i] = 0.
            for f in range(1+2*self.pol):
                if self.beam[f][i]==0: self.beam_lm[f][self.l_arr==i] = 0.
                    
        # Define C^-1 matrix
        if self.pol:
            # Compute full matrix of C^XY_lm and C^XY_l
            Cl_lm_mat = np.moveaxis(np.asarray([[self.Cl_lm[0],self.Cl_lm[1],self.Cl_lm[2]],
                                                [self.Cl_lm[1],self.Cl_lm[3],self.Cl_lm[4]],
                                                [self.Cl_lm[2],self.Cl_lm[4],self.Cl_lm[5]]]),[2,1,0],[0,2,1])
            Cl_mat = np.moveaxis(np.asarray([[self.Cl[0],self.Cl[1],self.Cl[2]],
                                                [self.Cl[1],self.Cl[3],self.Cl[4]],
                                                [self.Cl[2],self.Cl[4],self.Cl[5]]]),[2,1,0],[0,2,1])
            
            # Check that matrix is well-posed 
            assert (np.linalg.det(Cl_lm_mat)>0).all(), "Determinant of Cl^{XY} matrix is <= 0; are the input power spectra set correctly?"
            
            # Invert matrix for each l,m
            self.inv_Cl_lm_mat = np.moveaxis(np.linalg.inv(Cl_lm_mat),[0,1,2],[2,0,1])
            self.inv_Cl_mat = np.moveaxis(np.linalg.inv(Cl_mat),[0,1,2],[2,0,1])
        else:
            # Create trivial arrays for single field
            self.inv_Cl_lm_mat = (1./self.Cl_lm[0]).reshape(1,1,-1)
            self.inv_Cl_mat = (1./self.Cl[0]).reshape(1,1,-1)
                    
        # Define 3j calculation
        wig.wig_table_init(self.lmax*2,9)
        wig.wig_temp_init(self.lmax*2)
        self.tj0 = lambda l1,l2,l3: wig.wig3jj(2*l1,2*l2,2*l3,0,0,0)
        self.tj_sym = lambda l1,l2,l3: (wig.wig3jj(2*l1,2*l2,2*l3,-2,-2,4)+wig.wig3jj(2*l1,2*l2,2*l3,4,-2,-2)+wig.wig3jj(2*l1,2*l2,2*l3,-2,4,-2))/3.

        # Counter for SHTs
        self.n_SHTs_forward = 0
        self.n_SHTs_reverse = 0

    # Basic harmonic transform functions
    def to_lm(self, input_map):
        """Convert from map-space to harmonic-space. If three fields are supplied, this transforms TQU -> TEB.
        
        This uses either HEALPix or Libsharp, depending on the backend chosen."""
        self.n_SHTs_forward += 1+(len(input_map)==3)
        if self.backend=='healpix':
            if len(input_map)==3:
                return healpy.map2alm(input_map, pol=True, iter=0) # no iteration, to match Libsharp
            else:
                return healpy.map2alm(input_map[0], pol=False, iter=0).reshape(1,-1)
        elif self.backend=='libsharp':
            if len(input_map)==3:
                T_lm = self.sht_func.map2alm([input_map[0]])
                E_lm,B_lm = self.sht_func.map2alm(input_map[1:],spin=2)
                return np.asarray([T_lm.ravel(),E_lm,B_lm])
            else:
                return self.sht_func.map2alm(input_map[0]).reshape(1,-1)
            
    def to_map(self, input_lm):
        """Convert from harmonic-space to map-space. If three fields are supplied, this transforms TEB -> TQU.
        
        This uses either HEALPix or Libsharp, depending on the backend chosen."""
        self.n_SHTs_reverse += 1+(len(input_lm)==3)
        if self.backend=='healpix':
            if len(input_lm)==3:
                return healpy.alm2map(input_lm, self.Nside, pol=True)
            else:
                return healpy.alm2map(input_lm[0], self.Nside, pol=False).reshape(1,-1)
        elif self.backend=='libsharp':
            if len(input_lm)==3:
                T = self.sht_func.alm2map([input_lm[0]])
                Q,U = self.sht_func.alm2map(input_lm[1:],spin=2)
                return np.asarray([T.ravel(),Q,U])
            else:
                return self.sht_func.alm2map(input_lm[0]).reshape(1,-1)
            
    def to_lm_spin(self, input_map_plus, input_map_minus, spin):
        """Convert (+-s)A from map-space to harmonic-space, weighting by (+-s)Y_{lm}. Our convention definitions follow HEALPix.
        
        The inputs are [(+s)M(n), (-s)M(n)] and the outputs are [(+s)M_lm, (-s)M_lm]
        """
        assert spin>=1, "Spin must be positive!"
        assert type(spin)==int, "Spin must be an integer!"
        self.n_SHTs_forward += 1
        
        # Define inputs
        map_inputs = [np.real((input_map_plus+input_map_minus)/2.), np.real((input_map_plus-input_map_minus)/(2.0j))]

        # Perform transformation
        if self.backend=='healpix':
            lm_outputs = healpy.map2alm_spin(map_inputs,spin,self.lmax)
        elif self.backend=='libsharp':
            lm_outputs = self.sht_func.map2alm(map_inputs,spin=spin)
        
        # Reconstruct output
        lm_plus = -(lm_outputs[0]+1.0j*lm_outputs[1])
        lm_minus = -1*(-1)**spin*(lm_outputs[0]-1.0j*lm_outputs[1])
        
        return np.asarray([lm_plus, lm_minus])
    
    def to_map_spin(self, input_lm_plus, input_lm_minus, spin):
        """Convert (+-s)A_lm from harmonic-space to map-space, weighting by (+-s)Y_{lm}. Our convention definitions follow HEALPix.
        
        The inputs are [(+s)M_lm, (-s)M_lm] and the outputs are [(+s)M(n), (-s)M(n)]
        """
        assert spin>=1, "Spin must be positive!"
        assert type(spin)==int, "Spin must be an integer!"
        
        self.n_SHTs_reverse += 1
        
        # Define inputs
        lm_inputs = [-(input_lm_plus+(-1)**spin*input_lm_minus)/2.,-(input_lm_plus-(-1)**spin*input_lm_minus)/(2.0j)]
        
        # Perform transformation
        if self.backend=='healpix':
            map_outputs = healpy.alm2map_spin(lm_inputs,self.Nside,spin,self.lmax)
        elif self.backend=='libsharp':
            map_outputs = self.sht_func.alm2map(lm_inputs,spin=spin)
        
        # Reconstruct output
        map_plus = map_outputs[0]+1.0j*map_outputs[1]
        map_minus = map_outputs[0]-1.0j*map_outputs[1]
        
        return map_plus, map_minus

    def compute_spin_transform_map(self, a_lms, spin):
        """
        Compute Sum_lm {}_sY_lm a_lm for a given spin s and a set of *scalar* maps a_lm. This calls the relevant SHT functions.
        """
        if spin>0:
            return [self.to_map_spin(a_lm,(-1.)**spin*a_lm,int(abs(spin)))[0] for a_lm in a_lms]
        if spin<0:
            return [self.to_map_spin((-1.)**spin*a_lm,a_lm,int(abs(spin)))[1] for a_lm in a_lms]
    
    def safe_divide(self, x, y):
        """Function to divide maps without zero errors."""
        out = np.zeros_like(x)
        out[y!=0] = x[y!=0]/y[y!=0]
        return out

    def generate_data(self, seed=None, Cl_input=[], output_type='map', b_input=None, add_B=False, remove_mean=True, sum_ells='even'):
        """
        Generate a full-sky map with a given set of C_ell^{XY} and (optionally) b_l1l2l3. 
        The input Cl are expected to by in the form {ClTT, ClTE, ClTB, ClEE, ClEB, ClBB} if polarized, else ClTT.

        We use the method of Smith & Zaldarriaga 2006, and assume that b_l1l2l3 is separable into three identical pieces.

        We optionally subtract off the mean of the map (numerically, but could be done analytically), since it is not guaranteed to be zero if we include a synthetic bispectrum.

        Finally, we can additionally choose if l1+l2+l3 is restricted to be "even" or "odd" (or "both").

        No mask is added at this stage, and the output can be in map- or harmonic-space.
        """
        assert output_type in ['harmonic','map'], "Valid output types are 'harmonic' and 'map' only!"
        
        # Define seed
        if seed!=None:
            np.random.seed(seed)
            
        # Define input power spectrum
        if len(Cl_input)==0:
            Cl_input = self.Cl
        if self.pol:
            assert len(Cl_input)==6, "Need to specify {ClTT, ClTE, ClTB, ClEE, ClEB, ClBB}"
        else:
            if len(Cl_input)!=1: Cl_input = [Cl_input]
        
        # Generate a_lm maps (either T or T,E,B)
        initial_lm = healpy.synalm(Cl_input,self.lmax, new=False)
        
        if not add_B:
            if output_type=='map': return self.to_map(initial_lm)
            else: return initial_lm
        
        # Interpolate Cl to all l values
        ls = np.arange(self.lmax+1)
        Cl_input_lm = [InterpolatedUnivariateSpline(ls, Cl_input[i])(self.l_arr) for i in range(self.n_Cl)]
        for i in range(self.lmax+1):
            for f in range(self.n_Cl):
                if Cl_input[f][i]==0: Cl_input_lm[f][self.l_arr==i] = 0.
        
        # Compute gradient map
        Cinv_lm = np.einsum('ijk,jk->ik',self.inv_Cl_lm_mat,initial_lm,order='C')
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
        output_lm = [initial_lm[i] + 1./3.*grad_lm[i] for i in range(len(Cinv_lm))]
        output_map = self.to_map(output_lm)

        # Optionally remove mean of map
        if remove_mean:
            
            # Compute mean of synthetic maps numerically
            if not hasattr(self, 'map_offset') or (hasattr(self,'map_offset') and self.saved_spec!=[Cl_input,b_input]):
                print("Computing offset for synthetic maps")
                map_offset = 0.
                for ii in range(100):
                    map_offset += self.generate_data(Cl_input=Cl_input, b_input=b_input, seed=int(1e6)+ii,add_B=True,remove_mean=False)/100.
                self.map_offset = map_offset
                self.saved_spec = [Cl_input, b_input]
            # Remove mean
            output_map -= self.map_offset
        
        return np.asarray(output_map)
    
    def applyAinv(self, input_map, input_type='map', output_type='map'):
        """Apply the exact inverse weighting A^{-1} to a map. This assumes a diagonal-in-ell C_l^{XY} weighting, as produced by generate_data.
        
        Note that the code has two input and output options: "harmonic" or "map", to avoid unnecessary transforms.
        """
    
        assert input_type in ['harmonic','map'], "Valid input types are 'harmonic' and 'map' only!"
        assert output_type in ['harmonic','map'], "Valid output types are 'harmonic' and 'map' only!"

        # Transform to harmonic space, if necessary
        if input_type=='map': input_map_lm = self.to_lm(input_map)
        else: input_map_lm = input_map.copy()
            
        # Divide by covariance
        output = np.einsum('ijk,jk->ik',self.inv_Cl_lm_mat,input_map_lm,order='C')
        
        # Optionally return to map-space
        if output_type=='map': return self.to_map(output)
        else: return output