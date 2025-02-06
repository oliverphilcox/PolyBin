# Code for ideal and unwindowed binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2025)
# This module includes code to weight general CMB maps.
import numpy as np, healpy

class Weightings():
    def __init__(self, base, mask, Cl_raw, noise_cov=None, inpainting_mask=None):
        """Class used to applying the S^-1 weight to a pixel-space map. This includes an idealized harmonic weighting and a full conjugate gradient descent solution.
        
        Inputs:
        - base: PolySpec base class.
        - mask: Pixel-space mask (optionally one for each spin). For ideal S^-1, this should be smoothed, but for optimal S^-1, a boolean mask is preferred.
        - Cl_raw: Dictionary containing fiducial power spectra (without the beam or noise)
        - noise_cov: Pixel-space noise covariance (optional, but required to compute the optimal weighting)
        - inpainting_mask: Boolean mask specifying which pixels to inpaint (optional, and used only in the idealized weighting) 
        """
        # Setup attributes
        self.base = base
        self.mask = mask
        self.Cl_raw = Cl_raw
        self.pol = self.base.pol
        self.pixel_area = 4.*np.pi/healpy.nside2npix(self.base.Nside)
        
        # Check noise covariance
        if noise_cov is not None:
            self.noise_cov = noise_cov
            # Define inverse noise covariance
            self.noise_icov = np.zeros_like(self.noise_cov)
            self.noise_icov[self.noise_cov!=0] = 1./self.noise_cov[self.noise_cov!=0]
            
        # Check inpainting mask
        if inpainting_mask is not None:
            assert np.mean(inpainting_mask)==np.mean(inpainting_mask**2), "Inpainting mask should be a boolean array!"
            self.inpainting_mask = inpainting_mask
            if np.mean(1.*self.inpainting_mask)>0.5:
                print("## %.2f%% of the pixels will be inpainted: is this mask correct?"%(np.mean(1.*self.inpainting_mask)*100))
            if self.pol:
                assert len(self.inpainting_mask)==3, "Must supply one mask for each polarization!"
            else:
                if len(self.inpainting_mask)!=1:
                    self.inpainting_mask = self.inpainting_mask.reshape(1,-1)
        
        # Define beam and signal
        self.beam_lm = self.base.beam_lm
        
        # Compute isotropic noise properties (accounting for mask)
        if hasattr(self, 'noise_cov'):
            self.Nl_iso = np.zeros(1+2*self.pol)
            for i in range(1+2*self.pol):
                self.Nl_iso[i] = np.mean(self.noise_cov[i][self.mask[i]>0.99])*self.pixel_area
            
            # Load raw power spectra and take inverse
            if self.pol:
                assert len(Cl_raw.keys())==6 or len(Cl_raw.keys())==4, "Must specify four or six input spectra: {ClTT, ClTE, ClEE, ClBB} or {ClTT, ClTE, ClTB, ClEE, ClEB, ClBB}"
                if len(Cl_raw.keys())==4:
                    Cl_raw['TB'] = Cl_raw['EB'] = 0.*Cl_raw['TT']
                self.Cl_lm_mat = np.moveaxis(np.asarray([[Cl_raw['TT'][self.base.l_arr],Cl_raw['TE'][self.base.l_arr],Cl_raw['TB'][self.base.l_arr]],
                                                        [Cl_raw['TE'][self.base.l_arr],Cl_raw['EE'][self.base.l_arr],Cl_raw['EB'][self.base.l_arr]],
                                                        [Cl_raw['TB'][self.base.l_arr],Cl_raw['EB'][self.base.l_arr],Cl_raw['BB'][self.base.l_arr]]]),[2,1,0],[0,2,1])
                
                # Compute inverse, being careful of B-modes
                if np.sum(Cl_raw['BB'])==0:
                    good_l = np.linalg.det(self.Cl_lm_mat[:,:2,:2])>0
                    self.inv_Cl_lm_mat = np.zeros((1+2*self.pol,1+2*self.pol,len(self.base.l_arr)),dtype=np.float64,order='C')
                    self.inv_Cl_lm_mat[:2,:2,good_l] = np.asarray(np.moveaxis(np.linalg.inv(self.Cl_lm_mat[good_l,:2,:2]),[0,1,2],[2,0,1]),order='C')
                else:
                    good_l = np.linalg.det(self.Cl_lm_mat)>0
                    self.inv_Cl_lm_mat = np.zeros((1+2*self.pol,1+2*self.pol,len(self.base.l_arr)),dtype=np.float64,order='C')
                    self.inv_Cl_lm_mat[:,:,good_l] = np.asarray(np.moveaxis(np.linalg.inv(self.Cl_lm_mat[good_l,:,:]),[0,1,2],[2,0,1]),order='C')
            else:
                self.Cl_lm_mat = Cl_raw['TT'][self.base.l_arr].reshape(-1,1,1)
                good_l = self.Cl_lm_mat[:,0,0]>0
                self.inv_Cl_lm_mat = np.zeros((1+2*self.pol,1+2*self.pol,len(self.base.l_arr)),dtype=np.float64,order='C')
                self.inv_Cl_lm_mat[:,:,good_l] = np.asarray(np.moveaxis(np.linalg.inv(self.Cl_lm_mat[good_l,:,:]),[0,1,2],[2,0,1]),order='C')
                
            # Compute idealized filtering (with non-zero values down to ell=0)
            inv_Slm = self.inv_Cl_lm_mat+np.einsum('il,jl,ij->ijl',self.beam_lm,self.beam_lm,np.diag(1./self.Nl_iso))
            inv_Slm[:,:,self.base.l_arr<2] = inv_Slm[:,:,[self.base.l_arr==2][0]]
            self.Sigmalm = np.asarray(np.moveaxis(np.linalg.inv(np.moveaxis(inv_Slm,[0,1,2],[1,2,0])),[0,1,2],[2,0,1]),order='C')
        
        # Check mask properties
        if not type(mask)==float or type(mask)==int:
            if len(mask)==1 or len(mask)==3:
                assert len(mask[0])==self.base.Npix, f'Mask has incorrect shape: {mask.shape}'
            else:
                assert len(mask)==self.base.Npix, f'Mask has incorrect shape: {mask.shape}'
    
    def inpaint_map(self, input_map, n_average=50):
        """
        Apply linear inpainting to a map. We replace each inpainted pixel with the average of its (non-zero) neighbors and iterate over unfilled pixels. 
        
        In the final step, we smooth the inpainted regions by replacing all inpainted pixels by the average of their neighbors, repeating this process n_average times.
        """
        assert hasattr(self, 'inpainting_mask'), "Must supply inpainting mask to use inpainting code!"
        
        # Starting map
        tmp_map = input_map.copy()
        
        # Check there's any inpainting to be done!
        inpainting_filt = (self.inpainting_mask==1)
        if np.sum(inpainting_filt)==0:
            return tmp_map

        # Iterate over fields
        for field in range(1+2*self.pol):
            this_map = tmp_map[field].copy()
            
            # Skip if the map is empty
            if np.sum(this_map)==0: continue 
            
            # Zero out inpainting regions 
            this_map[inpainting_filt[field]] = 0 
            inpaint_pix = np.where((tmp_map[field]==0)&inpainting_filt[field])[0]
            
            # Perform iterative impainting
            i = 0
            while True:
                
                # Identify nearest neighbors
                neighbors = healpy.get_all_neighbours(self.base.Nside, inpaint_pix)
                
                # Fill with mean of non-zero neighbors
                this_map[inpaint_pix] = np.average(this_map[neighbors],weights=this_map[neighbors]!=0+1e-30,axis=0)

                # Define next set of inpainting pixels
                next_inpaint = np.where((this_map==0)&inpainting_filt[field])[0]
                if len(next_inpaint)==0:
                    break
                if len(next_inpaint)==len(inpaint_pix):
                    print("# Inpainting field %d failed!"%field)
                    break
                
                inpaint_pix = next_inpaint
                i += 1
            
            # Iterate to smooth the map
            for i in range(n_average):
                inpaint_pix = np.where(inpainting_filt[field])[0]
                
                # Identify nearest neighbors
                neighbors = healpy.get_all_neighbours(self.base.Nside, inpaint_pix)
                
                # Fill with mean of non-zero neighbors
                this_map[inpaint_pix] = np.average(this_map[neighbors],axis=0)
            tmp_map[field] = this_map

        return tmp_map
        
    def applySinv_optimal(self, input_map, preconditioner='pseudo_inverse', nstep=25, thresh=1e-4, verb=False, input_type='map', lmax=None):
        """
        Apply the optimal S^-1 filtering to a map using conjugate gradient descent (CGD) methods.
        We start the iteration from the S^-1_ideal solution for efficiency. The output is returned in harmonic-space.
        
        Main Inputs:
        - input_map: Data vector to apply S^-1 to. This should be in map- or harmonic-space depending on "input_type"
        - nstep: Number of CGD steps to run.
        - thresh: Threshold for terminating the CGD algorithm.
        - preconditioner: "pseudo_inverse" or "harmonic". Choice of CGD preconditioner.
        """
        
        # Check inputs
        assert input_type=='map', "Optimal weighting should be applied only to pixel-space data!"
        if not hasattr(self, 'noise_cov'):
            raise Exception("The CGD solver requires a pixel-space noise covariance")
        if lmax==None:
            lmax = self.base.lmax
                    
       # Apply inpainting, if specified
        if hasattr(self, 'inpainting_mask'):
            tmp_map = self.inpaint_map(input_map.copy())
        else:
            tmp_map = input_map.copy()
        
        # Compute approximate solution ( = C.S^-1_{ideal} since we compute the Wiener-filtered map)
        x0 = np.einsum('iab,bi,bci,ci->ai',self.Cl_lm_mat, self.beam_lm, self.base.inv_Cl_tot_lm_mat, self.base.to_lm(tmp_map))

        # Initialize solver
        cgd = CGD(self, tmp_map, x0=x0, preconditioner=preconditioner, verb=verb)
        
        # Run solver
        cgd.run(nstep=nstep, thresh=thresh)
        
        # Return output, truncating at lmax
        lfilt = (self.base.l_arr<=lmax)
        return cgd.get_ivar_lm()[:,lfilt]
    
    def applySinv_ideal(self, input_map, input_type='map', lmax=None):
        """
        Apply the idealized S^-1 filtering. This inpaints small holes (if an inpainting mask has been supplied), then divides by the fiducial signal-plus-noise in harmonic-space, and finally multiplies by the beam.
        The output is returned in harmonic-space.
        """
        
        # Check inputs
        if lmax==None:
            lmax = self.base.lmax
        
        # Apply weighting in harmonic-space
        if input_type=='harmonic':
            assert np.mean(self.mask)==1, "Harmonic-space S^-1 requires a unit mask!"
            lfilt = self.base.l_arr<=lmax
            return self.beam_lm[:,lfilt]*np.einsum("abl,bl->al",self.base.inv_Cl_tot_lm_mat[:,:,lfilt], input_map)
        elif input_type=='map':
                
            # Apply inpainting, if specified
            if hasattr(self, 'inpainting_mask'):
                tmp_map = self.inpaint_map(input_map.copy())
            else:
                tmp_map = input_map.copy()
                
            # Transform to harmonic-space
            input_lm = self.base.to_lm(tmp_map)
                
            # Compute output map, truncating at lmax
            lfilt = (self.base.l_arr<=lmax)
            return self.beam_lm[:,lfilt]*np.einsum("abl,bl->al",self.base.inv_Cl_tot_lm_mat[:,:,lfilt], input_lm[:,lfilt])

### CGD Code    
class CGD():
    def __init__(self, weightings, d, x0=[], test_b=[], preconditioner='pseudo_inverse', verb=False):
        """Initialize the CGD iterator, which solves Ax=b[d] for x, i.e. x = [PCP^dag+N]^-1 d.
        
        This draws heavily on the pixell and optweight implementations.
        
        Inputs:
        - weightings: Weightings class, containing key attributes.
        - d: pixel-space data we wish to apply [PCP^dag+N]^-1 to.
        - x0: Initial guess at a harmonic-space solution. This should approximate the Wiener-filtered map d_WF.
        - test_b: Input b_lm used to test the estimator. If we set b = Ax for known x, the output should match x.
        - preconditioner: Choice of CGD preconditioner. Options include 'harmonic' and 'pseudo_inverse'.
        - verb: Verbosity for the convergence messages.
        """
        
        # Load attributes
        self.preconditioner = preconditioner
        self.base = weightings.base
        self.weightings = weightings
        self.verb = verb
        
        # Initialize CGD parameters
        if len(test_b)==0:
            b0 = self._get_b(d.copy())
        else:
            print("Running in test mode!")
            assert len(d)==0, "Cannot specify both d and test_b!"
            b0 = test_b.copy()
        if len(x0)!=0:
            self.x0 = x0
            b = b0-self._apply_A(self.x0)
        else:
            self.x0 = 0.
            b = b0
        self.x = np.zeros_like(b)
        self.r = b.copy()
        
        # Define dot product
        self.dot = lambda a, b: np.sum(a*np.conj(b)*(1.+(self.base.m_arr>0))).real
        
        # Internal work variables
        n = b.size
        z = self._apply_M(self.r)
        self.rz  = self.dot(self.r, z)
        self.rz0 = float(self.dot(b0, self._apply_M(b0)))
        self.p   = z
        self.i   = 0
        self.err = self.rz/self.rz0
        
    # FILTERING
    def _apply_beam(self, ilm):
        """Apply the beam in harmonic-space"""
        return ilm*self.weightings.beam_lm
        
    def _apply_mask(self, imap):
        """Apply W in pixel-space"""
        return imap*self.weightings.mask

    def _apply_Ninv(self, imap):
        """Apply N^-1 in pixel-space"""
        return imap*self.weightings.noise_icov/self.weightings.pixel_area

    def _apply_Cinv(self, ilm):
        """Apply C^-1 in harmonic-space"""
        return np.einsum('abi,bi->ai', self.weightings.inv_Cl_lm_mat, ilm)
        
    # FULL PROJECTION
    def _pointing(self, ilm):
        """Apply the pointing matrix: P = W Y B"""
        return self._apply_mask(self.base.to_map(self._apply_beam(ilm)))

    def _pointing_adjoint(self, imap):
        """Apply the adjoint of the pointing matrix: P = B^dag Y^dag W^dag"""
        return self._apply_beam(self.base.to_lm(self._apply_mask(imap)))

    def _get_b(self, d):
        """Compute b = P^dag N^-1 d"""
        return self._pointing_adjoint(self._apply_Ninv(d))
        
    # MAIN FUNCTIONS
    def _apply_A(self, ilm):
        """Apply the A matrix: A = C^-1 + P^dag N^-1 P"""
        return self._apply_Cinv(ilm)+self._pointing_adjoint(self._apply_Ninv(self._pointing(ilm)))
        
    def _apply_M(self, ilm):
        """Apply the preconditioner matrix."""
        
        if self.preconditioner=='harmonic':
            return np.einsum('abi,bi->ai',self.weightings.Sigmalm, ilm)
            
        elif self.preconditioner=='pseudo_inverse':

            # Weighting
            alm = np.einsum('abi,bi->ai',self.weightings.Sigmalm, ilm)
            
            # Signal: C^-1 a
            alm_signal = np.einsum('abi,bi->ai',self.weightings.inv_Cl_lm_mat, alm)

            # Noise: B.N_iso^-1.N.N_iso^-1.B^dagger
            alm_noise = alm*self.weightings.beam_lm/self.weightings.Nl_iso[:,np.newaxis]
            alm_noise = self.base.to_lm(self.weightings.noise_cov*self.base.to_map(alm_noise))*self.weightings.pixel_area
            alm_noise *= self.weightings.beam_lm/self.weightings.Nl_iso[:,np.newaxis]

            # Combination
            return np.einsum('abi,bi->ai',self.weightings.Sigmalm, alm_noise+alm_signal)
            
        else:
            raise Exception("Unknown preconditioner %s specified!"%self.preconditioner)    
            
    def step(self):
        """Take a single step in the CGD iteration. Results in .x, .i and .err being updated."""
        
        Ap = self._apply_A(self.p)
        alpha = self.rz/self.dot(self.p, Ap)
        self.x += alpha*self.p
        self.r -= alpha*Ap
        del Ap
        z       = self._apply_M(self.r)
        next_rz = self.dot(self.r, z)
        self.err = next_rz/self.rz0
        beta = next_rz/self.rz
        self.rz = next_rz
        self.p  = z + beta*self.p
        self.i += 1

    def run(self, nstep=10, thresh=1e-6):
        """Run the CGD algorithm for a given number of steps"""
        for i in range(nstep):
            self.step()
            if self.verb: print("CGD step %d: error = %.2e"%(self.i,self.err))
            if self.err<thresh:
                if self.verb: print("CGD terminating at step %d"%self.i)
                break
        if i==nstep-1 and self.err>thresh:
            print("CGD did not converge to %.2e threshold after %d steps. Final error = %.2e"%(thresh, nstep, self.err))
    
    def get_wiener_lm(self):
        """Return the Wiener-filtered map in harmonic-space"""
        return self.x+self.x0
        
    def get_wiener_map(self):
        """Return the Wiener-filtered map in pixel-space"""
        return self.base.to_map(self.x+self.x0)

    def get_ivar_lm(self):
        """Return the inverse-variance-filtered map in harmonic-space"""
        return self._apply_Cinv(self.x+self.x0)

    def get_ivar_map(self):
        """Return the inverse-variance-filtered map in pixel-space"""
        return self.base.to_map(self._apply_Cinv(self.x+self.x0))
    