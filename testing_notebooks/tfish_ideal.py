## IMPORTS
import time, os, sys, healpy
import numpy as np
sys.path.append('../')
import polybin as pb
from scipy.interpolate import InterpolatedUnivariateSpline

include_synthetic_b = False

# Binning parameters
dl = 20 # width
Nl = 5 # number
min_l = 2 # minimum

# HEALPix settings
Nside = 128
lmax = 3*Nside-1

print("binned lmax: %d,"%(min_l+dl*Nl),"HEALPix lmax: %d"%lmax)

# Whether to include bins only partially satisfying triangle conditions
include_partial_triangles = False

# whether to add a separable reduced bispectrum to the input maps
b_input_fac = lambda l1: np.exp(-(l1-2)/40.)*2e-6

# Galactic Mask
# Using GAL040 mask with 2-degree apodization for testing
root = '/projects/QUIJOTE/Oliver/planck/'
maskfile = 'HFI_Mask_GalPlane-apo2_2048_R2.00.fits'

from classy import Class
cosmo = Class()

# Define ell arrays
l = np.arange(lmax+1)

# Run CLASS
print("Loading CLASS")
cosmo.set({'output':'tCl,lCl,mPk','l_max_scalars':lmax+1,'lensing':'yes',
           'omega_b':0.022383,
           'non linear':'no',
           'omega_cdm':0.12011,
           'h':0.6732,
           'm_ncdm':0.06,
           'N_ncdm':1,
           'tau_reio':0.0543,
           'A_s':1e-10*np.exp(3.0448),
           'n_s':0.96605});
cosmo.compute()

# Compute signal C_ell
Cl_dict = cosmo.lensed_cl(lmax);
Cl_th = Cl_dict['tt']*cosmo.T_cmb()**2

# Compute noise C_ell
DeltaT = 60./60.*np.pi/180.*1e-6 # in K-radians
thetaFWHM = 5./60.*np.pi/180. # in radians
Nl_th = DeltaT**2*np.exp(l*(l+1)*thetaFWHM**2/(8.*np.log(2)))*(l>2)

mask = healpy.ud_grade(healpy.read_map(root+maskfile,field=1),Nside)

# Load class with fiducial Cl and Nside
base = pb.PolyBin(Nside, Cl_th+Nl_th+Cl_th[2]*(l<2))

# Generate unmasked data with known C_l and factorized b
# Cl is set to the fiducial spectrum unless otherwise specified
# No beam is included
print("Generating data")
raw_data = base.generate_data(seed=42, add_B=include_synthetic_b, b_input=b_input_fac)

# Mask the map
data = raw_data*mask

def applySinv(input_map):
    """Apply the optimal weighting to a map. 
    
    Here, we assume that the forward covariance is diagonal, in particular C_l, and invert this.
    This is not quite the exact solution (as it incorrectly treats W(n) factors), but should be unbiased."""
    
    # Transform to harmonic space
    input_map_lm = base.to_lm(input_map)
    # Divide by covariance and return to map-space
    Cinv_map = base.to_map(base.safe_divide(input_map_lm,base.Cl_lm))
    
    return Cinv_map

# Initialize power spectrum class
tspec = pb.TSpec(base, mask, applySinv, min_l, dl, Nl)

# Create Fisher matrix
t_init = time.time()
ideal_fish = tspec.compute_fisher_ideal('both',True,32) 

# IDEAL FISHER
fish_out = '/projects/QUIJOTE/Oliver/polybin_testing/tfish_ideal_%d_%d_%d.npy'%(min_l,dl,Nl)
np.save(fish_out,ideal_fish)

print("Computed ideal Fisher after %d seconds"%(time.time()-t_init))
