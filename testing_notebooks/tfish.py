## IMPORTS
import time, os, sys, healpy
import numpy as np
sys.path.append('../')
import polybin as pb
from scipy.interpolate import InterpolatedUnivariateSpline

if len(sys.argv)!=3:
    raise Exception("Need to specify fisher index and option!")
fish_index = int(sys.argv[1])
option = int(sys.argv[2])

# HEALPix settings
Nside = 128
lmax = 3*Nside-1

### Option 1:
if option==1:
    # unwindowed
    use_mask = False
elif option==2:
    # windowed
    use_mask = True

include_synthetic_b = False

# Binning parameters
dl = 20 # width
Nl = 6 # number
min_l = 2 # minimum
print("binned lmax: %d,"%(min_l+dl*Nl),"HEALPix lmax: %d"%lmax)

# Number of random iterations to create Fisher matrix
N_it = 100

# Number of simulations to use for testing
N_sim = 1000

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

if not use_mask:
    mask = 1.+0.*mask

# Load class with fiducial Cl and Nside
base = pb.PolyBin(Nside, Cl_th+Nl_th)

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

## Compute simulation
raw_sim = base.generate_data(fish_index,add_B=include_synthetic_b,b_input=b_input_fac)
sim = mask*raw_sim

# FISHER
fish_out = '/projects/QUIJOTE/Oliver/polybin_testing/tfish%d_%d.npy'%(option,fish_index)
if not os.path.exists(fish_out) and fish_index < N_it:
    init = time.time()
    this_fish = tspec.compute_fisher_contribution(fish_index, parity='both',verb=True)
    print("Fisher matrix %d computed in %.2f s"%(fish_index,time.time()-init))

    np.save(fish_out,this_fish)

# IDEAL NUMERATOR
ideal_out = '/projects/QUIJOTE/Oliver/polybin_testing/tnum%d_ideal%d.npy'%(option,fish_index)
if not os.path.exists(ideal_out) and fish_index < N_sim:
    
    init = time.time()
    num_ideal = tspec.Tl_numerator_ideal(sim,parity='both')
    print("Ideal numerator %d computed in %.2f s"%(fish_index,time.time()-init))
    
    np.save(ideal_out,num_ideal)

optimal_out = '/projects/QUIJOTE/Oliver/polybin_testing/tnum%d_opt%d.npy'%(option,fish_index)
optimal_out02 = '/projects/QUIJOTE/Oliver/polybin_testing/tnum%d_opt%d_no02.npy'%(option,fish_index)

if fish_index < N_sim:
    if not os.path.exists(optimal_out) or not os.path.exists(optimal_out02):
        # Generate debiasing simultions
        tspec.generate_sims(N_it, verb=True)

# OPTIMAL NUMERATOR
if not os.path.exists(optimal_out) and fish_index < N_sim:
    
    init = time.time()
    num = tspec.Tl_numerator(sim, parity='both',verb=True)
    print("Optimal numerator %d computed in %.2f s"%(fish_index,time.time()-init))
    
    np.save(optimal_out,num)

# OPTIMAL NUMERATOR (w/o disconnected subtraction)
if not os.path.exists(optimal_out02) and fish_index < N_sim:
    init = time.time()
    num02 = tspec.Tl_numerator(sim, parity='both',include_disconnected_term=False, verb=True)
    print("Optimal numerator %d (w/o 0+2-field) computed in %.2f s"%(fish_index,time.time()-init))
    
    np.save(optimal_out02,num02)


