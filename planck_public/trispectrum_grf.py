### Compute the odd-parity CMB trispectrum of Planck 2018 data or FFP10 simulations
# Here, we compute the numerator from a single GRF realization

########################### IMPORTS ###########################
import os, sys, healpy, fitsio, time, numpy as np
sys.path.append('../')
import polybin as pb
from scipy.interpolate import InterpolatedUnivariateSpline
init = time.time()

if len(sys.argv)!=2:
    raise Exception("Simulation index not specified!")
sim_id = int(sys.argv[1])

########################### SETTINGS ###########################

# HEALPix settings
Nside = 256
lmax = 3*Nside-1

# Accuracy settings
N_it = 100 # Number of random iterations to compute 2- and 0-field terms

# Binning parameters
l_bins = np.load('models/l_bins510.npy')
l_bins_squeeze = l_bins.copy()
L_bins = l_bins.copy()
print("binned lmax: %d, HEALPix lmax: %d"%(np.max(l_bins_squeeze),lmax))

# Whether to include bins only partially satisfying triangle conditions
include_partial_triangles = False

# Whether to include the pixel window function
# This should be set to True, unless we generate maps at the same realization we analyze them!
include_pixel_window = True

# whether to add a separable reduced bispectrum to the input maps
include_synthetic_b = False

# I/O
root = '/projects/QUIJOTE/Oliver/planck_maps/'
outroot = '/projects/QUIJOTE/Oliver/planck_trispectrum_hi/'
datafile = 'COM_CMB_IQU-smica_2048_R3.00_full.fits' # Data map (from 1905.05697, 2018 SMICA map)
if sim_id!=-1:
    # Simulation map (SMICA FFP10 from PLA - CMB + noise realizations)
    simfile_cmb = 'ffp10/dx12_v3_smica_cmb_mc_%s_raw.fits'%str(sim_id).zfill(5)
    simfile_noise = 'ffp10/dx12_v3_smica_noise_mc_%s_raw.fits'%str(sim_id).zfill(5)

# Beam
l = np.arange(lmax+1)
beam_dat = fitsio.read(root+datafile,ext=2)['INT_BEAM']
beam_int = InterpolatedUnivariateSpline(np.arange(len(beam_dat)),beam_dat)
beam = beam_int(l)*(l>=2)+(l<2)*1

# Base class
Sl_weighting = np.load('planck/Sl_weighting256.npy')
assert len(Sl_weighting)==lmax+1
base = pb.PolyBin(Nside, Sl_weighting, beam=beam, include_pixel_window=include_pixel_window)

# Galactic Mask (from Planck SMICA common sky mask)
smooth_mask_file = 'smooth_mask%d.fits'%Nside
inpainting_mask_file = 'inpainting_mask%d.fits'%Nside

# Check if output exists
outfile = outroot+'trispectrum100_grf-numerator%d_(%d,%d,%d).txt'%(sim_id,len(l_bins)-1,len(l_bins_squeeze)-1,len(L_bins)-1)

if os.path.exists(outfile):
    print("Trispectrum numerator already computed; exiting!")
    sys.exit()

########################### LOAD DATA ###########################
print("Generating GRF")
Sl_grf = np.load('planck/Sl_weighting2048.npy')
# use high-resolution Sl to generate, before downgrading
# this ensures that FFP10 + GRFs match
data = healpy.ud_grade(healpy.synfast(Sl_grf, 2048), Nside)

########################### LOAD MASK ###########################

print("Loading mask")
smooth_mask = healpy.read_map(root+smooth_mask_file)
inpainting_mask = healpy.read_map(root+inpainting_mask_file)

########################### WEIGHTING ###########################

# Define S+N weighting, ensuring l<2 modes do not blow up
Cl_filt = InterpolatedUnivariateSpline(l, healpy.pixwin(Nside)**2*Sl_weighting)(base.l_arr)   
    
def inpaint_map(input_map):
    """
    Apply linear inpainting to a map, given an inpainting mask
    """
    
    tmp_map = input_map.copy()
    
    # Zero out inpainting regions 
    tmp_map[inpainting_mask==1] = 0 

    # Perform iterative impainting
    for i in range(1000):

        inpaint_pix = np.where((tmp_map==0)&(inpainting_mask==1))[0]
        if len(inpaint_pix)==0:
            break
        # Identify four nearest neighbors
        neighbors = healpy.get_interp_weights(Nside,inpaint_pix)[0]
        tmp_map[inpaint_pix] = np.mean(tmp_map[neighbors],axis=0)

    return tmp_map

def applySinv(input_map):
    """
    Apply the quasi-optimal weighting, S^{-1} to a map. This firstly inpaints small holes in the data, applies a smooth mask, then weights by an ell-dependent factor.
    
    Note that this is neither diagonal nor invertible. The weighting is given by Cl_lm = B_l^2 C_l^TT + N_l here for beam B_l.
    """
    ## Step 1: inpaint the data
    tmp_map = inpaint_map(input_map)
    
    ## Step 2: mask out the large bad regions
    tmp_map *= smooth_mask
        
    ## Step 3: Apply S+N weighting in harmonic space
    Cinv_map = base.to_map(base.safe_divide(base.to_lm(tmp_map),Cl_filt))
    
    return Cinv_map

########################### COMPUTE NUMERATOR ###########################

# Initialize trispectrum class
tspec = pb.TSpec(base, 1.+0.*smooth_mask, applySinv, l_bins, l_bins_squeeze=l_bins_squeeze, L_bins=L_bins)

### Read in pairs of MC simulations created externally for the two-field and zero-field terms
alpha_sims = []

print("Loading simulations")
load_time = time.time()
for jj in range(sim_id+1,sim_id+1+N_it,2):
    ii = jj%300
    iip1 = (jj+1)%300
    print("Loading simulations %d and %d of %d"%(ii,iip1,N_it))
    
    # Load first simulations
    simfile_cmb_ii = 'ffp10/dx12_v3_smica_cmb_mc_%s_raw.fits'%str(ii).zfill(5)
    simfile_noise_ii = 'ffp10/dx12_v3_smica_noise_mc_%s_raw.fits'%str(ii).zfill(5)
    cmb_ii = healpy.ud_grade(healpy.read_map(root+simfile_cmb_ii,field=0),Nside)
    noise_ii = healpy.ud_grade(healpy.read_map(root+simfile_noise_ii,field=0),Nside)
    sim1 = cmb_ii + noise_ii

    # Load second simulation
    simfile_cmb_ii = 'ffp10/dx12_v3_smica_cmb_mc_%s_raw.fits'%str(iip1).zfill(5)
    simfile_noise_ii = 'ffp10/dx12_v3_smica_noise_mc_%s_raw.fits'%str(iip1).zfill(5)
    cmb_ii = healpy.ud_grade(healpy.read_map(root+simfile_cmb_ii,field=0),Nside)
    noise_ii = healpy.ud_grade(healpy.read_map(root+simfile_noise_ii,field=0),Nside)
    sim2 = cmb_ii + noise_ii
    
    # Add to outputs
    alpha_sims.append([sim1,sim2])

# Load into PolyBin
tspec.load_sims(alpha_sims, verb=True)
print("Finished loading simulations in %.2f s"%(time.time()-load_time))

# Compute trispectrum numerator
print("Starting numerator computation")
start = time.time()
numerator = tspec.Tl_numerator(data, parity='both', include_disconnected_term=True, verb=True)
print("Computed trispectrum contribution after %.2f s"%(time.time()-start))

#np.save(outfile,numerator)
np.savetxt(outfile,np.concatenate(numerator))
print("Output saved to %s; exiting after %.2f seconds"%(outfile,time.time()-init))
