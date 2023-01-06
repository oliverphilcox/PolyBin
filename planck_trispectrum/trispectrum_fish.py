### Compute the odd-parity CMB trispectrum of Planck 2018 data or FFP10 simulations
# Here, we compute the contribution to the Fisher matrix from a single realization

########################### IMPORTS ###########################
import os, sys, healpy, fitsio, time, numpy as np
sys.path.append('../')
import polybin as pb
from scipy.interpolate import InterpolatedUnivariateSpline
init = time.time()

if len(sys.argv)!=2:
    raise Exception("Fisher index not specified!")
index = int(sys.argv[1])

########################### SETTINGS ###########################

# HEALPix settings
Nside = 128
lmax = 3*Nside-1

# Binning parameters
l_bins = np.load('l_bins_data.npy')
l_bins_squeeze = l_bins.copy()
L_bins = l_bins.copy()
print("binned lmax: %d, HEALPix lmax: %d"%(np.max(l_bins_squeeze),lmax))

# Whether to include bins only partially satisfying triangle conditions
include_partial_triangles = False

# whether to add a separable reduced bispectrum to the input maps
include_synthetic_b = False

# I/O
root = '/projects/QUIJOTE/Oliver/planck_maps/'
outroot = '/projects/QUIJOTE/Oliver/planck_trispectrum/'
datafile = 'COM_CMB_IQU-smica_2048_R3.00_full.fits'  # Data map (from 1905.05697, 2018 SMICA map)

# Beam
l = np.arange(lmax+1)
beam_dat = fitsio.read(root+datafile,ext=2)['INT_BEAM']
beam_int = InterpolatedUnivariateSpline(np.arange(len(beam_dat)),beam_dat)
beam = beam_int(l)*(l>=2)+(l<2)*1

# Base class
Sl_weighting = np.load('planck/Sl_weighting.npy')
assert len(Sl_weighting)==lmax+1
base = pb.PolyBin(Nside, Sl_weighting, beam=beam)

# Galactic Mask (from 1905.05697, common T map, fsky = 77.9%)
maskfile = 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits'
mask_fwhm = 10. # smoothing in arcminutes

# Check if output exists
outfile = outroot+'trispectrum_fisher%d_(%d,%d,%d).npy'%(index,len(l_bins)-1,len(l_bins_squeeze)-1,len(L_bins)-1)

if os.path.exists(outfile):
    print("Fisher matrix already computed; exiting!")
    sys.exit()

########################### LOAD MASK ###########################
print("Loading mask")
mask = healpy.ud_grade(healpy.read_map(root+maskfile,field=0),Nside)
# Convert to binary mask
mask[mask!=1] = 0

### Divide mask into small and large holes
zero_pix = np.where(mask==0)[0] # Look at each zero point
neighbors = healpy.get_interp_weights(Nside,zero_pix,phi=None)[0] # Identify neighbors
neighbor_val = mask[neighbors] # Check if neighbors are zero
neighbors[neighbor_val!=0] = -1 # Find list of particles with 0 neighbors
cluster_id = np.arange(len(zero_pix)) # Assign ID to each point

# Iterate over points (this is expensive)
for i in range(len(zero_pix)):
    x = np.where(neighbors==zero_pix[i])[1]
    if len(x)==0: continue
    cluster_id[x] = min(cluster_id[x])

# Count cluster sizes + identify small clusters
cluster_count = np.bincount(cluster_id,minlength=len(cluster_id))
cluster_size = cluster_count[cluster_id]
small_clusters = (cluster_size>0)&(cluster_size<20*(Nside/128)**2)
    
# Create inpainting mask
inpainting_mask = 0.*mask
inpainting_mask[zero_pix[small_clusters]]=1

# Define smooth mask
smooth_mask = healpy.smoothing(mask+inpainting_mask,mask_fwhm/60.*np.pi/180.)

########################### WEIGHTING ###########################

# Define S+N weighting, ensuring l<2 modes do not blow up
Cl_filt = InterpolatedUnivariateSpline(l, Sl_weighting)(base.l_arr)   
    
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

########################### COMPUTE FISHER ###########################

# Initialize trispectrum class

tspec = pb.TSpec(base, 1.+0.*mask, applySinv, l_bins, l_bins_squeeze=l_bins_squeeze, L_bins=L_bins)

# Compute Fisher contribution
print("Starting Fisher matrix computation")
start = time.time()
fish = tspec.compute_fisher_contribution(index,'both',verb=True)
print("Computed Fisher matrix contribution after %.2f s"%(time.time()-start))

np.save(outfile,fish)
print("Output saved to %s; exiting after %.2f seconds"%(outfile,time.time()-init))
