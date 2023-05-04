### Compute the odd-parity CMB trispectrum of Planck 2018 T- or E-mode data or FFP10 simulations
# Here, we compute the numerator from a single realization

########################### IMPORTS ###########################
import os, sys, healpy, fitsio, time, numpy as np
sys.path.append('/mnt/home/ophilcox/PolyBin/')
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

fields = ['TTTT','TTTE','TTEE','TETE','TEEE','EEEE']
parity = 'odd'

# Accuracy settings
N_it = 100 # Number of random iterations to compute 2- and 0-field terms

# Binning parameters
l_bins = np.load('/mnt/home/ophilcox/PolyBin/planck_public/l_bins_data.npy')
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
root = '/mnt/ceph/users/ophilcox/Oliver/planck_maps/'
outroot = '/mnt/ceph/users/ophilcox/planck_trispectrum_pol/TE_odd/'
datafile = 'COM_CMB_IQU-smica_2048_R3.00_full.fits'  # Data map (from 1905.05697, 2018 SMICA map)
if sim_id!=-1:
    # Simulation map (SMICA FFP10 from PLA - CMB + noise realizations)
    simfile_cmb = 'ffp10/dx12_v3_smica_cmb_mc_%s_raw.fits'%str(sim_id).zfill(5)
    simfile_noise = 'ffp10/dx12_v3_smica_noise_mc_%s_raw.fits'%str(sim_id).zfill(5)

# Beam (temperature and polarization)
l = np.arange(lmax+1)
beam_datT = fitsio.read(root+datafile,ext=2)['INT_BEAM']
beam_intT = InterpolatedUnivariateSpline(np.arange(len(beam_datT)),beam_datT)
beamT = beam_intT(l)*(l>=2)+(l<2)*1
beam_datP = fitsio.read(root+datafile,ext=2)['POL_BEAM']
beam_intP = InterpolatedUnivariateSpline(np.arange(len(beam_datP)),beam_datP)
beamP = beam_intP(l)*(l>=2)+(l<2)*1
beam = [beamT, beamP]

# Base class
Sl_weighting = np.load('/mnt/home/ophilcox/PolyBin/planck_public/Sl_weighting_pol256.npy',allow_pickle=True).flat[0]
assert len(Sl_weighting['TT'])==lmax+1
base = pb.PolyBin(Nside, Sl_weighting, beam=beam, pol=True, backend='libsharp', include_pixel_window=include_pixel_window)

# Check if output exists
outfile = outroot+'trispectrum_numerator%d_(%d,%d,%d).txt'%(sim_id,len(l_bins)-1,len(l_bins_squeeze)-1,len(L_bins)-1)

if os.path.exists(outfile):
    print("Fisher matrix already computed; exiting!")
    sys.exit()

# Load masks
smooth_mask = healpy.read_map(root+'smooth_mask%d.fits'%Nside)
inpainting_mask = healpy.read_map(root+'inpainting_mask%d.fits'%Nside)

# Define S^-1 weighting for each l,m

# Interpolate S_l to all ell and m values
ls = np.arange(lmax+1)
Sl = [Sl_weighting['TT'],Sl_weighting['TE'],Sl_weighting['TB'],Sl_weighting['EE'],Sl_weighting['EB'],Sl_weighting['BB']]
Sl_lm = [InterpolatedUnivariateSpline(ls, Sl[i])(base.l_arr) for i in range(len(Sl))]

# Compute full matrix of C^XY_lm and C^XY_l
Sl_lm_mat = np.moveaxis(np.asarray([[Sl_lm[0],Sl_lm[1],Sl_lm[2]],
                                    [Sl_lm[1],Sl_lm[3],Sl_lm[4]],
                                    [Sl_lm[2],Sl_lm[4],Sl_lm[5]]]),[2,1,0],[0,2,1])

# Check that matrix is well-posed 
assert (np.linalg.det(Sl_lm_mat)>0).all(), "Determinant of S_l^{XY} matrix is <= 0; are the input power spectra set correctly?"

# Invert matrix for each l,m
inv_Sl_lm_mat = np.moveaxis(np.linalg.inv(Sl_lm_mat),[0,1,2],[2,0,1])

########################### LOAD DATA ###########################
if sim_id==-1:
    print("Using Planck data")
    data = healpy.ud_grade(healpy.read_map(root+datafile,field=[0,1,2]),Nside)
else:
    print("Using SMICA simulation %d"%sim_id)
    sim_cmb = healpy.ud_grade(healpy.read_map(root+simfile_cmb,field=[0,1,2]),Nside)
    sim_noise = healpy.ud_grade(healpy.read_map(root+simfile_noise,field=[0,1,2]),Nside)
    data = sim_cmb + sim_noise

########################### WEIGHTING ###########################

def inpaint_map(input_map):
    """
    Apply linear inpainting to a map, given an inpainting mask
    """
    
    tmp_map = input_map.copy()
    
    # Zero out inpainting regions 
    for i in range(len(tmp_map)):
        tmp_map[i][inpainting_mask==1] = 0 
    
    # Perform iterative impainting
    for i in range(1000):
        for f in range(len(tmp_map)):
            inpaint_pix = np.where((tmp_map[f]==0)&(inpainting_mask==1))[0]
            if len(inpaint_pix)==0:
                break
            # Identify four nearest neighbors
            neighbors = healpy.get_interp_weights(Nside,inpaint_pix)[0]
            tmp_map[f][inpaint_pix] = np.mean(tmp_map[f][neighbors],axis=0)

    return tmp_map

def applySinv(input_map, input_type='map', output_type='map'):
    """
    Apply the quasi-optimal weighting, S^{-1} to a map. This firstly inpaints small holes in the data, applies a smooth mask, then weights by an ell-dependent factor.
    
    Note that this is neither diagonal nor invertible. The weighting is given by the inverse of Cl^{XY}_lm = B_l^2 C_l^XY + Kronecker[X,Y] N_l^XX here for beam B_l.
    
    The code has two input and output options: "harmonic" or "map", to avoid unnecessary transforms.
    """
    assert input_type in ['harmonic','map'], "Valid input types are 'harmonic' and 'map' only!"
    assert output_type in ['harmonic','map'], "Valid output types are 'harmonic' and 'map' only!"
    
    ## Transform to real-space, if necessary
    if input_type=='harmonic': 
        input_map = base.to_map(input_map)
        
    ## Step 1: inpaint the data
    tmp_map = inpaint_map(input_map)
    
    ## Step 2: mask out the large bad regions
    tmp_map *= smooth_mask
        
    ## Step 3: Apply S+N weighting in harmonic space
    Cinv_tmp_lm = np.einsum('ijk,jk->ik',base.inv_Cl_lm_mat,base.to_lm(tmp_map),order='C')
    
    # Return to map-space, if necessary
    if output_type=='map': return base.to_map(Cinv_tmp_lm)
    else: return Cinv_tmp_lm

########################### COMPUTE NUMERATOR ###########################

# Initialize trispectrum class
tspec = pb.TSpec(base, 1.+0.*smooth_mask, applySinv, l_bins, l_bins_squeeze=l_bins_squeeze, L_bins=L_bins, fields=fields, parity=parity)

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
    cmb_ii = healpy.ud_grade(healpy.read_map(root+simfile_cmb_ii,field=[0,1,2]),Nside)
    noise_ii = healpy.ud_grade(healpy.read_map(root+simfile_noise_ii,field=[0,1,2]),Nside)
    sim1 = cmb_ii + noise_ii

    # Load second simulation
    simfile_cmb_ii = 'ffp10/dx12_v3_smica_cmb_mc_%s_raw.fits'%str(iip1).zfill(5)
    simfile_noise_ii = 'ffp10/dx12_v3_smica_noise_mc_%s_raw.fits'%str(iip1).zfill(5)
    cmb_ii = healpy.ud_grade(healpy.read_map(root+simfile_cmb_ii,field=[0,1,2]),Nside)
    noise_ii = healpy.ud_grade(healpy.read_map(root+simfile_noise_ii,field=[0,1,2]),Nside)
    sim2 = cmb_ii + noise_ii
    
    # Add to outputs
    alpha_sims.append([sim1,sim2])

# Load into PolyBin
tspec.load_sims(alpha_sims, verb=True, preload=False)

# Compute trispectrum numerator
print("Starting numerator computation")
start = time.time()
numerator = tspec.Tl_numerator(data, parity='both', include_disconnected_term=True, verb=True)
print("Computed trispectrum contribution after %.2f s"%(time.time()-start))

# Print some diagnostics
print("Computation complete using %d forward and %d reverse SHTs"%(base.n_SHTs_forward, base.n_SHTs_reverse))

np.savetxt(outfile,np.concatenate(numerator))
print("Output saved to %s; exiting after %.2f seconds"%(outfile,time.time()-init))