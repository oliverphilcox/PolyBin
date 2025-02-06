### SAMPLE CODE FOR RUNNING POLYSPEC
# This script demonstrates how to compute the binned CMB bispectra from Planck data.
# It is similar to the scripts used in Philcox 2024, and works at low resolution (lmax=500).
# This computes the T+E-mode squeezed bispectrum across a range of triangles.
# The full data result can be obtained via "result = np.linalg.inv(np.mean(fishs,axis=0))@BlCD""
# Note that including the linear term (using BlC instead of Bl3 below) is optional, but may lead to reduced variance. 
# See the tutorial for further details and "getting started" examples.
# Author: Oliver Philcox (2025)

# Import modules
import healpy, camb, os, sys, time, numpy as np, polyspec as ps

#################################### PARAMETERS ####################################

# Fields (see tutorial for list of available fields)
fields = ['TTT','TTE','TEE','EEE']
parity = 'even'

# Directories
data_dir = '/mnt/home/ophilcox/ceph/planck_npipe/' # Data directory
output_dir = '/mnt/ceph/users/ophilcox/polyspec_planck/binned_example/' # Output directory
if not os.path.exists(output_dir): os.makedirs(output_dir)

# Binning
Nside = 256
l_bins = np.array([2, 5, 10, 20]) # bin-edges for soft leg
l_bins_squeeze = np.array([2, 5, 10, 20, 100, 250, 500]) # bin-edges for hard leg
include_partial_triangles = False # Whether to include bins only partially satisfying triangle conditions

# Hyperparameters
N_linear = 50 # Number of simulations used in the linear term
N_fish = 10 # Number of Monte Carlo realizations for the Fisher matrix
N_sim = 50 # Number of simulations to analyze
backend = 'ducc' # Either 'healpix' or 'ducc' (ducc is recommended).

# Dataset
compsep = 'sevem' # Either 'sevem' or 'smica'

# Input files
smooth_maskfile = 'cosine2deg_maskTP_gal70_256.fits' # Smooth analysis mask
inpainting_maskfile = 'inpainting_maskTP_256.fits' # Inpainting mask
beam_file = 'beam_planck256.npy' # Planck beam
noise_cl_file = 'Nl_spectra_planck256.npy' # Planck noise power spectrum 

# Weighting
weighting_type = 'ideal' # either 'ideal' or 'optimal'. Optimal is more expensive
noise_cov = None # Pixel-space noise covariance. Only needed for optimal weighting.

# Fiducial cosmological parameters
H0, ombh2, omch2, tau, ns, deltaR2, mnu = 67.32117, 0.0223828, 0.1201075, 0.05430842, 0.9660499, 2.100549e-9, 0.06

# Miscellaneous
preload = True # Whether to hold disconnected simulations in memory (faster) or reload from disk when needed (less memory)

# CGD parameters (only needed for optimal weighting)
cgd_steps = 50 # Number of steps
cgd_thresh = 1e-5 # Convergence threshold
cgd_preconditioner = 'harmonic' # Choice of preconditioner: 'harmonic' or 'pseudo_inverse'

#################################### SET-UP ####################################

init_time = time.time()

# Masks
print("Loading masks...")
smooth_mask = healpy.read_map(data_dir+smooth_maskfile,field=[0,1,2])  
inpainting_mask = healpy.read_map(data_dir+inpainting_maskfile,field=[0,1,2])

# Drop any small values in mask
smooth_mask[smooth_mask<0.01] = 0.
inpainting_mask[smooth_mask==0]=0

# Load Planck 2018 cosmology
print("Loading cosmology...")
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, num_massive_neutrinos=1, mnu=mnu)
pars.InitPower.set_params(As=deltaR2, ns=ns, r=0, nt=0)
pars.set_for_lmax(3500)

# Compute CMB power spectra
results = camb.get_results(pars);
cls = results.get_cmb_power_spectra(pars,raw_cl=True,CMB_unit='K')
clTT, clEE, clBB, clTE = cls['lensed_scalar'].T

# Load temperature and polarization beam from file
# Note: This should include any transfer functions and pixel window functions
beam = np.load(data_dir+beam_file)[:,:3*Nside]

# Load noise power spectrum from file 
Nl = np.load(data_dir+noise_cl_file,allow_pickle=True).flat[0]
Nl = {k: Nl[k][:3*Nside] for k in Nl.keys()}

# Compute total power spectrum [C^fid(l) = B^2(l) C(l) + N(l)]
l_arr = np.arange(3*Nside)
Cl_fid = {'TT': (beam[0]**2*clTT[:3*Nside]+Nl['TT'])*(l_arr>=2),
          'TE': (beam[0]*beam[1]*clTE[:3*Nside])*(l_arr>=2),
          'TB': 0.*clTT[:3*Nside],
          'EE': (beam[1]**2*clEE[:3*Nside]+Nl['EE'])*(l_arr>=2),
          'EB': 0.*clTT[:3*Nside],
          'BB': (beam[1]**2*clBB[:3*Nside]+Nl['BB'])*(l_arr>=2)}

# Define base PolySpec class, loading in power spectra and beam
print("Loading base...")
base = ps.PolySpec(Nside, Cl_fid, beam, pol=True, backend=backend)

# Define weighting class and S^-1 weighting scheme
cl_dict = {'TT':clTT,'TE':clTE,'EE':clEE,'BB':clBB}
weightings = ps.Weightings(base, smooth_mask, cl_dict, noise_cov, inpainting_mask)

def applySinv(input_map, input_type='map', lmax=3*Nside-1):
    """
    Dummy function to call the S^-1 weighting with any relevant hyperparameters.
    """    
    if weighting_type=='ideal':
        return weightings.applySinv_ideal(input_map, input_type=input_type, lmax=lmax)
    elif weighting_type=='optimal':
        return weightings.applySinv_optimal(input_map, preconditioner=cgd_preconditioner, nstep=cgd_steps, thresh=cgd_thresh, verb=False, input_type=input_type, lmax=lmax)

def load_planck(sim_id):
    """Load the Planck data or FFP10 simulations. We use the PR4 NPIPE maps with SEVEM/SMICA component separation, where the SMICA maps are custom-made using the PR3 weights. We apply the smooth mask to all maps (with inpainting added later).
    
    sim_id specifies the simulation number: sim_id = -1 is the Planck data.
    
    We use simulations 300-399 for testing and 400-499 for the linear term.
    """
    
    # Planck data
    if sim_id==-1:
        if compsep=='sevem':  
            # Full dataset, downloaded from NERSC
            print("Loading Planck SEVEM")
            datafile = 'sevem/SEVEM_NPIPE_2019/npipe6v20_sevem_cmb_005a_2048.fits'
            data = healpy.ud_grade(healpy.read_map(data_dir+datafile,field=[0,1,2]),Nside)
        elif compsep=='smica':
            # Full dataset, created from PR4 frequency maps
            print("Loading Planck SMICA")
            datafile = 'smica/smica2_npipe_planck.fits'
            data = healpy.ud_grade(healpy.read_map(data_dir+datafile,field=[0,1,2]),Nside)
            
    # FFP10 Simulations
    else:
        if compsep=='sevem':
            
            # Input files (from NERSC, including "noisefix" simulations)
            simfile_cmb = 'sevem/SEVEM_NPIPE_sims/SEVEM_NPIPE_cmb_sim%s.fits'%str(sim_id).zfill(4)
            simfile_noise = 'sevem/SEVEM_NPIPE_sims/SEVEM_NPIPE_noise_sim%s.fits'%str(sim_id).zfill(4)
            simfile_noisefix = 'sevem/SEVEM_NPIPE_sims/SEVEM_NPIPE_noisefix_sim%s.fits'%str(sim_id).zfill(4)
    
            # Load CMB and noise separately and combine
            data = healpy.ud_grade(healpy.read_map(data_dir+simfile_noise,field=[0,1,2]),Nside)
            data += healpy.ud_grade(healpy.read_map(data_dir+simfile_cmb,field=[0,1,2]),Nside)
            data += healpy.ud_grade(healpy.read_map(data_dir+simfile_noisefix,field=[0,1,2]),Nside)
            
        elif compsep=='smica':
            
            # Input files (created from NERSC PR4 frequency maps)
            # These include CMB, noise and noisefix simulations
            simfile = 'smica/smica2_npipe_%d.fits'%sim_id
            data = healpy.ud_grade(healpy.read_map(data_dir+simfile,field=[0,1,2]),Nside)
        
    # Apply mask
    masked_data = smooth_mask * data
    
    # Return data
    return masked_data

# Define output files
fishfile = output_dir+'fish.npy'
Bl3file = output_dir+'Bl3.npy'
BlCfile = output_dir+'BlC.npy'
Bl3file_data = output_dir+'Bl3_data.npy'
BlCfile_data = output_dir+'BlC_data.npy'

# Initialize PolySpec bispectrum class
bspec = ps.BSpecBin(base, smooth_mask, applySinv, l_bins, l_bins_squeeze=l_bins_squeeze, fields=fields, parity=parity, include_partial_triangles=include_partial_triangles)

print("\n## Setup Time: %.2fs\n"%(time.time()-init_time))
init_time = time.time()

#################################### FISHER MATRIX ####################################

# Compute the Fisher matrix, if not present
if not os.path.exists(fishfile):
    print("## Computing Fisher matrix")
    
    # Compute Fisher matrix over Monte Carlo realizations
    fishs = []
    for sim_index in range(N_fish):
        fishs.append(bspec.compute_fisher_contribution(sim_index+int(1e4), verb=(sim_index==0)))
    np.save(fishfile,np.asarray(fishs))

    # Report diagnostics
    print("\n## Fisher Matrix Time: %.2fs"%(time.time()-init_time))
    print("## Number of SHTs: %d forward and %d reverse\n"%(base.n_SHTs_forward, base.n_SHTs_reverse))
    base.n_SHTs_forward, base.n_SHTs_reverse = 0,0
    init_time = time.time()

#################################### NUMERATORS ####################################

# Compute numerators, if not present
if not (os.path.exists(Bl3file) and os.path.exists(BlCfile)):
    
    print("## Computing simulation numerators")
    
    # Load linear simulations
    # If preload=False, we will load the disconnected sims on the fly
    bspec.load_sims(lambda ii: load_planck(300+ii), N_linear, verb=True, preload=preload)

    # Load and analyze simulations
    Bl3s = []
    BlCs = []
    for sim_index in range(300,300+N_sim):
        print("### SIMULATION %d"%(sim_index))
        
        # Load simulation
        sim = load_planck(sim_index)
        
        # Compute 3-field + connected bispectrum
        Bl3s.append(bspec.Bl_numerator(sim, include_linear_term=False, verb=(sim_index==1)))
        BlCs.append(bspec.Bl_numerator(sim, include_linear_term=True, verb=(sim_index==1)))

    # Save outputs
    np.save(Bl3file, np.asarray(Bl3s))
    np.save(BlCfile, np.asarray(BlCs))
    
    # Report diagnostics
    print("## Numerator Time (%d simulations): %.2fs"%(N_sim,time.time()-init_time))
    print("## Number of SHTs: %d forward and %d reverse"%(base.n_SHTs_forward, base.n_SHTs_reverse))
    base.n_SHTs_forward, base.n_SHTs_reverse = 0,0
    init_time = time.time()

# Compute data numerator, if not present
if not (os.path.exists(Bl3file_data) and os.path.exists(BlCfile_data)):
    
    print("## Computing data numerators")
    
    # Load linear simulations
    # If preload=False, we will load the disconnected sims on the fly
    if not hasattr(bspec, 'p1_H_maps'):
        bspec.load_sims(lambda ii: load_planck(300+ii), N_linear, verb=True, preload=preload)

    # Load Planck data
    data = load_planck(-1)
    Bl3D = bspec.Bl_numerator(data, include_linear_term=False, verb=False)
    BlCD = bspec.Bl_numerator(data, include_linear_term=True, verb=False)
    
    # Save outputs
    np.save(Bl3file_data, np.asarray(Bl3D))
    np.save(BlCfile_data, np.asarray(BlCD))
    
    # Report diagnostics
    print("\n## Numerator Time (data): %.2fs"%(time.time()-init_time))
    print("## Number of SHTs: %d forward and %d reverse\n"%(base.n_SHTs_forward, base.n_SHTs_reverse))
    base.n_SHTs_forward, base.n_SHTs_reverse = 0,0
    init_time = time.time()
    
print("## All complete!")