### SAMPLE CODE FOR RUNNING POLYSPEC
# This script demonstrates how to estimate local and lensing non-Gaussianity parameters from Planck data.
# It is similar to the scripts used in Philcox 2025c and works at full resolution (lmax=2048).
# This computes all parts of the estimator: optimization, the Fisher matrix, and the numerator.
# The full data result can be obtained via "result = np.linalg.inv(np.mean(fishs,axis=0))@TlCD""
# See the tutorial for further details and "getting started" examples.
# Author: Oliver Philcox (2025)

# Import modules
import healpy, camb, os, numpy as np, polyspec as ps

#################################### PARAMETERS ####################################

# Templates (see tutorial for list of available templates)
templates = ['gNL-loc','tauNL-loc','lensing']

# Directories
data_dir = '/mnt/home/ophilcox/ceph/planck_npipe/' # Data directory
output_dir = '/mnt/ceph/users/ophilcox/polyspec_planck/local_example/' # Output directory
if not os.path.exists(output_dir): os.makedirs(output_dir)

# Resolution
Nside = 1024 # This is sufficient for lmax = 2048
lmin, lmax = 2, 2048
Lmin, Lmax = 1, 30
Lmin_lens, Lmax_lens = 2, 2048

# Hyperparameters
N_disc = 100 # Number of disconnected simulations
N_fish = 10 # Number of Monte Carlo realizations for the Fisher matrix
N_sim = 100 # Number of simulations to analyze
backend = 'ducc' # Either 'healpix' or 'ducc' (ducc is recommended).

# Dataset
pol = True # Whether to include E+B modes
pol_only = False # If true, use *only* E+B modes
compsep = 'sevem' # Either 'sevem' or 'smica'

# Input files
smooth_maskfile = 'cosine2deg_maskTP_gal70_1024.fits' # Smooth analysis mask
inpainting_maskfile = 'inpainting_maskTP_1024.fits' # Inpainting mask
beam_file = 'beam_planck1024.npy' # Planck beam
noise_cl_file = 'Nl_spectra_planck1024.npy' # Planck noise power spectrum 
cl_correction_file = 'sevem_Cl_correction_N1024_gal70.npy' # Power spectrum correction needed to ensure simulations match data

# Weighting
weighting_type = 'ideal' # either 'ideal' or 'optimal'. Optimal is more expensive
noise_cov = None # Pixel-space noise covariance. Only needed for optimal weighting.

# Fiducial cosmological parameters
H0, ombh2, omch2, tau, ns, deltaR2, mnu = 67.32117, 0.0223828, 0.1201075, 0.05430842, 0.9660499, 2.100549e-9, 0.06

# Optimization settings
optim_thresh = 1e-4 # Tolerance for the optimization algorithm
N_fish_optim = 1 # Average tauNL Fisher matrix over this many iterations during optimization
stalled_iterations = 5 # Failsafe: stop if optimization stalled for this many iterations
N_split = None # Split optimization into this many chunks

# Miscellaneous
preload = True # Whether to hold disconnected simulations in memory (faster) or reload from disk when needed (less memory)
reduce_r = 2 # Downsampling for radial grid
refine_k = 4 # Precision of k-space grid ( = camb AccuracyBoost parameter)

# CGD parameters (only needed for optimal weighting)
cgd_steps = 50 # Number of steps
cgd_thresh = 1e-5 # Convergence threshold
cgd_preconditioner = 'harmonic' # Choice of preconditioner: 'harmonic' or 'pseudo_inverse'

#################################### SET-UP ####################################

## Masks
print("Loading masks...")
if pol:
    smooth_mask = healpy.read_map(data_dir+smooth_maskfile,field=[0,1,2])  
    inpainting_mask = healpy.read_map(data_dir+inpainting_maskfile,field=[0,1,2])
else:
    smooth_mask = healpy.read_map(data_dir+smooth_maskfile,field=0)[None]  
    inpainting_mask = healpy.read_map(data_dir+inpainting_maskfile,field=0)[None]

# Optionally remove temperature contributions
if pol_only:
    assert pol
    smooth_mask[0] = 0.

# Drop any small values in mask
smooth_mask[smooth_mask<0.01] = 0.
inpainting_mask[smooth_mask==0]=0

# Load Planck 2018 cosmology
print("Loading cosmology...")
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, num_massive_neutrinos=1, mnu=mnu)
pars.InitPower.set_params(As=deltaR2, ns=ns, r=0, nt=0)
pars.set_for_lmax(3500, lens_potential_accuracy=1, lens_margin=0, nonlinear=True, k_eta_fac=2.5)

# Compute k array and transfer function
transfer_file = data_dir+'camb_transfer_planck.npz'
if not os.path.exists(transfer_file):
    print("Computing transfer functions...")
    
    # Transfer function settings
    pars.DoLensing = True
    pars.BessIntBoost = 30
    pars.KmaxBoost=3
    pars.IntTolBoost=4
    pars.TimeStepBoost=5
    pars.SourcekAccuracyBoost=5
    pars.BesselBoost=5
    pars.IntkAccuracyBoost=5
    pars.AccurateBB=True
    pars.AccurateReionization=True
    pars.AccuratePolarization=True
    pars.set_accuracy(DoLateRadTruncation=False,AccuracyBoost=refine_k,lSampleBoost=50,lAccuracyBoost=2);

    # Compute transfer functions and spectra    
    results = camb.get_results(pars);
    cls = results.get_cmb_power_spectra(pars,raw_cl=True,CMB_unit='K')
    trans = results.get_cmb_transfer_data('scalar')
    ls, qs, DeltasT = trans.get_transfer(source=0)
    ls, qs, DeltasE = trans.get_transfer(source=1)

    # Rescale E mode amplitude for output (usually done later in camb)
    ls = ls.astype(int)
    prefactorE = np.sqrt((ls + 2) * (ls + 1) * ls * (ls - 1))
    DeltasE *= prefactorE[:,None]

    # Save transfer functions
    np.savez(transfer_file,l=ls,k=qs,transfer=np.asarray([DeltasT,DeltasE]),cls=cls)

# Load transfer functions
with np.load(transfer_file,allow_pickle=True) as transfer_inp:
    print("Loading transfer functions...")
    k_arr = transfer_inp['k']
    if pol:
        # Convert to K units and add ell=0,1 modes
        TlT_k_arr = np.vstack([[np.zeros_like(k_arr) for _ in range(2)],transfer_inp['transfer'][0,:3*Nside]*pars.TCMB])
        TlE_k_arr = np.vstack([[np.zeros_like(k_arr) for _ in range(2)],transfer_inp['transfer'][1,:3*Nside]*pars.TCMB])
        Tl_arr = [TlT_k_arr, TlE_k_arr]
    else:
        # Convert to K units and add ell=0,1 modes
        TlT_k_arr = np.vstack([[np.zeros_like(k_arr) for _ in range(2)],transfer_inp['transfer'][0,:3*Nside]*pars.TCMB])
        Tl_arr = [TlT_k_arr]

    # Load lensed power spectra
    clTT, clEE, clBB, clTE = transfer_inp['cls'].flat[0]['lensed_scalar'].T
    clPP = transfer_inp['cls'].flat[0]['lens_potential'][:,0]

# Load temperature and polarization beam from file
# Note: This should include any transfer functions and pixel window functions
beam = np.load(data_dir+beam_file)[:1+pol,:3*Nside]

# Load noise power spectrum from file 
Nl = np.load(data_dir+noise_cl_file,allow_pickle=True).flat[0]
Nl = {k: Nl[k][:3*Nside] for k in Nl.keys()}

# Compute total power spectrum [C^fid(l) = B^2(l) C(l) + N(l)]
l_arr = np.arange(3*Nside)
if pol:
    Cl_fid = {'TT': (beam[0]**2*clTT[:3*Nside]+Nl['TT'])*(l_arr>=2),
              'TE': (beam[0]*beam[1]*clTE[:3*Nside])*(l_arr>=2),
              'TB': 0.*clTT[:3*Nside],
              'EE': (beam[1]**2*clEE[:3*Nside]+Nl['EE'])*(l_arr>=2),
              'EB': 0.*clTT[:3*Nside],
              'BB': (beam[1]**2*clBB[:3*Nside]+Nl['BB'])*(l_arr>=2)}
else:
    Cl_fid = {'TT': (beam[0]**2*clTT[:3*Nside]+Nl['TT'])*(l_arr>=2)}

# Define base PolySpec class, loading in power spectra and beam
print("Loading base...")
base = ps.PolySpec(Nside, Cl_fid, beam, pol=pol, backend=backend)

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

def load_sim_pair(i):
    """
    Load a pair of simulations. These should have the same two-point function as the data.
    
    Here, we use the FFP10 simulations.
    """
    assert N_sim <= 100, "Cannot reuse simulations!"
    return [load_planck(i+400),load_planck(i+400+N_disc//2)]

def load_planck(sim_id):
    """Load the Planck data or FFP10 simulations. We use the PR4 NPIPE maps with SEVEM/SMICA component separation, where the SMICA maps are custom-made using the PR3 weights. We apply the smooth mask to all maps (with inpainting added later).
    
    sim_id specifies the simulation number: sim_id = -1 is the Planck data.
    
    We use simulations 300-399 for testing and 400-499 for the disconnected contributions.
    """
    
    # Planck data
    if sim_id==-1:
        if compsep=='sevem':  
            # Full dataset, downloaded from NERSC
            print("Loading Planck SEVEM")
            datafile = 'sevem/SEVEM_NPIPE_2019/npipe6v20_sevem_cmb_005a_2048.fits'
            if pol:
                data = healpy.ud_grade(healpy.read_map(data_dir+datafile,field=[0,1,2]),Nside)
            else:
                data = healpy.ud_grade(healpy.read_map(data_dir+datafile,field=[0]),Nside)[None]
        elif compsep=='smica':
            # Full dataset, created from PR4 frequency maps
            print("Loading Planck SMICA")
            datafile = 'smica/smica2_npipe_planck.fits'
            if pol:
                data = healpy.ud_grade(healpy.read_map(data_dir+datafile,field=[0,1,2]),Nside)
            else:
                data = healpy.ud_grade(healpy.read_map(data_dir+datafile,field=[0]),Nside)[None]
            
    # FFP10 Simulations
    else:
        if compsep=='sevem':
            
            # Input files (from NERSC, including "noisefix" simulations)
            simfile_cmb = 'sevem/SEVEM_NPIPE_sims/SEVEM_NPIPE_cmb_sim%s.fits'%str(sim_id).zfill(4)
            simfile_noise = 'sevem/SEVEM_NPIPE_sims/SEVEM_NPIPE_noise_sim%s.fits'%str(sim_id).zfill(4)
            simfile_noisefix = 'sevem/SEVEM_NPIPE_sims/SEVEM_NPIPE_noisefix_sim%s.fits'%str(sim_id).zfill(4)
    
            # Load CMB and noise separately and combine
            if pol:
                data = healpy.ud_grade(healpy.read_map(data_dir+simfile_noise,field=[0,1,2]),Nside)
                data += healpy.ud_grade(healpy.read_map(data_dir+simfile_cmb,field=[0,1,2]),Nside)
                data += healpy.ud_grade(healpy.read_map(data_dir+simfile_noisefix,field=[0,1,2]),Nside)
            else:
                data = healpy.ud_grade(healpy.read_map(data_dir+simfile_noise,field=[0]),Nside)
                data += healpy.ud_grade(healpy.read_map(data_dir+simfile_cmb,field=[0]),Nside)
                data += healpy.ud_grade(healpy.read_map(data_dir+simfile_noisefix,field=[0]),Nside)
                data = data[None]
        
        elif compsep=='smica':
            
            # Input files (created from NERSC PR4 frequency maps)
            # These include CMB, noise and noisefix simulations
            simfile = 'smica/smica2_npipe_%d.fits'%sim_id
            if pol:
                data = healpy.ud_grade(healpy.read_map(data_dir+simfile,field=[0,1,2]),Nside)
            else:
                data = healpy.ud_grade(healpy.read_map(data_dir+simfile,field=[0]),Nside)[None]

        # Correct the simulations by adding additional Gaussian noise to account for the high-ell C_l^TT deficit. This follows Marzouk et al. 2022
        if cl_correction_file is not None:
            DeltaClTT,DeltaClEE,DeltaClBB,DeltaClTE = np.load(data_dir+cl_correction_file)
            np.random.seed(sim_id)
            delta_map = healpy.synfast([DeltaClTT,DeltaClEE,DeltaClBB,DeltaClTE],Nside,new=True)
            if pol:
                data += delta_map
            else:
                data += delta_map[0][None]
            
    # Apply mask
    masked_data = smooth_mask * data
    
    # Return data
    return masked_data

# Define output files
weightfile = output_dir+'opt_radii.npz'
fishfile = output_dir+'fish.npy'
Tl0file = output_dir+'Tl0.npy'
TlCfile = output_dir+'TlC.npy'
Tl4file = output_dir+'Tl4.npy'
TlCfile_data = output_dir+'TlC_data.npy'
Tl4file_data = output_dir+'Tl4_data.npy'
Tl0file_data = output_dir+'Tl0_data.npy'

#################################### OPTIMIZATION ####################################

# Compute optimal weights if weightfile does not exist
if not os.path.exists(weightfile):
    print("## Computing optimization")
    
    # Load PolySpec template class
    tspec = ps.TSpecTemplate(base, smooth_mask, applySinv, templates,
                    lmin, lmax, k_arr=k_arr, Tl_arr=Tl_arr, Lmin=Lmin, Lmax=Lmax, Lmin_lens=Lmin_lens, Lmax_lens=Lmax_lens, 
                    k_pivot=pars.InitPower.pivot_scalar, C_phi = clPP, C_lens_weight = cl_dict, 
                    ns = pars.InitPower.ns, As = pars.InitPower.As)
    
    def run_optimization(split_index=None, N_split=None, initial_r_points=None):
        """Run 1D optimization, splitting into chunks if desired"""
        return tspec.optimize_radial_sampling_1d(reduce_r=reduce_r, tolerance=optim_thresh, N_fish_optim=N_fish_optim, stalled_iterations=stalled_iterations, verb=True, split_index=split_index, N_split=N_split, initial_r_points=initial_r_points)
                
    # Run 1D optimization, splitting computation into chunks if desired
    if N_split is None or N_split==1:
        
        # Run 1D optimization
        r_arr, r_weights = run_optimization()
        # Save outputs
        np.savez(weightfile,r_arr=r_arr,r_weights=r_weights, ideal_fisher=tspec.ideal_fisher)
    
    else:
        
        # Iterate over splits
        for split_index in range(N_split):
            print("Optimizing weights for split %d of %d"%(split_index+1,N_split))
            weightfile_split = output_dir+'opt_radii%d.npz'%split_index
    
            if not os.path.exists(weightfile_split):
                # Run 1D optimization
                r_arr, r_weights = run_optimization(split_index, N_split)
                # Save outputs (just in case)
                np.savez(weightfile_split, r_arr=r_arr, r_weights=r_weights, ideal_fisher=tspec.ideal_fisher)
        
        # Load all splits
        initial_r_points = []
        for split_index in range(N_split):
            weightfile_split = output_dir+'opt_radii%d.npz'%split_index
            initial_r_points.append(np.load(weightfile_split)['r_arr'])
        initial_r_points = np.concatenate(initial_r_points)
        
        # Run 1D optimization, starting from these points
        run_optimization(initial_r_points=initial_r_points)
        # Save outputs
        np.savez(weightfile,r_arr=r_arr, r_weights=r_weights, ideal_fisher=tspec.ideal_fisher)
        
    # Report diagnostics
    tspec.report_timings()
    tspec.reset_timings()
    
# Load weight array and initialize PolySpec template class
weight_arr = np.load(weightfile,allow_pickle=True)
tspec = ps.TSpecTemplate(base, smooth_mask, applySinv, templates, k_arr, Tl_arr, 
                         lmin, lmax, Lmin=Lmin, Lmax=Lmax, Lmin_lens=Lmin_lens, Lmax_lens=Lmax_lens,
                         k_pivot=pars.InitPower.pivot_scalar, C_phi = clPP, C_lens_weight = cl_dict, 
                         r_values = weight_arr['r_arr'], r_weights = weight_arr['r_weights'].flat[0],
                         ns = pars.InitPower.ns, As = pars.InitPower.As)

#################################### FISHER MATRIX ####################################

# Compute the Fisher matrix, if not present
if not os.path.exists(fishfile):
    print("## Computing Fisher matrix")
    
    # Compute Fisher matrix over Monte Carlo realizations
    fishs = []
    for sim_index in range(N_fish):
        fishs.append(tspec.compute_fisher_contribution(sim_index+int(1e4), verb=(sim_index==0)))
    np.save(fishfile,np.asarray(fishs))

    # Report diagnostics
    tspec.report_timings()
    tspec.reset_timings()
        
#################################### NUMERATORS ####################################

# Compute numerators, if not present
if not (os.path.exists(Tl4file) and os.path.exists(Tl0file) and os.path.exists(TlCfile)):
    
    print("## Computing simulation numerators")
    
    # Load disconnected simulations
    # If preload=False, we will load the disconnected sims on the fly
    tspec.load_sims(load_sim_pair, N_disc//2, preload=preload, verb=True)

    # Load and analyze simulations
    Tl4s = []
    TlCs = []
    for sim_index in range(300,300+N_sim):
        print("### SIMULATION %d"%(sim_index))
        
        # Load simulation
        sim = load_planck(sim_index)
        
        # Compute 4-field + connected trispectra
        Tl4s.append(tspec.Tl_numerator(sim, include_disconnected_term=False, verb=(sim_index==1)))
        TlCs.append(tspec.Tl_numerator(sim, include_disconnected_term=True, verb=(sim_index==1)))

    # Save outputs
    np.save(Tl4file, np.asarray(Tl4s))
    np.save(TlCfile, np.asarray(TlCs))
    np.save(Tl0file, tspec.t0_num)

    # Report diagnostics
    tspec.report_timings()

# Compute data numerator, if not present
if not (os.path.exists(Tl4file_data) and os.path.exists(Tl0file_data) and os.path.exists(TlCfile_data)):

    print("## Computing data numerators")

    # Load disconnected simulations
    # If preload=False, we will load the disconnected sims on the fly
    if not hasattr(tspec, 't0_num'):
        tspec.load_sims(load_sim_pair, N_disc//2, preload=preload, verb=True, input_type='map')
    
    # Load Planck data
    data = load_planck(-1)
    Tl4D = tspec.Tl_numerator(data, include_disconnected_term=False, verb=False)
    TlCD = tspec.Tl_numerator(data, include_disconnected_term=True, verb=False)

    # Save outputs
    np.save(Tl4file_data, np.asarray(Tl4D))
    np.save(TlCfile_data, np.asarray(TlCD))
    np.save(Tl0file_data, tspec.t0_num)
    
    # Report diagnostics
    tspec.report_timings()

print("## All complete!")