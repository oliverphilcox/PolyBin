## IMPORTS
import time, os, sys, healpy
import numpy as np
from classy import Class
sys.path.append('/mnt/home/ophilcox/PolyBin/')
import polybin as pb
from scipy.interpolate import InterpolatedUnivariateSpline

if len(sys.argv)!=3:
    raise Exception("No option or sim-no input supplied!")
option = int(sys.argv[1])
sim_no = int(sys.argv[2])

# HEALPix settings
Nside = 256
lmax = 3*Nside-1

# Whether to include a mask in practice
if option==1:
    flat_mask = False
elif option==2:
    flat_mask = True
else:
    raise Exception("Wrong option!")
    
pol=True
backend = 'libsharp'

outdir = '/mnt/ceph/users/ophilcox/polybin_testing/Cl/'

# Bin edges (could also be non-linearly spaced)
l_bins = np.arange(2,404,10)
Nl = len(l_bins)-1
print("binned lmax: %d, HEALPix lmax: %d"%(np.max(l_bins),lmax))
assert lmax>np.max(l_bins)

# Number of random iterations to create Fisher matrix
N_it = 100 # N ~ 50 is sufficient in practice, we'll use 10 for testing
assert sim_no<=10*N_it-10, "Sim number must be at most %d"%(10*N_it) 

# Number of simulations to use for testing
N_sim = 1000
assert sim_no<=100*N_sim-100, "Sim number must be at most %d"%(100*N_sim) 

# Number of CPUs to run code on
N_cpus = 40

# Whether to include the pixel window function
# This should be set to True, unless we generate maps at the same realization we analyze them!
include_pixel_window = False

# Whether to include bins only partially satisfying triangle conditions
include_partial_triangles = False

# Galactic Mask
# Using GAL040 mask with 2-degree apodization for testing
root = '/mnt/home/ophilcox/ceph/Oliver/planck/'
maskfile = 'HFI_Mask_GalPlane-apo2_2048_R2.00.fits'

cosmo = Class()

# Define ell arrays
l = np.arange(lmax+1)

# Run CLASS
cosmo.set({'output':'tCl,lCl,pCl,mPk','l_max_scalars':lmax+1,'lensing':'yes',
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

# Compute dictionary of signal C_ell (note that we use all real fields by convention)
Cl_dict = cosmo.lensed_cl(lmax);
Cl_th = {}
if pol:
    r_TB = 0.5 # correlation of T and B (usually set to zero)
    r_EB = 0.5 # correlation of E and B (usually set to zer0)
    Cl_th = {}
    Cl_th['TT'] = Cl_dict['tt']*cosmo.T_cmb()**2
    Cl_th['TE'] = Cl_dict['te']*cosmo.T_cmb()**2
    Cl_th['TB'] = r_TB*np.sqrt(Cl_dict['tt']*Cl_dict['bb'])*cosmo.T_cmb()**2
    Cl_th['EE'] = Cl_dict['ee']*cosmo.T_cmb()**2
    Cl_th['EB'] = r_EB*np.sqrt(Cl_dict['ee']*Cl_dict['bb'])*cosmo.T_cmb()**2
    Cl_th['BB'] = Cl_dict['bb']*cosmo.T_cmb()**2
else:
    Cl_th['TT'] = Cl_dict['tt']*cosmo.T_cmb()**2
    
# Compute noise C_ell
DeltaT = 60./60.*np.pi/180.*1e-6 # in K-radians
DeltaP = 60./60.*np.pi/180.*1e-6*np.sqrt(2) 
thetaFWHM = 5./60.*np.pi/180. # in radians
NlT = DeltaT**2*np.exp(l*(l+1)*thetaFWHM**2/(8.*np.log(2)))*(l>=2)
NlP = DeltaP**2*np.exp(l*(l+1)*thetaFWHM**2/(8.*np.log(2)))*(l>=2)
Nl_th = {}
if pol:
    Nl_th['TT'] = NlT
    Nl_th['TE'] = Nl_th['TB'] = Nl_th['EB'] = 0.*NlT
    Nl_th['EE'] = Nl_th['BB'] = NlP
else:
    Nl_th['TT'] = NlT

if flat_mask:
    mask = 1.+0*healpy.ud_grade(healpy.read_map(root+maskfile,field=1),Nside)
else:
    mask = healpy.ud_grade(healpy.read_map(root+maskfile,field=1),Nside)
    
# Define fiducial beam and signal+noise
if not pol:
    beam = [1.+0.*l]
else:
    beam = [1.+0.*l, 1.+0.*l] # Temperature and polarization
Sl_fiducial = {}
for f in Cl_th.keys(): Sl_fiducial[f] = beam[0]**2*Cl_th[f]+Nl_th[f]+(Cl_th[f][2]+Nl_th[f][2])*(l<2) # avoiding zeros at l<2

# Define class, optionally including polarization
base = pb.PolyBin(Nside, Sl_fiducial, beam, include_pixel_window=include_pixel_window, pol=pol, backend=backend)

# Generate unmasked data with known C_l and factorized b
# Cl^XY are set to the fiducial spectrum unless otherwise specified
# No beam is included
print("Generating data")
raw_data = base.generate_data(seed=42, add_B=False)

# Mask the map
data = (raw_data*mask).reshape(len(raw_data),-1)

def applySinv(input_map, input_type='map', output_type='map'):
    """Apply the quasi-optimal weighting, S^{-1} to a map in map- or harmonic-space. 
    
    Here, we assume that the forward covariance is diagonal in ell (though not in fields), in particular C_l, and invert this.
    This is not quite the exact solution (as it incorrectly treats W(n) factors), but will be unbiased.
    
    Note that the code has two input and output options: "harmonic" or "map", to avoid unnecessary transforms.
    """
    assert input_type in ['harmonic','map'], "Valid input types are 'harmonic' and 'map' only!"
    assert output_type in ['harmonic','map'], "Valid output types are 'harmonic' and 'map' only!"
    
    # Transform to harmonic space, if necessary
    if input_type=='map': input_map_lm = base.to_lm(input_map)
    else: input_map_lm = input_map.copy()
    
    # Divide by covariance
    Cinv_data_lm = np.einsum('ijk,jk->ik',base.inv_Cl_lm_mat,input_map_lm,order='C')
        
    # Return to map-space, if necessary
    if output_type=='map': return base.to_map(Cinv_data_lm)
    else: return Cinv_data_lm
    
# Initialize power spectrum class

# Define fields to use
if pol:
    fields=['TT','TE','TB','EE','EB','BB'] # can use any subset of these!
else:
    fields = ['TT']

# NB: use mask = 1.+0.*mask if including mask projection in S^-1 (as in applySinv_planck)
pspec = pb.PSpec(base, mask, applySinv, l_bins, fields=fields)

# Zero cou/mnt/home/ophilcox/ceph/ (for diagnostics only)
base.n_SHTs_forward, base.n_SHTs_reverse = 0, 0

# Compute Fisher matrix
out_fish = outdir+'Cl_fish%d_%d.npy'%(option,sim_no)
if not os.path.exists(out_fish):
    t1 = time.time()
    fishs = []
    for ii in range(10*sim_no,10*sim_no+10):
        print("## Computing Fisher %d of %d"%(ii,N_it))
        fishs.append(pspec.compute_fisher_contribution(ii, verb=True))
    fishs = np.asarray(fishs)
    np.save(out_fish, fishs)
    print("Fisher computation complete, avering %.2f time, %d / %d SHTs"%((time.time()-t1)/10.,base.n_SHTs_forward//10, base.n_SHTs_reverse//10))
#fish = pspec.compute_fisher(N_it, N_cpus=N_cpus, verb=False);

#np.save(outdir+'Cl_fish%d.npy'%option,fish)

### Ideal simulations
out_ideal = outdir+'Cl_ideal%d_%d.npy'%(option,sim_no)
if not os.path.exists(out_ideal):

    t1 = time.time()
    base.n_SHTs_forward, base.n_SHTs_reverse = 0, 0

    Cl_ideal = []
    for ii in range(100*sim_no,100*sim_no+100):
        print("Analyzing sim %d of %d"%(ii+1,N_sim))
        raw_sim = base.generate_data(ii,add_B=False)
        sim = mask*raw_sim
        Cl_ideal.append(pspec.Cl_numerator_ideal(sim))
    Cl_ideal = np.asarray(Cl_ideal)
    print("Ideal sim computation complete, averaging %.2f time, %d / %d SHTs"%((time.time()-t1)/100.,base.n_SHTs_forward//100,base.n_SHTs_reverse//100))
    
    np.save(out_ideal,Cl_ideal)

### Unwindowed simulations
out_unwindowed = outdir+'Cl_unwindowed%d_%d.npy'%(option,sim_no)
if not os.path.exists(out_unwindowed):

    t1 = time.time()
    base.n_SHTs_forward, base.n_SHTs_reverse = 0, 0

    Cl_unwindowed = []
    for ii in range(100*sim_no,100*sim_no+100):
        print("Analyzing sim %d of %d"%(ii+1,N_sim))
        raw_sim = base.generate_data(ii,add_B=False)
        sim = mask*raw_sim
        Cl_unwindowed.append(pspec.Cl_numerator(sim))
        
    Cl_unwindowed = np.asarray(Cl_unwindowed)
    print("Unwindowed sim computation complete, averaging %.2f time, %d / %d SHTs"%((time.time()-t1)/100.,base.n_SHTs_forward//100,base.n_SHTs_reverse//100))

    np.save(out_unwindowed,Cl_unwindowed)

print("PROCESS COMPLETE!")