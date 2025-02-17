![logo](logo.png)

# PolySpec
PolySpec (formerly PolyBin) is a Python code that estimates $N$-point correlation functions for 2D HEALPix maps, optionally with spin (e.g., CMB temperature, polarization, galaxy positions and cosmic shear). There are two main functionalities:
- **Templates**: This directly estimates the amplitude of inflationary templates in CMB data, such as gNL or tauNL, in addition to late time templates such as CMB lensing. This includes thirteen types of trispectrum estimator (see below), and provides quasi-optimal estimates of the template amplitudes, accounting for arbitrary beams, masks, and filtering. 
- **Binned Statistics**: This estimates the binned power spectrum, bispectrum and trispectrum of a 2D field, accounting for correlations between bins. For each statistic, two estimators are available: the standard/ideal estimators (i.e. pseudo-Cl), which do not take into account the mask, and window-deconvolved estimators, which do. In the second case, we require computation of a numerical Fisher matrix; this depends on binning and the mask, but does not need to be recomputed for each new simulation. For the bispectrum and trispectrum, we can compute both the *parity-even* and *parity-odd* components, accounting for any leakage between the two.

PolySpec contains the following main modules:
- `pspec_bin`: Binned power spectra
- `bspec_bin`: Binned bispectra
- `tspec_bin`: Binned trispectra
- `tspec_template`: Direct estimation of trispectrum template amplitudes

In the templates class, we can estimate the following types of trispectra:
- `gNL-loc`, `tauNL-loc`: Cubic local templates
- `gNL-con`: Featureless constant template
- `gNL-dotdot`, `gNL-dotdel`, `gNL-deldel`: Effective Field Theory of Inflation templates
- `tauNL-direc`, `tauNL-even`, `tauNL-odd`: Direction-dependent tauNL templates
- `tauNL-heavy`, `tauNL-light`: Cosmological collider signatures from massive spinning particles
- `lensing`, `point-source`: CMB lensing and point source amplitudes

For details on the binned estimators, see the [Binned Tutorial](Tutorial-Binned.ipynb). For details on the template estimators see the [Template Tutorial](Tutorial-Template.ipynb).

### Example Usage
Below, we demonstrate how to use PolySpec to compute gNL-loc and the lensing amplitude from a dataset. This depends on a number of inputs, which are discussed in the tutorial.
```
import polyspec as ps, numpy as np

# Load base class, specifying the fiducial spectrum and beam
base = ps.PolySpec(Nside, fiducial_Cl_tot, beam, backend="ducc")

# Load the trispectrum template class, specifying the templates to analyze
tspec = ps.TSpecTemplate(base, smooth_mask, applySinv, ["gNL-loc","lensing"], 
                         lmin, lmax, k_array, transfer_array, Lmin, Lmax,
                         C_phi=Cl_phi, C_lens_weight = Cl_lensed)

# Perform optimization to compute the radial integration points
tspec.optimize_radial_sampling_1d()

# Compute the Fisher matrix as a Monte Carlo sum
fish = np.mean([tspec.compute_fisher_contribution(seed) for seed in range(Nfish)],axis=0)

# Compute the trispectrum estimator
tspec.generate_sims(Nnum)
estimate = np.linalg.inv(fish)@tspec.Tl_numerator(data)

# Print run-time statistics
tspec.report_timings()

```
We can use similar code to compute binned polyspectra. Here's an example for the bispectrum:

```
# Load the binned bispectrum class, specifying the fields to analyze
bspec_bin = ps.BSpecBin(base, smooth_mask, applySinv, l_bins, fields=['TTT','TTE','TEE','EEE'])

# Compute the Fisher matrix as a Monte Carlo sum
fish = np.mean([bspec_bin.compute_fisher_contribution(seed) for seed in range(Nfish)],axis=0)

# Compute the bispectrum estimator
bspec_bin.generate_sims(Nnum)
estimate = np.linalg.inv(fish)@bspec_bin.Bl_numerator(data)
```
Further details can be found in the Tutorials. We additionally provide a sample scripts demonstrating the application of PolySpec to [Planck trispectrum templates](run_planck_local_trispectra.py) and [Planck binned bispectra](run_planck_binned_bispectrum.py) in a realistic set-up.

### Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Columbia / Stanford / Simons Foundation)

### Dependencies
- Python 3
- healpy, fitsio, tqdm (pip installable)
- pywigxjpf [for binned spectra]
- Cython [for template analyses]
- ducc0 [optional but recommended, for fast SHTs]
- [wignerSymbols](https://github.com/joeydumont/wignerSymbols) [included with the Cython module]

When using the template estimators, a number of Cython modules must be installed. This can be done using the following command:
```cd polyspec/cython; python setup.py build_ext --inplace; cd ../../```

### Code References
1. Philcox, O. H. E., "Optimal Estimation of the Binned Mask-Free Power Spectrum, Bispectrum, and Trispectrum on the Full Sky: Scalar Edition", (2023) ([arXiv](https://arxiv.org/abs/2303.08828))
2. Philcox, O. H. E., "Optimal Estimation of the Binned Mask-Free Power Spectrum, Bispectrum, and Trispectrum on the Full Sky: Tensor Edition", (2023) ([arXiv](https://arxiv.org/abs/2306.03915))
3. Philcox, O. H. E., "Searching for Inflationary Physics with the CMB Trispectrum: 1. Primordial Theory & Optimal Estimators", (2025) ([arXiv](https://arxiv.org/abs/2502.04434))
4. Philcox, O. H. E., "Searching for Inflationary Physics with the CMB Trispectrum: 2. Code & Validation", (2025) ([arXiv](http://arxiv.org/abs/2502.05258))

### Applications
5. Philcox, O. H. E., "Do the CMB Temperature Fluctuations Conserve Parity?", (2023) ([arXiv](https://arxiv.org/abs/2303.12106))
6. Philcox, O. H. E., Shiraishi, M., "Testing Parity Symmetry with the Polarized Cosmic Microwave Background", (2023) ([arXiv](https://arxiv.org/abs/2308.03831))
7. Philcox, O. H. E., Shiraishi, M., "Testing graviton parity and Gaussianity with Planck T-, E-, and B-mode bispectra", (2024) ([arXiv](https://arxiv.org/abs/2312.12498))
8. Philcox, O. H. E., Shiraishi, M., "Non-Gaussianity Beyond the Scalar Sector: A Search for Tensor and Mixed Tensor-Scalar Bispectra with Planck Data", (2024) ([arXiv](https://arxiv.org/abs/2409.10595))
9. Philcox, O. H. E., "Searching for Inflationary Physics with the CMB Trispectrum: 3. Constraints from Planck", (2025) ([arXiv](https://arxiv.org/abs/2502.06931))
