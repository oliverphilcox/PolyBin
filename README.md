![logo](logo.png)

# PolyBin
PolyBin is a Python code that estimates the binned power spectrum, bispectrum, and trispectrum for full-sky HEALPix maps such as the CMB, using the algorithms of [Philcox 2023a](https://arxiv.org/abs/2303.08828) and [Philcox 2023c](https://arxiv.org/abs/2306.03915). This can include both spin-0 and spin-2 fields, such as the CMB temperature and polarization, or galaxy positions and galaxy shear. Alternatively, one can use only scalar maps. For each statistic, two estimators are available: the standard (ideal) estimators, which do not take into account the mask, and window-deconvolved estimators. In the second case, we require computation of a Fisher matrix; this depends on binning and the mask, but does not need to be recomputed for each new simulation. For the bispectrum and trispectrum, we can compute both the *parity-even* and *parity-odd* components, accounting for any leakage between the two.

PolyBin contains the following modules:
- `pspec`: Binned power spectra
- `bspec`: Binned bispectra
- `tspec`: Binned trispectra

For usage details, see the [Tutorial](Tutorial.ipynb). 

In the [planck](planck_public/) directory, we include measurements and analysis of the Planck parity-odd temperature trispectrum, as in [Philcox 2023b](https://arxiv.org/abs/2303.12106). The [scalar](https://github.com/oliverphilcox/PolyBin/tree/scalar) branch contains legacy scalar-only code from the [Philcox 2023a](https://arxiv.org/abs/2303.08828) paper (without a number of optimizations).

### Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Columbia / Simons Foundation)

### Dependencies
- Python 3
- healpy, pywigxjpf, fitsio, tqdm (pip installable)
- pixell [optional, for faster SHTs with libsharp]

### References
**Code references:**
1. Philcox, O. H. E., "Optimal Estimation of the Binned Mask-Free Power Spectrum, Bispectrum, and Trispectrum on the Full Sky: Scalar Edition", (2023) ([arXiv](https://arxiv.org/abs/2303.08828))
2. Philcox, O. H. E., "Optimal Estimation of the Binned Mask-Free Power Spectrum, Bispectrum, and Trispectrum on the Full Sky: Tensor Edition", (2023) ([arXiv](https://arxiv.org/abs/2306.03915))

**List of papers using Polybin:**
1. Philcox, O. H. E., "Do the CMB Temperature Fluctuations Conserve Parity?", (2023) ([arXiv](https://arxiv.org/abs/2303.12106))
2. Philcox, O. H. E., Shiraishi, M., "Testing Parity Symmetry with the Polarized Cosmic Microwave Background", (2023) ([arXiv](http://arxiv.org/abs/2308.03831))
3. Philcox, O. H. E., Shiraishi, M., "Testing Graviton Parity and Gaussianity with Planck T -, E- and B-mode Bispectra", (2023) ([arXiv](https://arxiv.org/abs/2312.12498))
