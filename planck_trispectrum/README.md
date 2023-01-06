
This directory contains code and analysis products for the measurement of the large-scale parity-odd Planck temperature trispectrum in [Philcox 2023](https://arxiv.org/2301.XXXXX). In particular, we give code for computing spectra from Planck data, computing theoretical templates, and performing the analysis. Note that most routines should be run on a cluster. 

## Data
- ```trispectrum_num.py```: Compute the trispectrum numerator for Planck data or FFP10 simulations.
- ```trispectrum_fish.py```: Compute the trispectrum normalization matrix. This should be run for O(100) Monte Carlo iterations.
We additionally provide sample SLURM submission scripts for these calculations.

## Models
- ```ghost
- ```collider.py```: Compute the CMB trispectrum arising from inflationary exchange of a spin-1 particle, for mass nu and sound-speed cs
- ```gauge.py```: Compute the CMB trispectrum arising from an inflationary gauge field with Chern-Simons interactions and a non-vanishing vev.
- ```ghost.py```: Compute the CMB trispectrum arising from ghost inflation with parity-violating interactions.
- ```fish.py```: Compute the idealized trispectrum normalization matrix, for use in Fisher forecasting.

## Analysis pipeline
- ```Analysis.ipynb```: Read-in the Planck and FFP10 simulations and compute null tests and parameter constraints.
- ```Fisher.ipynb```: Forecast the detection significance as a function of ell-max.

