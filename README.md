# Fractal_Covid

**This repository is still in beta.**

This repository is dedicated to the ESMRMB presentation titled: *Temporal fractal analysis of the rs-fMRI signal in Covid-19 recovered patients.*

## Contents

- `synbold.sh`:  
  Performs preprocessing of fMRI files. Copies a `.fsf` FEAT file to each BIDS directory, performs minimal preprocessing with FEAT, and then applies SynBOLD-DisCo susceptibility distortion correction.

- `fmri_reg`:  
  Registers each subject to `MNI152_T1_2mm` included in the FSL distribution using SyN diffeomorphic registration.

- `DFAv2`:  
  Calculates the Hurst exponent for each image. Includes minor code optimizations for increased speed and a "wonky" progress bar. 😊

- `group_dfa.py`:  
  Calculates group-wise statistics in volumetric image form. The results are viewable in any `.nii.gz` viewer.

- `stat_test.py`:  
  Computes the average Hurst value and compares results according to the Yeo17 network. It also calculates the p-value and applies FDR correction.

## Acknowledgements

This repository is intended for research purposes only and is currently under development. Contributions, suggestions, and feedback are welcome.
