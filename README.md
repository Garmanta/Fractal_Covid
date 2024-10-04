# Fractal_Covid
This repository is still on beta
This repository is dedicated to the ESMRMB presentation titled: Temporal fractal analysis of the rs-fMRI 
signal in Covid-19 recovered patient.

DFAv2 : Calculates the Hurst exponent in each image. Contain small code optimizations for speed. Includes a wonky progress bar 8)
group_dfa.py : Calculates group wise statistics in volumetric image form, Viewable in any .nii.gz viewer.
stat_test.py : Computes average Hurst value and compares according to Yeo17 network. Then calculates p value and corrects using FDR.
