import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib

mask = nib.load(os.path.join('MNI152_T1_2mm_brain_mask.nii.gz'))
x,y,z = mask.get_fdata().shape
H = np.zeros([x,y,z,100])

#Control
for i in range(1,50):
    
    H[:,:,:,i-1] = np.load(f'control/sub-{i:02d}/sub-{i:02d}_H.npy')
    print('entry: ' +str(i-1) + f' sub-{i:02d}_H.npy')
    
                
#Treatment      
for i in range(1,52):
              
   H[:,:,:,i+48] = np.load(f'treatment/sub-{i:02d}/sub-{i:02d}_H.npy')      
   print('entry: ' +str(i+48) + f' sub-{i:02d}_H.npy')

                
control_group_meanH = np.nanmean(H[:,:,:,0:48], axis=3) 
control_group_varH = np.nanvar(H[:,:,:,0:48], axis=3)               
treatment_group_meanH = np.nanmean(H[:,:,:,49:99], axis=3) 
treatment_group_varH = np.nanvar(H[:,:,:,49:99], axis=3)              

img = nib.Nifti1Image(control_group_meanH.astype(np.float64), mask.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('control_group_meanH'))  # Save as NiBabel file

img = nib.Nifti1Image(control_group_varH.astype(np.float64), mask.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('control_group_varH'))  # Save as NiBabel file

img = nib.Nifti1Image(treatment_group_meanH.astype(np.float64), mask.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('treatment_group_meanH'))  # Save as NiBabel file

img = nib.Nifti1Image(treatment_group_meanH.astype(np.float64), mask.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('treatment_group_varH'))  # Save as NiBabel file
                
img = nib.Nifti1Image(H.astype(np.float64), mask.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('complete_H'))  # Save as NiBabel file
                
img = nib.Nifti1Image((control_group_meanH - treatment_group_meanH).astype(np.float64), mask.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('difference_H'))  # Save as NiBabel file


H_masked = np.where(mask, H, np.nan)
H2 = H.copy()    
plt.imshow(H[:,:,30])
plt.colorbar()

H_hist = H.flatten()
H_hist = H_hist[~np.isnan(H_hist)] 
H_hist = H_hist[H_hist != 0]


plt.hist(H_hist, bins = 30, density = True, alpha = 0.8)
plt.xlim([0,1.6])
plt.xlabel('H', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Distribucion exponentes de Hurst en Control', fontsize=14)
plt.grid()
plt.show()


H_masked = np.where(mask, H, np.nan)
plt.imshow(H_masked[:,:,30])
plt.colorbar()
plt.imshow(subject[:,:,30,1])
plt.colorbar()
plt.imshow(mask[:,:,30])
plt.colorbar()

        