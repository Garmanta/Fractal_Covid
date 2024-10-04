import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import fathon
from fathon import fathonUtils as fu
from tqdm import tqdm

# %%
#Obtaining control patients
n = 51
for i in range(49,n):

    #path = [f'home/sphyrna/storage/subjects/tests/complete_synbold/sub-{i:02d}/func/sub-{i:02d}_fmri_in_mni_warp.nii.gz']
    path = [f'sub-{i:02d}/func/sub-{i:02d}_rest_in_mni_warp.nii.gz']
    path_mask = ['MNI152_T1_2mm_brain_mask.nii.gz']
    print(path)
    
    preproc_subject = nib.load(os.path.join(path[0]))
    subject = preproc_subject.get_fdata()
    mask = nib.load(os.path.join(path_mask[0])).get_fdata()
    x,y,z,t = subject.shape


    windows = fu.linRangeByStep(10,80)    
    F = np.zeros([x,y,z,windows.size] , dtype = np.float16)
    H = np.zeros([x,y,z] , dtype = np.float16)
    H_intercept = np.zeros([x,y,z] , dtype = np.float16)
    
    total_voxels = np.sum(mask)
    print("---------------------------------------")
    print("|")
    print("| Processing DFA image for subject " + str(i))
    print("|")
    print("---------------------------------------")
    pbar = tqdm(total=total_voxels, desc='Processing Voxels', dynamic_ncols=False)


    for xi in range(0 , x):
        for yi in range(0 , y):
            for zi in range(0 , z):
                if mask[xi,yi,zi]:
                    DFA = fathon.DFA(fu.toAggregated(subject[xi,yi,zi,:].astype(np.float64)))
                    _, F[xi,yi,zi,:] = DFA.computeFlucVec(windows, revSeg=False, polOrd = 1)
                    H[xi,yi,zi], H_intercept[xi,yi,zi] = DFA.fitFlucVec()
                    pbar.update(1)
  
    print("")
    print("Finished!")
    pbar.close()
         
    np.save(f'../fractal/control/sub-{i:02d}/sub-{i:02d}_H.npy', H)
    np.save(f'../fractal/control/sub-{i:02d}/sub-{i:02d}_H_intercept.npy', H_intercept)
    np.save(f'../fractal/control/sub-{i:02d}/sub-{i:02d}_F.npy', F)
    np.save(f'../fractal/control/sub-{i:02d}/sub-{i:02d}_windows.npy', windows)
    plt.imshow(H[:,:,30])
    plt.savefig(f'../fractal/treatment/sub-{i:02d}/sub-{i:02d}_H.png')
    
    img = nib.Nifti1Image(H.astype(np.float64), preproc_subject.affine)  # Save axis for data (just identity)
    img.header.get_xyzt_units()
    img.to_filename(os.path.join(f'../fractal/control/sub-{i:02d}/sub-{i:02d}_H.nii.gz'))  # Save as NiBabel file
    



