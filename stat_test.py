import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import stats
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


Yeo17_image = nib.load(os.path.join('Yeo2011_17Networks_N1000.split_components.FSL_MNI152_2mm.nii.gz'))
Yeo17 = Yeo17_image.get_fdata()



# VisCent = [1,2,58,59]
# VisPeri = [3,4,5,60,61,62]
# SomMotA = [6,63]
# SomMotB = [7,8,9,10,64,65,66,67]
# DorsAttnA = [11,12,13,68,69,70]
# DorsAttnB = [14,15,16,17,71,72,73,74]
# SalVentAttnA = [18,19,20,21,22,75,76,77,78,79,80]
# SalVentAttnB = [23,24,25,26,27,28,81,82,83,84,85,86,87]
# LimbicA = [29,88]
# LimbicB = [30,89]
# ContA = [31,32,33,34,35,36,90,91,92,93,94]
# ContB = [37,38,39,40,41,42,95,96,97,98,99]
# ContC = [43,44,100,101]
# DefaultA = [45,46,47,48,102,103,104,105,106]
# DefaultB = [49,50,51,52,53,107,108,109,110]
# DefaultC = [54,55,56,111,112,113]
# TempPar = [57,114]


#VisCent, VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB, SalVentAttnA, SalVentAttnB, 
#LimbicA, LimbicB, ContA, ContB, ContA, DefaultA, DefaultB, DefaultC, TempPar
# Define the new cluster groups
clusters = {
    1: [1, 2, 58, 59],
    2: [3, 4, 5, 60, 61, 62],
    3: [6, 63],
    4: [7, 8, 9, 10, 64, 65, 66, 67],
    5: [11, 12, 13, 68, 69, 70],
    6: [14, 15, 16, 17, 71, 72, 73, 74],
    7: [18, 19, 20, 21, 22, 75, 76, 77, 78, 79, 80],
    8: [23, 24, 25, 26, 27, 28, 81, 82, 83, 84, 85, 86, 87],
    9: [29, 88],
    10: [30, 89],
    11: [31, 32, 33, 34, 35, 36, 90, 91, 92, 93, 94],
    12: [37, 38, 39, 40, 41, 42, 95, 96, 97, 98, 99],
    13: [43, 44, 100, 101],
    14: [45, 46, 47, 48, 102, 103, 104, 105, 106],
    15: [49, 50, 51, 52, 53, 107, 108, 109, 110],
    16: [54, 55, 56, 111, 112, 113],
    17: [57, 114]
}

# Create a mapping from old cluster IDs to new ones
id_map = {}
for new_id, ids in clusters.items():
    for old_id in ids:
        id_map[old_id] = new_id

map_func = np.vectorize(id_map.get, otypes=[np.float64], cache=False)
new_Yeo17 = map_func(Yeo17, 0)  # Using 0 for any missing old_id values
# Replace elements in the matrix according to the mapping

#%%


img = nib.Nifti1Image(new_Yeo17.astype(np.float64), Yeo17_image.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('new_Yeo17'))  # Save as NiBabel file


#%%
def statistical_results(atlas, image, n_x, n_y):
    results = []
    # Loop through each cluster ID in new_matrix
    for cluster_id in range(1, int(np.max(atlas)+1)) :
        
        mask = atlas == cluster_id
    
        # Extract trials for X and Y
        x_data = image[:,:,:,0:n_x][mask]
        y_data = image[:,:,:,n_x:n_y][mask]
    
        # Calculate mean and variance
        x_mean = np.mean(x_data)
        x_var = np.var(x_data, ddof = 1)
        y_mean = np.mean(y_data)
        y_var = np.var(y_data, ddof=1)
    
    
        t_stat = (x_mean - y_mean) / np.sqrt(x_var / n_x + y_var / n_y)
        df = ((x_var / n_x + y_var / n_y)**2 /((x_var / n_x)**2 / (n_x - 1) + (y_var / n_y)**2 / (n_y - 1)))
        p_value = 2 * t.cdf(-np.abs(t_stat), df)
        
        p_value_corrected = min(p_value * int(np.max(atlas)+1), 1)
        
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = ''
        
        # Perform t-test
       # t_stat, p_value = stats.ttest_ind(x_data, y_data, equal_var=False)  # Welch's t-test
    
        # Collect results
        results.append({
            'cluster_id': cluster_id,
            'x_mean': x_mean,
            'y_mean': y_mean,
            'x_variance': x_var,
            'y_variance': y_var,
            't_stat': t_stat,
            'p_value': p_value,
            'p_value_corrected':p_value_corrected,
            'significance':significance
        })
    
    # Optionally, print or save results
    for result in results:
        print(f"Cluster {result['cluster_id']}:")
        print(f"  X Mean: {result['x_mean']}, Y Mean: {result['y_mean']}")
        print(f"  X Variance: {result['x_variance']}, Y Variance: {result['y_variance']}")
        print(f"  T-Statistic: {result['t_stat']}, P-value: {result['p_value']}")
        print(f"  Corrected P-value: {result['p_value_corrected']}, Signifance: {result['significance']}\n")
        
    return results


def extract_significant_clusters(results):
    # Use filter with lambda to get significant clusters with p-value < 0.05
    significant_clusters = list(filter(lambda result: result['p_value'] < 0.05, results))
    return significant_clusters

def create_significant_clusters_mask(Yeo17, significant_clusters):
    # Extract the significant cluster IDs
    significant_ids = [cluster['cluster_id'] for cluster in significant_clusters]
    
    # Initialize the mask with 0.0 and set dtype to float64
    mask = np.zeros(Yeo17.shape, dtype=np.float64)
    
    # Set the mask to 1.0 for each significant cluster ID
    for cluster_id in significant_ids:
        mask[Yeo17 == cluster_id] = 1.0

    return mask


def update_p_values(Yeo17_results, adjusted_p_values):
    # Ensure adjusted_p_values is a numpy array
    adjusted_p_values = np.array(adjusted_p_values)
    
    # Make a copy of Yeo17_results with updated p-values
    updated_results = []
    for original, new_p_value in zip(Yeo17_results, adjusted_p_values):
        updated_dict = original.copy()  # Copy original dictionary
        updated_dict['p_value'] = new_p_value
        updated_results.append(updated_dict)
    
    return updated_results

#%%

complete_H = nib.load(os.path.join('complete_H.nii')).get_fdata()
n_x = 49
n_y = 51
new_Yeo17_results = statistical_results(new_Yeo17, complete_H, n_x, n_y)
Yeo17_results = statistical_results(Yeo17, complete_H, n_x, n_y)

significant_clusters = extract_significant_clusters(Yeo17_results)
Yeo17_significant_mask = create_significant_clusters_mask(Yeo17, significant_clusters)

img = nib.Nifti1Image(Yeo17_significant_mask.astype(np.float64), Yeo17_image.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('Yeo17_significant_mask'))  # Save as NiBabel file

p_values =  np.array([result['p_value'] for result in Yeo17_results])
bool_result, adjusted_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

Yeo17_results_adjusted = update_p_values(Yeo17_results, adjusted_p_values)

significant_clusters = extract_significant_clusters(Yeo17_results_adjusted)
Yeo17_significant_mask = create_significant_clusters_mask(Yeo17, significant_clusters)

img = nib.Nifti1Image(Yeo17_significant_mask.astype(np.float64), Yeo17_image.affine)  # Save axis for data (just identity)
img.header.get_xyzt_units()
img.to_filename(os.path.join('Yeo17_significant_adjusted_mask'))  # Save as NiBabel file


for result in Yeo17_results:
    print(f"Cluster {result['cluster_id']}:")
    print(f"  Corrected P-value: {result['p_value_corrected']}, Signifance: {result['significance']}\n")

for result in significant_clusters:
    print(f"Cluster {result['cluster_id']}:")
    print(f"  Corrected P-value: {result['p_value_corrected']}, Signifance: {result['significance']}\n")



