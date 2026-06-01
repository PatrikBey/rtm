import nibabel as nib
import numpy as np
import os
import glob
import progress.bar

atlas_img = nib.load('/mnt/h/RT/data/ATLAS/AAL3v1_MNI.nii.gz')
atlas_data = atlas_img.get_fdata().astype(np.int32)

roi_names = {}
# with open('/mnt/h/RT/data/ATLAS/AAL3v1_coords_lobes.txt') as f:
#     for line in f:
#         cols = line.strip().split()
#         if cols:
#             name = cols[0]
#             idx = len(roi_names) + 1
#             roi_names[idx] = name

lut = np.genfromtxt('/mnt/h/RT/data/ATLAS/AAL3v1_coords_lobes.txt', dtype=str, delimiter='\t')
rois = lut[1:,0].tolist()
for i in range(1,lut.shape[0]):
    roi_names[i] = rois[i-1]



all_roi_indices = sorted(roi_names.keys())

roi_volumes = {idx: np.sum(atlas_data == idx) for idx in all_roi_indices}

lesion_files = sorted(glob.glob('/mnt/h/RT/data/LESIONS_MNI/sub-*_lesion.nii.gz'))

results = {}
with progress.bar.Bar('| COMPUTING LESION LOADS |', max=len(lesion_files)) as bar:
    for lf in lesion_files:
        patient = os.path.basename(lf).replace('.nii.gz', '')
        les_data = nib.load(lf).get_fdata()
        les_mask = (les_data > 0).astype(np.int32)
        patient_results = {}
        for idx in all_roi_indices:
            roi_mask = (atlas_data == idx).astype(np.int32)
            overlap = np.sum(les_mask * roi_mask)
            vol = roi_volumes[idx]
            patient_results[idx] = round(100.0 * overlap / vol, 2) if vol > 0 else 0.0
        results[patient] = patient_results
        bar.next()


header = 'Participant_id\t' + '\t'.join(roi_names[idx] for idx in all_roi_indices)
# print(header)
# for patient in sorted(results.keys()):
#     row = [patient] + [f'{results[patient][idx]:.2f}' for idx in all_roi_indices]
#     print('\t'.join(row))

# --- save results to file --- #
output_file = '/mnt/h/RT/data/lesion_loads.tsv'
with open(output_file, 'w') as f:
    f.write(header + '\n')
    for patient in sorted(results.keys()):
        row = [patient] + [f'{results[patient][idx]:.2f}' for idx in all_roi_indices]
        f.write('\t'.join(row) + '\n')



# import nilearn.image

# template = nib.load('/mnt/h/RT/data/ATLAS/MNI152_icbm_T1_1mm_brain.nii.gz')

# mni = nilearn.image.resample_img(atlas_img, target_affine=template.affine, target_shape=template.shape, interpolation='nearest', copy=True, order='F', clip=True, fill_value=0, force_resample=True, copy_header=True)


# nib.save(mni, '/mnt/h/RT/data/ATLAS/AAL3v1_MNI.nii.gz')
# mni_data = mni.get_fdata().astype(np.int32)