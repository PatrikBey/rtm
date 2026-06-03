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



import numpy, os

loc = numpy.genfromtxt('/mnt/h/RT/data/LocationCategories.csv', dtype=str, delimiter=',')

loads = numpy.genfromtxt('/mnt/h/RT/data/Patient_Lesion_Loads.csv', dtype=str, delimiter=',')

rois = loads[1,1:].tolist()
locations = loads[0,1:].tolist()



subjects_loc = loc[1:,0].tolist()
match_count = 0
mismatch_count = 0
for sub in subjects_loc:
    if f'sub-{sub}' not in loads[:,0]:
        print(f'{sub} is missing in loads')
    else:
        initial_loc = loc[loc[:,0] == sub, 1][0]
        if initial_loc == '1':
            initial_loc = 'FL'
        elif initial_loc == '2':
            initial_loc = 'FR'
        elif initial_loc == '3':
            initial_loc = 'NFL'
        elif initial_loc == '4':
            initial_loc = 'NFR'
        idx = numpy.where(loads[:,0] == f'sub-{sub}')[0]
        tmp_loads = loads[idx,1:].astype(float)
        tmp_loads = numpy.where(tmp_loads > 5, tmp_loads, 0)  # Set values <= 5% to 0
        if numpy.sum(tmp_loads) == 0:
            print(f'{sub} has no significant lesion load after thresholding')
            continue
        nz_idx = numpy.where(tmp_loads > 0)[1]
        tmp_rois = [rois[i] for i in nz_idx]
        tmp_locs = [locations[i] for i in nz_idx]
        frontal_ratio = tmp_locs.count('Frontal') / len(tmp_locs)
        left = sum(1 for i in tmp_rois if i.endswith('_L'))
        right = sum(1 for i in tmp_rois if i.endswith('_R'))
        if left > right:
            if frontal_ratio > 0.1:
                sub_loc = 'FL'
            else:
                sub_loc = 'NFL'
        elif right > left:
            if frontal_ratio > 0.25:
                sub_loc = 'FR'
            else:
                sub_loc = 'NFR'
        elif left == right == 0:
            sub_loc = 'Unknown'
        if sub_loc == initial_loc:
            print(f'{sub} location matches: {sub_loc}')
            match_count +=1
        else:
            mismatch_count +=1
            print(f'{sub} location mismatch: initial {initial_loc} vs computed {sub_loc}')



sub='41626009'


initial_loc = loc[loc[:,0] == sub, 1][0]
if initial_loc == '1':
    initial_loc = 'FL'
elif initial_loc == '2':
    initial_loc = 'FR'
elif initial_loc == '3':
    initial_loc = 'NFL'
elif initial_loc == '4':
    initial_loc = 'NFR'
idx = numpy.where(loads[:,0] == f'sub-{sub}')[0]
tmp_loads = loads[idx,1:].astype(float)
nz_idx = numpy.where(tmp_loads > 0)[1]
tmp_rois = [rois[i] for i in nz_idx]
tmp_locs = [locations[i] for i in nz_idx]
frontal_ratio = tmp_locs.count('Frontal') / len(tmp_locs)
left = sum(1 for i in tmp_rois if i.endswith('_L'))
right = sum(1 for i in tmp_rois if i.endswith('_R'))
if left > right:
    if frontal_ratio > 0.1:
        sub_loc = 'FL'
    else:
        sub_loc = 'NFL'
elif right > left:
    if frontal_ratio > 0.25:
        sub_loc = 'FR'
    else:
        sub_loc = 'NFR'
elif left == right == 0:
    sub_loc = 'Unknown'
if sub_loc == initial_loc:
    print(f'{sub} location matches: {sub_loc}')
    match_count +=1
else:
    mismatch_count +=1
    print(f'{sub} location mismatch: initial {initial_loc} vs computed {sub_loc}')



mismatch_subjects = ['40456931',
'41624966',
'41626009',
'41224272',
'41052308',
'41269681',
'40751153',
'M444701',
'41266250',
'41708477',
'41713022',
'41723290',
'41727659',
'41725457',
'21024366',
'21071881',
'41557575',
'40993138',
'21128955',
'21133577',
'41269709',
'21167759',
'40703075',
'21210301',
'21207608',
'21217066',
'21257849',
'21280654',
'21284689']

for sub in mismatch_subjects:
    initial_loc = loc[loc[:,0] == sub, 1][0]
    idx = numpy.where(loads[:,0] == f'sub-{sub}')[0]
    tmp_loads = loads[idx,1:].astype(float)
    nz_idx = numpy.where(tmp_loads > 0)[1]
    tmp_rois = [rois[i] for i in nz_idx]
    tmp_locs = [locations[i] for i in nz_idx]
    print(f'{sub} computed locations: {numpy.unique(tmp_locs, return_counts=True)[0]} with counts {numpy.unique(tmp_locs, return_counts=True)[1]}')


minor_lesions=['41646032',
'41708477',
'21096887',
'21322447']


for sub in minor_lesions:
    initial_loc = loc[loc[:,0] == sub, 1][0]
    idx = numpy.where(loads[:,0] == f'sub-{sub}')[0]
    tmp_loads = loads[idx,1:].astype(float)
    nz_idx = numpy.where(tmp_loads > 0)[1]
    tmp_rois = [rois[i] for i in nz_idx]
    tmp_locs = [locations[i] for i in nz_idx]
    print(f'{sub} initial location: {initial_loc}')
    print(f'computed locations: {numpy.unique(tmp_locs, return_counts=True)[0]} with counts {numpy.unique(tmp_locs, return_counts=True)[1]}')




sub='40455361'

initial_loc = loc[loc[:,0] == sub, 1][0]
idx = numpy.where(loads[:,0] == f'sub-{sub}')[0]
tmp_loads = loads[idx,1:].astype(float)
nz_idx = numpy.where(tmp_loads > 5)[1]
tmp_rois = [rois[i] for i in nz_idx]
tmp_roi_loads = [tmp_loads[0,i] for i in nz_idx]
tmp_locs = [locations[i] for i in nz_idx]
print(f'{sub} initial location: {initial_loc}')
print(f'computed locations: {numpy.unique(tmp_locs, return_counts=True)[0]} with counts {numpy.unique(tmp_locs, return_counts=True)[1]}')




# 40456931 computed locations: ['Frontal' 'Parietal'] with counts [2 3]
# 41624966 computed locations: ['Cerebellum' 'Frontal' 'Insula' 'Medial temporal' 'Occipital' 'Parietal'
#  'Subcortical' 'Temporal' 'Temporo-occipital' 'Thalamus'] with counts [3 8 1 3 5 4 3 6 1 9]
# 41626009 computed locations: ['Frontal' 'Insula' 'Medial temporal' 'Subcortical' 'Temporal'
#  'Temporo-occipital' 'Thalamus'] with counts [5 1 3 2 6 1 3]
# 41224272 computed locations: ['Frontal' 'Insula' 'Medial temporal' 'Temporal' 'Temporo-occipital'] with counts [4 1 2 5 1]
# 41052308 computed locations: ['Cingulate' 'Frontal' 'Insula' 'Occipital' 'Parietal' 'Subcortical'
#  'Temporal'] with counts [2 5 1 5 5 1 6]
# 41269681 computed locations: ['Cerebellum' 'Cingulate' 'Frontal' 'Insula' 'Medial temporal' 'Midbrain'
#  'Occipital' 'Parietal' 'Subcortical' 'Temporal' 'Temporo-occipital'
#  'Thalamus'] with counts [ 3  2 10  1  3  3  6  5  3  6  1 14]
# 40751153 computed locations: ['Frontal' 'Insula' 'Parietal' 'Subcortical' 'Temporal'] with counts [4 1 2 1 1]
# M444701 computed locations: ['Frontal' 'Insula' 'Subcortical' 'Temporal' 'Thalamus'] with counts [5 1 3 1 3]
# 41266250 computed locations: ['Frontal' 'Insula' 'Medial temporal' 'Midbrain' 'Occipital' 'Parietal'
#  'Subcortical' 'Temporal' 'Temporo-occipital' 'Thalamus'] with counts [ 8  1  3  2  2  3  3  6  1 13]
# 41708477 computed locations: ['Frontal' 'Insula' 'Parietal'] with counts [2 1 1]
# 41713022 computed locations: ['Frontal' 'Parietal' 'Temporal'] with counts [1 2 2]
# 41723290 computed locations: ['Frontal' 'Insula' 'Medial temporal' 'Occipital' 'Parietal' 'Subcortical'
#  'Temporal' 'Temporo-occipital'] with counts [8 1 1 2 2 1 6 1]
# 41727659 computed locations: ['Frontal' 'Insula' 'Subcortical' 'Temporal' 'Thalamus'] with counts [3 1 2 1 2]
# 41725457 computed locations: ['Cingulate' 'Frontal' 'Insula' 'Medial temporal' 'Occipital' 'Parietal'
#  'Subcortical' 'Temporal' 'Thalamus'] with counts [2 3 1 1 5 6 1 3 2]
# 21024366 computed locations: ['Cingulate' 'Frontal' 'Insula' 'Medial temporal' 'Subcortical' 'Temporal'
#  'Thalamus'] with counts [ 6 11  1  3  4  3 15]
# 21071881 computed locations: ['Cingulate' 'Frontal' 'Insula' 'Medial temporal' 'Subcortical' 'Temporal'
#  'Thalamus'] with counts [ 4 10  1  1  3  1  3]
# 41557575 computed locations: ['Cingulate' 'Frontal' 'Occipital' 'Parietal'] with counts [1 3 1 4]
# 40993138 computed locations: ['Frontal' 'Insula' 'Medial temporal' 'Subcortical' 'Temporal'
#  'Temporo-occipital' 'Thalamus'] with counts [9 1 3 3 6 1 1]
# 21128955 computed locations: ['Cerebellum' 'Cingulate' 'Frontal' 'Insula' 'Medial temporal' 'Midbrain'
#  'Occipital' 'Parietal' 'Subcortical' 'Temporal' 'Temporo-occipital'
#  'Thalamus'] with counts [ 2  1 10  1  3  1  2  3  3  6  1  9]
# 21133577 computed locations: ['Cingulate' 'Frontal' 'Insula' 'Occipital' 'Parietal' 'Subcortical'
#  'Temporal'] with counts [2 4 1 4 6 1 3]
# 41269709 computed locations: ['Frontal' 'Insula' 'Medial temporal' 'Occipital' 'Parietal' 'Subcortical'
#  'Temporal' 'Temporo-occipital' 'Thalamus'] with counts [3 1 3 2 4 3 5 1 6]
# 21167759 computed locations: ['Cingulate' 'Frontal' 'Insula' 'Medial temporal' 'Occipital' 'Parietal'
#  'Subcortical' 'Temporal' 'Thalamus'] with counts [ 4  5  1  1  7  7  2  3 12]
# 40703075 computed locations: ['Frontal' 'Insula' 'Medial temporal' 'Subcortical' 'Temporal'
#  'Temporo-occipital'] with counts [7 1 3 2 6 1]
# 21210301 computed locations: ['Frontal' 'Insula' 'Parietal' 'Temporal'] with counts [7 1 2 6]
# 21207608 computed locations: ['Frontal' 'Insula' 'Occipital' 'Parietal' 'Temporal'] with counts [2 1 1 5 3]
# 21217066 computed locations: ['Frontal' 'Insula' 'Medial temporal' 'Parietal' 'Temporal'] with counts [1 1 1 2 3]
# 21257849 computed locations: ['Brainstem' 'Cerebellum' 'Cingulate' 'Frontal' 'Insula' 'Medial temporal'
#  'Midbrain' 'Occipital' 'Subcortical' 'Temporal' 'Temporo-occipital'
#  'Thalamus'] with counts [ 1 10  6 28  2  6  8 13  6  9  2 24]
# 21280654 computed locations: ['Cerebellum' 'Frontal' 'Insula' 'Medial temporal' 'Midbrain' 'Parietal'
#  'Subcortical' 'Temporal' 'Temporo-occipital'] with counts [ 2 10  1  3  1  1  3  6  1]
# 21284689 computed locations: ['Frontal' 'Insula' 'Subcortical' 'Thalamus'] with counts [1 1 3 2]