#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                            RT MODELLING                               #
#                                                                       #
# The following script generates synthetic behvaiours for use in        #
# downstream SBM community detection                                    #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/06/04.                                              #
#                                                                       #
#                                                                       #
#########################################################################




import numpy, os, nibabel, progress.bar

atlas_img = nibabel.load('/mnt/h/RT/data/ATLAS/AAL3v1_MNI.nii.gz')
atlas_data = atlas_img.get_fdata().astype(numpy.int32)

lesion_files = [f for f in os.listdir('/mnt/h/RT/data/LESIONS_MNI/') if f.endswith('.nii.gz')]

# V1_idx = 23
# V1_mask = numpy.where(atlas_data == V1_idx, 1, 0)

# nibabel.save(nibabel.Nifti1Image(V1_mask.astype(numpy.float32), atlas_img.affine), '/mnt/h/RT/data/ATLAS/V1_L_mask.nii.gz')

mask = nibabel.load('/mnt/h/RT/data/synth/RF_mask.nii.gz')
mask = mask.get_fdata().astype(numpy.int32)

# la = nilearn.image.resample_to_img(mask, atlas_img, interpolation='nearest')
# nibabel.save(nibabel.Nifti1Image(la.get_fdata().astype(numpy.float32), atlas_img.affine), '/mnt/h/RT/data/ATLAS/RF_mask.nii.gz')


synth = numpy.zeros((len(lesion_files), 2), dtype=object)
with progress.bar.Bar('| GEN SYNTH SCORES|', max=len(lesion_files)) as bar:
    for i, file in enumerate(lesion_files):
        lesion_img = nibabel.load(os.path.join('/mnt/h/RT/data/LESIONS_MNI/', file))
        lesion_data = lesion_img.get_fdata().astype(numpy.int32)
        # Compute synthetic score as overlap with V1 mask
        overlap = numpy.sum(lesion_data * mask)
        synth[i, 0] = file.split('_')[0]
        synth[i, 1] = overlap / mask.sum() + numpy.random.normal(0, 0.1)  # Add noise
        bar.next()

numpy.savetxt('/mnt/h/RT/data/synth_scores.csv', synth, delimiter=',', fmt='%s', header='ID,Synth_Score')