#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                            RT MODELLING                               #
# 
# The following script performs MRI processing of raw patient MRI data  #
#                                                                       #
# performed stpes include:                                              #
#  1. coregistration of native image to MNI152                          #
#     1.1 brain extraction                                              #
#     1.2 enantiomorphic normalization                                  #
#     1.3 transformation to MNI152                                      #
#     1.4 application of transformation matrix to raw image and mask    #
#  2.                                                                   #
#  3.                                                                   #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# requirements:                                                         #
#   - LeAPP docker container (dockerhub: patrikneuro/leapp:processing)  #
#                                                                       #
#                                                                       #
# last update: 2026/05/29.                                              #
#                                                                       #
#                                                                       #
#########################################################################





import SimpleITK as sitk, os, progress.bar

FILES = os.listdir('./LESIONS')
with progress.bar.Bar('Processing', max=len(FILES)) as bar:
  for f in FILES:
    img = sitk.ReadImage('./LESIONS/' + f)
    img_ras = sitk.DICOMOrient(img, 'RAS')
    sitk.WriteImage(img_ras, './LESIONS_RAS/' + f)
    bar.next()


hd-bet -i input_ras.nii.gz -o test_ras.nii.gz --save_bet_mask



# 41148603 >> corrupted image
# LESION WITHOUT BRAIN IMAGES:
# missing file: 40880998_DWI_smask.nii
# missing file: 41654673_T2_smask.nii
# BRAIN IMAGES WITHOUT LESION MASKS:
# missing file: 21040673_T2.nii.gz

# BRAINS_RAS	LESIONS_RAS
# 03066619_T2.nii.gz	03066619_smask.nii (no modality)
# 3008481_T2.nii.gz	3008481_T2mask.nii
# 40116713.nii.gz (no modality)	40116713_T2_smask.nii
# 40350786_T2.nii.gz	40350786_T2smask.nii
# 40662637_T2.nii.gz	40662637_T2mask.nii
# 41250622_T2.nii.gz	41250622_smask.nii (no modality)
# 41574866_T2.nii.gz	41574866_T2smask.nii
# 41708477_DWI.nii.gz	41708477_T2_smask.nii
# 41716240_T2.nii.gz	41716240_T2mask.nii
# 98003835_T2.nii.gz	98003835_smask.nii (no modality)
# 04036564_T2.nii.gz	04036564_t2_smask.nii (case only)




FILES=$( ls LESIONS_RAS)
for f in ${FILES}; do
  id="$( cut -d '_' -f 1 <<< "$f" )"
  mv LESIONS_RAS/${f} LESIONS_RAS/${id}_lesion.nii
done

FILES=$( ls BRAINS_RAS)
for f in ${FILES}; do
  id="$( cut -d '_' -f 1 <<< "$f" )"
  mv BRAINS_RAS/${f} BRAINS_RAS/${id}_brain.nii.gz
done


FILES=$( ls LESIONS_RAS)
i=0
for f in ${FILES}; do
    fslmaths LESIONS_RAS/$f -mul 1 LESIONS_RAS/$f
    i=$((i+1))
    echo "${i}/309"
done


# RERUN TOMORROW:

hd-bet -i BRAINS_RAS -o BRAINS_BET


docker run --rm -it -v /mnt/h/RT/data/:/data -e Steps="vbt" -e SubID="test" -e Image="sub-test.nii.gz" -e Sigma=4 -e CostMask="sub-test_lesion.nii.gz" patrikneuro/leapp:processing bash

FILES=$(ls BRAINS_BET)
export SmoothingFactor=10 # adjusting for tumour topology and sub mm resolution

for f in ${FILES}; do
  id="$( cut -d '_' -f 1 <<< "$f" )"
  ${LEAPP_STRUCTDIR}/Scripts/VirtualBrainTransplant.sh \
      --workingdir=/data \
      --subject=test \
      --session=test-01 \
      --in=BRAINS_BET/${f} \
      --costmask=LESIONS_RAS/${id}_lesion.nii.gz \
      --out="/data/BRAINS_VBT/${f%.nii.gz}"
done


FILES=$(ls)
i=0
for f in ${FILES}; do
  fslmaths ${f} -bin ${f}
  i=$((i+1))
  echo "${i}/309"
done






RegisterBrain() {
  # Register native brain image and lesion mask to MNI152 space using cost function masking in native space
  # after enantiomophric normalization
  # ${1} = PATIENT ID | e.g. 01005027

  if [ ! -d "/data/LESIONS_INV" ]; then
    mkdir "/data/LESIONS_INV"
  fi
  if [ ! -d "/data/LESIONS_MNI" ]; then
    mkdir "/data/LESIONS_MNI"
  fi
  if [ ! -d "/data/BRAINS_MNI" ]; then
    mkdir "/data/BRAINS_MNI"
  fi
  ${FSLDIR}/bin/fslmaths \
    "/data/LESIONS_RAS/${1}_lesion.nii.gz" \
    -binv \
    "/data/LESIONS_INV/${1}_invert.nii.gz"
    
  ${FSLDIR}/bin/flirt \
    -cost "mutualinfo" \
    -in "/data/ATLAS/MNI152_icbm_T1_1mm_brain.nii.gz" \
    -refweight "/data/LESIONS_INV/${1}_invert.nii.gz" \
    -ref "/data/BRAINS_VBT/${1}_brain.nii.gz" \
    -omat "/tmp/${1}MNI2ref.mat"

  ${FSLDIR}/bin/convert_xfm \
      "/tmp/${1}MNI2ref.mat"  \
      -inverse \
      -omat "/tmp/${1}ref2MNI.mat" 
  # ---- apply transformation to lesioned brain ---- #
  ${FSLDIR}/bin/flirt -applyxfm -usesqform \
      -init "/tmp/${1}ref2MNI.mat"  \
      -in "/data/BRAINS_BET/${1}_brain.nii.gz" \
      -ref "/data/ATLAS/MNI152_icbm_T1_1mm_brain.nii.gz" \
      -out "/data/BRAINS_MNI/sub-${1}_brain.nii.gz"
  # ---- apply transformation to lesion mask ---- #
  ${FSLDIR}/bin/flirt -applyxfm -usesqform \
      -init "/tmp/${1}ref2MNI.mat"  \
      -in "/data/LESIONS_RAS/${1}_lesion.nii.gz" \
      -ref "/data/ATLAS/MNI152_icbm_T1_1mm_brain.nii.gz" \
      -out "/data/LESIONS_MNI/sub-${1}_lesion.nii.gz"

  # ---- ensure binary mask ---- #
  ${FSLDIR}/bin/fslmaths \
      "/data/LESIONS_MNI/sub-${1}_lesion.nii.gz" \
      -bin "/data/LESIONS_MNI/sub-${1}_lesion.nii.gz"
  }



MAX_JOBS=10
for f in ${FILES}; do
    id="$( cut -d '_' -f 1 <<< "$f" )"
    RegisterBrain "$( cut -d '_' -f 1 <<< "$f" )" &
        while [ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]; do
            wait -n 2>/dev/null || true
        done
done
wait  # wait for remaining jobs to finish


for f in ${FILES}; do
  fslstats LESIONS_MNI/${f} -M
done



MAX_JOBS=20
for f in ${FILES}; do
  fslmaths LESIONS_MNI/${f} -mul ATLAS/MNI152_icbm_T1_1mm_mask.nii.gz LESIONS_MASKED/${f} &
  while [ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]; do
      wait -n 2>/dev/null || true
  done
done

for f in $FILES; do
  echo "${f}: $(fslstats LESIONS_MASKED/${f} -M )"
  fslmaths LesionAggregate.nii.gz -add LESIONS_MASKED/${f} LesionAggregate.nii.gz
done

