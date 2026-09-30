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
# The following script performs disconnectome extraction using ARISE    #
#                                                                       #
# performed steps include:                                              #
#  1. loop over all lesion masks                                        #
#  2. call docker arise:0.4 with atlas hemisphere split AAL3            #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# requirements:                                                         #
#   - ARISE docker container (dockerhub: patrikneuro/arise:0.4)         #
#                                                                       #
#                                                                       #
# last update: 2026/05/29.                                              #
#                                                                       #
#                                                                       #
#########################################################################



Path="/data/patrik/RT/DATA"
# Path="/mnt/h/RT/data"
FILES=$(ls ${Path}/LESIONS)


MAX_JOBS=20

for f in $FILES; do
    docker run --rm --cpus="3" -v ${Path}/LESIONS:/data -v ${Path}/DISCONNECTOMES:/output -e Seed="${f}" -e Atlas="Schaefer2018-400" -e OutDir="/output" -e tck_keep="False" patrikneuro/arise:0.5 &
    while [ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]; do
      wait -n 2>/dev/null || true
  done
done





###########################
#                         #
#  SBM BLOCK CONNECTOMES  # 
#                         # 
###########################



# tck="${TEMPLATEDIR}/Tractograms/dTOR_2m_tractogram.tck"
# SCORES="Foreperiod_Long_tau GoNoGo_tau SATO_Accuracy_tau"


# # ---- 1. get full connectome ---- #
# for score in ${SCORES}; do
#   tck2connectome -force -symmetric -zero_diagonal -quiet -scale_invnodevol \
#     "${tck}" \
#     "/data/SBM_Schaefer2018-400_${score}_singleflip/block_niftis/${score}_parcellation.nii.gz"  \
#     "/data/SBM_Schaefer2018-400_${score}_singleflip/Lvl0_block_connectome_${score}.tsv"
# done


######################

Path="/data/patrik/RT/DATA"
# Path="/mnt/h/RT/data"
FILES=$(ls ${Path}/LESIONS)


MAX_JOBS=20

for f in $FILES; do
    docker run --rm --cpus="3" -v ${Path}/LESIONS:/data -v ${Path}/DISCONNECTOMES:/output -e Seed="${f}" -e Atlas="Schaefer2018-400" -e OutDir="/output" -e tck_keep="False" patrikneuro/arise:0.5 &
    while [ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]; do
      wait -n 2>/dev/null || true
  done
done



Path="/data/patrik/RT/RTM"
FILES=$(ls ${Path}/LESIONS)
mkdir -p "${Path}/DISCONNECTOMES"

MAX_JOBS=20
for f in $FILES; do
  if [[ ${f} == sub-*.nii.gz ]]; then
    docker run --rm --cpus="3" --user "$(id -u):$(id -g)" -v ${Path}/LESIONS:/data -v ${Path}/DISCONNECTOMES:/output -e Seed="${f}" -e Atlas="Schaefer2018-1000" -e OutDir="/output" -e tck_keep="False" -e Tracts="tracts_1M.tck" patrikneuro/arise:0.5 &
  fi
    while [ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]; do
      wait -n 2>/dev/null || true
    done
done



# VOXEL ATLAS BASED DISCONNECTOMES
#
# NOTE (2026-09-25): a single subject's VoxelAtlas container previously used
# 70-108GB RSS, because add_lut_disconnectome round-tripped the whole dense
# matrix through numpy (genfromtxt + astype(str)). Running MAX_JOBS=20 of
# those concurrently OOM-killed the whole machine (128GB RAM) on 2026-09-24.
# Fixed at the source in ARISE 0.6 (streaming label-prepend instead of the
# numpy round-trip) and switched output to OutFormat=npz: voxel-resolution
# disconnectomes are >99.9% sparse, so npz stores only the nonzero
# (row, col, value) triplets instead of the dense matrix -- ~500-8000x
# smaller on disk, and measured at ~300MB peak RSS per container (3 CPUs),
# so no per-container --memory cap is needed and concurrency can go back up.

Path="/data/patrik/RT/RTM"
FILES=$(ls ${Path}/LESIONS)
mkdir -p "${Path}/DISCONNECTOMES"

MAX_JOBS=20
for f in $FILES; do
  if [[ ${f} == sub-*.nii.gz ]]; then
    docker run --rm --cpus="3" --user "$(id -u):$(id -g)" -v ${Path}/LESIONS:/data -v ${Path}/DISCONNECTOMES:/output -e Seed="${f}" -e Atlas="VoxelAtlas_4mm_cortex, VoxelAtlas_4mm_brain,VoxelAtlas_6mm_cortex, VoxelAtlas_6mm_brain,VoxelAtlas_8mm_cortex, VoxelAtlas_8mm_brain" -e OutDir="/output" -e tck_keep="False" -e Tracts="tracts_1M.tck" -e OutFormat="npz" patrikneuro/arise:0.6 &
  fi
    while [ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]; do
      wait -n 2>/dev/null || true
    done
done

