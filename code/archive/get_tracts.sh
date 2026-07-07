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
# The following script processed volume maps to extract white matter    #
# coonectivity maps.                                                    #
#                                                                       #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# requirements:                                                         #
#   - LeAPP docker container (dockerhub: patrikneuro/leapp:processing)  #
#                                                                       #
#                                                                       #
# last update: 2026/07/05.                                              #
#                                                                       #
#                                                                       #
#########################################################################

# usage: get_tracts.sh <input_image.nii.gz>

set -e

Input="$1"

Base=$(basename "${Input}")
Base="${Base%.nii.gz}"
Base="${Base%.nii}"
OutDir=$(dirname "${Input}")

BinMask="${OutDir}/${Base}_bin.nii.gz"
OutsideMask="${OutDir}/${Base}_outside_mask.nii.gz"
FullMask="${OutDir}/${Base}_full_mask.nii.gz"
Output="${OutDir}/${Base}_subset.tck"

fslmaths "${Input}" -bin "${BinMask}"

mrcalc "${BinMask}" 0 -eq "${OutsideMask}" -datatype int

fslmaths "${Input}" -mul 0 -add 1 -bin "${FullMask}"

tckedit -force -quiet \
    "${TEMPLATEDIR}/Tractograms/dTOR_2m_tractogram.tck" \
    -include "${Input}" \
    -exclude "${OutsideMask}" \
    -ends_only \
    -mask "${FullMask}" \
    "${Output}"

rm -f "${BinMask}" "${OutsideMask}" "${FullMask}"


fslmaths test.nii.gz -bin /data/test_bin.nii.gz


tck2connectome -force -symmetric -zero_diagonal -quiet \
    "${TEMPLATEDIR}/Tractograms/dTOR_2m_tractogram.tck" \
    "/data/test2.nii.gz"  \
    -out_assignments "/data/assignments2.txt" \
    "/data/tmp.tsv"

# ---- 2. extract tract subset ---- #
connectome2tck -force -exclusive -quiet -files single \
    "${TEMPLATEDIR}/Tractograms/dTOR_2m_tractogram.tck" \
    "/data/assignments2.txt" \
    "/data/tracts_subset2.tck" -nodes "1,2,2,3,4,5,6,7,8,9,10,11,12,13"
