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



# Path="/data/patrik/RT/DATA"
Path="/mnt/h/RT/data"
FILES=$(ls ${Path}/LESIONS)


MAX_JOBS=6

for f in $FILES; do
    docker run --rm --cpus="3" -v ${Path}/LESIONS:/data -v ${Path}/DISCONNECTOMES:/output -e Seed="${f}" -e Atlas="HCP-MMP1" -e OutDir="/output" -e tck_keep="False" patrikneuro/arise:0.5 &
    while [ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]; do
      wait -n 2>/dev/null || true
  done
done