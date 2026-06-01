#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                     TRANSLATIONAL TRACTOGRAPHY                        #
#                                                                       #
# This script performs node based mapping of behavioural values.        #
# It uses the AAL3v1 atlas in combination with lesion loads and lesion  #
# induced disruption networks generated from ARISE.                     #
#                                                                       #
# Author: Patrik Bey, patrik.bey@ucl.ac.uk                              #
#                                                                       #
# last update: 2026/05/24.                                              #
#                                                                       #
#                                                                       #
#########################################################################



# MASKS=$(ls masks_binary_2mm)
# Path=$(pwd)
# OUTDIR=$Path/DISCONNECTOMES

# MAX_JOBS=4

# for mask in ${MASKS}; do
#     docker run --rm --cpus=5 -v ${Path}/masks_binary_2mm:/data -v ${OUTDIR}:/output \
#         -e Seed=$mask -e OutDir=/output -e tck_keep=False arise:0.4 &
#     # if 4 jobs are running, wait for one to finish before continuing
#     while [ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]; do
#         wait -n 2>/dev/null || true
#     done
# done
# wait  # wait for remaining jobs to finish


# ---- load libraries ---- #
import os, numpy, matplotlib.pyplot as plt, progress.bar


# ---- set paths ---- #
DATA_DIR = '/mnt/h/RT/data'
OUT_DIR = '/mnt/h/RT/plots'



# ---- load data ---- #

atlas = numpy.genfromtxt(os.path.join(DATA_DIR,'atlas', 'AAL3v1_coords_lobes.txt'), delimiter = '\t', dtype = str)
rois = atlas[1:,0].tolist()
coords = atlas[1:,1:3].astype(float)
lobes = atlas[1:,-1].tolist()

network = numpy.genfromtxt(os.path.join(DATA_DIR,'networks', 'sub-40456931.tsv'), delimiter = '\t', dtype = float)
plt.imshow(numpy.where(network>0,network,numpy.nan), cmap = 'plasma')
plt.show()

degrees = numpy.sum(numpy.where(network>0,1,0), axis = 0)

plt.scatter(coords[:,0], coords[:,1], s = degrees*20)



# ---- map RT to network ---- #

