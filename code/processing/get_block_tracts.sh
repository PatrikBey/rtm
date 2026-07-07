#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                GRAPH REPRESENTATION OF INTELLIGENCE                   #
#                                                                       #
# The following script extracts tractogram subsets for all SBM          #
# based community blocks.                                               #
#                                                                       #
#                                                                       #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/06.                                              #
#                                                                       #
#                                                                       #
#########################################################################


Path="/mnt/h/RT/data/RESULTS/split_threshold"

TASKS="${Path}"/SBM*


for t in ${TASKS}; do
    echo "Processing ${t}"
    FILES="${t}"/block_niftis/*tau_regions.nii.gz
    for f in ${FILES}; do
        echo "Processing ${f}"
        python3 get_tracts.py --input_image "${f}" --tractogram "${Path}/dTOR_1M.tck"
    done
done





# python3 /mnt/h/GitHub/rtm/code/get_tracts.py --input_image combi.nii.gz --tractogram "dTOR_1M.tck"




# import numpy as np
# from nibabel.streamlines import load, save, Tractogram

# a = load("combi_subset.tck")
# b = load("block8subset.tck")

# def key(s):
#     return s.round(3).tobytes()

# b_keys = {key(s) for s in b.streamlines}
# kept = [s for s in a.streamlines if key(s) not in b_keys]

# save(Tractogram(kept, affine_to_rasmm=a.tractogram.affine_to_rasmm), "combi_minus_block8.tck")



# a = load("combi_minus_block8.tck")
# b = load("block9subset.tck")

# def key(s):
#     return s.round(3).tobytes()

# b_keys = {key(s) for s in b.streamlines}
# kept = [s for s in a.streamlines if key(s) not in b_keys]

# save(Tractogram(kept, affine_to_rasmm=a.tractogram.affine_to_rasmm), "combi_minus_block9.tck")


