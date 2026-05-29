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




hd - bet 04054774_T2.nii brain.nii.gz


i=0
for f in ${FILES}; do
    fslreorient2std VOLUMES/$f BRAINS/${f%.nii.gz}.nii.gz
    i=$((i+1))
    echo "${i}/309"
done


214/309
Image Exception : #63 :: No image files match: /opt/miniforge/data/standard/MNI152_T1_2mm_brain
terminate called after throwing an instance of 'std::runtime_error'
  what():  No image files match: /opt/miniforge/data/standard/MNI152_T1_2mm_brain
Aborted (core dumped)

f="41590175_T1.nii"