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
# Compute per-ROI lesion load: for each lesion mask in a directory,     #
# the percentage overlap between the (binarised) lesion and each atlas  #
# ROI, relative to that ROI's total volume. Results are written to a    #
# single TSV with subjects as rows and ROIs as columns.                 #
#                                                                       #
# usage: get_lesion_loads.py --lesion_dir /path/to/lesions               #
#                            --atlas Schaefer2018-400                    #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/19.                                              #
#                                                                       #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import argparse
import csv
import glob
import os

import nibabel as nib
import numpy as np
import progress.bar


#################################
#      LOGGING UTILITIES        #
#################################

def log_msg(_string):
    '''
    logging function printing date, scriptname & input string to stdout
    '''
    import datetime, sys
    print(f'{datetime.date.today().strftime("%a %B %d %H:%M:%S %Z %Y")} {str(os.path.basename(sys.argv[0]))}: {str(_string)}')


#################################
#       PARSE PARAMETERS        #
#################################

parser = argparse.ArgumentParser(
    description='Compute per-ROI lesion load (percentage overlap of each lesion '
                'mask with each atlas ROI, relative to ROI volume).'
)
parser.add_argument('--data_path', type=str, default='/mnt/h/RT/data',
                    help='Path to the data directory containing the ATLAS folder and '
                         'the lesion mask folder')
parser.add_argument('--lesion_dir', type=str, default='LESIONS',
                    help='Name of the folder (inside data_path) containing nifti lesion '
                         'masks (*.nii.gz) (default: LESIONS)')
parser.add_argument('--atlas', type=str, default='Schaefer2018-400',
                    help='Atlas name; looks up {data_path}/ATLAS/{atlas}.nii.gz and '
                         '{data_path}/ATLAS/{atlas}_areas.txt (default: Schaefer2018-400)')
parser.add_argument('--output', type=str, default=None,
                    help='Output TSV path (default: {data_path}/{atlas}_lesion_loads.tsv)')
args = parser.parse_args()

lesion_dir  = os.path.join(args.data_path, args.lesion_dir)
output_path = args.output or os.path.join(args.data_path, f'{args.atlas}_lesion_loads.tsv')

log_msg(f"| START | Computing lesion loads")
log_msg(f"| UPDATE | Lesion directory: {lesion_dir}")
log_msg(f"| UPDATE | Atlas: {args.atlas}")


#################################
#          LOAD ATLAS           #
#################################

atlas_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
areas_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt')

atlas_img  = nib.load(atlas_path)
atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int32)

# Parcellation convention: integer value v in the atlas NIfTI corresponds to
# roi_names[v-1] (1-indexed, matching the atlas areas file row order).
areas_raw = np.genfromtxt(areas_path, dtype=str, delimiter='\t')
roi_names = areas_raw[1:, 0].tolist()
n_rois    = len(roi_names)

roi_volumes = {idx: int(np.sum(atlas_data == idx)) for idx in range(1, n_rois + 1)}

log_msg(f"| UPDATE | Atlas loaded: {atlas_path} ({n_rois} ROIs)")


#################################
#        LOAD LESIONS           #
#################################

lesion_files = sorted(glob.glob(os.path.join(lesion_dir, '*.nii.gz')))
if not lesion_files:
    raise SystemExit(f'No nifti lesion masks found in {lesion_dir}')

log_msg(f"| UPDATE | Found {len(lesion_files)} lesion masks")


#################################
#      COMPUTE LESION LOADS     #
#################################

results = {}
with progress.bar.Bar('| COMPUTING LESION LOADS |', max=len(lesion_files)) as bar:
    for lf in lesion_files:
        subject = os.path.basename(lf)
        subject = subject.split('.')[0]
        les_data = np.asarray(nib.load(lf).dataobj)
        les_mask = les_data > 0
        patient_results = {}
        for idx in range(1, n_rois + 1):
            vol = roi_volumes[idx]
            if vol == 0:
                patient_results[idx] = 0.0
                continue
            overlap = np.sum(les_mask & (atlas_data == idx))
            patient_results[idx] = round(100.0 * overlap / vol, 2)
        results[subject] = patient_results
        bar.next()

log_msg(f"| UPDATE | Lesion loads computed for {len(results)} subjects")


#################################
#          SAVE OUTPUT          #
#################################

with open(output_path, 'w', newline='') as fh:
    writer = csv.writer(fh, delimiter='\t')
    writer.writerow(['Participant_id'] + roi_names)
    for subject in sorted(results.keys()):
        row = [subject] + [f'{results[subject][idx]:.2f}' for idx in range(1, n_rois + 1)]
        writer.writerow(row)

log_msg(f"| UPDATE | Lesion loads saved -> {output_path}")
log_msg(f"| FINISHED | All outputs saved")
