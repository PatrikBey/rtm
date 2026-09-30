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
# archive/summarize_null.py adapted for run_null_noconv.py's            #
# SBMNULL_NOCONV output (same roi_block_assignments/blockzscores.nii.gz #
# file formats, just a different null_dir naming pattern that includes  #
# a fit_suffix). Summarizes the 100-permutation null into:              #
#   1. permutation_zscores.tsv - roi x permutation table of level-{n}   #
#      ROI block z-scores, concatenated across all perm_XXXXX dirs.     #
#   2. {atlas}_lvl{n}_blockzscores_{score}_permutation_mean.nii.gz -     #
#      the per-voxel mean across all permutations' level-{n} block      #
#      z-score NIfTIs.                                                  #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import sys
import csv
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sbm'))
from utils import log_msg


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Summarize SBMNULL_NOCONV (run_null_noconv.py) permutation outputs.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--score', type=str, default='foreperiod_hexg_tau', help='Behaviour score analyzed')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--fit_suffix', type=str, default='_singleflip',
                  help='Fit-suffix on the SBMNULL_NOCONV null directory name (default: _singleflip)')
args.add_argument('--level', type=int, default=0, help='Hierarchy level to summarize (default: 0)')
args = args.parse_args()

null_dir = os.path.join(args.data_path, 'SBMNULL_NOCONV', f'SBM_{args.atlas}_{args.score}_NULL{args.fit_suffix}')

log_msg(f"| START | Null model summary")
log_msg(f"| UPDATE | Null directory: {null_dir}")

perm_dirs = sorted(d for d in os.listdir(null_dir) if d.startswith('perm_') and
                   os.path.isdir(os.path.join(null_dir, d)))
log_msg(f"| UPDATE | Permutations found: {len(perm_dirs)}")


#################################
#   ROI x PERMUTATION ZSCORES   #
#################################

roi_names   = None
roi_zscores = []
zscore_col  = f'zscore_{args.level}'

for perm in perm_dirs:
    roi_path = os.path.join(null_dir, perm, f'roi_block_assignments_{args.score}.csv')
    with open(roi_path, newline='') as fh:
        reader = csv.DictReader(fh)
        names   = []
        zscores = []
        for row in reader:
            names.append(row['roi_name'])
            zscores.append(float(row[zscore_col]))
    if roi_names is None:
        roi_names = names
    elif names != roi_names:
        raise ValueError(f"ROI order/names in {roi_path} do not match the first permutation")
    roi_zscores.append(zscores)

# rows = ROIs, columns = permutations
zscore_table = np.array(roi_zscores, dtype=np.float64).T

tsv_path = os.path.join(null_dir, f'permutation_zscores_lvl{args.level}.tsv')
with open(tsv_path, 'w', newline='') as fh:
    writer = csv.writer(fh, delimiter='\t')
    writer.writerow(['roi_name'] + perm_dirs)
    for roi_name, row in zip(roi_names, zscore_table):
        writer.writerow([roi_name] + [round(float(v), 6) for v in row])
log_msg(f"| UPDATE | ROI x permutation z-score table saved "
        f"({zscore_table.shape[0]} ROIs x {zscore_table.shape[1]} permutations, level {args.level}) -> {tsv_path}")


#################################
#   MEAN BLOCK Z-SCORE NIFTI    #
#################################

sum_data   = None
ref_img    = None
n_included = 0

for perm in perm_dirs:
    nii_path = os.path.join(null_dir, perm, f'{args.atlas}_lvl{args.level}_blockzscores_{args.score}.nii.gz')
    if not os.path.isfile(nii_path):
        log_msg(f"| WARNING | Missing {nii_path}, skipping")
        continue
    img  = nib.load(nii_path)
    data = np.asarray(img.dataobj, dtype=np.float64)
    if sum_data is None:
        sum_data = data
        ref_img  = img
    else:
        sum_data += data
    n_included += 1

mean_data = (sum_data / n_included).astype(np.float32)
mean_path = os.path.join(null_dir, f'{args.atlas}_lvl{args.level}_blockzscores_{args.score}_permutation_mean.nii.gz')
mean_img  = nib.Nifti1Image(mean_data, ref_img.affine, ref_img.header)
nib.save(mean_img, mean_path)
log_msg(f"| UPDATE | Per-voxel mean block z-score NIfTI saved "
        f"({n_included} permutations, level {args.level}) -> {mean_path}")

log_msg(f"| FINISHED | Null model summary saved -> {null_dir}")
