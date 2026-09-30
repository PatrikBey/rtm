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
# Maps the per-ROI behaviour_degree and cooccurrence_degree columns     #
# already saved in SBMNOCONV20K's roi_block_assignments_{task}.csv      #
# (run_noconv.py, weighted degree of each node within the behaviour-    #
# weighted layer / lesion-cooccurrence layer of the observed multilayer #
# graph -- not a new computation, just onto the atlas volume) into two  #
# NIfTI volumes per task, one per layer.                                #
#                                                                       #
# Outputs (per task x fit_suffix):                                      #
#   {atlas}_behaviour_degree_{task}{fit_suffix}.nii.gz                  #
#   {atlas}_cooccurrence_degree_{task}{fit_suffix}.nii.gz               #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import csv
import argparse
import numpy as np
import nibabel as nib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sbm'))
from utils import log_msg


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Map SBMNOCONV20K per-ROI behaviour/cooccurrence degree to NIfTI.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+', default=['foreperiod_hexg_tau', 'gonogo_hexg_tau'],
                  help='Observed (SBMNOCONV20K) task names to map')
args.add_argument('--fit_suffix', type=str, default='_singleflip',
                  help='Suffix of the observed SBMNOCONV20K run directory (default: _singleflip)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args = args.parse_args()

out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


#################################
#      LOAD ATLAS TEMPLATE      #
#################################

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)


#################################
#     PER-TASK COMPUTATION      #
#################################

for task in args.tasks:
    fit_dir  = os.path.join(args.data_path, 'SBMNOCONV20K', f'SBM_{args.atlas}_{task}{args.fit_suffix}')
    roi_path = os.path.join(fit_dir, f'roi_block_assignments_{task}.csv')
    with open(roi_path, newline='') as fh:
        rows = list(csv.DictReader(fh))

    roi_index = np.array([int(r['roi_index']) for r in rows])
    beh_deg   = np.array([float(r['behaviour_degree'])    for r in rows])
    occ_deg   = np.array([float(r['cooccurrence_degree']) for r in rows])
    order     = np.argsort(roi_index)
    roi_index, beh_deg, occ_deg = roi_index[order], beh_deg[order], occ_deg[order]

    beh_by_index = np.zeros(int(roi_index.max()) + 1)
    occ_by_index = np.zeros(int(roi_index.max()) + 1)
    beh_by_index[roi_index] = beh_deg
    occ_by_index[roi_index] = occ_deg

    valid_mask = (atlas_data > 0) & (atlas_data <= len(beh_by_index))

    beh_data = np.zeros_like(atlas_data, dtype=np.float32)
    beh_data[valid_mask] = beh_by_index[atlas_data[valid_mask] - 1]
    beh_path = os.path.join(out_dir, f'{args.atlas}_behaviour_degree_{task}{args.fit_suffix}.nii.gz')
    nib.save(nib.Nifti1Image(beh_data, atlas_img.affine, atlas_img.header), beh_path)
    log_msg(f"| UPDATE | {task}: behaviour-layer degree NIfTI saved "
            f"(range [{beh_deg.min():.2f}, {beh_deg.max():.2f}]) -> {beh_path}")

    occ_data = np.zeros_like(atlas_data, dtype=np.float32)
    occ_data[valid_mask] = occ_by_index[atlas_data[valid_mask] - 1]
    occ_path = os.path.join(out_dir, f'{args.atlas}_cooccurrence_degree_{task}{args.fit_suffix}.nii.gz')
    nib.save(nib.Nifti1Image(occ_data, atlas_img.affine, atlas_img.header), occ_path)
    log_msg(f"| UPDATE | {task}: cooccurrence-layer degree NIfTI saved "
            f"(range [{occ_deg.min():.2f}, {occ_deg.max():.2f}]) -> {occ_path}")

log_msg(f"| FINISHED | Degree NIfTIs saved -> {out_dir}")
