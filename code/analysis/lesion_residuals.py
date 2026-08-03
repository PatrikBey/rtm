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
# Implements Approach 2 from doc/ROISelection.md: residualize the       #
# behaviour-layer signal against the lesion-occurrence signal.          #
#                                                                       #
# The predictor and outcome come from two independently-fit models:     #
#   x = lesion-only base fit's block score (SBMBASE, run_base.py)       #
#       - consistency-weighted mean cooccurrence_degree per block,      #
#         broadcast to member ROIs                                      #
#   y = observed multilayer fit's block score (SBMFITTING, run.py)      #
#       - consistency-weighted mean behaviour_degree per block,         #
#         broadcast to member ROIs                                      #
#                                                                       #
# An earlier version used cooccurrence_degree and behaviour_degree from #
# the SAME multilayer fit, but those are computed from edges of the     #
# same constructed graph, and behaviour_weight is itself built as       #
# lesion-connectivity-weighted-by-behaviour-score - so the two were not #
# independent by construction and the residual mostly reflected that    #
# shared origin rather than genuine joint signal. Using the lesion-only #
# base fit's own (behaviour-blind) block structure for x removes that   #
# circularity: x now comes from a model that never saw behaviour at     #
# all, while y comes from the model that saw both layers jointly.       #
# Fitting one linear trend of y on x across all ROIs and keeping the    #
# residual isolates the part of each ROI's multilayer-fit behavioural   #
# signal that isn't explained by the independent lesion-only model's    #
# own view of that ROI - ROIs with a large positive residual carry      #
# behavioural information beyond pure lesion occurrence.                #
#                                                                       #
# Uses only the observed (unpermuted) fits - this is a cross-model      #
# regression across ROIs, not a permutation-referenced test like        #
# Approach 1.                                                           #
#                                                                       #
# Outputs (per task):                                                   #
#   lesion_residuals_{task}.tsv    - roi_name, lesion_only_score,       #
#                                     multilayer_behaviour_score,        #
#                                     predicted, residual, residual_z    #
#   {atlas}_residual_{task}.nii.gz - per-ROI residual, i.e. the         #
#                                     behaviour-beyond-lesion signal,    #
#                                     for spatial pattern inspection     #
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

args = argparse.ArgumentParser(description='Residualize multilayer behaviour-block score against the lesion-only fit\'s block score (ROISelection.md Approach 2).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+', default=['Foreperiod_Long_tau', 'GoNoGo_tau'],
                  help='Behaviour scores to evaluate (default: both available tasks)')
args.add_argument('--level', type=int, default=0, help='Hierarchy level to use for both fits (default: 0)')
args.add_argument('--fit_suffix', type=str, default='_singleflip',
                  help='Suffix of the observed multilayer SBMFITTING run directory (default: _singleflip)')
args.add_argument('--base_suffix', type=str, default='_base_singleflip',
                  help='Suffix of the lesion-only SBMBASE run directory (default: _base_singleflip)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args = args.parse_args()

out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

log_msg(f"| START | Lesion-residual ROI signal (Approach 2) | tasks: {args.tasks}")


#################################
#      LOAD ATLAS TEMPLATE      #
#################################

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)


#################################
#   PER-ROI BLOCK SCORE LOADER  #
#################################

def load_roi_block_score(fit_dir, task, level):
    '''Per-ROI block score: block id per ROI (roi_block_assignments) joined
    with that block's score (SBM_block_scores_lvl{level}).'''
    roi_path = os.path.join(fit_dir, f'roi_block_assignments_{task}.csv')
    with open(roi_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    roi_index = np.array([int(r['roi_index']) for r in rows])
    roi_name  = np.array([r['roi_name'] for r in rows])
    roi_block = np.array([int(r[f'level_{level}']) for r in rows])
    order     = np.argsort(roi_index)
    roi_index, roi_name, roi_block = roi_index[order], roi_name[order], roi_block[order]

    block_path = os.path.join(fit_dir, f'SBM_block_scores_lvl{level}_{task}.csv')
    with open(block_path, newline='') as fh:
        block_score = {int(r['block']): float(r['score']) for r in csv.DictReader(fh)}

    roi_score = np.array([block_score[b] for b in roi_block])
    return roi_index, roi_name, roi_score


#################################
#     PER-TASK COMPUTATION      #
#################################

for task in args.tasks:
    fit_dir  = os.path.join(args.data_path, 'SBMFITTING', f'SBM_{args.atlas}_{task}{args.fit_suffix}')
    base_dir = os.path.join(args.data_path, 'SBMBASE', f'SBM_{args.atlas}_{task}{args.base_suffix}')

    _,         multi_roi_name, y = load_roi_block_score(fit_dir, task, args.level)
    roi_index, base_roi_name,  x = load_roi_block_score(base_dir, task, args.level)

    if not np.array_equal(multi_roi_name, base_roi_name):
        raise ValueError(f"[{task}] ROI order differs between multilayer and lesion-only fits")
    roi_name = multi_roi_name

    # ---- single global regression across ROIs: multilayer score ~ lesion-only score ---- #
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual  = y - predicted
    resid_std = residual.std()
    residual_z = (residual - residual.mean()) / resid_std if resid_std > 0 else np.zeros_like(residual)
    r = np.corrcoef(x, y)[0, 1]

    log_msg(f"| UPDATE | {task}: slope={slope:.6f} intercept={intercept:.4f} r={r:.4f}")
    log_msg(f"| UPDATE | {task}: top residual ROI = {roi_name[np.argmax(residual)]} "
            f"(z={residual_z[np.argmax(residual)]:.2f})")

    # ---- per-ROI table ---- #
    table_path = os.path.join(out_dir, f'lesion_residuals_{task}.tsv')
    with open(table_path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['roi_name', 'lesion_only_score', 'multilayer_behaviour_score',
                         'predicted', 'residual', 'residual_z'])
        for i in range(len(roi_name)):
            writer.writerow([roi_name[i], round(float(x[i]), 6), round(float(y[i]), 6),
                             round(float(predicted[i]), 6), round(float(residual[i]), 6),
                             round(float(residual_z[i]), 6)])
    log_msg(f"| UPDATE | {task}: residual table saved → {table_path}")

    # ---- residual NIfTI: spatial pattern of behaviour beyond lesion occurrence ---- #
    residual_by_index = np.zeros(int(roi_index.max()) + 1)
    residual_by_index[roi_index] = residual

    residual_data = np.zeros_like(atlas_data, dtype=np.float32)
    valid_mask    = (atlas_data > 0) & (atlas_data <= len(residual_by_index))
    residual_data[valid_mask] = residual_by_index[atlas_data[valid_mask] - 1]

    nii_path = os.path.join(out_dir, f'{args.atlas}_residual_{task}.nii.gz')
    nib.save(nib.Nifti1Image(residual_data, atlas_img.affine, atlas_img.header), nii_path)
    log_msg(f"| UPDATE | {task}: residual NIfTI saved → {nii_path}")

log_msg(f"| FINISHED | Lesion-residual ROI signal saved → {out_dir}")
