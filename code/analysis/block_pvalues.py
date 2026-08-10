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
# Block-level variant of ROISelection.md Approach 1, using a max-       #
# statistic permutation test rather than per-ROI z-scores.              #
#                                                                       #
# roi_pvalues.py tests every ROI, but a block's score is broadcast      #
# identically to every ROI it contains, so testing 400 ROIs is really   #
# only testing the handful of distinct block values (pseudo-            #
# replication). Testing at the block level directly avoids that, but    #
# block ids aren't comparable across fits - each SBM fit (observed or   #
# permuted) discovers its own independent partition, so "block 3" in    #
# permutation 47 has no relation to "block 3" in the observed fit, and  #
# forcing a permutation's null value onto the observed block's specific #
# ROI membership would be a form of double-dipping (that membership was #
# chosen by fitting on the very behaviour scores being tested).         #
#                                                                       #
# Instead this uses a max-statistic test: for each permutation, take    #
# the single highest block score among that permutation's own freely-   #
# discovered blocks (already saved in                                   #
# perm_XXXXX/SBM_block_scores_lvl{level}_{task}.csv, "score" column -   #
# the consistency-weighted mean behaviour_degree per block, computed    #
# before any across-block z-scoring). Across all permutations this      #
# gives a null distribution of "the most coherent block chance alone    #
# can produce" for that task. Every observed block is then compared     #
# against this same null_max distribution. Because the null is already #
# built from the maximum over each permutation's blocks, it controls    #
# the family-wise error rate across blocks by construction - no         #
# separate multiple-comparisons correction (e.g. BH-FDR) is needed on   #
# top, and none is applied here.                                        #
#                                                                       #
# The raw "score" is used throughout rather than the z-scored version:  #
# the z-score just re-normalizes across blocks within one fit, which    #
# isn't needed for this permutation-based comparison and would only add #
# fit-dependent noise (block count differs from fit to fit).            #
#                                                                       #
# Because blocks are fit independently per task, results are written    #
# one table per task rather than a combined block x task table.         #
#                                                                       #
# Outputs (per task):                                                   #
#   block_pvalues_lvl{level}_{task}.tsv                                 #
#   {atlas}_lvl{level}_significant_blocks_{task}.nii.gz                 #
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

args = argparse.ArgumentParser(description='Block-level max-statistic permutation test (ROISelection.md Approach 1, block variant).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+',
                  default=['Foreperiod_Long_tau', 'GoNoGo_tau', 'SATO_Accuracy_tau'],
                  help='Behaviour scores to evaluate (default: all three available tasks)')
args.add_argument('--level', type=int, default=0, help='Hierarchy level to test (default: 0)')
args.add_argument('--alpha', type=float, default=0.05, help='Significance threshold (default: 0.05; '
                  'already family-wise controlled by the max-statistic, no further correction applied)')
args.add_argument('--fit_suffix', type=str, default='_singleflip',
                  help='Suffix of the observed SBMFITTING run directory (default: _singleflip)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args = args.parse_args()

out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

log_msg(f"| START | Block max-statistic p-values lvl{args.level} | tasks: {args.tasks}")


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
    fit_dir  = os.path.join(args.data_path, 'SBMFITTING', f'SBM_{args.atlas}_{task}{args.fit_suffix}')
    null_dir = os.path.join(args.data_path, 'SBMNULL', f'SBM_{args.atlas}_{task}_NULL')

    # ---- observed lvl-{level} block assignment + raw block scores ---- #
    roi_path = os.path.join(fit_dir, f'roi_block_assignments_{task}.csv')
    with open(roi_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    roi_index = np.array([int(r['roi_index']) for r in rows])
    roi_name  = np.array([r['roi_name'] for r in rows])
    roi_block = np.array([int(r[f'level_{args.level}']) for r in rows])
    order     = np.argsort(roi_index)
    roi_index, roi_name, roi_block = roi_index[order], roi_name[order], roi_block[order]

    block_path = os.path.join(fit_dir, f'SBM_block_scores_lvl{args.level}_{task}.csv')
    with open(block_path, newline='') as fh:
        block_rows = list(csv.DictReader(fh))
    block_ids     = np.array([int(r['block']) for r in block_rows])
    block_n_nodes = {int(r['block']): int(r['n_nodes']) for r in block_rows}
    block_score   = {int(r['block']): float(r['score']) for r in block_rows}

    # ---- null_max: highest raw block score per permutation, own free partition ---- #
    perm_dirs = sorted(d for d in os.listdir(null_dir) if d.startswith('perm_') and
                       os.path.isdir(os.path.join(null_dir, d)))
    null_max = []
    for perm in perm_dirs:
        perm_block_path = os.path.join(null_dir, perm, f'SBM_block_scores_lvl{args.level}_{task}.csv')
        with open(perm_block_path, newline='') as fh:
            perm_scores = [float(r['score']) for r in csv.DictReader(fh)]
        null_max.append(max(perm_scores))
    null_max = np.array(null_max, dtype=np.float64)
    n_perm   = len(null_max)

    log_msg(f"| UPDATE | {task}: {n_perm} permutations, {len(block_ids)} observed blocks at level {args.level}")
    log_msg(f"| UPDATE | {task}: null_max range [{null_max.min():.4f}, {null_max.max():.4f}], "
            f"mean {null_max.mean():.4f}")

    # ---- test every observed block against the same null_max distribution ---- #
    p_value = np.array([(np.sum(null_max >= block_score[b]) + 1) / (n_perm + 1) for b in block_ids])
    n_sig   = int(np.sum(p_value < args.alpha))
    log_msg(f"| UPDATE | {task}: {n_sig}/{len(block_ids)} blocks significant at p < {args.alpha}")

    # ---- block-level table ---- #
    table_path = os.path.join(out_dir, f'block_pvalues_lvl{args.level}_{task}.tsv')
    with open(table_path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['block', 'n_nodes', 'observed_score', 'p_value'])
        for i, b in enumerate(block_ids):
            writer.writerow([int(b), block_n_nodes[b], round(float(block_score[b]), 6),
                             round(float(p_value[i]), 6)])
    log_msg(f"| UPDATE | {task}: block p-value table saved → {table_path}")

    # ---- significant blocks x member ROI names ---- #
    sig_block_ids = set(block_ids[p_value < args.alpha])
    sig_p_value   = {int(b): float(p_value[i]) for i, b in enumerate(block_ids)}

    n_sig_rois = int(np.isin(roi_block, list(sig_block_ids)).sum())
    rois_path  = os.path.join(out_dir, f'block_significant_rois_lvl{args.level}_{task}.tsv')
    with open(rois_path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['block', 'p_value', 'roi_name'])
        for b in sorted(sig_block_ids):
            for name in roi_name[roi_block == b]:
                writer.writerow([int(b), round(sig_p_value[b], 6), name])
    log_msg(f"| UPDATE | {task}: {len(sig_block_ids)} significant block(s), {n_sig_rois} ROIs → {rois_path}")

    # ---- significant-block NIfTI ---- #
    sig_score_by_roi = np.array([block_score[b] if b in sig_block_ids else 0.0 for b in roi_block])
    sig_by_index     = np.zeros(int(roi_index.max()) + 1)
    sig_by_index[roi_index] = sig_score_by_roi

    sig_data   = np.zeros_like(atlas_data, dtype=np.float32)
    valid_mask = (atlas_data > 0) & (atlas_data <= len(sig_by_index))
    sig_data[valid_mask] = sig_by_index[atlas_data[valid_mask] - 1]

    sig_path = os.path.join(out_dir, f'{args.atlas}_lvl{args.level}_significant_blocks_{task}.nii.gz')
    nib.save(nib.Nifti1Image(sig_data, atlas_img.affine, atlas_img.header), sig_path)
    log_msg(f"| UPDATE | {task}: significant-block NIfTI saved → {sig_path}")

    # ---- full-brain p-value NIfTI: every block, not just significant ones ---- #
    p_value_by_block = {int(b): float(p_value[i]) for i, b in enumerate(block_ids)}
    p_by_roi    = np.array([p_value_by_block[b] for b in roi_block])
    p_by_index  = np.zeros(int(roi_index.max()) + 1)
    p_by_index[roi_index] = p_by_roi

    p_data = np.zeros_like(atlas_data, dtype=np.float32)
    p_data[valid_mask] = p_by_index[atlas_data[valid_mask] - 1]

    p_path = os.path.join(out_dir, f'{args.atlas}_lvl{args.level}_block_pvalues_{task}.nii.gz')
    nib.save(nib.Nifti1Image(p_data, atlas_img.affine, atlas_img.header), p_path)
    log_msg(f"| UPDATE | {task}: full-brain block p-value NIfTI saved → {p_path}")

log_msg(f"| FINISHED | Block p-values saved → {out_dir}")
