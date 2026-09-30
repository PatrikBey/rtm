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
# block_pvalues_noconv.py adapted for the lesion-only (cooccurrence-    #
# only, no behaviour) SBMBASE20K fits (run_base_noconv.py). Same max-   #
# statistic permutation logic (see block_pvalues_noconv.py header):     #
# block ids aren't comparable across independently-fit partitions, so   #
# each null permutation's single highest block statistic is pooled      #
# into a null distribution, and every observed block is compared        #
# against that same null_max distribution.                              #
#                                                                       #
# The lesion-only model has no behavioural covariate, so there is no    #
# matched behaviour-permutation null for it (unlike SBMNOCONV20K vs.    #
# SBMNULL_NOCONV). Per explicit instruction, this instead reuses the    #
# existing SBMNULL_NOCONV runs (the JOINT two-layer model, behaviour    #
# permuted) as an approximate reference: for each null permutation, the #
# per-ROI cooccurrence_degree + level_{level}/consistency_{level}       #
# columns already saved in roi_block_assignments_{task}.csv are         #
# regrouped into blocks and scored with the SAME formula the observed   #
# lesion-only fit uses (consistency-weighted mean cooccurrence_degree   #
# per block) -- since SBMNULL_NOCONV never saved a cooccurrence-based   #
# block-score CSV directly (its own 'score' column is behaviour-based). #
#                                                                       #
# Caveat (accepted, not fixed here): the null's blocks come from a      #
# JOINT two-layer partition (nudged by permuted-behaviour noise), while #
# the observed fit's blocks come from a single-layer, cooccurrence-only #
# partition -- a structural mismatch between null and observed model,   #
# not a like-for-like matched null. Treat results as approximate.       #
#                                                                       #
# Outputs (per task x level):                                           #
#   block_pvalues_base_noconv_lvl{level}_{task}{fit_suffix}.tsv          #
#   block_significant_rois_base_noconv_lvl{level}_{task}{fit_suffix}.tsv #
#   {atlas}_lvl{level}_teststat_base_noconv_{task}{fit_suffix}.nii.gz    #
#   {atlas}_lvl{level}_invp_base_noconv_{task}{fit_suffix}.nii.gz        #
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

args = argparse.ArgumentParser(description='Block-level max-statistic permutation test: '
                               'SBMBASE20K (lesion-only) observed fits vs. SBMNULL_NOCONV '
                               '(cooccurrence-degree reconstructed per block).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+', default=['foreperiod_hexg_tau', 'gonogo_hexg_tau'])
args.add_argument('--levels', type=int, nargs='+', default=[0, 1],
                  help='Hierarchy levels to test (default: 0 1 -- level 2 is a single trivial block)')
args.add_argument('--alpha', type=float, default=0.05, help='Significance threshold (default: 0.05; '
                  'already family-wise controlled by the max-statistic, no further correction applied)')
args.add_argument('--fit_suffix', type=str, default='_singleflip')
args.add_argument('--null_fit_suffix', type=str, default='_singleflip')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args = args.parse_args()

out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

log_msg(f"| START | Block max-statistic p-values (lesion-only base_noconv vs SBMNULL_NOCONV) | tasks: {args.tasks}")


#################################
#      LOAD ATLAS TEMPLATE      #
#################################

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)


#################################
#  RECONSTRUCT NULL BLOCK SCORE #
#################################

def null_max_cooccurrence_scores(null_dir, task, level):
    '''
    For each permutation in null_dir, regroups the per-ROI
    cooccurrence_degree column by level_{level} block id and computes the
    SAME consistency-weighted-mean-degree score run_base_noconv.py uses
    for the observed lesion-only fit. Returns the array of per-permutation
    MAX block scores (the null_max distribution).
    '''
    perm_dirs = sorted(d for d in os.listdir(null_dir) if d.startswith('perm_') and
                       os.path.isdir(os.path.join(null_dir, d)))
    null_max = []
    for perm in perm_dirs:
        roi_path = os.path.join(null_dir, perm, f'roi_block_assignments_{task}.csv')
        with open(roi_path, newline='') as fh:
            rows = list(csv.DictReader(fh))
        occ_degree = np.array([float(r['cooccurrence_degree'])       for r in rows])
        block      = np.array([int(r[f'level_{level}'])              for r in rows])
        consist    = np.array([float(r[f'consistency_{level}'])      for r in rows])

        n_blocks = int(block.max()) + 1
        block_scores = np.zeros(n_blocks)
        for blk in range(n_blocks):
            idx = np.where(block == blk)[0]
            if len(idx) == 0:
                continue
            w = np.clip(consist[idx], 0, None)
            vals = occ_degree[idx]
            block_scores[blk] = np.average(vals, weights=w) if w.sum() > 0 else vals.mean()
        null_max.append(block_scores.max())
    return np.array(null_max, dtype=np.float64), len(perm_dirs)


#################################
#     PER-TASK COMPUTATION      #
#################################

for task in args.tasks:
    fit_dir  = os.path.join(args.data_path, 'SBMBASE20K', f'SBM_{args.atlas}_{task}_base{args.fit_suffix}')
    null_dir = os.path.join(args.data_path, 'SBMNULL_NOCONV', f'SBM_{args.atlas}_{task}_NULL{args.null_fit_suffix}')

    roi_path = os.path.join(fit_dir, f'roi_block_assignments_{task}.csv')
    with open(roi_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    roi_index_all = np.array([int(r['roi_index']) for r in rows])
    roi_name_all  = np.array([r['roi_name'] for r in rows])

    for level in args.levels:
        roi_block = np.array([int(r[f'level_{level}']) for r in rows])
        order     = np.argsort(roi_index_all)
        roi_index, roi_name, roi_block_sorted = roi_index_all[order], roi_name_all[order], roi_block[order]

        block_path = os.path.join(fit_dir, f'SBM_block_scores_lvl{level}_{task}.csv')
        with open(block_path, newline='') as fh:
            block_rows = list(csv.DictReader(fh))
        block_ids     = np.array([int(r['block']) for r in block_rows])
        block_n_nodes = {int(r['block']): int(r['n_nodes']) for r in block_rows}
        block_score   = {int(r['block']): float(r['score'])  for r in block_rows}
        block_zscore  = {int(r['block']): float(r['zscore']) for r in block_rows}

        if len(block_ids) <= 1:
            log_msg(f"| UPDATE | {task} lvl{level}: single trivial block, skipping")
            continue

        null_max, n_perm = null_max_cooccurrence_scores(null_dir, task, level)
        log_msg(f"| UPDATE | {task} lvl{level}: {n_perm} null permutations, "
                f"{len(block_ids)} observed blocks; null_max range "
                f"[{null_max.min():.4f}, {null_max.max():.4f}], mean {null_max.mean():.4f}")

        p_value = np.array([(np.sum(null_max >= block_score[b]) + 1) / (n_perm + 1) for b in block_ids])
        p_value_by_block = {int(b): float(p_value[i]) for i, b in enumerate(block_ids)}
        n_sig = int(np.sum(p_value < args.alpha))
        log_msg(f"| UPDATE | {task} lvl{level}: {n_sig}/{len(block_ids)} blocks significant at p < {args.alpha}")

        # ---- block-level table ---- #
        table_path = os.path.join(out_dir, f'block_pvalues_base_noconv_lvl{level}_{task}{args.fit_suffix}.tsv')
        with open(table_path, 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter='\t')
            writer.writerow(['block', 'n_nodes', 'observed_score', 'observed_zscore', 'p_value'])
            for b in block_ids:
                writer.writerow([int(b), block_n_nodes[b], round(block_score[b], 6),
                                 round(block_zscore[b], 6), round(p_value_by_block[int(b)], 6)])
        log_msg(f"| UPDATE | {task} lvl{level}: block p-value table saved -> {table_path}")

        # ---- significant blocks x member ROI names ---- #
        sig_block_ids = set(block_ids[p_value < args.alpha])
        n_sig_rois = int(np.isin(roi_block_sorted, list(sig_block_ids)).sum())
        rois_path  = os.path.join(out_dir, f'block_significant_rois_base_noconv_lvl{level}_{task}{args.fit_suffix}.tsv')
        with open(rois_path, 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter='\t')
            writer.writerow(['block', 'p_value', 'roi_name'])
            for b in sorted(sig_block_ids):
                for name in roi_name[roi_block_sorted == b]:
                    writer.writerow([int(b), round(p_value_by_block[b], 6), name])
        log_msg(f"| UPDATE | {task} lvl{level}: {len(sig_block_ids)} significant block(s), "
                f"{n_sig_rois} ROIs -> {rois_path}")

        # ---- test-statistic NIfTI (score) -- every block ---- #
        stat_by_roi   = np.array([block_score[b] for b in roi_block_sorted])
        stat_by_index = np.zeros(int(roi_index.max()) + 1)
        stat_by_index[roi_index] = stat_by_roi

        valid_mask = (atlas_data > 0) & (atlas_data <= len(stat_by_index))
        teststat_data = np.zeros_like(atlas_data, dtype=np.float32)
        teststat_data[valid_mask] = stat_by_index[atlas_data[valid_mask] - 1]

        teststat_path = os.path.join(out_dir, f'{args.atlas}_lvl{level}_teststat_base_noconv_{task}{args.fit_suffix}.nii.gz')
        nib.save(nib.Nifti1Image(teststat_data, atlas_img.affine, atlas_img.header), teststat_path)
        log_msg(f"| UPDATE | {task} lvl{level}: test-statistic (score) NIfTI saved -> {teststat_path}")

        # ---- inverted p-value (1-p) NIfTI -- every block ---- #
        invp_by_roi   = np.array([1.0 - p_value_by_block[b] for b in roi_block_sorted])
        invp_by_index = np.zeros(int(roi_index.max()) + 1)
        invp_by_index[roi_index] = invp_by_roi

        invp_data = np.zeros_like(atlas_data, dtype=np.float32)
        invp_data[valid_mask] = invp_by_index[atlas_data[valid_mask] - 1]

        invp_path = os.path.join(out_dir, f'{args.atlas}_lvl{level}_invp_base_noconv_{task}{args.fit_suffix}.nii.gz')
        nib.save(nib.Nifti1Image(invp_data, atlas_img.affine, atlas_img.header), invp_path)
        log_msg(f"| UPDATE | {task} lvl{level}: inverted p-value (1-p) NIfTI saved -> {invp_path}")

log_msg(f"| FINISHED | Block p-values (lesion-only base_noconv vs SBMNULL_NOCONV) saved -> {out_dir}")
