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
# block_pvalues.py adapted to test the SBMNOCONV20K (run_noconv.py,     #
# fixed-iteration, no change-point criterion) Schaefer2018-400 fits     #
# against a matching permutation null. Same max-statistic logic and     #
# rationale as block_pvalues.py (see that file's header) -- block ids   #
# aren't comparable across independently-fit partitions, so each        #
# permutation's single highest block statistic is pooled into a null    #
# distribution, and every observed block is compared against that same  #
# null_max distribution. This already controls the family-wise error    #
# rate across blocks by construction; no separate correction is added.  #
#                                                                       #
# --null_root selects which null to test against:                       #
#                                                                       #
# "SBMNULL_NOCONV" (default) -- run_null_noconv.py's null, built with   #
#    the SAME fitting method, edge_threshold, atlas AND (critically)    #
#    the same post-rescaling behaviour scores as SBMNOCONV20K itself    #
#    (task name matches directly, no NULL_TASK_MAP needed). Because     #
#    null and observed are now genuinely matched, this uses each        #
#    block's RAW "score" (--stat_column, default) rather than a         #
#    within-fit z-score -- checked directly: SBMNULL_NOCONV perm_00000  #
#    level-0 raw scores span [25.6, 234.3], vs. SBMNOCONV20K's observed  #
#    [0.0, 457.4] -- same order of magnitude, a valid direct comparison. #
#                                                                       #
# "SBMNULL" (legacy) -- the original run_null.py null (old              #
#    convergence-based fit, pre-2026-09-08 un-rescaled behaviour,       #
#    Foreperiod_Long_tau/GoNoGo_tau only). Neither the fitting method    #
#    nor the behaviour scale match SBMNOCONV20K, so this path forces     #
#    --stat_column zscore (dimensionless, fit-relative, survives the    #
#    scale mismatch) and NULL_TASK_MAP (pairs each observed task with    #
#    the nearest available legacy-null task by identity, an accepted    #
#    approximation, not a like-for-like match). Kept only for            #
#    comparison against earlier results; SBMNULL_NOCONV is preferred    #
#    whenever available for a given task.                               #
#                                                                       #
# Outputs (per task x fit_suffix):                                      #
#   block_pvalues_noconv_lvl{level}_{task}{fit_suffix}.tsv               #
#   block_significant_rois_noconv_lvl{level}_{task}{fit_suffix}.tsv      #
#   {atlas}_lvl{level}_teststat_noconv_{task}{fit_suffix}.nii.gz         #
#       (every block's own statistic per --stat_column -- not just the  #
#       significant ones)                                               #
#   {atlas}_lvl{level}_invp_noconv_{task}{fit_suffix}.nii.gz             #
#       (every block's 1 - p_value, so higher = more significant)       #
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

NULL_TASK_MAP = {
    'foreperiod_hexg_tau': 'Foreperiod_Long_tau',
    'gonogo_hexg_tau':     'GoNoGo_tau',
}

args = argparse.ArgumentParser(description='Block-level max-statistic permutation test: '
                               'SBMNOCONV20K observed fits vs. a permutation null.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+', default=['foreperiod_hexg_tau', 'gonogo_hexg_tau'],
                  help='Observed (SBMNOCONV20K) task names to evaluate '
                       '(default: the only two tasks SBMNOCONV20K has)')
args.add_argument('--level', type=int, default=0, help='Hierarchy level to test (default: 0)')
args.add_argument('--alpha', type=float, default=0.05, help='Significance threshold (default: 0.05; '
                  'already family-wise controlled by the max-statistic, no further correction applied)')
args.add_argument('--fit_suffix', type=str, default='_singleflip',
                  help='Suffix of the observed SBMNOCONV20K run directory (default: _singleflip)')
args.add_argument('--null_root', type=str, default='SBMNULL_NOCONV', choices=['SBMNULL_NOCONV', 'SBMNULL'],
                  help='Which null to test against (default: SBMNULL_NOCONV, the matched null -- '
                       'see file header). SBMNULL is the legacy, scale-mismatched null.')
args.add_argument('--null_fit_suffix', type=str, default='_singleflip',
                  help='Fit-suffix on the SBMNULL_NOCONV null directory name (default: _singleflip; '
                       'ignored for --null_root SBMNULL, which has none)')
args.add_argument('--stat_column', type=str, default=None, choices=['score', 'zscore'],
                  help='Which SBM_block_scores column to test (default: "score" for SBMNULL_NOCONV, '
                       '"zscore" for SBMNULL -- see file header for why each default is forced)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args = args.parse_args()

if args.stat_column is None:
    args.stat_column = 'score' if args.null_root == 'SBMNULL_NOCONV' else 'zscore'

out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

log_msg(f"| START | Block max-statistic p-values (noconv vs null) lvl{args.level} | tasks: {args.tasks}")


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
    if args.null_root == 'SBMNULL_NOCONV':
        null_task = task  # matched null: same task name, no mapping needed
        null_dir  = os.path.join(args.data_path, 'SBMNULL_NOCONV',
                                 f'SBM_{args.atlas}_{null_task}_NULL{args.null_fit_suffix}')
    else:
        if task not in NULL_TASK_MAP:
            log_msg(f"| WARNING | {task}: no null-task mapping defined, skipping "
                    f"(add it to NULL_TASK_MAP if a matching SBMNULL run exists)")
            continue
        null_task = NULL_TASK_MAP[task]
        null_dir  = os.path.join(args.data_path, 'SBMNULL', f'SBM_{args.atlas}_{null_task}_NULL')

    fit_dir = os.path.join(args.data_path, 'SBMNOCONV20K', f'SBM_{args.atlas}_{task}{args.fit_suffix}')

    # ---- observed lvl-{level} block assignment + raw score + z-score ---- #
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
    block_score   = {int(r['block']): float(r['score'])  for r in block_rows}
    block_zscore  = {int(r['block']): float(r['zscore']) for r in block_rows}
    block_stat    = block_score if args.stat_column == 'score' else block_zscore  # test statistic

    # ---- null_max: highest within-fit statistic per permutation ---- #
    perm_dirs = sorted(d for d in os.listdir(null_dir) if d.startswith('perm_') and
                       os.path.isdir(os.path.join(null_dir, d)))
    null_max = []
    for perm in perm_dirs:
        perm_block_path = os.path.join(null_dir, perm, f'SBM_block_scores_lvl{args.level}_{null_task}.csv')
        with open(perm_block_path, newline='') as fh:
            perm_stats = [float(r[args.stat_column]) for r in csv.DictReader(fh)]
        null_max.append(max(perm_stats))
    null_max = np.array(null_max, dtype=np.float64)
    n_perm   = len(null_max)

    log_msg(f"| UPDATE | {task} (null: {args.null_root}/{null_task}, stat={args.stat_column}): "
            f"{n_perm} permutations, {len(block_ids)} observed blocks at level {args.level}")
    log_msg(f"| UPDATE | {task}: null_max ({args.stat_column}) range "
            f"[{null_max.min():.4f}, {null_max.max():.4f}], mean {null_max.mean():.4f}")

    # ---- test every observed block's statistic against the same null_max distribution ---- #
    p_value = np.array([(np.sum(null_max >= block_stat[b]) + 1) / (n_perm + 1) for b in block_ids])
    p_value_by_block = {int(b): float(p_value[i]) for i, b in enumerate(block_ids)}
    n_sig   = int(np.sum(p_value < args.alpha))
    log_msg(f"| UPDATE | {task}: {n_sig}/{len(block_ids)} blocks significant at p < {args.alpha}")

    # ---- block-level table ---- #
    table_path = os.path.join(out_dir, f'block_pvalues_noconv_lvl{args.level}_{task}{args.fit_suffix}.tsv')
    with open(table_path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['block', 'n_nodes', 'observed_score', 'observed_zscore', 'p_value'])
        for b in block_ids:
            writer.writerow([int(b), block_n_nodes[b], round(block_score[b], 6),
                             round(block_zscore[b], 6), round(p_value_by_block[int(b)], 6)])
    log_msg(f"| UPDATE | {task}: block p-value table saved -> {table_path}")

    # ---- significant blocks x member ROI names ---- #
    sig_block_ids = set(block_ids[p_value < args.alpha])

    n_sig_rois = int(np.isin(roi_block, list(sig_block_ids)).sum())
    rois_path  = os.path.join(out_dir, f'block_significant_rois_noconv_lvl{args.level}_{task}{args.fit_suffix}.tsv')
    with open(rois_path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['block', 'p_value', 'roi_name'])
        for b in sorted(sig_block_ids):
            for name in roi_name[roi_block == b]:
                writer.writerow([int(b), round(p_value_by_block[b], 6), name])
    log_msg(f"| UPDATE | {task}: {len(sig_block_ids)} significant block(s), {n_sig_rois} ROIs -> {rois_path}")

    # ---- test-statistic NIfTI: every block's own statistic, not just significant ones ---- #
    stat_by_roi   = np.array([block_stat[b] for b in roi_block])
    stat_by_index = np.zeros(int(roi_index.max()) + 1)
    stat_by_index[roi_index] = stat_by_roi

    valid_mask = (atlas_data > 0) & (atlas_data <= len(stat_by_index))
    teststat_data = np.zeros_like(atlas_data, dtype=np.float32)
    teststat_data[valid_mask] = stat_by_index[atlas_data[valid_mask] - 1]

    teststat_path = os.path.join(out_dir, f'{args.atlas}_lvl{args.level}_teststat_noconv_{task}{args.fit_suffix}.nii.gz')
    nib.save(nib.Nifti1Image(teststat_data, atlas_img.affine, atlas_img.header), teststat_path)
    log_msg(f"| UPDATE | {task}: test-statistic ({args.stat_column}) NIfTI saved -> {teststat_path}")

    # ---- inverted p-value (1-p) NIfTI: every block, not just significant ones ---- #
    invp_by_roi   = np.array([1.0 - p_value_by_block[b] for b in roi_block])
    invp_by_index = np.zeros(int(roi_index.max()) + 1)
    invp_by_index[roi_index] = invp_by_roi

    invp_data = np.zeros_like(atlas_data, dtype=np.float32)
    invp_data[valid_mask] = invp_by_index[atlas_data[valid_mask] - 1]

    invp_path = os.path.join(out_dir, f'{args.atlas}_lvl{args.level}_invp_noconv_{task}{args.fit_suffix}.nii.gz')
    nib.save(nib.Nifti1Image(invp_data, atlas_img.affine, atlas_img.header), invp_path)
    log_msg(f"| UPDATE | {task}: inverted p-value (1-p) NIfTI saved -> {invp_path}")

log_msg(f"| FINISHED | Block p-values (noconv vs null) saved -> {out_dir}")
