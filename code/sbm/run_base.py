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
# The following script performs modelling of various intelligence       #
# related cognitive domains using a single-layer nested                 #
# stochastic block modelling (SBM) framework                            #
#                                                                       #
# Base model: identical to run.py except it contains no behavioural     #
# information — only the lesion cooccurrence layer (as defined in       #
# run.py) is built and modelled.                                        #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/06/30.                                              #
#                                                                       #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################


import os
import argparse
import graph_tool.all as gt
from graph_tool import draw
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
import nibabel as nib
from functions import create_cooccurrence_graph
from functions import fit_nested_sbm
from utils import load_graphs, log_msg, get_cooccurrence_layer



#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Run single-layer nested SBM (cooccurrence only) on disconnectome data.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
# args.add_argument('--data_path', type=str, default='/data/patrik/RT/DATA', help='Path to the data directory')
args.add_argument('--score', type=str, default='Foreperiod_Long_tau', help='Behaviour score to analyze')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--max_iter', type=int, default=10000,
                  help='Maximum MCMC sweeps for change-point detection (default: 10000)')
args.add_argument('--window_size', type=int, default=250,
                  help='Window size for change-point detection and accumulation (default: 250)')
args.add_argument('--shift_factor', type=float, default=0.75,
                  help='Mean-shift threshold: change point when sliding window mean drops '
                       'below first_window_mean - shift_factor * first_window_std (default: 0.75)')
args.add_argument('--seed', type=int, default=42,
                  help='Random seed for graph_tool RNG (default: 42)')
args.add_argument('--cooccurrence_dist', type=str, default='normal',
                  choices=['normal', 'poisson'],
                  help='Assumed distribution for the cooccurrence layer (default: normal). '
                       'Set to poisson to apply [0,1] rescaling and use discrete-Poisson DL.')
args = args.parse_args()

log_msg(f"| START | Running single-layer nested SBM (cooccurrence only) on disconnectome data")
log_msg(f"| UPDATE | Data path: {args.data_path}")
log_msg(f"| UPDATE | Behaviour score: {args.score}")

output_dir = os.path.join(args.data_path, 'SBMBASE', f'SBM_{args.atlas}_{args.score}_base_singleflip')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

#################################
#          LOAD DATA            #
#################################

# ---- disconnectomes and behaviour ---- #
discos = os.listdir(os.path.join(args.data_path, 'DISCONNECTOMES'))
subject_list = [f.split('_')[0] for f in discos if f.endswith(f'_{args.atlas}.tsv')]
part = np.genfromtxt(os.path.join(args.data_path, 'participants.tsv'), dtype=str, delimiter='\t')
score_col = np.where(part[0] == args.score)[0][0]

# ---- graph nodes ---- #
atlas_meta = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[0, :].tolist()
node_names = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, atlas_meta.index('label')].tolist()
locations  = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, atlas_meta.index('region')].tolist()
dim = len(node_names)

subject_list_clean, behaviour, adj_matrices, subjects_missing_score, empty_subjects = load_graphs(args.data_path, args.atlas, subject_list, part, score_col)

log_msg(f"| UPDATE | Total subjects: {len(subject_list)}")
log_msg(f"| UPDATE | Included: {len(subject_list_clean)}")
log_msg(f"| UPDATE | Missing {args.score}: {len(subjects_missing_score)}")
log_msg(f"| UPDATE | Empty disconnectome: {len(empty_subjects)}")



#################################
#         BUILD GRAPH           #
#################################

graph = create_cooccurrence_graph(adj_matrices, node_names, edge_threshold=75,
                                  cooccurrence_dist=args.cooccurrence_dist)

graph_path = os.path.join(output_dir, f'SBM_graph_{args.score}.gt')
graph.save(graph_path)
log_msg(f"| UPDATE | Single-layer graph saved (graph-tool format) → {graph_path}")


occ_layer = get_cooccurrence_layer(graph)

np.savetxt(os.path.join(output_dir, f'SBM_layer_{args.score}_cooccurrence.txt'), occ_layer, fmt='%.6f')



#################################
#          FIT MODEL            #
#################################

real_results = fit_nested_sbm(
    graph,
    max_iter=args.max_iter,
    window_size=args.window_size,
    shift_factor=args.shift_factor,
    cooccurrence_dist=args.cooccurrence_dist,
    seed=args.seed
)

state_nested      = real_results['state']
g                 = state_nested.g
meaningful_levels = real_results['meaningful_levels']
modal_assignments  = real_results['modal_assignments']   # {level: (n_nodes,) array}
block_connectivity = real_results['block_connectivity']  # {level: (B×B) joint matrix}
node_consistency   = real_results['node_consistency']    # {level: (n_nodes,) array in [0,1]}

final_graph_path = os.path.join(output_dir, f'SBM_final_graph_{args.score}.gt')
g.save(final_graph_path)
log_msg(f"| UPDATE | Final MCMC graph saved (graph-tool format) → {final_graph_path}")



#################################
#       PRINT BLOCK STRUCTURE   #
#################################

log_msg(f"| UPDATE | Total model entropy: {real_results['entropy']:.2f}")
log_msg(f"| UPDATE | Hierarchy levels: {real_results['n_levels']} total, "
        f"{len(meaningful_levels)} meaningful ({meaningful_levels})")
log_msg(f"| UPDATE | Converged: {real_results['converged']} "
        f"(iteration {real_results['convergence_iteration']}, "
        f"{real_results['n_converged_samples']} accumulation samples)")



#################################
#        SCALAR OUTPUTS         #
#################################

# --- Entropy / description length trajectory ---
np.save(os.path.join(output_dir, f'entropy_trajectory_{args.score}.npy'),
        real_results['entropy_trajectory'])
np.save(os.path.join(output_dir, f'entropy_converged_{args.score}.npy'),
        real_results['entropy_converged'])
log_msg(f"| UPDATE | Entropy / DL trajectory saved "
        f"({len(real_results['entropy_trajectory'])} MCMC iters + "
        f"{len(real_results['entropy_converged'])} accumulation samples)")

# --- ROI × level block-assignment table ---
# Rows = ROIs, columns = meaningful level indices.
# Values are 0-indexed block IDs from the modal partition.

# Weighted degree per node, cooccurrence layer (the only layer present).
occ_degree = np.zeros(g.num_vertices())
for e in g.edges():
    w = g.ep.cooccurrence_weight[e]
    occ_degree[int(e.source())] += w
    occ_degree[int(e.target())] += w

# Posterior mean/variance of each node's own-block internal edge mass
# (diagonal of edge_mean / edge_var — accumulated over the accumulation window).
block_edge_mean = np.diag(real_results['edge_mean'])
block_edge_var  = np.diag(real_results['edge_var'])

import csv
roi_level_path  = os.path.join(output_dir, f'roi_block_assignments_{args.score}.csv')
level_cols      = [f'level_{k}'       for k in meaningful_levels]
consist_cols    = [f'consistency_{k}' for k in meaningful_levels]
header          = ['roi_index', 'roi_name', 'cooccurrence_degree',
                   'block_edge_mean', 'block_edge_var'] + level_cols + consist_cols
with open(roi_level_path, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(header)
    for node_idx, roi_name in enumerate(node_names):
        block_vals   = [int(modal_assignments[k][node_idx])              for k in meaningful_levels]
        consist_vals = [round(float(node_consistency[k][node_idx]), 6)   for k in meaningful_levels]
        row = [node_idx, roi_name,
               round(float(occ_degree[node_idx]), 6),
               round(float(block_edge_mean[node_idx]), 6),
               round(float(block_edge_var[node_idx]), 6)] + block_vals + consist_vals
        writer.writerow(row)
log_msg(f"| UPDATE | ROI block-assignment table saved ({len(node_names)} ROIs × "
        f"{len(meaningful_levels)} levels) → {roi_level_path}")



#################################
#        VISUALISATIONS         #
#################################

# ---- Entropy trajectory ---- #
conv_iter    = real_results['convergence_iteration']
n_conv       = real_results['n_converged_samples']
converged    = real_results['converged']
dS           = real_results['entropy_trajectory']
dS_conv      = real_results['entropy_converged']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.plot(dS, linewidth=1, alpha=0.7, color='steelblue', label='MCMC entropy')
if converged:
    ax.axvline(conv_iter, color='firebrick', linewidth=1.5, linestyle='--',
               label=f'Mean-shift change point (iter {conv_iter})')
    ax.axhline(real_results.get('threshold', np.nan), color='firebrick',
               linewidth=1, linestyle=':', alpha=0.6, label='Shift threshold')
else:
    ax.axvline(len(dS) - 1, color='darkorange', linewidth=1.5, linestyle='--',
               label=f'No shift detected within {args.max_iter} iter')
ax.set_xlabel('MCMC Iteration', fontsize=11)
ax.set_ylabel('Model Entropy (Description Length)', fontsize=11)
ax.set_title('Entropy Trajectory — Mean-Shift Detection', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(dS_conv, linewidth=1, alpha=0.8, color='seagreen',
        label=f'Accumulation window ({n_conv} samples)')
ax.set_xlabel('Accumulation Iteration', fontsize=11)
ax.set_ylabel('Model Entropy (Description Length)', fontsize=11)
ax.set_title('Accumulation Window: Entropy', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

y_all = np.concatenate([dS, dS_conv])
y_pad = (y_all.max() - y_all.min()) * 0.05
for ax in axes:
    ax.set_ylim(y_all.min() - y_pad, y_all.max() + y_pad)

plt.suptitle('MCMC Entropy — Weighted Nested Hierarchical SBM (single-layer, cooccurrence only)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'SBM_entropy_trajectory_{args.score}.png'),
            dpi=150, bbox_inches='tight')
log_msg(f"| UPDATE | Entropy trajectory saved "
        f"(converged={converged}, iter={conv_iter}, {n_conv} accumulation samples)")
plt.close()



#################################
#    BLOCK COMMUNITY VOLUME     #
#################################

# For each meaningful level, write a NIfTI image where every voxel belonging
# to a parcellation region is replaced by that region's modal block ID.
#
# Parcellation convention: the integer value v in the NIfTI corresponds to
# node_names[v-1] (1-indexed, matching the atlas areas file row order).
# Block IDs are stored as (block_index + 1) so that 0 unambiguously marks
# background voxels.
#
# Only blocks that are informative about the lesion cooccurrence structure
# are saved: per block, score = consistency-weighted mean cooccurrence_degree
# of its member nodes (node_consistency acts as a confidence weight on how
# reliably each node belongs to that block under the model; negative kappa
# values are clipped to 0 so unreliable nodes don't invert the weighting).
# Scores are z-scored across blocks within a level; only blocks with z > 0
# (above-average informativeness) are written out.

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)

for level_idx in meaningful_levels:
    b_level  = modal_assignments[level_idx]
    n_blocks = int(b_level.max()) + 1

    # ---- Score blocks by cooccurrence informativeness ---- #
    block_scores = np.zeros(n_blocks)
    for blk in range(n_blocks):
        nodes_in_block = np.where(b_level == blk)[0]
        if len(nodes_in_block) == 0:
            continue
        w = np.clip(node_consistency[level_idx][nodes_in_block], 0, None)
        vals = occ_degree[nodes_in_block]
        block_scores[blk] = np.average(vals, weights=w) if w.sum() > 0 else vals.mean()

    score_std    = block_scores.std()
    block_zscore = (block_scores - block_scores.mean()) / score_std if score_std > 0 \
                   else np.zeros_like(block_scores)
    selected_blocks = np.where(block_zscore > 0)[0]

    score_path = os.path.join(output_dir, f'SBM_block_scores_lvl{level_idx}_{args.score}.csv')
    with open(score_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['block', 'n_nodes', 'score', 'zscore', 'selected'])
        for blk in range(n_blocks):
            writer.writerow([blk, int((b_level == blk).sum()),
                             round(float(block_scores[blk]), 6),
                             round(float(block_zscore[blk]), 6),
                             blk in selected_blocks])
    log_msg(f"| UPDATE | Level {level_idx}: {len(selected_blocks)}/{n_blocks} blocks "
            f"selected (z > 0) for NIfTI output → {score_path}")

    # ---- Combined NIfTI: z-score corrected value for ALL blocks (not just selected) ---- #
    zscore_by_node = block_zscore[b_level]      # (n_nodes,) z-score of each node's own block
    zscore_data    = np.zeros_like(atlas_data, dtype=np.float32)
    valid_mask     = (atlas_data > 0) & (atlas_data <= len(zscore_by_node))
    zscore_data[valid_mask] = zscore_by_node[atlas_data[valid_mask] - 1]
    zscore_nii_path = os.path.join(output_dir,
                                   f'{args.atlas}_lvl{level_idx}_blockzscores_{args.score}.nii.gz')
    zscore_img = nib.Nifti1Image(zscore_data, atlas_img.affine, atlas_img.header)
    nib.save(zscore_img, zscore_nii_path)
    log_msg(f"| UPDATE | Combined block z-score NIfTI saved: level {level_idx} "
            f"(all {n_blocks} blocks) → {zscore_nii_path}")

    # One NIfTI + txt mapping per selected block, plus a single combined image
    # where every voxel of a selected block carries that block's weighted
    # cooccurrence score (consistency-weighted mean cooccurrence_degree) and 0 elsewhere.
    score_data = np.zeros_like(atlas_data, dtype=np.float32)
    for blk in selected_blocks:
        nodes_in_block = np.where(b_level == blk)[0]
        if len(nodes_in_block) == 0:
            continue
        block_data    = np.zeros_like(atlas_data, dtype=np.int32)
        mapping_lines = []
        for region_idx, node_idx in enumerate(nodes_in_block):
            parcel_val = node_idx + 1       # 1-indexed parcel code in atlas NIfTI
            region_num = region_idx + 1     # 1-indexed label within this block
            block_data[atlas_data == parcel_val] = region_num
            score_data[atlas_data == parcel_val] = block_scores[blk]
            mapping_lines.append(f"{region_num}\t{node_names[node_idx]}\t{locations[node_idx]}")
        nii_path = os.path.join(output_dir,
                                f'{args.atlas}_lvl{level_idx}_block{blk}_{args.score}.nii.gz')
        txt_path = os.path.join(output_dir,
                                f'{args.atlas}_lvl{level_idx}_block{blk}_{args.score}.txt')
        out_img = nib.Nifti1Image(block_data, atlas_img.affine, atlas_img.header)
        nib.save(out_img, nii_path)
        with open(txt_path, 'w') as fh:
            fh.write('\n'.join(mapping_lines) + '\n')
        log_msg(f"| UPDATE | Block NIfTI saved: level {level_idx} block {blk} "
                f"({len(nodes_in_block)} regions) → {nii_path}")

    score_nii_path = os.path.join(output_dir,
                                  f'{args.atlas}_lvl{level_idx}_blockscores_{args.score}.nii.gz')
    score_img = nib.Nifti1Image(score_data, atlas_img.affine, atlas_img.header)
    nib.save(score_img, score_nii_path)
    log_msg(f"| UPDATE | Combined block-score NIfTI saved: level {level_idx} "
            f"({len(selected_blocks)} blocks) → {score_nii_path}")
    # Block connectivity matrix — CSV
    bmat     = block_connectivity[level_idx]
    csv_path = os.path.join(output_dir,
                            f'SBM_block_connectivity_lvl{level_idx}_{args.score}.csv')
    np.savetxt(csv_path, bmat, delimiter=',', fmt='%.6f')
    log_msg(f"| UPDATE | Block connectivity matrix saved: level {level_idx} "
            f"({n_blocks}×{n_blocks}) → {csv_path}")


log_msg(f"| FINISHED | All outputs saved")
