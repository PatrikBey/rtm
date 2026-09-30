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
# Variant of run.py with the mean-shift change-point convergence        #
# criterion removed: earlier fits (run.py, --shift_factor 0.75) all     #
# looked like they were still drifting toward lower description length  #
# when the change point fired, so this instead runs a single FIXED-     #
# length MCMC sweep (--fixed_iter, default 5000) with no early stop,    #
# and reuses the LAST --window_size iterations of that same sweep as    #
# the accumulation window (block assignments, mrs matrices, entropy)    #
# instead of running a separate additional round of sweeps afterward.   #
#                                                                       #
# run.py itself is untouched -- this is a standalone copy with only the #
# fitting loop (fit_nested_sbm_layered_noconv, in functions.py so       #
# run_null_noconv.py can reuse it too) and the CLI args around it       #
# changed; everything downstream (ROI tables, NIfTI outputs, block      #
# scores, entropy plot) is identical to run.py.                        #
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

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from functions import create_multilayer_graph_streaming, fit_nested_sbm_layered_noconv
from utils import log_msg, get_graph_layers, get_disco_format


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Run multi-layer nested SBM (fixed iteration count, no change-point criterion).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--score', type=str, default='Foreperiod_Long_tau', help='Behaviour score to analyze')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--fixed_iter', type=int, default=5000,
                  help='Fixed number of main-loop MCMC sweeps, no early stop (default: 5000)')
args.add_argument('--window_size', type=int, default=250,
                  help='Number of trailing iterations reused as the accumulation window (default: 250)')
args.add_argument('--multiflip', action='store_true', default=False,
                  help='Use multiflip_mcmc_sweep instead of mcmc_sweep')
args.add_argument('--seed', type=int, default=42, help='Random seed for graph_tool RNG (default: 42)')
args.add_argument('--behaviour_dist', type=str, default='normal', choices=['normal', 'poisson'])
args.add_argument('--cooccurrence_dist', type=str, default='normal', choices=['normal', 'poisson'])
args.add_argument('--combined_layers', action=argparse.BooleanOptionalAction, default=False)
args.add_argument('--output_dir', type=str, default=None,
                  help='Explicit output directory, overriding the default '
                       '{data_path}/SBMNOCONV/SBM_{atlas}_{score}_{singleflip,multiflip} naming convention')
args.add_argument('--participants', type=str, default=None,
                  help='Path to participants.tsv (default: {data_path}/participants.tsv)')
args.add_argument('--disconnectomes_dir', type=str, default=None,
                  help='Explicit disconnectomes directory, overriding the default '
                       '{data_path}/DISCONNECTOMES (e.g. DISCONNECTOMES400 for the 400-ROI atlas)')
args.add_argument('--edge_threshold', type=float, default=75,
                  help='Percentile threshold applied independently to each layer when building '
                       'the multilayer graph -- only the top (100-edge_threshold)%% of nonzero '
                       'node pairs in each layer survive (default: 75, i.e. top 25%%)')
args = args.parse_args()

log_msg(f"| START | Running fixed-iteration multi-layer nested SBM (no change-point criterion)")
log_msg(f"| UPDATE | Data path: {args.data_path}")
log_msg(f"| UPDATE | Behaviour score: {args.score}")

if args.output_dir:
    output_dir = args.output_dir
elif args.multiflip:
    output_dir = os.path.join(args.data_path, 'SBMNOCONV', f'SBM_{args.atlas}_{args.score}_multiflip')
else:
    output_dir = os.path.join(args.data_path, 'SBMNOCONV', f'SBM_{args.atlas}_{args.score}_singleflip')

os.makedirs(output_dir, exist_ok=True)


#################################
#          LOAD DATA            #
#################################

disconnectomes_dir = args.disconnectomes_dir or os.path.join(args.data_path, 'DISCONNECTOMES')
disco_format = get_disco_format(disconnectomes_dir, args.atlas)
discos = os.listdir(disconnectomes_dir)
subject_list = [f.split('_')[0] for f in discos if f.endswith(f'_{args.atlas}.{disco_format}')]
participants_path = args.participants or os.path.join(args.data_path, 'participants.tsv')
part = np.genfromtxt(participants_path, dtype=str, delimiter='\t')
score_col = np.where(part[0] == args.score)[0][0]

atlas_meta = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[0, :].tolist()
node_names = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, atlas_meta.index('label')].tolist()
location_col = 'region' if 'region' in atlas_meta else 'AAL3'  # VoxelAtlas_* areas.txt has no 'region' column
locations  = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, atlas_meta.index(location_col)].tolist()
dim = len(node_names)

#################################
#         BUILD GRAPH           #
#################################
#
# streaming: loads and thresholds one subject's disconnectome at a time and
# accumulates it directly into the two running layer matrices, instead of
# stacking every subject's dense (n_nodes, n_nodes) matrix in memory at
# once -- required at voxel-resolution atlas sizes (tens of thousands of
# nodes), where the old stack-then-build approach scales to hundreds of
# GB-to-TB and is not feasible.

graph, subject_list_clean, behaviour, subjects_missing_score, empty_subjects = create_multilayer_graph_streaming(
    disconnectomes_dir, args.atlas, subject_list, part, score_col, node_names, edge_threshold=args.edge_threshold,
    behaviour_dist=args.behaviour_dist,
    cooccurrence_dist=args.cooccurrence_dist,
    combined_layers=args.combined_layers)

log_msg(f"| UPDATE | Total subjects: {len(subject_list)}")
log_msg(f"| UPDATE | Included: {len(subject_list_clean)}")
log_msg(f"| UPDATE | Missing {args.score}: {len(subjects_missing_score)}")
log_msg(f"| UPDATE | Empty disconnectome: {len(empty_subjects)}")

graph_path = os.path.join(output_dir, f'SBM_graph_{args.score}.gt')
graph.save(graph_path)
log_msg(f"| UPDATE | Multilayer graph saved (graph-tool format) -> {graph_path}")

occ_layer, beh_layer = get_graph_layers(graph)
np.savetxt(os.path.join(output_dir, f'SBM_layer_{args.score}_cooccurrence.txt'), occ_layer, fmt='%.6f')
np.savetxt(os.path.join(output_dir, f'SBM_layer_{args.score}_behaviour.txt'), beh_layer, fmt='%.6f')


#################################
#          FIT MODEL            #
#################################

real_results = fit_nested_sbm_layered_noconv(
    graph,
    fixed_iter=args.fixed_iter,
    window_size=args.window_size,
    behaviour_dist=args.behaviour_dist,
    cooccurrence_dist=args.cooccurrence_dist,
    multiflip=args.multiflip,
    seed=args.seed
)

state_nested      = real_results['state']
g                 = state_nested.g
meaningful_levels = real_results['meaningful_levels']
modal_assignments  = real_results['modal_assignments']
block_connectivity = real_results['block_connectivity']
node_consistency   = real_results['node_consistency']

final_graph_path = os.path.join(output_dir, f'SBM_final_graph_{args.score}.gt')
g.save(final_graph_path)
log_msg(f"| UPDATE | Final MCMC graph saved (graph-tool format) -> {final_graph_path}")


#################################
#       PRINT BLOCK STRUCTURE   #
#################################

log_msg(f"| UPDATE | Total model entropy: {real_results['entropy']:.2f}")
log_msg(f"| UPDATE | Hierarchy levels: {real_results['n_levels']} total, "
        f"{len(meaningful_levels)} meaningful ({meaningful_levels})")
log_msg(f"| UPDATE | Fixed-iteration fit: {args.fixed_iter} iterations, "
        f"last {real_results['n_converged_samples']} reused as accumulation window "
        f"(no change-point criterion applied)")


#################################
#        SCALAR OUTPUTS         #
#################################

np.save(os.path.join(output_dir, f'entropy_trajectory_{args.score}.npy'),
        real_results['entropy_trajectory'])
np.save(os.path.join(output_dir, f'entropy_converged_{args.score}.npy'),
        real_results['entropy_converged'])
log_msg(f"| UPDATE | Entropy / DL trajectory saved "
        f"({len(real_results['entropy_trajectory'])} MCMC iters + "
        f"{len(real_results['entropy_converged'])} accumulation samples)")

beh_degree = np.zeros(g.num_vertices())
occ_degree = np.zeros(g.num_vertices())
for e in g.edges():
    if g.ep.layer[e] == 0:
        w = g.ep.behaviour_weight[e]
        beh_degree[int(e.source())] += w
        beh_degree[int(e.target())] += w
    else:
        w = g.ep.cooccurrence_weight[e]
        occ_degree[int(e.source())] += w
        occ_degree[int(e.target())] += w

block_edge_mean = np.diag(real_results['edge_mean'])
block_edge_var  = np.diag(real_results['edge_var'])

roi_level_path  = os.path.join(output_dir, f'roi_block_assignments_{args.score}.csv')
level_cols      = [f'level_{k}'       for k in meaningful_levels]
consist_cols    = [f'consistency_{k}' for k in meaningful_levels]
header          = ['roi_index', 'roi_name', 'behaviour_degree', 'cooccurrence_degree',
                   'block_edge_mean', 'block_edge_var'] + level_cols + consist_cols
with open(roi_level_path, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(header)
    for node_idx, roi_name in enumerate(node_names):
        block_vals   = [int(modal_assignments[k][node_idx])              for k in meaningful_levels]
        consist_vals = [round(float(node_consistency[k][node_idx]), 6)   for k in meaningful_levels]
        row = [node_idx, roi_name,
               round(float(beh_degree[node_idx]), 6),
               round(float(occ_degree[node_idx]), 6),
               round(float(block_edge_mean[node_idx]), 6),
               round(float(block_edge_var[node_idx]), 6)] + block_vals + consist_vals
        writer.writerow(row)
log_msg(f"| UPDATE | ROI block-assignment table saved ({len(node_names)} ROIs x "
        f"{len(meaningful_levels)} levels) -> {roi_level_path}")


#################################
#        ENTROPY FIGURE         #
#################################

dS      = real_results['entropy_trajectory']
dS_conv = real_results['entropy_converged']
collect_from = real_results['convergence_iteration']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.plot(dS, linewidth=1, alpha=0.7, color='steelblue', label='MCMC entropy')
ax.axvspan(collect_from, len(dS) - 1, color='seagreen', alpha=0.15,
          label=f'Accumulation window (last {args.window_size} iters)')
ax.set_xlabel('MCMC Iteration', fontsize=11)
ax.set_ylabel('Model Entropy (Description Length)', fontsize=11)
ax.set_title(f'Entropy Trajectory — fixed {args.fixed_iter} iterations, no change-point criterion',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(dS_conv, linewidth=1, alpha=0.8, color='seagreen',
        label=f'Last {args.window_size} iterations (reused as accumulation window)')
ax.set_xlabel('Accumulation Iteration', fontsize=11)
ax.set_ylabel('Model Entropy (Description Length)', fontsize=11)
ax.set_title('Accumulation Window: Entropy', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

y_all = np.concatenate([dS, dS_conv])
y_pad = (y_all.max() - y_all.min()) * 0.05
for ax in axes:
    ax.set_ylim(y_all.min() - y_pad, y_all.max() + y_pad)

plt.suptitle(f'MCMC Entropy — {args.score} ({args.atlas}) — fixed iterations, no change-point criterion',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'SBM_entropy_trajectory_{args.score}.png'),
            dpi=150, bbox_inches='tight')
log_msg(f"| UPDATE | Entropy trajectory saved")
plt.close()


#################################
#    BLOCK COMMUNITY VOLUME     #
#################################

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)

for level_idx in meaningful_levels:
    b_level  = modal_assignments[level_idx]
    n_blocks = int(b_level.max()) + 1

    block_scores = np.zeros(n_blocks)
    for blk in range(n_blocks):
        nodes_in_block = np.where(b_level == blk)[0]
        if len(nodes_in_block) == 0:
            continue
        w = np.clip(node_consistency[level_idx][nodes_in_block], 0, None)
        vals = beh_degree[nodes_in_block]
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
            f"selected (z > 0) for NIfTI output -> {score_path}")

    zscore_by_node = block_zscore[b_level]
    zscore_data    = np.zeros_like(atlas_data, dtype=np.float32)
    valid_mask     = (atlas_data > 0) & (atlas_data <= len(zscore_by_node))
    zscore_data[valid_mask] = zscore_by_node[atlas_data[valid_mask] - 1]
    zscore_nii_path = os.path.join(output_dir,
                                   f'{args.atlas}_lvl{level_idx}_blockzscores_{args.score}.nii.gz')
    zscore_img = nib.Nifti1Image(zscore_data, atlas_img.affine, atlas_img.header)
    nib.save(zscore_img, zscore_nii_path)
    log_msg(f"| UPDATE | Combined block z-score NIfTI saved: level {level_idx} "
            f"(all {n_blocks} blocks) -> {zscore_nii_path}")

    score_data = np.zeros_like(atlas_data, dtype=np.float32)
    for blk in selected_blocks:
        nodes_in_block = np.where(b_level == blk)[0]
        if len(nodes_in_block) == 0:
            continue
        block_data    = np.zeros_like(atlas_data, dtype=np.int32)
        mapping_lines = []
        for region_idx, node_idx in enumerate(nodes_in_block):
            parcel_val = node_idx + 1
            region_num = region_idx + 1
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
                f"({len(nodes_in_block)} regions) -> {nii_path}")

    score_nii_path = os.path.join(output_dir,
                                  f'{args.atlas}_lvl{level_idx}_blockscores_{args.score}.nii.gz')
    score_img = nib.Nifti1Image(score_data, atlas_img.affine, atlas_img.header)
    nib.save(score_img, score_nii_path)
    log_msg(f"| UPDATE | Combined block-score NIfTI saved: level {level_idx} "
            f"({len(selected_blocks)} blocks) -> {score_nii_path}")

    bmat     = block_connectivity[level_idx]
    csv_path = os.path.join(output_dir,
                            f'SBM_block_connectivity_lvl{level_idx}_{args.score}.csv')
    np.savetxt(csv_path, bmat, delimiter=',', fmt='%.6f')
    log_msg(f"| UPDATE | Block connectivity matrix saved: level {level_idx} "
            f"({n_blocks}x{n_blocks}) -> {csv_path}")


log_msg(f"| FINISHED | All outputs saved")
