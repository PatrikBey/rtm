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
# run_null.py adapted to the "noconv" framework: fixed-iteration, no    #
# change-point MCMC fit (fit_nested_sbm_layered_noconv, functions.py)   #
# instead of the original convergence-based fit_nested_sbm_layered.     #
# Loading/graph-building stay on load_graphs + create_multilayer_graph  #
# (the classic dense path, also from functions.py/utils.py) rather than #
# run_noconv.py's streaming builder: streaming exists to avoid holding  #
# every subject's dense adjacency matrix in memory at once, which only  #
# matters at voxel-atlas node counts -- at Schaefer2018-400 (400 nodes) #
# holding ~200 dense 400x400 matrices is ~128MB, a non-issue. It's also #
# not usable here regardless: create_multilayer_graph_streaming reads   #
# behaviour straight from participants.tsv on each call and has no way  #
# to accept an externally-permuted behaviour vector, whereas            #
# create_multilayer_graph takes the behaviour array directly -- exactly #
# what permutation testing needs (same adjacency matrices reused across #
# permutations, only the behaviour-to-subject pairing shuffled).        #
#                                                                       #
# This exists because the original SBMNULL (run_null.py) was generated  #
# before the 2026-09-08 participants.tsv behaviour rescaling and using  #
# the old convergence-based fit -- neither the behaviour scale nor the  #
# fitting procedure match SBMNOCONV20K's foreperiod_hexg_tau /          #
# gonogo_hexg_tau fits, so it could only be compared against them via a #
# within-fit z-score workaround (see block_pvalues_noconv.py), not a    #
# direct raw-score permutation test. Output lands in SBMNULL_NOCONV/ --  #
# a new top-level directory, not merged into SBMNULL/ -- so the old     #
# (still valid for the old tau scores/old run.py fit) null is untouched.#
#                                                                       #
# Per-permutation output is otherwise identical in format to            #
# run_null.py's (down to the SBM_block_scores_lvl{level}_{score}.csv    #
# columns block_pvalues_noconv.py reads), so an eventual raw-score-     #
# based test can reuse the same downstream analysis code unchanged --   #
# only the null_dir root and the statistic column ("score" instead of   #
# "zscore") would need to change back once this null is available.      #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import gc
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

import graph_tool.all as gt

from functions import create_multilayer_graph, fit_nested_sbm_layered_noconv, permute_behaviour
from utils import load_graphs, log_msg, get_graph_layers


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Behaviour-permutation null model, noconv (fixed-iteration) framework.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--score', type=str, default='foreperiod_hexg_tau', help='Behaviour score to analyze')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--disconnectomes_dir', type=str, default=None,
                  help='Explicit disconnectomes directory, overriding the default '
                       '{data_path}/DISCONNECTOMES (e.g. DISCONNECTOMES400 for the 400-ROI atlas -- '
                       'DISCONNECTOMES itself currently holds VoxelAtlas_* files only)')
args.add_argument('--edge_threshold', type=float, default=75,
                  help='Percentile threshold applied independently to each layer (default: 75, '
                       'matching SBMNOCONV20K and the original run_null.py)')
args.add_argument('--fixed_iter', type=int, default=20000,
                  help='Fixed number of main-loop MCMC sweeps, no early stop (default: 20000, '
                       'matching SBMNOCONV20K)')
args.add_argument('--window_size', type=int, default=250,
                  help='Number of trailing iterations reused as the accumulation window (default: 250)')
args.add_argument('--multiflip', action='store_true', default=False,
                  help='Use multiflip_mcmc_sweep instead of mcmc_sweep (default: singleflip -- '
                       'multiflip is far more expensive, see file header cost estimate)')
args.add_argument('--seed', type=int, default=42, help='Base random seed (default: 42)')
args.add_argument('--behaviour_dist', type=str, default='normal', choices=['normal', 'poisson'])
args.add_argument('--cooccurrence_dist', type=str, default='normal', choices=['normal', 'poisson'])
args.add_argument('--combined_layers', action=argparse.BooleanOptionalAction, default=False)
args.add_argument('--n_permutations', type=int, default=100,
                  help='Number of behaviour-permutation null iterations (default: 100)')
args.add_argument('--start_perm', type=int, default=0,
                  help='Permutation index to start from (default: 0). Use to resume a run '
                       'without overwriting already-completed perm_XXXXX directories.')
args.add_argument('--n_threads', type=int, default=1,
                  help='OpenMP threads for graph_tool MCMC fitting (default: 1 -- keep this at 1 '
                       'when running several permutation processes in parallel on the same '
                       'machine, to avoid oversubscribing cores)')
args = args.parse_args()

gt.openmp_set_num_threads(args.n_threads)

fit_suffix = '_multiflip' if args.multiflip else '_singleflip'
output_dir = os.path.join(args.data_path, 'SBMNULL_NOCONV', f'SBM_{args.atlas}_{args.score}_NULL{fit_suffix}')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

log_msg(f"| START | Behaviour-permutation null model (noconv, fixed_iter={args.fixed_iter}, "
        f"{'multiflip' if args.multiflip else 'singleflip'})")
log_msg(f"| UPDATE | Data path: {args.data_path}")
log_msg(f"| UPDATE | Behaviour score: {args.score}")
log_msg(f"| UPDATE | Permutations: {args.n_permutations} (starting at {args.start_perm})")


#################################
#          LOAD DATA            #
#################################

disconnectomes_dir = args.disconnectomes_dir or os.path.join(args.data_path, 'DISCONNECTOMES')
discos = os.listdir(disconnectomes_dir)
subject_list = [f.split('_')[0] for f in discos if f.endswith(f'_{args.atlas}.tsv')]
part = np.genfromtxt(os.path.join(args.data_path, 'participants.tsv'), dtype=str, delimiter='\t')
score_col = np.where(part[0] == args.score)[0][0]

atlas_meta = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[0, :].tolist()
node_names = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, atlas_meta.index('label')].tolist()

subject_list_clean, behaviour, adj_matrices, subjects_missing_score, empty_subjects = load_graphs(
    args.data_path, args.atlas, subject_list, part, score_col, disconnectomes_dir=disconnectomes_dir)

log_msg(f"| UPDATE | Total subjects: {len(subject_list)}")
log_msg(f"| UPDATE | Included: {len(subject_list_clean)}")
log_msg(f"| UPDATE | Missing {args.score}: {len(subjects_missing_score)}")
log_msg(f"| UPDATE | Empty disconnectome: {len(empty_subjects)}")


#################################
#           ATLAS DATA          #
#################################

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)


#################################
#     SAVE ONE PERMUTATION      #
#################################

def _save_permutation_outputs(perm_dir, graph, results):
    """
    Same output set as run_null.py's _save_permutation_outputs (final
    graph, entropy trajectories, ROI block-assignment table with per-level
    block z-scores, entropy plot, block scores/connectivity tables) --
    identical file names/columns so block_pvalues_noconv.py-style
    downstream analysis works unchanged against either null.
    """

    state_nested       = results['state']
    g                  = state_nested.g
    meaningful_levels  = results['meaningful_levels']
    modal_assignments  = results['modal_assignments']
    block_connectivity = results['block_connectivity']
    node_consistency   = results['node_consistency']

    final_graph_path = os.path.join(perm_dir, f'SBM_final_graph_{args.score}.gt')
    g.save(final_graph_path)

    log_msg(f"| UPDATE | Total model entropy: {results['entropy']:.2f}")
    log_msg(f"| UPDATE | Hierarchy levels: {results['n_levels']} total, "
            f"{len(meaningful_levels)} meaningful ({meaningful_levels})")

    #################################
    #        SCALAR OUTPUTS         #
    #################################

    np.save(os.path.join(perm_dir, f'entropy_trajectory_{args.score}.npy'),
            results['entropy_trajectory'])
    np.save(os.path.join(perm_dir, f'entropy_converged_{args.score}.npy'),
            results['entropy_converged'])

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

    block_edge_mean = np.diag(results['edge_mean'])
    block_edge_var  = np.diag(results['edge_var'])

    #################################
    #    PER-LEVEL BLOCK Z-SCORES   #
    #################################

    block_scores_by_level = {}
    block_zscore_by_level = {}
    node_zscore_by_level  = {}
    for level_idx in meaningful_levels:
        b_level  = modal_assignments[level_idx]
        n_blocks = int(b_level.max()) + 1

        block_scores = np.zeros(n_blocks)
        for blk in range(n_blocks):
            nodes_in_block = np.where(b_level == blk)[0]
            if len(nodes_in_block) == 0:
                continue
            w    = np.clip(node_consistency[level_idx][nodes_in_block], 0, None)
            vals = beh_degree[nodes_in_block]
            block_scores[blk] = np.average(vals, weights=w) if w.sum() > 0 else vals.mean()

        score_std    = block_scores.std()
        block_zscore = (block_scores - block_scores.mean()) / score_std if score_std > 0 \
                       else np.zeros_like(block_scores)

        block_scores_by_level[level_idx] = block_scores
        block_zscore_by_level[level_idx] = block_zscore
        node_zscore_by_level[level_idx]  = block_zscore[b_level]

    roi_level_path = os.path.join(perm_dir, f'roi_block_assignments_{args.score}.csv')
    level_cols     = [f'level_{k}'       for k in meaningful_levels]
    consist_cols   = [f'consistency_{k}' for k in meaningful_levels]
    zscore_cols    = [f'zscore_{k}'      for k in meaningful_levels]
    header         = ['roi_index', 'roi_name', 'behaviour_degree', 'cooccurrence_degree',
                      'block_edge_mean', 'block_edge_var'] + level_cols + consist_cols + zscore_cols
    with open(roi_level_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for node_idx, roi_name in enumerate(node_names):
            block_vals   = [int(modal_assignments[k][node_idx])            for k in meaningful_levels]
            consist_vals = [round(float(node_consistency[k][node_idx]), 6) for k in meaningful_levels]
            zscore_vals  = [round(float(node_zscore_by_level[k][node_idx]), 6) for k in meaningful_levels]
            row = [node_idx, roi_name,
                   round(float(beh_degree[node_idx]), 6),
                   round(float(occ_degree[node_idx]), 6),
                   round(float(block_edge_mean[node_idx]), 6),
                   round(float(block_edge_var[node_idx]), 6)] + block_vals + consist_vals + zscore_vals
            writer.writerow(row)
    log_msg(f"| UPDATE | ROI block-assignment table saved ({len(node_names)} ROIs x "
            f"{len(meaningful_levels)} levels, incl. block z-scores) -> {roi_level_path}")

    #################################
    #        VISUALISATIONS         #
    #################################

    dS      = results['entropy_trajectory']
    dS_conv = results['entropy_converged']
    collect_from = results['convergence_iteration']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.plot(dS, linewidth=1, alpha=0.7, color='steelblue', label='MCMC entropy')
    ax.axvspan(collect_from, len(dS) - 1, color='seagreen', alpha=0.15,
              label=f'Accumulation window (last {args.window_size} iters)')
    ax.set_xlabel('MCMC Iteration', fontsize=11)
    ax.set_ylabel('Model Entropy (Description Length)', fontsize=11)
    ax.set_title(f'Entropy Trajectory -- fixed {args.fixed_iter} iterations, no change-point criterion',
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

    plt.suptitle(f'MCMC Entropy -- {args.score} ({args.atlas}) -- Null Permutation, '
                 f'fixed iterations, no change-point criterion',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(perm_dir, f'SBM_entropy_trajectory_{args.score}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    log_msg(f"| UPDATE | Entropy trajectory saved")

    #################################
    #    BLOCK COMMUNITY VOLUME     #
    #################################

    for level_idx in meaningful_levels:
        b_level      = modal_assignments[level_idx]
        n_blocks     = int(b_level.max()) + 1
        block_scores = block_scores_by_level[level_idx]
        block_zscore = block_zscore_by_level[level_idx]

        score_path = os.path.join(perm_dir, f'SBM_block_scores_lvl{level_idx}_{args.score}.csv')
        with open(score_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['block', 'n_nodes', 'score', 'zscore'])
            for blk in range(n_blocks):
                writer.writerow([blk, int((b_level == blk).sum()),
                                 round(float(block_scores[blk]), 6),
                                 round(float(block_zscore[blk]), 6)])
        log_msg(f"| UPDATE | Level {level_idx}: block scores/z-scores saved "
                f"({n_blocks} blocks) -> {score_path}")

        zscore_by_node = node_zscore_by_level[level_idx]
        zscore_data    = np.zeros_like(atlas_data, dtype=np.float32)
        valid_mask     = (atlas_data > 0) & (atlas_data <= len(zscore_by_node))
        zscore_data[valid_mask] = zscore_by_node[atlas_data[valid_mask] - 1]
        zscore_nii_path = os.path.join(perm_dir,
                                       f'{args.atlas}_lvl{level_idx}_blockzscores_{args.score}.nii.gz')
        zscore_img = nib.Nifti1Image(zscore_data, atlas_img.affine, atlas_img.header)
        nib.save(zscore_img, zscore_nii_path)

        bmat     = block_connectivity[level_idx]
        csv_path = os.path.join(perm_dir,
                                f'SBM_block_connectivity_lvl{level_idx}_{args.score}.csv')
        np.savetxt(csv_path, bmat, delimiter=',', fmt='%.6f')


#################################
#     PERMUTATION NULL LOOP     #
#################################

for perm_idx in range(args.start_perm, args.n_permutations):
    perm_seed = args.seed + perm_idx
    # Per-permutation RNG keyed on perm_seed (not one shared RNG advanced
    # across the loop) -- see run_null.py's identical comment for why this
    # matters for --start_perm resumability.
    rng = np.random.default_rng(perm_seed)
    perm_dir  = os.path.join(output_dir, f'perm_{perm_idx:05d}')
    if not os.path.isdir(perm_dir):
        os.makedirs(perm_dir)

    log_msg(f"| UPDATE | Permutation {perm_idx + 1}/{args.n_permutations}")

    permuted_behaviour = permute_behaviour(behaviour, rng)

    #################################
    #         BUILD GRAPH           #
    #################################

    graph = create_multilayer_graph(adj_matrices, permuted_behaviour, node_names,
                                   edge_threshold=args.edge_threshold,
                                   behaviour_dist=args.behaviour_dist,
                                   cooccurrence_dist=args.cooccurrence_dist,
                                   combined_layers=args.combined_layers)

    graph_path = os.path.join(perm_dir, f'SBM_graph_{args.score}.gt')
    graph.save(graph_path)

    occ_layer, beh_layer = get_graph_layers(graph)
    np.savetxt(os.path.join(perm_dir, f'SBM_layer_{args.score}_cooccurrence.txt'), occ_layer, fmt='%.6f')
    np.savetxt(os.path.join(perm_dir, f'SBM_layer_{args.score}_behaviour.txt'), beh_layer, fmt='%.6f')

    #################################
    #          FIT MODEL            #
    #################################

    results = fit_nested_sbm_layered_noconv(
        graph,
        fixed_iter=args.fixed_iter,
        window_size=args.window_size,
        behaviour_dist=args.behaviour_dist,
        cooccurrence_dist=args.cooccurrence_dist,
        multiflip=args.multiflip,
        seed=perm_seed
    )

    _save_permutation_outputs(perm_dir, graph, results)

    # see run_null.py's identical comment: breaks reference cycles between
    # the NestedBlockState and the accumulation buffers so memory doesn't
    # grow unbounded across permutations.
    del results, graph
    gc.collect()

log_msg(f"| FINISHED | Null model outputs saved -> {output_dir}")
