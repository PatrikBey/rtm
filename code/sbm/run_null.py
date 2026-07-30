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
# Behaviour-permutation null model for the multi-layer nested SBM       #
# pipeline (run.py). Shuffles the subject -> behaviour-score assignment #
# (lesion data untouched), rebuilds the graph, and re-runs the full     #
# MCMC-based convergence fit used in run.py, n_permutations times, so   #
# that the real (unpermuted) fit can later be compared against the     #
# resulting distribution of null outcomes.                              #
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

from functions import create_multilayer_graph, permute_behaviour
from functions import fit_nested_sbm_layered
from utils import load_graphs, log_msg, get_graph_layers


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Behaviour-permutation null model for the nested SBM pipeline.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
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
                  help='Base random seed (default: 42)')
args.add_argument('--behaviour_dist', type=str, default='normal',
                  choices=['normal', 'poisson'],
                  help='Assumed distribution for the behaviour layer (default: normal)')
args.add_argument('--cooccurrence_dist', type=str, default='normal',
                  choices=['normal', 'poisson'],
                  help='Assumed distribution for the cooccurrence layer (default: normal). '
                       'Set to poisson to apply [0,1] rescaling and use discrete-Poisson DL.')
args.add_argument('--combined_layers', action=argparse.BooleanOptionalAction, default=False,
                  help='If set, threshold both layers jointly so they share the same edge '
                       'structure (intersection). Must match the setting used for the real run.')
args.add_argument('--n_permutations', type=int, default=1000,
                  help='Number of behaviour-permutation null iterations (default: 1000)')
args.add_argument('--start_perm', type=int, default=0,
                  help='Permutation index to start from (default: 0). Use to resume a run '
                       'without overwriting already-completed perm_XXXXX directories.')
args = args.parse_args()

log_msg(f"| START | Behaviour-permutation null model")
log_msg(f"| UPDATE | Data path: {args.data_path}")
log_msg(f"| UPDATE | Behaviour score: {args.score}")
log_msg(f"| UPDATE | Permutations: {args.n_permutations}")


output_dir = os.path.join(args.data_path, 'SBMNULL', f'SBM_{args.atlas}_{args.score}_NULL_29')

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


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

subject_list_clean, behaviour, adj_matrices, subjects_missing_score, empty_subjects = load_graphs(
    args.data_path, args.atlas, subject_list, part, score_col)

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
    Persist a single permutation's fit outputs: final graph, entropy
    trajectories, ROI block-assignment table (including per-ROI block
    z-scores for every meaningful level), entropy-trajectory plot, block
    scores/connectivity tables, and the combined block z-score NIfTI (the
    only NIfTI output produced here). Block z-scores are the statistic later
    compared against the null distribution built by concatenating
    roi_block_assignments across permutations.

    No per-permutation state-drawing visualisation is produced here — the
    saved final graph (.gt) and roi_block_assignments CSV (per-node block
    labels per level) carry everything visualizations/create_figures.py's
    utils.plot_sbm_state() needs to redraw the block state later, on demand,
    from disk (see how it's used for the real/unpermuted run's outputs).
    """

    state_nested       = results['state']
    g                  = state_nested.g
    meaningful_levels  = results['meaningful_levels']
    modal_assignments  = results['modal_assignments']   # {level: (n_nodes,) array}
    block_connectivity = results['block_connectivity']  # {level: (B×B) joint matrix}
    node_consistency   = results['node_consistency']    # {level: (n_nodes,) array in [0,1]}

    final_graph_path = os.path.join(perm_dir, f'SBM_final_graph_{args.score}.gt')
    g.save(final_graph_path)

    log_msg(f"| UPDATE | Total model entropy: {results['entropy']:.2f}")
    log_msg(f"| UPDATE | Hierarchy levels: {results['n_levels']} total, "
            f"{len(meaningful_levels)} meaningful ({meaningful_levels})")
    log_msg(f"| UPDATE | Converged: {results['converged']} "
            f"(iteration {results['convergence_iteration']}, "
            f"{results['n_converged_samples']} accumulation samples)")

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

    # Block score = consistency-weighted mean behaviour degree within the block;
    # z-scored across blocks within each level, then broadcast back to nodes so
    # the ROI table carries a per-ROI z-score for every meaningful level. This is
    # the statistic later compared against the null distribution (concatenating
    # roi_block_assignments z-scores across permutations).
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
    log_msg(f"| UPDATE | ROI block-assignment table saved ({len(node_names)} ROIs × "
            f"{len(meaningful_levels)} levels, incl. block z-scores) → {roi_level_path}")

    #################################
    #        VISUALISATIONS         #
    #################################

    # ---- Entropy trajectory ---- #
    conv_iter = results['convergence_iteration']
    n_conv    = results['n_converged_samples']
    converged = results['converged']
    dS        = results['entropy_trajectory']
    dS_conv   = results['entropy_converged']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.plot(dS, linewidth=1, alpha=0.7, color='steelblue', label='MCMC entropy')
    if converged:
        ax.axvline(conv_iter, color='firebrick', linewidth=1.5, linestyle='--',
                   label=f'Mean-shift change point (iter {conv_iter})')
        ax.axhline(results.get('threshold', np.nan), color='firebrick',
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

    plt.suptitle('MCMC Entropy — Weighted Nested Hierarchical SBM (Null Permutation)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(perm_dir, f'SBM_entropy_trajectory_{args.score}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    log_msg(f"| UPDATE | Entropy trajectory saved "
            f"(converged={converged}, iter={conv_iter}, {n_conv} accumulation samples)")

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
                f"({n_blocks} blocks) → {score_path}")

        zscore_by_node = node_zscore_by_level[level_idx]
        zscore_data    = np.zeros_like(atlas_data, dtype=np.float32)
        valid_mask     = (atlas_data > 0) & (atlas_data <= len(zscore_by_node))
        zscore_data[valid_mask] = zscore_by_node[atlas_data[valid_mask] - 1]
        zscore_nii_path = os.path.join(perm_dir,
                                       f'{args.atlas}_lvl{level_idx}_blockzscores_{args.score}.nii.gz')
        zscore_img = nib.Nifti1Image(zscore_data, atlas_img.affine, atlas_img.header)
        nib.save(zscore_img, zscore_nii_path)
        log_msg(f"| UPDATE | Combined block z-score NIfTI saved: level {level_idx} "
                f"(all {n_blocks} blocks) → {zscore_nii_path}")

        bmat     = block_connectivity[level_idx]
        csv_path = os.path.join(perm_dir,
                                f'SBM_block_connectivity_lvl{level_idx}_{args.score}.csv')
        np.savetxt(csv_path, bmat, delimiter=',', fmt='%.6f')
        log_msg(f"| UPDATE | Block connectivity matrix saved: level {level_idx} "
                f"({n_blocks}×{n_blocks}) → {csv_path}")


#################################
#     PERMUTATION NULL LOOP     #
#################################

for perm_idx in range(args.start_perm, args.n_permutations):
    perm_seed = args.seed + perm_idx
    # Per-permutation RNG keyed on perm_seed (rather than one shared RNG
    # advanced across the whole loop) so each permutation's behaviour shuffle
    # only depends on its own index, not on how many permutations ran before
    # it — this lets a run be resumed at --start_perm and still reproduce
    # exactly the shuffles a single unbroken 0..n_permutations run would have
    # produced for those indices.
    rng = np.random.default_rng(perm_seed)
    perm_dir  = os.path.join(output_dir, f'perm_{perm_idx:05d}')
    if not os.path.isdir(perm_dir):
        os.makedirs(perm_dir)

    log_msg(f"| UPDATE | Permutation {perm_idx + 1}/{args.n_permutations}")

    permuted_behaviour = permute_behaviour(behaviour, rng)

    #################################
    #         BUILD GRAPH           #
    #################################

    graph = create_multilayer_graph(adj_matrices, permuted_behaviour, node_names, edge_threshold=75,
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

    results = fit_nested_sbm_layered(
        graph,
        max_iter=args.max_iter,
        window_size=args.window_size,
        shift_factor=args.shift_factor,
        behaviour_dist=args.behaviour_dist,
        cooccurrence_dist=args.cooccurrence_dist,
        seed=perm_seed
    )

    _save_permutation_outputs(perm_dir, graph, results)

    # `results['state']` (graph_tool NestedBlockState) and the accumulation
    # buffers inside fit_nested_sbm_layered form reference cycles (nested
    # closures, parent/child block-state links) that plain refcounting can't
    # reclaim. Breaking the references here and forcing a cyclic collection
    # keeps peak memory bounded across permutations instead of growing until
    # the process is OOM-killed.
    del results, graph
    gc.collect()

log_msg(f"| FINISHED | Null model outputs saved → {output_dir}")
