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
# related cognitive domains using a multi-layer nested                  #
# stochastic block modelling (SBM) framework                            #
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
from functions import create_multilayer_graph
from functions import fit_nested_sbm_layered, fit_nested_sbm_layered_multiflip
from utils import load_graphs, log_msg, get_graph_layers



#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Run multi-layer nested SBM on disconnectome data.')
args.add_argument('--data_path', type=str, default='/mnt/h/RT/data', help='Path to the data directory')
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
args.add_argument('--multiflip', action='store_true', default=False,
                  help='Use multiflip_mcmc_sweep instead of mcmc_sweep (better local-minima escape)')
args.add_argument('--seed', type=int, default=42,
                  help='Random seed for graph_tool RNG (default: 42)')
args.add_argument('--behaviour_dist', type=str, default='normal',
                  choices=['normal', 'poisson'],
                  help='Assumed distribution for the behaviour layer (default: normal)')
args.add_argument('--cooccurrence_dist', type=str, default='normal',
                  choices=['normal', 'poisson'],
                  help='Assumed distribution for the cooccurrence layer (default: normal). '
                       'Set to poisson to apply [0,1] rescaling and use discrete-Poisson DL.')
args.add_argument('--combined_layers', action=argparse.BooleanOptionalAction, default=False,
                  help='If set (default), threshold both layers jointly so they share the same '
                       'edge structure (intersection). Use --no-combined-layers to threshold each '
                       'layer independently, giving each its own edge set.')
args = args.parse_args()

log_msg(f"| START | Running multi-layer nested SBM on disconnectome data")
log_msg(f"| UPDATE | Data path: {args.data_path}")
log_msg(f"| UPDATE | Behaviour score: {args.score}")

if args.multiflip:
    output_dir = os.path.join(args.data_path, 'RESULTS', f'SBM_{args.atlas}_{args.score}_multiflip')
else:
    output_dir = os.path.join(args.data_path, 'RESULTS', f'SBM_{args.atlas}_{args.score}_singleflip')

# try:
#     os.makedirs(output_dir, exist_ok=True)
# except FileExistsError:
#     pass  # WSL/NTFS can raise EEXIST spuriously

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

graph = create_multilayer_graph(adj_matrices, behaviour, node_names, edge_threshold=75,
                               behaviour_dist=args.behaviour_dist,
                               cooccurrence_dist=args.cooccurrence_dist,
                               combined_layers=args.combined_layers)

graph_path = os.path.join(output_dir, f'SBM_graph_{args.score}.gt')
graph.save(graph_path)
log_msg(f"| UPDATE | Multilayer graph saved (graph-tool format) → {graph_path}")


occ_layer, beh_layer = get_graph_layers(graph)

np.savetxt(os.path.join(output_dir, f'SBM_layer_{args.score}_cooccurrence.txt'), occ_layer, fmt='%.6f')
np.savetxt(os.path.join(output_dir, f'SBM_layer_{args.score}_behaviour.txt'), beh_layer, fmt='%.6f')

# plt.figure(figsize=(8, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(np.where(occ_layer == 0, np.nan, occ_layer), cmap='plasma')
# plt.colorbar()
# plt.title('Cooccurrence layer')
# plt.subplot(1, 2, 2)
# plt.imshow(np.where(beh_layer == 0, np.nan, beh_layer), cmap='plasma')
# plt.colorbar()
# plt.title('Behaviour layer')
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, f'RF_weighted_nested_graph_layers_{args.score}.png'),
#             dpi=150, bbox_inches='tight')
# plt.close()




#################################
#          FIT MODEL            #
#################################

if args.multiflip:
    real_results = fit_nested_sbm_layered_multiflip(
        graph,
        max_iter=args.max_iter,
        window_size=args.window_size,
        shift_factor=args.shift_factor,
        behaviour_dist=args.behaviour_dist,
        cooccurrence_dist=args.cooccurrence_dist,
        seed=args.seed
    )
else:
    real_results = fit_nested_sbm_layered(
        graph,
        max_iter=args.max_iter,
        window_size=args.window_size,
        shift_factor=args.shift_factor,
        behaviour_dist=args.behaviour_dist,
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

# print(f"\nHierarchical Block Structure:")
# for level_idx in range(real_results['n_levels']):
#     n_b  = real_results['levels_n_blocks'][level_idx]
#     entr = real_results['levels_entropy'][level_idx]
#     tag  = '  <-- meaningful' if level_idx in meaningful_levels else ''
#     print(f"  Level {level_idx}: {n_b} blocks, entropy {entr:.2f}{tag}")

# print(f"\nModal Block Assignments (converged posterior):")
# for level_idx in meaningful_levels:
#     b    = modal_assignments[level_idx]
#     n_b  = int(b.max()) + 1
#     print(f"\n  Level {level_idx}  ({n_b} blocks)")
#     print(f"  {'Node':<6}  {'Block':>5}  Name")
#     print(f"  {'-'*40}")
#     for node_idx in range(len(b)):
#         print(f"  {node_idx:<6}  {b[node_idx]:>5}  {node_names[node_idx]}")



#################################
#        SCALAR OUTPUTS         #
#################################

# --- Entropy / description length trajectory ---
# In graph_tool, state.entropy() returns the description length (DL) directly —
# the two quantities are identical in the MDL/Bayesian SBM formulation.
# entropy_trajectory = DL at each main-loop MCMC iteration;
# entropy_converged  = DL across the fixed accumulation window.
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

# Weighted degree per node, behaviour layer (layer 0) and cooccurrence layer (layer 1).
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

# Posterior mean/variance of each node's own-block internal edge mass
# (diagonal of edge_mean / edge_var — accumulated over the accumulation window).
block_edge_mean = np.diag(real_results['edge_mean'])
block_edge_var  = np.diag(real_results['edge_var'])

import csv
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
log_msg(f"| UPDATE | ROI block-assignment table saved ({len(node_names)} ROIs × "
        f"{len(meaningful_levels)} levels) → {roi_level_path}")



#################################
#        VISUALISATIONS         #
#################################

# # ---- Node colours by anatomical location ---- #
# # Locations are a fixed mapping from atlas ROIs to canonical anatomical areas —
# # consistent across atlases, so colours are hardcoded to those areas.
# loc_colours = [
#     'mediumvioletred', 'deeppink', 'indigo', 'mediumslateblue', 'steelblue',
#     'deepskyblue', 'teal', 'mediumturquoise', 'darkgreen', 'limegreen',
#     'olivedrab', 'yellowgreen', 'darkorange', 'gold', 'firebrick', 'lightcoral'
# ]
# locations        = [f'{loc}_L' if idx < dim / 2 else f'{loc}_R'
#                     for idx, loc in enumerate(locations)]
# unique_locations = sorted(set(locations))
# node_color       = [loc_colours[unique_locations.index(loc)] for loc in locations]


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

plt.suptitle('MCMC Entropy — Weighted Nested Hierarchical SBM',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'SBM_entropy_trajectory_{args.score}.png'),
            dpi=150, bbox_inches='tight')
log_msg(f"| UPDATE | Entropy trajectory saved "
        f"(converged={converged}, iter={conv_iter}, {n_conv} accumulation samples)")
plt.close()

# # plt.show()


# # ---- Block connectivity heatmaps (one figure per meaningful level) ---- #
# # Single joint matrix per level — model-internal mrs, joint across both layers.
# from matplotlib.colors import to_rgba

# for level_idx in meaningful_levels:
#     bmat = block_connectivity[level_idx]
#     b    = modal_assignments[level_idx]
#     n_b  = int(b.max()) + 1
#     fig, ax = plt.subplots(figsize=(7, 6))
#     im = ax.imshow(np.where(bmat == 0, np.nan, bmat), cmap='plasma', interpolation='nearest')
#     plt.colorbar(im, ax=ax, label='Mean block-to-block edge count (mrs)')
#     ax.set_title(f'Joint Block Connectivity\nLevel {level_idx} — {n_b} blocks', fontsize=11)
#     ax.set_xlabel('Block index')
#     ax.set_ylabel('Block index')
#     ax.set_xticks(range(n_b))
#     ax.set_yticks(range(n_b))
#     plt.suptitle(f'Block Connectivity — Level {level_idx} Modal Partition  |  {args.score}',
#                  fontsize=12, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(f'{output_dir}/SBM_block_matrix_level{level_idx}_{args.score}.png',
#                 dpi=150, bbox_inches='tight')
#     log_msg(f"| UPDATE | Block matrix saved for level {level_idx}")
#     plt.close()


# # ---- Graph visualisation (graph_tool draw, level-0 modal partition) ---- #
# print("\nGenerating graph visualisation...")

# _coord_data = np.loadtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_circle_coords_sorted.txt'),
#                          delimiter='\t', skiprows=1, usecols=(0, 1))


# pos = g.new_vertex_property("vector<double>")
# for v in g.vertices():
#     pos[v] = _coord_data[int(v)].tolist()

# unique_location_names   = sorted(set(locations))
# location_name_to_idx    = {name: idx for idx, name in enumerate(unique_location_names)}
# n_locations             = len(unique_location_names)

# distinct_colors = [to_rgba(c) for c in loc_colours]
# cmap            = ListedColormap(distinct_colors)
# norm            = Normalize(vmin=0, vmax=max(n_locations - 1, 1))

# vertex_color = g.new_vertex_property("vector<double>")
# vertex_shape = g.new_vertex_property("int")
# for v in g.vertices():
#     node_idx     = int(v)
#     location     = locations[node_idx]
#     side         = location.rsplit('_', 1)[1]
#     location_idx = location_name_to_idx[location]
#     rgba         = cmap(norm(location_idx))
#     vertex_color[v] = rgba
#     vertex_shape[v] = 0 if side == 'L' else 1

# degree_map   = g.degree_property_map("in")
# vertex_sizes = gt.prop_to_size(degree_map, mi=20, ma=50)

# # Edge alpha: behaviour layer weight for each node pair.
# # Build a lookup by node pair from behaviour layer edges (layer == 0).
# # Cooccurrence edges share the same node pair so they pick up the same weight.
# # Z-scored weights can be negative (below-mean edges); these clip to alpha_min
# # so only edges with meaningful positive behaviour signal are opaque.
# _e_alpha_min   = 0.03
# _e_alpha_gamma = 1.0

# beh_weight_lookup = {}
# for e in g.edges():
#     if g.ep.layer[e] == 0:
#         key = (min(int(e.source()), int(e.target())),
#                max(int(e.source()), int(e.target())))
#         beh_weight_lookup[key] = float(g.ep.behaviour_weight[e])

# raw_beh = np.array([
#     beh_weight_lookup.get(
#         (min(int(e.source()), int(e.target())),
#          max(int(e.source()), int(e.target()))), 0.0)
#     for e in g.edges()
# ])

# clipped  = np.maximum(raw_beh, 0.0)
# w_max    = clipped.max() if clipped.max() > 0 else 1.0
# edge_alpha_arr = _e_alpha_min + (1.0 - _e_alpha_min) * (clipped / w_max) ** _e_alpha_gamma

# edge_color = g.new_edge_property("vector<double>")
# for idx, e in enumerate(g.edges()):
#     src_c   = vertex_color[e.source()]
#     tgt_c   = vertex_color[e.target()]
#     avg_rgb = [(src_c[c] + tgt_c[c]) / 2 for c in range(3)]
#     avg_rgb.append(float(edge_alpha_arr[idx]))
#     edge_color[e] = tuple(avg_rgb)

# state_nested.draw(
#     pos=pos,
#     vertex_fill_color=vertex_color,
#     vertex_shape=vertex_shape,
#     vertex_size=vertex_sizes,
#     vertex_pen_width=0.5,
#     edge_color=edge_color,
#     edge_pen_width=gt.prop_to_size(g.ep.behaviour_weight, mi=0.5, ma=3),
#     edge_gradient=[],
#     vertex_text=g.vp.label,
#     vertex_text_color='black',
#     vertex_text_position=0,
#     vertex_font_size=10,
#     output=os.path.join(output_dir, f"RF_weighted_nested_block_state_draw_{args.score}.png"),
#     output_size=(1200, 1200)
# )
# log_msg(f"| UPDATE | Graph visualisation saved")


# # ---- Location legend ---- #
# fig, ax = plt.subplots(figsize=(3, 2))
# legend_elements = []
# for loc_idx, location_name in enumerate(unique_location_names):
#     rgba   = cmap(norm(loc_idx))
#     n_node = np.sum(np.array(locations) == location_name)
#     marker = 'o' if location_name.endswith('_L') else '^'
#     legend_elements.append(
#         Line2D([0], [0], marker=marker, color='w',
#                markerfacecolor=rgba, markersize=10,
#                markeredgecolor='black', markeredgewidth=1.5,
#                label=f"{location_name} ({n_node} nodes)")
#     )

# ax.legend(handles=legend_elements, loc='center', fontsize=10, frameon=True,
#           title="Node colour by location (circles=L, triangles=R)",
#           title_fontsize=12, ncol=2, fancybox=True, shadow=True)
# ax.axis('off')
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, f'RF_weighted_nested_location_legend_{args.score}.png'),
#             dpi=150, bbox_inches='tight')
# plt.close()


# # ---- Connectome plot: three-ring hierarchical circle ---- #
# #
# # Layout:
# #   Outer ring   — anatomical nodes at atlas coordinates, coloureds by location,
# #                  shaped by hemisphere (circles = L, triangles = R).
# #   Middle ring  — level-0 block nodes, evenly spaced at 66% of outer radius.
# #                  Spokes connect each outer node to its level-0 block (Bézier,
# #                  control point = outer-circle point at the block's radial angle).
# #   Inner ring   — level-1 block nodes, evenly spaced at 35% of outer radius.
# #                  Straight edges connect each level-0 block node to its level-2
# #                  block (majority-vote mapping over member nodes).

# outer_coord_data = np.loadtxt(
#     os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_circle_coords_sorted.txt'),
#     delimiter='\t', skiprows=1, usecols=(0, 1)
# ).astype(float)

# b0   = modal_assignments[meaningful_levels[0]]
# n_b0 = int(b0.max()) + 1

# # Inner ring uses the second meaningful level if it exists; otherwise omit it.
# _has_inner = len(meaningful_levels) >= 2
# b1   = modal_assignments[meaningful_levels[1]] if _has_inner else None
# n_b1 = int(b1.max()) + 1                       if _has_inner else 0

# # Outer circle geometry
# cx_outer = outer_coord_data[:, 0].mean()
# cy_outer = outer_coord_data[:, 1].mean()
# r_outer  = np.hypot(outer_coord_data[:, 0] - cx_outer,
#                     outer_coord_data[:, 1] - cy_outer).max()
# r_middle = r_outer * 0.66   # level-0 blocks
# r_inner  = r_outer * 0.35   # level-2 blocks

# # Middle ring: level-0 block positions
# angles_b0 = np.linspace(0, 2 * np.pi, n_b0, endpoint=False)
# middle_xy = np.column_stack([
#     cx_outer + r_middle * np.cos(angles_b0),
#     cy_outer + r_middle * np.sin(angles_b0)
# ])

# # Bézier control points for outer→middle spokes
# ctrl_xy = np.column_stack([
#     cx_outer + r_outer * np.cos(angles_b0),
#     cy_outer + r_outer * np.sin(angles_b0)
# ])

# # Inner ring: second meaningful level block positions (omitted if only one level)
# if _has_inner:
#     angles_b1 = np.linspace(0, 2 * np.pi, n_b1, endpoint=False)
#     inner_xy  = np.column_stack([
#         cx_outer + r_inner * np.cos(angles_b1),
#         cy_outer + r_inner * np.sin(angles_b1)
#     ])
#     b0_to_b1 = np.zeros(n_b0, dtype=int)
#     for blk0 in range(n_b0):
#         nodes_in = np.where(b0 == blk0)[0]
#         if nodes_in.size > 0:
#             b0_to_b1[blk0] = int(np.bincount(b1[nodes_in], minlength=n_b1).argmax())
# else:
#     inner_xy = None
#     b0_to_b1 = None

# # Normalise behavioural weighted degree to [alpha_min, 1.0] for node/spoke alpha.
# # A power transform (gamma > 1) strongly compresses low-degree nodes toward
# # alpha_min so they are barely visible, while high-degree nodes stay opaque.
# _deg_min    = float(beh_degree.min())
# _deg_max    = float(beh_degree.max())
# _deg_range  = _deg_max - _deg_min if _deg_max > _deg_min else 1.0
# _alpha_min  = 0.03
# _alpha_gamma = 1.0   # increase to push low-degree nodes further toward invisible
# log_msg(f"| UPDATE | Behaviour degree: "
#         f"min={_deg_min:.3f}  max={_deg_max:.3f}  range={_deg_range:.3f}")

# t_bez = np.linspace(0, 1, 80)
# idx_L = [i for i, loc in enumerate(locations) if loc.endswith('_L')]
# idx_R = [i for i, loc in enumerate(locations) if loc.endswith('_R')]

# n_cols    = min(n_b0, 4)
# n_rows    = int(np.ceil(n_b0 / n_cols))
# fig, axes = plt.subplots(n_rows, n_cols,
#                          figsize=(5.5 * n_cols, 5.5 * n_rows),
#                          squeeze=False)
# axes_flat = axes.flatten()

# for blk_feat in range(n_b0):
#     ax = axes_flat[blk_feat]
#     ax.set_aspect('equal')
#     ax.axis('off')
#     members     = np.where(b0 == blk_feat)[0]
#     non_members = np.where(b0 != blk_feat)[0]
#     # ---- Spokes: outer node → its level-0 block node ---- #
#     # Alpha = normalised behavioural weighted degree.
#     # High-degree nodes are opaque; low-degree nodes are transparent.
#     for node_idx in members:
#         x1, y1    = outer_coord_data[node_idx]
#         x2, y2    = middle_xy[blk_feat]
#         cpx, cpy  = ctrl_xy[blk_feat]
#         alpha_node = float(_alpha_min + (1.0 - _alpha_min) *
#                            ((beh_degree[node_idx] - _deg_min) / _deg_range) ** _alpha_gamma)
#         xc = (1 - t_bez)**2 * x1 + 2*(1 - t_bez)*t_bez * cpx + t_bez**2 * x2
#         yc = (1 - t_bez)**2 * y1 + 2*(1 - t_bez)*t_bez * cpy + t_bez**2 * y2
#         ax.plot(xc, yc, color=node_color[node_idx],
#                 alpha=alpha_node, linewidth=0.7, zorder=2)
#     # ---- Middle→inner edges and inner ring (only if a second level exists) ---- #
#     if _has_inner:
#         for blk0 in range(n_b0):
#             blk1 = int(b0_to_b1[blk0])
#             ax.plot([middle_xy[blk0, 0], inner_xy[blk1, 0]],
#                     [middle_xy[blk0, 1], inner_xy[blk1, 1]],
#                     color='dimgray', alpha=0.5, linewidth=1.5, zorder=3)
#         ax.scatter(inner_xy[:, 0], inner_xy[:, 1],
#                    s=150, c='white', edgecolors='dimgray', linewidths=1.5, zorder=4)
#         for b in range(n_b1):
#             ax.text(inner_xy[b, 0], inner_xy[b, 1], str(b),
#                     ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
#     # ---- Middle ring: level-0 block nodes (featured block highlighted) ---- #
#     c_mid  = ['gold' if b == blk_feat else 'white' for b in range(n_b0)]
#     lw_mid = [2.5   if b == blk_feat else 1.5     for b in range(n_b0)]
#     ax.scatter(middle_xy[:, 0], middle_xy[:, 1],
#                s=200, c=c_mid, edgecolors='dimgray', linewidths=lw_mid, zorder=6)
#     for b in range(n_b0):
#         ax.text(middle_xy[b, 0], middle_xy[b, 1], str(b),
#                 ha='center', va='center', fontsize=8, fontweight='bold', zorder=7)
#     # ---- Outer nodes: non-members grayed, members colored by location ---- #
#     # Member node alpha matches spoke alpha (normalised behavioural degree).
#     nm_L = [i for i in non_members if locations[i].endswith('_L')]
#     nm_R = [i for i in non_members if locations[i].endswith('_R')]
#     m_L  = [i for i in members     if locations[i].endswith('_L')]
#     m_R  = [i for i in members     if locations[i].endswith('_R')]
#     def _node_rgba(node_idx):
#         a   = _alpha_min + (1.0 - _alpha_min) * \
#               ((beh_degree[node_idx] - _deg_min) / _deg_range) ** _alpha_gamma
#         r, g, b, _ = to_rgba(node_color[node_idx])
#         return (r, g, b, float(a))
#     if nm_L:
#         ax.scatter(outer_coord_data[nm_L, 0], outer_coord_data[nm_L, 1],
#                    s=20, c='lightgray', marker='o', edgecolors='none',
#                    alpha=0.3, zorder=6)
#     if nm_R:
#         ax.scatter(outer_coord_data[nm_R, 0], outer_coord_data[nm_R, 1],
#                    s=20, c='lightgray', marker='^', edgecolors='none',
#                    alpha=0.3, zorder=6)
#     if m_L:
#         ax.scatter(outer_coord_data[m_L, 0], outer_coord_data[m_L, 1],
#                    s=80, c=[_node_rgba(i) for i in m_L],
#                    marker='o', edgecolors='none', zorder=8)
#     if m_R:
#         ax.scatter(outer_coord_data[m_R, 0], outer_coord_data[m_R, 1],
#                    s=80, c=[_node_rgba(i) for i in m_R],
#                    marker='^', edgecolors='none', zorder=8)
#     ax.set_title(f'Block {blk_feat}  ({len(members)} nodes)',
#                  fontsize=10, fontweight='bold')

# # Hide unused subplot panels
# for ax in axes_flat[n_b0:]:
#     ax.set_visible(False)

# fig.suptitle(f'Level-0 Block Connectome — per block  |  {args.score}',
#              fontsize=13, fontweight='bold')
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, f'SBM_block_connectome_per_block_{args.score}.svg'),
#             dpi=150, bbox_inches='tight')

# plt.close()


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
# Only blocks that are informative about the JOINT lesion/behaviour structure
# are saved: per block, score = consistency-weighted mean behaviour_degree
# of its member nodes (node_consistency acts as a confidence weight on how
# reliably each node belongs to that block under the joint model; negative
# kappa values are clipped to 0 so unreliable nodes don't invert the weighting).
# Scores are z-scored across blocks within a level; only blocks with z > 0
# (above-average joint informativeness) are written out.

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)

for level_idx in meaningful_levels:
    b_level  = modal_assignments[level_idx]
    n_blocks = int(b_level.max()) + 1

    # ---- Score blocks by joint informativeness ---- #
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
    # behaviour score (consistency-weighted mean behaviour_degree) and 0 elsewhere.
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