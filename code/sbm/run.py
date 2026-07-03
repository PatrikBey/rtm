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
from functions import fit_nested_sbm_layered
from utils import load_graphs, log_msg, get_graph_layers



#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Run multi-layer nested SBM on disconnectome data.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/DATA', help='Path to the data directory')
args.add_argument('--score', type=str, default='Foreperiod_Long_tau', help='Behaviour score to analyze')
args.add_argument('--atlas', type=str, default='HCP-MMP1', help='Atlas name')
args.add_argument('--mcmc_samples', type=int, default=50000, help='Number of MCMC samples for fitting the SBM')
args.add_argument('--burn_in', type=int, default=10, help='Number of burn-in iterations for MCMC')
args.add_argument('--annealing_temps', type=float, nargs=2, default=(1, 5), help='Annealing temperatures for MCMC')
args.add_argument('--annealing_steps', type=int, default=1, help='Number of annealing steps for CMC')
args.add_argument('--convergence_fraction', type=float, default=0.05,
                  help='Fraction of total entropy reduction used to define convergence threshold. '
                       'Only MCMC iterations in the converged tail (remaining reduction <= this '
                       'fraction of total) are used for partition accumulation (default: 0.05)')
args = args.parse_args()

log_msg(f"| START | Running multi-layer nested SBM on disconnectome data")
log_msg(f"| UPDATE | Data path: {args.data_path}")
log_msg(f"| UPDATE | Behaviour score: {args.score}")

output_dir = os.path.join(args.data_path, 'RESULTS', f'SBM_{args.atlas}_{args.score}')

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
node_names = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, 0].tolist()
locations  = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, 2].tolist()
dim = len(node_names)

subject_list_clean, behaviour, adj_matrices, subjects_missing_score, empty_subjects = load_graphs(args.data_path, args.atlas, subject_list, part, score_col)

log_msg(f"| UPDATE | Total subjects: {len(subject_list)}")
log_msg(f"| UPDATE | Included: {len(subject_list_clean)}")
log_msg(f"| UPDATE | Missing {args.score}: {len(subjects_missing_score)}")
log_msg(f"| UPDATE | Empty disconnectome: {len(empty_subjects)}")



#################################
#         BUILD GRAPH           #
#################################

graph = create_multilayer_graph(adj_matrices, behaviour, node_names, edge_threshold=25)
occ_layer, beh_layer = get_graph_layers(graph)

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.where(occ_layer == 0, np.nan, occ_layer), cmap='plasma')
plt.colorbar()
plt.title('Cooccurrence layer')
plt.subplot(1, 2, 2)
plt.imshow(np.where(beh_layer == 0, np.nan, beh_layer), cmap='plasma')
plt.colorbar()
plt.title('Behaviour layer')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'RF_weighted_nested_graph_layers_{args.score}.png'),
            dpi=150, bbox_inches='tight')
plt.close()




#################################
#          FIT MODEL            #
#################################

real_results = fit_nested_sbm_layered(
    graph,
    mcmc_samples=args.mcmc_samples,
    burn_in=args.burn_in,
    annealing_temps=(args.annealing_temps[0], args.annealing_temps[1]),
    annealing_steps=args.annealing_steps,
    convergence_fraction=args.convergence_fraction
)

state_nested      = real_results['state']
g                 = state_nested.g
meaningful_levels = real_results['meaningful_levels']
modal_assignments  = real_results['modal_assignments']   # {level: (n_nodes,) array}
block_connectivity = real_results['block_connectivity']  # {level: (B×B) joint matrix}
node_consistency   = real_results['node_consistency']    # {level: (n_nodes,) array in [0,1]}



#################################
#       PRINT BLOCK STRUCTURE   #
#################################

log_msg(f"| UPDATE | Total model entropy: {real_results['entropy']:.2f}")
log_msg(f"| UPDATE | Hierarchy levels: {real_results['n_levels']} total, "
        f"{len(meaningful_levels)} meaningful ({meaningful_levels})")
log_msg(f"| UPDATE | Convergence at iteration {real_results['convergence_iteration']} "
        f"({real_results['n_converged_samples']} accumulation samples)")

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
# entropy_trajectory = DL at each Phase 1 iteration;
# entropy_converged  = DL at each converged Phase 2 iteration.
np.save(os.path.join(output_dir, f'entropy_trajectory_{args.score}.npy'),
        real_results['entropy_trajectory'])
np.save(os.path.join(output_dir, f'entropy_converged_{args.score}.npy'),
        real_results['entropy_converged'])
log_msg(f"| UPDATE | Entropy / DL trajectory saved "
        f"({len(real_results['entropy_trajectory'])} Phase-1 + "
        f"{len(real_results['entropy_converged'])} Phase-2 samples)")

# --- ROI × level block-assignment table ---
# Rows = ROIs, columns = meaningful level indices.
# Values are 0-indexed block IDs from the modal partition.
import csv
roi_level_path = os.path.join(output_dir, f'roi_block_assignments_{args.score}.csv')
level_cols     = [f'level_{k}' for k in meaningful_levels]
with open(roi_level_path, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['roi_index', 'roi_name'] + level_cols)
    for node_idx, roi_name in enumerate(node_names):
        row = [node_idx, roi_name] + [int(modal_assignments[k][node_idx])
                                       for k in meaningful_levels]
        writer.writerow(row)
log_msg(f"| UPDATE | ROI block-assignment table saved ({len(node_names)} ROIs × "
        f"{len(meaningful_levels)} levels) → {roi_level_path}")



#################################
#        VISUALISATIONS         #
#################################

# ---- Node colours by anatomical location ---- #
loc_colours = [
    'mediumvioletred', 'deeppink', 'indigo', 'mediumslateblue', 'steelblue',
    'deepskyblue', 'teal', 'mediumturquoise', 'darkgreen', 'limegreen',
    'olivedrab', 'yellowgreen', 'darkorange', 'gold', 'firebrick', 'lightcoral'
]
locations      = [f'{loc}_L' if idx < dim / 2 else f'{loc}_R'
                  for idx, loc in enumerate(locations)]
unique_locations = sorted(set(locations))
node_color     = [loc_colours[unique_locations.index(loc)] for loc in locations]


# ---- Entropy trajectory ---- #
t_star  = real_results['convergence_iteration']
n_conv  = real_results['n_converged_samples']
dS      = real_results['entropy_trajectory']
dS_conv = real_results['entropy_converged']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.plot(dS, linewidth=1, alpha=0.7, color='steelblue', label='Phase 1 (tracking)')
ax.axvline(t_star, color='firebrick', linewidth=1.5, linestyle='--',
           label=f't* = {t_star}  ({args.convergence_fraction * 100:.0f}% threshold)')
ax.axhspan(dS.min(), dS.min() + args.convergence_fraction * (dS[0] - dS.min()),
           alpha=0.08, color='firebrick', label='Converged entropy band')
ax.set_xlabel('MCMC Iteration', fontsize=11)
ax.set_ylabel('Model Entropy (Description Length)', fontsize=11)
ax.set_title('Phase 1: Entropy Tracking', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(dS_conv, linewidth=1, alpha=0.8, color='seagreen',
        label=f'Phase 2 ({n_conv} converged samples)')
ax.set_ylim(dS.min(), dS.max())
ax.set_xlabel('MCMC Iteration', fontsize=11)
ax.set_ylabel('Model Entropy (Description Length)', fontsize=11)
ax.set_title('Phase 2: Converged Accumulation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('MCMC Entropy — Weighted Nested Hierarchical SBM',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'RF_weighted_nested_entropy_trajectory_50_{args.score}.png'),
            dpi=150, bbox_inches='tight')
log_msg(f"| UPDATE | Entropy trajectory saved (t*={t_star}, {n_conv} accumulation samples)")
plt.close()

# plt.show()


# ---- Block connectivity heatmaps (one figure per meaningful level) ---- #
# Single joint matrix per level — model-internal mrs, joint across both layers.
from matplotlib.colors import to_rgba

for level_idx in meaningful_levels:
    bmat = block_connectivity[level_idx]
    b    = modal_assignments[level_idx]
    n_b  = int(b.max()) + 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.where(bmat == 0, np.nan, bmat), cmap='plasma', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Mean block-to-block edge count (mrs)')
    ax.set_title(f'Joint Block Connectivity\nLevel {level_idx} — {n_b} blocks', fontsize=11)
    ax.set_xlabel('Block index')
    ax.set_ylabel('Block index')
    ax.set_xticks(range(n_b))
    ax.set_yticks(range(n_b))
    plt.suptitle(f'Block Connectivity — Level {level_idx} Modal Partition  |  {args.score}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/SBM_block_matrix_level{level_idx}_{args.score}.png',
                dpi=150, bbox_inches='tight')
    log_msg(f"| UPDATE | Block matrix saved for level {level_idx}")
    plt.close()


# ---- Graph visualisation (graph_tool draw, level-0 modal partition) ---- #
print("\nGenerating graph visualisation...")

_coord_data = np.loadtxt(os.path.join(args.data_path, 'ATLAS', 'circle_coords_360_sorted.txt'),
                         delimiter='\t', skiprows=1, usecols=(0, 1))


pos = g.new_vertex_property("vector<double>")
for v in g.vertices():
    pos[v] = _coord_data[int(v)].tolist()

unique_location_names   = sorted(set(locations))
location_name_to_idx    = {name: idx for idx, name in enumerate(unique_location_names)}
n_locations             = len(unique_location_names)

distinct_colors = [to_rgba(c) for c in loc_colours]
cmap            = ListedColormap(distinct_colors)
norm            = Normalize(vmin=0, vmax=max(n_locations - 1, 1))

vertex_color = g.new_vertex_property("vector<double>")
vertex_shape = g.new_vertex_property("int")
for v in g.vertices():
    node_idx     = int(v)
    location     = locations[node_idx]
    side         = location.rsplit('_', 1)[1]
    location_idx = location_name_to_idx[location]
    rgba         = cmap(norm(location_idx))
    vertex_color[v] = rgba
    vertex_shape[v] = 0 if side == 'L' else 1

degree_map   = g.degree_property_map("in")
vertex_sizes = gt.prop_to_size(degree_map, mi=20, ma=50)

# Edge alpha: normalised joint block connectivity strength under level-0 modal partition
b0       = modal_assignments[0]
bmat0    = block_connectivity[0]
bmat_max = bmat0.max() if bmat0.max() > 0 else 1.0

edge_alpha_arr = np.array([
    bmat0[b0[int(e.source())], b0[int(e.target())]] / bmat_max
    for e in g.edges()
])
edge_alpha_arr = np.clip(edge_alpha_arr, 0.01, 1.0)

edge_color = g.new_edge_property("vector<double>")
for idx, e in enumerate(g.edges()):
    src_c   = vertex_color[e.source()]
    tgt_c   = vertex_color[e.target()]
    avg_rgb = [(src_c[c] + tgt_c[c]) / 2 for c in range(3)]
    avg_rgb.append(float(edge_alpha_arr[idx]))
    edge_color[e] = tuple(avg_rgb)

state_nested.draw(
    pos=pos,
    vertex_fill_color=vertex_color,
    vertex_shape=vertex_shape,
    vertex_size=vertex_sizes,
    vertex_pen_width=0.5,
    edge_color=edge_color,
    edge_pen_width=gt.prop_to_size(g.ep.behaviour_weight, mi=0.5, ma=3),
    edge_gradient=[],
    vertex_text=g.vp.label,
    vertex_text_color='black',
    vertex_text_position=0,
    vertex_font_size=10,
    output=os.path.join(output_dir, f"RF_weighted_nested_block_state_draw_{args.score}.png"),
    output_size=(1200, 1200)
)
log_msg(f"| UPDATE | Graph visualisation saved")


# ---- Location legend ---- #
fig, ax = plt.subplots(figsize=(3, 2))
legend_elements = []
for loc_idx, location_name in enumerate(unique_location_names):
    rgba   = cmap(norm(loc_idx))
    n_node = np.sum(np.array(locations) == location_name)
    marker = 'o' if location_name.endswith('_L') else '^'
    legend_elements.append(
        Line2D([0], [0], marker=marker, color='w',
               markerfacecolor=rgba, markersize=10,
               markeredgecolor='black', markeredgewidth=1.5,
               label=f"{location_name} ({n_node} nodes)")
    )

ax.legend(handles=legend_elements, loc='center', fontsize=10, frameon=True,
          title="Node colour by location (circles=L, triangles=R)",
          title_fontsize=12, ncol=2, fancybox=True, shadow=True)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'RF_weighted_nested_location_legend_{args.score}.png'),
            dpi=150, bbox_inches='tight')
plt.close()


# ---- Connectome plot: three-ring hierarchical circle ---- #
#
# Layout:
#   Outer ring   — anatomical nodes at atlas coordinates, coloureds by location,
#                  shaped by hemisphere (circles = L, triangles = R).
#   Middle ring  — level-0 block nodes, evenly spaced at 66% of outer radius.
#                  Spokes connect each outer node to its level-0 block (Bézier,
#                  control point = outer-circle point at the block's radial angle).
#   Inner ring   — level-1 block nodes, evenly spaced at 35% of outer radius.
#                  Straight edges connect each level-0 block node to its level-2
#                  block (majority-vote mapping over member nodes).

outer_coord_data = np.loadtxt(
    os.path.join(args.data_path,'ATLAS', 'circle_coords_360_sorted.txt'),
    delimiter='\t', skiprows=1, usecols=(0, 1)
).astype(float)

b0   = modal_assignments[0]
b1   = modal_assignments[1]
n_b0 = int(b0.max()) + 1
n_b1 = int(b1.max()) + 1

# Outer circle geometry
cx_outer = outer_coord_data[:, 0].mean()
cy_outer = outer_coord_data[:, 1].mean()
r_outer  = np.hypot(outer_coord_data[:, 0] - cx_outer,
                    outer_coord_data[:, 1] - cy_outer).max()
r_middle = r_outer * 0.66   # level-0 blocks
r_inner  = r_outer * 0.35   # level-2 blocks

# Middle ring: level-0 block positions
angles_b0 = np.linspace(0, 2 * np.pi, n_b0, endpoint=False)
middle_xy = np.column_stack([
    cx_outer + r_middle * np.cos(angles_b0),
    cy_outer + r_middle * np.sin(angles_b0)
])

# Bézier control points for outer→middle spokes
ctrl_xy = np.column_stack([
    cx_outer + r_outer * np.cos(angles_b0),
    cy_outer + r_outer * np.sin(angles_b0)
])

# Inner ring: level-1 block positions
angles_b1 = np.linspace(0, 2 * np.pi, n_b1, endpoint=False)
inner_xy  = np.column_stack([
    cx_outer + r_inner * np.cos(angles_b1),
    cy_outer + r_inner * np.sin(angles_b1)
])

# Mapping: level-0 block → level-1 block (majority vote over member nodes)
b0_to_b1 = np.zeros(n_b0, dtype=int)
for blk0 in range(n_b0):
    nodes_in = np.where(b0 == blk0)[0]
    if nodes_in.size > 0:
        b0_to_b1[blk0] = int(np.bincount(b1[nodes_in], minlength=n_b1).argmax())

# Normalise level-0 consistency scores to [0, 1] across all nodes so that
# relative differences are visible even when all values are clustered near 1.
# The least-consistent node maps to alpha=0.05; the most-consistent to 1.0.
_cons0      = node_consistency[0]
_cons_min   = float(_cons0.min())
_cons_max   = float(_cons0.max())
_cons_range = _cons_max - _cons_min if _cons_max > _cons_min else 1.0
_alpha_min  = 0.05
log_msg(f"| UPDATE | Node consistency (level 0): "
        f"min={_cons_min:.3f}  max={_cons_max:.3f}  range={_cons_range:.3f}")

t_bez = np.linspace(0, 1, 80)
idx_L = [i for i, loc in enumerate(locations) if loc.endswith('_L')]
idx_R = [i for i, loc in enumerate(locations) if loc.endswith('_R')]

n_cols    = min(n_b0, 4)
n_rows    = int(np.ceil(n_b0 / n_cols))
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5.5 * n_cols, 5.5 * n_rows),
                         squeeze=False)
axes_flat = axes.flatten()

for blk_feat in range(n_b0):
    ax = axes_flat[blk_feat]
    ax.set_aspect('equal')
    ax.axis('off')
    members     = np.where(b0 == blk_feat)[0]
    non_members = np.where(b0 != blk_feat)[0]
    # ---- Spokes: outer node → its level-0 block node ---- #
    # Alpha = posterior marginal probability of modal block assignment.
    # Consistent members are opaque; ambiguous nodes are transparent.
    for node_idx in members:
        x1, y1    = outer_coord_data[node_idx]
        x2, y2    = middle_xy[blk_feat]
        cpx, cpy  = ctrl_xy[blk_feat]
        alpha_node = float(_alpha_min + (1.0 - _alpha_min) *
                           (_cons0[node_idx] - _cons_min) / _cons_range)
        xc = (1 - t_bez)**2 * x1 + 2*(1 - t_bez)*t_bez * cpx + t_bez**2 * x2
        yc = (1 - t_bez)**2 * y1 + 2*(1 - t_bez)*t_bez * cpy + t_bez**2 * y2
        ax.plot(xc, yc, color=node_color[node_idx],
                alpha=alpha_node, linewidth=0.7, zorder=2)
    # ---- Middle→inner edges: shown in every subplot ---- #
    for blk0 in range(n_b0):
        blk1 = int(b0_to_b1[blk0])
        ax.plot([middle_xy[blk0, 0], inner_xy[blk1, 0]],
                [middle_xy[blk0, 1], inner_xy[blk1, 1]],
                color='dimgray', alpha=0.5, linewidth=1.5, zorder=3)
    # ---- Inner ring: level-1 block nodes ---- #
    ax.scatter(inner_xy[:, 0], inner_xy[:, 1],
               s=150, c='white', edgecolors='dimgray', linewidths=1.5, zorder=4)
    for b in range(n_b1):
        ax.text(inner_xy[b, 0], inner_xy[b, 1], str(b),
                ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
    # ---- Middle ring: level-0 block nodes (featured block highlighted) ---- #
    c_mid  = ['gold' if b == blk_feat else 'white' for b in range(n_b0)]
    lw_mid = [2.5   if b == blk_feat else 1.5     for b in range(n_b0)]
    ax.scatter(middle_xy[:, 0], middle_xy[:, 1],
               s=200, c=c_mid, edgecolors='dimgray', linewidths=lw_mid, zorder=6)
    for b in range(n_b0):
        ax.text(middle_xy[b, 0], middle_xy[b, 1], str(b),
                ha='center', va='center', fontsize=8, fontweight='bold', zorder=7)
    # ---- Outer nodes: non-members grayed, members colored by location ---- #
    # Member node alpha matches spoke alpha (normalised consistency score).
    nm_L = [i for i in non_members if locations[i].endswith('_L')]
    nm_R = [i for i in non_members if locations[i].endswith('_R')]
    m_L  = [i for i in members     if locations[i].endswith('_L')]
    m_R  = [i for i in members     if locations[i].endswith('_R')]
    def _node_rgba(node_idx):
        a   = _alpha_min + (1.0 - _alpha_min) * (_cons0[node_idx] - _cons_min) / _cons_range
        r, g, b, _ = to_rgba(node_color[node_idx])
        return (r, g, b, float(a))
    if nm_L:
        ax.scatter(outer_coord_data[nm_L, 0], outer_coord_data[nm_L, 1],
                   s=20, c='lightgray', marker='o', edgecolors='none',
                   alpha=0.3, zorder=6)
    if nm_R:
        ax.scatter(outer_coord_data[nm_R, 0], outer_coord_data[nm_R, 1],
                   s=20, c='lightgray', marker='^', edgecolors='none',
                   alpha=0.3, zorder=6)
    if m_L:
        ax.scatter(outer_coord_data[m_L, 0], outer_coord_data[m_L, 1],
                   s=80, c=[_node_rgba(i) for i in m_L],
                   marker='o', edgecolors='none', zorder=8)
    if m_R:
        ax.scatter(outer_coord_data[m_R, 0], outer_coord_data[m_R, 1],
                   s=80, c=[_node_rgba(i) for i in m_R],
                   marker='^', edgecolors='none', zorder=8)
    ax.set_title(f'Block {blk_feat}  ({len(members)} nodes)',
                 fontsize=10, fontweight='bold')

# Hide unused subplot panels
for ax in axes_flat[n_b0:]:
    ax.set_visible(False)

fig.suptitle(f'Level-0 Block Connectome — per block  |  {args.score}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'SBM_block_connectome_per_block_{args.score}.svg'),
            dpi=150, bbox_inches='tight')

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

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)

for level_idx in meaningful_levels:
    b_level  = modal_assignments[level_idx]
    n_blocks = int(b_level.max()) + 1
    # One NIfTI + txt mapping per block
    for blk in range(n_blocks):
        nodes_in_block = np.where(b_level == blk)[0]
        if len(nodes_in_block) == 0:
            continue
        block_data    = np.zeros_like(atlas_data, dtype=np.int32)
        mapping_lines = []
        for region_idx, node_idx in enumerate(nodes_in_block):
            parcel_val = node_idx + 1       # 1-indexed parcel code in atlas NIfTI
            region_num = region_idx + 1     # 1-indexed label within this block
            block_data[atlas_data == parcel_val] = region_num
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
    # Block connectivity matrix — CSV
    bmat     = block_connectivity[level_idx]
    csv_path = os.path.join(output_dir,
                            f'SBM_block_connectivity_lvl{level_idx}_{args.score}.csv')
    np.savetxt(csv_path, bmat, delimiter=',', fmt='%.6f')
    log_msg(f"| UPDATE | Block connectivity matrix saved: level {level_idx} "
            f"({n_blocks}×{n_blocks}) → {csv_path}")


log_msg(f"| FINISHED | All outputs saved")