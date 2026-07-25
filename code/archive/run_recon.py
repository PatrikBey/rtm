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
# The following script performs latent-multigraph stochastic block       #
# modelling (SBM) reconstruction of brain-region lesion co-occurrence    #
# data, with and without a behaviour indicator node. The earlier         #
# Ising-model (PseudoIsingBlockState) approach is kept below as          #
# commented-out legacy code for later evaluation — it was superseded     #
# after failing to reliably recover known block structure even on       #
# synthetic data (see project_run_recon_ising_fit_investigation memory). #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/20.                                              #
#                                                                       #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################


import os
import csv
import argparse
import pickle
import graph_tool.all as gt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from utils import log_msg



#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Run latent-multigraph SBM reconstruction on lesion co-occurrence data.')
args.add_argument('--data_file', type=str, default='/mnt/h/RT/data/Schaefer2018-400_lesion_loads.tsv',
                  help='Path to the co-occurrence data .tsv file')
args.add_argument('--out_dir', type=str, default='/mnt/h/RT/data/RESULTS/SBMRECONLOADS', help='Path to the output directory')
args.add_argument('--threshold', type=float, default=50.0,
                  help='Threshold applied to the subject x node matrix before binarization. '
                       'Interpreted according to --threshold_mode: a raw value on the data\'s '
                       'own scale (e.g. 50 for a lesion load >= 50%%) in "raw" mode, or a '
                       'quantile in [0,1] (e.g. 0.5 for a per-subject median) in "quantile" '
                       'mode (default: 50.0)')
args.add_argument('--threshold_mode', type=str, default='raw', choices=['raw', 'quantile'],
                  help='"raw": a single global cutoff (value > threshold), appropriate for '
                       'lesion load percentages already on a comparable 0-100 scale across '
                       'subjects. "quantile": a per-subject quantile cutoff computed from '
                       'that subject\'s own nonzero values, appropriate for degree/node-'
                       'strength data whose absolute scale is not comparable across subjects '
                       '(default: raw)')
args.add_argument('--data_path', type=str, default='/mnt/h/RT/data',
                  help='Path to the data directory containing participants.tsv')
args.add_argument('--atlas', type=str, default='Schaefer2018-400',
                  help='Atlas name; looks up {data_path}/ATLAS/{atlas}.nii.gz for block-'
                       'selection NIfTI output (default: Schaefer2018-400)')
args.add_argument('--score', type=str, default='Foreperiod_Long_tau',
                  help='Behaviour score column in participants.tsv to match against')
args.add_argument('--split_quantile', type=float, default=0.5,
                  help='Quantile of the matched behaviour vector used as the binarization '
                       'threshold: value > quantile(behaviour, split_quantile) -> 1, else -1. '
                       'Robust to non-normal distributions, unlike z-scoring (default: 0.5, '
                       'i.e. a median split)')
args.add_argument('--pclabel_bins', type=int, default=3,
                  help='Number of quantile bins used to discretize the per-region '
                       'behaviour-relevance score into pclabel categories (default: 3)')
args.add_argument('--max_iter', type=int, default=10000,
                  help='Maximum MCMC sweeps for change-point detection (default: 10000)')
args.add_argument('--window_size', type=int, default=250,
                  help='Window size for change-point detection (default: 250)')
args.add_argument('--shift_factor', type=float, default=0.5,
                  help='Mean-shift threshold: change point when sliding window mean drops '
                       'below first_window_mean - shift_factor * first_window_std (default: 0.5)')
args.add_argument('--seed', type=int, default=42,
                  help='Random seed for graph_tool RNG (default: 42)')
args = args.parse_args()

log_msg(f"| START | Running latent-multigraph SBM reconstruction on lesion co-occurrence data")
log_msg(f"| UPDATE | Data file: {args.data_file}")
log_msg(f"| UPDATE | Output directory: {args.out_dir}")

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)



#################################
#           FUNCTIONS           #
#################################

# ---- LEGACY: PseudoIsingBlockState reconstruction ---- #
# Kept for later evaluation (see project_run_recon_ising_fit_investigation
# memory for the full investigation). Superseded because, even after fixing
# the original beta=inf greedy-convergence bug, this approach could not
# reliably find or KEEP known block structure on synthetic test data: a
# multiflip partition search could find real structure, but the very next
# joint mcmc_sweep() call judged it against not-yet-adapted edge parameters
# and reverted it. LatentMultigraphBlockState below does not show this
# failure mode on matched-sparsity synthetic data.
#
# def fit_ising_sbm(S, pclabel=None, max_iter=10000, window_size=250, shift_factor=0.75, seed=42):
#     '''
#     fit a nested Ising-model block state via finite-temperature MCMC with
#     mean-shift change-point detection, mirroring the equilibration protocol
#     already validated in functions.fit_nested_sbm_layered (used by run.py).
#     if pclabel is given (one categorical label per node), it biases the
#     block partition toward agreeing with those external labels via
#     graph_tool's partition-constraint mechanism, without adding behaviour
#     as a node. graph_tool requires pclabel as a PropertyMap (not a plain
#     array); a scratch graph matching S's node count/order is used only to
#     create that map.
#     '''
#     gt.seed_rng(seed)
#
#     if pclabel is not None:
#         g_pclabel  = gt.Graph(directed=False)
#         g_pclabel.add_vertex(S.shape[0])
#         pclabel    = g_pclabel.new_vertex_property('int', vals=pclabel)
#         state_args = dict(pclabel=pclabel)
#     else:
#         state_args = {}
#     state = gt.PseudoIsingBlockState(S, self_loops=False, nested=True, state_args=state_args)
#
#     entropy_traj      = []
#     converged         = False
#     conv_iter         = max_iter
#     first_window_mean = None
#     first_window_std  = None
#     threshold         = None
#
#     for i in range(max_iter):
#         state.mcmc_sweep(niter=1)
#         entropy_traj.append(state.entropy())
#
#         if i == window_size - 1:
#             first_window_mean = sum(entropy_traj) / window_size
#             first_window_std  = (sum((x - first_window_mean) ** 2 for x in entropy_traj) / window_size) ** 0.5
#             threshold = first_window_mean - shift_factor * first_window_std
#         elif threshold is not None:
#             w_mean = sum(entropy_traj[-window_size:]) / window_size
#             if w_mean < threshold:
#                 converged = True
#                 conv_iter = i
#                 break
#
#     if converged:
#         log_msg(f"| UPDATE | fit_ising_sbm: mean-shift change point at iteration {conv_iter} "
#                 f"(ref={first_window_mean:.2f}, threshold={threshold:.2f})")
#     else:
#         log_msg(f"| WARNING | fit_ising_sbm: no mean shift detected within {max_iter} iterations "
#                 f"(ref={first_window_mean}, threshold={threshold}) — using final state. "
#                 f"Consider increasing max_iter or reducing shift_factor (current: {shift_factor}).")
#
#     return state


def build_cooccurrence_graph(X, behaviour_high=None):
    '''
    build a co-occurrence multigraph from a subject x node binary matrix:
    edge multiplicity between nodes i and j = number of subjects with both
    lesioned. If behaviour_high (subjects,) boolean array is given, one
    extra vertex is appended representing "behaviour high", with edge
    multiplicity to each node = number of subjects with both that node
    lesioned and behaviour_high true. This is the measured multigraph
    LatentMultigraphBlockState reconstructs a latent network from.
    '''
    n_nodes    = X.shape[1]
    n_vertices = n_nodes + 1 if behaviour_high is not None else n_nodes
    g = gt.Graph(directed=False)
    g.add_vertex(n_vertices)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            count = int(np.sum((X[:, i] == 1) & (X[:, j] == 1)))
            if count > 0:
                g.add_edge_list([(i, j)] * count)
    if behaviour_high is not None:
        beh_vertex = n_nodes
        for i in range(n_nodes):
            count = int(np.sum((X[:, i] == 1) & behaviour_high))
            if count > 0:
                g.add_edge_list([(i, beh_vertex)] * count)
    return g


def fit_latent_multigraph(g, pclabel=None, max_iter=10000, window_size=250, shift_factor=0.75, seed=42):
    '''
    fit a latent-multigraph nested block state (Peixoto 2020, "Latent
    Poisson models for networks with heterogeneous density") via finite-
    temperature MCMC with mean-shift change-point detection, on a
    co-occurrence multigraph built by build_cooccurrence_graph(). The
    model treats observed edge multiplicities as noisy Poisson draws of a
    latent "true" network and reconstructs it with a degree-corrected
    null, jointly with an SBM partition. Unlike PseudoIsingBlockState,
    mcmc_sweep() here uses multiflip partition moves by default, which
    was empirically necessary to escape trivial local optima (see
    project_run_recon_ising_fit_investigation memory).

    if pclabel is given (one categorical label per node), it biases the
    block partition toward agreeing with those external labels via
    graph_tool's partition-constraint mechanism, without adding behaviour
    as a node.

    Returns (state, entropy_traj) — entropy_traj is the full per-iteration
    entropy trace, for post-hoc convergence inspection (see plot_entropy).
    '''
    gt.seed_rng(seed)

    if pclabel is not None:
        g_pclabel  = gt.Graph(directed=False)
        g_pclabel.add_vertex(g.num_vertices())
        pclabel    = g_pclabel.new_vertex_property('int', vals=pclabel)
        state_args = dict(pclabel=pclabel)
    else:
        state_args = {}
    state = gt.LatentMultigraphBlockState(g, nested=True, self_loops=False, state_args=state_args)

    entropy_traj      = []
    converged         = False
    conv_iter         = max_iter
    first_window_mean = None
    first_window_std  = None
    threshold         = None

    for i in range(max_iter):
        state.mcmc_sweep(niter=1)
        entropy_traj.append(state.entropy())

        if i == window_size - 1:
            first_window_mean = sum(entropy_traj) / window_size
            first_window_std  = (sum((x - first_window_mean) ** 2 for x in entropy_traj) / window_size) ** 0.5
            threshold = first_window_mean - shift_factor * first_window_std
        elif threshold is not None:
            w_mean = sum(entropy_traj[-window_size:]) / window_size
            if w_mean < threshold:
                converged = True
                conv_iter = i
                break

    if converged:
        log_msg(f"| UPDATE | fit_latent_multigraph: mean-shift change point at iteration {conv_iter} "
                f"(ref={first_window_mean:.2f}, threshold={threshold:.2f})")
    else:
        log_msg(f"| WARNING | fit_latent_multigraph: no mean shift detected within {max_iter} iterations "
                f"(ref={first_window_mean}, threshold={threshold}) — using final state. "
                f"Consider increasing max_iter or reducing shift_factor (current: {shift_factor}).")

    return state, entropy_traj


def plot_entropy(entropy_traj, label, output):
    '''
    plot the entropy (description length) trajectory across all MCMC
    iterations of a fit_latent_multigraph() run, to visually inspect
    convergence behaviour.
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(entropy_traj, linewidth=1)
    plt.xlabel('MCMC iteration')
    plt.ylabel('Entropy (description length)')
    plt.title(f'Entropy trajectory — {label}')
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def region_behaviour_relevance(X, behaviour):
    '''
    per-region Spearman correlation between lesion presence across subjects
    (X: subjects x nodes, binary) and the raw, un-split behaviour score.
    Regions with zero lesion variance in this sample get relevance 0.
    '''
    n_nodes   = X.shape[1]
    relevance = np.zeros(n_nodes)
    for i in range(n_nodes):
        if X[:, i].std() == 0:
            continue
        relevance[i], _ = spearmanr(X[:, i], behaviour)
    return np.nan_to_num(relevance)


def relevance_to_pclabel(relevance, n_bins=3):
    '''
    quantile-bin a continuous per-node relevance score into categorical
    labels suitable for graph_tool's pclabel partition constraint
    '''
    edges = np.quantile(relevance, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(relevance, edges)


def select_relevant_blocks(blocks, relevance):
    '''
    score each block by its members' mean behavioural relevance, z-score
    across blocks, and select blocks with above-average (z > 0) relevance
    -- mirroring run.py's block-selection approach (there: consistency-
    weighted mean behaviour_degree per block, z-scored, z > 0 selected;
    here, a plain per-block mean relevance since this reconstruction has
    no posterior node-consistency weighting to fold in).

    Returns selected_blocks (array of block ids with z > 0) and, in the
    same order as blocks/relevance, each ROI's own block's raw score,
    z-score, and whether that block was selected -- for writing a
    roi_assignment table and a combined block-values NIfTI.
    '''
    unique_blocks = np.unique(blocks)
    block_scores  = np.array([relevance[blocks == blk].mean() for blk in unique_blocks])
    score_std     = block_scores.std()
    block_zscore  = (block_scores - block_scores.mean()) / score_std \
                    if score_std > 0 else np.zeros_like(block_scores)
    selected_blocks = unique_blocks[block_zscore > 0]

    score_by_block  = dict(zip(unique_blocks.tolist(), block_scores.tolist()))
    zscore_by_block = dict(zip(unique_blocks.tolist(), block_zscore.tolist()))
    node_score    = np.array([score_by_block[b] for b in blocks])
    node_zscore   = np.array([zscore_by_block[b] for b in blocks])
    node_selected = np.isin(blocks, selected_blocks)

    return selected_blocks, node_score, node_zscore, node_selected


def save_block_niftis(blocks, selected_blocks, atlas_img, atlas_data, out_dir, label):
    '''
    write one NIfTI per selected block: voxels of each member ROI carry a
    1-indexed within-block region number, 0 elsewhere. Parcellation
    convention: atlas parcel value v corresponds to node index v-1 (row
    order of the input data file), matching run.py's convention.
    '''
    for blk in selected_blocks:
        member_nodes = np.where(blocks == blk)[0]
        if member_nodes.size == 0:
            continue
        block_data = np.zeros_like(atlas_data, dtype=np.int32)
        for region_idx, node_idx in enumerate(member_nodes):
            parcel_val = node_idx + 1
            region_num = region_idx + 1
            block_data[atlas_data == parcel_val] = region_num
        nii_path = os.path.join(out_dir, f'{label}_block{blk}.nii.gz')
        nib.save(nib.Nifti1Image(block_data, atlas_img.affine, atlas_img.header), nii_path)
        log_msg(f"| UPDATE | Block NIfTI saved: {label} block {blk} "
                f"({len(member_nodes)} regions) → {nii_path}")


def save_block_value_nifti(node_score, atlas_img, atlas_data, out_path):
    '''
    write a single combined NIfTI covering every node (not just selected
    blocks): each voxel carries its own block's mean behavioural relevance
    value, so all blocks' behavioural relevance can be inspected spatially
    at once -- analogous to run.py's combined block-score NIfTI.
    '''
    value_data = np.zeros_like(atlas_data, dtype=np.float32)
    for node_idx, val in enumerate(node_score):
        parcel_val = node_idx + 1
        value_data[atlas_data == parcel_val] = val
    nib.save(nib.Nifti1Image(value_data, atlas_img.affine, atlas_img.header), out_path)
    log_msg(f"| UPDATE | Block behavioural-value NIfTI saved → {out_path}")


def save_roi_assignment_table(node_names, blocks, relevance, node_score, node_zscore, node_selected, out_path):
    '''
    write a per-ROI summary table: block assignment, behavioural relevance,
    that ROI's own block's raw score and z-score, and whether that block
    was selected (z > 0) -- analogous to run.py's roi_block_assignments csv.
    '''
    with open(out_path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(['roi_index', 'roi_name', 'block', 'relevance',
                         'block_score', 'block_zscore', 'selected'])
        for i, name in enumerate(node_names):
            writer.writerow([i, name, int(blocks[i]), round(float(relevance[i]), 6),
                             round(float(node_score[i]), 6), round(float(node_zscore[i]), 6),
                             bool(node_selected[i])])
    log_msg(f"| UPDATE | ROI assignment summary saved → {out_path}")


# ---- LEGACY: Ising-specific graph extraction (see fit_ising_sbm above) ---- #
# def extract_graph(state):
#     '''
#     extract the graph from a fitted block state, registering the Ising couplings as an internal edge property
#     '''
#     g = state.get_graph()
#     g.ep['x'] = state.get_x()  # register as internal property so it survives g.save()
#     return g


def extract_latent_graph(state):
    '''
    extract the inferred latent graph from a fitted LatentMultigraphBlockState.
    Unlike PseudoIsingBlockState there is no single scalar "coupling strength"
    per edge to attach (no get_x() analog); block structure is carried via
    get_block_state() as usual.
    '''
    return state.get_graph()


def brain_only_graph(g, behaviour_node):
    '''
    materialize a standalone copy of a graph with the behaviour node removed
    '''
    keep = g.new_vertex_property('bool', val=True)
    keep[g.vertex(behaviour_node)] = False
    gv = gt.GraphView(g, vfilt=keep)
    return gt.Graph(gv, prune=True)


def behaviour_only_graph(g, behaviour_node):
    '''
    materialize a standalone copy of a graph keeping only edges incident to
    the behaviour node (i.e. the full with-behaviour graph minus
    brain_only_graph, edge-wise): isolates which regions behaviour actually
    connects to, dropping every pure brain-to-brain edge. All vertices
    (including brain regions with no behaviour edge) are kept.
    '''
    keep = g.new_edge_property('bool', val=False)
    for e in g.vertex(behaviour_node).all_edges():
        keep[e] = True
    gv = gt.GraphView(g, efilt=keep)
    return gt.Graph(gv, prune=True)


def get_blocks(state, level=0):
    '''
    get the per-node block assignment array at a given hierarchy level from
    a (possibly nested) block state. level=0 is the base partition; higher
    levels are projected down to per-node assignments via
    NestedBlockState.project_partition(level, 0), which broadcasts each
    higher-level block id to every original node whose level-0 block
    belongs to it.
    '''
    bstate = state.get_block_state()
    if not hasattr(bstate, 'levels'):
        return bstate.get_blocks().a
    if level == 0:
        return bstate.levels[0].get_blocks().a
    proj = bstate.project_partition(level, 0)
    return proj.a if hasattr(proj, 'a') else np.array([proj[v] for v in bstate.g.vertices()])


def is_same_partition(labels_a, labels_b):
    '''
    check whether two label arrays induce the exact same grouping of nodes,
    up to a relabelling (i.e. every group in a maps onto exactly one group
    in b, and vice versa)
    '''
    for label in np.unique(labels_a):
        if len(np.unique(labels_b[labels_a == label])) != 1:
            return False
    for label in np.unique(labels_b):
        if len(np.unique(labels_a[labels_b == label])) != 1:
            return False
    return True


def partition_agreement(labels_a, labels_b):
    '''
    quantify how similar two vertex partitions are, to check whether a
    pclabel-informed fit is reproducing its input labels verbatim (biasing
    turned into a hard constraint in practice) or finding block structure
    that is informed by, but free to diverge from, those input labels.
    AMI/ARI == 1.0 and identical_partition == True together indicate exact
    reproduction; lower values indicate the model found something else.
    '''
    return {
        'ami':                adjusted_mutual_info_score(labels_a, labels_b),
        'ari':                adjusted_rand_score(labels_a, labels_b),
        'identical_partition': is_same_partition(labels_a, labels_b),
    }


def log_hierarchy(state, label):
    '''
    log per-level block and edge counts of a (possibly nested) block state,
    to distinguish "collapsed to a single block" from "level graph has no
    recorded edges despite multiple blocks"
    '''
    bstate = state.get_block_state()
    levels = bstate.levels if hasattr(bstate, 'levels') else [bstate]
    for i, level in enumerate(levels):
        n_blocks = int(level.get_nonempty_B())
        n_edges  = level.g.num_edges()
        log_msg(f"| UPDATE | {label} level {i}: {n_blocks} blocks, {level.g.num_vertices()} vertices, {n_edges} edges")


def draw_state(state, relevance, output, node_names=None, output_size=(1200, 1200), cmap='plasma',
               edge_alpha=0.6, min_edge_alpha=0.15, alpha_gamma=4.0, edge_visible_percentile=90,
               arrow_colour='black', vertex_font_size=8, vertex_size=None,
               vertex_text_position=0, vertex_text_color='white'):
    '''
    draw a (possibly nested) block state's hierarchy via state.draw(),
    colouring nodes and edges by behavioural relevance (region_behaviour_
    relevance output) rather than plain block colour: node colour reflects
    signed relevance via a diverging colormap, edge colour is the average
    of its endpoints' colours.

    Real co-occurrence graphs are near-complete (hundreds of nodes, tens of
    thousands of edges) — a purely continuous alpha falloff still leaves
    every edge with some nonzero alpha, and thousands of overlapping
    semi-transparent lines visually accumulate into a solid mass that looks
    like the nodes themselves are overlapping even when they aren't. So
    edges are only drawn at all if their endpoints' mean |relevance| is
    above the edge_visible_percentile across all edges (default: top 10%);
    everything else gets exact alpha 0.0, a true no-op in Cairo regardless
    of how many such edges are stacked. Edges that pass the cutoff still
    get their alpha shaped by alpha_gamma (> 1 compresses the low end of
    the surviving range toward min_edge_alpha) between min_edge_alpha and
    edge_alpha.

    relevance must be given in the same vertex order as the state's base
    graph; any extra vertex beyond len(relevance) (e.g. an added behaviour
    node) is treated as neutral (relevance 0) and, if node_names is given,
    labelled "behaviour". node_names, if given, are used as per-node text
    labels (ROI names), matching create_figures.py's convention of
    labelling nodes directly on the drawing. vertex_text_position=0 places
    the label beside each node rather than centred on top of it (graph_tool
    renders text outside the vertex at that angle when given a value >= 0,
    vs. centred/overlapping the fill when left at the default -1) — with
    hundreds of densely-packed nodes, centred text overlaps neighbouring
    nodes and reads as if the nodes themselves were overlapping.

    vertex_size defaults to None, which auto-scales it down as vertex count
    grows (inversely with sqrt(n)) so nodes don't overlap regardless of
    graph size; pass a number to override.
    '''
    bstate = state.get_block_state()
    g = bstate.g if hasattr(bstate, 'g') else bstate.levels[0].g

    if vertex_size is None:
        vertex_size = max(6, min(20, 120 / max(g.num_vertices(), 1) ** 0.5))

    rel = np.zeros(g.num_vertices())
    rel[:len(relevance)] = relevance

    cmap_obj  = plt.get_cmap(cmap)
    max_abs   = np.max(np.abs(rel)) if np.any(rel) else 1.0
    colour_norm = Normalize(vmin=-max_abs, vmax=max_abs)

    vertex_color = g.new_vertex_property('vector<double>')
    for v in g.vertices():
        vertex_color[v] = cmap_obj(colour_norm(rel[int(v)]))

    abs_rel   = np.abs(rel)
    edge_rels = np.array([(abs_rel[int(e.source())] + abs_rel[int(e.target())]) / 2 for e in g.edges()])
    visible_cutoff = np.percentile(edge_rels, edge_visible_percentile) if edge_rels.size else 0.0
    visible_rels   = edge_rels[edge_rels >= visible_cutoff]
    alpha_norm = Normalize(vmin=visible_rels.min(), vmax=visible_rels.max()) \
                 if visible_rels.size and visible_rels.max() > visible_rels.min() else Normalize(0, 1)

    edge_color = g.new_edge_property('vector<double>')
    for e, edge_rel in zip(g.edges(), edge_rels):
        src_c, tgt_c = vertex_color[e.source()], vertex_color[e.target()]
        avg_rgb = [(src_c[c] + tgt_c[c]) / 2 for c in range(3)]
        if edge_rel < visible_cutoff:
            alpha = 0.0
        else:
            alpha = min_edge_alpha + (edge_alpha - min_edge_alpha) * (alpha_norm(edge_rel) ** alpha_gamma)
        edge_color[e] = (*avg_rgb, alpha)

    draw_kwargs = dict(
        vertex_fill_color=vertex_color,
        vertex_size=vertex_size,
        edge_color=edge_color,
        edge_gradient=[],
        hedge_color=arrow_colour,
        hvertex_fill_color=arrow_colour,
        hvertex_color=arrow_colour,
        output=output,
        output_size=output_size,
    )

    if node_names is not None:
        vertex_text = g.new_vertex_property('string')
        for v in g.vertices():
            idx = int(v)
            vertex_text[v] = node_names[idx] if idx < len(node_names) else 'behaviour'
        draw_kwargs['vertex_text']          = vertex_text
        draw_kwargs['vertex_font_size']     = vertex_font_size
        draw_kwargs['vertex_text_position'] = vertex_text_position
        draw_kwargs['vertex_text_color']    = vertex_text_color

    bstate.draw(**draw_kwargs)


def binarize(vals, threshold=0.5, threshold_mode='raw'):
    '''
    binarize a subject x node value matrix. Two thresholding modes:
      'raw'      - a single global cutoff (value > threshold), appropriate
                   for lesion load percentages: already on a comparable
                   0-100 scale across subjects and ROIs, so no per-subject
                   normalization is needed.
      'quantile' - a per-subject quantile cutoff, computed from that
                   subject's own nonzero values (value > quantile(row[row>0],
                   threshold)). Appropriate for degree/node-strength data,
                   whose absolute scale is not comparable across subjects
                   (depends on each subject's own tractogram reconstruction).
                   Note: scale-invariant per row, so uniformly rescaling a
                   subject's whole row (e.g. by a behaviour weight) before
                   calling this in 'quantile' mode has no effect on the
                   result -- only 'raw' mode is sensitive to pre-scaling.
    '''
    if threshold_mode == 'raw':
        return (vals > threshold).astype(int)
    elif threshold_mode == 'quantile':
        X = np.zeros_like(vals, dtype=int)
        for i in range(vals.shape[0]):
            row = vals[i]
            nz  = row[row > 0]
            if nz.size == 0:
                continue
            cutoff = np.quantile(nz, threshold)
            X[i] = (row > cutoff).astype(int)
        return X
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode!r} (expected 'raw' or 'quantile')")


def rescale_01(x):
    '''min-max rescale a vector to [0, 1]; constant input maps to all-1s'''
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return np.ones_like(x) if span == 0 else (x - x.min()) / span


def load_node_matrix(data_file, threshold=0.5, threshold_mode='raw'):
    '''
    load the subject x node matrix and return a binarized data matrix, the
    raw (pre-binarization) values, the subject list, and the node/ROI names
    (from the file's header row). See binarize() for threshold_mode.
    '''
    data       = np.genfromtxt(data_file, delimiter='\t', dtype=str)
    node_names = data[0, 1:].tolist()
    vals       = data[1:, 1:].astype(float)
    X          = binarize(vals, threshold=threshold, threshold_mode=threshold_mode)
    return X, vals, data[1:, 0].tolist(), node_names


def load_data(data_file, data_path, score='Foreperiod_Long_tau', threshold=0.5,
             threshold_mode='raw', split_quantile=0.5):
    '''
    load the subject x node matrix (see load_node_matrix/binarize for
    threshold_mode: 'raw' for lesion loads, 'quantile' for degree/node-
    strength), match subjects to their behaviour score in participants.tsv,
    drop subjects with missing behaviour from the list and both data
    matrices, and return the clean binarized data matrix, the clean raw
    (pre-binarization) values, a quantile-thresholded behaviour vector
    (value > quantile(behaviour, split_quantile) -> 1, else -1), the
    subject list, missing-score list, and node/ROI names (unaffected by
    subject-level filtering). Quantile-based splitting is robust to
    non-normal behaviour distributions, unlike z-scoring.
    '''
    X, raw_vals, subject_list, node_names = load_node_matrix(
        data_file, threshold=threshold, threshold_mode=threshold_mode)

    part      = np.genfromtxt(os.path.join(data_path, 'participants.tsv'), dtype=str, delimiter='\t')
    score_col = np.where(part[0] == score)[0][0]

    keep_idx  = []
    behaviour = []
    subjects_missing_score = []
    for i, subject in enumerate(subject_list):
        val = part[part[:, 0] == subject, score_col]
        if val.size == 0 or val[0] in ('', 'nan', 'NaN'):
            subjects_missing_score.append(subject)
            continue
        keep_idx.append(i)
        behaviour.append(float(val[0]))

    X_clean             = X[keep_idx]
    raw_vals_clean      = raw_vals[keep_idx]
    subject_list_clean  = [subject_list[i] for i in keep_idx]
    behaviour           = np.array(behaviour, dtype=np.float64)

    # ---- mask subjects with zero variance across all nodes ---- #
    # A subject whose binarized lesion vector is the same at every ROI (all-0:
    # no ROI cleared the threshold anywhere; all-1: every ROI flagged) carries
    # no informative co-occurrence signal for the reconstruction, and an all-1
    # row in particular almost certainly reflects a data/loading artifact
    # rather than real pathology. Log each one's mean before dropping it.
    row_std           = X_clean.std(axis=1)
    zero_var_subjects = np.where(row_std == 0)[0]
    for i in zero_var_subjects:
        log_msg(f"| WARNING | subject {subject_list_clean[i]} has zero variance across all "
                f"{X_clean.shape[1]} nodes (mean={X_clean[i].mean():.3f}) — masking from further analysis")

    keep_var            = row_std > 0
    X_clean             = X_clean[keep_var]
    raw_vals_clean      = raw_vals_clean[keep_var]
    behaviour           = behaviour[keep_var]
    subject_list_clean  = [s for s, k in zip(subject_list_clean, keep_var) if k]

    cutoff = np.quantile(behaviour, split_quantile)
    y      = np.where(behaviour > cutoff, 1, -1)

    return X_clean, raw_vals_clean, y, behaviour, subject_list_clean, subjects_missing_score, node_names



#################################
#          LOAD DATA            #
#################################

X, raw_vals, y, behaviour, subject_list, subjects_missing_score, node_names = load_data(
    args.data_file, args.data_path, args.score,
    threshold=args.threshold, threshold_mode=args.threshold_mode, split_quantile=args.split_quantile
)

log_msg(f"| UPDATE | Subjects: {X.shape[0]}, nodes: {X.shape[1]}")
log_msg(f"| UPDATE | Binarization threshold: {args.threshold} (mode: {args.threshold_mode})")
log_msg(f"| UPDATE | Behaviour score: {args.score} (split quantile: {args.split_quantile})")
log_msg(f"| UPDATE | Subjects missing behaviour: {len(subjects_missing_score)}")
log_msg(f"| UPDATE | Subjects with positive behaviour: {np.sum(y == 1)}")

# ---- diagnose whether the binarized matrix is too degenerate to carry structure ---- #
# A trivial single-block SBM fit is the correct MDL answer if X is nearly all
# 0s/1s: there is then no per-node variance across subjects for the
# reconstruction to explain, regardless of how behaviour is (or isn't) integrated.
zero_var_nodes    = int(np.sum(X.std(axis=0) == 0))
zero_var_subjects = int(np.sum(X.std(axis=1) == 0))
log_msg(f"| UPDATE | X fraction of ones: {X.mean():.4f} "
        f"(0.5 = balanced; near 0 or 1 means the threshold is degenerate for this data's scale)")
log_msg(f"| UPDATE | Zero-variance nodes: {zero_var_nodes}/{X.shape[1]}, "
        f"zero-variance subjects: {zero_var_subjects}/{X.shape[0]} "
        f"(should be 0 here — load_data() already masks zero-variance subjects)")

# ---- LEGACY: Ising {-1,+1} spin observation matrices (see fit_ising_sbm above) ---- #
# S_brain = np.where(X.T.astype(int) > 0, 1, -1)      # (N_nodes, N_subjects)
# S_full  = np.vstack([S_brain, y[np.newaxis, :]])    # (N_nodes + 1, N_subjects)

# ---- behaviour indicator used to add an optional behaviour vertex to the ---- #
# ---- co-occurrence multigraph (build_cooccurrence_graph) ---- #
behaviour_high = y == 1

# ---- behaviour-weighted lesion loads: weight each subject's raw (pre- ---- #
# ---- binarization) values by their own 0-1 rescaled behaviour score, ---- #
# ---- then binarize -- so behaviourally low-weighted subjects' lesions  ---- #
# ---- are less likely to clear the threshold and contribute to co-      ---- #
# ---- occurrence counts. Only meaningful under threshold_mode='raw':    ---- #
# ---- 'quantile' mode is scale-invariant per subject row, so uniformly  ---- #
# ---- rescaling a row has no effect on that subject's own binarization. ---- #
if args.threshold_mode == 'quantile':
    log_msg(f"| WARNING | threshold_mode='quantile' is scale-invariant per subject row — "
            f"behaviour-weighting before thresholding will have no effect on X_beh_weighted")

behaviour_weight_01 = rescale_01(behaviour)
X_beh_weighted = binarize(raw_vals * behaviour_weight_01[:, np.newaxis],
                          threshold=args.threshold, threshold_mode=args.threshold_mode)
log_msg(f"| UPDATE | Behaviour-weighted X fraction of ones: {X_beh_weighted.mean():.4f} "
        f"(unweighted: {X.mean():.4f})")



#################################
#          FIT MODEL            #
#################################

fit_kwargs = dict(max_iter=args.max_iter, window_size=args.window_size,
                  shift_factor=args.shift_factor, seed=args.seed)

# ---- LEGACY: Ising SBM fits (see fit_ising_sbm / extract_graph above) ---- #
# log_msg(f"| UPDATE | Fitting Ising SBM without behaviour node")
# state_no_beh  = fit_ising_sbm(S_brain, **fit_kwargs)
# g_no_beh      = extract_graph(state_no_beh)
# blocks_no_beh = get_blocks(state_no_beh)
# log_msg(f"| UPDATE | Fit complete without behaviour node ({len(np.unique(blocks_no_beh))} blocks)")
# log_hierarchy(state_no_beh, 'no-behaviour')
#
# log_msg(f"| UPDATE | Fitting Ising SBM with behaviour node")
# state_with_beh   = fit_ising_sbm(S_full, **fit_kwargs)
# g_with_beh       = extract_graph(state_with_beh)
# blocks_with_beh  = get_blocks(state_with_beh)
# behaviour_node   = S_full.shape[0] - 1
# log_hierarchy(state_with_beh, 'with-behaviour')
# g_with_beh_brain = brain_only_graph(g_with_beh, behaviour_node)
# log_msg(f"| UPDATE | Fit complete with behaviour node ({len(np.unique(blocks_with_beh))} blocks)")
#
# relevance    = region_behaviour_relevance(X, behaviour)
# pclabel      = relevance_to_pclabel(relevance, n_bins=args.pclabel_bins)
# pclabel_bins = np.bincount(pclabel, minlength=args.pclabel_bins)
# log_msg(f"| UPDATE | Fitting Ising SBM with behaviour-informed pclabel")
# state_pclabel  = fit_ising_sbm(S_brain, pclabel=pclabel, **fit_kwargs)
# g_pclabel      = extract_graph(state_pclabel)
# blocks_pclabel = get_blocks(state_pclabel)
# log_hierarchy(state_pclabel, 'pclabel')
# log_msg(f"| UPDATE | Fit complete with pclabel ({len(np.unique(blocks_pclabel))} blocks)")

# ---- co-occurrence graph (no behaviour node) ---- #
log_msg(f"| UPDATE | Building co-occurrence graph without behaviour node")
g_cooc_no_beh = build_cooccurrence_graph(X)
log_msg(f"| UPDATE | Co-occurrence graph: {g_cooc_no_beh.num_vertices()} nodes, "
        f"{g_cooc_no_beh.num_edges()} multi-edges")

log_msg(f"| UPDATE | Fitting latent multigraph SBM without behaviour node")
state_no_beh, entropy_no_beh = fit_latent_multigraph(g_cooc_no_beh, **fit_kwargs)
g_no_beh      = extract_latent_graph(state_no_beh)
blocks_no_beh = get_blocks(state_no_beh)
log_msg(f"| UPDATE | Fit complete without behaviour node ({len(np.unique(blocks_no_beh))} blocks)")
log_hierarchy(state_no_beh, 'no-behaviour')

# ---- co-occurrence graph from behaviour-weighted lesion loads (no ---- #
# ---- behaviour node -- behaviour is baked into which lesions survive  ---- #
# ---- thresholding, not injected as a separate vertex or partition bias) ---- #
log_msg(f"| UPDATE | Building co-occurrence graph from behaviour-weighted lesion loads")
g_cooc_beh_weighted = build_cooccurrence_graph(X_beh_weighted)
log_msg(f"| UPDATE | Co-occurrence graph (behaviour-weighted): {g_cooc_beh_weighted.num_vertices()} nodes, "
        f"{g_cooc_beh_weighted.num_edges()} multi-edges")

log_msg(f"| UPDATE | Fitting latent multigraph SBM on behaviour-weighted co-occurrence")
state_beh_weighted, entropy_beh_weighted = fit_latent_multigraph(g_cooc_beh_weighted, **fit_kwargs)
g_beh_weighted      = extract_latent_graph(state_beh_weighted)
blocks_beh_weighted = get_blocks(state_beh_weighted)
log_msg(f"| UPDATE | Fit complete on behaviour-weighted co-occurrence ({len(np.unique(blocks_beh_weighted))} blocks)")
log_hierarchy(state_beh_weighted, 'behaviour-weighted')

# ---- co-occurrence graph with an added behaviour vertex ---- #
log_msg(f"| UPDATE | Building co-occurrence graph with behaviour node")
g_cooc_with_beh = build_cooccurrence_graph(X, behaviour_high=behaviour_high)
behaviour_node  = X.shape[1]
log_msg(f"| UPDATE | Co-occurrence graph (with behaviour): {g_cooc_with_beh.num_vertices()} nodes, "
        f"{g_cooc_with_beh.num_edges()} multi-edges")

log_msg(f"| UPDATE | Fitting latent multigraph SBM with behaviour node")
state_with_beh, entropy_with_beh = fit_latent_multigraph(g_cooc_with_beh, **fit_kwargs)
g_with_beh       = extract_latent_graph(state_with_beh)
blocks_with_beh  = get_blocks(state_with_beh)
log_hierarchy(state_with_beh, 'with-behaviour')
# three projections of the with-behaviour graph: (1) behaviour node removed,
# (2) full graph as fitted, (3) only edges incident to behaviour (2 minus 1,
# edge-wise) -- isolates which regions behaviour actually connects to.
g_with_beh_brain      = brain_only_graph(g_with_beh, behaviour_node)
g_with_beh_behaviour  = behaviour_only_graph(g_with_beh, behaviour_node)
log_msg(f"| UPDATE | Fit complete with behaviour node ({len(np.unique(blocks_with_beh))} blocks)")

# ---- behaviour-informed partition prior (pclabel), no behaviour node ---- #
relevance    = region_behaviour_relevance(X, behaviour)
pclabel      = relevance_to_pclabel(relevance, n_bins=args.pclabel_bins)
pclabel_bins = np.bincount(pclabel, minlength=args.pclabel_bins)
log_msg(f"| UPDATE | Region-behaviour relevance computed")
log_msg(f"| UPDATE | pclabel bin counts: {pclabel_bins.tolist()} ({args.pclabel_bins} bins, {len(pclabel)} regions)")

log_msg(f"| UPDATE | Fitting latent multigraph SBM with behaviour-informed pclabel")
state_pclabel, entropy_pclabel = fit_latent_multigraph(g_cooc_no_beh, pclabel=pclabel, **fit_kwargs)
g_pclabel      = extract_latent_graph(state_pclabel)
blocks_pclabel = get_blocks(state_pclabel)
log_hierarchy(state_pclabel, 'pclabel')
log_msg(f"| UPDATE | Fit complete with pclabel ({len(np.unique(blocks_pclabel))} blocks)")

# ---- validate: is the pclabel-informed fit reproducing pclabel verbatim, ---- #
# ---- or diverging from it toward brain-driven (unconstrained) structure? ---- #
vs_pclabel = partition_agreement(pclabel, blocks_pclabel)
log_msg(f"| UPDATE | learned blocks vs input pclabel: AMI={vs_pclabel['ami']:.3f}, "
        f"ARI={vs_pclabel['ari']:.3f}, identical={vs_pclabel['identical_partition']}")

vs_unconstrained = partition_agreement(blocks_no_beh, blocks_pclabel)
log_msg(f"| UPDATE | learned blocks vs unconstrained (no-behaviour) fit: AMI={vs_unconstrained['ami']:.3f}, "
        f"ARI={vs_unconstrained['ari']:.3f}, identical={vs_unconstrained['identical_partition']}")

if vs_pclabel['identical_partition']:
    log_msg(f"| UPDATE | WARNING: pclabel fit exactly reproduces the input pclabel — "
            f"the reconstruction is not free to evolve beyond the relevance-derived grouping")



#################################
#       BLOCK SELECTION         #
#################################

# For each of the three fits, at both level 0 (base partition) and level 1
# (one level up, projected down to per-node assignments via
# get_blocks(state, level=1)), select the blocks whose members are, on
# average, more behaviourally relevant than chance (z > 0 across blocks —
# see select_relevant_blocks): write one NIfTI per selected block, a single
# combined NIfTI carrying every block's behavioural value (not just
# selected ones), and a per-ROI roi_assignment summary table. Mirrors
# run.py's BLOCK COMMUNITY VOLUME section, adapted to this reconstruction
# (plain per-block mean relevance instead of consistency-weighted
# behaviour_degree, since this model has no posterior node-consistency to
# weight by) and to create_figures.py's per-level (lvl0/lvl1) convention.

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)

for label, state in [('no_beh', state_no_beh),
                     ('beh_weighted', state_beh_weighted),
                     ('beh_node', state_with_beh),
                     ('pclabel', state_pclabel)]:
    for level in [0, 1]:
        # with-behaviour blocks include one extra entry for the behaviour
        # vertex itself, which isn't an anatomical ROI — drop it before
        # atlas mapping (a no-op for the other two variants).
        blk = get_blocks(state, level=level)[:len(relevance)]

        selected_blocks, node_score, node_zscore, node_selected = select_relevant_blocks(blk, relevance)
        log_msg(f"| UPDATE | {label} lvl{level}: {len(selected_blocks)}/{len(np.unique(blk))} blocks "
                f"selected (z > 0) for NIfTI output")

        save_block_niftis(blk, selected_blocks, atlas_img, atlas_data, args.out_dir,
                          f'{args.atlas}_{label}_lvl{level}')

        value_nii_path = os.path.join(args.out_dir, f'{args.atlas}_{label}_lvl{level}_blockvalues.nii.gz')
        save_block_value_nifti(node_score, atlas_img, atlas_data, value_nii_path)

        roi_assignment_path = os.path.join(args.out_dir, f'roi_assignment_summary_{label}_lvl{level}.tsv')
        save_roi_assignment_table(node_names, blk, relevance, node_score, node_zscore, node_selected,
                                  roi_assignment_path)



#################################
#        SAVE OUTPUTS           #
#################################

# ---- states ---- #
state_no_beh_path = os.path.join(args.out_dir, 'recon_no_beh_state.pkl')
with open(state_no_beh_path, 'wb') as f:
    pickle.dump(state_no_beh, f)
log_msg(f"| UPDATE | Block state saved (no behaviour node) → {state_no_beh_path}")

state_beh_weighted_path = os.path.join(args.out_dir, 'recon_beh_weighted_state.pkl')
with open(state_beh_weighted_path, 'wb') as f:
    pickle.dump(state_beh_weighted, f)
log_msg(f"| UPDATE | Block state saved (behaviour-weighted) → {state_beh_weighted_path}")

state_with_beh_path = os.path.join(args.out_dir, 'recon_beh_node_state.pkl')
with open(state_with_beh_path, 'wb') as f:
    pickle.dump(state_with_beh, f)
log_msg(f"| UPDATE | Block state saved (with behaviour node) → {state_with_beh_path}")

state_pclabel_path = os.path.join(args.out_dir, 'recon_pclabel_state.pkl')
with open(state_pclabel_path, 'wb') as f:
    pickle.dump(state_pclabel, f)
log_msg(f"| UPDATE | Block state saved (pclabel) → {state_pclabel_path}")

# ---- region-behaviour relevance / pclabel ---- #
relevance_path = os.path.join(args.out_dir, 'recon_region_behaviour_relevance.npy')
np.save(relevance_path, relevance)
log_msg(f"| UPDATE | Region-behaviour relevance saved → {relevance_path}")

pclabel_path = os.path.join(args.out_dir, 'recon_pclabel.npy')
np.save(pclabel_path, pclabel)
log_msg(f"| UPDATE | pclabel saved → {pclabel_path}")

validation_path = os.path.join(args.out_dir, 'recon_pclabel_validation.csv')
with open(validation_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['comparison', 'ami', 'ari', 'identical_partition'])
    writer.writerow(['blocks_pclabel_vs_pclabel', vs_pclabel['ami'], vs_pclabel['ari'], vs_pclabel['identical_partition']])
    writer.writerow(['blocks_pclabel_vs_no_beh', vs_unconstrained['ami'], vs_unconstrained['ari'], vs_unconstrained['identical_partition']])
log_msg(f"| UPDATE | pclabel partition-agreement validation saved → {validation_path}")

# ---- graphs ---- #
# without-behaviour fit: brain nodes only (this is the natural output of that fit)
graph_no_beh_path = os.path.join(args.out_dir, 'recon_no_beh_graph.gt')
g_no_beh.save(graph_no_beh_path)
log_msg(f"| UPDATE | Graph saved (no behaviour node) → {graph_no_beh_path}")

graph_pclabel_path = os.path.join(args.out_dir, 'recon_pclabel_graph.gt')
g_pclabel.save(graph_pclabel_path)
log_msg(f"| UPDATE | Graph saved (pclabel) → {graph_pclabel_path}")

graph_beh_weighted_path = os.path.join(args.out_dir, 'recon_beh_weighted_graph.gt')
g_beh_weighted.save(graph_beh_weighted_path)
log_msg(f"| UPDATE | Graph saved (behaviour-weighted) → {graph_beh_weighted_path}")

# with-behaviour fit: the full (brain + behaviour) graph, a brain-only
# projection (behaviour node removed), and a behaviour-only projection
# (only edges incident to behaviour -- full minus brain-only, edge-wise)
graph_with_beh_full_path = os.path.join(args.out_dir, 'recon_beh_node_graph_full.gt')
g_with_beh.save(graph_with_beh_full_path)
log_msg(f"| UPDATE | Graph saved (with behaviour node, full) → {graph_with_beh_full_path}")

graph_with_beh_brain_path = os.path.join(args.out_dir, 'recon_beh_node_graph_brain_only.gt')
g_with_beh_brain.save(graph_with_beh_brain_path)
log_msg(f"| UPDATE | Graph saved (with behaviour node, brain-only projection) → {graph_with_beh_brain_path}")

graph_with_beh_behaviour_path = os.path.join(args.out_dir, 'recon_beh_node_graph_behaviour_only.gt')
g_with_beh_behaviour.save(graph_with_beh_behaviour_path)
log_msg(f"| UPDATE | Graph saved (with behaviour node, behaviour-only projection) → {graph_with_beh_behaviour_path}")

# ---- visualisations ---- #
# node/edge colour + alpha reflect behavioural relevance (region_behaviour_
# relevance), not plain block colour — see draw_state()
state_no_beh_draw_path = os.path.join(args.out_dir, 'recon_no_beh_state_draw.png')
draw_state(state_no_beh, relevance, output=state_no_beh_draw_path, node_names=node_names)
log_msg(f"| UPDATE | Block state visualisation saved (no behaviour node) → {state_no_beh_draw_path}")

state_beh_weighted_draw_path = os.path.join(args.out_dir, 'recon_beh_weighted_state_draw.png')
draw_state(state_beh_weighted, relevance, output=state_beh_weighted_draw_path, node_names=node_names)
log_msg(f"| UPDATE | Block state visualisation saved (behaviour-weighted) → {state_beh_weighted_draw_path}")

state_with_beh_draw_path = os.path.join(args.out_dir, 'recon_beh_node_state_draw.png')
draw_state(state_with_beh, relevance, output=state_with_beh_draw_path, node_names=node_names)
log_msg(f"| UPDATE | Block state visualisation saved (with behaviour node) → {state_with_beh_draw_path}")

state_pclabel_draw_path = os.path.join(args.out_dir, 'recon_pclabel_state_draw.png')
draw_state(state_pclabel, relevance, output=state_pclabel_draw_path, node_names=node_names)
log_msg(f"| UPDATE | Block state visualisation saved (pclabel) → {state_pclabel_draw_path}")

# ---- entropy trajectories ---- #
entropy_no_beh_path = os.path.join(args.out_dir, 'recon_no_beh_entropy.png')
plot_entropy(entropy_no_beh, 'no-behaviour', entropy_no_beh_path)
log_msg(f"| UPDATE | Entropy trajectory saved (no behaviour node) → {entropy_no_beh_path}")

entropy_beh_weighted_path = os.path.join(args.out_dir, 'recon_beh_weighted_entropy.png')
plot_entropy(entropy_beh_weighted, 'behaviour-weighted', entropy_beh_weighted_path)
log_msg(f"| UPDATE | Entropy trajectory saved (behaviour-weighted) → {entropy_beh_weighted_path}")

entropy_with_beh_path = os.path.join(args.out_dir, 'recon_beh_node_entropy.png')
plot_entropy(entropy_with_beh, 'with-behaviour', entropy_with_beh_path)
log_msg(f"| UPDATE | Entropy trajectory saved (with behaviour node) → {entropy_with_beh_path}")

entropy_pclabel_path = os.path.join(args.out_dir, 'recon_pclabel_entropy.png')
plot_entropy(entropy_pclabel, 'pclabel', entropy_pclabel_path)
log_msg(f"| UPDATE | Entropy trajectory saved (pclabel) → {entropy_pclabel_path}")

# ---- behaviour edges (from the with-behaviour fit) ---- #
# Edges incident to the behaviour vertex in the inferred latent graph.
# LatentMultigraphBlockState has no single scalar "coupling strength" per
# edge (no get_x() analog, unlike PseudoIsingBlockState); weight is instead
# the raw observed co-occurrence count between that region and behaviour_high,
# which for this count-based reconstruction is the natural, directly
# interpretable quantity.
beh_edges    = []
seen_regions = set()
for e in g_with_beh.edges():
    s, t = int(e.source()), int(e.target())
    if s == behaviour_node or t == behaviour_node:
        region = t if s == behaviour_node else s
        if region in seen_regions:
            continue
        seen_regions.add(region)
        weight = int(np.sum((X[:, region] == 1) & behaviour_high))
        beh_edges.append((region, weight))

beh_edges_path = os.path.join(args.out_dir, 'recon_beh_node_behaviour_edges.npy')
np.save(beh_edges_path, np.array(beh_edges))
log_msg(f"| UPDATE | Behaviour edges saved ({len(beh_edges)} edges) → {beh_edges_path}")

log_msg(f"| FINISHED | All outputs saved")
