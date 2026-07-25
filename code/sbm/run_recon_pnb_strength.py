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
# The following script performs PseudoNormalBlockState reconstruction   #
# of brain-region structure directly from continuous per-subject node   #
# STRENGTH values (normative-disconnectome-based), with no              #
# binarization/co-occurrence-counting step. Copied from                 #
# run_recon_pnb.py, which assumes 0-100 lesion-load values already      #
# comparable across subjects. Node strength has no such cross-subject   #
# comparability: it reflects each subject's own individual connection-  #
# strength profile, so a GLOBAL min-max rescale would conflate real     #
# between-subject differences in absolute strength with the node-to-    #
# node PATTERN this model should reconstruct. This script therefore     #
# rescales each subject's own row to [0, 1] independently               #
# (rescale_01_within_subject) instead of run_recon_pnb.py's global      #
# rescale_01 -- everything else (fit procedure, behaviour weighting,    #
# outputs) is unchanged.                                                #
#                                                                       #
# Reconstruction is fit via plain single-flip mcmc_sweep() (NOT         #
# multiflip), which was found to be essential — multiflip search never  #
# finds structure for this model, while single-flip search reliably    #
# finds and keeps it (see project_run_recon_ising_fit_investigation     #
# memory, "REVERSAL" section).                                          #
#                                                                       #
# Two variants are fit: no_beh (rescaled node-strength values) and      #
# beh_weighted (multiplied by a rescaled behaviour score before         #
# reconstruction). Appending behaviour as an extra correlated node was  #
# tested and empirically found to never change the outcome relative to #
# no_beh (this SBM groups nodes by pairwise correlation profile; it has #
# no mechanism to condition on a third variable) — so that variant was  #
# dropped in favour of weighting, the only mechanism that showed any    #
# ability to recover structure that behaviour-blind reconstruction      #
# alone could not (see memory, "behaviour-gated interaction synthetic   #
# data" section).                                                       #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/23.                                              #
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
from tqdm import tqdm
from utils import log_msg



#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Run PseudoNormalBlockState reconstruction directly on continuous node-strength data.')
args.add_argument('--data_file', type=str, default='/mnt/h/RT/data/Schaefer2018-400_node_strength.tsv',
                  help='Path to the subject x node continuous-value .tsv file')
args.add_argument('--out_dir', type=str, default='/mnt/h/RT/data/RESULTS/SBMRECONPNB_STRENGTH', help='Path to the output directory')
args.add_argument('--data_path', type=str, default='/mnt/h/RT/data',
                  help='Path to the data directory containing participants.tsv')
args.add_argument('--atlas', type=str, default='Schaefer2018-400',
                  help='Atlas name; looks up {data_path}/ATLAS/{atlas}.nii.gz for block-'
                       'selection NIfTI output (default: Schaefer2018-400)')
args.add_argument('--score', type=str, default='Foreperiod_Long_tau',
                  help='Behaviour score column in participants.tsv to match against')
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

log_msg(f"| START | Running PseudoNormalBlockState reconstruction on continuous node-strength data")
log_msg(f"| UPDATE | Data file: {args.data_file}")
log_msg(f"| UPDATE | Output directory: {args.out_dir}")

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)



#################################
#           FUNCTIONS           #
#################################

def fit_pseudo_normal(s, max_iter=10000, window_size=250, shift_factor=0.75, seed=42):
    '''
    fit a nested PseudoNormalBlockState via finite-temperature, single-flip
    MCMC (plain state.mcmc_sweep(), never multiflip_mcmc_sweep()) with
    mean-shift change-point detection, mirroring run_recon.py's
    fit_latent_multigraph loop structure. Single-flip search is essential
    here: multiflip search on PseudoNormalBlockState never finds known
    synthetic structure at all (best AMI ~0.03-0.08 across 2000 sweeps),
    while plain single-flip search reliably finds and keeps it (AMI=1.000,
    stable, reproducible across seeds and across the full realistic
    density range) — see project_run_recon_ising_fit_investigation memory,
    "REVERSAL" section. This is the opposite of LatentMultigraphBlockState,
    whose default mcmc_sweep() already includes multiflip and needs no
    special handling.

    s must be a nodes x subjects continuous-value array (no binarization).

    Returns (state, entropy_traj) — entropy_traj is the full per-iteration
    entropy trace, for post-hoc convergence inspection (see plot_entropy).
    '''
    gt.seed_rng(seed)

    state = gt.PseudoNormalBlockState(s, nested=True)

    entropy_traj      = []
    converged         = False
    conv_iter         = max_iter
    first_window_mean = None
    first_window_std  = None
    threshold         = None

    with tqdm(total=max_iter, desc='MCMC', unit='iter') as pbar:
        for i in range(max_iter):
            state.mcmc_sweep(niter=1)
            entropy_traj.append(state.entropy())

            if i == window_size - 1:
                first_window_mean = sum(entropy_traj) / window_size
                first_window_std  = (sum((x - first_window_mean) ** 2 for x in entropy_traj) / window_size) ** 0.5
                threshold = first_window_mean - shift_factor * first_window_std
                pbar.set_postfix(ref=f'{first_window_mean:.1f}', thr=f'{threshold:.1f}')
            elif threshold is not None:
                w_mean = sum(entropy_traj[-window_size:]) / window_size
                if w_mean < threshold:
                    converged = True
                    conv_iter = i
                    pbar.set_postfix(DL=f'{entropy_traj[-1]:.1f}', w_mean=f'{w_mean:.1f}', status='SHIFT')
                    pbar.update(1)
                    break

            if i % 100 == 0:
                pbar.set_postfix(DL=f'{entropy_traj[-1]:.1f}')
            pbar.update(1)

    if converged:
        log_msg(f"| UPDATE | fit_pseudo_normal: mean-shift change point at iteration {conv_iter} "
                f"(ref={first_window_mean:.2f}, threshold={threshold:.2f})")
    else:
        log_msg(f"| WARNING | fit_pseudo_normal: no mean shift detected within {max_iter} iterations "
                f"(ref={first_window_mean}, threshold={threshold}) — using final state. "
                f"Consider increasing max_iter or reducing shift_factor (current: {shift_factor}).")

    return state, entropy_traj


def plot_entropy(entropy_traj, label, output, skip_iters=20):
    '''
    plot the entropy (description length) trajectory across all MCMC
    iterations of a fit_pseudo_normal() run, to visually inspect
    convergence behaviour. PseudoNormalBlockState's initial entropy (before
    any adaptation) can be many orders of magnitude larger than where it
    settles -- a single plot spanning the whole range makes every
    post-burn-in iteration look like a flat line at zero, hiding whether
    the search is actually still moving there. Mirrors run.py's two-panel
    entropy plot: left panel is the full trajectory (with the initial
    collapse visible), right panel re-scales to entropy_traj[skip_iters:]
    only, on its own y-axis, so genuine post-burn-in dynamics are visible.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(entropy_traj, linewidth=1, color='steelblue')
    axes[0].set_xlabel('MCMC iteration')
    axes[0].set_ylabel('Entropy (description length)')
    axes[0].set_title('Full trajectory')

    tail = entropy_traj[skip_iters:] if len(entropy_traj) > skip_iters else entropy_traj
    axes[1].plot(range(len(entropy_traj) - len(tail), len(entropy_traj)), tail,
                linewidth=1, color='seagreen')
    axes[1].set_xlabel('MCMC iteration')
    axes[1].set_ylabel('Entropy (description length)')
    axes[1].set_title(f'Post-burn-in (iteration {skip_iters}+)')

    plt.suptitle(f'Entropy trajectory — {label}')
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def region_behaviour_relevance(vals, behaviour):
    '''
    per-region Spearman correlation between per-subject node values (raw
    continuous node-strength here — Spearman is rank-based so this works
    identically for binary or continuous input) and the raw behaviour
    score. Regions with zero variance in this sample get relevance 0.
    '''
    n_nodes   = vals.shape[1]
    relevance = np.zeros(n_nodes)
    for i in range(n_nodes):
        if vals[:, i].std() == 0:
            continue
        relevance[i], _ = spearmanr(vals[:, i], behaviour)
    return np.nan_to_num(relevance)


def select_relevant_blocks(blocks, relevance):
    '''
    score each block by its members' mean behavioural relevance, z-score
    across blocks, and select blocks with above-average (z > 0) relevance
    -- mirroring run.py's block-selection approach.

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


def extract_graph(state):
    '''
    extract the reconstructed latent graph from a fitted
    PseudoNormalBlockState, registering the fitted coupling strength as an
    internal edge property ('x') so it survives g.save().
    '''
    g = state.get_graph()
    g.ep['x'] = state.get_x()
    return g


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
    of its endpoints' colours. See run_recon.py's draw_state for the full
    rationale behind the edge_visible_percentile cutoff and
    vertex_text_position choice — reused verbatim here.

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


def rescale_01(x):
    '''min-max rescale a vector to [0, 1]; constant input maps to all-1s'''
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return np.ones_like(x) if span == 0 else (x - x.min()) / span


def rescale_01_within_subject(vals):
    '''
    min-max rescale each SUBJECT'S OWN ROW to [0, 1] independently, rather
    than a single global min-max across all subjects (rescale_01). Node-
    strength values from the normative disconnectome reflect each
    subject's own individual connection-strength profile -- cross-subject
    normalization doesn't make sense here, since a global rescale would
    conflate real between-subject differences in absolute strength with
    the node-to-node pattern of relative strength this model should
    reconstruct. A subject with a fully constant row maps to all-1s,
    matching rescale_01's convention.
    '''
    vals    = np.asarray(vals, dtype=np.float64)
    row_min = vals.min(axis=1, keepdims=True)
    row_max = vals.max(axis=1, keepdims=True)
    span    = row_max - row_min
    return np.where(span == 0, 1.0, (vals - row_min) / np.where(span == 0, 1.0, span))


def load_node_matrix(data_file):
    '''
    load the subject x node continuous-value matrix, no binarization --
    PseudoNormalBlockState consumes raw values directly. Returns the raw
    values, the subject list, and the node/ROI names (from the file's
    header row).
    '''
    data       = np.genfromtxt(data_file, delimiter='\t', dtype=str)
    node_names = data[0, 1:].tolist()
    vals       = data[1:, 1:].astype(float)
    return vals, data[1:, 0].tolist(), node_names


def load_data(data_file, data_path, score='Foreperiod_Long_tau'):
    '''
    load the subject x node continuous-value matrix (no binarization), match
    subjects to their behaviour score in participants.tsv, drop subjects
    with missing behaviour, mask subjects with zero variance across all
    nodes (uninformative for reconstruction -- and, for an all-constant
    nonzero row, likely a data/loading artifact), and return the clean raw
    values, the raw behaviour, the subject list, missing-score list, and
    node/ROI names.
    '''
    raw_vals, subject_list, node_names = load_node_matrix(data_file)

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

    raw_vals_clean      = raw_vals[keep_idx]
    subject_list_clean  = [subject_list[i] for i in keep_idx]
    behaviour            = np.array(behaviour, dtype=np.float64)

    row_std           = raw_vals_clean.std(axis=1)
    zero_var_subjects = np.where(row_std == 0)[0]
    for i in zero_var_subjects:
        log_msg(f"| WARNING | subject {subject_list_clean[i]} has zero variance across all "
                f"{raw_vals_clean.shape[1]} nodes (mean={raw_vals_clean[i].mean():.3f}) — masking from further analysis")

    keep_var            = row_std > 0
    raw_vals_clean       = raw_vals_clean[keep_var]
    behaviour            = behaviour[keep_var]
    subject_list_clean  = [s for s, k in zip(subject_list_clean, keep_var) if k]

    return raw_vals_clean, behaviour, subject_list_clean, subjects_missing_score, node_names



#################################
#          LOAD DATA            #
#################################

raw_vals, behaviour, subject_list, subjects_missing_score, node_names = load_data(
    args.data_file, args.data_path, args.score
)

log_msg(f"| UPDATE | Subjects: {raw_vals.shape[0]}, nodes: {raw_vals.shape[1]}")
log_msg(f"| UPDATE | Behaviour score: {args.score}")
log_msg(f"| UPDATE | Subjects missing behaviour: {len(subjects_missing_score)}")

# S_brain: nodes x subjects, continuous, as PseudoNormalBlockState expects.
# raw_vals is rescaled to [0, 1] WITHIN EACH SUBJECT'S OWN ROW (not a
# global min-max across subjects -- see rescale_01_within_subject), since
# node-strength values aren't comparable across subjects the way lesion-
# load percentages are. behaviour is still rescaled globally (a single
# scalar per subject, not a profile with its own cross-subject scale
# issue).
# beh_weighted: rescaled node-strength values multiplied by the rescaled
# behaviour score (per subject) before reconstruction -- amplifies each
# subject's strength pattern in proportion to their behaviour score,
# rather than adding behaviour as a separate correlated dimension (which
# was tested and found to never change the recovered structure relative
# to no_beh at all).
raw_vals_01      = rescale_01_within_subject(raw_vals)
behaviour_weight = rescale_01(behaviour)
raw_vals_weighted = raw_vals_01 * behaviour_weight[:, np.newaxis]

S_brain    = raw_vals_01.T
S_weighted = raw_vals_weighted.T



#################################
#          FIT MODEL            #
#################################

fit_kwargs = dict(max_iter=args.max_iter, window_size=args.window_size,
                  shift_factor=args.shift_factor, seed=args.seed)

log_msg(f"| UPDATE | Fitting PseudoNormalBlockState without behaviour weighting (single-flip search)")
state_no_beh, entropy_no_beh = fit_pseudo_normal(S_brain, **fit_kwargs)
g_no_beh      = extract_graph(state_no_beh)
blocks_no_beh = get_blocks(state_no_beh)
log_msg(f"| UPDATE | Fit complete without behaviour weighting ({len(np.unique(blocks_no_beh))} blocks)")
log_hierarchy(state_no_beh, 'no-behaviour')

log_msg(f"| UPDATE | Fitting PseudoNormalBlockState with behaviour-weighted node-strength values (single-flip search)")
state_beh_weighted, entropy_beh_weighted = fit_pseudo_normal(S_weighted, **fit_kwargs)
g_beh_weighted      = extract_graph(state_beh_weighted)
blocks_beh_weighted = get_blocks(state_beh_weighted)
log_msg(f"| UPDATE | Fit complete with behaviour-weighted node-strength values ({len(np.unique(blocks_beh_weighted))} blocks)")
log_hierarchy(state_beh_weighted, 'behaviour-weighted')

relevance = region_behaviour_relevance(raw_vals, behaviour)
log_msg(f"| UPDATE | Region-behaviour relevance computed")



#################################
#       BLOCK SELECTION         #
#################################

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)

for label, state in [('no_beh', state_no_beh),
                     ('beh_weighted', state_beh_weighted)]:
    for level in [0, 1]:
        blk = get_blocks(state, level=level)

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
state_no_beh_path = os.path.join(args.out_dir, 'recon_pnb_no_beh_state.pkl')
with open(state_no_beh_path, 'wb') as f:
    pickle.dump(state_no_beh, f)
log_msg(f"| UPDATE | Block state saved (no behaviour) → {state_no_beh_path}")

state_beh_weighted_path = os.path.join(args.out_dir, 'recon_pnb_beh_weighted_state.pkl')
with open(state_beh_weighted_path, 'wb') as f:
    pickle.dump(state_beh_weighted, f)
log_msg(f"| UPDATE | Block state saved (behaviour-weighted) → {state_beh_weighted_path}")

# ---- region-behaviour relevance ---- #
relevance_path = os.path.join(args.out_dir, 'recon_pnb_region_behaviour_relevance.npy')
np.save(relevance_path, relevance)
log_msg(f"| UPDATE | Region-behaviour relevance saved → {relevance_path}")

# ---- graphs ---- #
graph_no_beh_path = os.path.join(args.out_dir, 'recon_pnb_no_beh_graph.gt')
g_no_beh.save(graph_no_beh_path)
log_msg(f"| UPDATE | Graph saved (no behaviour) → {graph_no_beh_path}")

graph_beh_weighted_path = os.path.join(args.out_dir, 'recon_pnb_beh_weighted_graph.gt')
g_beh_weighted.save(graph_beh_weighted_path)
log_msg(f"| UPDATE | Graph saved (behaviour-weighted) → {graph_beh_weighted_path}")

# ---- visualisations ---- #
# node/edge colour + alpha reflect behavioural relevance (region_behaviour_
# relevance), not plain block colour — see draw_state()
state_no_beh_draw_path = os.path.join(args.out_dir, 'recon_pnb_no_beh_state_draw.png')
draw_state(state_no_beh, relevance, output=state_no_beh_draw_path, node_names=node_names)
log_msg(f"| UPDATE | Block state visualisation saved (no behaviour) → {state_no_beh_draw_path}")

state_beh_weighted_draw_path = os.path.join(args.out_dir, 'recon_pnb_beh_weighted_state_draw.png')
draw_state(state_beh_weighted, relevance, output=state_beh_weighted_draw_path, node_names=node_names)
log_msg(f"| UPDATE | Block state visualisation saved (behaviour-weighted) → {state_beh_weighted_draw_path}")

# ---- entropy trajectories ---- #
# raw per-iteration values saved alongside the plot, matching run.py's
# entropy_trajectory_{score}.npy pattern -- lets the trajectory be
# re-inspected (e.g. log-scale, different burn-in cutoff) without rerunning
# the fit.
entropy_no_beh_traj_path = os.path.join(args.out_dir, 'recon_pnb_no_beh_entropy_trajectory.npy')
np.save(entropy_no_beh_traj_path, np.array(entropy_no_beh))
log_msg(f"| UPDATE | Entropy trajectory data saved (no behaviour) → {entropy_no_beh_traj_path}")

entropy_no_beh_path = os.path.join(args.out_dir, 'recon_pnb_no_beh_entropy.png')
plot_entropy(entropy_no_beh, 'no-behaviour', entropy_no_beh_path)
log_msg(f"| UPDATE | Entropy trajectory plot saved (no behaviour) → {entropy_no_beh_path}")

entropy_beh_weighted_traj_path = os.path.join(args.out_dir, 'recon_pnb_beh_weighted_entropy_trajectory.npy')
np.save(entropy_beh_weighted_traj_path, np.array(entropy_beh_weighted))
log_msg(f"| UPDATE | Entropy trajectory data saved (behaviour-weighted) → {entropy_beh_weighted_traj_path}")

entropy_beh_weighted_path = os.path.join(args.out_dir, 'recon_pnb_beh_weighted_entropy.png')
plot_entropy(entropy_beh_weighted, 'behaviour-weighted', entropy_beh_weighted_path)
log_msg(f"| UPDATE | Entropy trajectory plot saved (behaviour-weighted) → {entropy_beh_weighted_path}")

log_msg(f"| FINISHED | All outputs saved")
