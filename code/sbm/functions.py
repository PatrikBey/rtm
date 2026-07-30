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
# The following script contains wrapper functions used in modelling     #
# of various intelligence related cognitive domains using a multi-layer #
# nested stochastic block modelling (SBM) framework                     #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/01.                                              #
#                                                                       #
#                                                                       #
#########################################################################


# ---- import libraries ---- #
import numpy as np
import graph_tool.all as gt
from graph_tool import inference
from tqdm import tqdm
from utils import log_msg

_DIST_TO_REC = {'normal': 'real-normal', 'poisson': 'discrete-poisson'}



#########################
#   BUILD GRAPH OBJECT  #
#########################

def create_multilayer_graph(adjacency_matrices, behavioral_values, node_names,
                            edge_threshold=50,
                            behaviour_dist='normal', cooccurrence_dist='normal',
                            combined_layers=True):
    """
    Create a redundant two-layer graph for weighted nested SBM inference.

    Both layers default to real-normal, which is scale-invariant — raw weights
    are used directly. If cooccurrence_dist='poisson', the cooccurrence layer is
    [0,1] min-max scaled to reduce scale impact (discrete-Poisson DL scales
    linearly with weight magnitude).

    Edge structure: each layer is thresholded independently at edge_threshold
    percentile; only edges surviving in both layers are retained (intersection).

    Parameters
    ----------
    adjacency_matrices : ndarray, shape (n_patients, n_nodes, n_nodes)
    behavioral_values  : ndarray, shape (n_patients,)  — raw tau values (> 0)
    node_names         : list, length n_nodes
    edge_threshold     : float, percentile threshold applied per layer (default 50)

    Returns
    -------
    graph : graph_tool.Graph with edge properties
            'behaviour_weight', 'cooccurrence_weight', 'layer'
    """
    n_patients, n_nodes_dim1, n_nodes_dim2 = adjacency_matrices.shape
    assert n_nodes_dim1 == n_nodes_dim2
    assert len(behavioral_values) == n_patients
    assert len(node_names) == n_nodes_dim1
    assert 0 <= edge_threshold <= 100

    behaviour_weighted = np.zeros((n_nodes_dim1, n_nodes_dim1))
    for i in range(n_patients):
        behaviour_weighted += adjacency_matrices[i] * behavioral_values[i]
    cooccurrence_binary = np.sum(adjacency_matrices, axis=0).astype(float)

    # If cooccurrence layer uses Poisson, min-max scale to [0,1] to reduce
    # scale impact (Poisson DL is not scale-invariant; raw counts would dominate)
    if cooccurrence_dist == 'poisson':
        nz = cooccurrence_binary[cooccurrence_binary > 0]
        if nz.size > 0:
            lo, hi = nz.min(), nz.max()
            cooccurrence_binary[cooccurrence_binary > 0] = \
                (cooccurrence_binary[cooccurrence_binary > 0] - lo) / (hi - lo) \
                if hi > lo else 1.0

    # Threshold each layer independently on raw (or scaled) values
    beh_nz = behaviour_weighted[behaviour_weighted > 0]
    occ_nz = cooccurrence_binary[cooccurrence_binary > 0]
    beh_thresh = np.percentile(beh_nz, edge_threshold) if beh_nz.size > 0 else 0
    occ_thresh = np.percentile(occ_nz, edge_threshold) if occ_nz.size > 0 else 0
    behaviour_weighted[behaviour_weighted < beh_thresh] = 0
    cooccurrence_binary[cooccurrence_binary < occ_thresh] = 0
    threshold_value = (beh_thresh, occ_thresh)

    # Edge masks after thresholding
    beh_mask = behaviour_weighted > 0
    occ_mask = cooccurrence_binary > 0

    if combined_layers:
        # Intersection: enforce identical edge structure across both layers
        shared = beh_mask & occ_mask
        beh_mask = shared
        occ_mask = shared

    g = gt.Graph(directed=False)
    g.add_vertex(n_nodes_dim1)

    node_label_prop = g.new_vertex_property("string")
    for idx in range(n_nodes_dim1):
        node_label_prop[g.vertex(idx)] = str(node_names[idx])
    g.vp.label = node_label_prop

    behaviour_weight_prop    = g.new_edge_property("double")
    cooccurrence_weight_prop = g.new_edge_property("double")
    layer_prop               = g.new_edge_property("int")

    for i in range(n_nodes_dim1):
        for j in range(i + 1, n_nodes_dim1):
            has_beh = bool(beh_mask[i, j])
            has_occ = bool(occ_mask[i, j])
            if not (has_beh or has_occ):
                continue

            if has_beh:
                e0 = g.add_edge(g.vertex(i), g.vertex(j))
                behaviour_weight_prop[e0]    = float(behaviour_weighted[i, j])
                cooccurrence_weight_prop[e0] = 0.0
                layer_prop[e0]               = 0

            if has_occ:
                e1 = g.add_edge(g.vertex(i), g.vertex(j))
                behaviour_weight_prop[e1]    = 0.0
                cooccurrence_weight_prop[e1] = float(cooccurrence_binary[i, j])
                layer_prop[e1]               = 1

    g.ep.behaviour_weight    = behaviour_weight_prop
    g.ep.cooccurrence_weight = cooccurrence_weight_prop
    g.ep.layer               = layer_prop

    g.gp.n_patients              = g.new_graph_property("int",    n_patients)
    g.gp.edge_threshold_applied  = g.new_graph_property("string", str(threshold_value))

    return g


#########################
#      NULL MODELS      #
#########################

def permute_behaviour(behaviour, rng):
    """
    Shuffle the subject -> behaviour-score assignment, leaving lesion data
    untouched. Used to build the behaviour-permutation null: it breaks the
    subject-specific link between lesion pattern and behaviour while
    preserving the cohort's lesion topology and score distribution.

    Parameters
    ----------
    behaviour : sequence of float, length n_patients
    rng       : np.random.Generator

    Returns
    -------
    list of float, length n_patients
    """
    return rng.permutation(np.asarray(behaviour, dtype=float)).tolist()


def quick_fit_dl(graph, behaviour_dist='normal', cooccurrence_dist='normal', seed=42):
    """
    Fast, single-shot nested-SBM fit used as the test statistic for null-model
    comparisons. Uses the same LayeredBlockState construction as
    fit_nested_sbm_layered but skips the MCMC change-point search and
    posterior accumulation — a single gt.minimize_nested_blockmodel_dl call
    is enough to compare relative description length / block count across
    many permutations without the cost of the full pipeline.

    Parameters
    ----------
    graph              : graph_tool.Graph from create_multilayer_graph()
    behaviour_dist     : 'normal' or 'poisson'
    cooccurrence_dist  : 'normal' or 'poisson'
    seed               : int — graph_tool RNG seed

    Returns
    -------
    dict with keys: 'entropy', 'n_levels', 'meaningful_levels', 'level0_n_blocks'
    """
    gt.seed_rng(seed)
    g = graph.copy()

    state = gt.minimize_nested_blockmodel_dl(
        g,
        state_args=dict(
            base_type=gt.LayeredBlockState,
            state_args=dict(
                ec=g.ep.layer,
                recs=[g.ep.behaviour_weight, g.ep.cooccurrence_weight],
                rec_types=[_DIST_TO_REC[behaviour_dist], _DIST_TO_REC[cooccurrence_dist]],
                layers=True,
                deg_corr=True
            )
        )
    )

    levels    = state.get_levels()
    n_levels  = len(levels)
    entropies = [lv.entropy() for lv in levels]
    floor     = min(entropies)
    meaningful_levels = [k for k in range(n_levels) if entropies[k] > floor]

    return {
        'entropy':           state.entropy(),
        'n_levels':          n_levels,
        'meaningful_levels': meaningful_levels,
        'level0_n_blocks':   levels[0].get_nonempty_B(),
    }


#########################
#     SBM FITTING       #
#########################

def fit_nested_sbm_layered(graph,
                           max_iter=10000,
                           window_size=500,
                           shift_factor=0.5,
                           behaviour_dist='normal',
                           cooccurrence_dist='poisson',
                           seed=42):
    """
    Fit nested layered stochastic block model to multi-layer weighted graph.

    Sampling strategy
    -----------------
    After annealing and a short burn-in, a single MCMC loop runs up to max_iter
    sweeps. The first window_size iterations establish a reference mean and
    standard deviation. The change point is the first subsequent iteration where
    the sliding window mean drops below:

        threshold = first_window_mean - shift_factor * first_window_std

    This detects a meaningful shift in the mean entropy level. The standard
    deviation of the initial window defines the natural scale of variability;
    shift_factor controls how many of those units constitute a real shift rather
    than noise. Once the change point is detected, window_size additional
    accumulation sweeps are run and block assignments are collected from those.
    If max_iter is reached without a shift a warning is logged and accumulation
    proceeds from the final state.

    Parameters
    ----------
    graph            : graph_tool.Graph from create_multilayer_graph()
    max_iter         : int   — maximum MCMC sweeps for change-point search (default 10000)
    window_size      : int   — rolling window length and accumulation sample count
                               (default 500)
    shift_factor     : float — threshold = first_window_mean - shift_factor * first_window_std;
                               change point fires when sliding window mean drops below
                               this value (default 0.5)

    Returns
    -------
    results : dict
        'state'                 : final NestedBlockState
        'entropy'               : final model entropy (description length)
        'n_levels'              : total hierarchy levels
        'meaningful_levels'     : list of level indices with > 1 block
        'levels_n_blocks'       : non-empty block count per level
        'levels_entropy'        : entropy per level
        'entropy_trajectory'    : entropy at every main-loop iteration
        'entropy_converged'     : entropy across the window_size accumulation sweeps
        'convergence_iteration' : iteration of detected change point (max_iter if none)
        'n_converged_samples'   : window_size
        'converged'             : bool — True if change point was detected
        'modal_assignments'     : {level: (n_nodes,) int array}
        'block_connectivity'    : {level: (B x B) float ndarray}
        'node_consistency'      : {level: (n_nodes,) float array in [0,1]}
        'edge_mean'             : (n_nodes x n_nodes) posterior mean mrs[b_i,b_j]
        'edge_var'              : (n_nodes x n_nodes) posterior variance
        'max_iter'              : value used
        'window_size'           : value used
        'shift_factor'          : value used
    """

    gt.seed_rng(seed)
    g = graph.copy()

    # ---- Initialise nested block state ---- #
    state = gt.minimize_nested_blockmodel_dl(
        g,
        state_args=dict(
            base_type=gt.LayeredBlockState,
            state_args=dict(
                ec=g.ep.layer,
                recs=[g.ep.behaviour_weight, g.ep.cooccurrence_weight],
                rec_types=[_DIST_TO_REC[behaviour_dist], _DIST_TO_REC[cooccurrence_dist]],
                layers=True,
                deg_corr=True
            )
        )
    )

    n_verts = g.num_vertices()

    # ---- Main MCMC loop with mean-shift change-point detection ---- #
    # The first window_size iterations establish a reference mean and std.
    # The threshold is first_window_mean - shift_factor * first_window_std.
    # Change point = first iteration where the sliding window mean drops below
    # that threshold, indicating a meaningful shift in entropy level.
    entropy_traj      = []
    converged         = False
    conv_iter         = max_iter
    first_window_mean = None
    first_window_std  = None
    threshold         = None

    log_msg('| UPDATE | starting MCMC change-point detection loop')
    with tqdm(total=max_iter, desc='MCMC', unit='iter') as pbar:
        for i in range(max_iter):
            state.mcmc_sweep(niter=1)
            entropy_traj.append(state.entropy())

            if i == window_size - 1:
                first_window_mean = sum(entropy_traj) / window_size
                first_window_std  = (
                    sum((x - first_window_mean) ** 2 for x in entropy_traj) / window_size
                ) ** 0.5
                threshold = first_window_mean - shift_factor * first_window_std
                pbar.set_postfix(ref=f'{first_window_mean:.1f}',
                                 thr=f'{threshold:.1f}')

            elif threshold is not None:
                w_mean = sum(entropy_traj[-window_size:]) / window_size
                if w_mean < threshold:
                    converged = True
                    conv_iter = i
                    pbar.set_postfix(DL=f'{entropy_traj[-1]:.1f}',
                                     w_mean=f'{w_mean:.1f}',
                                     status='SHIFT')
                    pbar.update(1)
                    break

            if i % 100 == 0:
                pbar.set_postfix(DL=f'{entropy_traj[-1]:.1f}')
            pbar.update(1)

    dS = np.array(entropy_traj)

    if converged:
        log_msg(f'| UPDATE | mean-shift change point at iteration {conv_iter} '
                f'| ref={first_window_mean:.2f}, std={first_window_std:.2f}, '
                f'threshold={threshold:.2f}, shift_factor={shift_factor}')
    else:
        log_msg(f'| WARNING | no mean shift detected within {max_iter} iterations '
                f'| ref={first_window_mean:.2f}, threshold={threshold:.2f} '
                f'— using last {window_size} iterations. '
                f'Consider increasing max_iter or reducing shift_factor '
                f'(current: {shift_factor}).')

    # ---- Meaningful levels ---- #
    _levels_init    = state.get_levels()
    n_levels        = len(_levels_init)
    _entropies_init = [lv.entropy() for lv in _levels_init]
    entropy_floor   = min(_entropies_init)
    meaningful_levels = [k for k in range(n_levels)
                         if _entropies_init[k] > entropy_floor]

    # ---- Accumulation: window_size sweeps from converged state ---- #
    # Runs a fresh set of window_size sweeps from the current state and collects
    # block assignments and joint mrs matrices for inference. Memory is bounded:
    # only window_size samples are ever stored simultaneously.
    b_history    = {k: [] for k in meaningful_levels}
    b0_history   = []
    mrs0_history = []

    edge_M1 = np.zeros((n_verts, n_verts))
    edge_M2 = np.zeros((n_verts, n_verts))

    dS_converged = np.zeros(window_size)
    log_msg(f'| UPDATE | accumulating {window_size} samples from converged state')
    with tqdm(total=window_size, desc='Accumulation', unit='iter') as pbar:
        for i in range(window_size):
            state.mcmc_sweep(niter=1)
            dS_converged[i] = state.entropy()

            lv0         = state.get_levels()[0]
            b0_arr      = lv0.get_blocks().a.copy()
            mrs0_sparse = lv0.get_matrix()
            mrs0_arr    = mrs0_sparse.toarray().astype(float) \
                          if hasattr(mrs0_sparse, 'toarray') \
                          else np.array(mrs0_sparse, dtype=float)
            b0_history.append(b0_arr)
            mrs0_history.append(mrs0_arr)

            B0_size = mrs0_arr.shape[0]
            b_clip  = np.minimum(b0_arr, B0_size - 1).astype(int)
            vals    = mrs0_arr[np.ix_(b_clip, b_clip)]
            edge_M1 += vals
            edge_M2 += vals ** 2

            for k in meaningful_levels:
                proj  = state.project_partition(k, 0)
                b_arr = proj.a.copy() if hasattr(proj, 'a') else \
                        np.array([proj[v] for v in g.vertices()])
                b_history[k].append(b_arr)

            if i % 50 == 0:
                pbar.set_postfix(DL=f'{dS_converged[i]:.1f}')
            pbar.update(1)

    edge_mean = edge_M1 / window_size
    edge_var  = np.maximum(edge_M2 / window_size - edge_mean ** 2, 0.0)

    # ---- Modal partitions + node assignment consistency (Cohen's Kappa) ---- #
    # PartitionModeState resolves label switching and gives:
    #   get_max(g)      — modal (most probable) block per node
    #   get_marginal(g) — posterior probability of each block per node
    # node_consistency[k][i] = chance-corrected consistency (Cohen's Kappa):
    #   kappa = (P_modal - 1/n_blocks) / (1 - 1/n_blocks)
    # kappa=1: always in modal block; kappa=0: at chance level; kappa<0: below chance.
    # This accounts for the fact that with more blocks, even moderate raw probabilities
    # represent strong consistency relative to random assignment.
    modal_assignments = {}
    node_consistency  = {}
    for k in meaningful_levels:
        pmode    = gt.PartitionModeState(b_history[k], converge=True)
        b_mode   = pmode.get_max(g)
        b_modal  = b_mode.a.copy() if hasattr(b_mode, 'a') else \
                   np.array([b_mode[v] for v in g.vertices()])
        modal_assignments[k] = b_modal

        n_blocks  = int(b_modal.max()) + 1
        chance    = 1.0 / n_blocks
        n_samples = len(b_history[k])
        marginals = pmode.get_marginal(g)
        raw = np.array([
            float(marginals[g.vertex(i)][int(b_modal[i])]) / n_samples
            if int(b_modal[i]) < len(marginals[g.vertex(i)]) else 0.0
            for i in range(n_verts)
        ])
        node_consistency[k] = (raw - chance) / (1.0 - chance)

    # ---- Joint block connectivity (model-internal mrs) ---- #
    # For each meaningful level k, aggregate the level-0 mrs upward using the
    # projected level-k assignments, remap to modal labels, and average.
    # The result is a single B×B matrix that directly reflects what the joint
    # SBM inferred — not a per-layer projection of original edge weights.
    def _aggregate_mrs_to_level(k, b_modal_k):
        B_modal = int(b_modal_k.max()) + 1
        accum   = np.zeros((B_modal, B_modal))

        for i in range(window_size):
            b0_iter  = b0_history[i]        # node → level-0 block label
            mrs0_i   = mrs0_history[i]      # (B0 x B0) joint edge count matrix
            b_k_iter = b_history[k][i]      # node → current level-k block label

            B0_cur = mrs0_i.shape[0]
            B0_nv  = int(b0_iter.max()) + 1

            # Map level-0 block label → current level-k block label
            # (uses node co-membership; first assignment wins per level-0 block)
            b0_to_bk = np.zeros(max(B0_cur, B0_nv), dtype=int)
            for node in range(n_verts):
                b0_to_bk[b0_iter[node]] = b_k_iter[node]

            # Map current level-k label → modal level-k label (majority vote)
            B_k_cur = int(b_k_iter.max()) + 1
            remap   = np.zeros(B_k_cur, dtype=int)
            for r in range(B_k_cur):
                nodes_r = np.where(b_k_iter == r)[0]
                if nodes_r.size > 0:
                    remap[r] = int(
                        np.bincount(b_modal_k[nodes_r],
                                    minlength=B_modal).argmax()
                    )

            # Aggregate mrs0_i into (B_modal x B_modal) modal label space
            for r0 in range(min(B0_cur, B0_nv)):
                r_modal = remap[b0_to_bk[r0]]
                for s0 in range(min(B0_cur, B0_nv)):
                    s_modal = remap[b0_to_bk[s0]]
                    accum[r_modal, s_modal] += mrs0_i[r0, s0]

        return accum / window_size

    block_connectivity = {}
    for k in meaningful_levels:
        log_msg(f'| UPDATE | aggregating mrs to level {k}')
        block_connectivity[k] = _aggregate_mrs_to_level(k, modal_assignments[k])
        log_msg(f'| UPDATE | finished aggregating mrs to level {k}')

    # _aggregate_mrs_to_level closes over b0_history/mrs0_history/b_history,
    # which creates a reference cycle (frame -> function -> cell -> frame) that
    # keeps those large per-sample buffers alive until the next cyclic GC pass.
    # Drop the closure and buffers now that they're no longer needed.
    del _aggregate_mrs_to_level, b0_history, mrs0_history, b_history

    # ---- Compile results ---- #
    _levels_final = state.get_levels()

    results = {
        'state':                 state,
        'entropy':               state.entropy(),
        'n_levels':              n_levels,
        'meaningful_levels':     meaningful_levels,
        'levels_n_blocks':       [lv.get_nonempty_B() for lv in _levels_final],
        'levels_entropy':        [lv.entropy()         for lv in _levels_final],
        'entropy_trajectory':    dS,
        'entropy_converged':     dS_converged,
        'convergence_iteration': conv_iter,
        'n_converged_samples':   window_size,
        'converged':             converged,
        'threshold':             threshold,
        'first_window_mean':     first_window_mean,
        'first_window_std':      first_window_std,
        'modal_assignments':     modal_assignments,
        'block_connectivity':    block_connectivity,
        'node_consistency':      node_consistency,
        'edge_mean':             edge_mean,
        'edge_var':              edge_var,
        'max_iter':              max_iter,
        'window_size':           window_size,
        'shift_factor':          shift_factor,
    }

    return results


def fit_nested_sbm_layered_multiflip(graph,
                                     max_iter=10000,
                                     window_size=500,
                                     shift_factor=0.5,
                                     behaviour_dist='normal',
                                     cooccurrence_dist='poisson',
                                     seed=42):
    """
    Identical to fit_nested_sbm_layered but uses multiflip_mcmc_sweep instead of
    mcmc_sweep. Multiflip proposes merge/split moves which can escape local minima
    that single-node proposals cannot, making it more effective when
    minimize_nested_blockmodel_dl has already converged to a local optimum.

    Parameters and return value are identical to fit_nested_sbm_layered.
    """

    gt.seed_rng(seed)
    g = graph.copy()

    state = gt.minimize_nested_blockmodel_dl(
        g,
        state_args=dict(
            base_type=gt.LayeredBlockState,
            state_args=dict(
                ec=g.ep.layer,
                recs=[g.ep.behaviour_weight, g.ep.cooccurrence_weight],
                rec_types=[_DIST_TO_REC[behaviour_dist], _DIST_TO_REC[cooccurrence_dist]],
                layers=True,
                deg_corr=True
            )
        )
    )

    n_verts = g.num_vertices()

    entropy_traj      = []
    converged         = False
    conv_iter         = max_iter
    first_window_mean = None
    first_window_std  = None
    threshold         = None

    log_msg('| UPDATE | starting multiflip MCMC change-point detection loop')
    with tqdm(total=max_iter, desc='MCMC-mf', unit='iter') as pbar:
        for i in range(max_iter):
            state.multiflip_mcmc_sweep(niter=1)
            entropy_traj.append(state.entropy())

            if i == window_size - 1:
                first_window_mean = sum(entropy_traj) / window_size
                first_window_std  = (
                    sum((x - first_window_mean) ** 2 for x in entropy_traj) / window_size
                ) ** 0.5
                threshold = first_window_mean - shift_factor * first_window_std
                pbar.set_postfix(ref=f'{first_window_mean:.1f}',
                                 thr=f'{threshold:.1f}')

            elif threshold is not None:
                w_mean = sum(entropy_traj[-window_size:]) / window_size
                if w_mean < threshold:
                    converged = True
                    conv_iter = i
                    pbar.set_postfix(DL=f'{entropy_traj[-1]:.1f}',
                                     w_mean=f'{w_mean:.1f}',
                                     status='SHIFT')
                    pbar.update(1)
                    break

            if i % 100 == 0:
                pbar.set_postfix(DL=f'{entropy_traj[-1]:.1f}')
            pbar.update(1)

    dS = np.array(entropy_traj)

    if converged:
        log_msg(f'| UPDATE | multiflip mean-shift change point at iteration {conv_iter} '
                f'| ref={first_window_mean:.2f}, std={first_window_std:.2f}, '
                f'threshold={threshold:.2f}, shift_factor={shift_factor}')
    else:
        log_msg(f'| WARNING | multiflip: no mean shift detected within {max_iter} iterations '
                f'| ref={first_window_mean:.2f}, threshold={threshold:.2f} '
                f'— using last {window_size} iterations. '
                f'Consider increasing max_iter or reducing shift_factor '
                f'(current: {shift_factor}).')

    _levels_init    = state.get_levels()
    n_levels        = len(_levels_init)
    _entropies_init = [lv.entropy() for lv in _levels_init]
    entropy_floor   = min(_entropies_init)
    meaningful_levels = [k for k in range(n_levels)
                         if _entropies_init[k] > entropy_floor]

    b_history    = {k: [] for k in meaningful_levels}
    b0_history   = []
    mrs0_history = []

    edge_M1 = np.zeros((n_verts, n_verts))
    edge_M2 = np.zeros((n_verts, n_verts))

    dS_converged = np.zeros(window_size)
    log_msg(f'| UPDATE | accumulating {window_size} samples from converged state (multiflip)')
    with tqdm(total=window_size, desc='Accum-mf', unit='iter') as pbar:
        for i in range(window_size):
            state.multiflip_mcmc_sweep(niter=1)
            dS_converged[i] = state.entropy()

            lv0         = state.get_levels()[0]
            b0_arr      = lv0.get_blocks().a.copy()
            mrs0_sparse = lv0.get_matrix()
            mrs0_arr    = mrs0_sparse.toarray().astype(float) \
                          if hasattr(mrs0_sparse, 'toarray') \
                          else np.array(mrs0_sparse, dtype=float)
            b0_history.append(b0_arr)
            mrs0_history.append(mrs0_arr)

            B0_size = mrs0_arr.shape[0]
            b_clip  = np.minimum(b0_arr, B0_size - 1).astype(int)
            vals    = mrs0_arr[np.ix_(b_clip, b_clip)]
            edge_M1 += vals
            edge_M2 += vals ** 2

            for k in meaningful_levels:
                proj  = state.project_partition(k, 0)
                b_arr = proj.a.copy() if hasattr(proj, 'a') else \
                        np.array([proj[v] for v in g.vertices()])
                b_history[k].append(b_arr)

            if i % 50 == 0:
                pbar.set_postfix(DL=f'{dS_converged[i]:.1f}')
            pbar.update(1)

    edge_mean = edge_M1 / window_size
    edge_var  = np.maximum(edge_M2 / window_size - edge_mean ** 2, 0.0)

    modal_assignments = {}
    node_consistency  = {}
    for k in meaningful_levels:
        pmode    = gt.PartitionModeState(b_history[k], converge=True)
        b_mode   = pmode.get_max(g)
        b_modal  = b_mode.a.copy() if hasattr(b_mode, 'a') else \
                   np.array([b_mode[v] for v in g.vertices()])
        modal_assignments[k] = b_modal

        n_blocks  = int(b_modal.max()) + 1
        chance    = 1.0 / n_blocks
        n_samples = len(b_history[k])
        marginals = pmode.get_marginal(g)
        raw = np.array([
            float(marginals[g.vertex(i)][int(b_modal[i])]) / n_samples
            if int(b_modal[i]) < len(marginals[g.vertex(i)]) else 0.0
            for i in range(n_verts)
        ])
        node_consistency[k] = (raw - chance) / (1.0 - chance)

    def _aggregate_mrs_to_level(k, b_modal_k):
        B_modal = int(b_modal_k.max()) + 1
        accum   = np.zeros((B_modal, B_modal))

        for i in range(window_size):
            b0_iter  = b0_history[i]
            mrs0_i   = mrs0_history[i]
            b_k_iter = b_history[k][i]

            B0_cur = mrs0_i.shape[0]
            B0_nv  = int(b0_iter.max()) + 1

            b0_to_bk = np.zeros(max(B0_cur, B0_nv), dtype=int)
            for node in range(n_verts):
                b0_to_bk[b0_iter[node]] = b_k_iter[node]

            B_k_cur = int(b_k_iter.max()) + 1
            remap   = np.zeros(B_k_cur, dtype=int)
            for r in range(B_k_cur):
                nodes_r = np.where(b_k_iter == r)[0]
                if nodes_r.size > 0:
                    remap[r] = int(
                        np.bincount(b_modal_k[nodes_r],
                                    minlength=B_modal).argmax()
                    )

            for r0 in range(min(B0_cur, B0_nv)):
                r_modal = remap[b0_to_bk[r0]]
                for s0 in range(min(B0_cur, B0_nv)):
                    s_modal = remap[b0_to_bk[s0]]
                    accum[r_modal, s_modal] += mrs0_i[r0, s0]

        return accum / window_size

    block_connectivity = {}
    for k in meaningful_levels:
        log_msg(f'| UPDATE | aggregating mrs to level {k} (multiflip)')
        block_connectivity[k] = _aggregate_mrs_to_level(k, modal_assignments[k])
        log_msg(f'| UPDATE | finished aggregating mrs to level {k} (multiflip)')

    # _aggregate_mrs_to_level closes over b0_history/mrs0_history/b_history,
    # which creates a reference cycle (frame -> function -> cell -> frame) that
    # keeps those large per-sample buffers alive until the next cyclic GC pass.
    # Drop the closure and buffers now that they're no longer needed.
    del _aggregate_mrs_to_level, b0_history, mrs0_history, b_history

    _levels_final = state.get_levels()

    results = {
        'state':                 state,
        'entropy':               state.entropy(),
        'n_levels':              n_levels,
        'meaningful_levels':     meaningful_levels,
        'levels_n_blocks':       [lv.get_nonempty_B() for lv in _levels_final],
        'levels_entropy':        [lv.entropy()         for lv in _levels_final],
        'entropy_trajectory':    dS,
        'entropy_converged':     dS_converged,
        'convergence_iteration': conv_iter,
        'n_converged_samples':   window_size,
        'converged':             converged,
        'threshold':             threshold,
        'first_window_mean':     first_window_mean,
        'first_window_std':      first_window_std,
        'modal_assignments':     modal_assignments,
        'block_connectivity':    block_connectivity,
        'node_consistency':      node_consistency,
        'edge_mean':             edge_mean,
        'edge_var':              edge_var,
        'max_iter':              max_iter,
        'window_size':           window_size,
        'shift_factor':          shift_factor,
    }

    return results



#########################
#     SBM FITTING       #
#########################
