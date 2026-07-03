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
from utils import log_msg



#########################
#   BUILD GRAPH OBJECT  #
#########################

def create_multilayer_graph(adjacency_matrices, behavioral_values, node_names,
                            edge_threshold=50, th_apply='split'):
    """
    Create a redundant two-layer graph for weighted nested SBM inference.

    Each node pair (i, j) appears in both layers with complementary weights:
    - Layer 0: behaviour_weight active, cooccurrence_weight = 0
    - Layer 1: cooccurrence_weight active, behaviour_weight = 0

    Parameters
    ----------
    adjacency_matrices : ndarray, shape (n_patients, n_nodes, n_nodes)
    behavioral_values  : ndarray, shape (n_patients,)
    node_names         : list, length n_nodes
    edge_threshold     : float, percentile threshold for edge filtering (default 50)
    th_apply           : 'split' applies threshold per layer; else applies jointly

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
    cooccurrence_binary = np.sum(adjacency_matrices, axis=0)

    if th_apply == 'split':
        beh_nz = behaviour_weighted[behaviour_weighted > 0]
        occ_nz = cooccurrence_binary[cooccurrence_binary > 0]
        beh_thresh = np.percentile(beh_nz, edge_threshold) if beh_nz.size > 0 else 0
        occ_thresh = np.percentile(occ_nz, edge_threshold) if occ_nz.size > 0 else 0
        behaviour_weighted[behaviour_weighted < beh_thresh] = 0
        cooccurrence_binary[cooccurrence_binary < occ_thresh] = 0
        threshold_value = (beh_thresh, occ_thresh)
    else:
        combined_nz = np.concatenate([
            behaviour_weighted[behaviour_weighted > 0],
            cooccurrence_binary[cooccurrence_binary > 0]
        ])
        threshold_value = np.percentile(combined_nz, edge_threshold)
        behaviour_weighted[behaviour_weighted < threshold_value] = 0
        cooccurrence_binary[cooccurrence_binary < threshold_value] = 0

    for layer_mat in [behaviour_weighted, cooccurrence_binary]:
        mask = layer_mat > 0
        if mask.any():
            mu, sigma = layer_mat[mask].mean(), layer_mat[mask].std()
            if sigma > 0:
                layer_mat[mask] = (layer_mat[mask] - mu) / sigma

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
            has_beh = behaviour_weighted[i, j] > 0
            has_occ = cooccurrence_binary[i, j] > 0
            if has_beh or has_occ:
                beh_val = float(behaviour_weighted[i, j]) if has_beh else 0.0
                occ_val = float(cooccurrence_binary[i, j]) if has_occ else 0.0

                e0 = g.add_edge(g.vertex(i), g.vertex(j))
                behaviour_weight_prop[e0]    = beh_val
                cooccurrence_weight_prop[e0] = 0.0
                layer_prop[e0]               = 0

                e1 = g.add_edge(g.vertex(i), g.vertex(j))
                behaviour_weight_prop[e1]    = 0.0
                cooccurrence_weight_prop[e1] = occ_val
                layer_prop[e1]               = 1

    g.ep.behaviour_weight    = behaviour_weight_prop
    g.ep.cooccurrence_weight = cooccurrence_weight_prop
    g.ep.layer               = layer_prop

    g.gp.n_patients              = g.new_graph_property("int",    n_patients)
    g.gp.edge_threshold_applied  = g.new_graph_property("string", str(threshold_value))

    return g


#########################
#     SBM FITTING       #
#########################

def fit_nested_sbm_layered(graph,
                           mcmc_samples=100000,
                           burn_in=50000,
                           annealing_temps=(1, 10),
                           annealing_steps=100,
                           convergence_fraction=0.05):
    """
    Fit nested layered stochastic block model to multi-layer weighted graph.

    Two-phase sampling strategy
    ---------------------------
    Phase 1 (mcmc_samples sweeps): records entropy only. Used to locate t_star,
    the first iteration where remaining entropy reduction <=
    convergence_fraction * total_reduction. No partition accumulation occurs here.

    Phase 2 (n_converged = mcmc_samples - t_star sweeps from warm state):
    At each converged sweep the function records:
      - projected node-level block assignments at each meaningful level
        (for PartitionModeState label-switching correction)
      - the level-0 mrs matrix (model-internal block-to-block edge counts,
        joint across both graph layers)

    After Phase 2:
      - PartitionModeState finds the modal (most probable) partition at each
        meaningful level, resolving label switching.
      - The level-0 mrs matrices are aggregated upward to each level using the
        projected assignments, remapped to modal labels, and averaged across
        converged samples to yield block_connectivity: a single joint
        block-to-block connectivity matrix per level that reflects the model's
        internal representation — not a projection onto individual graph layers.

    Parameters
    ----------
    graph               : graph_tool.Graph from create_multilayer_graph()
    mcmc_samples        : int   — Phase 1 entropy-tracking sweeps (default 100000)
    burn_in             : int   — burn-in iterations before Phase 1 (default 50000)
    annealing_temps     : tuple — (min_temp, max_temp) for annealing (default (1,10))
    annealing_steps     : int   — number of annealing steps (default 100)
    convergence_fraction: float — fraction of total entropy reduction defining
                                  the convergence threshold (default 0.05)

    Returns
    -------
    results : dict
        'state'                 : final NestedBlockState
        'entropy'               : final model entropy (description length)
        'n_levels'              : total hierarchy levels
        'meaningful_levels'     : list of level indices with > 1 block
        'levels_n_blocks'       : non-empty block count per level
        'levels_entropy'        : entropy per level
        'entropy_trajectory'    : entropy at each Phase 1 iteration
        'entropy_converged'     : entropy at each Phase 2 iteration
        'convergence_iteration' : t_star
        'n_converged_samples'   : number of Phase 2 samples
        'convergence_fraction'  : value used
        'modal_assignments'     : {level: (n_nodes,) int array}
                                  Most probable block assignment per node at each
                                  meaningful level, recovered via PartitionModeState.
                                  Encodes the joint community structure derived from
                                  both graph layers simultaneously.
        'block_connectivity'    : {level: (B x B) float ndarray}
                                  Posterior mean of the model-internal block-to-block
                                  edge count matrix (mrs), aggregated to each level
                                  and aligned to modal labels. A single joint matrix
                                  — not split by graph layer.
        'mcmc_samples'          : Phase 1 sample count
        'burn_in'               : burn-in used
    """

    g = graph.copy()

    # ---- Initialise nested block state ---- #
    state = gt.minimize_nested_blockmodel_dl(
        g,
        state_args=dict(
            base_type=gt.LayeredBlockState,
            state_args=dict(
                ec=g.ep.layer,
                recs=[g.ep.behaviour_weight, g.ep.cooccurrence_weight],
                rec_types=["real-normal", "discrete-poisson"],
                layers=True,
                deg_corr=True
            )
        )
    )

    # Annealing
    log_msg('| UPDATE | starting annealing')
    for temp in np.linspace(annealing_temps[1], annealing_temps[0], annealing_steps):
        state.mcmc_sweep(niter=10, beta=1.0 / temp)
    log_msg('| UPDATE | annealing complete')

    # Burn-in
    log_msg('| UPDATE | starting burn-in')
    for _ in range(burn_in):
        state.mcmc_sweep(niter=1)
    log_msg('| UPDATE | burn-in complete')

    n_verts = g.num_vertices()

    # ---- Phase 1: entropy tracking only ---- #
    dS = np.zeros(mcmc_samples)
    log_msg('| UPDATE | starting phase 1 mcmc')
    _step1 = max(1, mcmc_samples // 10)
    for i in range(mcmc_samples):
        state.mcmc_sweep(niter=1)
        dS[i] = state.entropy()
        if (i + 1) % _step1 == 0:
            log_msg(f'| UPDATE | finished {(i + 1) * 100 // mcmc_samples}% of mcmc iterations')

    # ---- Convergence threshold ---- #
    total_reduction = dS[0] - dS.min()
    if total_reduction > 0:
        conv_threshold = dS.min() + convergence_fraction * total_reduction
        crossing = np.where(dS <= conv_threshold)[0]
        t_star = int(crossing[0]) if crossing.size > 0 else mcmc_samples - 1
    else:
        t_star = 0
    n_converged = mcmc_samples - t_star

    # ---- Meaningful levels ---- #
    # get_nonempty_B() is unreliable for layered nested states (can return 1
    # even for levels with genuine block structure). Use entropy instead:
    # the degenerate single-block levels all collapse to the same floor value,
    # so any level with entropy strictly above that floor is meaningful.
    _levels_init    = state.get_levels()
    n_levels        = len(_levels_init)
    _entropies_init = [lv.entropy() for lv in _levels_init]
    entropy_floor   = min(_entropies_init)
    meaningful_levels = [k for k in range(n_levels)
                         if _entropies_init[k] > entropy_floor]

    # ---- Phase 2: converged accumulation ---- #
    # Collect per-iteration:
    #   b_history[k]  — projected node-level assignment at level k
    #   b0_history    — level-0 block assignments (for mrs aggregation)
    #   mrs0_history  — level-0 mrs matrix (joint, both layers, model-internal)
    #   edge_M1/M2    — online accumulators for edge variance:
    #                   for each node pair (i,j), track mrs[b_i, b_j] to compute
    #                   mean and variance across converged iterations. This is
    #                   "edge existence in the final model partition, informed by
    #                   both layers" — low variance = consistently identified edge.
    b_history    = {k: [] for k in meaningful_levels}
    b0_history   = []
    mrs0_history = []

    edge_M1 = np.zeros((n_verts, n_verts))   # running sum
    edge_M2 = np.zeros((n_verts, n_verts))   # running sum of squares

    dS_converged = np.zeros(n_converged)
    log_msg('| UPDATE | starting phase 2 accumulation')
    _step2 = max(1, n_converged // 10)
    for i in range(n_converged):
        state.mcmc_sweep(niter=1)
        dS_converged[i] = state.entropy()

        # Level-0 state: block assignments and joint mrs
        lv0          = state.get_levels()[0]
        b0_arr       = lv0.get_blocks().a.copy()
        mrs0_sparse  = lv0.get_matrix()
        mrs0_arr     = mrs0_sparse.toarray().astype(float) \
                       if hasattr(mrs0_sparse, 'toarray') \
                       else np.array(mrs0_sparse, dtype=float)
        b0_history.append(b0_arr)
        mrs0_history.append(mrs0_arr)

        # Accumulate mrs[b_i, b_j] for every node pair (vectorised)
        B0_size = mrs0_arr.shape[0]
        b_clip  = np.minimum(b0_arr, B0_size - 1).astype(int)
        vals    = mrs0_arr[np.ix_(b_clip, b_clip)]
        edge_M1 += vals
        edge_M2 += vals ** 2

        # Projected node-level assignments for each meaningful level
        for k in meaningful_levels:
            proj  = state.project_partition(k, 0)
            b_arr = proj.a.copy() if hasattr(proj, 'a') else \
                    np.array([proj[v] for v in g.vertices()])
            b_history[k].append(b_arr)

        if (i + 1) % _step2 == 0:
            log_msg(f'| UPDATE | finished {(i + 1) * 100 // n_converged}% of mcmc iterations')

    edge_mean = edge_M1 / n_converged
    edge_var  = np.maximum(edge_M2 / n_converged - edge_mean ** 2, 0.0)

    # ---- Modal partitions + node assignment consistency ---- #
    # PartitionModeState resolves label switching and gives:
    #   get_max(g)       — modal (most probable) block per node
    #   get_marginals(g) — posterior probability of each block per node,
    #                      in the same canonical labelling as get_max().
    # node_consistency[k][i] = P(node i in its modal block) across converged
    # iterations, properly accounting for label switching.
    # Range [0, 1]: 1 = always placed in that block, 0 = never.
    # Used as spoke alpha: bright = consistent member, transparent = ambiguous.
    modal_assignments = {}
    node_consistency  = {}
    for k in meaningful_levels:
        pmode    = gt.PartitionModeState(b_history[k], converge=True)
        b_mode   = pmode.get_max(g)
        b_modal  = b_mode.a.copy() if hasattr(b_mode, 'a') else \
                   np.array([b_mode[v] for v in g.vertices()])
        modal_assignments[k] = b_modal

        marginals = pmode.get_marginal(g)
        node_consistency[k] = np.array([
            float(marginals[g.vertex(i)][int(b_modal[i])])
            if int(b_modal[i]) < len(marginals[g.vertex(i)]) else 0.0
            for i in range(n_verts)
        ])

    # ---- Joint block connectivity (model-internal mrs) ---- #
    # For each meaningful level k, aggregate the level-0 mrs upward using the
    # projected level-k assignments, remap to modal labels, and average.
    # The result is a single B×B matrix that directly reflects what the joint
    # SBM inferred — not a per-layer projection of original edge weights.
    def _aggregate_mrs_to_level(k, b_modal_k):
        B_modal = int(b_modal_k.max()) + 1
        accum   = np.zeros((B_modal, B_modal))

        for i in range(n_converged):
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

        return accum / n_converged

    block_connectivity = {}
    for k in meaningful_levels:
        log_msg(f'| UPDATE | aggregating mrs to level {k}')
        block_connectivity[k] = _aggregate_mrs_to_level(k, modal_assignments[k])
        log_msg(f'| UPDATE | finished aggregating mrs to level {k}')

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
        'convergence_iteration': t_star,
        'n_converged_samples':   n_converged,
        'convergence_fraction':  convergence_fraction,
        'modal_assignments':     modal_assignments,    # {level: (n_nodes,) array}
        'block_connectivity':    block_connectivity,   # {level: (B x B) array}
        'node_consistency':      node_consistency,     # {level: (n_nodes,) array in [0,1]} — assignment consistency
        'edge_mean':             edge_mean,            # (n_nodes x n_nodes) posterior mean mrs[b_i,b_j]
        'edge_var':              edge_var,             # (n_nodes x n_nodes) posterior variance — low = stable
        'mcmc_samples':          mcmc_samples,
        'burn_in':               burn_in,
    }

    return results



#########################
#     SBM FITTING       #
#########################
