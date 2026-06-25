#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                   STROKE SUB-SCORE PREDICTION                         #
#                                                                       #
# The following script runs weighted nested SBM-based community        #
# detection on the multi-layer graph object using LayeredBlockState.   #
#                                                                       #
# Edges are strictly separated into two layers (behaviour and          #
# co-occurrence), with edges appearing in both layers duplicated.      #
# The nested hierarchy is inferred using minimize_nested_blockmodel_dl #
# with LayeredBlockState as the base type, enabling proper Bayesian   #
# inference where both edge covariates (Gaussian and Poisson           #
# likelihoods) influence block detection at every hierarchy level.     #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/06/05.                                              #
#                                                                       #
#                                                                       #
#########################################################################


import numpy as np
import graph_tool.all as gt
from graph_tool import inference


# ---- Weighted nested SBM community detection with LayeredBlockState ---- #
def fit_nested_sbm_layered(graph,
                           mcmc_samples=100000,
                           burn_in=50000,
                           annealing_temps=(1, 10),
                           annealing_steps=100):
    """
    Fit nested layered stochastic block model to multi-layer weighted graph.

    Uses LayeredBlockState as base for NestedBlockState to handle separate
    weight likelihoods per layer during hierarchical inference.

    Parameters
    ----------
    graph : graph_tool.Graph
        Graph from create_multilayer_graph() with properties:
        - 'behaviour_weight' : layer 0 weights (real-valued)
        - 'cooccurrence_weight' : layer 1 weights (integer counts)
        - 'layer' : layer assignment (0 or 1)
    mcmc_samples : int, optional
        Posterior MCMC samples (default: 100000).
    burn_in : int, optional
        MCMC burn-in iterations (default: 50000).
    annealing_temps : tuple of float, optional
        (min_temp, max_temp) for simulated annealing (default: (1, 10)).
    annealing_steps : int, optional
        Number of annealing steps (default: 100).

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'state' : NestedBlockState object
        - 'entropy' : Final model entropy
        - 'n_levels' : Number of hierarchy levels
        - 'block_structure_level_0' : Level-0 block assignments
        - 'n_blocks_level_0' : Number of blocks at level 0
        - 'levels_n_blocks' : Block counts per level
        - 'levels_entropy' : Entropy per level
        - 'entropy_trajectory' : Entropy at each sample
        - 'mcmc_samples' : Samples generated
        - 'burn_in' : Burn-in used
    """

    g = graph.copy()

    # Initialize nested block state with LayeredBlockState base
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
    for temp in np.linspace(annealing_temps[1], annealing_temps[0], annealing_steps):
        state.mcmc_sweep(niter=10, beta=1.0 / temp)

    # Burn-in
    for _ in range(burn_in):
        state.mcmc_sweep(niter=1)

    # Posterior sampling with marginal accumulation
    n_verts = g.num_vertices()
    pv = None                                      # vertex marginals (block assignment distributions)
    edge_mean = np.zeros((n_verts, n_verts))       # Welford running mean of edge rates
    edge_M2   = np.zeros((n_verts, n_verts))       # Welford running sum of squared deviations

    dS = np.zeros(mcmc_samples)
    for i in range(mcmc_samples):
        state.mcmc_sweep(niter=1)
        dS[i] = state.entropy()

        # Vertex marginals: accumulate block assignment counts per node
        pv = state.collect_vertex_marginals(pv)

        # Edge rate matrix: implied connection rate between each node pair
        # under the current partition, using block-to-block edge counts
        level0 = state.get_levels()[0]
        b   = level0.get_blocks().a                        # block id per vertex
        mrs = level0.get_matrix()                          # sparse (B x B) edge count matrix
        nr  = np.bincount(b, minlength=int(b.max()) + 1)  # block sizes

        edge_rate = np.zeros((n_verts, n_verts))
        for e in g.edges():
            u, v = int(e.source()), int(e.target())
            r, s = b[u], b[v]
            denom = nr[r] * nr[s] if r != s else nr[r] * (nr[r] - 1) / 2
            rate = float(mrs[r, s]) / denom if denom > 0 else 0.0
            edge_rate[u, v] = edge_rate[v, u] = rate

        # Welford online update
        delta       = edge_rate - edge_mean
        edge_mean  += delta / (i + 1)
        edge_M2    += delta * (edge_rate - edge_mean)

    # Extract hierarchy
    levels = state.get_levels()
    block_structure_level_0 = state.project_partition(0, 0)
    
    if hasattr(block_structure_level_0, 'a'):
        block_structure_array = block_structure_level_0.a
    else:
        block_structure_array = np.array([block_structure_level_0[v] for v in state.g.vertices()])
    
    results = {
        'state': state,
        'entropy': state.entropy(),
        'n_levels': len(levels),
        'block_structure_level_0': block_structure_array,
        'n_blocks_level_0': block_structure_array.max() + 1,
        'levels_n_blocks': [level.get_nonempty_B() for level in levels],
        'levels_entropy': [level.entropy() for level in levels],
        'entropy_trajectory': dS,
        'vertex_marginals': pv,
        'edge_prob_mean': edge_mean,
        'edge_prob_var': edge_M2 / mcmc_samples,
        'mcmc_samples': mcmc_samples,
        'burn_in': burn_in
    }

    return results
