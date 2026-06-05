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
# detection on the multi-layer graph object.                           #
#                                                                       #
# Extends sbm_community_detection.py by replacing the flat BlockState  #
# with NestedBlockState. Edge covariates (recs) are passed through     #
# state_args to the base-level BlockState, so the Gaussian likelihood  #
# on behaviour_weight and Poisson likelihood on cooccurrence_weight    #
# are both preserved across all levels of the hierarchy.               #
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


# ---- Weighted nested SBM community detection ---- #
def fit_sbm_model(graph,
                  mcmc_samples=100000,
                  burn_in=50000,
                  annealing_temps=(1, 10),
                  annealing_steps=100):
    """
    Fit weighted nested stochastic block model to multi-layer graph.

    Extends the flat BlockState approach in sbm_community_detection.py by
    using NestedBlockState. Edge covariates are passed via state_args to the
    base-level BlockState so weighted likelihoods are preserved at every
    level of the hierarchy.

    Based on the framework from Cipolotti et al. (2023) BRAIN 146: 167-181.

    Parameters
    ----------
    graph : graph_tool.Graph
        Multi-layer graph from create_multilayer_graph() with edge properties:
        - 'behaviour_weight': BEHAVIOUR layer weights (real-valued)
        - 'cooccurrence_weight': co-occurrence layer weights (integer counts)
    mcmc_samples : int, optional
        Number of posterior MCMC samples (default: 100000).
    burn_in : int, optional
        MCMC burn-in iterations (default: 50000).
    annealing_temps : tuple of float, optional
        (min_temp, max_temp) for simulated annealing (default: (1, 10)).
    annealing_steps : int, optional
        Number of annealing temperature steps (default: 100).

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'entropy'           : Total model entropy (lower = better fit)
        - 'block_structure'   : Level-0 vertex partition array
        - 'n_blocks'          : Number of blocks at level 0
        - 'state'             : The NestedBlockState object for further analysis
        - 'n_levels'          : Number of levels in the hierarchy
        - 'levels_n_blocks'   : List of block counts per level
        - 'levels_entropy'    : List of entropy values per level
        - 'mcmc_samples'      : Number of samples used
        - 'burn_in'           : Burn-in used
        - 'entropy_trajectory': Entropy at each posterior sample

    Notes
    -----
    The NestedBlockState wraps a hierarchy of BlockState objects. The recs
    and rec_types are forwarded to the base-level (level 0) BlockState via
    state_args. Higher levels operate on the block membership graph and do
    not require separate rec specifications.

    rec_types:
    - 'real-normal'      : Gaussian likelihood for behaviour_weight.
      Prior: Normal-inverse-chi-squared with m0=0, k0=1, v0=1, nu0=3.
    - 'discrete-poisson' : Poisson likelihood for cooccurrence_weight.
      Prior: Gamma with alpha=1, beta=1.
    """

    # Copy graph to avoid modifying original
    g = graph.copy()

    # Extract edge properties (two layers)
    behaviour_weights = g.ep.behaviour_weight
    cooccurrence_weights = g.ep.cooccurrence_weight

    # Initialize nested block state with weighted edge covariates at base level.
    # state_args is forwarded to the underlying BlockState at level 0.
    state = gt.NestedBlockState(
        g,
        state_args=dict(
            recs=[behaviour_weights, cooccurrence_weights],
            rec_types=["real-normal", "discrete-poisson"],
            rec_params=[
                dict(m0=0., k0=1, v0=1., nu0=3),  # Normal-inverse-chi-squared
                dict(alpha=1, beta=1.)             # Gamma prior on Poisson rate
            ]
        )
    )

    # Simulated annealing: sweep from high temperature down to 1
    for temp in np.linspace(annealing_temps[1], annealing_temps[0], annealing_steps):
        state.mcmc_sweep(niter=10, beta=1.0 / temp)

    # Burn-in
    for i in range(burn_in):
        state.mcmc_sweep(niter=1)

    # Posterior sampling with entropy tracking
    dS = np.zeros(mcmc_samples)
    for i in range(mcmc_samples):
        state.mcmc_sweep(niter=1)
        dS[i] = state.entropy()

    # Extract level-wise information
    levels_n_blocks = [level.get_B() for level in state.levels]
    levels_entropy  = [level.entropy() for level in state.levels]

    # Level-0 partition (finest resolution)
    block_structure = state.levels[0].get_blocks()
    n_blocks        = state.levels[0].get_B()

    results = {
        'entropy':            state.entropy(),
        'block_structure':    block_structure.a,
        'n_blocks':           n_blocks,
        'state':              state,
        'n_levels':           len(state.levels),
        'levels_n_blocks':    levels_n_blocks,
        'levels_entropy':     levels_entropy,
        'mcmc_samples':       mcmc_samples,
        'burn_in':            burn_in,
        'entropy_trajectory': dS
    }

    return results
