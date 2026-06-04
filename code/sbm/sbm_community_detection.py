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
# The following script runs SBM-based community detection on the       #
# multi-layer graph object                                             #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/06/03.                                              #
#                                                                       #
#                                                                       #
#########################################################################


import numpy as np
import graph_tool.all as gt
from graph_tool import inference


# ---- SBM community detection on real data ---- #
def fit_sbm_model(graph, 
                  mcmc_samples=100000, 
                  burn_in=50000,
                  annealing_temps=(1, 10),
                  annealing_steps=100):
    """
    Fit hierarchical weighted stochastic block model to multi-layer graph.
    
    Based on the framework from Cipolotti et al. (2023) BRAIN 146: 167-181,
    this function fits a Bayesian non-parametric hierarchical stochastic block
    model to identify community structure in the multi-layer graph, disentangling
    behavioral effects from pathological co-occurrence effects.
    
    The model:
    - Treats BEHAVIOUR layer weights as Gaussian distribution
    - Treats co-occurrence layer weights as Poisson distribution
    - Uses simulated annealing to optimize community structure
    - Compares layers to identify blocks driven by behavior vs. lesion patterns
    
    Parameters
    ----------
    graph : graph_tool.Graph
        Multi-layer graph from create_multilayer_graph() with edge properties:
        - 'behaviour_weight': BEHAVIOUR layer weights
        - 'cooccurrence_weight': co-occurrence layer weights
    mcmc_samples : int, optional
        Number of posterior MCMC samples (default: 100000, from paper).
    burn_in : int, optional
        MCMC burn-in iterations (default: 50000, from paper).
    annealing_temps : tuple of float, optional
        (min_temp, max_temp) for simulated annealing (default: (1, 10), from paper).
    annealing_steps : int, optional
        Number of annealing iterations for optimization.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'entropy': Model entropy value (lower is better fit)
        - 'block_structure': Vertex partition array, block assignment per node
        - 'n_blocks': Number of blocks in optimal partition
        - 'state': The BlockStateNested object for further analysis
        - 'mcmc_samples': Number of samples used
        - 'burn_in': Burn-in used
    
    Notes
    -----
    The function uses graph_tool's hierarchical stochastic block model with:
    - Simulated annealing to find optimal partition
    - Nested block structure (hierarchical)
    - Weighted edge model with mixed distributions per layer
    
    Lower entropy indicates better fit of community structure.
    """
    
    # Copy graph to avoid modifying original
    g = graph.copy()
    
    # Extract edge properties (two layers)
    behaviour_weights = g.ep.behaviour_weight
    cooccurrence_weights = g.ep.cooccurrence_weight
    
    # Initialize nested block state with both layers as edge weights
    # We create two separate edge property maps for the two weight types
    state = inference.BlockStateNested(
        g,
        state_args=dict(
            recs=[behaviour_weights, cooccurrence_weights],
            rec_types=["real-normal", "poisson"],  # BEHAVIOUR: Gaussian, co-occurrence: Poisson
            rec_params=[
                dict(mu=0., sigma=1.),
                dict(mu=0., sigma=1.)
            ]
        ),
        base_type=inference.BlockState
    )
    
    # Simulated annealing optimization
    for temp in np.linspace(annealing_temps[1], annealing_temps[0], annealing_steps):
        # MCMC sampling at each temperature
        state.mcmc_sweep(
            niter=10,  # iterations at this temperature
            beta=1.0/temp,  # inverse temperature
            force_move=False,
            allow_vacuous=False
        )
    
    # Final MCMC sampling with recorded entropy
    dS = np.zeros(mcmc_samples)
    for i in range(burn_in):
        state.mcmc_sweep(niter=1, force_move=False, allow_vacuous=False)
    
    for i in range(mcmc_samples):
        state.mcmc_sweep(niter=1, force_move=False, allow_vacuous=False)
        dS[i] = state.entropy()
    
    # Extract results
    entropy = state.entropy()
    block_structure = state.get_blocks()
    n_blocks = block_structure.a.max() + 1
    
    results = {
        'entropy': entropy,
        'block_structure': block_structure.a,
        'n_blocks': n_blocks,
        'state': state,
        'mcmc_samples': mcmc_samples,
        'burn_in': burn_in,
        'entropy_trajectory': dS
    }
    
    return results
