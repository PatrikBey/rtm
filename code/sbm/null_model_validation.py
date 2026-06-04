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
# The following script generates null model distribution via behavioral #
# score permutation and compares real data against the null distribution #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/06/03.                                              #
#                                                                       #
#                                                                       #
#########################################################################


import numpy as np
from get_graphobject import create_multilayer_graph
from sbm_community_detection import fit_sbm_model


# ---- null model generation via permutation ---- #
def generate_null_models(adjacency_matrices, behavioral_values, node_names,
                         n_null_models=100,
                         edge_threshold=None,
                         mcmc_samples=100000,
                         burn_in=50000,
                         annealing_temps=(1, 10),
                         annealing_steps=100):
    """
    Generate empirical null distribution via behavioral score permutation.
    
    Based on the framework from Cipolotti et al. (2023) BRAIN 146: 167-181,
    this function creates N null models by randomly permuting behavioral scores
    when constructing multi-layer graphs. Comparing the real model entropy against
    the null distribution tests whether the structure of the behavioral layer
    is significantly different from random.
    
    The null hypothesis is that behavioral scores are unrelated to network
    structure. Each permutation breaks any relationship between behavior and
    lesion patterns while preserving the co-occurrence layer structure.
    
    Parameters
    ----------
    adjacency_matrices : ndarray, shape (n_patients, 166, 166)
        Binary adjacency matrices for each patient.
    behavioral_values : ndarray, shape (n_patients,)
        Behavioral scores corresponding to each patient.
    node_names : list or ndarray, shape (166,)
        Names/labels for the 166 brain regions/nodes.
    n_null_models : int, optional
        Number of null models to generate (default: 100).
    edge_threshold : float or None, optional
        Percentile threshold for edge filtering (passed to create_multilayer_graph).
    mcmc_samples : int, optional
        Number of MCMC samples for SBM fitting (default: 100000).
    burn_in : int, optional
        MCMC burn-in iterations (default: 50000).
    annealing_temps : tuple of float, optional
        (min_temp, max_temp) for simulated annealing (default: (1, 10)).
    annealing_steps : int, optional
        Number of annealing iterations.
    
    Returns
    -------
    null_distribution : dict
        Dictionary containing:
        - 'entropies': ndarray, shape (n_null_models,) - model entropy from each null model
        - 'mean_entropy': float - mean of null entropy distribution
        - 'std_entropy': float - standard deviation of null entropies
        - 'percentiles': dict - {5, 25, 50, 75, 95} percentiles of null distribution
        - 'null_models': list of n_null_models result dicts from fit_sbm_model()
        - 'n_null_models': int - number of null models generated
    
    Notes
    -----
    Each iteration:
    1. Randomly permute behavioral_values (fully randomized, no structure preserved)
    2. Create multi-layer graph with permuted behavioral scores
    3. Fit SBM model
    4. Extract entropy
    
    The resulting entropy distribution represents the expected entropy if
    behavioral scores were unrelated to network structure.
    """
    
    null_entropies = np.zeros(n_null_models)
    null_models = []
    
    for null_idx in range(n_null_models):
        # Fully randomize behavioral scores
        permuted_behavior = np.random.permutation(behavioral_values)
        
        # Create null graph with permuted behavioral values
        null_graph = create_multilayer_graph(
            adjacency_matrices, 
            permuted_behavior, 
            node_names,
            edge_threshold=edge_threshold
        )
        
        # Fit SBM model on null graph
        null_result = fit_sbm_model(
            null_graph,
            mcmc_samples=mcmc_samples,
            burn_in=burn_in,
            annealing_temps=annealing_temps,
            annealing_steps=annealing_steps
        )
        
        null_entropies[null_idx] = null_result['entropy']
        null_models.append(null_result)
    
    # Compute distribution statistics
    percentiles_dict = {
        5: np.percentile(null_entropies, 5),
        25: np.percentile(null_entropies, 25),
        50: np.percentile(null_entropies, 50),
        75: np.percentile(null_entropies, 75),
        95: np.percentile(null_entropies, 95)
    }
    
    null_distribution = {
        'entropies': null_entropies,
        'mean_entropy': np.mean(null_entropies),
        'std_entropy': np.std(null_entropies),
        'percentiles': percentiles_dict,
        'null_models': null_models,
        'n_null_models': n_null_models
    }
    
    return null_distribution


# ---- statistical comparison ---- #
def compare_with_null(real_entropy, null_distribution):
    """
    Compare real model entropy against null distribution.
    
    Tests the null hypothesis that the real data entropy is consistent with
    random behavioral scores. Low p-values indicate that the observed
    community structure is significantly better than expected by chance.
    
    Parameters
    ----------
    real_entropy : float
        Model entropy from real data (from fit_sbm_model output).
    null_distribution : dict
        Output dictionary from generate_null_models().
    
    Returns
    -------
    comparison : dict
        Dictionary containing:
        - 'z_score': float - Z-score of real entropy vs null mean
        - 'percentile_rank': float - Percentile rank (0-100) of real entropy in null distribution
        - 'p_value': float - Proportion of null models with lower entropy than real
        - 'is_significant': bool - True if p_value < 0.05
        - 'n_below': int - Number of null models with lower entropy
        - 'n_null_models': int - Total null models
    
    Notes
    -----
    Interpretation:
    - Lower entropy = better fit of community structure
    - If real_entropy is in the lower tail of null distribution → community structure
      is more meaningful than expected by random association
    - p_value = P(null_entropy < real_entropy) = proportion of null models
      outperforming the real model
    - Small p-values (< 0.05) suggest behavioral layer structure is significant
    """
    
    null_entropies = null_distribution['entropies']
    null_mean = null_distribution['mean_entropy']
    null_std = null_distribution['std_entropy']
    
    # Z-score
    z_score = (real_entropy - null_mean) / null_std if null_std > 0 else 0.0
    
    # Percentile rank: what percentile is real entropy at?
    percentile_rank = (np.sum(null_entropies <= real_entropy) / len(null_entropies)) * 100.0
    
    # P-value: proportion of null models better than (lower entropy than) real
    n_below = np.sum(null_entropies < real_entropy)
    p_value = n_below / len(null_entropies)
    
    # Significance test
    is_significant = p_value < 0.05
    
    comparison = {
        'z_score': z_score,
        'percentile_rank': percentile_rank,
        'p_value': p_value,
        'is_significant': is_significant,
        'n_below': n_below,
        'n_null_models': len(null_entropies),
        'real_entropy': real_entropy,
        'null_mean': null_mean,
        'null_std': null_std
    }
    
    return comparison
