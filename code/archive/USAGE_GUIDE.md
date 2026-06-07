#########################################################################
#                    INTERACTIVE USAGE GUIDE                            #
#                                                                       #
#          SBM-Based Lesion-Behavior Network Analysis Framework         #
#                                                                       #
#              Implementing: Cipolotti et al. (2023) BRAIN               #
#                     146: 167-181, doi: 10.1093/brain/awac304         #
#                                                                       #
#########################################################################


QUICK START - INTERACTIVE PYTHON SHELL
=======================================

Step 1: Load your data
-----------------------
import numpy as np
from get_graphobject import create_multilayer_graph
from sbm_community_detection import fit_sbm_model
from null_model_validation import generate_null_models, compare_with_null

# Load your data (example structure)
adj_matrices = np.load('adjacency_matrices.npy')  # shape: (n_patients, 166, 166)
behavioral_scores = np.load('behavioral_scores.npy')  # shape: (n_patients,)
with open('node_names.txt', 'r') as f:
    node_names = [line.strip() for line in f]  # 166 node labels


Step 2: Create multi-layer graph
--------------------------------
# Option A: No edge filtering (keep all edges)
graph = create_multilayer_graph(adj_matrices, behavioral_scores, node_names)

# Option B: With edge filtering (keep top 50% connected nodes)
graph = create_multilayer_graph(adj_matrices, behavioral_scores, node_names, 
                                edge_threshold=50)

print(f"Graph created: {graph.num_vertices()} nodes, {graph.num_edges()} edges")


Step 3: Fit SBM model on real data
----------------------------------
# Run with default parameters from paper
real_results = fit_sbm_model(
    graph,
    mcmc_samples=100000,  # 100k posterior samples (paper default)
    burn_in=50000,        # 50k burn-in (paper default)
    annealing_temps=(1, 10),  # Temperature range (paper default)
    annealing_steps=100
)

print(f"Real model entropy: {real_results['entropy']:.2f}")
print(f"Communities found: {real_results['n_blocks']}")


Step 4: Generate null model distribution
-----------------------------------------
# Generate 100 null models with permuted behavioral scores
null_dist = generate_null_models(
    adj_matrices, 
    behavioral_scores, 
    node_names,
    n_null_models=100,  # Generate 100 null models
    edge_threshold=50,  # Same threshold as real data
    mcmc_samples=100000,
    burn_in=50000,
    annealing_temps=(1, 10),
    annealing_steps=100
)

print(f"Null distribution mean entropy: {null_dist['mean_entropy']:.2f}")
print(f"Null distribution std entropy: {null_dist['std_entropy']:.2f}")
print(f"Null percentiles: {null_dist['percentiles']}")


Step 5: Compare real vs null
----------------------------
comparison = compare_with_null(real_results['entropy'], null_dist)

print(f"\n=== STATISTICAL COMPARISON ===")
print(f"Real entropy: {comparison['real_entropy']:.2f}")
print(f"Null mean: {comparison['null_mean']:.2f}")
print(f"Null std: {comparison['null_std']:.2f}")
print(f"Z-score: {comparison['z_score']:.3f}")
print(f"Percentile rank: {comparison['percentile_rank']:.1f}%")
print(f"P-value: {comparison['p_value']:.4f}")
print(f"Significant (p<0.05): {comparison['is_significant']}")


DETAILED PARAMETER GUIDE
========================

create_multilayer_graph() parameters:
-------------------------------------
- adjacency_matrices (required): Binary matrices (n_patients, 166, 166)
- behavioral_values (required): Scores per patient (n_patients,)
- node_names (required): Labels for 166 nodes
- edge_threshold (optional): Keep top X% connected nodes
    * None: keep all edges (default)
    * 50: keep top 50% connected nodes
    * 75: keep top 25% most connected nodes
  
  Output layers:
  - Layer 1 (BEHAVIOUR): adjacency[i] × behavioral_value[i] for each patient
  - Layer 2 (Co-occurrence): sum of all binary adjacencies


fit_sbm_model() parameters (defaults from Cipolotti et al. 2023):
-----------------------------------------------------------------
- graph (required): Output from create_multilayer_graph()
- mcmc_samples: Posterior samples (default: 100000)
- burn_in: MCMC burn-in (default: 50000)
- annealing_temps: (min_temp, max_temp) tuple (default: (1, 10))
- annealing_steps: Iterations during annealing (default: 100)

  Output:
  - 'entropy': Model entropy (lower = better fit)
  - 'block_structure': Array of block assignments per node
  - 'n_blocks': Number of identified communities
  - 'state': BlockStateNested object for advanced analysis
  - 'entropy_trajectory': Entropy trace over MCMC samples


generate_null_models() parameters:
-----------------------------------
- adjacency_matrices, behavioral_values, node_names (required)
- n_null_models: Number of permutations (default: 100)
- edge_threshold: Same as create_multilayer_graph() (optional)
- mcmc_samples, burn_in, annealing_temps, annealing_steps: Same as fit_sbm_model()

  Each iteration:
  1. Permutes behavioral_values randomly
  2. Reconstructs multi-layer graph with shuffled behavior
  3. Fits SBM model
  4. Extracts entropy

  Output:
  - 'entropies': Array of N entropy values
  - 'mean_entropy': Mean of null distribution
  - 'std_entropy': Standard deviation
  - 'percentiles': Dict with {5, 25, 50, 75, 95} percentiles
  - 'null_models': Full results from each null SBM fit


compare_with_null() parameters:
-------------------------------
- real_entropy (float): From real_results['entropy']
- null_distribution (dict): From generate_null_models()

  Output:
  - 'z_score': Standard deviations from null mean
  - 'percentile_rank': Where real falls in null distribution (0-100%)
  - 'p_value': Proportion of nulls better than real (lower entropy)
  - 'is_significant': Boolean, True if p_value < 0.05


INTERPRETATION GUIDE
====================

Model Entropy:
- Lower entropy = better fit of community structure
- Real entropy << null mean: behavioral layer has meaningful structure
- Real entropy ≈ null mean: behavioral layer ≈ random

Statistical Significance (from compare_with_null):
- p_value < 0.05: Community structure significantly differs from random
- percentile_rank < 5: Real entropy in lower 5% of null distribution
- z_score < -2: Real entropy > 2 SD below null mean (very significant)

Community Structure (from real_results):
- n_blocks: Number of distinct communities identified
- block_structure: Which nodes belong to which community
  * Extract with: real_results['block_structure']
  * Values from 0 to (n_blocks - 1)

Back-Projection:
- Extract per-node community assignments:
  blocks = real_results['block_structure']
  for node_idx, block_id in enumerate(blocks):
      print(f"Node {node_names[node_idx]} → Community {block_id}")


ADVANCED: Accessing BlockState object
======================================

For detailed analysis, the BlockState object is available:
  state = real_results['state']
  
Some useful methods:
  - state.entropy(): Get entropy value
  - state.get_blocks(): Get community assignments (vertex partition)
  - state.get_bg(): Get background graph structure
  - state.mcmc_sweep(): Run additional MCMC iterations


EXPECTED WORKFLOW
=================

1. Load data and explore shapes:
   print(adj_matrices.shape, behavioral_scores.shape, len(node_names))

2. Create graph:
   graph = create_multilayer_graph(...)

3. Fit real model (takes time, depends on n_patients and edge count):
   real_results = fit_sbm_model(graph)

4. Generate null models (parallel step, or sequential):
   # Can run in parallel by batch if needed
   null_dist = generate_null_models(...)

5. Compare results:
   comparison = compare_with_null(real_results['entropy'], null_dist)

6. Examine community structure:
   blocks = real_results['block_structure']
   for i, block in enumerate(np.unique(blocks)):
       nodes_in_block = np.where(blocks == block)[0]
       print(f"Block {block}: {[node_names[n] for n in nodes_in_block]}")

7. If significant, visualize or export results:
   - Export block assignments
   - Map to anatomical regions
   - Compare with literature findings


TROUBLESHOOTING
===============

Graph has no edges:
  - Check behavioral_values contains non-zero values
  - Check adjacency_matrices are binary (0 or 1)
  - If using edge_threshold, may be filtering out all nodes

SBM takes very long:
  - Reduce mcmc_samples (e.g., 50000 instead of 100000)
  - Reduce annealing_steps (e.g., 50 instead of 100)
  - Apply stricter edge_threshold (e.g., 75 instead of 50)

High entropy values:
  - Normal if many nodes/edges
  - Compare relative to null distribution, not absolute values
  - More important: is real entropy lower than null mean?

#########################################################################
