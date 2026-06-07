===============================================================================
                    IMPLEMENTATION SUMMARY
           SBM-Based Lesion-Behavior Network Analysis Framework
                 Based on: Cipolotti et al. (2023) BRAIN 146
===============================================================================

OVERVIEW
========
Three complementary Python scripts implementing a Bayesian hierarchical 
stochastic block model (SBM) framework for lesion-deficit mapping with 
behavioral validation.

Framework separates behavioral effects from pathological co-occurrence 
effects through a two-layer network model.

===============================================================================

SCRIPT 1: get_graphobject.py
============================
Creates multi-layer graph object from patient lesion and behavioral data.

Function: create_multilayer_graph()
Inputs:
  - adjacency_matrices: (n_patients, 166, 166) binary matrices
  - behavioral_values: (n_patients,) behavioral scores  
  - node_names: (166,) node labels from txt file
  - edge_threshold: optional, keep top X% connected nodes

Process:
  Layer 1 (BEHAVIOUR):
    For each patient i: weighted_adj[i] = adjacency[i] × behavioral_score[i]
    Aggregates all weighted adjacencies across patients
    
  Layer 2 (Co-occurrence):
    Sum of all binary adjacency matrices
    Shows how many patients had each edge lesioned together

Output:
  - graph_tool.Graph object with 2 layers
  - Edge properties: 'behaviour_weight', 'cooccurrence_weight'
  - 166 nodes with labels
  - Metadata: n_patients, edge_threshold applied

Example:
  graph = create_multilayer_graph(adj_matrices, behavioral_scores, 
                                  node_names, edge_threshold=50)

===============================================================================

SCRIPT 2: sbm_community_detection.py
===================================
Fits hierarchical weighted stochastic block model to identify community 
structure in real data graph.

Function: fit_sbm_model()
Inputs:
  - graph: from create_multilayer_graph()
  - mcmc_samples: 100000 (paper default)
  - burn_in: 50000 (paper default)
  - annealing_temps: (1, 10) (paper default)
  - annealing_steps: 100

Process:
  1. Initialize BlockStateNested with two layers
     - BEHAVIOUR layer: Gaussian distribution
     - Co-occurrence layer: Poisson distribution
  2. Simulated annealing optimization
  3. Final MCMC sampling and entropy calculation
  4. Extract community block structure

Output:
  Dict with:
  - 'entropy': model entropy value (lower = better fit)
  - 'block_structure': community assignments per node
  - 'n_blocks': number of communities identified
  - 'state': BlockStateNested object for further analysis
  - 'entropy_trajectory': entropy over MCMC samples

Example:
  real_results = fit_sbm_model(graph, mcmc_samples=100000)
  print(f"Entropy: {real_results['entropy']}, Blocks: {real_results['n_blocks']}")

===============================================================================

SCRIPT 3: null_model_validation.py
==================================
Generates empirical null distribution via behavioral score permutation.
Compares real model against null to test significance.

Functions: generate_null_models() and compare_with_null()

generate_null_models():
Inputs:
  - adjacency_matrices, behavioral_values, node_names (same as Script 1)
  - n_null_models: 100 (default)
  - edge_threshold, mcmc_samples, burn_in, annealing_temps, annealing_steps

Process (for each of N iterations):
  1. Fully randomize behavioral_values: permuted_behavior = np.random.permutation()
  2. Create null graph with permuted scores
  3. Fit SBM model
  4. Extract entropy
  5. Store in distribution

Output:
  Dict with:
  - 'entropies': array of N model entropies
  - 'mean_entropy': mean of null distribution
  - 'std_entropy': standard deviation
  - 'percentiles': {5, 25, 50, 75, 95} percentiles
  - 'null_models': full results from each fit

Example:
  null_dist = generate_null_models(adj_matrices, behavioral_scores, 
                                   node_names, n_null_models=100)

compare_with_null():
Inputs:
  - real_entropy: from real_results['entropy']
  - null_distribution: from generate_null_models()

Output:
  Dict with:
  - 'z_score': (real - null_mean) / null_std
  - 'percentile_rank': where real entropy falls (0-100%)
  - 'p_value': proportion of nulls with lower entropy
  - 'is_significant': boolean, True if p_value < 0.05

Example:
  comparison = compare_with_null(real_results['entropy'], null_dist)
  print(f"P-value: {comparison['p_value']:.4f}")
  print(f"Significant: {comparison['is_significant']}")

===============================================================================

DATA FLOW
=========

Patient Data (numpy arrays + txt file)
├─ adjacency_matrices (n_patients, 166, 166)
├─ behavioral_values (n_patients,)
└─ node_names.txt → 166 labels

            ↓ Script 1: create_multilayer_graph()

Multi-Layer Graph Object (graph_tool.Graph)
├─ Layer 1: BEHAVIOUR-weighted edges
├─ Layer 2: Binary co-occurrence edges
└─ 166 nodes with properties

            ↓ Script 2: fit_sbm_model()
            
Real Model Results
├─ entropy: X.XX (lower = better)
├─ block_structure: community assignments
├─ n_blocks: number of communities
└─ state: BlockStateNested object

            ↓ Script 3: generate_null_models() + compare_with_null()

Null Distribution (100 models with permuted behavior)
├─ entropies: [X, Y, Z, ...]
├─ mean_entropy: avg null entropy
└─ Statistical Comparison:
   ├─ Z-score
   ├─ P-value
   └─ Significance

===============================================================================

QUICK REFERENCE
===============

# Load and prepare data
import numpy as np
from get_graphobject import create_multilayer_graph
from sbm_community_detection import fit_sbm_model
from null_model_validation import generate_null_models, compare_with_null

adj = np.load('adjacencies.npy')
behavior = np.load('scores.npy')
with open('node_names.txt') as f:
    nodes = [line.strip() for line in f]

# Create graph
graph = create_multilayer_graph(adj, behavior, nodes)

# Fit real model
real = fit_sbm_model(graph)

# Generate and compare null
null = generate_null_models(adj, behavior, nodes, n_null_models=100)
comp = compare_with_null(real['entropy'], null)

# Results
print(f"Real entropy: {real['entropy']:.2f}")
print(f"P-value: {comp['p_value']:.4f}")
print(f"Significant: {comp['is_significant']}")

===============================================================================

KEY FEATURES
============

✓ Multi-layer graph: BEHAVIOUR (weighted) + Co-occurrence (binary)
✓ Hierarchical SBM: Non-parametric Bayesian community detection
✓ Paper-based parameters: Defaults from Cipolotti et al. (2023)
✓ Null model validation: 100 permutations with randomized behavior
✓ Statistical comparison: Z-score, p-value, percentile rank
✓ Interactive shell ready: No main blocks, all functions for import
✓ graph_tool integration: Native support for weighted multi-layer models
✓ Fully parameterized: All paper defaults exposed as adjustable arguments

===============================================================================

PAPER REFERENCE
===============

Cipolotti L, et al. Graph lesion-deficit mapping of fluid intelligence.
BRAIN. 2023;146(1):167-181.
doi: 10.1093/brain/awac304

Key innovation: Layered hierarchical stochastic block model that 
disentangles functional (behavioral) effects from pathological 
(lesion co-occurrence) confounds.

===============================================================================

USAGE DOCUMENTATION
===================

For detailed usage examples, parameter descriptions, interpretation guide,
and troubleshooting, see: USAGE_GUIDE.md

===============================================================================
