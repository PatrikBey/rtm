╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║                          IMPLEMENTATION INDEX                                 ║
║              SBM-Based Lesion-Behavior Network Analysis Framework             ║
║                                                                                ║
║            Based on: Cipolotti et al. (2023) BRAIN 146: 167-181              ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝


GETTING STARTED
================

1. START HERE: QUICK_REFERENCE.txt
   └─ One-page overview with basic usage and commands
   └─ 5 minutes to understand the framework

2. NEXT: USAGE_GUIDE.md
   └─ Step-by-step interactive examples
   └─ Detailed parameter descriptions
   └─ 15 minutes to learn usage

3. DEEP DIVE: ARCHITECTURE.txt
   └─ Complete data flow diagrams
   └─ Algorithmic descriptions
   └─ Interpretation framework
   └─ 30 minutes to understand completely

4. REFERENCE: README.md
   └─ Implementation overview
   └─ File descriptions
   └─ Key features summary
   └─ Keep open while coding


IMPLEMENTATION FILES
====================

Script 1: get_graphobject.py
────────────────────────────
Function: create_multilayer_graph(adjacency_matrices, behavioral_values, 
                                   node_names, edge_threshold=None)

Purpose: Build multi-layer graph from patient data
Layers:
  - Layer 1: BEHAVIOUR-weighted adjacencies
  - Layer 2: Binary co-occurrence

Returns: graph_tool.Graph object

Time to understand: 10 minutes
Lines of code: 140


Script 2: sbm_community_detection.py
────────────────────────────────────
Function: fit_sbm_model(graph, mcmc_samples=100000, burn_in=50000,
                         annealing_temps=(1,10), annealing_steps=100)

Purpose: Fit hierarchical SBM to real data graph
Model:
  - BEHAVIOUR layer: Gaussian distribution
  - Co-occurrence layer: Poisson distribution

Returns: Dictionary with entropy, block_structure, n_blocks, state

Time to understand: 15 minutes
Lines of code: 139


Script 3: null_model_validation.py
───────────────────────────────────
Functions:
  - generate_null_models(adjacency_matrices, behavioral_values, node_names,
                         n_null_models=100, ...)
  - compare_with_null(real_entropy, null_distribution)

Purpose: Generate empirical null distribution and test significance
Process:
  - N=100 permutations with randomized behavioral scores
  - Statistical comparison (z-score, p-value, percentile rank)

Returns: Null distribution and comparison statistics

Time to understand: 20 minutes
Lines of code: 207


DOCUMENTATION FILES
===================

README.md (7.8 KB)
──────────────
Sections:
  - Overview of framework
  - Script descriptions
  - Key features
  - Data structures
  - Quick reference

Use for: Understanding what was implemented and why

USAGE_GUIDE.md (8.8 KB)
──────────────────────
Sections:
  - Step-by-step examples
  - Parameter descriptions
  - Interpretation guide
  - Troubleshooting
  - Advanced topics

Use for: Learning how to use the scripts

ARCHITECTURE.txt (22 KB)
──────────────────────
Sections:
  - Data flow diagrams
  - Input/output specifications
  - Visual representations
  - Algorithmic descriptions
  - Interpretation framework

Use for: Understanding how data flows through the system

QUICK_REFERENCE.txt (10 KB)
──────────────────────────
Sections:
  - Basic usage in shell
  - Function signatures
  - Parameter quick lookup
  - Interpretation rules
  - Troubleshooting tips

Use for: Quick lookup while coding


TYPICAL WORKFLOW
================

Step 1: Load Data (5 minutes)
─────────────────────────────
import numpy as np
from get_graphobject import create_multilayer_graph
from sbm_community_detection import fit_sbm_model
from null_model_validation import generate_null_models, compare_with_null

adj = np.load('adjacencies.npy')
scores = np.load('behavioral_scores.npy')
with open('node_names.txt') as f:
    names = [line.strip() for line in f]


Step 2: Create Graph (5 minutes)
────────────────────────────────
graph = create_multilayer_graph(adj, scores, names)
print(f"Graph created: {graph.num_vertices()} nodes, {graph.num_edges()} edges")


Step 3: Fit Real Model (30-60 minutes)
──────────────────────────────────────
real_results = fit_sbm_model(graph)
print(f"Entropy: {real_results['entropy']:.2f}")
print(f"Communities: {real_results['n_blocks']}")


Step 4: Generate Null Models (2-4 hours)
────────────────────────────────────────
null_dist = generate_null_models(adj, scores, names, n_null_models=100)
print(f"Null mean entropy: {null_dist['mean_entropy']:.2f}")
print(f"Null std entropy: {null_dist['std_entropy']:.2f}")


Step 5: Compare Results (1 minute)
──────────────────────────────────
comparison = compare_with_null(real_results['entropy'], null_dist)
print(f"P-value: {comparison['p_value']:.4f}")
print(f"Significant: {comparison['is_significant']}")


Step 6: Extract Communities (5 minutes)
──────────────────────────────────────
blocks = real_results['block_structure']
for i in range(real_results['n_blocks']):
    nodes = np.where(blocks == i)[0]
    regions = [names[j] for j in nodes]
    print(f"Community {i}: {regions}")


COMMON QUESTIONS
=================

Q: Where do I start?
A: Read QUICK_REFERENCE.txt first (5 min), then USAGE_GUIDE.md (15 min)

Q: How do I run the analysis?
A: See "WORKFLOW EXAMPLE" section in QUICK_REFERENCE.txt

Q: What's the difference between the 3 scripts?
A: See ARCHITECTURE.txt for complete data flow diagram

Q: What do the parameters mean?
A: See USAGE_GUIDE.md "DETAILED PARAMETER GUIDE" section

Q: How do I interpret the results?
A: See QUICK_REFERENCE.txt "INTERPRETATION RULES" section

Q: What if something goes wrong?
A: See USAGE_GUIDE.md "TROUBLESHOOTING" section

Q: Can I change the parameters?
A: Yes, all are function parameters with defaults from the paper

Q: How long does analysis take?
A: Real model: 30-60 min, Null models: 2-4 hours (100 iterations)


PARAMETER DEFAULTS
===================

Script 1: create_multilayer_graph()
────────────────────────────────────
edge_threshold: None              (keep all edges)


Script 2: fit_sbm_model()
─────────────────────────
mcmc_samples: 100000              (paper default)
burn_in: 50000                    (paper default)
annealing_temps: (1, 10)          (paper default)
annealing_steps: 100              (reasonable default)


Script 3: generate_null_models()
────────────────────────────────
n_null_models: 100                (reasonable default)
edge_threshold: None              (match real analysis)
mcmc_samples: 100000              (match real analysis)
burn_in: 50000                    (match real analysis)
annealing_temps: (1, 10)          (match real analysis)
annealing_steps: 100              (match real analysis)


OUTPUT SUMMARY
===============

Real Model Results:
  entropy:               Model quality (lower = better)
  block_structure:       Community assignment per node
  n_blocks:              Number of communities found
  state:                 BlockStateNested object
  entropy_trajectory:    MCMC entropy trace

Null Distribution:
  entropies:             Array of N entropy values
  mean_entropy:          Null mean
  std_entropy:           Null standard deviation
  percentiles:           5, 25, 50, 75, 95 percentiles

Comparison Results:
  z_score:               Statistical measure
  percentile_rank:       Where real falls (0-100%)
  p_value:               Significance test (< 0.05 = significant)
  is_significant:        Boolean result


FILE TREE
===========

/mnt/h/GitHub/sssp/code/sbm/
├── Implementation Scripts
│   ├── get_graphobject.py (140 lines) ← Script 1: Graph construction
│   ├── sbm_community_detection.py (139 lines) ← Script 2: Real SBM
│   └── null_model_validation.py (207 lines) ← Script 3: Null validation
│
├── Documentation
│   ├── QUICK_REFERENCE.txt (10 KB) ← START HERE
│   ├── USAGE_GUIDE.md (9 KB) ← Next: learn usage
│   ├── ARCHITECTURE.txt (22 KB) ← Deep dive: data flow
│   ├── README.md (8 KB) ← Reference: overview
│   └── INDEX (this file)
│
└── [When running analysis, user creates]
    ├── adjacencies.npy (input data)
    ├── behavioral_scores.npy (input data)
    ├── node_names.txt (input data)
    └── [Results stored in Python variables, not files]


REQUIREMENTS
=============

Python Libraries:
  • numpy (data structures)
  • graph_tool (multi-layer SBM)

Python Version:
  • Python 3.7+

Installation:
  conda install -c conda-forge graph-tool
  # or: pip install graph-tool (if not available, use conda)


PAPER REFERENCE
================

Cipolotti L, Ruffle JK, Mole J, Xu T, Hyare H, Shallice T, Chan E, Nachev P.
Graph lesion-deficit mapping of fluid intelligence.
BRAIN. 2023;146(1):167-181.
doi: 10.1093/brain/awac304

Key Innovation: Layered hierarchical stochastic block model distinguishing 
behavioral (APM performance) from pathological (lesion co-occurrence) effects


SUPPORT & REFERENCES
=====================

For questions about:
  - Graph construction: See get_graphobject.py docstring
  - SBM fitting: See sbm_community_detection.py docstring
  - Null validation: See null_model_validation.py docstring
  - Usage examples: See USAGE_GUIDE.md
  - Data flow: See ARCHITECTURE.txt
  - Quick lookup: See QUICK_REFERENCE.txt
  - General overview: See README.md


═════════════════════════════════════════════════════════════════════════════════

Start with QUICK_REFERENCE.txt → Proceed to USAGE_GUIDE.md → Consult 
ARCHITECTURE.txt as needed → Keep README.md as reference

Total time to learn framework: ~1 hour
Total time for analysis: ~3-4 hours (plus computation time)

═════════════════════════════════════════════════════════════════════════════════
