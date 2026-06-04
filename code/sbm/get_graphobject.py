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
# The following script generates input graph object for SBM based       #
# community detection                                                   #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/06/03.                                              #
#                                                                       #
#                                                                       #
#########################################################################





# ---- import libraries ---- #
import numpy as np
import graph_tool.all as gt
from graph_tool import inference


# ---- multilayer graph construction ---- #
def create_multilayer_graph(adjacency_matrices, behavioral_values, node_names, 
                            edge_threshold=None):
    """
    Create multi-layer graph object from patient lesion and behavioral data.
    
    Based on the framework from Cipolotti et al. (2023) BRAIN 146: 167-181,
    this function constructs a layered graph where:
    - Layer 1 (BEHAVIOUR): Adjacency matrices weighted by behavioral scores
    - Layer 2 (Co-occurrence): Binary concatenation of adjacency matrices
    
    The two layers can be statistically compared to disentangle functional
    (behavioral) from pathological (co-occurrence) effects.
    
    Parameters
    ----------
    adjacency_matrices : ndarray, shape (n_patients, 166, 166)
        Binary adjacency matrices for each patient, where adjacency_matrices[i, j, k]
        indicates presence (1) or absence (0) of an edge between nodes j and k
        for patient i.
    behavioral_values : ndarray, shape (n_patients,)
        Behavioral scores (e.g., APM performance, BEHAVIOUR variable) corresponding 
        to each patient. Values used directly without normalization.
    node_names : list or ndarray, shape (166,)
        Names/labels for the 166 brain regions/nodes.
    edge_threshold : float or None, optional
        Percentile threshold for edge filtering (0-100). If specified, only nodes
        in the top X% by degree are retained. If None (default), all edges are kept.
    
    Returns
    -------
    graph : graph_tool.Graph
        Multi-layer graph_tool Graph object with:
        - 166 nodes with 'label' property containing node names
        - Edges organized into two layers:
          * Layer 1: 'behaviour_weight' edge property (behavioral-weighted)
          * Layer 2: 'cooccurrence_weight' edge property (binary co-occurrence)
        - Graph properties: 'n_patients', 'edge_threshold_applied'
    
    Notes
    -----
    Layer 1 (BEHAVIOUR) construction:
    - For each patient i: weighted_adj[i] = adjacency_matrices[i] * behavioral_values[i]
    - Aggregates all weighted adjacencies across patients
    
    Layer 2 (Co-occurrence) construction:
    - Sum of all binary adjacency matrices across patients
    - Represents how many times each edge was present (co-occurrence pattern)
    
    The graph is undirected and weighted.
    """
    
    n_patients, n_nodes_dim1, n_nodes_dim2 = adjacency_matrices.shape
    assert n_nodes_dim1 == n_nodes_dim2 == 166, "Adjacency matrices must be 166x166"
    assert len(behavioral_values) == n_patients, "behavioral_values length must match n_patients"
    assert len(node_names) == 166, "node_names must contain exactly 166 labels"
    
    # Layer 1: BEHAVIOUR - Weight each patient's adjacency by their behavioral score
    behaviour_weighted = np.zeros((166, 166))
    for i in range(n_patients):
        behaviour_weighted += adjacency_matrices[i] * behavioral_values[i]
    
    # Layer 2: Co-occurrence - Sum binary adjacency matrices
    cooccurrence_binary = np.sum(adjacency_matrices, axis=0)
    
    # Optional edge filtering: keep only top X% connected nodes by degree
    if edge_threshold is not None:
        combined_adj = (behaviour_weighted > 0) | (cooccurrence_binary > 0)
        node_degrees = np.sum(combined_adj, axis=1)
        degree_threshold = np.percentile(node_degrees, edge_threshold)
        nodes_to_keep = np.where(node_degrees >= degree_threshold)[0]
    else:
        nodes_to_keep = np.arange(166)
    
    # Create graph_tool Graph
    g = gt.Graph(directed=False)
    
    # Add nodes with labels
    nodes = g.add_vertex(len(nodes_to_keep))
    node_label_prop = g.new_vertex_property("string")
    for idx, node_id in enumerate(nodes_to_keep):
        node_label_prop[g.vertex(idx)] = str(node_names[node_id])
    g.vp.label = node_label_prop
    
    # Create mapping from old node indices to new indices in filtered graph
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep)}
    
    # Add edges with layer-specific weights
    behaviour_weight_prop = g.new_edge_property("double")
    cooccurrence_weight_prop = g.new_edge_property("double")
    
    for i in nodes_to_keep:
        for j in nodes_to_keep:
            if i < j:  # Undirected: only add each edge once
                if behaviour_weighted[i, j] > 0 or cooccurrence_binary[i, j] > 0:
                    edge = g.add_edge(g.vertex(old_to_new_idx[i]), 
                                     g.vertex(old_to_new_idx[j]))
                    behaviour_weight_prop[edge] = float(behaviour_weighted[i, j])
                    cooccurrence_weight_prop[edge] = float(cooccurrence_binary[i, j])
    
    g.ep.behaviour_weight = behaviour_weight_prop
    g.ep.cooccurrence_weight = cooccurrence_weight_prop
    
    # Store metadata
    g.gp.n_patients = g.new_graph_property("int", n_patients)
    g.gp.edge_threshold_applied = g.new_graph_property("double", 
                                                        edge_threshold if edge_threshold else -1.0)
    
    return g

