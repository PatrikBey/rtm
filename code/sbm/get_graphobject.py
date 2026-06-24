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
                            edge_threshold=50):
    """
    Create a redundant two-layer graph for weighted nested SBM inference.
    
    Each node pair (i, j) appears in both layers with complementary weights:
    - Layer 0: behaviour_weight active, cooccurrence_weight = 0
    - Layer 1: cooccurrence_weight active, behaviour_weight = 0
    
    Parameters
    ----------
    adjacency_matrices : ndarray, shape (n_patients, 166, 166)
        Binary adjacency matrices per patient.
    behavioral_values : ndarray, shape (n_patients,)
        Behavioral scores per patient.
    node_names : list or ndarray, shape (166,)
        Brain region labels.
    edge_threshold : float, optional
        Percentile threshold (0-100) for edge filtering (default: 50).
    
    Returns
    -------
    graph : graph_tool.Graph
        Graph with properties:
        - 'behaviour_weight' : real-valued weights for layer 0
        - 'cooccurrence_weight' : integer counts for layer 1
        - 'layer' : layer assignment (0 or 1) for LayeredBlockState
        - Graph properties: 'n_patients', 'edge_threshold_applied'
    """
    
    n_patients, n_nodes_dim1, n_nodes_dim2 = adjacency_matrices.shape
    assert n_nodes_dim1 == n_nodes_dim2, "Adjacency matrices must be square"
    assert len(behavioral_values) == n_patients, "behavioral_values length must match n_patients"
    assert len(node_names) == n_nodes_dim1, "node_names length must match adjacency matrix size"
    assert 0 <= edge_threshold <= 100, "edge_threshold must be between 0 and 100"
    
    # Compute layer weights
    behaviour_weighted = np.zeros((n_nodes_dim1, n_nodes_dim1))
    for i in range(n_patients):
        behaviour_weighted += adjacency_matrices[i] * behavioral_values[i]
    
    cooccurrence_binary = np.sum(adjacency_matrices, axis=0)
    
    # Apply shared percentile threshold
    combined_nonzero = np.concatenate([
        behaviour_weighted[behaviour_weighted > 0],
        cooccurrence_binary[cooccurrence_binary > 0]
    ])
    threshold_value = np.percentile(combined_nonzero, edge_threshold)
    
    behaviour_weighted[behaviour_weighted < threshold_value] = 0
    cooccurrence_binary[cooccurrence_binary < threshold_value] = 0
    
    # Filter nodes with edges
    combined_adj = (behaviour_weighted > 0) | (cooccurrence_binary > 0)
    nodes_to_keep = np.where(np.sum(combined_adj, axis=1) > 0)[0]
    
    # Create graph
    g = gt.Graph(directed=False)
    g.add_vertex(len(nodes_to_keep))
    
    # Add node labels
    node_label_prop = g.new_vertex_property("string")
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep)}
    for idx, node_id in enumerate(nodes_to_keep):
        node_label_prop[g.vertex(idx)] = str(node_names[node_id])
    g.vp.label = node_label_prop
    
    # Add redundant edges with dual weights
    behaviour_weight_prop = g.new_edge_property("double")
    cooccurrence_weight_prop = g.new_edge_property("double")
    layer_prop = g.new_edge_property("int")
    
    for i in nodes_to_keep:
        for j in nodes_to_keep:
            if i < j:
                has_behaviour = behaviour_weighted[i, j] > 0
                has_cooccurrence = cooccurrence_binary[i, j] > 0
                
                if has_behaviour or has_cooccurrence:
                    behaviour_val = float(behaviour_weighted[i, j]) if has_behaviour else 0.0
                    cooccurrence_val = float(cooccurrence_binary[i, j]) if has_cooccurrence else 0.0
                    
                    # Layer 0: behaviour weight active
                    e0 = g.add_edge(g.vertex(old_to_new_idx[i]), g.vertex(old_to_new_idx[j]))
                    behaviour_weight_prop[e0] = behaviour_val
                    cooccurrence_weight_prop[e0] = 0.0
                    layer_prop[e0] = 0
                    
                    # Layer 1: cooccurrence weight active
                    e1 = g.add_edge(g.vertex(old_to_new_idx[i]), g.vertex(old_to_new_idx[j]))
                    behaviour_weight_prop[e1] = 0.0
                    cooccurrence_weight_prop[e1] = cooccurrence_val
                    layer_prop[e1] = 1
    
    g.ep.behaviour_weight = behaviour_weight_prop
    g.ep.cooccurrence_weight = cooccurrence_weight_prop
    g.ep.layer = layer_prop
    
    g.gp.n_patients = g.new_graph_property("int", n_patients)
    g.gp.edge_threshold_applied = g.new_graph_property("double", threshold_value)
    
    return g

