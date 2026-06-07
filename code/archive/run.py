




import os
import graph_tool.all as gt
from graph_tool import draw
import matplotlib.pyplot as plt
import numpy as np
from get_graphobject import create_multilayer_graph
from sbm_community_detection import fit_sbm_model
from null_model_validation import generate_null_models, compare_with_null




path = '/mnt/h/RT/data'



discos = os.listdir(os.path.join(path,'DISCONNECTOMES'))


scores = np.genfromtxt(os.path.join(path, 'synth_rt.csv'), dtype = str, delimiter = ',', skip_header=1)
behaviour = scores[:,2].astype(np.float32)
lesion_locations = scores[:,1].tolist()
subject_list = scores[:,0].tolist()

node_names = np.genfromtxt(os.path.join(path, 'ATLAS','AAL3v1_coords_lobes.txt'), dtype=str, delimiter = '\t')[1:,0].tolist()

# ---- get disconnectomes ---- #
adj_matrices = np.zeros([len(subject_list),166,166], dtype=np.int32)  # To store adjacency matrices for each subject
empty_subjects = []  # To keep track of subjects with no disconnections
for subject in subject_list:
    tmp = np.genfromtxt(os.path.join(path,'DISCONNECTOMES', f'sub-{subject}_lesion_AAL3.tsv'), delimiter = '\t')
    data = tmp[1:,1:].astype(np.float32)
    if np.sum(data) == 0:
        print(f"Subject {subject} has no disconnections.")
        empty_subjects.append(subject)
        continue
    adj_matrices[subject_list.index(subject)] = np.where(data >= np.quantile(data[data>0],.5),1,0)

print(f"Total subjects: {len(subject_list)}, Empty subjects: {len(empty_subjects)}")




graph = create_multilayer_graph(adj_matrices, behaviour, node_names)




# beh_graph, structure_graph = extract_layers_from_graph(graph)

# color_lim = max(beh_graph.max(), structure_graph.max())
# plt.subplot(1, 2, 1)
# plt.imshow(beh_graph, cmap='viridis')
# plt.clim(0, color_lim)
# plt.title('BEHAVIOUR Layer')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(structure_graph, cmap='viridis')
# plt.clim(0, color_lim)
# plt.title('STRUCTURAL Layer')
# plt.colorbar()
# plt.show()


real_results = fit_sbm_model(
    graph,
    mcmc_samples=10000,  # 100k posterior samples (paper default)
    burn_in=500,        # 50k burn-in (paper default)
    annealing_temps=(1, 10),  # Temperature range (paper default)
    annealing_steps=100
)

# ---- Visualize detected communities and edges ---- #
print(f"\nDetected {real_results['n_blocks']} communities")
print(f"Model entropy: {real_results['entropy']:.2f}")

# Extract block state
state_best = real_results['state']
g = state_best.g
blocks_final = state_best.get_blocks().a

# Create circular layout with blocks arranged around the circle
n_nodes = g.num_vertices()
n_blocks = real_results['n_blocks']

# Sort nodes by their block assignment
node_order = np.argsort(blocks_final)
blocks_sorted = blocks_final[node_order]

# Compute circular positions grouped by blocks
pos = g.new_vertex_property("vector<double>")
angle_per_node = 2 * np.pi / n_nodes

for idx, node_id in enumerate(node_order):
    angle = angle_per_node * idx
    x = np.cos(angle)
    y = np.sin(angle)
    pos[g.vertex(node_id)] = [x, y]

# Get edge weights for visualization (using the behaviour layer weights)
edge_weights = []
edge_colors = g.new_edge_property("vector<double>")  # RGBA colors for edges

for edge in g.edges():
    src = int(edge.source())
    dst = int(edge.target())
    weight = g.ep.behaviour_weight[edge]
    edge_weights.append(weight)
    
    # Color edges based on block structure from state_best.b
    if state_best.b[g.vertex(src)] == state_best.b[g.vertex(dst)]:
        # Within-block edge: green
        edge_colors[edge] = [0.2, 0.8, 0.2, 0.8]  # Green with transparency
    else:
        # Between-block edge: red
        edge_colors[edge] = [0.8, 0.2, 0.2, 0.6]  # Red with more transparency

edge_weights = np.array(edge_weights)

# Create edge pen width based on behaviour layer weights
if len(edge_weights) > 0 and edge_weights.max() > 0:
    edge_weight_prop = g.new_edge_property("double")
    for i, edge in enumerate(g.edges()):
        edge_weight_prop[edge] = edge_weights[i]
    
    edge_pen_widths = gt.prop_to_size(
        edge_weight_prop,
        .1, 3, power=1
    )
else:
    edge_pen_widths = None

# Draw the circular block state visualization
print("\nGenerating circular block state visualization...")
draw.graph_draw(
    g,
    pos=pos,
    vertex_fill_color=state_best.b,
    vertex_color=state_best.b,
    edge_color=edge_colors,
    edge_pen_width=edge_pen_widths,
    vcmap=draw.default_cm,
    output="/mnt/h/RT/data/circular_block_state.png",
    output_size=(1200, 1200)
)
print("Circular block state visualization saved to: /mnt/h/RT/data/circular_block_state.png")

# Create entropy trajectory plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(real_results['entropy_trajectory'], linewidth=1, alpha=0.7, color='steelblue')
ax.set_xlabel('MCMC Iteration', fontsize=11)
ax.set_ylabel('Model Entropy', fontsize=11)
ax.set_title('Entropy Trajectory During MCMC Sampling', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/h/RT/data/entropy_trajectory.png', dpi=150, bbox_inches='tight')
print("Entropy trajectory saved to: /mnt/h/RT/data/entropy_trajectory.png")
plt.show()

# Print community assignments
print("\nCommunity Assignments:")
print(f"Total nodes: {len(blocks_final)}")
print(f"Total communities: {real_results['n_blocks']}")
for block_id in range(real_results['n_blocks']):
    nodes_in_block = np.where(blocks_final == block_id)[0]
    print(f"  Community {block_id}: {len(nodes_in_block)} nodes")



























# Direct access to layer weights per edge
import numpy as np
import graph_tool.all as gt
def extract_layers_from_graph(graph):
    """
    Extract the different layers from a multi-layer graph_tool graph object
    as separate adjacency matrices.
    
    Parameters
    ----------
    graph : graph_tool.Graph
        Multi-layer graph with edge properties:
        - 'behaviour_weight': BEHAVIOUR layer weights
        - 'cooccurrence_weight': co-occurrence layer weights
    
    Returns
    -------
    behaviour_matrix : numpy.ndarray, shape (n_nodes, n_nodes)
        Adjacency matrix for the BEHAVIOUR layer
    cooccurrence_matrix : numpy.ndarray, shape (n_nodes, n_nodes)
        Adjacency matrix for the co-occurrence layer
    """
    
    # Extract node count
    n_nodes = graph.num_vertices()
    
    # Initialize layer matrices
    behaviour_matrix = np.zeros((n_nodes, n_nodes))
    cooccurrence_matrix = np.zeros((n_nodes, n_nodes))
    
    # Extract edge properties
    behaviour_weights = graph.ep.behaviour_weight
    cooccurrence_weights = graph.ep.cooccurrence_weight
    
    # Populate matrices from edge properties
    for edge in graph.edges():
        src = int(edge.source())
        dst = int(edge.target())
        
        behaviour_val = behaviour_weights[edge]
        cooccurrence_val = cooccurrence_weights[edge]
        
        # For undirected graphs, mirror the values
        behaviour_matrix[src, dst] = behaviour_val
        behaviour_matrix[dst, src] = behaviour_val
        
        cooccurrence_matrix[src, dst] = cooccurrence_val
        cooccurrence_matrix[dst, src] = cooccurrence_val
    
    return behaviour_matrix, cooccurrence_matrix