



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


scores = np.genfromtxt(os.path.join(path, 'synth_scores.csv'), dtype = str, delimiter = ',', skip_header=1)
behaviour = scores[:,1].astype(np.float32)
behaviour = (behaviour - np.mean(behaviour)) / np.std(behaviour)  # Standardize scores
# lesion_locations = scores[:,1].tolist()
subject_list = scores[:,0].tolist()

node_names = np.genfromtxt(os.path.join(path, 'ATLAS','AAL3v1_coords_lobes.txt'), dtype=str, delimiter = '\t')[1:,0].tolist()

# ---- get disconnectomes ---- #
adj_matrices = np.zeros([len(subject_list),166,166], dtype=np.int32)  # To store adjacency matrices for each subject
empty_subjects = []  # To keep track of subjects with no disconnections
for subject in subject_list:
    tmp = np.genfromtxt(os.path.join(path,'DISCONNECTOMES', f'{subject}_lesion_AAL3.tsv'), delimiter = '\t')
    data = tmp[1:,1:].astype(np.float32)
    if np.sum(data) == 0:
        print(f"Subject {subject} has no disconnections.")
        empty_subjects.append(subject)
        continue
    adj_matrices[subject_list.index(subject)] = np.where(data >= np.quantile(data[data>0],.5),1,0)

print(f"Total subjects: {len(subject_list)}, Empty subjects: {len(empty_subjects)}")




graph = create_multilayer_graph(adj_matrices, behaviour, node_names)

# ---- Binarize layers: keep only top 50% of edges ---- #
print("\nBinarizing edge layers...")
n_nodes = graph.num_vertices()

# Extract layers as adjacency matrices
behaviour_matrix = np.zeros((n_nodes, n_nodes))
cooccurrence_matrix = np.zeros((n_nodes, n_nodes))

for edge in graph.edges():
    src = int(edge.source())
    dst = int(edge.target())
    behaviour_val = graph.ep.behaviour_weight[edge]
    cooccurrence_val = graph.ep.cooccurrence_weight[edge]
    
    behaviour_matrix[src, dst] = behaviour_val
    behaviour_matrix[dst, src] = behaviour_val
    
    cooccurrence_matrix[src, dst] = cooccurrence_val
    cooccurrence_matrix[dst, src] = cooccurrence_val

# Binarize: keep only top 50% edges for each layer
behaviour_nonzero = behaviour_matrix[behaviour_matrix > 0]
cooccurrence_nonzero = cooccurrence_matrix[cooccurrence_matrix > 0]

behaviour_threshold = np.percentile(behaviour_nonzero, 50)
cooccurrence_threshold = np.percentile(cooccurrence_nonzero, 50)

# Binarize to 0/1 only
behaviour_binary = (behaviour_matrix >= behaviour_threshold).astype(int)
cooccurrence_binary = (cooccurrence_matrix >= cooccurrence_threshold).astype(int)

print(f"Behaviour layer: kept {np.sum(behaviour_binary > 0) // 2} edges (threshold: {behaviour_threshold:.4f})")
print(f"Cooccurrence layer: kept {np.sum(cooccurrence_binary > 0) // 2} edges (threshold: {cooccurrence_threshold:.4f})")

# Create a new graph with binarized edges
g_bin = gt.Graph(directed=False)
g_bin.add_vertex(n_nodes)

# Add node labels
node_label_prop = g_bin.new_vertex_property("string")
for idx in range(n_nodes):
    node_label_prop[g_bin.vertex(idx)] = str(node_names[idx])
g_bin.vp.label = node_label_prop

# Add edges from binarized layers (binary 0/1 only)
behaviour_weight_prop = g_bin.new_edge_property("int")
cooccurrence_weight_prop = g_bin.new_edge_property("int")

added_edges = set()
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        if behaviour_binary[i, j] > 0 or cooccurrence_binary[i, j] > 0:
            edge = g_bin.add_edge(g_bin.vertex(i), g_bin.vertex(j))
            behaviour_weight_prop[edge] = int(behaviour_binary[i, j])
            cooccurrence_weight_prop[edge] = int(cooccurrence_binary[i, j])
            added_edges.add((i, j))

g_bin.ep.behaviour_weight = behaviour_weight_prop
g_bin.ep.cooccurrence_weight = cooccurrence_weight_prop

# Store metadata
g_bin.gp.n_patients = g_bin.new_graph_property("int", len(subject_list))
g_bin.gp.edge_threshold_applied = g_bin.new_graph_property("double", -1.0)

print(f"Binarized graph: {g_bin.num_vertices()} nodes, {g_bin.num_edges()} edges")
print(f"All edges are now binary (0 or 1)")

# ---- Fit Hierarchical Nested Block Model ---- #
print("\nFitting hierarchical nested block model...")

# Initialize nested block state without records (binary edges only)
# This allows hierarchy detection on the binarized graph
state_nested = gt.NestedBlockState(g_bin)

# MCMC sampling
print("Running MCMC sampling...")
mcmc_samples = 10000
burn_in = 500
annealing_temps = (1, 10)
annealing_steps = 50

# Simulated annealing
for temp in np.linspace(annealing_temps[1], annealing_temps[0], annealing_steps):
    state_nested.mcmc_sweep(niter=10, beta=1.0/temp)

# Burn-in
for i in range(burn_in):
    state_nested.mcmc_sweep(niter=1)

# Sampling with entropy tracking
dS = np.zeros(mcmc_samples)
for i in range(mcmc_samples):
    state_nested.mcmc_sweep(niter=1)
    dS[i] = state_nested.entropy()

print(f"Final entropy: {state_nested.entropy():.2f}")

# ---- Visualize nested hierarchical structure ---- #
# print("\nVisualizing hierarchical block structure...")

# # Draw using graph_tool's default visualization showing hierarchy
# draw.graph_draw(
#     g_bin,
#     pos=gt.sfdp_layout(g_bin, cooling_step=0.99, epsilon=1e-3, max_iter=100),
#     vertex_fill_color=state_nested.levels[0].b,  # Color vertices by block at level 0
#     vertex_size=30,
#     edge_pen_width=1.0,
#     vcmap=draw.default_cm,
#     output="/mnt/h/RT/data/V1_synth_score_hierarchy_structure.png",
#     output_size=(1200, 1200)
# )
# print("Hierarchical block structure visualization saved to: /mnt/h/RT/data/V1_synth_score_hierarchy_structure.png")

# Print hierarchy information
print("\nHierarchical Block Structure:")
print(f"Number of levels: {len(state_nested.levels)}")
for level_idx, level_state in enumerate(state_nested.levels):
    n_blocks = level_state.get_B()
    print(f"  Level {level_idx}: {n_blocks} blocks, entropy: {level_state.entropy():.2f}")

# Create entropy trajectory plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dS, linewidth=1, alpha=0.7, color='steelblue')
ax.set_xlabel('MCMC Iteration', fontsize=11)
ax.set_ylabel('Model Entropy', fontsize=11)
ax.set_title('Entropy Trajectory During MCMC Sampling (Nested Hierarchical)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/h/RT/data/RF_synth_score_entropy_trajectory.png', dpi=150, bbox_inches='tight')
print("Entropy trajectory saved to: /mnt/h/RT/data/RF_synth_score_entropy_trajectory.png")
plt.show()

# Get and display block assignments
blocks_level_0 = state_nested.levels[0].get_blocks().a
print("\nBlock Assignments (Level 0):")
print(f"Total nodes: {len(blocks_level_0)}")
print(f"Total blocks: {state_nested.levels[0].get_B()}")
for block_id in range(state_nested.levels[0].get_B()):
    nodes_in_block = np.where(blocks_level_0 == block_id)[0]
    print(f"  Block {block_id}: {len(nodes_in_block)} nodes")

# ---- Visualization of nested block structure using state.draw() ---- #
print("\nGenerating nested block structure visualization...")

import matplotlib.cm
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

# Get vertex colors from the block assignments
blocks_level_0 = state_nested.levels[0].get_blocks().a
n_blocks = state_nested.levels[0].get_B()

# Create a colormap for the blocks
cmap = plt.cm.get_cmap('tab20' if n_blocks <= 20 else 'hsv')
norm = Normalize(vmin=0, vmax=n_blocks-1)

# Get the position layout from the draw (we need to capture it)
# First, do the initial draw to get positions
pos = gt.sfdp_layout(g_bin, cooling_step=0.99, epsilon=1e-3, max_iter=100)

# Create vertex color property based on block assignments
vertex_color = g_bin.new_vertex_property("vector<double>")
vertex_text_color = g_bin.new_vertex_property("vector<double>")
for v in g_bin.vertices():
    block_id = blocks_level_0[int(v)]
    rgba = cmap(norm(block_id))
    vertex_color[v] = rgba
    vertex_text_color[v] = rgba

# Draw the nested block state showing hierarchy and community structure
state_nested.draw(
    pos=pos,
    vertex_fill_color=vertex_color,
    edge_color=gt.prop_to_size(g_bin.ep.behaviour_weight,
                               power=1,
                               log=False),
    ecmap=(matplotlib.cm.inferno, 0.6),
    eorder=g_bin.ep.behaviour_weight,
    edge_pen_width=gt.prop_to_size(g_bin.ep.behaviour_weight,
                                   0.5, 3,
                                   power=1,
                                   log=False),
    edge_gradient=[],
    vertex_text=g_bin.vp.label,
    vertex_text_color='black',
    vertex_text_position=0,
    vertex_font_size=10,
    output="/mnt/h/RT/data/RF_synth_score_nested_block_state_draw.png",
    output_size=(1200, 1200)
)
print("Nested block state visualization saved to: /mnt/h/RT/data/RF_synth_score_nested_block_state_draw.png")
