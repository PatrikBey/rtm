


import os
import graph_tool.all as gt
from graph_tool import draw
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import numpy as np
from get_graphobject import create_multilayer_graph
from sbm_community_detection_weight import fit_sbm_model
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




real_results = fit_sbm_model(
    graph,
    mcmc_samples=100000,
    burn_in=50000,
    annealing_temps=(1, 10),
    annealing_steps=100
)

# ---- Print hierarchical structure ---- #
state_nested = real_results['state']
g = state_nested.g

print(f"\nDetected {real_results['n_blocks']} communities at level 0")
print(f"Total model entropy: {real_results['entropy']:.2f}")
print(f"\nHierarchical Block Structure:")
print(f"Number of levels: {real_results['n_levels']}")
for level_idx in range(real_results['n_levels']):
    print(f"  Level {level_idx}: {real_results['levels_n_blocks'][level_idx]} blocks, "
          f"entropy: {real_results['levels_entropy'][level_idx]:.2f}")

# ---- Block assignments at level 0 (finest resolution) ---- #
blocks_level_0 = real_results['block_structure']
n_blocks = real_results['n_blocks']

print("\nBlock Assignments (Level 0):")
print(f"Total nodes: {len(blocks_level_0)}")
print(f"Total blocks: {n_blocks}")
for block_id in range(n_blocks):
    nodes_in_block = np.where(blocks_level_0 == block_id)[0]
    print(f"  Block {block_id}: {len(nodes_in_block)} nodes")

# ---- Entropy trajectory plot ---- #
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(real_results['entropy_trajectory'], linewidth=1, alpha=0.7, color='steelblue')
ax.set_xlabel('MCMC Iteration', fontsize=11)
ax.set_ylabel('Model Entropy', fontsize=11)
ax.set_title('Entropy Trajectory During MCMC Sampling (Weighted Nested Hierarchical)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/h/RT/data/weighted_nested_entropy_trajectory.png', dpi=150, bbox_inches='tight')
print("\nEntropy trajectory saved to: /mnt/h/RT/data/weighted_nested_entropy_trajectory.png")
plt.show()

# ---- Visualization of nested block structure ---- #
print("\nGenerating nested block structure visualization...")

# Compute layout on the weighted graph
pos = gt.sfdp_layout(g, cooling_step=0.99, epsilon=1e-3, max_iter=100)

# Vertex colors from level-0 block assignments
cmap = plt.cm.get_cmap('tab20' if n_blocks <= 20 else 'hsv')
norm = Normalize(vmin=0, vmax=max(n_blocks - 1, 1))

vertex_color = g.new_vertex_property("vector<double>")
for v in g.vertices():
    block_id = blocks_level_0[int(v)]
    rgba = cmap(norm(block_id))
    vertex_color[v] = rgba

# Draw using state_nested.draw() to show hierarchical layout
state_nested.draw(
    pos=pos,
    vertex_fill_color=vertex_color,
    edge_color=gt.prop_to_size(g.ep.behaviour_weight,
                               power=1,
                               log=False),
    ecmap=(matplotlib.cm.inferno, 0.6),
    eorder=g.ep.behaviour_weight,
    edge_pen_width=gt.prop_to_size(g.ep.behaviour_weight,
                                   0.5, 3,
                                   power=1,
                                   log=False),
    edge_gradient=[],
    vertex_text=g.vp.label,
    vertex_text_color='black',
    vertex_text_position=0,
    vertex_font_size=10,
    output="/mnt/h/RT/data/weighted_nested_block_state_draw.png",
    output_size=(1200, 1200)
)
print("Nested block state visualization saved to: /mnt/h/RT/data/weighted_nested_block_state_draw.png")
