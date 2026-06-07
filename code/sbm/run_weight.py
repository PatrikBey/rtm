

import os
import graph_tool.all as gt
from graph_tool import draw
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import numpy as np
from get_graphobject import create_multilayer_graph
from sbm_community_detection_weight import fit_nested_sbm_layered
# from null_model_validation import generate_null_models, compare_with_null




path = '/mnt/h/RT/data'



discos = os.listdir(os.path.join(path,'DISCONNECTOMES'))


scores = np.genfromtxt(os.path.join(path, 'synth_scores_raw.csv'), dtype = str, delimiter = ',', skip_header=1)
behaviour = scores[:,1].astype(np.float32)
# lesion_locations = scores[:,1].tolist()
subject_list = scores[:,0].tolist()

node_names = np.genfromtxt(os.path.join(path, 'ATLAS','AAL3v1_coords_lobes.txt'), dtype=str, delimiter = '\t')[1:,0].tolist()
locations = np.genfromtxt(os.path.join(path, 'ATLAS','AAL3v1_coords_lobes.txt'), dtype=str, delimiter = '\t')[1:,-1].tolist()


'''
## FUNCTIONAL GROUPINGS (for node color coding)
Network/Group	Regions
Salience Network	Cingulate + Insula
Default Mode / Limbic	Cingulate (posterior) + Medial temporal
Executive / Frontoparietal	Frontal + Parietal
Visual Stream	Occipital + Temporo-occipital
Temporal/Limbic	Temporal + Medial temporal
Subcortical	Thalamus + Subcortical
BCM	Brainstem + Midbrain + Cerebellum

## ANATOMICAL GROUPINGS (for node color coding)
Brainstem	Brainstem + Midbrain — Midbrain is literally a subdivision of the brainstem (along with pons and medulla)
Cerebellum	Cerebellum — Standalone; functionally and anatomically distinct
Frontal Lobe	Frontal — Motor, executive, prefrontal functions
Parietal Lobe	Parietal — Somatosensory, spatial, parietal association
Temporal Lobe	Temporal + Medial temporal + Temporo-occipital — All temporal lobe territory; memory, auditory, language
Occipital Lobe	Occipital — Primary visual cortex (can include Temporo-occipital if emphasizing visual stream)
Limbic / Paralimbic	Cingulate + Insula — Often grouped as "cingulo-insular" cortex; core of salience network and emotional processing
Subcortical	Thalamus + Subcortical — Deep gray matter structures (note: thalamus IS subcortical, so "Subcortical" likely refers to basal ganglia/other nuclei here)

'''

func_locations = ['Salience' if loc in ['Cingulate', 'Insula'] else
                  'Default' if loc in ['Cingulate (posterior)', 'Medial temporal'] else
                  'Frontoparietal' if loc in ['Frontal', 'Parietal'] else
                  'Visual' if loc in ['Occipital', 'Temporo-occipital'] else
                  'Temporal' if loc in ['Temporal', 'Medial temporal'] else
                  'Subcortical' if loc in ['Thalamus', 'Subcortical'] else
                  'BCM' if loc in ['Brainstem', 'Midbrain','Cerebellum'] else 'Other' for loc in locations]


loc_colours = ['crimson','fuchsia','purple','indigo','mediumslateblue','cornflowerblue','powderblue','cyan','teal','limegreen','olive','gold','darkorange']

func_loc_colours = ['fuchsia','purple','cornflowerblue','teal','limegreen','gold','darkorange']

locations = [f'{loc}_L' if idx < 166/2 else f'{loc}_R' for idx, loc in enumerate(func_locations)]

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




graph = create_multilayer_graph(adj_matrices, behaviour, node_names, edge_threshold=75)




real_results = fit_nested_sbm_layered(
    graph,
    mcmc_samples=100000,
    burn_in=10000,
    annealing_temps=(1, 10),
    annealing_steps=100
)

# ---- Print hierarchical structure ---- #
state_nested = real_results['state']
g = state_nested.g

print(f"\nDetected {real_results['n_blocks_level_0']} communities at level 0")
print(f"Total model entropy: {real_results['entropy']:.2f}")
print(f"\nHierarchical Block Structure:")
print(f"Number of levels: {real_results['n_levels']}")
for level_idx in range(real_results['n_levels']):
    print(f"  Level {level_idx}: {real_results['levels_n_blocks'][level_idx]} blocks, "
          f"entropy: {real_results['levels_entropy'][level_idx]:.2f}")

# ---- Block assignments at level 0 (finest resolution) ---- #
blocks_level_0 = real_results['block_structure_level_0']
n_blocks = real_results['n_blocks_level_0']

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
plt.savefig('/mnt/h/RT/data/figures/RF_weighted_nested_entropy_trajectory_75.png', dpi=150, bbox_inches='tight')
print("\nEntropy trajectory saved to: /mnt/h/RT/data/figures/RF_weighted_nested_entropy_trajectory.png")
plt.show()

# ---- Visualization of nested block structure ---- #
print("\nGenerating nested block structure visualization...")

# Compute layout on the weighted graph
pos = gt.sfdp_layout(g, cooling_step=0.99, epsilon=1e-3, max_iter=100)

# Extract unique location names (without _L/_R suffix)
unique_location_names = sorted(set([loc.rsplit('_', 1)[0] for loc in locations]))
location_name_to_idx = {name: idx for idx, name in enumerate(unique_location_names)}
n_locations = len(unique_location_names)

# Convert func_loc_colours (named colors) to RGB tuples
from matplotlib.colors import to_rgba
distinct_colors = [to_rgba(color) for color in func_loc_colours]

# Create a custom colormap from these distinct colors
from matplotlib.colors import ListedColormap
cmap = ListedColormap(distinct_colors)
norm = Normalize(vmin=0, vmax=max(n_locations - 1, 1))

vertex_color = g.new_vertex_property("vector<double>")
vertex_shape = g.new_vertex_property("int")  # 0=circle, 1=triangle
for v in g.vertices():
    node_idx = int(v)
    location = locations[node_idx]
    # Extract location name (remove _L/_R suffix)
    location_name = location.rsplit('_', 1)[0]
    side = location.rsplit('_', 1)[1]  # 'L' or 'R'
    location_idx = location_name_to_idx[location_name]
    rgba = cmap(norm(location_idx))
    vertex_color[v] = rgba
    # Set shape: 0 for circle (left), 1 for triangle (right)
    vertex_shape[v] = 0 if side == 'L' else 1

# Compute node sizes based on degree
degree_map = g.degree_property_map("in")  # Get degree property map
vertex_sizes = gt.prop_to_size(degree_map, 
                               mi=20, 
                               ma=50)  # Node size range: 20 to 50 points

# Create edge colors matching the node color gradient
edge_color = g.new_edge_property("vector<double>")

for e in g.edges():
    # Get colors of the source and target nodes
    src_color = vertex_color[e.source()]
    tgt_color = vertex_color[e.target()]
    
    # Average the colors of the two endpoints (RGB only)
    avg_color = list((src_color[i] + tgt_color[i]) / 2 for i in range(3))
    
    # Add transparency (alpha = 0.4 for semi-transparent edges)
    avg_color.append(0.4)
    
    edge_color[e] = tuple(avg_color)

# Draw using state_nested.draw() to show hierarchical layout
state_nested.draw(
    pos=pos,
    vertex_fill_color=vertex_color,
    vertex_shape=vertex_shape,  # Circle for left, triangle for right
    vertex_size=vertex_sizes,  # Size based on node degree
    vertex_pen_width=0.5,  # Thin outline
    edge_color=edge_color,  # Match node color gradient
    edge_pen_width=gt.prop_to_size(g.ep.behaviour_weight,
                                   mi=0.5,
                                   ma=3),  # Edge width based on behaviour_weight
    edge_gradient=[],
    vertex_text=g.vp.label,
    vertex_text_color='black',
    vertex_text_position=0,
    vertex_font_size=10,
    output="/mnt/h/RT/data/figures/RF_weighted_nested_block_state_draw.png",
    output_size=(1200, 1200)
)
print("Nested block state visualization saved to: /mnt/h/RT/data/figures/RF_weighted_nested_block_state_draw_75.png")

# ---- Create location legend ---- #
print("\nGenerating location legend...")

fig, ax = plt.subplots(figsize=(14, 10))

# Create legend entries with colors and location names
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = []
legend_labels = []

for loc_idx, location_name in enumerate(unique_location_names):
    rgba = cmap(norm(loc_idx))
    nodes_in_location = np.where(np.array([loc.rsplit('_', 1)[0] for loc in locations]) == location_name)[0]
    
    # Create legend entries for both left and right sides
    # Left side (circle)
    circle_element = Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=rgba, markersize=10, 
                           markeredgecolor='black', markeredgewidth=1.5,
                           label=f"{location_name}_L ({len([n for n in nodes_in_location if locations[n].endswith('_L')])} nodes)")
    legend_elements.append(circle_element)
    
    # Right side (triangle)
    triangle_element = Line2D([0], [0], marker='^', color='w', 
                             markerfacecolor=rgba, markersize=10, 
                             markeredgecolor='black', markeredgewidth=1.5,
                             label=f"{location_name}_R ({len([n for n in nodes_in_location if locations[n].endswith('_R')])} nodes)")
    legend_elements.append(triangle_element)

ax.legend(handles=legend_elements, 
          loc='center', fontsize=10, frameon=True, 
          title="Node Color and Shape by Location and Side\n(Circles=Left, Triangles=Right)", 
          title_fontsize=12, ncol=2, 
          fancybox=True, shadow=True)
ax.axis('off')
plt.tight_layout()
plt.savefig('/mnt/h/RT/data/figures/RF_weighted_nested_location_legend.png', dpi=150, bbox_inches='tight')
print("Location legend saved to: /mnt/h/RT/data/figures/RF_weighted_nested_location_legend.png")
plt.show()

# ---- Print location distribution summary ---- #
print("\nLocation Distribution Summary:")
print(f"Total unique locations: {n_locations}")
for location_name in unique_location_names:
    nodes_in_location_l = np.where(np.array(locations) == f'{location_name}_L')[0]
    nodes_in_location_r = np.where(np.array(locations) == f'{location_name}_R')[0]
    print(f"  {location_name}_L: {len(nodes_in_location_l)} nodes (circles)")
    print(f"  {location_name}_R: {len(nodes_in_location_r)} nodes (triangles)")
