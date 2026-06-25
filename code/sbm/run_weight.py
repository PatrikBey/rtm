

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
score = 'GoNoGo_tau'
# score = 'Foreperiod_Long_tau'

ATLAS = 'HCP-MMP1'

discos = os.listdir(os.path.join(path,'DISCONNECTOMES'))
subject_list = [f.split('_')[0] for f in discos if f.endswith(f'_{ATLAS}.tsv')]







part = np.genfromtxt(os.path.join(path, 'participants.tsv'), dtype=str, delimiter='\t')

score_col = np.where(part[0] == score)[0][0]





node_names = np.genfromtxt(os.path.join(path, 'ATLAS', f'{ATLAS}_areas.txt'), dtype=str, delimiter = '\t')[1:,0].tolist()
locations = np.genfromtxt(os.path.join(path, 'ATLAS', f'{ATLAS}_areas.txt'), dtype=str, delimiter = '\t')[1:,2].tolist()
dim = len(node_names)

'''
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

# func_locations = ['Salience' if loc in ['Cingulate', 'Insula'] else
#                   'Default' if loc in ['Cingulate (posterior)', 'Medial temporal'] else
#                   'Frontoparietal' if loc in ['Frontal', 'Parietal'] else
#                   'Visual' if loc in ['Occipital', 'Temporo-occipital'] else
#                   'Temporal' if loc in ['Temporal', 'Medial temporal'] else
#                   'Subcortical' if loc in ['Thalamus', 'Subcortical'] else
#                   'BCM' if loc in ['Brainstem', 'Midbrain','Cerebellum'] else 'Other' for loc in locations]


loc_colours = ['crimson','fuchsia','purple','indigo','mediumslateblue','cornflowerblue','powderblue','cyan','teal','limegreen','olive','gold','darkorange']

# func_loc_colours = ['fuchsia','purple','cornflowerblue','teal','limegreen','gold','darkorange']

locations = [f'{loc}_L' if idx < dim/2 else f'{loc}_R' for idx, loc in enumerate(locations)]

# ---- get disconnectomes and behaviour (aligned, clean subjects only) ---- #

subject_list_clean = []
behaviour = []
adj_matrices_list = []
subjects_missing_score = []
empty_subjects = []

for subject in subject_list:
    val = part[part[:, 0] == subject, score_col]
    if val.size == 0 or val[0] in ('', 'nan', 'NaN'):
        subjects_missing_score.append(subject)
        continue
    tmp = np.genfromtxt(os.path.join(path, 'DISCONNECTOMES', f'{subject}_{ATLAS}.tsv'), delimiter='\t')
    data = tmp[1:, 1:].astype(np.float32)
    if np.sum(data) == 0:
        empty_subjects.append(subject)
        continue
    subject_list_clean.append(subject)
    behaviour.append(float(val[0]))
    adj_matrices_list.append(np.where(data >= np.quantile(data[data > 0], .5), 1, 0))

adj_matrices = np.stack(adj_matrices_list).astype(np.int32)
behaviour = list((np.array(behaviour) - np.mean(behaviour)) / np.std(behaviour))

print(f"Total subjects: {len(subject_list)}")
print(f"  Included: {len(subject_list_clean)}")
print(f"  Missing {score}: {len(subjects_missing_score)} — {subjects_missing_score}")
print(f"  Empty disconnectome: {len(empty_subjects)} — {empty_subjects}")





graph = create_multilayer_graph(adj_matrices, behaviour, node_names, edge_threshold=50)

n = graph.num_vertices()
occ_layer = np.zeros((n, n))
beh_layer = np.zeros((n, n))
for e in graph.edges():
    i, j = int(e.source()), int(e.target())
    if graph.ep.layer[e] == 0:
        beh_layer[i, j] = beh_layer[j, i] = graph.ep.behaviour_weight[e]
    else:
        occ_layer[i, j] = occ_layer[j, i] = graph.ep.cooccurrence_weight[e]


plt.subplot(1,2,1)
plt.imshow(np.where(occ_layer==0,np.nan,occ_layer),cmap='plasma')
plt.colorbar()
plt.title('occurence layer')
plt.subplot(1,2,2)
plt.imshow(np.where(beh_layer==0,np.nan,beh_layer),cmap='plasma')
plt.colorbar()
plt.title('behaviour layer')
plt.show()


real_results = fit_nested_sbm_layered(
    graph,
    mcmc_samples=10000,
    burn_in=100,
    annealing_temps=(1, 5),
    annealing_steps=1
)

edge_var = real_results['edge_prob_var']   # (n_nodes x n_nodes) posterior variance of edge rates

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
plt.savefig(f'/mnt/h/RT/data/figures/RF_weighted_nested_entropy_trajectory_50_{score}.png', dpi=150, bbox_inches='tight')
print(f"\nEntropy trajectory saved to: /mnt/h/RT/data/figures/RF_weighted_nested_entropy_trajectory_50_{score}.png")
plt.show()

# ---- Visualization of nested block structure ---- #
print("\nGenerating nested block structure visualization...")

# Compute layout on the weighted graph
pos = gt.sfdp_layout(g, cooling_step=0.99, epsilon=1e-3, max_iter=100)

# Extract unique location names (without _L/_R suffix)
unique_location_names = sorted(set([loc.rsplit('_', 1)[0] for loc in locations]))
unique_location_names = sorted(set(locations))
location_name_to_idx = {name: idx for idx, name in enumerate(unique_location_names)}
n_locations = len(unique_location_names)

# Convert func_loc_colours (named colors) to RGB tuples
from matplotlib.colors import to_rgba
distinct_colors = [to_rgba(color) for color in loc_colours]

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
    location_name = location#.rsplit('_', 1)[0]
    # side = location.rsplit('_', 1)[1]  # 'L' or 'R'
    location_idx = location_name_to_idx[location_name]
    rgba = cmap(norm(location_idx))
    vertex_color[v] = rgba
    # Set shape: 0 for circle (left), 1 for triangle (right)
    # vertex_shape[v] = 0 if side == 'L' else 1

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
    output=f"/mnt/h/RT/data/figures/RF_weighted_nested_block_state_draw_{score}.png",
    output_size=(1200, 1200)
)
print(f"Nested block state visualization saved to: /mnt/h/RT/data/figures/RF_weighted_nested_block_state_draw_{score}.png")

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
    n_nodes = np.sum(np.array(locations) == location_name)
    marker = 'o' if location_name.endswith('_L') else '^'
    element = Line2D([0], [0], marker=marker, color='w',
                     markerfacecolor=rgba, markersize=10,
                     markeredgecolor='black', markeredgewidth=1.5,
                     label=f"{location_name} ({n_nodes} nodes)")
    legend_elements.append(element)

ax.legend(handles=legend_elements,
          loc='center', fontsize=10, frameon=True,
          title="Node Color by Location (circles=L, triangles=R)",
          title_fontsize=12, ncol=2,
          fancybox=True, shadow=True)
ax.axis('off')
plt.tight_layout()
plt.savefig(f'/mnt/h/RT/data/figures/RF_weighted_nested_location_legend_{score}.png', dpi=150, bbox_inches='tight')
print("Location legend saved to: /mnt/h/RT/data/figures/RF_weighted_nested_location_legend.png")
plt.show()

# # ---- Print location distribution summary ---- #
# print("\nLocation Distribution Summary:")
# print(f"Total unique locations: {n_locations}")
# for location_name in unique_location_names:
#     nodes_in_location_l = np.where(np.array(locations) == f'{location_name}_L')[0]
#     nodes_in_location_r = np.where(np.array(locations) == f'{location_name}_R')[0]
#     print(f"  {location_name}_L: {len(nodes_in_location_l)} nodes (circles)")
#     print(f"  {location_name}_R: {len(nodes_in_location_r)} nodes (triangles)")
