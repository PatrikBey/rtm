


import numpy, os, argparse, progress.bar
import matplotlib.pyplot as plt


# ---- use lesion loads for RT analysis ---- #


# ---- use disconnection network for RT analysis ---- #


# tmp = numpy.genfromtxt('/data/patrik/WAIS/EXAMPLE/arise/sub-40183110_lesion_bin_AAL3v1.tsv', delimiter='\t', dtype = str)
tmp = numpy.genfromtxt('/data/patrik/WAIS/EXAMPLE/arise/sub-40183110_lesion_bin_AAL3.tsv', delimiter='\t', dtype = str)

sc = tmp[1:,1:].astype(float)
rois = tmp[1:,0].tolist()

q95 = numpy.quantile(sc[sc>0], 0.95)
thsc = numpy.where(sc >= q95, q95, sc)
thsc = numpy.where(thsc<= numpy.quantile(sc[sc>0], 0.75), 0, thsc)


plt.figure(figsize=(20,20))
plt.imshow(numpy.where(thsc==0, numpy.nan, thsc), cmap='Greys')
plt.colorbar()
for xy, color, label in [(( 125,125),'darkorange','Precentral_R'),((102,102),'teal','Frontal_Sup_2_R'),((101,101),'royalblue','Frontal_Mid_R'),((103,103),'cornflowerblue','Frontal_Sup_Medial_R'),((133,133),'crimson','Supp_Motor_Area_R'),((94,94),'plum','Cingulate_Mid_R'),((78,78),'orchid','ACC_pre_R'),((80,80),'mediumvioletred','ACC_sup_R')]:
    plt.scatter(*xy, s=100, c=color, alpha=0.33)
    # plt.annotate(label, xy, textcoords='offset points', xytext=(6,0), va='center', fontsize=8)

# plt.yticks(numpy.arange(len(rois)), rois)
plt.title('Structural Connectivity Matrix')
# plt.tight_layout()
plt.show()


# Precentral_R 81
# Frontal_Sup_2_R 82
# Frontal_Mid_R 83
# Supp_Motor_Area_R 88
# Frontal_Sup_Medial_R 90
# Cingulate_Mid_R 99
# ACC_pre_R 153
# ACC_sup_R 154



# Precentral_R	4.64359656906241
# Frontal_Sup_2_R	41.7479516191963
# Frontal_Mid_2_R	10.3703703703704
# Supp_Motor_Area_R	53.2686630113876
# Frontal_Sup_Medial_R	23.1958762886598
# Cingulate_Mid_R	32.5465274625511
# ACC_pre_R	15.2777777777778
# ACC_sup_R	29.4559099437148

coords = numpy.genfromtxt('/data/patrik/WAIS/EXAMPLE/AAL3_coords.tsv', delimiter='\t', dtype = str)
rois = coords[:,1].tolist()

idx = [ rois.index(roi) for roi in ['Precentral_R', 'Frontal_Sup_2_R', 'Frontal_Mid_2_R', 'Supp_Motor_Area_R', 'Frontal_Sup_Medial_R', 'Cingulate_Mid_R', 'ACC_pre_R', 'ACC_sup_R'] ]

loads = [4.64359656906241, 41.7479516191963, 10.3703703703704, 53.2686630113876, 23.1958762886598, 32.5465274625511, 15.2777777777778, 29.4559099437148]

fig, ax = plt.subplots(figsize=(10, 8))

xy = coords[1:, 2:4].astype(float)  # all node coordinates (x, y)

# base layer: all nodes
ax.scatter(xy[:, 0], xy[:, 1], s=50, c='lightgray', alpha=0.5, zorder=2)

# highlighted nodes scaled by lesion load
roi_labels = ['Precentral_R', 'Frontal_Sup_2_R', 'Frontal_Mid_2_R', 'Supp_Motor_Area_R', 'Frontal_Sup_Medial_R', 'Cingulate_Mid_R', 'ACC_pre_R', 'ACC_sup_R']
hx, hy = coords[idx, 2].astype(float), coords[idx, 3].astype(float)
ax.scatter(hx, hy, s=50 * numpy.array(loads), c='crimson', alpha=0.5, zorder=3)
for x, y, label in zip(hx, hy, roi_labels):
    ax.annotate(label, (x, y), textcoords='offset points', xytext=(6, 0), va='center', fontsize=8)

# edges from upper triangle of thresholded adjacency matrix
rows, cols = numpy.triu_indices(thsc.shape[0], k=1)
for r, c in zip(rows, cols):
    if not numpy.isnan(thsc[r, c]) and thsc[r, c] > 0:
        ax.plot([xy[r, 0], xy[c, 0]], [xy[r, 1], xy[c, 1]],
                color='lightgrey', alpha=0.3, linewidth=0.5, zorder=1)

ax.set_title('Structural Connectivity Graph (thresholded)')
plt.tight_layout()
plt.show()

import networkx as nx

g = nx.from_numpy_array(numpy.triu(thsc))
node_labels = {i: (roi_labels[idx.index(i)] if i in idx else '') for i in g.nodes()}
node_colors = ['crimson' if i in idx else 'lightgray' for i in g.nodes()]
nx.draw_circular(g, labels=node_labels, node_color=node_colors, edge_color='lightgrey', with_labels=True, node_size=50, alpha=0.33)
plt.show()

import nilearn.plotting, nibabel
nii = nibabel.load('/data/patrik/WAIS/EXAMPLE/sub-40183110_lesion_bin.nii.gz')
nilearn.plotting.plot_connectome(thsc, node_coords=coords[1:,2:5].astype(float), node_color=node_colors, edge_threshold='80%', title='Structural Connectivity Graph (thresholded)')
nilearn.plotting.plot_glass_brain(nii, alpha = 0.25)
plt.show()