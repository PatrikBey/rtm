#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                GRAPH REPRESENTATION OF INTELLIGENCE                   #
#                                                                       #
# The following script generates visualisations of the SBM based        #
# modelling of intelligence. Visualizations include:                    #
# 1. lesion distribution                                                #
# 2. disconnectomes                                                     #
# 3. graph layers                                                       #
# 4. community structures                                               #
# 5. block connectivity                                                 #
#                                                                       #
#                                                                       #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/06.                                              #
#                                                                       #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################


from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.tracking.streamline import Streamlines

import numpy as np
import nibabel as nib
import fury




#################################
#      input file paths         #
#################################

tck_path = "combi_subset.tck"    # .tck streamline file
roi_path = "combi.nii.gz"        # nifti ROI, also used as reference space for the .tck
brain_path = "MNI152_icbm_T1_1mm_brain.nii.gz"    # full-brain reference volume (e.g. T1 / brain mask)

# label value -> hex color for each distinct ROI mask
roi_colors = {
    1: "#3d8ce6",   # cyan
    2: "#c90f0f",   # magenta
}
surface_opacity = 0.05
streamline_opacity = 0.2
brain_color = "#808080"
brain_opacity = 0.05


def hex_to_rgb(hex_color):
    """Convert a '#rrggbb' (or 'rrggbb') hex string to an (r, g, b) tuple in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


roi_colors = {label: hex_to_rgb(color) for label, color in roi_colors.items()}
brain_color = hex_to_rgb(brain_color)


#################################
#      load ROI + streamlines   #
#################################

roi_img = nib.load(roi_path)
roi_data = roi_img.get_fdata()
affine = roi_img.affine

roi_masks = {label: (roi_data == label) for label in roi_colors}

brain_img = nib.load(brain_path)
brain_data = brain_img.get_fdata()
brain_affine = brain_img.affine
brain_mask = brain_data > 0

# .tck files store coordinates in mm (RASMM) space; the ROI nifti supplies
# the reference geometry needed to interpret them.
tractogram = load_tractogram(tck_path, roi_img, to_space=Space.RASMM)
streamlines = Streamlines(tractogram.streamlines)


#################################
#  color by ROI endpoint pair   #
#################################

inv_affine = np.linalg.inv(affine)


def endpoint_label(point):
    """Return the ROI label containing this RASMM point, or None."""
    ijk = tuple(np.round(nib.affines.apply_affine(inv_affine, point)).astype(int))
    if any(idx < 0 or idx >= dim for idx, dim in zip(ijk, roi_data.shape)):
        return None
    for label, mask in roi_masks.items():
        if mask[ijk]:
            return label
    return None


def streamline_color(sl):
    """Return per-point colors for sl, or None if it doesn't connect two labeled ROIs."""
    start_label = endpoint_label(sl[0])
    end_label = endpoint_label(sl[-1])

    if start_label is None or end_label is None:
        return None

    if start_label == end_label:
        return np.tile(roi_colors[start_label], (len(sl), 1))

    c0 = np.array(roi_colors[start_label], dtype=float)
    c1 = np.array(roi_colors[end_label], dtype=float)
    mixed_color = (c0 + c1) / 2
    return np.tile(mixed_color, (len(sl), 1))


# keep only streamlines that start/end inside one of the labeled ROIs
colored = [(sl, streamline_color(sl)) for sl in streamlines]
colored = [(sl, c) for sl, c in colored if c is not None]

streamlines = Streamlines([sl for sl, _ in colored])

# concatenated (total_points, 3) array, in the same point order fury/vtk uses
# internally when flattening the list of streamlines
streamline_colors = np.vstack([c for _, c in colored])

streamlines_actor = fury.actor.line(streamlines, colors=streamline_colors, opacity=streamline_opacity)

roi_actors = [
    fury.actor.contour_from_roi(
        mask, affine=affine, color=roi_colors[label], opacity=surface_opacity
    )
    for label, mask in roi_masks.items()
]

brain_actor = fury.actor.contour_from_roi(
    brain_mask, affine=brain_affine, color=brain_color, opacity=brain_opacity
)



scene = fury.window.Scene()
scene.background((1, 1, 1))
scene.add(brain_actor)
scene.add(streamlines_actor)
for roi_actor in roi_actors:
    scene.add(roi_actor)




# interactive = False
# if interactive:
fury.window.show(scene)

# scene.zoom(1.5)
# scene.reset_clipping_range()

# fury.window.record(scene, out_path="contour_from_roi_tutorial.png", size=(600, 600))


########################################
#                                      #
#          DISCONNECTOMES              #
#                                      #
########################################


def draw_curved_line(x1, y1, x2, y2, ax, color='blue', alpha=0.3, curve_factor=0.5, center=(0,0)):
    """
    Draw a curved line between two points that curves through the center
    """
    # Control point at the center with some offset for curvature
    cx, cy = center  # Center of circle
    # Create parametric curve
    t = numpy.linspace(0, 1, 100)
    # Quadratic Bézier curve: P(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    x_curve = (1-t)**2 * x1 + 2*(1-t)*t * (cx * curve_factor) + t**2 * x2
    y_curve = (1-t)**2 * y1 + 2*(1-t)*t * (cy * curve_factor) + t**2 * y2
    ax.plot(x_curve, y_curve, color=color, alpha=alpha, linewidth=1)

def get_color_from_scaled_colormap(value, N, colormap_name='Greys'):
    """Get a color from a colormap rescaled to fit N values."""
    cmap = plt.colormaps.get_cmap(colormap_name)
    normalized_value = value / N
    return cmap(normalized_value)

def plot_disconnectome(adj_matrix, coords, title='Disconnectome', cmap = 'plasma', node_colours = None, lobes=None, save_path=None):
    """
    Plot disconnectome given adjacency matrix and coordinates.
    
    Args:
        adj_matrix: Adjacency matrix (numpy array)
        coords: List of (x,y,z) coordinates for each node
        title: Title of the plot
        cmap: Colormap to use
        node_colours: List of colors for nodes
        lobes: List of lobe/region names corresponding to nodes (for legend)
        save_path: If provided, save the plot to this path
    """
    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(111)
    ax.set_facecolor('#F1EAEF')
    # cmap = plt.get_cmap('Purples')
    # Plot nodes
    x, y = coords[:,0], coords[:,1]
    if node_colours is None:
        ax.scatter(x, y, s=75, c='black', alpha=0.75)
        node_colours = [get_color_from_scaled_colormap(i, adj_matrix.shape[0]-4, cmap) for i in range(adj_matrix.shape[0]-4)]
    else:
        # ax.scatter(x, y, s=50, c=node_colours, edgecolors='#240E3C', alpha=1)
        ax.scatter(x, y, s=125, c=node_colours, edgecolors='white', alpha=1, linewidth=3)

    # Plot edges
    num_nodes = adj_matrix.shape[0]-4
    color_counter = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adj_matrix[i, j] > 0:
                color_id = (color_counter / adj_matrix.sum()) * 256
                draw_curved_line(coords[i][0], coords[i][1], coords[j][0], coords[j][1], ax, color=node_colours[i], alpha=0.5, center=(0,0))
                # draw_curved_line(coords[i][0], coords[i][1], coords[j][0], coords[j][1], ax, color=get_color_from_scaled_colormap(color_id, adj_matrix.sum(), cmap), alpha=0.25, center=(0,0))

                color_counter += 1
    
    # Add legend if lobes are provided
    if lobes is not None:
        import matplotlib.patches as mpatches
        # Get unique lobes while preserving order
        unique_lobes = []
        for lobe in lobes:
            if lobe not in unique_lobes:
                unique_lobes.append(lobe)
        
        # Create legend patches with corresponding colors
        legend_patches = []
        for idx, lobe in enumerate(unique_lobes):
            color = node_colours[lobes.index(lobe)]
            legend_patches.append(mpatches.Patch(color=color, label=lobe))
        
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fontsize=10)
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path) if save_path else plt.show()
    plt.close()



# ---- create plots ----- #


import numpy, os, matplotlib.pyplot as plt

path = '/mnt/h/RT/data/RESULTS/split_threshold'


coords = numpy.genfromtxt(os.path.join('/mnt/h/github/rtm/data/atlas/Schaefer2018-400_coords.txt'), delimiter = ',')
coords = numpy.genfromtxt(os.path.join('/mnt/h/github/rtm/data/atlas/Schaefer2018-400_circle_coords_sorted.txt'), delimiter = '\t')


for score in ['Foreperiod_Long_tau', 'GoNoGO_tau', 'SATO_Accuracy_tau']:
    occ_layer = numpy.genfromtxt(os.path.join(path, f'SBM_Schaefer2018-400_{score}_singleflip', f'SBM_layer_{score}_cooccurrence.txt'), delimiter = ' ')
    beh_layer = numpy.genfromtxt(os.path.join(path, f'SBM_Schaefer2018-400_{score}_singleflip', f'SBM_layer_{score}_behaviour.txt'), delimiter = ' ')
    plot_disconnectome(adj_matrix = occ_layer, coords = coords[:,:2], title = 'co-occurrence', cmap = 'plasma',save_path = os.path.join(path, f'SBM_Schaefer2018-400_{score}_singleflip', f'SBM_layer_{score}_cooccurrence_disconnectome.png'))
    plot_disconnectome(adj_matrix = beh_layer, coords = coords[:,:2], title = 'behaviour', cmap = 'plasma', save_path = os.path.join(path, f'SBM_Schaefer2018-400_{score}_singleflip', f'SBM_layer_{score}_behaviour_disconnectome.png'))





########################################
#                                      #
#       SBM vs CONNECTOME              #
#                                      #
########################################

import numpy, os, matplotlib.pyplot as plt

path = '/mnt/h/RT/data/RESULTS/split_threshold'



for score in ['Foreperiod_Long_tau', 'GoNoGO_tau', 'SATO_Accuracy_tau']:
    tmp = numpy.genfromtxt(os.path.join(path, f'SBM_Schaefer2018-400_{score}_singleflip', f'Lvl0_block_connectome_{score}.tsv'), delimiter = '\t')
    plt.imshow(numpy.where(tmp==0,numpy.nan,tmp), cmap = 'plasma', interpolation = 'nearest')
    plt.title(f'Anatomical connectivity between SBM blocks | {score}', fontsize=11)
    plt.xlabel('Block index')
    plt.ylabel('Block index')
    plt.xticks(range(tmp.shape[0]))
    plt.yticks(range(tmp.shape[0]))
    plt.colorbar()
    plt.show()

