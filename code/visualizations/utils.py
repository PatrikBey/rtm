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
# The following script contains utility functions for use in            #
# create_figures.py                                                     #
#                                                                       #
# 1. lesion distribution                                                #
# 2. disconnectome example                                              #
# 3. graph layers                                                       #
# 4. SBM blocks                                                         #
# 5. block connectivity                                                 #
#                                                                       #
#                                                                       #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/14.                                              #
#                                                                       #
#                                                                       #
#########################################################################


import os

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh


def hex_to_rgb(hex_color):
    """Convert a '#rrggbb' (or 'rrggbb') hex string to an (r, g, b) tuple in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


#################################
#      shared view angles       #
#################################

VIEWS = [
    ('axial',    90,  -90),
    ('coronal',   0,  -90),
    ('sagittal',  0,    0),
]

#################################
#      shared colours           #
#################################

def make_cmap(colors, n=256):
    """
    Create a linear segmented colormap from a list of color names.

    :param colors: Sequence of matplotlib-recognised color names
        (e.g. ``['blue', 'white', 'red']``).
    :type colors: list[str]
    :param n: Number of discrete RGB levels in the colormap.
        Default is ``256``.
    :type n: int
    :return: A matplotlib colormap.
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap', colors, N=n
    )

#################################
#      figure / axes helpers    #
#################################

def setup_views_figure(n_views=3, figsize=(18, 6)):
    """Create a 1xN grid of 3-D axes, one per anatomical view."""
    fig, axes = plt.subplots(1, n_views, figsize=figsize,
                              subplot_kw={'projection': '3d'})
    fig.patch.set_facecolor('white')
    return fig, axes


def finalize_view(ax, view_name, elev, azim, aspect):
    """Apply the shared view angle / styling used across all figures."""
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(aspect)
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_title(view_name, fontsize=11)


#################################
#      glass cortical surface   #
#################################

def load_glass_surface(mesh_name='fsaverage5', surface='pial'):
    """Fetch fsaverage meshes for both hemispheres.

    Returns a dict {hemi: (vertices, faces)} and the bounding-box aspect
    ratio used to keep 3-D axes proportioned like the brain.
    """
    fsaverage = fetch_surf_fsaverage(mesh=mesh_name)
    meshes = {}
    for hemi in ('left', 'right'):
        mesh = load_surf_mesh(fsaverage[f'{surface}_{hemi}'])
        meshes[hemi] = (mesh.coordinates, mesh.faces)

    all_verts = np.vstack([v for v, _ in meshes.values()])
    aspect = [all_verts[:, i].max() - all_verts[:, i].min() for i in range(3)]
    return meshes, aspect


def plot_glass_surface(ax, meshes, color=(0.8, 0.8, 0.8), opacity=0.05):
    """Draw a transparent cortical 'glass' surface for both hemispheres onto ax."""
    for vertices, faces in meshes.values():
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=faces,
            color=color,
            alpha=opacity,
            shade=False,
            linewidth=0,
        )


def plot_glass_surface_2d(ax, meshes, color=(0.8, 0.8, 0.8), opacity=0.05, axes=(0, 1)):
    """Draw a flat 2-D projection of the transparent cortical 'glass' surface.

    Projects the fsaverage mesh vertices onto the two coordinate axes given
    by `axes` (default (0, 1) -> x, y / axial view) and renders the mesh
    faces as a translucent polygon collection, for overlaying behind 2-D
    graph/node plots that share the same coordinate projection.
    """
    a, b = axes
    for vertices, faces in meshes.values():
        tris = vertices[faces][:, :, (a, b)]
        poly = PolyCollection(tris, facecolor=color, edgecolor='none',
                               alpha=opacity, zorder=0)
        ax.add_collection(poly)
    ax.autoscale_view()


#################################
#      streamlines              #
#################################

def plot_tracts(ax, streamlines, colors=None, opacity=0.25, linewidth=0.8,
                 default_color=(0.8, 0.2, 0.2)):
    """Plot streamlines onto ax as individual 3-D lines.

    `colors` may be a single RGB(A) tuple applied to every streamline, a
    sequence of per-streamline colors (same length/order as `streamlines`),
    or None to fall back to `default_color`.
    """
    is_single_color = colors is None or np.ndim(colors) == 1
    color_seq = [colors or default_color] * len(streamlines) if is_single_color else colors

    for sl, color in zip(streamlines, color_seq):
        ax.plot(sl[:, 0], sl[:, 1], sl[:, 2],
                color=color,
                alpha=opacity,
                linewidth=linewidth)


#################################
#      ROI / lesion volume      #
#################################

def plot_mask(ax, mask_file, color=(0.85, 0.1, 0.1), opacity=0.6, level=0.5):
    """Render a binary ROI/lesion mask as a solid 3-D volume.

    Unlike surface-based ROI plotting (vol_to_surf projection onto a
    cortical mesh), this extracts a marching-cubes isosurface of the mask
    itself in RASMM space, so the ROI renders as its own volumetric shape
    rather than being mapped onto the cortex.
    """
    mask_img  = nib.load(mask_file) if isinstance(mask_file, (str, os.PathLike)) else mask_file
    mask_data = mask_img.get_fdata() > 0

    if not mask_data.any():
        return

    verts, faces, _, _ = marching_cubes(mask_data.astype(float), level=level)
    verts_native = nib.affines.apply_affine(mask_img.affine, verts)

    poly = Poly3DCollection(verts_native[faces], zsort='average')
    poly.set_facecolor((*color, opacity))
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)




#################################
#     CURVED LINE CONNECTION    #
#################################

def draw_curved_line(ax, x1, y1, x2, y2, color='blue', alpha=0.3, curve_factor=0.1, linewidth=1):
    """
    Draw a curved line between two points that curves through the center
    """
    # Control point at the center with some offset for curvature
    cx, cy = 0, 0  # Center of circle
    # Create parametric curve
    t = np.linspace(0, 1, 100)
    # Quadratic Bézier curve: P(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    x_curve = (1-t)**2 * x1 + 2*(1-t)*t * (cx * curve_factor) + t**2 * x2
    y_curve = (1-t)**2 * y1 + 2*(1-t)*t * (cy * curve_factor) + t**2 * y2
    ax.plot(x_curve, y_curve, color=color, alpha=alpha, linewidth=linewidth)



def plot_graph(ax, adj, coords, colours=('paleturquoise', '#D3238A'), node_size=400, min_node_size=10,
               size_exponent=3, edge_alpha=0.05, min_edge_width=0.1, max_edge_width=3, top_pct=0.2):
    """
    Plot a graph given an adjacency matrix and coordinates for each node.

    All edges are drawn, coloured by edge weight via a continuous colormap
    built from `edge_colors`. Only the top `top_pct` fraction of nodes by
    degree are scattered on top (e.g. the default 0.05 keeps the top 5%);
    all others, including unconnected (degree 0) nodes, are omitted from
    the node markers. Node marker colour scales linearly with node degree,
    via a continuous colormap built from `node_colors`; marker size scales
    with degree raised to `size_exponent` (>1 makes high-degree hub nodes
    stand out disproportionately over lower-degree ones).
    """
    adj = np.asarray(adj)

    # degree per node, excluding self-loops
    binary = (adj > 0).astype(int)
    np.fill_diagonal(binary, 0)
    degree = binary.sum(axis=1)

    connected = degree > 0
    if not connected.any():
        return

    # keep only the top `top_pct` of connected nodes by degree
    threshold = np.percentile(degree[connected], 100 * (1 - top_pct))
    keep = connected & (degree >= threshold)
    if not keep.any():
        return

    kept_degree = degree[keep]
    degree_frac = kept_degree / kept_degree.max()
    node_sizes = min_node_size + (node_size - min_node_size) * degree_frac ** size_exponent

    node_cmap = make_cmap(list(colours))
    node_norm = plt.Normalize(vmin=kept_degree.min(), vmax=kept_degree.max())
    node_color_vals = node_cmap(node_norm(kept_degree))

    # each undirected edge drawn/coloured once
    src, dst = np.where(np.triu(adj, k=1) > 0)
    if len(src) > 0:
        weights = adj[src, dst]
        edge_cmap = make_cmap(list(colours))
        edge_norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
        for i, j, w in zip(src, dst, weights):
            w_norm = edge_norm(w)
            linewidth = min_edge_width + (max_edge_width - min_edge_width) * w_norm
            draw_curved_line(ax, coords[i][0], coords[i][1], coords[j][0], coords[j][1],
                              color=edge_cmap(w_norm), alpha=edge_alpha, linewidth=linewidth)

    # Draw kept nodes only, sized and coloured by degree
    ax.scatter(coords[keep, 0], coords[keep, 1],
               color=node_color_vals, s=node_sizes, zorder=3)
