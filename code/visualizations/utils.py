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
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

import graph_tool.all as gt

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh, vol_to_surf


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

def setup_views_figure(n_views=3, figsize=None):
    """Create a 1xN grid of 3-D axes, one per anatomical view."""
    if figsize is None:
        figsize = (6 * n_views, 6)
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
#   scalar volume -> surface    #
#################################

def project_to_surface(img, mesh_name='fsaverage5', surface='pial', interpolation='linear'):
    """Project a scalar nifti volume onto both hemispheres of a surface mesh.

    Returns {hemi: (vertices, faces, projected_values)}.
    """
    fsaverage = fetch_surf_fsaverage(mesh=mesh_name)
    projected = {}
    for hemi in ('left', 'right'):
        mesh_path = fsaverage[f'{surface}_{hemi}']
        mesh = load_surf_mesh(mesh_path)
        values = vol_to_surf(img, mesh_path, interpolation=interpolation)
        projected[hemi] = (mesh.coordinates, mesh.faces, values)
    return projected


def plot_surface_scalar(ax, vertices, faces, values, cmap='viridis', vmin=None, vmax=None,
                         background_color=(0.8, 0.8, 0.8), background_opacity=0.05, opacity=0.9,
                         positive_only=False):
    """Colour surface faces by the mean of their vertices' scalar `values`.

    Faces with no data (NaN — e.g. medial wall or non-cortical parcels)
    are drawn as translucent background cortex; all other faces are
    colour-mapped regardless of sign, unless `positive_only` is set, in
    which case zero/negative faces are treated as background too.
    `vmin`/`vmax` default to the data's own (colour-mapped) range if not
    given.
    """
    values = np.asarray(values)
    face_vals = np.nanmean(values[faces], axis=1)
    valid = ~np.isnan(face_vals)
    if positive_only:
        valid &= face_vals > 0

    cmap_obj = cm.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(face_vals[valid]) if valid.any() else 0
    if vmax is None:
        vmax = np.nanmax(face_vals[valid]) if valid.any() else 1
    norm = Normalize(vmin=vmin, vmax=vmax)

    face_rgba = np.empty((len(faces), 4))
    if valid.any():
        face_rgba[valid] = [(*cmap_obj(norm(v))[:3], opacity) for v in face_vals[valid]]
    face_rgba[~valid] = (*background_color, background_opacity)

    poly = Poly3DCollection(vertices[faces], zsort='average')
    poly.set_facecolors(face_rgba)
    poly.set_edgecolors('none')
    ax.add_collection3d(poly)

    return cmap_obj, norm


def plot_block_surface(img, cmap='viridis', surface_opacity=0.05, brain_opacity=0.05, roi_opacity=0.3,
                        positive_only=False, views=None):
    """Project a scalar volume (e.g. SBM block z-scores) onto the cortical
    surface and render it across the given anatomical views.

    Takes the volume and a colormap — projection, view setup and styling
    are all handled internally so callers don't need to know about
    surface-projection details. `brain_opacity` sets the alpha of the
    no-data background cortex and `roi_opacity` the alpha of the
    colour-mapped (data) faces, mirroring the disconnectome example's
    local parameters. `surface_opacity` is accepted for parity with that
    example but currently unused. If `positive_only` is set, only values
    greater than zero are colour-mapped (rest drawn as background).
    `views` selects which of the shared VIEWS entries to render (default:
    all three -- axial, coronal, sagittal).

    Returns (fig, axes).
    """
    background_color = (0.8, 0.8, 0.8)
    views = views if views is not None else VIEWS

    hemi_surface = project_to_surface(img)
    all_verts = np.vstack([v for v, _, _ in hemi_surface.values()])
    brain_aspect = [all_verts[:, i].max() - all_verts[:, i].min() for i in range(3)]

    all_vals = np.concatenate([vals for _, _, vals in hemi_surface.values()])
    valid_vals = all_vals[~np.isnan(all_vals)]
    if positive_only:
        valid_vals = valid_vals[valid_vals > 0]
    vmin, vmax = (valid_vals.min(), valid_vals.max()) if valid_vals.size else (0, 1)

    fig, axes = setup_views_figure(n_views=len(views))
    axes = np.atleast_1d(axes)
    for ax, (view_name, elev, azim) in zip(axes, views):
        for vertices, faces, values in hemi_surface.values():
            plot_surface_scalar(ax, vertices, faces, values, cmap=cmap, vmin=vmin, vmax=vmax,
                                 background_color=background_color, background_opacity=brain_opacity,
                                 opacity=roi_opacity, positive_only=positive_only)
        finalize_view(ax, view_name, elev, azim, brain_aspect)

    return fig, axes


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


#################################
#      SBM state + legend       #
#################################

def load_joint_adjacency(graph_path):
    """Load a saved multilayer graph_tool graph and collapse its behaviour
    and cooccurrence edge weights into a single joint adjacency matrix —
    the same combined structure the final MCMC-fitted multilayer SBM was
    run on, as opposed to each layer's own separate adjacency.
    """
    g = gt.load_graph(graph_path)
    n = g.num_vertices()
    adj = np.zeros((n, n))
    for e in g.edges():
        i, j = int(e.source()), int(e.target())
        adj[i, j] = adj[j, i] = g.ep.behaviour_weight[e] + g.ep.cooccurrence_weight[e]
    return adj


def load_cooccurrence_adjacency(graph_path):
    """Load a saved lesion-only base graph (run_base.py output) into a
    plain adjacency matrix, using its single 'cooccurrence_weight' edge
    property -- the lesion-only counterpart of load_joint_adjacency, which
    expects both a behaviour_weight and cooccurrence_weight property on a
    run.py multilayer graph. run_base.py's graph never had a behaviour
    layer at all, so there is nothing to collapse.
    """
    g = gt.load_graph(graph_path)
    n = g.num_vertices()
    adj = np.zeros((n, n))
    for e in g.edges():
        i, j = int(e.source()), int(e.target())
        adj[i, j] = adj[j, i] = g.ep.cooccurrence_weight[e]
    return adj


def load_pnb_adjacency(graph_path):
    """Load a saved single-layer PseudoNormalBlockState reconstruction
    graph (run_recon_pnb.py / run_recon_pnb_strength.py output) into a
    plain adjacency matrix, using the fitted coupling-strength edge
    property ('x') as edge weight -- the single-layer counterpart of
    load_joint_adjacency, which collapses a run.py multilayer graph's two
    separate weight properties instead.
    """
    g = gt.load_graph(graph_path)
    n = g.num_vertices()
    adj = np.zeros((n, n))
    for e in g.edges():
        i, j = int(e.source()), int(e.target())
        adj[i, j] = adj[j, i] = g.ep.x[e]
    return adj


def nested_bs_from_node_levels(level_arrays):
    """Convert a list of per-node block-assignment arrays — one per
    hierarchy level, each length n_nodes (e.g. the level_0, level_1, ...
    columns of roi_block_assignments_{task}.csv) — into the block-to-block
    `bs` format graph_tool's NestedBlockState expects: bs[0] is the
    per-node level-0 assignment; each subsequent bs[k] maps every
    level-(k-1) block id to its level-k id, taken as the majority level-k
    value among that block's members (levels are assumed nested/consistent,
    though modal per-level partitions computed via posterior consensus can
    disagree slightly — majority vote resolves that).

    A trivial single root group is appended on top if the topmost given
    level isn't already a single block, since NestedBlockState requires
    the hierarchy to terminate in one group.
    """
    bs = [np.asarray(level_arrays[0], dtype=int)]
    for lower, upper in zip(level_arrays[:-1], level_arrays[1:]):
        lower = np.asarray(lower, dtype=int)
        upper = np.asarray(upper, dtype=int)
        n_blocks = lower.max() + 1
        mapping = np.zeros(n_blocks, dtype=int)
        for blk in range(n_blocks):
            members = np.where(lower == blk)[0]
            if len(members):
                vals, counts = np.unique(upper[members], return_counts=True)
                mapping[blk] = vals[np.argmax(counts)]
        bs.append(mapping)

    if bs[-1].max() > 0:
        bs.append(np.zeros(bs[-1].max() + 1, dtype=int))

    return bs


def block_edge_weight_image(adj, block_of_node, block_id, atlas_img):
    """Build a NIfTI where each ROI belonging to `block_id` carries its own
    mean edge weight to other members of that block, normalized by the
    block's own maximum (no z-scoring), and 0 elsewhere.

    `block_of_node` is an array of block IDs in the same row order as
    `adj` (block_of_node[i] is the block of node i, atlas parcel i+1).
    """
    atlas_data = np.asarray(atlas_img.dataobj)
    members = np.where(block_of_node == block_id)[0]

    weights = np.zeros(len(members))
    for idx, node in enumerate(members):
        peers = members[members != node]
        edge_vals = adj[node, peers]
        nonzero = edge_vals[edge_vals > 0]
        weights[idx] = nonzero.mean() if nonzero.size else 0.0

    max_w = weights.max()
    if max_w > 0:
        weights = weights / max_w

    data = np.zeros(atlas_data.shape, dtype=np.float32)
    for node, w in zip(members, weights):
        data[atlas_data == node + 1] = w

    return nib.Nifti1Image(data, atlas_img.affine, atlas_img.header)


def block_membership_image(block_of_node, block_id, atlas_img):
    """Build a NIfTI marking every ROI belonging to `block_id` with a
    constant value of 1 (0 elsewhere) -- pure membership, no edge weight
    or other value attached. Meant to be rendered with plot_block_surface
    (positive_only=True): since every member voxel shares the same value,
    the colormap's normalization collapses to a single flat colour, so
    membership is shown regardless of whether that block happens to have
    any intra-block edges (which block_edge_weight_image would render as
    empty, even for a real, z-score-selected block).

    `block_of_node` is an array of block IDs in the same row order as the
    atlas parcellation (block_of_node[i] is the block of node i, atlas
    parcel i+1).
    """
    atlas_data = np.asarray(atlas_img.dataobj)
    members = np.where(block_of_node == block_id)[0]

    data = np.zeros(atlas_data.shape, dtype=np.float32)
    for node in members:
        data[atlas_data == node + 1] = 1.0

    return nib.Nifti1Image(data, atlas_img.affine, atlas_img.header)


def plot_sbm_state(adj, node_groups, output_prefix, cmap='plasma', arrow_colour='black', edge_alpha=0.5,
                    min_edge_alpha=0.05, block_of_node=None, relevance=None):
    """
    Fit a single (non-layered, non-annealed) nested SBM to `adj`, then draw
    that same fitted state twice with graph_tool's state.draw() — once per
    colour scheme — so node ordering and edges (which state.draw() lays out
    according to the fitted block structure) are identical between the two
    and only the colouring differs:

    1. '{output_prefix}_blocks.svg' — nodes coloured by their SBM block
       (tab20); one representative node per block labelled with its block
       index (to avoid clutter); a legend ('{output_prefix}_blocks_legend.svg')
       maps each block colour to the majority `node_groups` entry among that
       block's members.
    2. '{output_prefix}_weights.svg' — nodes coloured AND sized by degree.
       Each edge is drawn as a genuine two-colour GRADIENT running from its
       source endpoint's own colour to its target endpoint's own colour
       (not a single flat average), where each endpoint's colour comes from
       `relevance` (a per-node behavioural-relevance score, e.g.
       region_behaviour_relevance's output) if given -- so this view
       reflects the behavioural variable rather than the raw fitted edge
       weight (which is a ROI-ROI coupling strength with no direct
       relationship to behaviour -- see project discussion). Falls back to
       a flat colour by the edge's own |weight| (the original behaviour)
       if `relevance` is None. Edge transparency is always driven by
       |weight| regardless (see edge_alpha/min_edge_alpha below) -- only
       the colour channel changes.

    `node_groups` must be an array/list of group labels in the same row
    order as `adj` (node_groups[i] is the group of node i).

    If `block_of_node` is given, that partition is used as-is instead of
    independently re-fitting a new nested SBM — so the block IDs
    drawn/labelled are the caller's own original IDs rather than a fresh
    0..n_blocks-1 relabelling from an unrelated fit. It may be either a
    single array of per-node level-0 block IDs (in the same row order as
    `adj`; wrapped in a trivial single-group root level), or a list of
    such arrays, one per hierarchy level (level_0, level_1, ... — see
    `nested_bs_from_node_levels`), to also show the higher-level hierarchy
    overlay.

    `arrow_colour` (default 'black') sets the colour of the hierarchy overlay
    that state.draw() adds on top of the base graph — the block marker glyphs
    and the connectors ("arrows"/"rectangles") from each node to its block.

    Edge transparency scales with edge weight rather than being constant:
    `edge_alpha` (default 0.5) is the alpha of the highest-weight edge and
    `min_edge_alpha` (default 0.05) the alpha of the lowest-weight edge, so
    higher-weight edges are less transparent than lower-weight ones, in
    both plots.

    Returns the graph_tool NestedBlockState (fitted, or wrapping the given
    `block_of_node` partition).
    """
    node_groups = np.asarray(node_groups)

    # edge presence is sign-agnostic (adj != 0, not adj > 0): run.py's
    # tractography/cooccurrence weights are always >= 0 so this is a no-op
    # there, but run_recon_pnb.py's fitted coupling strengths ('x') are
    # frequently negative (anti-correlated pairs) -- filtering to > 0 was
    # silently discarding the large majority of real edges (observed
    # ~90-99% negative in practice), producing near-empty graphs. Alpha/
    # colour scaling below uses |weight| throughout for the same reason:
    # coupling MAGNITUDE, not sign, is what should drive visual prominence.
    g = gt.Graph(directed=False)
    g.add_vertex(adj.shape[0])
    weight = g.new_edge_property('double')
    src, dst = np.where(np.triu(adj, k=1) != 0)
    for i, j in zip(src, dst):
        e = g.add_edge(i, j)
        weight[e] = adj[i, j]

    # log-scaled: coupling magnitudes here commonly span multiple orders of
    # magnitude (e.g. observed range [1.1, 9978] for one beh_weighted fit)
    # -- a linear Normalize compresses the bulk of edges toward one end of
    # the colormap, with only a few extreme outliers spanning the rest
    # (reported as "most edges bright yellow, only a few purple" in
    # plasma). LogNorm spreads colour differences evenly across the full
    # magnitude range instead.
    weights   = np.abs(np.array([weight[e] for e in g.edges()]))
    edge_norm = LogNorm(vmin=weights[weights > 0].min(), vmax=weights.max()) if weights.size else Normalize(0, 1)
    edge_weight_alpha = g.new_edge_property('double')
    for e in g.edges():
        edge_weight_alpha[e] = min_edge_alpha + (edge_alpha - min_edge_alpha) * edge_norm(abs(weight[e]))

    if block_of_node is not None:
        if isinstance(block_of_node, (list, tuple)):
            bs = nested_bs_from_node_levels(block_of_node)
        else:
            b0 = np.asarray(block_of_node, dtype=int)
            bs = [b0, np.zeros(b0.max() + 1, dtype=int)]
        state = gt.NestedBlockState(g, bs=bs)
    else:
        state = gt.minimize_nested_blockmodel_dl(g)

    # ---- work around a graph_tool crash, do not change which draw ---- #
    # function is called: draw_hierarchy() (invoked by state.draw() below)
    # calls label_self_loops(level_graph).fa.max() for every hierarchy
    # level -- on a totally edgeless level graph that .fa array is
    # zero-size, and .max() raises ValueError. A coarse/sparse partition
    # (e.g. a level with disconnected super-blocks) can genuinely produce
    # such a level. Adding one self-loop is purely a rendering-time fix
    # (it doesn't touch the fitted partition state.draw() reads from).
    for level in state.levels:
        if level.g.num_edges() == 0:
            v0 = level.g.vertex(0)
            level.g.add_edge(v0, v0)

    # ---- 1. block-coloured version + legend ---- #
    b = state.levels[0].get_blocks()
    block_of_node = np.array([b[v] for v in g.vertices()])
    unique_blocks = sorted(set(block_of_node.tolist()))

    tab_cmap    = cm.get_cmap('tab20')
    block_color = {blk: tab_cmap(i % 20) for i, blk in enumerate(unique_blocks)}

    block_fill_color = g.new_vertex_property('vector<double>')
    for v in g.vertices():
        block_fill_color[v] = block_color[b[v]]

    block_majority_group = {}
    for blk in unique_blocks:
        groups_in_block = node_groups[block_of_node == blk]
        labels, counts   = np.unique(groups_in_block, return_counts=True)
        block_majority_group[blk] = labels[np.argmax(counts)]

    # label only one representative node per block with its block index,
    # to avoid cluttering the plot
    representative_node = {}
    for v in g.vertices():
        representative_node.setdefault(b[v], v)

    vertex_text = g.new_vertex_property('string')
    for blk, v in representative_node.items():
        vertex_text[v] = str(blk)

    # edges blended from their endpoints' block colours, with weight-scaled
    # transparency (edge_gradient=[] so this explicit colour is used as-is,
    # rather than graph_tool's own vertex-to-vertex gradient)
    block_edge_color = g.new_edge_property('vector<double>')
    for e in g.edges():
        src_c, tgt_c = block_fill_color[e.source()], block_fill_color[e.target()]
        avg_rgb = [(src_c[c] + tgt_c[c]) / 2 for c in range(3)]
        block_edge_color[e] = (*avg_rgb, edge_weight_alpha[e])

    state.draw(
        vertex_fill_color=block_fill_color,
        vertex_color=block_fill_color,
        vertex_text=vertex_text,
        vertex_font_size=10,
        edge_color=block_edge_color,
        edge_gradient=[],
        hedge_color=arrow_colour,
        hvertex_fill_color=arrow_colour,
        hvertex_color=arrow_colour,
        output=f'{output_prefix}_blocks.svg',
        output_size=(800, 800),
    )

    legend_elements = [
        Patch(facecolor=block_color[blk], edgecolor='black', label=f'Block {blk}: {block_majority_group[blk]}')
        for blk in unique_blocks
    ]
    fig, ax = plt.subplots(figsize=(4, max(2, 0.3 * len(unique_blocks))))
    ax.legend(handles=legend_elements, loc='center', frameon=True, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_blocks_legend.svg', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ---- 2. degree-coloured/sized nodes, relevance- (or weight-) coloured edges ---- #
    cmap_obj = cm.get_cmap(cmap)

    degree      = (adj != 0).sum(axis=1)
    deg_norm    = Normalize(vmin=degree.min(), vmax=degree.max())
    degree_fill_color = g.new_vertex_property('vector<double>')
    degree_size       = g.new_vertex_property('double')
    min_size, max_size = 5.0, 20.0
    for v in g.vertices():
        d = deg_norm(degree[int(v)])
        degree_fill_color[v] = cmap_obj(d)
        degree_size[v]       = min_size + (max_size - min_size) * d

    if relevance is not None:
        # true two-colour gradient along each edge, from the source
        # endpoint's own relevance colour to the target's -- NOT the
        # average of the two (a flat colour). graph_tool's edge_gradient
        # property takes the literal format [stop0, r0,g0,b0,a0, stop1,
        # r1,g1,b1,a1] per edge; passing it explicitly (rather than
        # leaving it at the default []) fully replaces edge_color, and is
        # independent of vertex_fill_color (which stays degree-based here).
        relevance   = np.asarray(relevance, dtype=np.float64)
        rel_norm    = Normalize(vmin=relevance.min(), vmax=relevance.max())
        vertex_rel_color = {v: cmap_obj(rel_norm(relevance[int(v)])) for v in g.vertices()}

        edge_gradient_prop = g.new_edge_property('vector<double>')
        for e in g.edges():
            r0, g0, b0, _ = vertex_rel_color[e.source()]
            r1, g1, b1, _ = vertex_rel_color[e.target()]
            a = edge_weight_alpha[e]
            edge_gradient_prop[e] = [0, r0, g0, b0, a, 1, r1, g1, b1, a]
        edge_draw_kwargs = dict(edge_gradient=edge_gradient_prop)
    else:
        edge_color = g.new_edge_property('vector<double>')
        for e in g.edges():
            r, gg, bb, _ = cmap_obj(edge_norm(abs(weight[e])))
            edge_color[e] = (r, gg, bb, edge_weight_alpha[e])
        edge_draw_kwargs = dict(edge_color=edge_color, edge_gradient=[])

    state.draw(
        vertex_fill_color=degree_fill_color,
        vertex_size=degree_size,
        **edge_draw_kwargs,
        hedge_color=arrow_colour,
        hvertex_fill_color=arrow_colour,
        hvertex_color=arrow_colour,
        output=f'{output_prefix}_weights.svg',
        output_size=(800, 800),
    )

    return state
