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
# Combines fury-based tractography (streamlines + ROI/brain volumes)    #
# with an fsaverage cortical surface in a single interactive VTK scene. #
#                                                                       #
# nilearn.plotting.plot_surf_stat_map() itself can't be embedded here — #
# it returns a matplotlib figure, and fury is a separate VTK renderer.  #
# Instead this script reuses the same data-prep nilearn relies on       #
# (fetch_surf_fsaverage + surface.vol_to_surf) to project the ROI       #
# labels onto the fsaverage mesh, then renders that mesh as a fury      #
# actor colored the same way, alongside the streamlines and volumes.    #
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

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import vol_to_surf, load_surf_mesh
from scipy.spatial import cKDTree

import numpy as np
import nibabel as nib
import fury
import os


#################################
#      input file paths         #
#################################

path = '/mnt/h/RT/data/RESULTS/split_threshold'
task = 'SATO_Accuracy_tau'
_files = os.listdir(os.path.join(path, f'SBM_Schaefer2018-400_{task}_singleflip', 'block_niftis'))
tck_files = [ f for f in _files if f.endswith('.tck')]
roi_files = [ f for f in _files if f.endswith('_regions.nii.gz')]

tck_path = "test.tck"     # .tck streamline file
roi_path = "test.nii.gz"         # nifti ROI (MNI space), also the reference space for the .tck

roi_path = os.path.join(path, f'SBM_Schaefer2018-400_{task}_singleflip', 'block_niftis', roi_files[0])
tck_path = os.path.join(path, f'SBM_Schaefer2018-400_{task}_singleflip', 'block_niftis', tck_files[0])
def hex_to_rgb_local(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

# Hemisphere-aware region name → RGB colour (kept as ground truth)
REGION_COLORS = {
    'Frontal_L':       hex_to_rgb_local('#4B0082'),
    'Parietal_L':      hex_to_rgb_local('#FF8C00'),
    'Cingulate_L':     hex_to_rgb_local('#C71585'),
    'Temporal_L':      hex_to_rgb_local('#B22222'),
    'Insula_L':        hex_to_rgb_local('#4682B4'),
    'Orbitofrontal_L': hex_to_rgb_local('#32CD32'),
    'Occipital_L':     hex_to_rgb_local('#48D1CC'),
    'Parietal_R':      hex_to_rgb_local('#FFD700'),
    'Cingulate_R':     hex_to_rgb_local('#FF1493'),
    'Frontal_R':       hex_to_rgb_local('#7B68EE'),
    'Temporal_R':      hex_to_rgb_local('#D77272'),
    'Orbitofrontal_R': hex_to_rgb_local('#9ACD32'),
    'Insula_R':        hex_to_rgb_local('#008080'),
    'Occipital_R':     hex_to_rgb_local('#006400'),
}

# Build roi_colors only for the indices actually present in this block's NIfTI.
# This ensures vol_to_surf values from other regions stay grey (unassigned).
txt_path = roi_path.replace('_regions.nii.gz', '_regions.txt')
region_index_map = {}   # int index → region name, for this block only
with open(txt_path) as fh:
    next(fh)            # skip header
    for line in fh:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            region_index_map[int(parts[0])] = parts[1]

roi_colors = {
    idx: REGION_COLORS[name]
    for idx, name in region_index_map.items()
    if name in REGION_COLORS
}

streamline_opacity = 0.1
streamline_linewidth = 2.5

# --- fsaverage cortical surface options ---
fsaverage_mesh_name = "fsaverage5"     # resolution: 'fsaverage5' (~10k verts/hemi), 'fsaverage', ...
fsaverage_surface = "pial"             # mesh to render: 'pial', 'white', 'infl'
roi_interpolation = "nearest"          # keep discrete labels intact when projecting onto vertices
cortex_unassigned_color = "#cccccc"    # vertices with no ROI signal
cortex_opacity = 0.05          # opacity of the base cortical surface (non-ROI faces)
roi_surface_opacity = 0.25     # opacity of the faces touching an ROI-labeled vertex


cortex_unassigned_color = hex_to_rgb_local(cortex_unassigned_color)


#################################
#      load ROI + streamlines   #
#################################

roi_img = nib.load(roi_path)
roi_data = roi_img.get_fdata()
affine = roi_img.affine

roi_masks = {label: (roi_data == label) for label in roi_colors}

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

streamlines_actor = fury.actor.line(
    streamlines, colors=streamline_colors, opacity=streamline_opacity, linewidth=streamline_linewidth
)

# note: roi_masks is still used above by endpoint_label() to color streamlines;
# the ROIs themselves are shown via the fsaverage surface projection below
# instead of a volumetric contour_from_roi actor.


#########################################
#  project ROI labels onto fsaverage    #
#########################################

# fsaverage meshes are registered to the same (MNI-aligned) template space
# roi_img is assumed to be in; vol_to_surf handles sampling the volume at
# each surface vertex. 'nearest' interpolation keeps integer ROI labels
# discrete instead of blending them across the vertex neighborhood.
fsaverage = fetch_surf_fsaverage(mesh=fsaverage_mesh_name)

cortex_actors = []
for hemi in ("left", "right"):
    mesh_path = fsaverage[f"{fsaverage_surface}_{hemi}"]
    mesh = load_surf_mesh(mesh_path)
    vertices, faces = mesh.coordinates, mesh.faces

    projected_labels = vol_to_surf(roi_img, mesh_path, interpolation=roi_interpolation)
    projected_labels = np.round(projected_labels).astype(int)

    # Actor 1: full cortex at low opacity for spatial context
    bg_colors = np.tile(cortex_unassigned_color, (len(vertices), 1)).astype(float)
    bg_actor = fury.actor.surface(vertices, faces=faces, colors=bg_colors)
    bg_actor.GetProperty().SetOpacity(cortex_opacity)
    bg_actor.GetProperty().SetAmbient(1.0)
    bg_actor.GetProperty().SetDiffuse(0.0)
    bg_actor.GetProperty().SetSpecular(0.0)
    cortex_actors.append(bg_actor)

    # Actor 2: ROI faces only at higher opacity
    roi_vertex_mask = np.zeros(len(vertices), dtype=bool)
    roi_vertex_colors = np.tile(cortex_unassigned_color, (len(vertices), 1)).astype(float)
    for label, color in roi_colors.items():
        mask = projected_labels == label
        roi_vertex_colors[mask] = color
        roi_vertex_mask |= mask

    roi_face_mask = roi_vertex_mask[faces].any(axis=1)
    roi_faces = faces[roi_face_mask]
    if len(roi_faces) > 0:
        roi_actor = fury.actor.surface(vertices, faces=roi_faces, colors=roi_vertex_colors)
        roi_actor.GetProperty().SetOpacity(roi_surface_opacity)
        roi_actor.GetProperty().SetAmbient(1.0)
        roi_actor.GetProperty().SetDiffuse(0.0)
        roi_actor.GetProperty().SetSpecular(0.0)
        cortex_actors.append(roi_actor)


#################################
#      assemble the scene       #
#################################

scene = fury.window.Scene()
scene.background((1, 1, 1))
scene.add(streamlines_actor)
for cortex_actor in cortex_actors:
    scene.add(cortex_actor)


# interactive = False
# if interactive:
fury.window.show(scene)

# scene.zoom(1.5)
# scene.reset_clipping_range()

# fury.window.record(scene, out_path="tracts_surf_combined.png", size=(600, 600))
