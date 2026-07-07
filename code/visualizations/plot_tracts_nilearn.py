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
# Streamline + surface ROI visualisation using nilearn surface          #
# functions and matplotlib 3-D rendering (no fury / VTK required).      #
#                                                                       #
# Iterates over all _regions.nii.gz files in the block_niftis dir and  #
# produces one three-view PNG per block.                                #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/07.                                              #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import re

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.tracking.streamline import Streamlines

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import vol_to_surf, load_surf_mesh


#################################
#      input file paths         #
#################################

path = '/mnt/h/RT/data/RESULTS/split_threshold'
task = 'GoNoGo_tau'

_block_dir = os.path.join(path, f'SBM_Schaefer2018-400_{task}_singleflip', 'block_niftis')
_files     = os.listdir(_block_dir)
tck_files  = [f for f in _files if f.endswith('.tck')]
roi_files  = sorted(f for f in _files if f.endswith('_regions.nii.gz'))

fig_dir = os.path.join(_block_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)


#################################
#   hemisphere-aware colours    #
#################################

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

REGION_COLORS = {
    'Frontal_L':       hex_to_rgb('#4B0082'),
    'Parietal_L':      hex_to_rgb('#FF8C00'),
    'Cingulate_L':     hex_to_rgb('#C71585'),
    'Temporal_L':      hex_to_rgb('#B22222'),
    'Insula_L':        hex_to_rgb('#4682B4'),
    'Orbitofrontal_L': hex_to_rgb('#32CD32'),
    'Occipital_L':     hex_to_rgb('#48D1CC'),
    'Parietal_R':      hex_to_rgb('#FFD700'),
    'Cingulate_R':     hex_to_rgb('#FF1493'),
    'Frontal_R':       hex_to_rgb('#7B68EE'),
    'Temporal_R':      hex_to_rgb('#D77272'),
    'Orbitofrontal_R': hex_to_rgb('#9ACD32'),
    'Insula_R':        hex_to_rgb('#008080'),
    'Occipital_R':     hex_to_rgb('#006400'),
}


#################################
#      render options           #
#################################

fsaverage_mesh_name  = 'fsaverage5'
fsaverage_surface    = 'pial'
roi_interpolation    = 'nearest'

cortex_color         = (0.8, 0.8, 0.8)
cortex_opacity       = 0.05
roi_surface_opacity  = 0.15
streamline_opacity   = 0.25
streamline_linewidth = 0.8

VIEWS = [
    ('axial',    90,  -90),
    ('coronal',   0,  -90),
    ('sagittal',  0,    0),
]


#################################
#  pre-load fsaverage meshes    #
#################################

# Vertices and faces are the same for every block; only the projected
# labels change. Load once here and reproject inside the loop.
fsaverage = fetch_surf_fsaverage(mesh=fsaverage_mesh_name)

meshes = {}
for hemi in ('left', 'right'):
    mesh_path       = fsaverage[f'{fsaverage_surface}_{hemi}']
    mesh            = load_surf_mesh(mesh_path)
    meshes[hemi]    = (mesh.coordinates, mesh.faces, mesh_path)

all_verts    = np.vstack([v for v, _, _ in meshes.values()])
brain_aspect = [all_verts[:, i].max() - all_verts[:, i].min() for i in range(3)]


#################################
#      per-block loop           #
#################################

for roi_file in roi_files:
    roi_path = os.path.join(_block_dir, roi_file)

    # Extract block number from filename for output naming and tck matching
    m = re.search(r'_block(\d+)_', roi_file)
    block_id = f'block{m.group(1)}' if m else 'block_unknown'

    # Match corresponding tck file by block number
    matching_tck = [f for f in tck_files if f'_{block_id}_' in f]
    if not matching_tck:
        print(f'No tck found for {roi_file}, skipping.')
        continue
    tck_path = os.path.join(_block_dir, matching_tck[0])

    # --- roi_colors: universal 14-region mapping filtered to this block ---
    txt_path = roi_path.replace('_regions.nii.gz', '_regions.txt')
    region_index_map = {}
    with open(txt_path) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                region_index_map[int(parts[0])] = parts[1]

    roi_colors = {
        idx: REGION_COLORS[name]
        for idx, name in region_index_map.items()
        if name in REGION_COLORS
    }

    # --- load ROI volume ---
    roi_img    = nib.load(roi_path)
    roi_data   = roi_img.get_fdata()
    affine     = roi_img.affine
    inv_affine = np.linalg.inv(affine)
    roi_masks  = {label: (roi_data == label) for label in roi_colors}

    # --- load and colour streamlines ---
    tractogram  = load_tractogram(tck_path, roi_img, to_space=Space.RASMM)
    streamlines = Streamlines(tractogram.streamlines)

    def endpoint_label(point):
        ijk = tuple(np.round(nib.affines.apply_affine(inv_affine, point)).astype(int))
        if any(i < 0 or i >= d for i, d in zip(ijk, roi_data.shape)):
            return None
        for label, mask in roi_masks.items():
            if mask[ijk]:
                return label
        return None

    colored_sl = []
    for sl in streamlines:
        s_lbl = endpoint_label(sl[0])
        e_lbl = endpoint_label(sl[-1])
        if s_lbl is None or e_lbl is None:
            continue
        if s_lbl == e_lbl:
            color = roi_colors[s_lbl]
        else:
            color = tuple((a + b) / 2
                          for a, b in zip(roi_colors[s_lbl], roi_colors[e_lbl]))
        colored_sl.append((sl, color))

    # --- project labels onto both hemispheres ---
    hemi_data = {}
    for hemi, (vertices, faces, mesh_path) in meshes.items():
        projected = np.round(
            vol_to_surf(roi_img, mesh_path, interpolation=roi_interpolation)
        ).astype(int)
        hemi_data[hemi] = (vertices, faces, projected)

    # --- render ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw={'projection': '3d'})
    fig.patch.set_facecolor('white')

    for ax, (view_name, elev, azim) in zip(axes, VIEWS):

        for hemi, (vertices, faces, projected) in hemi_data.items():

            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces,
                color=cortex_color,
                alpha=cortex_opacity,
                shade=False,
                linewidth=0,
            )

            roi_vertex_mask = np.isin(projected, list(roi_colors.keys()))
            roi_face_mask   = roi_vertex_mask[faces].any(axis=1)
            roi_faces       = faces[roi_face_mask]

            if len(roi_faces) > 0:
                fvl = projected[roi_faces]
                face_rgba = []
                for row in fvl:
                    roi_lbls = [l for l in row if l in roi_colors]
                    if roi_lbls:
                        dominant = max(set(roi_lbls), key=roi_lbls.count)
                        face_rgba.append((*roi_colors[dominant], roi_surface_opacity))
                    else:
                        face_rgba.append((*cortex_color, cortex_opacity))

                poly = Poly3DCollection(vertices[roi_faces], zsort='average')
                poly.set_facecolors(face_rgba)
                poly.set_edgecolors('none')
                ax.add_collection3d(poly)

        for sl, color in colored_sl:
            ax.plot(sl[:, 0], sl[:, 1], sl[:, 2],
                    color=color,
                    alpha=streamline_opacity,
                    linewidth=streamline_linewidth)

        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(brain_aspect)
        ax.set_facecolor('white')
        ax.axis('off')
        ax.set_title(view_name, fontsize=11)

    plt.suptitle(f'{task}  –  {block_id}', fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(fig_dir, f'{task}_{block_id}_tracts_nilearn.svg')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved → {out_path}')
    plt.close(fig)
