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
# Surface visualisation of the Schaefer atlas coloured by              #
# hemisphere-aware anatomical region (14 colours, same scheme as        #
# plot_tracts_nilearn.py). No streamlines.                              #
#                                                                       #
# Input:  {atlas}.nii.gz       (atlas parcellation)                    #
#         {atlas}_areas.txt    (roi → region mapping)                  #
#                                                                       #
# Output: {atlas}_regions_surface.png                                   #
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

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import vol_to_surf, load_surf_mesh


#################################
#      input file paths         #
#################################

data_path  = '/mnt/h/RT/data'
atlas      = 'Schaefer2018-400'

atlas_path = os.path.join(data_path, 'ATLAS', f'{atlas}.nii.gz')
areas_path = os.path.join(data_path, 'ATLAS', f'{atlas}_areas.txt')


#################################
#   hemisphere-aware colours    #
#################################

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

# Universal 14-region mapping — consistent across all tasks and images
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

fsaverage_mesh_name = 'fsaverage5'
fsaverage_surface   = 'pial'
roi_interpolation   = 'nearest'

cortex_color        = (0.8, 0.8, 0.8)
cortex_opacity      = 0.05
roi_surface_opacity = 0.5


#################################
#  build region-coloured volume #
#################################

atlas_img  = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata().astype(int)

# Load areas file: row i → region name for atlas parcel (i+1)
areas_raw  = np.genfromtxt(areas_path, dtype=str, delimiter='\t')
areas_hdr  = areas_raw[0].tolist()
areas_data = areas_raw[1:]
region_col = areas_hdr.index('region')

roi_regions = [
    f"{areas_data[i, region_col]}_{areas_data[i, 0].split('_')[-1]}"
    for i in range(len(areas_data))
]

# Stable region → index mapping (same order as get_block_niftis.py)
seen = {}
region_index_map = {}
for reg in roi_regions:
    if reg not in seen:
        seen[reg] = True
        region_index_map[reg] = len(region_index_map) + 1

# roi_colors: index (1-based) → RGB, covering all 14 regions
roi_colors = {
    idx: REGION_COLORS[name]
    for name, idx in region_index_map.items()
    if name in REGION_COLORS
}

# Map every atlas parcel to its region index
vol_region = np.zeros(atlas_data.shape, dtype=np.int32)
for atlas_val in range(1, len(roi_regions) + 1):
    reg_name = roi_regions[atlas_val - 1]
    reg_idx  = region_index_map.get(reg_name, 0)
    if reg_idx > 0:
        vol_region[atlas_data == atlas_val] = reg_idx

region_img = nib.Nifti1Image(vol_region, atlas_img.affine, atlas_img.header)


#################################
#  surface projection + render  #
#################################

fsaverage = fetch_surf_fsaverage(mesh=fsaverage_mesh_name)

VIEWS = [
    ('axial',    90,  -90),
    ('coronal',   0,  -90),
    ('sagittal',  0,    0),
]

# Pre-project both hemispheres once
hemi_data = {}
for hemi in ('left', 'right'):
    mesh_path       = fsaverage[f'{fsaverage_surface}_{hemi}']
    mesh            = load_surf_mesh(mesh_path)
    vertices, faces = mesh.coordinates, mesh.faces
    projected       = np.round(
        vol_to_surf(region_img, mesh_path, interpolation=roi_interpolation)
    ).astype(int)
    hemi_data[hemi] = (vertices, faces, projected)

all_verts    = np.vstack([v for v, _, _ in hemi_data.values()])
brain_aspect = [all_verts[:, i].max() - all_verts[:, i].min() for i in range(3)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         subplot_kw={'projection': '3d'})
fig.patch.set_facecolor('white')

for ax, (view_name, elev, azim) in zip(axes, VIEWS):

    for hemi, (vertices, faces, projected) in hemi_data.items():

        # Background cortex
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=faces,
            color=cortex_color,
            alpha=cortex_opacity,
            shade=False,
            linewidth=0,
        )

        # ROI faces with majority-vote colour
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

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(brain_aspect)
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_title(view_name, fontsize=11)

plt.suptitle(f'{atlas}  –  anatomical regions', fontsize=14)
plt.tight_layout()

out_path = os.path.join(data_path, 'ATLAS', f'{atlas}_regions_surface.svg')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out_path}')
plt.show()
