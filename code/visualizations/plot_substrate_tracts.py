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
# Streamline + fsaverage cortical-surface visualisation for the         #
# substrate tractography subsets (data/substrates/arise/*.tck), one     #
# PNG per substrate. Same nilearn + dipy + matplotlib approach as       #
# plot_tracts_nilearn.py (no fury/VTK, so it runs headless) -- adapted  #
# from a multi-region ROI atlas to a single binary substrate mask, so   #
# there's no per-region colour table or endpoint-pair mixing.           #
#                                                                       #
# Each streamline is coloured along its own length by Euclidean        #
# distance (mm) from each point to THAT SUBSTRATE'S OWN centre of      #
# mass, reversed plasma (near COM = bright yellow, far = dark purple). #
# The colour-scale ceiling is not the observed max streamline distance #
# but the theoretical maximum: the distance from this substrate's own  #
# COM to the single farthest voxel of the whole-brain mask substrate_  #
# gen.py originally drew every substrate from -- so colour reflects    #
# how close a point is to the substrate relative to how far away a    #
# point COULD possibly be in the brain, not just how far the plotted   #
# streamlines happen to reach. This ceiling differs per substrate      #
# (each has its own COM), unlike a single figure-wide constant.        #
# Rendered as a single Line3DCollection per view for tractability at   #
# up to ~15k streamlines. The substrate mask's own cortical projection  #
# is drawn as a fixed neutral-grey patch so it isn't mistaken for a    #
# point on the distance colour scale.                                  #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import re
import glob
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.tracking.streamline import Streamlines

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import vol_to_surf, load_surf_mesh

import utils


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Streamline + fsaverage surface figures for substrate '
                                            'tractography subsets.')
args.add_argument('--repo_path', type=str,
                  default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                  help='Path to the repo root (default: inferred from this script\'s location)')
args.add_argument('--substrate_dir', type=str, default=None,
                  help='Directory containing substrate_XX.nii.gz masks (default: {repo_path}/data/substrates)')
args.add_argument('--arise_dir', type=str, default=None,
                  help='Directory containing substrate_XX_subset.tck files (default: {substrate_dir}/arise)')
args.add_argument('--n_substrates', type=int, default=10,
                  help='Total number of substrates in the trajectory, for plasma colour indexing '
                       '(default: 10)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory for the PNGs (default: {arise_dir}/FIGURES)')
args.add_argument('--max_streamlines', type=int, default=3000,
                  help='Random subsample cap per substrate (fixed seed) -- matplotlib\'s '
                       'Line3DCollection does not scale to tens of thousands of streamlines '
                       'rendered 3 times (ran out of practical memory/time at ~15k). '
                       'Set to 0 to disable subsampling (default: 3000)')
args.add_argument('--brain_mask', type=str,
                  default='/data/patrik/RT/RTM/ATLAS/MNI152_icbm_T1_1mm_mask.nii.gz',
                  help='The whole-brain mask substrate_gen.py originally drew every substrate '
                       'from; its non-zero voxels define the candidate pool used to compute, '
                       'per substrate, the theoretical maximum possible distance from that '
                       'substrate\'s own centre of mass (the colour-scale ceiling)')
args.add_argument('--atlas', type=str, default=None,
                  help='Parcellation NIfTI whose per-ROI centres of mass are overlaid as scatter '
                       'points for anatomical reference (default: {repo_path}/data/ATLAS/'
                       'Schaefer2018-400.nii.gz). Pass an empty string to disable.')
args.add_argument('--substrates', type=str, nargs='+', default=None,
                  help='Substrate names to plot, e.g. sub_03_IFG (default: all sub_*_subset.tck '
                       'found in arise_dir)')
args = args.parse_args()

substrate_dir = args.substrate_dir or os.path.join(args.repo_path, 'data', 'substrates')
arise_dir     = args.arise_dir or os.path.join(substrate_dir, 'arise')
out_dir       = args.out_dir or os.path.join(arise_dir, 'FIGURES')
os.makedirs(out_dir, exist_ok=True)

tck_paths = sorted(glob.glob(os.path.join(arise_dir, 'sub_*_subset.tck')))
if not tck_paths:
    raise SystemExit(f'No sub_*_subset.tck files found in {arise_dir}')
if args.substrates:
    wanted   = set(args.substrates)
    tck_paths = [p for p in tck_paths if re.match(r'(sub_\d+_\w+)_subset\.tck', os.path.basename(p)).group(1) in wanted]
    if not tck_paths:
        raise SystemExit(f'None of --substrates {args.substrates} matched a sub_*_subset.tck in {arise_dir}')

# Whole-brain candidate pool substrate_gen.py drew every substrate from --
# used below, per substrate, to find the single farthest voxel from that
# substrate's own COM (the colour-scale ceiling), the same way substrate_
# gen.py itself found the furthest point from the lesion peak.
brain_mask_img = nib.load(args.brain_mask)
brain_mask_data = np.asarray(brain_mask_img.dataobj) > 0
brain_mask_ijk = np.array(np.nonzero(brain_mask_data)).T
brain_mask_xyz = nib.affines.apply_affine(brain_mask_img.affine, brain_mask_ijk)
print(f'Brain mask: {args.brain_mask} ({brain_mask_ijk.shape[0]} candidate voxels)')

# Per-ROI centres of mass from the parcellation, purely for anatomical
# reference (dark-grey, semi-transparent scatter) -- not part of any
# distance calculation above.
atlas_path = args.atlas if args.atlas is not None else \
    os.path.join(args.repo_path, 'data', 'ATLAS', 'Schaefer2018-400.nii.gz')
roi_com_xyz = None
if atlas_path:
    atlas_img  = nib.load(atlas_path)
    atlas_data = np.asarray(atlas_img.dataobj)
    labels     = np.unique(atlas_data)
    labels     = labels[labels > 0]
    roi_com_xyz = np.array([
        nib.affines.apply_affine(atlas_img.affine, np.array(np.nonzero(atlas_data == lbl)).T.mean(axis=0))
        for lbl in labels
    ])
    print(f'Atlas: {atlas_path} ({len(labels)} ROI centres of mass)')


#################################
#      render options           #
#################################

fsaverage_mesh_name  = 'fsaverage5'
fsaverage_surface    = 'pial'
roi_interpolation    = 'nearest_most_frequent'  # this nilearn version renamed 'nearest' to this

cortex_color         = (0.8, 0.8, 0.8)
cortex_opacity       = 0.05
roi_marker_color     = (0.3, 0.3, 0.3)   # fixed neutral colour: marks the seed region, not on the distance scale
roi_surface_opacity  = 0.25
streamline_opacity   = 0.15
streamline_linewidth = 0.6


#################################
#  pre-load fsaverage meshes    #
#################################

fsaverage = fetch_surf_fsaverage(mesh=fsaverage_mesh_name)

meshes = {}
for hemi in ('left', 'right'):
    mesh_path    = fsaverage[f'{fsaverage_surface}_{hemi}']
    mesh         = load_surf_mesh(mesh_path)
    meshes[hemi] = (mesh.coordinates, mesh.faces, mesh_path)

all_verts    = np.vstack([v for v, _, _ in meshes.values()])
brain_aspect = [all_verts[:, i].max() - all_verts[:, i].min() for i in range(3)]


#################################
#     per-substrate loop        #
#################################

for tck_path in tck_paths:
    fname = os.path.basename(tck_path)
    m = re.match(r'(sub_(\d+)_\w+)_subset\.tck', fname)
    if not m:
        print(f'Skipped {fname}: filename does not match sub_NN_TAG_subset.tck')
        continue
    substrate, idx = m.group(1), int(m.group(2))

    roi_path = os.path.join(substrate_dir, f'{substrate}.nii.gz')
    if not os.path.isfile(roi_path):
        print(f'Skipped {substrate}: no mask at {roi_path}')
        continue

    # --- load substrate mask (single binary ROI, unlike the multi-region atlas case) ---
    roi_img  = nib.load(roi_path)
    roi_data = np.asarray(roi_img.dataobj) > 0

    # centre of mass in mm, same convention as get_substrate_distances.py
    com_ijk = np.array(np.nonzero(roi_data)).T.mean(axis=0)
    com_xyz = nib.affines.apply_affine(roi_img.affine, com_ijk)

    # theoretical ceiling: farthest whole-brain-mask voxel from this substrate's own COM
    vmax_mm = float(np.linalg.norm(brain_mask_xyz - com_xyz, axis=1).max())
    print(f'{substrate}: COM (mm) = {com_xyz.round(1).tolist()}, '
          f'max possible distance within brain mask = {vmax_mm:.2f} mm')

    # --- load streamlines (.tck stores RASMM mm coords; roi_img supplies reference geometry) ---
    tractogram  = load_tractogram(tck_path, roi_img, to_space=Space.RASMM)
    streamlines = Streamlines(tractogram.streamlines)

    if args.max_streamlines and len(streamlines) > args.max_streamlines:
        rng  = np.random.default_rng(42)
        keep = rng.choice(len(streamlines), size=args.max_streamlines, replace=False)
        print(f'{substrate}: subsampling {len(streamlines)} -> {args.max_streamlines} streamlines for rendering')
        streamlines = Streamlines([streamlines[i] for i in keep])

    # --- per-point distance to this substrate's own COM ---
    # Segments (pairs of consecutive points) + a colour value per segment
    # (mean of its two endpoints' distances), pooled across every
    # streamline into one Line3DCollection per view for speed.
    all_segments  = []
    all_dist_vals = []
    for sl in streamlines:
        d = np.linalg.norm(sl - com_xyz, axis=1)
        segs = np.stack([sl[:-1], sl[1:]], axis=1)          # (n_points-1, 2, 3)
        all_segments.append(segs)
        all_dist_vals.append((d[:-1] + d[1:]) / 2)

    all_segments  = np.concatenate(all_segments, axis=0)
    all_dist_vals = np.concatenate(all_dist_vals, axis=0)
    if all_dist_vals.max() > vmax_mm:
        print(f'  NOTE: {substrate} has streamline points up to {all_dist_vals.max():.1f} mm from its '
              f'own COM, beyond the theoretical {vmax_mm:.2f} mm ceiling -- those will clip to the '
              f'darkest colour (should not happen: the brain mask should bound every streamline point)')
    norm = Normalize(vmin=0, vmax=vmax_mm)
    # reversed plasma: near COM (norm~0) -> bright yellow, far (norm~1) -> dark purple
    seg_colors = plt.get_cmap('plasma_r')(norm(all_dist_vals))

    # --- project the substrate mask onto both hemispheres ---
    hemi_data = {}
    for hemi, (vertices, faces, mesh_path) in meshes.items():
        projected = vol_to_surf(roi_img, mesh_path, interpolation=roi_interpolation) > 0
        hemi_data[hemi] = (vertices, faces, projected)

    # --- render ---
    fig, axes = utils.setup_views_figure(n_views=3)

    for ax, (view_name, elev, azim) in zip(axes, utils.VIEWS):

        for hemi, (vertices, faces, projected) in hemi_data.items():
            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces,
                color=cortex_color,
                alpha=cortex_opacity,
                shade=False,
                linewidth=0,
            )

            roi_face_mask = projected[faces].any(axis=1)
            roi_faces     = faces[roi_face_mask]
            if len(roi_faces) > 0:
                poly = Poly3DCollection(vertices[roi_faces], zsort='average')
                poly.set_facecolors([(*roi_marker_color, roi_surface_opacity)] * len(roi_faces))
                poly.set_edgecolors('none')
                ax.add_collection3d(poly)

        lc = Line3DCollection(all_segments, colors=seg_colors,
                              alpha=streamline_opacity, linewidths=streamline_linewidth)
        ax.add_collection3d(lc)

        if roi_com_xyz is not None:
            ax.scatter(roi_com_xyz[:, 0], roi_com_xyz[:, 1], roi_com_xyz[:, 2],
                      color='dimgray', alpha=0.5, s=8, linewidths=0, depthshade=False)

        utils.finalize_view(ax, view_name, elev, azim, brain_aspect)

    mappable = cm.ScalarMappable(norm=norm, cmap='plasma_r')
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('Distance to substrate centre of mass (mm)', fontsize=10)

    plt.suptitle(f'{substrate} tractography subset ({len(streamlines)} streamlines, '
                f'scale ceiling {vmax_mm:.0f} mm)', fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{substrate}_tracts_surf.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {out_path}')
    plt.close(fig)
