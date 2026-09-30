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
# Surface-mapped SBM block z-score figures for the synthetic-substrate  #
# fits (SBMSYNTH, run.py --score substrate_XX_dist_mm). Same rendering  #
# as create_figures.py section 4 ("SBM BLOCKS") applied to substrate    #
# fits instead of behavioural tasks: each substrate's combined block    #
# z-score NIfTI is projected onto the cortical surface (utils.          #
# plot_block_surface), titled with that substrate's own distance to    #
# the lesion-aggregate peak (from substrate_gen.py's distance table).   #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import csv
import glob
import argparse

import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import utils


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Surface-mapped SBM block z-score figures for SBMSYNTH substrate fits.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--synth_dir', type=str, default='SBMSYNTH',
                  help='Name of the folder (inside data_path) containing per-substrate SBM fits '
                       '(default: SBMSYNTH)')
args.add_argument('--distances_table', type=str, default=None,
                  help='Path to the substrate distance table written by substrate_gen_roi.py '
                       '(default: {data_path}/SUBSTRATES/sub_distances.tsv)')
args.add_argument('--level', type=int, default=0, help='Hierarchy level to visualise (default: 0)')
args.add_argument('--substrates', type=str, nargs='+', default=None,
                  help='Substrate names to visualise (default: auto-discover sub_* '
                       'subdirectories of synth_dir)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory for the PNGs (default: {data_path}/{synth_dir}/FIGURES)')
args.add_argument('--views', type=str, nargs='+', default=None,
                  choices=['axial', 'coronal', 'sagittal'],
                  help='Subset of views to render (default: all three)')
args.add_argument('--threshold', type=float, default=None,
                  help='Only render z-score voxels greater than this value (rest drawn as '
                       'background cortex); also appends "_th" to the output filename. '
                       'Default: None, i.e. show the full range')
args.add_argument('--brain_opacity', type=float, default=0.05,
                  help='Alpha of the no-data background cortex (default: 0.05)')
args.add_argument('--roi_opacity', type=float, default=0.3,
                  help='Alpha of the colour-mapped (data) faces (default: 0.3). Ignored if '
                       '--roi_opacity_split is given.')
args.add_argument('--roi_opacity_split', type=float, nargs=3, default=None,
                  metavar=('PIVOT', 'ALPHA_BELOW', 'ALPHA_ABOVE'),
                  help='Split alpha by value instead of a single --roi_opacity, e.g. '
                       '"0 0.5 1.0" makes faces below 0 half-transparent and faces >= 0 '
                       'fully opaque, independent of the colour mapping')
args.add_argument('--brain_color', type=str, default='0.8,0.8,0.8',
                  help='RGB of the no-data background cortex, comma-separated 0-1 floats '
                       '(default: 0.8,0.8,0.8 -- grey)')
args.add_argument('--shade_min', type=float, default=0.15,
                  help='Darkest a fully-shadowed face can get, as a fraction of its base '
                       'colour brightness (default: 0.15; lower = darker/more contrast, '
                       '1.0 = no shading)')
args.add_argument('--cmap', type=str, default='plasma',
                  help='Matplotlib colormap name for the data-mapped faces (default: plasma), '
                       'or a comma-separated list of 2+ hex colours (e.g. "#361a54,#993bff,'
                       '#30d6ff") to build a custom linear colormap on the fly')
args.add_argument('--cmap_range', type=float, nargs=2, default=[0.0, 1.0], metavar=('LO', 'HI'),
                  help='Truncate --cmap to this sub-range (0-1) before mapping data onto it, '
                       'e.g. "0.1 0.9" to avoid the very darkest/brightest ends of a colormap '
                       '(default: 0.0 1.0, i.e. the full colormap)')
args.add_argument('--surface_mesh', type=str, default='pial', choices=['pial', 'white', 'infl'],
                  help='fsaverage surface geometry to project onto (default: pial)')
args.add_argument('--interpolation', type=str, default='linear',
                  choices=['linear', 'nearest_most_frequent'],
                  help='vol_to_surf sampling interpolation (default: linear; '
                       'nearest_most_frequent avoids blending values across ROI borders)')
args.add_argument('--mesh_density', type=str, default='fsaverage5',
                  choices=['fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6', 'fsaverage7', 'fsaverage'],
                  help='fsaverage mesh resolution (default: fsaverage5, 10242 nodes/hemisphere; '
                       'fsaverage6 = 40962, fsaverage7/fsaverage = 163842 -- finer mesh = '
                       'smoother-looking ROI boundaries, more compute/memory')
args.add_argument('--roi_outlines', action='store_true', default=False,
                  help='Draw the parcellation\'s own ROI boundaries as thin outlines on top '
                       'of the z-score fill, independent of colour/threshold (default: off)')
args.add_argument('--outline_color', type=str, default='black',
                  help='Colour of the ROI outlines (default: black)')
args.add_argument('--outline_linewidth', type=float, default=0.3,
                  help='Line width of the ROI outlines (default: 0.3)')
args = args.parse_args()

brain_color = tuple(float(c) for c in args.brain_color.split(','))
cmap = (LinearSegmentedColormap.from_list('custom', args.cmap.split(','))
        if args.cmap.startswith('#') else args.cmap)
if tuple(args.cmap_range) != (0.0, 1.0):
    import numpy as np
    base = plt.get_cmap(cmap)
    lo, hi = args.cmap_range
    cmap = LinearSegmentedColormap.from_list(
        'truncated', base(np.linspace(lo, hi, 256)))

views = [v for v in utils.VIEWS if v[0] in args.views] if args.views else None
filename_suffix = ('_' + '_'.join(args.views)) if args.views else ''
filename_suffix += '_th' if args.threshold is not None else ''

synth_dir  = os.path.join(args.data_path, args.synth_dir)
out_dir    = args.out_dir or os.path.join(synth_dir, 'FIGURES')
os.makedirs(out_dir, exist_ok=True)

distances_path = args.distances_table or os.path.join(args.data_path, 'SUBSTRATES', 'sub_distances.tsv')
with open(distances_path, newline='') as fh:
    dist_rows = list(csv.DictReader(fh, delimiter='\t'))
dist_by_substrate = {row['filename'].split('.')[0]: float(row['achieved_distance_mm']) for row in dist_rows}
roi_name_by_substrate = {row['filename'].split('.')[0]: row['roi_name'] for row in dist_rows
                         if 'roi_name' in row}

substrates = args.substrates or sorted(
    os.path.basename(d) for d in glob.glob(os.path.join(synth_dir, 'sub_*'))
    if os.path.isdir(d)
)


#####################################
#                                   #
# SURFACE-MAPPED BLOCK Z-SCORES     #
#                                   #
#####################################

surface_opacity = 0.05
brain_opacity   = args.brain_opacity
roi_opacity     = tuple(args.roi_opacity_split) if args.roi_opacity_split else args.roi_opacity
roi_atlas_img   = (nib.load(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz'))
                   if args.roi_outlines else None)

for substrate in substrates:
    fit_dir  = os.path.join(synth_dir, substrate)
    matches  = glob.glob(os.path.join(fit_dir, f'{args.atlas}_lvl{args.level}_blockzscores_*.nii.gz'))
    if not matches:
        print(f'Skipped {substrate}: no level-{args.level} block z-score NIfTI in {fit_dir}')
        continue
    img_path = matches[0]

    # strip a trailing "_inv" (inverted-score variant, see get_substrate_distances.py)
    # to look up the substrate's own distance-to-peak, which is unaffected by that inversion
    is_inverted   = substrate.endswith('_inv')
    base_substrate = substrate[:-len('_inv')] if is_inverted else substrate
    dist = dist_by_substrate.get(base_substrate)

    block_img = nib.load(img_path)

    fig, axes = utils.plot_block_surface(block_img, cmap=cmap, surface_opacity=surface_opacity,
                                         brain_opacity=brain_opacity, roi_opacity=roi_opacity,
                                         threshold=args.threshold, views=views, brain_color=brain_color,
                                         shade_min=args.shade_min, surface_mesh=args.surface_mesh,
                                         interpolation=args.interpolation, mesh_density=args.mesh_density,
                                         roi_atlas_img=roi_atlas_img, outline_color=args.outline_color,
                                         outline_linewidth=args.outline_linewidth)

    roi_name = roi_name_by_substrate.get(base_substrate)
    label = f'{substrate} ({roi_name})' if roi_name else substrate
    title = f'{label} -- distance to lesion peak: {dist:.2f} mm' if dist is not None else label
    if is_inverted:
        title += ' (inverted score)'
    if args.threshold is not None:
        title += f' (z > {args.threshold:g})'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'SBM_blocks_surface_{substrate}_lvl{args.level}{filename_suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {out_path}')
    plt.close(fig)
