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
                  help='Path to the substrate distance table written by substrate_gen.py '
                       '(default: {data_path}/SUBSTRATES/substrate_distances.tsv)')
args.add_argument('--level', type=int, default=0, help='Hierarchy level to visualise (default: 0)')
args.add_argument('--substrates', type=str, nargs='+', default=None,
                  help='Substrate names to visualise (default: auto-discover substrate_* '
                       'subdirectories of synth_dir)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory for the PNGs (default: {data_path}/{synth_dir}/FIGURES)')
args.add_argument('--views', type=str, nargs='+', default=None,
                  choices=['axial', 'coronal', 'sagittal'],
                  help='Subset of views to render (default: all three)')
args = args.parse_args()

views = [v for v in utils.VIEWS if v[0] in args.views] if args.views else None
filename_suffix = ('_' + '_'.join(args.views)) if args.views else ''

synth_dir  = os.path.join(args.data_path, args.synth_dir)
out_dir    = args.out_dir or os.path.join(synth_dir, 'FIGURES')
os.makedirs(out_dir, exist_ok=True)

distances_path = args.distances_table or os.path.join(args.data_path, 'SUBSTRATES', 'substrate_distances.tsv')
with open(distances_path, newline='') as fh:
    dist_rows = list(csv.DictReader(fh, delimiter='\t'))
dist_by_substrate = {row['filename'].split('.')[0]: float(row['achieved_distance_mm']) for row in dist_rows}

substrates = args.substrates or sorted(
    os.path.basename(d) for d in glob.glob(os.path.join(synth_dir, 'substrate_*'))
    if os.path.isdir(d)
)


#####################################
#                                   #
# SURFACE-MAPPED BLOCK Z-SCORES     #
#                                   #
#####################################

surface_opacity = 0.05
brain_opacity   = 0.05
roi_opacity     = 0.3

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

    fig, axes = utils.plot_block_surface(block_img, cmap='plasma', surface_opacity=surface_opacity,
                                         brain_opacity=brain_opacity, roi_opacity=roi_opacity,
                                         views=views)

    title = f'Distance to lesion peak: {dist:.2f} mm' if dist is not None else substrate
    if is_inverted:
        title += ' (inverted score)'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'SBM_blocks_surface_{substrate}_lvl{args.level}{filename_suffix}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {out_path}')
    plt.close(fig)
