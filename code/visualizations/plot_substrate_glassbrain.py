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
# Single nilearn glass-brain figure showing all synthetic substrate     #
# masks (SUBSTRATES/substrate_XX.nii.gz, see substrate_gen.py) at once, #
# each coloured by its own trajectory index on the plasma colormap --   #
# substrate_00 (furthest from the lesion peak) darkest/lowest, the      #
# last substrate (at the peak) brightest/highest. Unlike the cortical   #
# surface projection (plot_block_surface), a glass brain renders the    #
# full volumetric extent, so it doesn't matter whether a substrate      #
# sits in cortex, white matter or a subcortical structure.              #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import glob
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Glass-brain figure of all synthetic substrate masks, '
                                            'colour-coded by trajectory index (plasma).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--substrate_dir', type=str, default='SUBSTRATES',
                  help='Name of the folder (inside data_path) containing substrate mask NIfTIs '
                       '(default: SUBSTRATES)')
args.add_argument('--substrate_glob', type=str, default='substrate_*.nii.gz',
                  help='Glob (within substrate_dir) selecting substrate mask NIfTIs, in trajectory '
                       'order (default: substrate_*.nii.gz)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory for the PNG (default: {data_path}/{substrate_dir}/FIGURES)')
args.add_argument('--output', type=str, default='substrate_glassbrain_all.png',
                  help='Output filename (default: substrate_glassbrain_all.png)')
args = args.parse_args()

substrate_dir = os.path.join(args.data_path, args.substrate_dir)
out_dir       = args.out_dir or os.path.join(substrate_dir, 'FIGURES')
os.makedirs(out_dir, exist_ok=True)

substrate_paths = sorted(glob.glob(os.path.join(substrate_dir, args.substrate_glob)))
if not substrate_paths:
    raise SystemExit(f'No substrate masks found in {substrate_dir} matching {args.substrate_glob}')
n_substrates = len(substrate_paths)


#################################
#   BUILD COMBINED VOLUME       #
#################################

# Background = 0, substrate i's value = i+1 (1..n). nilearn's glass-brain
# resampling silently replaces NaN background with 0 (see the "Non-finite
# values detected" warning), which previously collided with substrate_00's
# own colour value of 0 and washed the whole brain in plasma's dark-blue
# zero-colour. Shifting substrates to 1..n and thresholding out values
# <= 0.5 below keeps the true background out of the colour-mapped range
# entirely, leaving it white.
ref_img  = nib.load(substrate_paths[0])
combined = np.zeros(ref_img.shape, dtype=np.float32)

n_overlap = 0
for i, sp in enumerate(substrate_paths):
    data = np.asarray(nib.load(sp).dataobj) > 0
    n_overlap += int(np.sum((combined > 0) & data))
    combined[data] = i + 1

if n_overlap > 0:
    print(f'WARNING: {n_overlap} voxel(s) belong to more than one substrate '
          f'-- later substrates (higher index) take precedence')

combined_img = nib.Nifti1Image(combined, ref_img.affine, ref_img.header)


#################################
#       GLASS BRAIN PLOT        #
#################################

out_path = os.path.join(out_dir, args.output)

fig = plt.figure(figsize=(14, 5))
display = plotting.plot_glass_brain(
    combined_img,
    figure=fig,
    display_mode='ortho',
    cmap='plasma',
    vmin=1,
    vmax=n_substrates,
    threshold=0.5,           # hides only the true (0-valued) background
    black_bg=False,
    plot_abs=False,
    symmetric_cbar=False,
    resampling_interpolation='nearest',
    colorbar=True,
    title=f'Synthetic substrate trajectory (1 = furthest, {n_substrates} = lesion peak)',
)
display.savefig(out_path, dpi=150)
display.close()
print(f'Saved -> {out_path}')
