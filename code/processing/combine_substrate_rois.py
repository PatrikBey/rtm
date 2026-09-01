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
# Union two (or more) existing substrate ROI masks (sub_NN_TAG.nii.gz)  #
# into a single combined mask, for seeding a genuinely JOINT tractogram #
# subset -- as opposed to gen_dist_synth.py's post-hoc multiplicative   #
# combination of two independently-generated single-ROI connectomes,   #
# which was found to leak a shared lesion-severity confound into both  #
# factors. A single tractogram/connectome seeded from the union mask   #
# gives one real connectivity measure spanning both regions, computed  #
# once, rather than a product of two separately-derived proxies.       #
#                                                                       #
# This script only creates the combined ROI mask NIfTI -- the actual   #
# tractogram subset + connectome for it (data/substrates/arise/        #
# {output_name}_subset.tck, {output_name}_Schaefer2018-400.tsv) still  #
# need to be generated externally, the same way the existing per-ROI   #
# substrates' arise/ files were. Once that connectome exists, use      #
# get_substrate_connectome_behaviour.py directly (NOT gen_dist_synth.py #
# -- there's now only one substrate, not two to multiply) to derive    #
# the per-patient behaviour score.                                     #
#                                                                       #
# usage: combine_substrate_rois.py --substrates sub_01_V1 sub_03_IFG   #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import argparse
import os
import re

import nibabel as nib
import numpy as np


#################################
#      LOGGING UTILITIES        #
#################################

def log_msg(_string):
    '''
    logging function printing date, scriptname & input string to stdout
    '''
    import datetime, sys
    print(f'{datetime.date.today().strftime("%a %B %d %H:%M:%S %Z %Y")} {str(os.path.basename(sys.argv[0]))}: {str(_string)}')


#################################
#       PARSE PARAMETERS        #
#################################

parser = argparse.ArgumentParser(
    description='Union two or more existing substrate ROI masks into a single combined mask, '
                'for seeding a genuinely joint tractogram subset/connectome (rather than '
                'post-hoc multiplying two independently-derived single-ROI scores).'
)
parser.add_argument('--substrate_dir', type=str, default='/data/patrik/RT/RTM/SUBSTRATES',
                    help='Directory containing sub_NN_TAG.nii.gz masks (default: '
                         '/data/patrik/RT/RTM/SUBSTRATES)')
parser.add_argument('--substrates', type=str, nargs='+', default=['sub_01_V1', 'sub_03_IFG'],
                    help='Substrate names to combine, matching sub_NN_TAG.nii.gz filenames '
                         '(default: sub_01_V1 sub_03_IFG)')
parser.add_argument('--output_name', type=str, default=None,
                    help='Output filename stem, written as {output_name}.nii.gz (default: '
                         'dist_{TAG1}_{TAG2}[..._TAGn], derived from --substrates)')
args = parser.parse_args()

tags = []
for s in args.substrates:
    m = re.match(r'sub_\d+_(\w+)', s)
    tags.append(m.group(1) if m else s)
output_name = args.output_name or 'dist_' + '_'.join(tags)

log_msg(f"| START | Combining ROI masks: {' + '.join(args.substrates)} -> {output_name}.nii.gz")


#################################
#      LOAD + UNION MASKS       #
#################################

ref_img = None
combined = None
for substrate in args.substrates:
    roi_path = os.path.join(args.substrate_dir, f'{substrate}.nii.gz')
    if not os.path.isfile(roi_path):
        raise SystemExit(f'No mask found for substrate "{substrate}" at {roi_path}')
    img = nib.load(roi_path)
    data = np.asarray(img.dataobj) > 0

    if ref_img is None:
        ref_img = img
        combined = np.zeros(data.shape, dtype=bool)
    elif data.shape != combined.shape:
        raise SystemExit(f'Shape mismatch: {substrate} is {data.shape}, expected {combined.shape}')
    elif not np.allclose(img.affine, ref_img.affine, atol=1e-3):
        raise SystemExit(f'Affine mismatch: {substrate} does not share the reference grid')

    overlap_with_existing = int((combined & data).sum())
    n_voxels = int(data.sum())
    combined |= data
    log_msg(f"| UPDATE | {substrate}: {n_voxels} voxels "
            f"({overlap_with_existing} overlapping already-combined voxels)")

n_combined = int(combined.sum())
log_msg(f"| UPDATE | Combined mask: {n_combined} voxels "
        f"(vs. sum of individual masks -- overlap, if any, was logged above)")


#################################
#      SAVE COMBINED MASK       #
#################################

out_path = os.path.join(args.substrate_dir, f'{output_name}.nii.gz')
nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine, ref_img.header), out_path)
log_msg(f"| UPDATE | Combined mask saved -> {out_path}")
log_msg(f"| FINISHED | Next step: generate {output_name}_subset.tck and "
        f"{output_name}_Schaefer2018-400.tsv in {args.substrate_dir}/arise/ externally "
        f"(same process used for the existing per-ROI substrates), then run "
        f"get_substrate_connectome_behaviour.py on this single combined substrate.")
