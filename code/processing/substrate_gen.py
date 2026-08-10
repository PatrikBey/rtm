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
# Generate a set of synthetic binary "substrate" masks that trace a     #
# straight-line trajectory, in real-world (mm) space, from the point    #
# furthest away from the lesion-aggregate peak to the peak itself.      #
#                                                                       #
# Definitions:                                                          #
#   peak voxel   - the voxel of highest value in the lesion-aggregate   #
#                  image (argmax), i.e. the most frequently lesioned    #
#                  location across the patient sample.                  #
#   distance      - Euclidean distance, in mm (world space via the      #
#                  NIfTI affine, not raw voxel-index space), between a  #
#                  mask's own centre of mass and the peak voxel.        #
#   furthest point - the brain-mask voxel that maximises that distance; #
#                  this is the centre of the first (index 0) mask.      #
#                                                                       #
# --n_maps evenly spaced centres (default 10) are linearly interpolated #
# in world space between the furthest point (map 0) and the peak (last  #
# map, which is therefore centred exactly on the peak and so overlaps   #
# it). At each centre, every candidate voxel is restricted to the       #
# non-zero voxels of the input brain mask, scored by an isotropic       #
# Gaussian falloff from that centre (--sigma_mm), and the --n_voxels    #
# highest-scoring candidates are binarised on -- giving a compact,      #
# roughly spherical blob of identical volume (voxel count) for every    #
# map, entirely contained within the brain mask.                        #
#                                                                       #
# Because Gaussian-weighted top-N selection is asymmetric near the      #
# brain-mask boundary, a map's actual (achieved) centre of mass can     #
# drift slightly from its intended (planned) centre -- both the planned #
# and the achieved distance to the peak are written to the output       #
# table for every map.                                                  #
#                                                                       #
# usage: substrate_gen.py --brain_mask MNI152_mask.nii.gz               #
#                         --lesion_aggregate LesionAggregate.nii.gz     #
#                         --out_dir SUBSTRATES                          #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import argparse
import csv
import os

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine


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
    description='Generate synthetic binary substrate masks tracing a straight-line '
                'trajectory (in mm space) from the point furthest from the lesion-'
                'aggregate peak to the peak itself.'
)
parser.add_argument('--brain_mask', type=str, required=True,
                    help='Path to a brain mask NIfTI; its non-zero voxels define the '
                         'set of candidate locations every synthetic mask is drawn from')
parser.add_argument('--lesion_aggregate', type=str, required=True,
                    help='Path to a lesion-aggregate NIfTI (all patient lesion masks '
                         'overlapped into one image); its peak (argmax) voxel is the '
                         'convergence target of the trajectory')
parser.add_argument('--out_dir', type=str, default='SUBSTRATES',
                    help='Output directory for the synthetic masks and distance table '
                         '(default: SUBSTRATES)')
parser.add_argument('--prefix', type=str, default='substrate',
                    help='Filename prefix for the generated NIfTIs (default: substrate)')
parser.add_argument('--n_maps', type=int, default=10,
                    help='Number of synthetic masks to generate (default: 10)')
parser.add_argument('--n_voxels', type=int, default=1000,
                    help='Target volume of every synthetic mask, in voxel count -- '
                         'identical across all masks (default: 1000)')
parser.add_argument('--sigma_mm', type=float, default=10.0,
                    help='Standard deviation (mm) of the isotropic Gaussian used to '
                         'score candidate voxels around each mask centre before '
                         'selecting the top --n_voxels (default: 10.0)')
args = parser.parse_args()

if args.n_maps < 2:
    raise SystemExit('--n_maps must be >= 2 (need at least a start and an end point)')

os.makedirs(args.out_dir, exist_ok=True)

log_msg(f"| START | Generating {args.n_maps} synthetic substrate masks")
log_msg(f"| UPDATE | Brain mask: {args.brain_mask}")
log_msg(f"| UPDATE | Lesion aggregate: {args.lesion_aggregate}")


#################################
#          LOAD IMAGES          #
#################################

mask_img  = nib.load(args.brain_mask)
mask_data = np.asarray(mask_img.dataobj)

agg_img  = nib.load(args.lesion_aggregate)
agg_data = np.asarray(agg_img.dataobj)

if mask_data.shape != agg_data.shape:
    raise SystemExit(f'Shape mismatch: brain_mask {mask_data.shape} vs '
                     f'lesion_aggregate {agg_data.shape} -- both inputs must already '
                     f'share the same voxel grid')
if not np.allclose(mask_img.affine, agg_img.affine, atol=1e-3):
    raise SystemExit('Affine mismatch between brain_mask and lesion_aggregate -- '
                     'both inputs must already share the same space')

affine   = mask_img.affine
mask_ijk = np.array(np.nonzero(mask_data > 0)).T          # (n_candidates, 3)
mask_xyz = apply_affine(affine, mask_ijk)                 # (n_candidates, 3), mm

n_candidates = mask_ijk.shape[0]
log_msg(f"| UPDATE | Brain mask candidate voxels: {n_candidates}")

if args.n_voxels > n_candidates:
    raise SystemExit(f'--n_voxels ({args.n_voxels}) exceeds the number of non-zero '
                     f'brain-mask voxels ({n_candidates})')


#################################
#        LOCATE PEAK            #
#################################

peak_ijk    = np.array(np.unravel_index(np.argmax(agg_data), agg_data.shape))
peak_xyz    = apply_affine(affine, peak_ijk)
n_peak_ties = int(np.sum(agg_data == agg_data[tuple(peak_ijk)]))

log_msg(f"| UPDATE | Lesion-aggregate peak voxel (ijk): {peak_ijk.tolist()}, "
        f"value={agg_data[tuple(peak_ijk)]:.4g}, {n_peak_ties} voxel(s) share this max value")

if mask_data[tuple(peak_ijk)] <= 0:
    log_msg(f"| WARNING | Peak voxel lies outside the non-zero brain mask -- the last "
            f"mask will get as close as the mask boundary allows but cannot literally "
            f"include the peak voxel")


#################################
#     LOCATE FURTHEST POINT     #
#################################

dist_to_peak = np.linalg.norm(mask_xyz - peak_xyz, axis=1)
start_idx    = int(np.argmax(dist_to_peak))
start_xyz    = mask_xyz[start_idx]
max_dist     = float(dist_to_peak[start_idx])

log_msg(f"| UPDATE | Furthest point (ijk): {mask_ijk[start_idx].tolist()}, "
        f"distance to peak = {max_dist:.2f} mm")


#################################
#     BUILD MASK TRAJECTORY     #
#################################

# n_maps evenly spaced centres in mm space, linearly interpolated from the
# furthest point (t=0) to the peak itself (t=1, so the last mask is centred
# exactly on the peak).
t_steps = np.linspace(0.0, 1.0, args.n_maps)
centres = start_xyz[np.newaxis, :] + t_steps[:, np.newaxis] * (peak_xyz - start_xyz)[np.newaxis, :]

rows = []
for i, centre_xyz in enumerate(centres):
    planned_dist = float(np.linalg.norm(centre_xyz - peak_xyz))

    # Gaussian falloff score of every candidate voxel around this centre;
    # the top --n_voxels candidates (by score) become the binary blob.
    d2      = np.sum((mask_xyz - centre_xyz) ** 2, axis=1)
    score   = np.exp(-0.5 * d2 / args.sigma_mm ** 2)
    top_idx = np.argpartition(-score, args.n_voxels - 1)[:args.n_voxels]

    blob_ijk = mask_ijk[top_idx]

    vol = np.zeros(mask_data.shape, dtype=np.uint8)
    vol[blob_ijk[:, 0], blob_ijk[:, 1], blob_ijk[:, 2]] = 1

    # Achieved centre of mass (equal-weight binary blob -> mean voxel position),
    # which can drift slightly from the planned centre near the mask boundary.
    achieved_ijk = blob_ijk.mean(axis=0)
    achieved_xyz = apply_affine(affine, achieved_ijk)
    achieved_dist = float(np.linalg.norm(achieved_xyz - peak_xyz))

    out_path = os.path.join(args.out_dir, f'{args.prefix}_{i:02d}.nii.gz')
    nib.save(nib.Nifti1Image(vol, affine, mask_img.header), out_path)

    rows.append({
        'map_index':          i,
        'n_voxels':           int(vol.sum()),
        'planned_distance_mm':  round(planned_dist, 4),
        'achieved_distance_mm': round(achieved_dist, 4),
        'centre_x_mm':          round(float(centre_xyz[0]), 2),
        'centre_y_mm':          round(float(centre_xyz[1]), 2),
        'centre_z_mm':          round(float(centre_xyz[2]), 2),
        'filename':            os.path.basename(out_path),
    })
    log_msg(f"| UPDATE | Map {i:02d}: planned dist={planned_dist:.2f} mm, "
            f"achieved dist={achieved_dist:.2f} mm -> {out_path}")


#################################
#      SAVE DISTANCE TABLE      #
#################################

table_path = os.path.join(args.out_dir, f'{args.prefix}_distances.tsv')
with open(table_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

log_msg(f"| UPDATE | Distance table saved -> {table_path}")
log_msg(f"| FINISHED | {args.n_maps} substrate masks saved -> {args.out_dir}")
