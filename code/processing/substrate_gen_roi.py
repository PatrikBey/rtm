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
# Generate 10 "substrate" masks from REAL atlas ROIs (not synthetic     #
# Gaussian blobs, cf. substrate_gen.py) that trace the same distance-   #
# to-lesion-peak trajectory concept: sub_01 is anchored on primary      #
# visual cortex (striate cortex), the single most consistently distant  #
# ROI from the lesion-aggregate peak; the remaining nine are real ROIs  #
# picked at evenly spaced RANKS through the full peak-distance ordering #
# of every atlas parcel, from V1's own rank down to the closest-to-peak #
# parcel (sub_10).                                                      #
#                                                                       #
# Output filenames are {prefix}_{01..10}_{roi_tag}.nii.gz, e.g.         #
# sub_01_V1.nii.gz -- roi_tag is the ROI's own atlas name with its      #
# trailing _<index>_<hemisphere> suffix stripped (e.g. 'ST_2_L' ->      #
# 'ST'), except where --roi_tag_overrides substitutes a more recognisable #
# label (default: Striate -> V1).                                       #
#                                                                       #
# Each substrate mask is simply that ROI's own parcel (atlas_data ==    #
# parcel_index) -- no volume-matching across substrates, unlike the     #
# Gaussian-blob generator: these are genuine anatomical regions of      #
# whatever size they naturally are, not artificially equalised probes.  #
#                                                                       #
# Definitions (matching substrate_gen.py / get_substrate_distances.py): #
#   peak voxel - the voxel of highest value in the lesion-aggregate     #
#                image (argmax).                                        #
#   distance    - Euclidean distance, mm, world space via each image's  #
#                own affine, between an ROI's centre of mass and the    #
#                peak.                                                  #
#                                                                       #
# Downstream note: data/substrates/arise/'s connectome + .tck files     #
# were generated externally against the OLD (Gaussian-blob) substrate   #
# locations. They must be regenerated against these new ROI-based       #
# masks before get_substrate_connectome_behaviour.py / plot_substrate_  #
# tracts.py results are valid again.                                    #
#                                                                       #
# usage: substrate_gen_roi.py --out_dir SUBSTRATES                      #
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
import re

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
    description='Generate 10 real-ROI substrate masks tracing a decreasing-distance-to-'
                'lesion-peak trajectory, anchored at primary visual cortex.'
)
parser.add_argument('--repo_path', type=str,
                    default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='Path to the repo root (default: inferred from this script\'s location)')
parser.add_argument('--lesion_aggregate', type=str, default='/data/patrik/RT/RTM/LesionAggregate.nii.gz',
                    help='Path to the lesion-aggregate NIfTI; its peak (argmax) voxel is the '
                         'convergence target of the trajectory')
parser.add_argument('--atlas', type=str, default=None,
                    help='Parcellation NIfTI (default: {repo_path}/data/ATLAS/Schaefer2018-400.nii.gz)')
parser.add_argument('--atlas_areas', type=str, default=None,
                    help='Atlas areas file supplying ROI names, same row order as parcel index '
                         '(default: {repo_path}/data/ATLAS/Schaefer2018-400_areas.txt)')
parser.add_argument('--out_dir', type=str, default='/data/patrik/RT/RTM/SUBSTRATES',
                    help='Output directory for the substrate masks and distance table '
                         '(default: /data/patrik/RT/RTM/SUBSTRATES)')
parser.add_argument('--prefix', type=str, default='sub',
                    help='Filename prefix for the generated NIfTIs (default: sub)')
parser.add_argument('--roi_tag_overrides', type=str, default='Striate:V1',
                    help='Comma-separated base-name:tag pairs overriding the auto-derived '
                         'filename tag for specific ROI base names (default: "Striate:V1", '
                         'so the V1/striate-cortex substrate is tagged V1 not Striate)')
parser.add_argument('--n_maps', type=int, default=10,
                    help='Number of substrates to generate (default: 10)')
parser.add_argument('--v1_pattern', type=str, default='striate',
                    help='Case-insensitive substring identifying primary-visual-cortex ROI '
                         'candidates in atlas_areas (default: striate)')
args = parser.parse_args()

if args.n_maps < 2:
    raise SystemExit('--n_maps must be >= 2')

atlas_path       = args.atlas or os.path.join(args.repo_path, 'data', 'ATLAS', 'Schaefer2018-400.nii.gz')
atlas_areas_path = args.atlas_areas or os.path.join(args.repo_path, 'data', 'ATLAS', 'Schaefer2018-400_areas.txt')
os.makedirs(args.out_dir, exist_ok=True)

log_msg(f"| START | Generating {args.n_maps} ROI-based substrate masks")
log_msg(f"| UPDATE | Atlas: {atlas_path}")
log_msg(f"| UPDATE | Lesion aggregate: {args.lesion_aggregate}")


#################################
#          LOAD IMAGES          #
#################################

atlas_img  = nib.load(atlas_path)
atlas_data = np.asarray(atlas_img.dataobj).astype(int)

with open(atlas_areas_path, newline='') as fh:
    reader = csv.reader(fh, delimiter='\t')
    next(reader)
    roi_names = [row[0] for row in reader]
n_rois = len(roi_names)
if atlas_data.max() != n_rois:
    log_msg(f"| WARNING | atlas has max parcel value {atlas_data.max()} but atlas_areas has "
            f"{n_rois} rows -- proceeding, but double-check these correspond")

agg_img  = nib.load(args.lesion_aggregate)
agg_data = np.asarray(agg_img.dataobj)

peak_ijk = np.array(np.unravel_index(np.argmax(agg_data), agg_data.shape))
peak_xyz = apply_affine(agg_img.affine, peak_ijk)
log_msg(f"| UPDATE | Lesion-aggregate peak voxel (ijk): {peak_ijk.tolist()}, world (mm): {peak_xyz.round(2).tolist()}")


#################################
#   PER-ROI COM + PEAK DISTANCE #
#################################

roi_com_xyz = np.zeros((n_rois, 3))
roi_n_voxels = np.zeros(n_rois, dtype=int)
for i in range(n_rois):
    parcel = i + 1
    ijk = np.array(np.nonzero(atlas_data == parcel)).T
    roi_n_voxels[i] = ijk.shape[0]
    if ijk.shape[0] == 0:
        roi_com_xyz[i] = np.nan
        continue
    com_ijk = ijk.mean(axis=0)
    roi_com_xyz[i] = apply_affine(atlas_img.affine, com_ijk)

roi_dist = np.linalg.norm(roi_com_xyz - peak_xyz, axis=1)
valid = roi_n_voxels > 0
log_msg(f"| UPDATE | Computed centre of mass + peak distance for {valid.sum()}/{n_rois} ROIs "
        f"({(~valid).sum()} empty parcels skipped)")


#################################
#   V1 ANCHOR (sub_01)          #
#################################

v1_candidates = np.array([i for i in range(n_rois)
                          if valid[i] and args.v1_pattern.lower() in roi_names[i].lower()])
if v1_candidates.size == 0:
    raise SystemExit(f'No ROI name matched --v1_pattern "{args.v1_pattern}" in {atlas_areas_path}')

anchor_idx = v1_candidates[np.argmax(roi_dist[v1_candidates])]
log_msg(f"| UPDATE | Primary visual cortex candidates ({args.v1_pattern!r}): "
        f"{[roi_names[i] for i in v1_candidates]}")
log_msg(f"| UPDATE | V1 anchor (sub_01): {roi_names[anchor_idx]} (parcel {anchor_idx + 1}), "
        f"distance to peak = {roi_dist[anchor_idx]:.2f} mm")


#################################
#   RANK-BASED ROI SELECTION    #
#################################

# Every valid ROI, ordered furthest -> closest to the lesion peak.
order = np.where(valid)[0][np.argsort(-roi_dist[valid])]
anchor_rank = int(np.where(order == anchor_idx)[0][0])
log_msg(f"| UPDATE | V1 anchor's own rank by peak distance: {anchor_rank}/{valid.sum() - 1} "
        f"(0 = single furthest ROI overall, not necessarily V1 itself)")

# n_maps evenly spaced RANKS from the anchor's own rank through the very
# last (closest-to-peak) rank; round + deduplicate by nudging forward on
# collision so all n_maps selected ROIs are distinct.
target_ranks = np.linspace(anchor_rank, len(order) - 1, args.n_maps)
selected_ranks = []
last = -1
for r in target_ranks:
    r = max(int(round(r)), last + 1)
    selected_ranks.append(r)
    last = r
if selected_ranks[-1] >= len(order):
    raise SystemExit('Not enough distinct ROIs to build a non-overlapping trajectory of this length')

selected_idx = order[selected_ranks]
selected_idx[0] = anchor_idx   # exact anchor, in case rounding drifted


#################################
#     ROI NAME -> FILENAME TAG  #
#################################

tag_overrides = {}
for pair in args.roi_tag_overrides.split(','):
    pair = pair.strip()
    if not pair:
        continue
    base, tag = pair.split(':')
    tag_overrides[base.strip().lower()] = tag.strip()


def roi_tag(name):
    '''
    'Striate_1_L' -> 'Striate' -> 'V1' (via tag_overrides); 'ST_2_L' -> 'ST';
    falls back to the name unchanged if it doesn't match the trailing
    _<index>_<hemisphere> pattern.
    '''
    base = re.sub(r'_\d+_[LR]$', '', name)
    return tag_overrides.get(base.lower(), base)


#################################
#      WRITE SUBSTRATE MASKS    #
#################################

rows = []
for map_i, roi_i in enumerate(selected_idx):
    parcel = roi_i + 1
    vol = (atlas_data == parcel).astype(np.uint8)

    sub_num = map_i + 1   # 1-indexed, e.g. sub_01_V1.nii.gz
    tag = roi_tag(roi_names[roi_i])
    out_path = os.path.join(args.out_dir, f'{args.prefix}_{sub_num:02d}_{tag}.nii.gz')
    nib.save(nib.Nifti1Image(vol, atlas_img.affine, atlas_img.header), out_path)

    rows.append({
        'map_index':            map_i,
        'sub_num':              sub_num,
        'roi_tag':               tag,
        'roi_name':             roi_names[roi_i],
        'roi_index':            parcel,
        'n_voxels':             int(roi_n_voxels[roi_i]),
        'planned_distance_mm':  round(float(roi_dist[roi_i]), 4),
        'achieved_distance_mm': round(float(roi_dist[roi_i]), 4),
        'centre_x_mm':          round(float(roi_com_xyz[roi_i, 0]), 2),
        'centre_y_mm':          round(float(roi_com_xyz[roi_i, 1]), 2),
        'centre_z_mm':          round(float(roi_com_xyz[roi_i, 2]), 2),
        'filename':             os.path.basename(out_path),
    })
    log_msg(f"| UPDATE | {os.path.basename(out_path)}: {roi_names[roi_i]} (parcel {parcel}, "
            f"{int(roi_n_voxels[roi_i])} voxels), dist={roi_dist[roi_i]:.2f} mm")


#################################
#      SAVE DISTANCE TABLE      #
#################################

table_path = os.path.join(args.out_dir, f'{args.prefix}_distances.tsv')
with open(table_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

log_msg(f"| UPDATE | Distance table saved -> {table_path}")
log_msg(f"| FINISHED | {args.n_maps} ROI-based substrate masks saved -> {args.out_dir}")
