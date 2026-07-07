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
# Identify block communities significantly related to the behavioural   #
# variable of interest and export per-block NIfTI volumes with          #
# [0,1]-rescaled behavioural z-scores for the relevant nodes.           #
#                                                                       #
# Only level-0 blocks are considered.                                   #
#                                                                       #
# usage: get_block_niftis.py --data_path /path/to/data                  #
#                            --score Foreperiod_Long_tau                 #
#                            --atlas Schaefer2018-400                    #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/07.                                              #
#                                                                       #
#                                                                       #
#########################################################################

import argparse
import os

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(
    description='Export per-block NIfTI volumes with [0,1]-rescaled behavioural '
                'z-scores for nodes significantly related to the behavioural variable.'
)
parser.add_argument('--data_path',  type=str, default='/mnt/h/RT/data',
                    help='Path to the data directory')
parser.add_argument('--score',      type=str, default='Foreperiod_Long_tau',
                    help='Behaviour score / task name')
parser.add_argument('--atlas',      type=str, default='Schaefer2018-400',
                    help='Atlas name')
parser.add_argument('--multiflip',  action='store_true', default=False,
                    help='Use the multiflip results directory')
parser.add_argument('--z_thresh',   type=float, default=0.5,
                    help='Z-score threshold for block mean and node inclusion (default: 0.5)')
args = parser.parse_args()

# ---- paths ---- #
suffix     = 'multiflip' if args.multiflip else 'singleflip'
results_dir = os.path.join(args.data_path, 'RESULTS/split_threshold',
                           f'SBM_{args.atlas}_{args.score}_{suffix}')
csv_path   = os.path.join(results_dir, f'roi_block_assignments_{args.score}.csv')
atlas_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
out_dir    = os.path.join(results_dir, 'block_niftis')
os.makedirs(out_dir, exist_ok=True)

# ---- load CSV ---- #
with open(csv_path) as fh:
    header = fh.readline().strip().split(',')

data = np.genfromtxt(csv_path, delimiter=',', dtype=str, skip_header=1)

col = {name: i for i, name in enumerate(header)}

roi_index       = data[:, col['roi_index']].astype(int)
behaviour_degree = data[:, col['behaviour_degree']].astype(float)
block_level0    = data[:, col['level_0']].astype(int)

# ---- step 1: z-score behaviour degree ---- #
bd_mean = behaviour_degree.mean()
bd_std  = behaviour_degree.std()
z_scores = (behaviour_degree - bd_mean) / bd_std if bd_std > 0 else np.zeros_like(behaviour_degree)

# ---- step 2: unique blocks ---- #
blocks = np.unique(block_level0)

# ---- step 3: subset relevant blocks (at least one node z >= threshold) ---- #
relevant_blocks = [blk for blk in blocks
                   if np.any(z_scores[block_level0 == blk] >= args.z_thresh)]
print(f"Relevant blocks (>= 1 node with z >= {args.z_thresh}): {relevant_blocks}")

if not relevant_blocks:
    print("No relevant blocks found. Exiting.")
    raise SystemExit(0)

# ---- load atlas NIfTI ---- #
atlas_img  = nib.load(atlas_path)
atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int32)

# ---- load atlas areas for region mapping ---- #
areas_path  = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt')
areas_raw   = np.genfromtxt(areas_path, dtype=str, delimiter='\t')
areas_hdr   = areas_raw[0].tolist()
areas_data  = areas_raw[1:]
region_col  = areas_hdr.index('region')

# roi_regions[i] = hemisphere-sensitive region name (e.g. "Frontal_L", "Frontal_R")
# Hemisphere is the last '_'-separated token of the ROI label (column 0)
roi_regions = [
    f"{areas_data[i, region_col]}_{areas_data[i, 0].split('_')[-1]}"
    for i in range(len(areas_data))
]

# Build a stable region → index mapping (preserving order of first appearance)
seen = {}
region_index_map = {}   # hemisphere-sensitive region name -> 1-based integer index
for reg in roi_regions:
    if reg not in seen:
        seen[reg] = True
        region_index_map[reg] = len(region_index_map) + 1

# Save global region index mapping — consistent across all blocks and tasks
global_map_path = os.path.join(out_dir, f'{args.atlas}_region_index_map.txt')
with open(global_map_path, 'w') as fh:
    fh.write('index\tregion\n')
    for reg_name, reg_idx in region_index_map.items():
        fh.write(f'{reg_idx}\t{reg_name}\n')
print(f"Region index map ({len(region_index_map)} regions) -> {global_map_path}")

# ---- steps 4 & 5: per-block NIfTIs for nodes surviving threshold ---- #
# Two volumes per block:
#   _zdeg.nii.gz    — float32, voxel = raw behaviour z-score
#   _regions.nii.gz — int32,   voxel = anatomical region index
# Parcellation convention: integer value v in the atlas NIfTI corresponds to
# roi_index v (1-indexed), i.e. node at row index v-1 in the CSV.
for blk in relevant_blocks:
    blk_mask  = (block_level0 == blk)
    blk_nodes = roi_index[blk_mask]
    blk_z     = z_scores[blk_mask]

    # step 4: retain only nodes with z-score >= threshold
    node_mask      = blk_z >= args.z_thresh
    relevant_nodes = blk_nodes[node_mask]
    relevant_z     = blk_z[node_mask]

    if len(relevant_nodes) == 0:
        print(f"Block {blk}: no nodes at or above z-threshold, skipping.")
        continue

    # step 5a: raw z-scores as voxel values — NIfTI 1
    vol_z = np.zeros(atlas_data.shape, dtype=np.float32)
    for node_idx, z_val in zip(relevant_nodes, relevant_z):
        parcel_val = int(node_idx) + 1
        vol_z[atlas_data == parcel_val] = float(z_val)

    out_path_z = os.path.join(out_dir,
                              f'{args.atlas}_lvl0_block{blk}_{args.score}_zdeg.nii.gz')
    nib.save(nib.Nifti1Image(vol_z, atlas_img.affine, atlas_img.header), out_path_z)
    print(f"Block {blk}: z-score NIfTI -> {out_path_z}")

    # step 5b: brain region index — NIfTI 2
    vol_reg  = np.zeros(atlas_data.shape, dtype=np.int32)
    reg_seen = {}   # region_index -> region_name, for the txt mapping

    for node_idx in relevant_nodes:
        parcel_val = int(node_idx) + 1
        reg_name   = roi_regions[int(node_idx)]
        reg_idx    = region_index_map[reg_name]
        vol_reg[atlas_data == parcel_val] = reg_idx
        reg_seen[reg_idx] = reg_name

    out_path_reg = os.path.join(out_dir,
                                f'{args.atlas}_lvl0_block{blk}_{args.score}_regions.nii.gz')
    nib.save(nib.Nifti1Image(vol_reg, atlas_img.affine, atlas_img.header), out_path_reg)

    # txt mapping: index <tab> region_name  (only regions present in this block)
    txt_path = os.path.join(out_dir,
                            f'{args.atlas}_lvl0_block{blk}_{args.score}_regions.txt')
    with open(txt_path, 'w') as fh:
        fh.write('index\tregion\n')
        for reg_idx, reg_name in sorted(reg_seen.items()):
            fh.write(f'{reg_idx}\t{reg_name}\n')

    print(f"Block {blk}: region NIfTI -> {out_path_reg}  ({len(reg_seen)} regions)")

# ---- full parcellation NIfTI: all blocks, voxel = block index (1-based) ---- #
vol_parc = np.zeros(atlas_data.shape, dtype=np.int32)
for node_idx, blk in zip(roi_index, block_level0):
    parcel_val = int(node_idx) + 1          # 1-indexed parcel code in atlas NIfTI
    vol_parc[atlas_data == parcel_val] = int(blk) + 1   # 1-based block index

parc_path = os.path.join(out_dir, f'{args.score}_parcellation.nii.gz')
nib.save(nib.Nifti1Image(vol_parc, atlas_img.affine, atlas_img.header), parc_path)
print(f"Parcellation NIfTI ({len(blocks)} blocks) -> {parc_path}")

print("Done.")
