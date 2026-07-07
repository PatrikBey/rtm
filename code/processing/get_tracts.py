#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                            RT MODELLING                               #
#                                                                       #
# Extract a tractogram subset whose streamlines start and end inside    #
# the non-zero voxels of an input label/mask volume, discarding         #
# hairpin / U-turn streamlines. Pure python, no MRtrix calls.           #
#                                                                       #
# usage: get_tracts.py <input_image.nii.gz> [tractogram.tck]           #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/05.                                              #
#                                                                       #
#                                                                       #
#########################################################################

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import progress.bar
from nibabel.streamlines import Tractogram, load as load_tck, save as save_tck
from scipy.spatial import cKDTree

SEARCH_RADIUS_MM = 4.0      # radial fallback search, mirrors tck2connectome default
MAX_TURN_ANGLE_DEG = 100.0  # reject streamlines whose tangent bends back > this vs chord

parser = argparse.ArgumentParser(
    description="Extract tractogram subset with streamlines starting/ending inside a "
                "label mask, dropping U-turn/hairpin streamlines."
)
parser.add_argument('--input_image', default='test.nii.gz', help="label/mask volume (e.g. test.nii.gz)")
parser.add_argument(
    '--tractogram', default=None,
    help="tractogram .tck file (default: ${TEMPLATEDIR}/Tractograms/dTOR_2m_tractogram.tck)",
)
args = parser.parse_args()


def mask_hit(points_world, label_data, affine, inv_affine, tree, radius_mm):
    """vectorised check whether each world-space point falls on/near a non-zero label voxel"""
    vox = nib.affines.apply_affine(inv_affine, points_world)
    idx = np.round(vox).astype(int)
    shape = np.array(label_data.shape)
    in_bounds = np.all((idx >= 0) & (idx < shape), axis=1)

    hit = np.zeros(len(points_world), dtype=bool)
    clipped = np.clip(idx, 0, shape - 1)
    vals = label_data[clipped[:, 0], clipped[:, 1], clipped[:, 2]]
    hit[in_bounds] = vals[in_bounds] != 0
    missed = ~hit
    if tree is not None and np.any(missed):
        dist, _ = tree.query(points_world[missed])
        hit[missed] = dist <= radius_mm
    return hit


def is_uturn_batch(streamlines, max_angle_deg):
    return [is_uturn(s, max_angle_deg) for s in streamlines]


def is_uturn(streamline, max_angle_deg):
    chord = streamline[-1] - streamline[0]
    chord_norm = np.linalg.norm(chord)
    if chord_norm < 1e-6:
        return True
    chord_dir = chord / chord_norm
    tangents = np.diff(streamline, axis=0)
    lengths = np.linalg.norm(tangents, axis=1)
    tangents = tangents[lengths > 1e-6]
    if tangents.size == 0:
        return True
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    max_angle = np.degrees(np.arccos(np.clip(tangents @ chord_dir, -1, 1))).max()
    return max_angle > max_angle_deg


input_path = args.input_image
tractogram_path = args.tractogram or os.path.join(
    os.environ.get("TEMPLATEDIR", ""), "Tractograms", "dTOR_2m_tractogram.tck"
)

base = os.path.basename(input_path)
for ext in (".nii.gz", ".nii"):
    if base.endswith(ext):
        base = base[: -len(ext)]
        break
output_path = os.path.join(os.path.dirname(input_path) or ".", f"{base}_subset.tck")

label_img = nib.load(input_path)
label_data = np.asarray(label_img.dataobj)
affine = label_img.affine
inv_affine = np.linalg.inv(affine)

nonzero_idx = np.argwhere(label_data != 0)
tree = cKDTree(nib.affines.apply_affine(affine, nonzero_idx)) if len(nonzero_idx) else None

trk = load_tck(tractogram_path)
streamlines = trk.streamlines

# --- vectorised endpoint-in-mask filtering ---
starts = np.array([s[0] for s in streamlines])
ends = np.array([s[-1] for s in streamlines])
start_ok = mask_hit(starts, label_data, affine, inv_affine, tree, SEARCH_RADIUS_MM)
end_ok = mask_hit(ends, label_data, affine, inv_affine, tree, SEARCH_RADIUS_MM)
candidate_idx = np.nonzero(start_ok & end_ok)[0]
candidates = [streamlines[i] for i in candidate_idx]

# --- parallel u-turn filtering, only on candidates that passed the mask check ---
n_workers = os.cpu_count() or 1
n_chunks = min(len(candidates), n_workers * 4) or 1
chunk_idx = np.array_split(np.arange(len(candidates)), n_chunks)
chunks = [[candidates[i] for i in idx] for idx in chunk_idx if len(idx)]

kept = []
with progress.bar.Bar("| FILTERING U-TURNS |", max=len(chunks)) as bar:
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(is_uturn_batch, list(c), MAX_TURN_ANGLE_DEG) for c in chunks]
        for chunk, future in zip(chunks, futures):
            flags = future.result()
            kept.extend(s for s, uturn in zip(chunk, flags) if not uturn)
            bar.next()

filtered = Tractogram(kept, affine_to_rasmm=trk.tractogram.affine_to_rasmm)
save_tck(filtered, output_path)
print(f"kept {len(kept)}/{len(streamlines)} streamlines -> {output_path}")
