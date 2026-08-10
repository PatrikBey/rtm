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
# For each patient's binary lesion mask, compute the Euclidean distance #
# (mm, world space via each NIfTI's own affine) between the lesion's    #
# centre of mass and the centre of mass of every synthetic substrate    #
# mask (see substrate_gen.py), then write one new column per substrate  #
# into participants.tsv -- {substrate}_dist_mm -- filled in on the row  #
# matching that patient's participant_id. Participants with no lesion   #
# mask on disk are left blank for every new column.                     #
#                                                                       #
# Also writes an inverted counterpart per substrate --                 #
# {substrate}_dist_mm_inv = max(dist_mm over patients) - dist_mm --     #
# so that a patient whose lesion sits ON a substrate gets the HIGHEST   #
# score for that substrate, rather than the lowest. Using raw distance  #
# directly as a synthetic behaviour score means patients lesioned near  #
# the population lesion-aggregate peak (far from early/distant          #
# substrates) always get the highest score, which just reproduces the  #
# lesion-cooccurrence peak regardless of which substrate is tested; the #
# inverted score instead lets substrate-proximal lesion patterns drive  #
# the synthetic behaviour signal.                                       #
#                                                                       #
# participants.tsv is modified in place; the pre-existing file is       #
# copied to participants.tsv.bak first (overwritten on every run) so    #
# the previous version is always recoverable.                           #
#                                                                       #
# usage: get_substrate_distances.py --data_path /data/patrik/RT/RTM     #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import argparse
import csv
import glob
import os
import shutil

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
    description='Compute per-patient lesion-to-substrate centre-of-mass distances '
                'and add one column per substrate to participants.tsv.'
)
parser.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM',
                    help='Path to the data directory')
parser.add_argument('--lesion_dir', type=str, default='LESIONS',
                    help='Name of the folder (inside data_path) containing binary '
                         'lesion mask NIfTIs, one per patient (default: LESIONS)')
parser.add_argument('--substrate_dir', type=str, default='SUBSTRATES',
                    help='Name of the folder (inside data_path) containing the '
                         'synthetic substrate mask NIfTIs (default: SUBSTRATES)')
parser.add_argument('--substrate_glob', type=str, default='substrate_*.nii.gz',
                    help='Glob (within substrate_dir) selecting substrate NIfTIs '
                         '(default: substrate_*.nii.gz)')
parser.add_argument('--participants', type=str, default=None,
                    help='Path to participants.tsv (default: {data_path}/participants.tsv)')
parser.add_argument('--id_col', type=str, default='participant_id',
                    help='Column in participants.tsv holding the subject id that '
                         'matches lesion mask filenames (default: participant_id)')
args = parser.parse_args()

participants_path = args.participants or os.path.join(args.data_path, 'participants.tsv')
lesion_dir         = os.path.join(args.data_path, args.lesion_dir)
substrate_dir       = os.path.join(args.data_path, args.substrate_dir)

log_msg(f"| START | Computing lesion-to-substrate distances")
log_msg(f"| UPDATE | Lesion directory: {lesion_dir}")
log_msg(f"| UPDATE | Substrate directory: {substrate_dir}")
log_msg(f"| UPDATE | participants.tsv: {participants_path}")


#################################
#     CENTRE-OF-MASS HELPER     #
#################################

def mask_centre_of_mass_mm(nii_path):
    '''
    load a binary-mask NIfTI and return its centre of mass in world (mm)
    space: the mean voxel-index position of all non-zero voxels, mapped
    through the image's own affine.
    '''
    img  = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    ijk  = np.array(np.nonzero(data > 0)).T
    if ijk.shape[0] == 0:
        return None
    com_ijk = ijk.mean(axis=0)
    return apply_affine(img.affine, com_ijk)


#################################
#      SUBSTRATE CENTROIDS      #
#################################

substrate_paths = sorted(glob.glob(os.path.join(substrate_dir, args.substrate_glob)))
if not substrate_paths:
    raise SystemExit(f'No substrate masks found in {substrate_dir} matching '
                     f'{args.substrate_glob}')

substrate_names = []
substrate_com   = {}
for sp in substrate_paths:
    name = os.path.basename(sp).split('.')[0]
    com  = mask_centre_of_mass_mm(sp)
    if com is None:
        log_msg(f"| WARNING | Substrate {name} is empty (no non-zero voxels) -- skipping")
        continue
    substrate_names.append(name)
    substrate_com[name] = com

log_msg(f"| UPDATE | Loaded {len(substrate_names)} substrate centroids: {substrate_names}")


#################################
#      LESION CENTROIDS         #
#################################

lesion_paths = sorted(glob.glob(os.path.join(lesion_dir, '*.nii.gz')))
if not lesion_paths:
    raise SystemExit(f'No lesion masks found in {lesion_dir}')

# subject -> {substrate_name: distance_mm}
subject_distances = {}
n_empty = 0
for lp in lesion_paths:
    subject = os.path.basename(lp).split('.')[0]
    com     = mask_centre_of_mass_mm(lp)
    if com is None:
        log_msg(f"| WARNING | Lesion mask for {subject} is empty -- leaving blank")
        n_empty += 1
        continue
    subject_distances[subject] = {
        name: round(float(np.linalg.norm(com - substrate_com[name])), 6)
        for name in substrate_names
    }

log_msg(f"| UPDATE | Computed distances for {len(subject_distances)}/{len(lesion_paths)} "
        f"lesion masks ({n_empty} empty)")


#################################
#      INVERT DISTANCES         #
#################################

max_dist = {
    name: max(dists[name] for dists in subject_distances.values())
    for name in substrate_names
}
for dists in subject_distances.values():
    for name in substrate_names:
        dists[f'{name}__inv'] = round(max_dist[name] - dists[name], 6)

log_msg(f"| UPDATE | Inverted distances per substrate (max dist_mm - dist_mm): "
        f"{ {name: max_dist[name] for name in substrate_names} }")


#################################
#      UPDATE participants.tsv  #
#################################

with open(participants_path, newline='') as fh:
    reader     = csv.DictReader(fh, delimiter='\t')
    fieldnames = list(reader.fieldnames)
    rows       = list(reader)

if args.id_col not in fieldnames:
    raise SystemExit(f'Column "{args.id_col}" not found in {participants_path} '
                     f'(columns: {fieldnames})')

new_cols     = [f'{name}_dist_mm'     for name in substrate_names]
new_cols_inv = [f'{name}_dist_mm_inv' for name in substrate_names]
for col in new_cols + new_cols_inv:
    if col not in fieldnames:
        fieldnames.append(col)

n_matched = 0
for row in rows:
    subject = row[args.id_col]
    dists   = subject_distances.get(subject)
    if dists is None:
        for col in new_cols + new_cols_inv:
            row.setdefault(col, '')
        continue
    n_matched += 1
    for name, col, col_inv in zip(substrate_names, new_cols, new_cols_inv):
        row[col]     = dists[name]
        row[col_inv] = dists[f'{name}__inv']

log_msg(f"| UPDATE | Filled distance columns for {n_matched}/{len(rows)} participants.tsv rows "
        f"({len(rows) - n_matched} left blank -- no matching lesion mask)")

backup_path = participants_path + '.bak'
shutil.copy2(participants_path, backup_path)
log_msg(f"| UPDATE | Backed up existing participants.tsv -> {backup_path}")

with open(participants_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

log_msg(f"| UPDATE | participants.tsv updated with columns: {new_cols + new_cols_inv}")
log_msg(f"| FINISHED | Lesion-to-substrate distances saved -> {participants_path}")
