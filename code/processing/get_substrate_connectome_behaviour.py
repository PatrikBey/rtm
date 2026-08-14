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
# For each substrate, derive a per-patient synthetic "behaviour" score  #
# from that substrate's ROI-ROI connectome (data/substrates/arise/      #
# sub_NN_TAG_Schaefer2018-400.tsv) and each patient's per-ROI lesion     #
# overlap (Schaefer2018-400_lesion_loads.tsv), then add one new column  #
# per substrate to participants.tsv.                                    #
#                                                                       #
# Definitions:                                                          #
#   connectome scaling - each substrate's 400x400 ROI-ROI connectome    #
#     is min-max rescaled to [0,1] (globally, across the whole matrix)  #
#     before use, so edge weight is comparable in [0,1] across          #
#     substrates regardless of each connectome's own raw scale.         #
#   ROI connection strength - a substrate's scaled connectome collapsed #
#     to one value per ROI: that ROI's own row sum (weighted degree)    #
#     in the scaled matrix.                                             #
#   ROI-lesion overlap - per patient, per ROI: (lesioned voxels inside  #
#     that ROI) / (ROI's own total voxel count), in [0,1]; 0 = no       #
#     overlap, 1 = ROI fully enclosed by the lesion. Already computed   #
#     by get_lesion_loads.py as a percentage; divided by 100 here.      #
#   behaviour(patient, substrate) - mean, over only that patient's      #
#     AFFECTED ROIs (overlap > 0), of overlap[roi] * connection_        #
#     strength[roi]. Restricting the mean to affected ROIs (rather      #
#     than averaging over all 400) avoids diluting the score by lesion  #
#     size -- a patient with a small lesion hitting only strongly-      #
#     connected ROIs should not score lower than one with a large,      #
#     mostly-irrelevant lesion just because more zero-overlap ROIs      #
#     entered the average. Patients with zero affected ROIs are left    #
#     blank.                                                            #
#                                                                       #
# ROI naming differs across the three input files (connectome tsv:      #
# "17networks_LH_DefaultA_FPole_1"; lesion_loads.tsv:                   #
# "DefaultA_FPole_1_L"; atlas areas.txt: "FPole_1_L") -- all three are   #
# converted to the areas.txt convention and joined by that canonical    #
# name, not by column position, before any arithmetic.                  #
#                                                                       #
# participants.tsv is modified in place; the pre-existing file is       #
# copied to participants.tsv.bak first (overwritten on every run).      #
#                                                                       #
# usage: get_substrate_connectome_behaviour.py --data_path /data/patrik/RT/RTM #
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
import re
import shutil

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
    description='Compute per-patient, per-substrate connectome-weighted lesion-overlap '
                'behaviour scores and add one column per substrate to participants.tsv.'
)
parser.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM',
                    help='Path to the data directory')
parser.add_argument('--repo_path', type=str,
                    default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='Path to the repo root (default: inferred from this script\'s location)')
parser.add_argument('--arise_dir', type=str, default=None,
                    help='Directory containing sub_NN_TAG_Schaefer2018-400.tsv connectomes '
                         '(default: {repo_path}/data/substrates/arise)')
parser.add_argument('--lesion_loads', type=str, default=None,
                    help='Path to the per-ROI lesion-load table, percentages 0-100 '
                         '(default: {data_path}/Schaefer2018-400_lesion_loads.tsv)')
parser.add_argument('--atlas_areas', type=str, default=None,
                    help='Path to the atlas areas file defining canonical ROI names/order '
                         '(default: {repo_path}/data/ATLAS/Schaefer2018-400_areas.txt)')
parser.add_argument('--participants', type=str, default=None,
                    help='Path to participants.tsv (default: {data_path}/participants.tsv)')
parser.add_argument('--id_col', type=str, default='participant_id',
                    help='Column in participants.tsv holding the subject id that matches '
                         'lesion_loads.tsv row labels (default: participant_id)')
args = parser.parse_args()

arise_dir         = args.arise_dir or os.path.join(args.repo_path, 'data', 'substrates', 'arise')
lesion_loads_path = args.lesion_loads or os.path.join(args.data_path, 'Schaefer2018-400_lesion_loads.tsv')
atlas_areas_path  = args.atlas_areas or os.path.join(args.repo_path, 'data', 'ATLAS', 'Schaefer2018-400_areas.txt')
participants_path = args.participants or os.path.join(args.data_path, 'participants.tsv')

log_msg(f"| START | Computing substrate connectome-weighted behaviour scores")
log_msg(f"| UPDATE | Arise directory: {arise_dir}")
log_msg(f"| UPDATE | Lesion loads: {lesion_loads_path}")
log_msg(f"| UPDATE | participants.tsv: {participants_path}")


#################################
#   CANONICAL ROI NAME HELPERS  #
#################################

def canon_from_connectome(name):
    '''
    '17networks_LH_DefaultA_FPole_1' -> 'FPole_1_L' (areas.txt convention)
    '''
    parts = name.split('_')
    hemi_letter = 'L' if parts[1] == 'LH' else 'R'
    region = '_'.join(parts[3:])
    return f'{region}_{hemi_letter}'


def canon_from_lesion_loads(name):
    '''
    'DefaultA_FPole_1_L' -> 'FPole_1_L' (areas.txt convention: drop network prefix)
    '''
    return '_'.join(name.split('_')[1:])


#################################
#      CANONICAL ROI ORDER      #
#################################

with open(atlas_areas_path, newline='') as fh:
    reader = csv.reader(fh, delimiter='\t')
    next(reader)
    canonical_names = [row[0] for row in reader]
n_rois = len(canonical_names)
canon_index = {name: i for i, name in enumerate(canonical_names)}
log_msg(f"| UPDATE | Canonical ROI order: {n_rois} ROIs (from {atlas_areas_path})")


#################################
#   LOAD PER-PATIENT OVERLAP     #
#################################

with open(lesion_loads_path, newline='') as fh:
    reader = csv.reader(fh, delimiter='\t')
    ll_header = reader.__next__()[1:]
    ll_rows   = list(reader)

ll_canon_idx = np.array([canon_index[canon_from_lesion_loads(h)] for h in ll_header])

overlap_by_subject = {}
for row in ll_rows:
    subject = row[0]
    vals    = np.array([float(v) for v in row[1:]], dtype=np.float64)
    frac    = np.zeros(n_rois, dtype=np.float64)
    frac[ll_canon_idx] = vals / 100.0     # percentage -> [0,1] fraction
    overlap_by_subject[subject] = frac

log_msg(f"| UPDATE | Loaded ROI-lesion overlap for {len(overlap_by_subject)} subjects")


#################################
#  LOAD + SCALE SUBSTRATE CONNECTOMES #
#################################

connectome_paths = sorted(glob.glob(os.path.join(arise_dir, 'sub_*_Schaefer2018-400.tsv')))
if not connectome_paths:
    raise SystemExit(f'No sub_*_Schaefer2018-400.tsv connectomes found in {arise_dir}')

strength_by_substrate = {}   # substrate name -> (n_rois,) connection-strength vector, canonical order
for cp in connectome_paths:
    fname = os.path.basename(cp)
    m = re.match(r'(sub_(\d+)_\w+)_Schaefer2018-400\.tsv', fname)
    if not m:
        log_msg(f"| WARNING | Skipping {fname}: unexpected filename")
        continue
    substrate = m.group(1)

    with open(cp, newline='') as fh:
        reader = csv.reader(fh, delimiter='\t')
        header = reader.__next__()[1:]
        rows   = list(reader)
    mat = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float64)

    conn_canon_idx = np.array([canon_index[canon_from_connectome(h)] for h in header])
    mat_canon = np.zeros((n_rois, n_rois), dtype=np.float64)
    mat_canon[np.ix_(conn_canon_idx, conn_canon_idx)] = mat

    # global [0,1] min-max scaling of the connectome itself, before reducing to per-ROI strength
    mat_min, mat_max = mat_canon.min(), mat_canon.max()
    mat_scaled = (mat_canon - mat_min) / (mat_max - mat_min) if mat_max > mat_min else np.zeros_like(mat_canon)

    strength = mat_scaled.sum(axis=1)   # per-ROI connection strength (weighted degree), canonical order
    strength_by_substrate[substrate] = strength
    log_msg(f"| UPDATE | {substrate}: connectome scaled (raw range [{mat_min:.4g}, {mat_max:.4g}]), "
            f"strength range [{strength.min():.4g}, {strength.max():.4g}], "
            f"{int((strength > 0).sum())}/{n_rois} ROIs with non-zero strength")

substrate_names = sorted(strength_by_substrate.keys(), key=lambda s: int(s.split('_')[1]))


#################################
#     PER-PATIENT BEHAVIOUR     #
#################################

# subject -> {substrate_name: behaviour or None}
behaviour_by_subject = {}
for subject, overlap in overlap_by_subject.items():
    affected = overlap > 0
    scores = {}
    for substrate in substrate_names:
        if not affected.any():
            scores[substrate] = None
            continue
        strength = strength_by_substrate[substrate]
        scores[substrate] = float(np.mean(overlap[affected] * strength[affected]))
    behaviour_by_subject[subject] = scores

n_no_affected = sum(1 for o in overlap_by_subject.values() if not (o > 0).any())
log_msg(f"| UPDATE | Computed behaviour scores for {len(behaviour_by_subject)} subjects "
        f"({n_no_affected} with zero affected ROIs, left blank)")


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

new_cols = [f'{s}_conn_behaviour' for s in substrate_names]
for col in new_cols:
    if col not in fieldnames:
        fieldnames.append(col)

n_matched = 0
for row in rows:
    subject = row[args.id_col]
    scores  = behaviour_by_subject.get(subject)
    if scores is None:
        for col in new_cols:
            row.setdefault(col, '')
        continue
    n_matched += 1
    for substrate, col in zip(substrate_names, new_cols):
        val = scores[substrate]
        row[col] = round(val, 6) if val is not None else ''

log_msg(f"| UPDATE | Filled connectome-behaviour columns for {n_matched}/{len(rows)} "
        f"participants.tsv rows ({len(rows) - n_matched} left blank -- no matching lesion-load row)")

backup_path = participants_path + '.bak'
shutil.copy2(participants_path, backup_path)
log_msg(f"| UPDATE | Backed up existing participants.tsv -> {backup_path}")

with open(participants_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

log_msg(f"| UPDATE | participants.tsv updated with columns: {new_cols}")
log_msg(f"| FINISHED | Substrate connectome-behaviour scores saved -> {participants_path}")
