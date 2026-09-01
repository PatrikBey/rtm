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
# Generate a DISTRIBUTED synthetic ground-truth behaviour score: unlike #
# get_substrate_connectome_behaviour.py's per-substrate scores (each    #
# derived from a single seed ROI's own connectome, and consequently     #
# near-linearly recoverable from that one ROI's node degree alone --    #
# see beh_regression.py's V1/IFG sanity checks, R^2 up to 0.60 for a    #
# single ROI), this combines TWO independent substrates multiplicatively #
# per patient, so a high score requires BOTH regions to be meaningfully #
# affected jointly -- a pattern no single ROI's own degree can recover  #
# on its own (the product of two roughly-independent quantities has     #
# little marginal correlation with either factor alone). This is        #
# exactly the class of signal a joint/block-partition method (the SBM   #
# framework) is suited to detect and a plain per-ROI regression         #
# (beh_regression.py) structurally is not -- the intended stress test.  #
#                                                                       #
# Default substrates: sub_01_V1 and sub_03_IFG (both left hemisphere:   #
# Striate_1_L, IFG_1_L).                                                #
#                                                                       #
# Per substrate, per patient: identical computation to                  #
# get_substrate_connectome_behaviour.py -- that substrate's own [0,1]-  #
# scaled connectome collapsed to per-ROI connection strength, then      #
# mean(overlap[roi] * strength[roi]) over the patient's own AFFECTED    #
# ROIs (overlap > 0) for that substrate's connectome. A patient with    #
# zero affected ROIs for a given substrate gets None for that factor.   #
#                                                                       #
# Combined score = product of all per-substrate scores. None (missing) #
# if ANY factor is None (a joint score is only defined if every        #
# component substrate has a well-defined score for that patient) --    #
# NOT the same as "zero lesion overlap", which would still be a         #
# perfectly well-defined near-zero product.                             #
#                                                                       #
# ROI naming differs across the three input files (connectome tsv:      #
# "17networks_LH_DefaultA_FPole_1"; lesion_loads.tsv:                   #
# "DefaultA_FPole_1_L"; atlas areas.txt: "FPole_1_L") -- all three are   #
# converted to the areas.txt convention and joined by that canonical    #
# name, not by column position, before any arithmetic (identical to     #
# get_substrate_connectome_behaviour.py).                               #
#                                                                       #
# participants.tsv is modified in place; the pre-existing file is       #
# copied to participants.tsv.bak first (overwritten on every run).      #
#                                                                       #
# usage: gen_dist_synth.py --data_path /data/patrik/RT/RTM              #
#                          --substrates sub_01_V1 sub_03_IFG            #
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
    description='Generate a distributed synthetic behaviour score by multiplicatively '
                'combining two (or more) substrates\' connectome-weighted lesion-overlap '
                'scores, so the result requires joint involvement of all component regions -- '
                'a signal no single ROI\'s own degree can recover in isolation.'
)
parser.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM',
                    help='Path to the data directory')
parser.add_argument('--repo_path', type=str,
                    default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='Path to the repo root (default: inferred from this script\'s location)')
parser.add_argument('--substrates', type=str, nargs='+', default=['sub_01_V1', 'sub_03_IFG'],
                    help='Substrate names to combine, matching sub_NN_TAG_Schaefer2018-400.tsv '
                         'connectome filenames in arise_dir (default: sub_01_V1 sub_03_IFG, i.e. '
                         'left striate cortex + left inferior frontal gyrus)')
parser.add_argument('--arise_dir', type=str, default=None,
                    help='Directory containing sub_NN_TAG_Schaefer2018-400.tsv connectomes '
                         '(default: {data_path}/SUBSTRATES/arise)')
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
parser.add_argument('--column_name', type=str, default=None,
                    help='Output column name (default: dist_{TAG1}_{TAG2}[..._TAGn]_conn_behaviour, '
                         'derived from --substrates)')
args = parser.parse_args()

arise_dir         = args.arise_dir or os.path.join(args.data_path, 'SUBSTRATES', 'arise')
lesion_loads_path = args.lesion_loads or os.path.join(args.data_path, 'Schaefer2018-400_lesion_loads.tsv')
atlas_areas_path  = args.atlas_areas or os.path.join(args.repo_path, 'data', 'ATLAS', 'Schaefer2018-400_areas.txt')
participants_path = args.participants or os.path.join(args.data_path, 'participants.tsv')

tags = []
for s in args.substrates:
    m = re.match(r'sub_\d+_(\w+)', s)
    tags.append(m.group(1) if m else s)
column_name = args.column_name or 'dist_' + '_'.join(tags) + '_conn_behaviour'

log_msg(f"| START | Generating distributed synthetic behaviour: {' x '.join(args.substrates)}")
log_msg(f"| UPDATE | Arise directory: {arise_dir}")
log_msg(f"| UPDATE | Output column: {column_name}")
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
#  LOAD + SCALE EACH SUBSTRATE  #
#  CONNECTOME (only the ones    #
#  named in --substrates)       #
#################################

strength_by_substrate = {}   # substrate name -> (n_rois,) connection-strength vector, canonical order
for substrate in args.substrates:
    cp = os.path.join(arise_dir, f'{substrate}_Schaefer2018-400.tsv')
    if not os.path.isfile(cp):
        raise SystemExit(f'No connectome found for substrate "{substrate}" at {cp}')

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


#################################
#   PER-SUBSTRATE, PER-PATIENT  #
#   SCORES (unchanged from       #
#   get_substrate_connectome_    #
#   behaviour.py's own logic)   #
#################################

# subject -> {substrate_name: score or None}
factor_by_subject = {}
for subject, overlap in overlap_by_subject.items():
    affected = overlap > 0
    scores = {}
    for substrate in args.substrates:
        if not affected.any():
            scores[substrate] = None
            continue
        strength = strength_by_substrate[substrate]
        scores[substrate] = float(np.mean(overlap[affected] * strength[affected]))
    factor_by_subject[subject] = scores


#################################
#   COMBINE MULTIPLICATIVELY    #
#################################

# Joint score = product of every substrate's own factor -- undefined (None) unless
# EVERY factor is itself defined, since a genuinely joint requirement can't be
# evaluated if any one component substrate had zero affected ROIs for this patient.
dist_score_by_subject = {}
n_all_defined = 0
n_some_missing = 0
for subject, scores in factor_by_subject.items():
    if any(v is None for v in scores.values()):
        dist_score_by_subject[subject] = None
        n_some_missing += 1
        continue
    product = 1.0
    for v in scores.values():
        product *= v
    dist_score_by_subject[subject] = product
    n_all_defined += 1

log_msg(f"| UPDATE | Distributed score defined for {n_all_defined}/{len(factor_by_subject)} subjects "
        f"({n_some_missing} missing at least one component substrate's factor)")

defined_vals = np.array([v for v in dist_score_by_subject.values() if v is not None])
if defined_vals.size:
    log_msg(f"| UPDATE | Distributed score range: [{defined_vals.min():.6g}, {defined_vals.max():.6g}], "
            f"mean {defined_vals.mean():.6g}")


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

if column_name not in fieldnames:
    fieldnames.append(column_name)

n_matched = 0
for row in rows:
    subject = row[args.id_col]
    val = dist_score_by_subject.get(subject)
    if val is None:
        row.setdefault(column_name, '')
        continue
    n_matched += 1
    row[column_name] = round(val, 8)

log_msg(f"| UPDATE | Filled {column_name} for {n_matched}/{len(rows)} participants.tsv rows "
        f"({len(rows) - n_matched} left blank)")

backup_path = participants_path + '.bak'
shutil.copy2(participants_path, backup_path)
log_msg(f"| UPDATE | Backed up existing participants.tsv -> {backup_path}")

with open(participants_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

log_msg(f"| UPDATE | participants.tsv updated with column: {column_name}")
log_msg(f"| FINISHED | Distributed synthetic behaviour saved -> {participants_path}")
