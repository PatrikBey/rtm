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
# Per-patient summary table: lesion volume, the two REAL behavioural    #
# scores (Foreperiod_Long_tau, GoNoGo_tau -- no synthetic substrate     #
# scores), and a coarse, human-readable lesion location, all in one     #
# simple_stats_summary.tsv.                                             #
#                                                                       #
# Lesion volume: voxel count of the patient's own binary lesion mask    #
# (LESIONS/{subject}.nii.gz) x voxel volume from that image's own       #
# header -- the true total lesion volume, including any voxels outside  #
# the 400-ROI cortical parcellation (e.g. white matter/subcortical),    #
# unlike a volume reconstructed from lesion_loads.tsv alone which would #
# only cover the parcellated cortex.                                    #
#                                                                       #
# Lesion location: derived from Schaefer2018-400_lesion_loads.tsv's     #
# non-zero per-ROI overlap entries, collapsed to the atlas's own coarse #
# "region" (lobe) grouping (Schaefer2018-400_areas.txt's region column  #
# -- the same coarse grouping already used elsewhere in this project    #
# for figure legends/node colouring) crossed with hemisphere, e.g.      #
# "Right Frontal", "Left Occipital". Occipital is relabelled "Visual    #
# cortex" in the output as a standard, uncontroversial synonym. NOTE:   #
# no separate "Motor cortex" category is produced -- the atlas's own    #
# coarse region grouping doesn't distinguish motor/premotor cortex from #
# the rest of the frontal lobe, and ROI naming isn't consistent enough  #
# (some precentral ROIs fall under the SomMot network, others under     #
# DorsAttn) to split it out reliably; "Frontal" here includes motor     #
# cortex undifferentiated from the rest of the frontal lobe.            #
#                                                                       #
# Two location columns per patient:                                     #
#   lesion_location         -- every "Hemisphere Region" group with any #
#                               non-zero overlap, comma-separated,       #
#                               ordered by total overlap extent          #
#                               (descending).                            #
#   primary_lesion_location -- just the single highest-extent group.     #
# "Extent" per group = sum of per-ROI overlap percentages across all     #
# ROIs in that (hemisphere, region) group -- a rough weighting by how    #
# much and how many ROIs in that group are affected, not a true volume.  #
#                                                                       #
# usage: simple_stats_summary.py --data_path /data/patrik/RT/RTM        #
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
    description='Per-patient summary: lesion volume, Foreperiod_Long_tau/GoNoGo_tau behaviour '
                'scores, and a coarse lesion location derived from lesion_loads.tsv.'
)
parser.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM',
                    help='Path to the data directory')
parser.add_argument('--repo_path', type=str,
                    default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')),
                    help='Path to the repo root (default: inferred from this script\'s location)')
parser.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
parser.add_argument('--lesion_dir', type=str, default='LESIONS',
                    help='Name of the folder (inside data_path) containing binary lesion mask '
                         'NIfTIs (default: LESIONS)')
parser.add_argument('--lesion_loads', type=str, default=None,
                    help='Path to the per-ROI lesion-load table (default: '
                         '{data_path}/{atlas}_lesion_loads.tsv)')
parser.add_argument('--atlas_areas', type=str, default=None,
                    help='Path to the atlas areas file supplying the coarse region/hemisphere '
                         'grouping (default: {repo_path}/data/ATLAS/{atlas}_areas.txt)')
parser.add_argument('--participants', type=str, default=None,
                    help='Path to participants.tsv (default: {data_path}/participants.tsv)')
parser.add_argument('--id_col', type=str, default='participant_id',
                    help='participants.tsv column holding the subject id (default: participant_id)')
parser.add_argument('--out_path', type=str, default=None,
                    help='Output TSV path (default: {data_path}/simple_stats_summary.tsv)')
args = parser.parse_args()

lesion_dir         = os.path.join(args.data_path, args.lesion_dir)
lesion_loads_path  = args.lesion_loads or os.path.join(args.data_path, f'{args.atlas}_lesion_loads.tsv')
atlas_areas_path   = args.atlas_areas or os.path.join(args.repo_path, 'data', 'ATLAS', f'{args.atlas}_areas.txt')
participants_path  = args.participants or os.path.join(args.data_path, 'participants.tsv')
out_path            = args.out_path or os.path.join(args.data_path, 'simple_stats_summary.tsv')

BEHAVIOUR_COLS = ['Foreperiod_Long_tau', 'GoNoGo_tau']
REGION_LABEL   = {'Occipital': 'Visual cortex'}   # everything else uses the atlas's own region name as-is

log_msg(f"| START | Building simple_stats_summary.tsv")
log_msg(f"| UPDATE | Lesion masks: {lesion_dir}")
log_msg(f"| UPDATE | Lesion loads: {lesion_loads_path}")
log_msg(f"| UPDATE | participants.tsv: {participants_path}")


#################################
#   COARSE ROI -> (HEMI,REGION) #
#################################

with open(atlas_areas_path, newline='') as fh:
    reader = csv.reader(fh, delimiter='\t')
    next(reader)
    area_rows = list(reader)
# area_rows columns: label_hemi (e.g. 'FPole_1_L'), region (e.g. 'Frontal')
region_by_shortname = {row[0]: row[1] for row in area_rows}


def canon_from_lesion_loads(name):
    '''
    'DefaultA_FPole_1_L' -> 'FPole_1_L' (drop the leading network token,
    matching Schaefer2018-400_areas.txt's short-form 'label_hemi' column)
    '''
    return '_'.join(name.split('_')[1:])


with open(lesion_loads_path, newline='') as fh:
    reader    = csv.reader(fh, delimiter='\t')
    ll_header = reader.__next__()[1:]
    ll_rows   = list(reader)

roi_hemi_region = []   # per lesion_loads.tsv column: (hemisphere, region_label)
for h in ll_header:
    shortname = canon_from_lesion_loads(h)
    region    = region_by_shortname.get(shortname)
    hemi      = 'Right' if shortname.endswith('_R') else ('Left' if shortname.endswith('_L') else '?')
    if region is None:
        log_msg(f"| WARNING | No region found for ROI '{h}' (looked up as '{shortname}') -- skipping "
                f"from coarse-location grouping")
    roi_hemi_region.append((hemi, region))

n_unmapped = sum(1 for _, r in roi_hemi_region if r is None)
log_msg(f"| UPDATE | Mapped {len(ll_header) - n_unmapped}/{len(ll_header)} ROIs to a coarse "
        f"(hemisphere, region) group ({n_unmapped} unmapped)")


#################################
#      LESION VOLUME PER        #
#      PATIENT (from masks)     #
#################################

volume_by_subject = {}
lesion_files = sorted(f for f in os.listdir(lesion_dir) if f.endswith('.nii.gz'))
for fname in lesion_files:
    subject = fname.split('.')[0]
    img     = nib.load(os.path.join(lesion_dir, fname))
    data    = np.asarray(img.dataobj)
    n_voxels = int((data > 0).sum())
    zooms    = img.header.get_zooms()[:3]
    voxel_vol = float(zooms[0]) * float(zooms[1]) * float(zooms[2])
    volume_by_subject[subject] = round(n_voxels * voxel_vol, 2)

log_msg(f"| UPDATE | Computed lesion volume for {len(volume_by_subject)} subjects (from {lesion_dir})")


#################################
#   COARSE LOCATION PER PATIENT #
#   (from lesion_loads.tsv)     #
#################################

location_by_subject = {}   # subject -> (lesion_location str, primary_lesion_location str)
for row in ll_rows:
    subject = row[0]
    vals    = np.array([float(v) for v in row[1:]], dtype=np.float64)

    extent_by_group = {}
    for (hemi, region), v in zip(roi_hemi_region, vals):
        if region is None or v <= 0:
            continue
        label = f'{hemi} {REGION_LABEL.get(region, region)}'
        extent_by_group[label] = extent_by_group.get(label, 0.0) + v

    if not extent_by_group:
        location_by_subject[subject] = ('', '')
        continue

    ranked = sorted(extent_by_group.items(), key=lambda kv: -kv[1])
    lesion_location         = ', '.join(label for label, _ in ranked)
    primary_lesion_location = ranked[0][0]
    location_by_subject[subject] = (lesion_location, primary_lesion_location)

n_no_location = sum(1 for v in location_by_subject.values() if v[0] == '')
log_msg(f"| UPDATE | Computed coarse lesion location for {len(location_by_subject)} subjects "
        f"({n_no_location} with no overlap in any mapped ROI)")


#################################
#     BEHAVIOUR SCORES          #
#################################

with open(participants_path, newline='') as fh:
    part_rows = list(csv.DictReader(fh, delimiter='\t'))

behaviour_by_subject = {}
for row in part_rows:
    subject = row[args.id_col]
    behaviour_by_subject[subject] = {c: row.get(c, '') for c in BEHAVIOUR_COLS}


#################################
#      ASSEMBLE + SAVE TABLE    #
#################################

all_subjects = sorted(set(volume_by_subject) | set(location_by_subject) | set(behaviour_by_subject))

fieldnames = ['participant_id', 'lesion_volume_mm3'] + BEHAVIOUR_COLS + \
             ['lesion_location', 'primary_lesion_location']

n_written = 0
with open(out_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for subject in all_subjects:
        if subject not in volume_by_subject or subject not in location_by_subject:
            continue   # need at least a lesion mask + lesion_loads row to be worth a row
        loc, primary_loc = location_by_subject[subject]
        beh = behaviour_by_subject.get(subject, {c: '' for c in BEHAVIOUR_COLS})
        writer.writerow({
            'participant_id':          subject,
            'lesion_volume_mm3':       volume_by_subject[subject],
            **{c: beh.get(c, '') for c in BEHAVIOUR_COLS},
            'lesion_location':         loc,
            'primary_lesion_location': primary_loc,
        })
        n_written += 1

log_msg(f"| UPDATE | {n_written} patient rows written -> {out_path}")
log_msg(f"| FINISHED | simple_stats_summary.tsv saved -> {out_path}")
