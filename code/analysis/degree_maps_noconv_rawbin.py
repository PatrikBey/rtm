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
# Variant of degree_maps_noconv.py: instead of reading the already-     #
# saved behaviour_degree/cooccurrence_degree columns from a completed   #
# SBMNOCONV20K fit (which binarize each subject's disconnectome at      #
# their own median, per load_graphs()/create_multilayer_graph_streaming #
# -- np.where(data >= median, 1, 0)), this rebuilds the two layers      #
# from scratch with each subject's disconnectome binarized on >0        #
# instead (np.where(data > 0, 1, 0)) -- i.e. any nonzero connection     #
# counts, not just the top half. No SBM fit is run here: only the two   #
# layer matrices are built (group-level edge_threshold still applied,   #
# same as create_multilayer_graph()) and per-ROI weighted degree is     #
# computed directly from them.                                          #
#                                                                       #
# Outputs (per task):                                                   #
#   {atlas}_behaviour_degree_rawbin_{task}.nii.gz                       #
#   {atlas}_cooccurrence_degree_rawbin_{task}.nii.gz                    #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import argparse
import numpy as np
import nibabel as nib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sbm'))
from utils import log_msg
from functions import _finalize_multilayer_graph


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Build behaviour/cooccurrence layers with >0 per-subject '
                                            'binarization (no median cut) and map node degree to NIfTI.')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--disconnectomes_dir', type=str, default=None,
                  help='Disconnectomes directory (default: {data_path}/DISCONNECTOMES400)')
args.add_argument('--tasks', type=str, nargs='+', default=['foreperiod_hexg_tau', 'gonogo_hexg_tau'],
                  help='Behaviour scores to build layers for')
args.add_argument('--edge_threshold', type=float, default=75,
                  help='Percentile threshold applied independently to each layer (default: 75, matching '
                       'the SBMNOCONV20K singleflip fits this is compared against)')
args.add_argument('--behaviour_dist', type=str, default='normal', choices=['normal', 'poisson'])
args.add_argument('--cooccurrence_dist', type=str, default='normal', choices=['normal', 'poisson'])
args.add_argument('--combined_layers', action=argparse.BooleanOptionalAction, default=False)
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory (default: {data_path}/ROISELECTION)')
args = args.parse_args()

disconnectomes_dir = args.disconnectomes_dir or os.path.join(args.data_path, 'DISCONNECTOMES400')
out_dir = args.out_dir or os.path.join(args.data_path, 'ROISELECTION')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


#################################
#      LOAD ATLAS METADATA      #
#################################

atlas_meta = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[0, :].tolist()
node_names = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas.txt'),
                           dtype=str, delimiter='\t')[1:, atlas_meta.index('label')].tolist()
n_nodes = len(node_names)

atlas_nii_path = os.path.join(args.data_path, 'ATLAS', f'{args.atlas}.nii.gz')
atlas_img      = nib.load(atlas_nii_path)
atlas_data     = np.asarray(atlas_img.dataobj, dtype=np.int32)

participants_path = os.path.join(args.data_path, 'participants.tsv')
part = np.genfromtxt(participants_path, dtype=str, delimiter='\t')
header = part[0]

discos = os.listdir(disconnectomes_dir)
subject_list = [f.split('_')[0] for f in discos if f.endswith(f'_{args.atlas}.tsv')]


#################################
#   BUILD LAYERS (>0 binarize)  #
#################################

for task in args.tasks:
    score_col = np.where(header == task)[0][0]

    behaviour_weighted  = np.zeros((n_nodes, n_nodes))
    cooccurrence_binary = np.zeros((n_nodes, n_nodes))
    n_included = 0
    n_missing  = 0
    n_empty    = 0

    for subject in subject_list:
        val = part[part[:, 0] == subject, score_col]
        if val.size == 0 or val[0] in ('', 'nan', 'NaN'):
            n_missing += 1
            continue
        file_path = os.path.join(disconnectomes_dir, f'{subject}_{args.atlas}.tsv')
        tmp = np.genfromtxt(file_path, delimiter='\t')
        data = tmp[1:, 1:].astype(np.float32)
        if np.sum(data) == 0:
            n_empty += 1
            continue
        behaviour_value = float(val[0])
        # >0 binarization -- no per-subject median cut
        tmp_adj = np.where(data > 0, 1, 0)
        behaviour_weighted  += tmp_adj * behaviour_value
        cooccurrence_binary += tmp_adj
        n_included += 1

    log_msg(f"| UPDATE | {task}: included {n_included}, missing score {n_missing}, empty disconnectome {n_empty}")

    graph = _finalize_multilayer_graph(behaviour_weighted, cooccurrence_binary, n_included,
                                       node_names, args.edge_threshold, args.cooccurrence_dist,
                                       args.combined_layers)

    beh_degree = np.zeros(graph.num_vertices())
    occ_degree = np.zeros(graph.num_vertices())
    for e in graph.edges():
        if graph.ep.layer[e] == 0:
            w = graph.ep.behaviour_weight[e]
            beh_degree[int(e.source())] += w
            beh_degree[int(e.target())] += w
        else:
            w = graph.ep.cooccurrence_weight[e]
            occ_degree[int(e.source())] += w
            occ_degree[int(e.target())] += w

    valid_mask = (atlas_data > 0) & (atlas_data <= n_nodes)

    beh_data = np.zeros_like(atlas_data, dtype=np.float32)
    beh_data[valid_mask] = beh_degree[atlas_data[valid_mask] - 1]
    beh_path = os.path.join(out_dir, f'{args.atlas}_behaviour_degree_rawbin_{task}.nii.gz')
    nib.save(nib.Nifti1Image(beh_data, atlas_img.affine, atlas_img.header), beh_path)
    log_msg(f"| UPDATE | {task}: behaviour-layer degree NIfTI saved "
            f"(range [{beh_degree.min():.2f}, {beh_degree.max():.2f}]) -> {beh_path}")

    occ_data = np.zeros_like(atlas_data, dtype=np.float32)
    occ_data[valid_mask] = occ_degree[atlas_data[valid_mask] - 1]
    occ_path = os.path.join(out_dir, f'{args.atlas}_cooccurrence_degree_rawbin_{task}.nii.gz')
    nib.save(nib.Nifti1Image(occ_data, atlas_img.affine, atlas_img.header), occ_path)
    log_msg(f"| UPDATE | {task}: cooccurrence-layer degree NIfTI saved "
            f"(range [{occ_degree.min():.2f}, {occ_degree.max():.2f}]) -> {occ_path}")

log_msg(f"| FINISHED | Raw-binarized degree NIfTIs saved -> {out_dir}")
