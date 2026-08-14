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
# Regenerates the SBM state visualisation for the observed multilayer   #
# fits (SBMFITTING, run.py) - create_figures.py section 5 ("SBM         #
# COMMUNITIES"), extracted into its own runnable script since the rest  #
# of create_figures.py depends on files/paths (lesion distribution,     #
# disconnectome example tractograms, dipy/fury) not needed here. The    #
# section 5 logic itself (load_joint_adjacency, block_of_node from      #
# roi_block_assignments, utils.plot_sbm_state) is unchanged.            #
#                                                                       #
# Outputs are written directly into each task's own SBMFITTING          #
# directory, alongside its other fit outputs:                           #
#   SBM_final_state_{task}_joint_blocks.svg (+ _blocks_legend.svg)      #
#   SBM_final_state_{task}_joint_weights.svg                            #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import os
import csv
import argparse
import numpy as np

import utils


#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='SBM state visualisation for observed multilayer fits (SBMFITTING).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+', default=['Foreperiod_Long_tau', 'GoNoGo_tau', 'SATO_Accuracy_tau'],
                  help='Behaviour scores to visualise (default: all three tasks)')
args.add_argument('--fit_suffix', type=str, default='_singleflip',
                  help='Suffix of the observed SBMFITTING run directory (default: _singleflip)')
args.add_argument('--out_dir', type=str, default=None,
                  help='Output directory for the figures (default: {data_path}/SBMFITTING/FIGURES)')
args.add_argument('--ext', type=str, default='svg', choices=['svg', 'png'],
                  help='Output image format (default: svg)')
args.add_argument('--perm', type=str, default=None,
                  help='If given (e.g. perm_00000), plot that behaviour-permutation null '
                       'example from SBMNULL instead of the observed SBMFITTING fit -- same '
                       'file layout (SBM_final_graph_{task}.gt, roi_block_assignments_{task}.csv) '
                       'written by run_null.py, one directory per task under '
                       '{data_path}/SBMNULL/SBM_{atlas}_{task}_NULL/{perm}/')
args = args.parse_args()

out_dir = args.out_dir or os.path.join(args.data_path, 'SBMFITTING', 'FIGURES')
os.makedirs(out_dir, exist_ok=True)

grouped_raw = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas_grouped.txt'),
                            dtype=str, delimiter='\t')
node_groups = grouped_raw[1:, 2]


#####################################
#                                   #
# 5. SBM COMMUNITIES (from          #
#    create_figures.py, unchanged)  #
#                                   #
#####################################

# ---- fit and plot the block communities of the final joint graph ---- #
# one adjacency per task (behaviour + cooccurrence collapsed into the
# final MCMC-fitted multilayer graph), same as create_figures.py section 5.
for task in args.tasks:
    if args.perm:
        fit_dir = os.path.join(args.data_path, 'SBMNULL', f'SBM_{args.atlas}_{task}_NULL', args.perm)
        tag = f'{task}_{args.perm}'
    else:
        fit_dir = os.path.join(args.data_path, 'SBMFITTING', f'SBM_{args.atlas}_{task}{args.fit_suffix}')
        tag = task

    graph_path = os.path.join(fit_dir, f'SBM_final_graph_{task}.gt')
    if not os.path.isfile(graph_path):
        print(f'Skipped {task}: no multilayer fit at {fit_dir}')
        continue
    adj = utils.load_joint_adjacency(graph_path)

    assignments_path = os.path.join(fit_dir, f'roi_block_assignments_{task}.csv')
    with open(assignments_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    level_cols    = sorted((c for c in rows[0] if c.startswith('level_')), key=lambda c: int(c.split('_')[1]))
    block_of_node = [np.array([int(row[c]) for row in rows]) for c in level_cols]

    output_prefix = os.path.join(out_dir, f'SBM_final_state_{tag}_joint')
    utils.plot_sbm_state(adj, node_groups, output_prefix, cmap='plasma', arrow_colour='gold',
                         block_of_node=block_of_node, output_ext=args.ext)
    print(f'Saved → {output_prefix}_blocks.{args.ext} (+ _blocks_legend.{args.ext}, _weights.{args.ext})')
