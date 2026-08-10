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
# Generates the SBM state visualisation for the lesion-only base fits   #
# (SBMBASE, run_base.py), the counterpart of create_figures.py section  #
# 5 ("SBM COMMUNITIES") for the observed multilayer fits (SBMFITTING).  #
#                                                                       #
# Uses the base fit's own real converged partition (roi_block_           #
# assignments_{task}.csv level_* columns) and final graph               #
# (SBM_final_graph_{task}.gt) via utils.plot_sbm_state - not an          #
# independent re-fit - since the lesion-only model has no behaviour      #
# layer to collapse, load_cooccurrence_adjacency (single 'cooccurrence_ #
# weight' edge property) is used in place of load_joint_adjacency.      #
#                                                                       #
# Outputs are written directly into each task's own SBMBASE directory,  #
# alongside its other fit outputs:                                      #
#   SBM_final_state_{task}_lesiononly_blocks.svg (+ _blocks_legend.svg) #
#   SBM_final_state_{task}_lesiononly_weights.svg                       #
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

args = argparse.ArgumentParser(description='SBM state visualisation for lesion-only base fits (SBMBASE).')
args.add_argument('--data_path', type=str, default='/data/patrik/RT/RTM', help='Path to the data directory')
args.add_argument('--atlas', type=str, default='Schaefer2018-400', help='Atlas name')
args.add_argument('--tasks', type=str, nargs='+', default=['Foreperiod_Long_tau', 'GoNoGo_tau', 'SATO_Accuracy_tau'],
                  help='Behaviour scores to visualise (default: all three tasks)')
args.add_argument('--base_suffix', type=str, default='_base_singleflip',
                  help='Suffix of the lesion-only SBMBASE run directory (default: _base_singleflip)')
args = args.parse_args()

grouped_raw  = np.genfromtxt(os.path.join(args.data_path, 'ATLAS', f'{args.atlas}_areas_grouped.txt'),
                             dtype=str, delimiter='\t')
node_groups  = grouped_raw[1:, 2]


#################################
#     PER-TASK VISUALISATION    #
#################################

for task in args.tasks:
    base_dir = os.path.join(args.data_path, 'SBMBASE', f'SBM_{args.atlas}_{task}{args.base_suffix}')

    graph_path = os.path.join(base_dir, f'SBM_final_graph_{task}.gt')
    if not os.path.isfile(graph_path):
        print(f'Skipped {task}: no lesion-only base fit at {base_dir}')
        continue
    adj = utils.load_cooccurrence_adjacency(graph_path)

    assignments_path = os.path.join(base_dir, f'roi_block_assignments_{task}.csv')
    with open(assignments_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    level_cols     = sorted((c for c in rows[0] if c.startswith('level_')), key=lambda c: int(c.split('_')[1]))
    block_of_node  = [np.array([int(row[c]) for row in rows]) for c in level_cols]

    output_prefix = os.path.join(base_dir, f'SBM_final_state_{task}_lesiononly')
    utils.plot_sbm_state(adj, node_groups, output_prefix, cmap='plasma', arrow_colour='gold',
                         block_of_node=block_of_node)
    print(f'Saved → {output_prefix}_blocks.svg (+ _blocks_legend.svg, _weights.svg)')
