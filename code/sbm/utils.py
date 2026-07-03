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
# The following script contains utility functions                       #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/06/30.                                              #
#                                                                       #
#                                                                       #
#########################################################################


import graph_tool.all as gt
import numpy as np
import os

#########################################
#                                       #
#         LOGGING UTILITIES             #
#                                       #
#########################################

def log_msg(_string):
    '''
    logging function printing date, scriptname & input string to stdout
    '''
    import datetime, os, sys
    print(f'{datetime.date.today().strftime("%a %B %d %H:%M:%S %Z %Y")} {str(os.path.basename(sys.argv[0]))}: {str(_string)}')



#########################################
#                                       #
#             I/O UTILITIES             #
#                                       #
#########################################


def load_graphs(path, atlas, subject_list, part, score_col):
    '''
    load disconnectomes and behaviour scores for subjects with valid data
    '''
    subject_list_clean = []
    behaviour = []
    adj_matrices_list = []
    subjects_missing_score = []
    empty_subjects = []
    for subject in subject_list:
        val = part[part[:, 0] == subject, score_col]
        if val.size == 0 or val[0] in ('', 'nan', 'NaN'):
            subjects_missing_score.append(subject)
            continue
        tmp = np.genfromtxt(os.path.join(path, 'DISCONNECTOMES', f'{subject}_{atlas}.tsv'), delimiter='\t')
        data = tmp[1:, 1:].astype(np.float32)
        if np.sum(data) == 0:
            empty_subjects.append(subject)
            continue
        subject_list_clean.append(subject)
        behaviour.append(float(val[0]))
        adj_matrices_list.append(np.where(data >= np.quantile(data[data > 0], .5), 1, 0))
    adj_matrices = np.stack(adj_matrices_list).astype(np.int32)
    behaviour = list((np.array(behaviour) - np.mean(behaviour)) / np.std(behaviour))
    return subject_list_clean, behaviour, adj_matrices, subjects_missing_score, empty_subjects


def get_graph_layers(graph):
    '''
    get adjacency matrices for each layer of a multilayer graph
    '''
    n = graph.num_vertices()
    occ_layer = np.zeros((n, n))
    beh_layer = np.zeros((n, n))
    for e in graph.edges():
        i, j = int(e.source()), int(e.target())
        if graph.ep.layer[e] == 0:
            beh_layer[i, j] = beh_layer[j, i] = graph.ep.behaviour_weight[e]
        else:
            occ_layer[i, j] = occ_layer[j, i] = graph.ep.cooccurrence_weight[e]
    return [occ_layer, beh_layer]
