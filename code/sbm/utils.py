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


def load_graphs(path, atlas, subject_list, part, score_col, disconnectomes_dir=None):
    '''
    load disconnectomes and behaviour scores for subjects with valid data.
    disconnectomes_dir overrides the default {path}/DISCONNECTOMES (e.g.
    DISCONNECTOMES400 for the 400-ROI atlas, now that DISCONNECTOMES itself
    holds VoxelAtlas_* files).
    '''
    disconnectomes_dir = disconnectomes_dir or os.path.join(path, 'DISCONNECTOMES')
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
        tmp = np.genfromtxt(os.path.join(disconnectomes_dir, f'{subject}_{atlas}.tsv'), delimiter='\t')
        data = tmp[1:, 1:].astype(np.float32)
        if np.sum(data) == 0:
            empty_subjects.append(subject)
            continue
        subject_list_clean.append(subject)
        behaviour.append(float(val[0]))
        adj_matrices_list.append(np.where(data >= np.quantile(data[data > 0], .5), 1, 0))
    adj_matrices = np.stack(adj_matrices_list).astype(np.int32)
    behaviour = [float(v) for v in behaviour]
    return subject_list_clean, behaviour, adj_matrices, subjects_missing_score, empty_subjects


def get_disco_format(disconnectomes_dir, atlas):
    '''
    detect whether a disconnectomes directory holds tsv (dense, labelled) or
    npz (sparse: row, col, data, shape, roi_names) files for the given atlas
    -- ARISE >=0.6 can emit either via its OutFormat flag. npz is preferred
    if both are present.
    '''
    discos = os.listdir(disconnectomes_dir)
    if any(f.endswith(f'_{atlas}.npz') for f in discos):
        return 'npz'
    return 'tsv'


def load_disco_matrix(file_path, file_ext):
    '''
    load a single subject's disconnectome as a dense float32 matrix,
    regardless of on-disk format.
    '''
    if file_ext == 'npz':
        with np.load(file_path) as npz:
            data = np.zeros(tuple(npz['shape']), dtype=np.float32)
            data[npz['row'], npz['col']] = npz['data']
        return data
    tmp = np.genfromtxt(file_path, delimiter='\t')
    return tmp[1:, 1:].astype(np.float32)


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


def get_cooccurrence_layer(graph):
    '''
    get adjacency matrix of a single-layer cooccurrence-only graph
    '''
    n = graph.num_vertices()
    occ_layer = np.zeros((n, n))
    for e in graph.edges():
        i, j = int(e.source()), int(e.target())
        occ_layer[i, j] = occ_layer[j, i] = graph.ep.cooccurrence_weight[e]
    return occ_layer
