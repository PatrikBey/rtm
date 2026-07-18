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
# The following script performs Ising-model stochastic block modelling  #
# (SBM) reconstruction of brain-region co-occurrence data, with and     #
# without a behaviour indicator node                                    #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/18.                                              #
#                                                                       #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################


import os
import argparse
import pickle
import graph_tool.all as gt
import numpy as np
from utils import log_msg



#################################
#       PARSE PARAMETERS        #
#################################

args = argparse.ArgumentParser(description='Run Ising-model SBM reconstruction on co-occurrence data.')
args.add_argument('--data_file', type=str, default='/mnt/h/WAIS/WAIS_voc_cooccurrence_binary.npy',
                  help='Path to the co-occurrence data .npy file')
args.add_argument('--out_dir', type=str, default='/mnt/h/WAIS', help='Path to the output directory')
args = args.parse_args()

log_msg(f"| START | Running Ising-model SBM reconstruction on co-occurrence data")
log_msg(f"| UPDATE | Data file: {args.data_file}")
log_msg(f"| UPDATE | Output directory: {args.out_dir}")

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)



#################################
#           FUNCTIONS           #
#################################

def fit_ising_sbm(S):
    '''
    fit a nested Ising-model block state and sweep the MCMC chain to convergence
    '''
    state      = gt.PseudoIsingBlockState(S, self_loops=False, nested=True)
    delta, *_  = state.mcmc_sweep(beta=np.inf, niter=1)
    threshold  = max(abs(delta) * 0.01, 1.0)
    while abs(delta) > threshold:
        delta, *_ = state.mcmc_sweep(beta=np.inf, niter=1)
    return state


def extract_graph(state):
    '''
    extract the graph from a fitted block state, registering the Ising couplings as an internal edge property
    '''
    g = state.get_graph()
    g.ep['x'] = state.get_x()  # register as internal property so it survives g.save()
    return g


def brain_only_graph(g, behaviour_node):
    '''
    materialize a standalone copy of a graph with the behaviour node removed
    '''
    keep = g.new_vertex_property('bool', val=True)
    keep[g.vertex(behaviour_node)] = False
    gv = gt.GraphView(g, vfilt=keep)
    return gt.Graph(gv, prune=True)


def get_blocks(state):
    '''
    get the level-0 block assignment array from a (possibly nested) block state
    '''
    bstate = state.get_block_state()
    return (bstate.levels[0].get_blocks().a
            if hasattr(bstate, 'levels')
            else bstate.get_blocks().a)



#################################
#          LOAD DATA            #
#################################

data = np.load(args.data_file, allow_pickle=True).item()
X    = data['data']              # (N_subjects, N_nodes), binary node co-occurrence
y    = np.array(data['labels'])  # (N_subjects,), binary behaviour indicator

log_msg(f"| UPDATE | Subjects: {X.shape[0]}, nodes: {X.shape[1]}")

# ---- build observation matrices ---- #
# Transpose to (N_nodes, N_subjects); Ising states use {-1, +1} spins
S_brain = np.where(X.T.astype(int) > 0, 1, -1)      # (N_nodes, N_subjects)
y_s     = np.where(y > 0, 1, -1)
S_full  = np.vstack([S_brain, y_s[np.newaxis, :]])  # (N_nodes + 1, N_subjects)



#################################
#          FIT MODEL            #
#################################

log_msg(f"| UPDATE | Fitting Ising SBM without behaviour node")
state_no_beh  = fit_ising_sbm(S_brain)
g_no_beh      = extract_graph(state_no_beh)
blocks_no_beh = get_blocks(state_no_beh)
log_msg(f"| UPDATE | Fit complete without behaviour node ({int(blocks_no_beh.max()) + 1} blocks)")

log_msg(f"| UPDATE | Fitting Ising SBM with behaviour node")
state_with_beh   = fit_ising_sbm(S_full)
g_with_beh       = extract_graph(state_with_beh)
blocks_with_beh  = get_blocks(state_with_beh)
behaviour_node   = S_full.shape[0] - 1
g_with_beh_brain = brain_only_graph(g_with_beh, behaviour_node)
log_msg(f"| UPDATE | Fit complete with behaviour node ({int(blocks_with_beh.max()) + 1} blocks)")



#################################
#        SAVE OUTPUTS           #
#################################

# ---- states ---- #
state_no_beh_path = os.path.join(args.out_dir, 'recon_no_beh_state.pkl')
with open(state_no_beh_path, 'wb') as f:
    pickle.dump(state_no_beh, f)
log_msg(f"| UPDATE | Block state saved (no behaviour node) → {state_no_beh_path}")

state_with_beh_path = os.path.join(args.out_dir, 'recon_beh_node_state.pkl')
with open(state_with_beh_path, 'wb') as f:
    pickle.dump(state_with_beh, f)
log_msg(f"| UPDATE | Block state saved (with behaviour node) → {state_with_beh_path}")

# ---- graphs ---- #
# without-behaviour fit: brain nodes only (this is the natural output of that fit)
graph_no_beh_path = os.path.join(args.out_dir, 'recon_no_beh_graph.gt')
g_no_beh.save(graph_no_beh_path)
log_msg(f"| UPDATE | Graph saved (no behaviour node) → {graph_no_beh_path}")

# with-behaviour fit: both the full (brain + behaviour) graph and a brain-only projection of it
graph_with_beh_full_path = os.path.join(args.out_dir, 'recon_beh_node_graph_full.gt')
g_with_beh.save(graph_with_beh_full_path)
log_msg(f"| UPDATE | Graph saved (with behaviour node, full) → {graph_with_beh_full_path}")

graph_with_beh_brain_path = os.path.join(args.out_dir, 'recon_beh_node_graph_brain_only.gt')
g_with_beh_brain.save(graph_with_beh_brain_path)
log_msg(f"| UPDATE | Graph saved (with behaviour node, brain-only projection) → {graph_with_beh_brain_path}")

# ---- behaviour edges (from the with-behaviour fit) ---- #
# Edges incident to the behaviour node encode brain-behaviour Ising couplings
beh_edges = []
for e in g_with_beh.edges():
    s, t = int(e.source()), int(e.target())
    if s == behaviour_node or t == behaviour_node:
        region = t if s == behaviour_node else s
        beh_edges.append((region, g_with_beh.ep['x'][e]))

beh_edges_path = os.path.join(args.out_dir, 'recon_beh_node_behaviour_edges.npy')
np.save(beh_edges_path, np.array(beh_edges))
log_msg(f"| UPDATE | Behaviour edges saved ({len(beh_edges)} edges) → {beh_edges_path}")

log_msg(f"| FINISHED | All outputs saved")
