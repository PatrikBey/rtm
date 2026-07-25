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
# The following script generates visualisations of the SBM based        #
# modelling of intelligence. Visualizations include:                    #
# 1. lesion distribution                                                #
# 2. disconnectome example                                              #
# 3. graph layers                                                       #
# 4. SBM blocks                                                         #
# 5. block connectivity                                                 #
#                                                                       #
#                                                                       #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/14.                                              #
#                                                                       #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################


from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.tracking.streamline import Streamlines

import numpy as np, os, csv
import nibabel as nib
import fury
import matplotlib.pyplot as plt

import utils







#################################
#      input file paths         #
#################################

path = '/mnt/h/RT/data/'
result_dir = os.path.join(path,'RESULTS/SBMRECONPNB_LOADS')
out_dir = os.path.join(result_dir,'FIGURES')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

TASKS = ['Foreperiod_Long_tau','GoNoGo_tau','SATO_Accuracy_tau']
MODELS = ['beh_weighted', 'no_beh']

# ---- loading template files ---- #
atlas = 'Schaefer2018-400'
atlas_img = nib.load(os.path.join(path, 'ATLAS',f'{atlas}.nii.gz'))
template_img = nib.load(os.path.join(path, 'ATLAS','MNI152_icbm_T1_1mm_brain.nii.gz'))

brain_data = template_img.get_fdata()
brain_affine = template_img.affine
brain_mask = brain_data > 0

brain_colour = "#36C2AF"
roi_colour = "#700845"
back_colour = "#B3B1B1"




#####################################
#                                   #
# 1. LESION DISTRIBUTION            #
#                                   #
#####################################

# ---- define local parameters ---- #
cmap = utils.make_cmap([brain_colour, roi_colour])


# ---- load local files ---- #
lesion_file = os.path.join(path, 'LesionAggregate.nii.gz')
lesion_img = nib.load(lesion_file).get_fdata()

brain = template_img.get_fdata()
slices = [40,55,70,85,100,115,130,145]
variances = []
for i in slices:
    vals = lesion_img[:,:,i][lesion_img[:,:,i] > 0]
    variances.append(np.var(vals) if len(vals) > 0 else 0)
max_var = max(variances)
levels_per_slice = [max(2, int(round(25 * v / max_var))) if max_var > 0 else 2 for v in variances]


# ---- contour filled plot ---- #
for idx_s, i in enumerate(slices):
    plt.subplot(2,4,idx_s+1)
    tmp = brain[:,:,i].T
    plt.imshow(np.where(tmp==0,np.nan,tmp), cmap='gray', origin='lower', alpha = 0.9, interpolation='quadric')
    tmp = lesion_img[:,:,i].T
    # plt.contourf(np.where(tmp==0,np.nan,tmp), levels = levels_per_slice[idx_s], cmap = cmap, origin='lower', alpha = .75, antialiased=False)
    plt.imshow(np.where(tmp==0,np.nan,tmp), cmap = 'plasma', interpolation='quadric', origin='lower', alpha = .66)
    plt.clim((0, lesion_img.max()))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # plt.colorbar()
    # plt.title(f'Slice {i}')

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'LesionDistribution.svg'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()




#####################################
#                                   #
# 2. DISCONNECTOME EXAMPLE          #
#                                   #
#####################################

# # ---- define local parameters ---- #
# surface_opacity = 0.05
# streamline_color = "#494949"
# streamline_opacity = 0.1
# streamline_linewidth = 0.75
# brain_opacity = 0.05
# roi_opacity = 0.3

# # ---- load local files ---- #
# lesion_file = os.path.join(path, 'example','example_lesion.nii.gz')
# tck_file = os.path.join(path, 'example','example_tracts_1K.tck')

# tractogram = load_tractogram(tck_file, lesion_file, to_space=Space.RASMM)
# streamlines = Streamlines(tractogram.streamlines)

# # ---- glass surface + streamlines + volumetric ROI ---- #
# glass_meshes, brain_aspect = utils.load_glass_surface()

# fig, axes = utils.setup_views_figure()

# for ax, (view_name, elev, azim) in zip(axes, utils.VIEWS):
#     utils.plot_glass_surface(ax, glass_meshes, color=utils.hex_to_rgb(back_colour), opacity=brain_opacity)
#     utils.plot_tracts(ax, streamlines, colors=utils.hex_to_rgb(streamline_color),
#                        opacity=streamline_opacity, linewidth=streamline_linewidth)
#     utils.plot_mask(ax, lesion_file, color=utils.hex_to_rgb(roi_colour), opacity=roi_opacity)
#     utils.finalize_view(ax, view_name, elev, azim, brain_aspect)

# plt.suptitle('Disconnectome example', fontsize=14)
# plt.tight_layout()

# out_path = os.path.join(out_dir, 'disconnectome_example.svg')
# plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
# print(f'Saved → {out_path}')
# plt.close(fig)


#####################################
#                                   #
# 3. NODE OCCURENCE MATRIX          #
#                                   #
#####################################

# ---- define local parameters ---- #
cmap = utils.make_cmap([back_colour, roi_colour])


# ---- duplicated from run_recon_pnb.py: same [0,1] rescaling / behaviour- ---- #
# ---- weighting pipeline, so these matrices match what that model        ---- #
# ---- actually fits on (kept as a standalone duplicate, not an import,   ---- #
# ---- matching this repo's convention of independent run scripts)        ---- #
def rescale_01(x):
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return np.ones_like(x) if span == 0 else (x - x.min()) / span


def load_node_matrix(data_file):
    data = np.genfromtxt(data_file, delimiter='\t', dtype=str)
    vals = data[1:, 1:].astype(float)
    return vals, data[1:,0].tolist()


def load_data(data_file, data_path, score):
    raw_vals, subject_list = load_node_matrix(data_file)

    part      = np.genfromtxt(os.path.join(data_path, 'participants.tsv'), dtype=str, delimiter='\t')
    score_col = np.where(part[0] == score)[0][0]

    keep_idx  = []
    behaviour = []
    for i, subject in enumerate(subject_list):
        val = part[part[:, 0] == subject, score_col]
        if val.size == 0 or val[0] in ('', 'nan', 'NaN'):
            continue
        keep_idx.append(i)
        behaviour.append(float(val[0]))

    raw_vals_clean = raw_vals[keep_idx]
    behaviour      = np.array(behaviour, dtype=np.float64)

    keep_var       = raw_vals_clean.std(axis=1) > 0
    raw_vals_clean = raw_vals_clean[keep_var]
    behaviour      = behaviour[keep_var]

    return raw_vals_clean, behaviour


lesion_file = os.path.join(path, 'Schaefer2018-400_node_strength.tsv')
raw_vals_all, node_names = load_node_matrix(lesion_file)
raw_vals_01 = rescale_01(raw_vals_all)

# ---- 1x4: raw matrix + one behaviour-weighted matrix per task ---- #
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

im = axes[0].imshow(np.where(raw_vals_01 == 0, np.nan, raw_vals_01), cmap='plasma',
                    interpolation='nearest', aspect='auto', vmin=0, vmax=1)
axes[0].set_xlabel('L   -   Brain regions   -   R')
axes[0].set_ylabel('Subjects')
axes[0].set_title('Raw node-strength matrix (0-1)')
plt.colorbar(im, ax=axes[0])

for ax, task in zip(axes[1:], TASKS):
    raw_vals_task, behaviour = load_data(lesion_file, path, task)
    raw_vals_task_01  = rescale_01(raw_vals_task)
    behaviour_weight  = rescale_01(behaviour)
    weighted          = raw_vals_task_01 * behaviour_weight[:, np.newaxis]

    im = ax.imshow(np.where(weighted == 0, np.nan, weighted), cmap='plasma',
                   interpolation='none', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('L   -   Brain regions   -   R')
    ax.set_ylabel('Subjects')
    ax.set_title(f'Behaviour-weighted — {task}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'NodeOccurrenceMatrix.svg'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)



#####################################
#                                   #
# 4. SBM BLOCKS                     #
#                                   #
#####################################

# ---- define local parameters ---- #
cmap = utils.make_cmap([brain_colour, roi_colour])
surface_opacity = 0.05
brain_opacity = 0.05
roi_opacity = 0.3



task = 'Foreperiod_Long_tau'
for task in TASKS:
    task=task.split('_t')[0]
    for m in MODELS:
        for layers in [0,1]:
            block_img = nib.load(os.path.join(result_dir, task, f'Schaefer2018-400_{m}_lvl{layers}_blockvalues.nii.gz'))

            fig, axes = utils.plot_block_surface(block_img, cmap='plasma', surface_opacity=surface_opacity, brain_opacity=brain_opacity, roi_opacity=roi_opacity)

            plt.suptitle(f'SBM block z-scores — {task} (layer {layers})', fontsize=14)
            plt.tight_layout()

            out_path = os.path.join(out_dir, f'SBM_blocks_surf_{task}_{m}_layer{layers}.svg')
            plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f'Saved → {out_path}')
            plt.close(fig)


# ---- per-block ROI membership (single flat colour, no value coding) ---- #
# run_recon_pnb.py already computes and saves block_zscore per ROI in
# roi_assignment_summary_{model}_lvl{level}.tsv -- read that directly to
# select blocks (z > 0.5). Plots pure membership (block_membership_image),
# not edge weight: block_edge_weight_image only looks at intra-block
# edges, which a real, behaviourally-relevant block can legitimately have
# none of (SBM blocks group nodes by similar connectivity PROFILE, not by
# mutual connection) -- that showed as an empty surface for many blocks.

roi_opacity = 0.5
for task in TASKS:
    task_short = task.split('_t')[0]
    for m in MODELS:
        for level in [0, 1]:
            assignments_path = os.path.join(result_dir, task_short, f'roi_assignment_summary_{m}_lvl{level}.tsv')
            with open(assignments_path, newline='') as fh:
                rows = list(csv.DictReader(fh, delimiter='\t'))

            block_of_node = np.array([int(row['block']) for row in rows])
            block_zscore  = np.array([float(row['block_zscore']) for row in rows])

            relevant_blocks = np.unique(block_of_node[block_zscore > 0.5])

            for blk in relevant_blocks:
                block_img = utils.block_membership_image(block_of_node, blk, atlas_img)

                fig, axes = utils.plot_block_surface(block_img, cmap='plasma_r', surface_opacity=surface_opacity,
                                                      brain_opacity=brain_opacity, roi_opacity=roi_opacity,
                                                      positive_only=True)

                plt.suptitle(f'Block ROI members — {task_short} {m} (level {level}, block {blk})', fontsize=14)
                plt.tight_layout()

                out_path = os.path.join(out_dir, f'SBM_block_members_{task_short}_{m}_lvl{level}_block{blk}.svg')
                plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
                print(f'Saved → {out_path}')
                plt.close(fig)




#####################################
#                                   #
# 5. SBM COMMUNITIES                #
#                                   #
#####################################

# ---- define local parameters ---- #
cmap = utils.make_cmap([brain_colour, roi_colour])

# ---- plot the block communities of each recon_pnb graph ---- #
# run_recon_pnb.py has no single "final joint" graph (no behaviour/
# cooccurrence layers to collapse) -- it fits one single-layer graph per
# model variant (no_beh / beh_weighted) instead, so this now runs once per
# task x model rather than once per task. block_of_node comes straight from
# roi_assignment_summary_{m}_lvl{level}.tsv's own 'block' column (already
# fitted by run_recon_pnb.py) for both levels, matching
# nested_bs_from_node_levels' list-of-per-level-arrays convention.
grouped_raw = np.genfromtxt(os.path.join(path, 'ATLAS', f'{atlas}_areas_grouped.txt'), dtype=str, delimiter='\t')
node_groups = grouped_raw[1:, 2]

for task in TASKS:
    task_short = task.split('_t')[0]
    # relevance is shared across model variants within a task (same
    # raw_vals/behaviour used to fit both no_beh and beh_weighted) -- one
    # load per task, reused for every model below.
    relevance_path = os.path.join(result_dir, task_short, 'recon_pnb_region_behaviour_relevance.npy')
    relevance = np.load(relevance_path)

    for m in MODELS:
        graph_path = os.path.join(result_dir, task_short, f'recon_pnb_{m}_graph.gt')
        adj = utils.load_pnb_adjacency(graph_path)

        block_of_node = []
        for level in [0, 1]:
            assignments_path = os.path.join(result_dir, task_short, f'roi_assignment_summary_{m}_lvl{level}.tsv')
            with open(assignments_path, newline='') as fh:
                rows = list(csv.DictReader(fh, delimiter='\t'))
            block_of_node.append(np.array([int(row['block']) for row in rows]))

        output_prefix = os.path.join(out_dir, f'SBM_final_state_{task_short}_{m}')
        state = utils.plot_sbm_state(adj, node_groups, output_prefix, cmap='plasma', arrow_colour='black',
                                      block_of_node=block_of_node, relevance=relevance)