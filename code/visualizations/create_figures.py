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
result_dir = os.path.join(path,'RESULTS/')
out_dir = os.path.join(path,'RESULTS','FIGURES')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

TASKS = ['Foreperiod_Long_tau','GoNoGO_tau','SATO_Accuracy_tau']


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

# ---- define local parameters ---- #
surface_opacity = 0.05
streamline_color = "#494949"
streamline_opacity = 0.1
streamline_linewidth = 0.75
brain_opacity = 0.05
roi_opacity = 0.3

# ---- load local files ---- #
lesion_file = os.path.join(path, 'example','example_lesion.nii.gz')
tck_file = os.path.join(path, 'example','example_tracts_1K.tck')

tractogram = load_tractogram(tck_file, lesion_file, to_space=Space.RASMM)
streamlines = Streamlines(tractogram.streamlines)

# ---- glass surface + streamlines + volumetric ROI ---- #
glass_meshes, brain_aspect = utils.load_glass_surface()

fig, axes = utils.setup_views_figure()

for ax, (view_name, elev, azim) in zip(axes, utils.VIEWS):
    utils.plot_glass_surface(ax, glass_meshes, color=utils.hex_to_rgb(back_colour), opacity=brain_opacity)
    utils.plot_tracts(ax, streamlines, colors=utils.hex_to_rgb(streamline_color),
                       opacity=streamline_opacity, linewidth=streamline_linewidth)
    utils.plot_mask(ax, lesion_file, color=utils.hex_to_rgb(roi_colour), opacity=roi_opacity)
    utils.finalize_view(ax, view_name, elev, azim, brain_aspect)

plt.suptitle('Disconnectome example', fontsize=14)
plt.tight_layout()

out_path = os.path.join(out_dir, 'disconnectome_example.svg')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out_path}')
plt.close(fig)


#####################################
#                                   #
# 3. GRAPH LAYERS                   #
#                                   #
#####################################

# ---- define local parameters ---- #
cmap = utils.make_cmap([brain_colour, roi_colour])

# ---- fit and plot a single SBM to one layer's adjacency matrix ---- #
# no MCMC annealing, no multi-layer model: just graph_tool's own
# minimize_nested_blockmodel_dl on a plain single-layer graph.
# task = 'Foreperiod_Long_tau'
for task in TASKS:
    for layer in ['behaviour', 'cooccurrence']:
        adj_path = os.path.join(result_dir, f'SBM_{atlas}_{task}_singleflip', f'SBM_layer_{task}_{layer}.txt')
        adj = np.genfromtxt(adj_path, delimiter=' ')
        grouped_raw = np.genfromtxt(os.path.join(path, 'ATLAS', f'{atlas}_areas_grouped.txt'), dtype=str, delimiter='\t')
        node_groups = grouped_raw[1:, 2]
        output_prefix = os.path.join(out_dir, f'SBM_state_{task}_{layer}')
        state = utils.plot_sbm_state(adj, node_groups, output_prefix, cmap=cmap, arrow_colour='gold')




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
    for layers in [0,1]:
        block_img = nib.load(os.path.join(result_dir, f'SBM_{atlas}_{task}_singleflip', f'Schaefer2018-400_lvl{layers}_blockzscores_{task}.nii.gz'))

        fig, axes = utils.plot_block_surface(block_img, cmap='plasma', surface_opacity=surface_opacity, brain_opacity=brain_opacity, roi_opacity=roi_opacity)

        plt.suptitle(f'SBM block z-scores — {task} (layer {layers})', fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f'SBM_blocks_surface_{task}_layer{layers}.svg')
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved → {out_path}')
        plt.close(fig)


# ---- per-block mean edge weight, normalized within each block (no z-score) ---- #
roi_opacity = 0.5
for task in TASKS:
    graph_path = os.path.join(result_dir, f'SBM_{atlas}_{task}_singleflip', f'SBM_final_graph_{task}.gt')
    adj = utils.load_joint_adjacency(graph_path)

    assignments_path = os.path.join(result_dir, f'SBM_{atlas}_{task}_singleflip', f'roi_block_assignments_{task}.csv')
    with open(assignments_path, newline='') as fh:
        rows = list(csv.DictReader(fh))

    for level in [0, 1]:
        level_col = f'level_{level}'
        if level_col not in rows[0]:
            continue
        block_of_node = np.array([int(row[level_col]) for row in rows])
        behaviour_degree = np.array([float(row['behaviour_degree']) for row in rows])

        blocks = np.unique(block_of_node)
        counts = np.array([(block_of_node == blk).sum() for blk in blocks])
        block_means = np.array([behaviour_degree[block_of_node == blk].mean() for blk in blocks])
        mean_std = block_means.std()
        block_zscores = (block_means - block_means.mean()) / mean_std if mean_std > 0 else np.zeros_like(block_means)

        relevant_blocks = blocks[(block_zscores >= 0) & (counts > 1)]

        for blk in relevant_blocks:
            block_img = utils.block_edge_weight_image(adj, block_of_node, blk, atlas_img)

            if not np.any(block_img.get_fdata() > 0):
                print(f'Skipped (no intra-block edges) — {task} lvl{level} block{blk}')
                continue

            fig, axes = utils.plot_block_surface(block_img, cmap='plasma', surface_opacity=surface_opacity,
                                                  brain_opacity=brain_opacity, roi_opacity=roi_opacity,
                                                  positive_only=True)

            plt.suptitle(f'Mean edge weight — {task} (level {level}, block {blk})', fontsize=14)
            plt.tight_layout()

            out_path = os.path.join(out_dir, f'SBM_block_edgeweights_{task}_lvl{level}_block{blk}.svg')
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

# ---- fit and plot the block communities of the final joint graph ---- #
# same as section 3, except one adjacency per task (behaviour +
# cooccurrence collapsed into the final MCMC-fitted multilayer graph)
# instead of two separate per-layer maps.
for task in TASKS:
    graph_path = os.path.join(result_dir, f'SBM_{atlas}_{task}_singleflip', f'SBM_final_graph_{task}.gt')
    adj = utils.load_joint_adjacency(graph_path)
    grouped_raw = np.genfromtxt(os.path.join(path, 'ATLAS', f'{atlas}_areas_grouped.txt'), dtype=str, delimiter='\t')
    node_groups = grouped_raw[1:, 2]

    assignments_path = os.path.join(result_dir, f'SBM_{atlas}_{task}_singleflip', f'roi_block_assignments_{task}.csv')
    with open(assignments_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    level_cols = sorted((c for c in rows[0] if c.startswith('level_')), key=lambda c: int(c.split('_')[1]))
    block_of_node = [np.array([int(row[c]) for row in rows]) for c in level_cols]

    output_prefix = os.path.join(out_dir, f'SBM_final_state_{task}_joint')
    state = utils.plot_sbm_state(adj, node_groups, output_prefix, cmap='plasma', arrow_colour='gold',
                                  block_of_node=block_of_node)

