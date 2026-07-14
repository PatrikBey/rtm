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

import numpy as np, os
import nibabel as nib
import fury
import matplotlib.pyplot as plt

import utils







#################################
#      input file paths         #
#################################

path = '/mnt/h/RT/data/'
result_dir = os.path.join(path,'RESULTS/split_threshold')


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
    plt.imshow(np.where(tmp==0,np.nan,tmp), cmap = cmap, interpolation='quadric', origin='lower', alpha = .66)
    plt.clim((0, lesion_img.max()))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # plt.colorbar()
    # plt.title(f'Slice {i}')

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(path, 'LesionDistribution.svg'), dpi=150, bbox_inches='tight', facecolor='white')
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
brain_color = "#B3B1B1"
brain_opacity = 0.05
roi_color = "#D3238A"
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
    utils.plot_glass_surface(ax, glass_meshes, color=utils.hex_to_rgb(brain_color), opacity=brain_opacity)
    utils.plot_tracts(ax, streamlines, colors=utils.hex_to_rgb(streamline_color),
                       opacity=streamline_opacity, linewidth=streamline_linewidth)
    utils.plot_mask(ax, lesion_file, color=utils.hex_to_rgb(roi_color), opacity=roi_opacity)
    utils.finalize_view(ax, view_name, elev, azim, brain_aspect)

plt.suptitle('Disconnectome example', fontsize=14)
plt.tight_layout()

out_path = os.path.join(path, 'example', 'disconnectome_example.svg')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved → {out_path}')
plt.close(fig)


#####################################
#                                   #
# 3. GRAPH LAYERS                   #
#                                   #
#####################################


# ---- glass surface + streamlines + volumetric ROI ---- #
glass_meshes, brain_aspect = utils.load_glass_surface()


# ---- plot brain mapping layers ---- #
coords = np.genfromtxt(os.path.join(path,'ATLAS',f'{atlas}_coords.txt'), delimiter=' ')

layers = dict()

fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

for ax, t in zip(axes[:3], TASKS):
    utils.plot_glass_surface_2d(ax, glass_meshes, color=utils.hex_to_rgb(back_colour), opacity=0.05)
    layers[f'{t}-beh'] = np.genfromtxt(os.path.join(result_dir,f'SBM_{atlas}_{t}_singleflip',f'SBM_layer_{t}_behaviour.txt'), delimiter=' ')
    utils.plot_graph(ax, layers[f'{t}-beh'],coords[:,:2], colours = ['midnightblue','crimson','gold'])
    ax.set_aspect('equal')
    ax.axis('off')

last_task = TASKS[-1]
ax = axes[3]
utils.plot_glass_surface_2d(ax, glass_meshes, color=utils.hex_to_rgb(back_colour), opacity=0.05)
layers[f'{last_task}-les'] = np.genfromtxt(os.path.join(result_dir,f'SBM_{atlas}_{last_task}_singleflip',f'SBM_layer_{last_task}_cooccurrence.txt'), delimiter=' ')
utils.plot_graph(ax, layers[f'{last_task}-les'],coords[:,:2], colours = ['midnightblue','crimson','gold'])
ax.set_aspect('equal')
ax.axis('off')

plt.savefig(os.path.join(path, 'GraphLayers.svg'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)


# --- plot layers in circular layout --- #
coords = np.genfromtxt(os.path.join(path,'ATLAS',f'{atlas}_circle_coords_sorted.txt'), delimiter='\t')[1:,:]
# coords = np.genfromtxt(os.path.join(path,'ATLAS',f'{atlas}_coords.txt'), delimiter=' ')

layers = dict()

fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

for ax, t in zip(axes[:3], TASKS):
    layers[f'{t}-beh'] = np.genfromtxt(os.path.join(result_dir,f'SBM_{atlas}_{t}_singleflip',f'SBM_layer_{t}_behaviour.txt'), delimiter=' ')
    utils.plot_graph(ax, layers[f'{t}-beh'],coords[:,:2], colours = ['midnightblue','crimson','gold'], node_size = 200, top_pct = .5)
    ax.set_aspect('equal')
    ax.axis('off')

last_task = TASKS[-1]
ax = axes[3]
layers[f'{last_task}-les'] = np.genfromtxt(os.path.join(result_dir,f'SBM_{atlas}_{last_task}_singleflip',f'SBM_layer_{last_task}_cooccurrence.txt'), delimiter=' ')
utils.plot_graph(ax, layers[f'{last_task}-les'],coords[:,:2], colours = ['midnightblue','crimson','gold'], node_size = 200, top_pct = .5)
ax.set_aspect('equal')
ax.axis('off')

plt.savefig(os.path.join(path, 'GraphLayersCircle.svg'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)






#####################################
#                                   #
# 4. SBM BLOCKS                     #
#                                   #
#####################################