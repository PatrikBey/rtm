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
# The following script generates figures of the current project.        #
#                                                                       #
# Generated figures include:                                            #
# * 1. lesion distribution                                              #
# * 2. reaction-time distributions / vs lesion volume                   #
# * 3. graph-layers (co-occurrence & behaviours)                        #
# * 4. joint fitted layer                                               #
# * 5. mapping of z-scores for tasks                                    #
# * 6. mapping of z-scores for validation                               #
# * 7. validation substrate mappings                                    #
#                                                                       #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/09/16                                               #
#                                                                       #
#                                                                       #
#########################################################################


#################################
#      prepare environment      #
#################################

import numpy as np, nilearn.plotting, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



#################################
#      prepare global variables      #
#################################

bright_colour = '#30d6ff'
dark_colour = '#361a54'
mig_colour = '#ba82ff'

pale2blue = ['#eedeff','#ddbdff','#ba82ff','#30d6ff']
pale2bluecmap = LinearSegmentedColormap.from_list('pale2blue', pale2blue)



part = np.genfromtxt('/data/patrik/RT/RTM/participants.tsv', dtype=str, delimiter='\t')
features = part[0,:].astype(str).tolist()


#######################################################################
#               2. reaction-time vs lesion volume
#######################################################################

x = part[1:,features.index('lesion_volume_mm3')]
y_gonogo = part[1:,features.index('gonogo_median_rt')]
y_foreperiod = part[1:,features.index('foreperiod_median_rt')]

x_gonogo, vals_gonogo, x_foreperiod, vals_foreperiod = [], [], [], []

for i in range(len(x)):
    if y_gonogo[i] == '' or y_foreperiod[i] == '' or x[i] == '':
        continue
    else:
        tmp_x = float(x[i])
        tmp_gonogo = float(y_gonogo[i])
        tmp_foreperiod = float(y_foreperiod[i])
        plt.scatter(tmp_x, tmp_gonogo, color=bright_colour, label='Go/No-go' if i == 1 else "")
        plt.scatter(tmp_x, tmp_foreperiod, color=dark_colour, label='Foreperiod long' if i == 1 else "")
        x_gonogo.append(tmp_x)  
        vals_gonogo.append(tmp_gonogo)
        x_foreperiod.append(tmp_x)
        vals_foreperiod.append(tmp_foreperiod)

for tmp_x, tmp_y, colour in [(x_gonogo, vals_gonogo, bright_colour), (x_foreperiod, vals_foreperiod, dark_colour)]:
    slope, intercept = np.polyfit(tmp_x, tmp_y, 1)
    fit_x = np.array([min(tmp_x), max(tmp_x)])
    plt.plot(fit_x, slope * fit_x + intercept, color=colour, linewidth=2)

plt.legend()
plt.xlabel(f'Lesion Volume (mm³)')
plt.ylabel(f'Reaction Time (ms)')
# plt.ylabel(f'Reaction Time (ms)')
plt.suptitle(f'Reaction Time vs Lesion Volume')
plt.title(f'| Foreperiod long r:{np.corrcoef(x_foreperiod, vals_foreperiod)[0,1]:.2f} | Go/No-go r:{np.corrcoef(x_gonogo, vals_gonogo)[0,1]:.2f} |')
plt.show()
