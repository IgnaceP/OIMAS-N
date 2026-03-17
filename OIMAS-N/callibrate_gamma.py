import os
os.chdir('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import sys
sys.path.append('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import pandas as pd
from scipy.stats import linregress

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import cmasher as cm

from rmse import rmse

from OIMAS import OIMAS_N
from read_C_obs_data import read_observation_data

matplotlib.style.use('ip01')

#%% load below-ground biomass
fn_bgb              = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/biomass/BGB2025.csv"
bgb                 = pd.read_csv(fn_bgb, skiprows=1, index_col=1)
footprint_bgb       = .05**2 * np.pi
bgb['mass_m2']      =  bgb['mass'] / footprint_bgb / 1000
bgb                 = bgb[bgb['primary_plant'] != 'bare']
bgb['depth']        = [(float(d.split('-')[-1][:-2]) - 5)/100 for d in bgb["Depth"]]
bgb.sort_values('primary_plant', inplace=True)

# load above-ground biomass
agb_fn              = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/biomass/summary.csv"
agb                 = pd.read_csv(agb_fn, skiprows=0, index_col=0)

total_RMSE = {'Atriplex prostrata': [], 'Bolboschoenus maritimus': [], 'Elytrigia atherica': [], 'Tripolium pannonicum': []}

#%% initialize figure

fig, axs = plt.subplots(ncols = 5, nrows = 6, figsize = (8,10), sharex = True, sharey = True)
axs = axs.flatten()
best_gamma = []
gamma = {'Atriplex prostrata': 0.17, 'Bolboschoenus maritimus': 0.2, 'Elytrigia atherica': 0.2, 'Tripolium pannonicum': 0.1}



RMSE_cores = {}

ax_count = 0

for i, plant in enumerate(['Atriplex prostrata', 'Bolboschoenus maritimus','Elytrigia atherica', 'Tripolium pannonicum']):

    # get observation data on belowground biomass
    cores_plant = bgb[bgb['primary_plant'] == plant]
    cores_plant["id"] = [int(i.split('y')[-1]) for i in cores_plant.index]

    # plot belowground biomass
    for id in np.unique(cores_plant.index):

        # get the core
        core = cores_plant.loc[id]
        core.sort_values('depth', inplace=True)

        # plot the mass per m3 from observations
        axs[ax_count].plot(core['mass_m2'] / .1, -1 * core.depth, marker = 'o', markersize = 3, c = 'grey', zorder = 1e9)

        # get the biomass of the upper layer
        b0 = core['mass_m2'].iloc[0]
        sim = b0 * np.exp(-(core['depth']-.05)/gamma[plant])

        # plot the mass per m3 from simulation
        axs[ax_count].plot(sim / .1, -1 * core.depth, marker = 'o', markersize = 3, c = ['C0','C1','C2','C3'][i], zorder = 1e9)
        axs[ax_count].text(0.55, 0.1, r'$\gamma:$ %.2f m' % gamma[plant], transform = axs[ax_count].transAxes, fontsize = 11)

        # calculate RMSE
        RMSE = rmse(core['mass_m2'] / .1, sim / .1)
        total_RMSE[plant].append(RMSE)
        RMSE_cores[core.index[0]] = RMSE

        # update ax title
        axs[ax_count].set_title(core.index[0] + '\n' + plant, fontsize = 8)

        ax_count += 1

for a in axs[25:]:
 a.set_xlabel('mass [kg/m3]')
for a in axs[[0, 5, 10, 15, 20, 25]]:
 a.set_ylabel('depth [m]')

fig.tight_layout()
fig.savefig('callibration_output/callibration_gamma.png', dpi = 300)

mean_RMSE = np.mean(list(RMSE_cores.values()))
mean_atri_RMSE = np.mean(total_RMSE['Atriplex prostrata'])
mean_bol_RMSE = np.mean(total_RMSE['Bolboschoenus maritimus'])
mean_ely_RMSE = np.mean(total_RMSE['Elytrigia atherica'])
mean_trip_RMSE = np.mean(total_RMSE['Tripolium pannonicum'])
print('gamma: ', gamma)
print('Atriplex RMSE', '%.3f' % mean_atri_RMSE)
print('Bolboschoenus RMSE', '%.3f' % mean_bol_RMSE)
print('Elytrigia RMSE', '%.3f' % mean_ely_RMSE)
print('Tripolium RMSE', '%.3f' % mean_trip_RMSE)
print('mean RMSE', '%.3f' % mean_RMSE )
print('----------------------------')
