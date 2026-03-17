import os
os.chdir('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import sys
sys.path.append('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import pandas as pd
from scipy.stats import linregress

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ip02')
import cmasher as cm
from matplotlib.patches import Patch


from rmse import rmse

from OIMAS import OIMAS_N

#%% load biomass data
agb_fn = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/biomass/AGB2025.csv"
agb = pd.read_csv(agb_fn, skiprows=1, index_col=1)

bgb_fn = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/biomass/BGB2025.csv"
bgb = pd.read_csv(bgb_fn, skiprows=1, index_col=1)

#%% translate to mass per m2
footprint_agb = .4**2
footprint_bgb = .05**2 * np.pi

# kg as units
agb['mass_m2'] =  agb['mass[g]'] / footprint_agb / 1000
bgb['mass_m2'] =  bgb['mass'] / footprint_bgb / 1000

# remove bare sites
agb = agb[agb['primary_plant'] != 'bare']
bgb = bgb[bgb['primary_plant'] != 'bare']

# add depth
bgb['depth'] = [(float(d.split('-')[-1][:-2]) - 5)/100 for d in bgb["Depth"]]
bgb['mass_m3'] = bgb['mass_m2'] * 0.1

#%% prepare grouping info
depths = np.sort(bgb['depth'].unique())
plants = agb['primary_plant'].unique()

# colors per plant
plant_colors = dict(zip(
    plants,
    [f'C{len(plants)-i}' for i in range(len(plants))]
))

#%% plot above ground biomass

fig, axs = plt.subplots(ncols = 4, nrows = 4, figsize = (15,8.5), height_ratios = [.3, .3, 1, .3])

# group dataframe per age zone
grouped_per_zone =  agb.groupby('zone')

# customize boxplot properties
boxprops = dict(linestyle='-', linewidth=0, color='w')
medianprops = dict(linestyle='-', linewidth=1, color='w')
whiskerprops = dict(linestyle='-', linewidth=1, color='w')
capprops = dict(linewidth=0)

for i, (group_i, group) in enumerate(grouped_per_zone):

    # group locations per species
    grouped_by_species = group.groupby('primary_plant')

    # plot boxplot
    bp = grouped_by_species.boxplot(ax = axs[0,i], subplots = False, column = 'mass_m2',
                         boxprops = boxprops, medianprops = medianprops,
                         whiskerprops = whiskerprops, capprops = capprops,
                         rot = 0, vert = False, patch_artist = True)

    plants_per_zone = grouped_by_species.groups.keys()

    for patch, plant in zip(bp.patches, plants_per_zone):
        patch.set_facecolor(plant_colors[plant])


    axs[0,i].set_yticklabels(['\n'.join(t.get_text().split(',')[0][1:].split(' ')) for t in axs[0,i].get_yticklabels()],
                             fontsize = 8)
    axs[0,i].set_xlim(-0.3,6)
    axs[0,i].set_xlabel(r'above-ground biomass [$kg / m^2$]')

    axs[0,i].set_title(group_i)

#%% plot total below ground biomass

bgb_int = bgb.groupby(bgb.index).agg({'mass_m2': lambda x: x.sum(skipna=True),
                                      'zone': lambda x: x.iloc[0],
                                      'primary_plant': lambda x: x.iloc[0]})

# group dataframe per age zone
grouped_per_zone =  bgb_int.groupby('zone')

for i, (group_i, group) in enumerate(grouped_per_zone):

    # group locations per species
    grouped_by_species = group.groupby('primary_plant')

    # plot boxplot
    bp = grouped_by_species.boxplot(ax = axs[1,i], subplots = False, column = 'mass_m2',
                         boxprops = boxprops, medianprops = medianprops,
                         whiskerprops = whiskerprops, capprops = capprops,
                         rot = 0, vert = False, patch_artist = True)

    plants_per_zone = grouped_by_species.groups.keys()

    for patch, plant in zip(bp.patches, plants_per_zone):
        patch.set_facecolor(plant_colors[plant])


    axs[1,i].set_yticklabels(['\n'.join(t.get_text().split(',')[0][1:].split(' ')) for t in axs[1,i].get_yticklabels()],
                             fontsize = 8)
    axs[1,i].set_xlim(-0.3,6)
    axs[1,i].set_xlabel(r'total below-ground biomass [$kg / m^2$]')

#%% below-ground biomass grouped by depth and primary plant

grouped_per_zone = bgb.groupby('zone')

# layout control
box_width = 0.025

for i, (zone_i, group) in enumerate(grouped_per_zone):

    ax = axs[2, i]

    for d in depths:
        depth_group = group[group['depth'] == d]

        off = -0.02
        for plant in plants:
            plant_group = depth_group[depth_group['primary_plant'] == plant]

            if plant_group.empty:
                continue

            ax.boxplot(
                plant_group['mass_m3'],
                positions=[d + off],
                widths=box_width,
                vert=False,
                patch_artist=True,
                boxprops=dict(
                    facecolor=plant_colors[plant],
                    edgecolor='none',
                    linewidth=1
                ),
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops
            )

            off += .04

    # axis formatting
    ax.invert_yaxis()
    ax.set_ylim(depths.max() + 0.3, depths.min() - 0.3)
    ax.set_yticks(depths)
    ax.set_yticklabels([])
    ax.set_xlim(-0.05, .3)
    ax.set_ylim(.75, 0)
    ax.set_xlabel(r'below-ground biomass [$kg / m^3$]')

# depth labels only on first column
axs[2, 0].set_yticklabels(np.unique(bgb['Depth']))
axs[2, 0].set_ylabel('depth')

# legend for primary plant
legend_handles = [
    Patch(facecolor=plant_colors[p], edgecolor='none', label=p)
    for p in plants
]

axs[0, -1].legend(
    handles=legend_handles,
    title='primary species',
    frameon=False,
    bbox_to_anchor=(1.05, 1.0)
)

fig.tight_layout()

#%% combine below-ground and above-ground biomass and plot root-shoot ratios

df = agb[['mass_m2']].join(bgb_int, lsuffix= 'agb', rsuffix= 'bgb', how = 'inner').dropna()
df['root-shoot'] = df['mass_m2bgb'] / df['mass_m2agb']


# group dataframe per age zone
grouped_per_zone =  df.groupby('zone')

for i, (group_i, group) in enumerate(grouped_per_zone):

    # group locations per species
    grouped_by_species = group.groupby('primary_plant')

    # plot boxplot
    bp = grouped_by_species.boxplot(ax = axs[3,i], subplots = False, column = 'root-shoot',
                         boxprops = boxprops, medianprops = medianprops,
                         whiskerprops = whiskerprops, capprops = capprops,
                         rot = 0, vert = False, patch_artist = True)

    plants_per_zone = grouped_by_species.groups.keys()

    for patch, plant in zip(bp.patches, plants_per_zone):
        patch.set_facecolor(plant_colors[plant])


    axs[3,i].set_yticklabels(['\n'.join(t.get_text().split(',')[0][1:].split(' ')) for t in axs[3,i].get_yticklabels()],
                             fontsize = 8)
    axs[3,i].set_xlim(-0.1,3)
    axs[3,i].set_xlabel(r'root-shoot ratio')

#%% get statistics

# group dataframe per age zone
grouped_per_zone =  agb.groupby('zone')

for i, (group_i, group) in enumerate(grouped_per_zone):
    median_agb = group.groupby("primary_plant").agg({'mass_m2': lambda x: x.median(skipna=True)})
    print(group_i)
    print(median_agb)
    print('------------')

# group dataframe per age zone
grouped_per_zone =  df.groupby('zone')

for i, (group_i, group) in enumerate(grouped_per_zone):
    median_rs = group.groupby("primary_plant").agg({'root-shoot': lambda x: x.median(skipna=True)})
    print(group_i)
    print(median_rs)
    print('------------')

# group dataframe per age zone
grouped_per_zone = df.groupby('zone')

for i, (group_i, group) in enumerate(grouped_per_zone):
    median_bgb = group.groupby("primary_plant").agg({'mass_m2bgb': lambda x: x.median(skipna=True)})
    print(group_i)
    print(median_bgb)
    print('------------')

#%% save df as summary
df.to_csv("/Users/ignace/Documents/WETCOAST/Data/Saefthinge/biomass/summary.csv")