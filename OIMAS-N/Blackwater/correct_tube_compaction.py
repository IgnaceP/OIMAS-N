import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.style.use("ip02")

#%% get compaction measurements
df = pd.read_excel("/Users/ignace/Documents/WETCOAST/Data/Blackwater/Mona/compaction_raw.xlsx", index_col=0)
df['auger'] = df.index
df["D"] = (60 - df["outer_length_cm"]) / 100
df["I"] = (60 - df["inner_length_cm"]) / 100
df["dI"] = df['D'] - df['I']

augers = df.groupby("auger")

# initiate figure
fig, ax = plt.subplots(nrows = 3, ncols = 6, sharex=True, sharey=True)
ax = ax.flatten()
ax_i = 0

def exponential(x, a, b):
    return a * b ** x

poly_coeff = {}
ax_id = {}
augers_comp_dbds = {}

# loop over all augers
for auger_id, auger in augers:
    # reset index
    auger.index = auger['D']

    # sort along depth
    auger.sort_index(inplace=True)

    # calculate segment lengths and incremental compaction depth
    auger['L'] = auger['D'].diff().fillna(auger['D'].iloc[0])
    auger['dC'] = auger['dI'].diff().fillna(auger['dI'].iloc[0])
    auger['dC/D'] = auger['dC'] / auger['D']

    # summation
    auger['S_dC/D'] = auger['dC/D'][::-1].cumsum()

    # segment length
    auger['l'] = auger['L'] * (1 - auger['S_dC/D'])

    # segmentation compaction
    auger['compaction'] = auger['l']/auger['L']

    # plot
    ax[ax_i].plot(100 * auger['compaction'], -1*auger['D'], marker = 'o')
    ax[ax_i].set_title(auger_id)
    ax[ax_i].grid(True)

    augers_comp_dbds[auger_id] = auger
    ax_id[auger_id] = ax_i

    ax_i += 1


[ax[i].set_xlabel("compaction [%]") for i in range(12,18)]
[ax[i].set_ylabel("depth [m]") for i in[0,6,12]]

#%% apply compaction correction to DBD

dbd_df = pd.read_excel('/Users/ignace/Documents/WETCOAST/Data/Blackwater/Mona/BD calculation_nocomp.xlsx', sheet_name="BD2", skiprows=1)

# keep true interval boundaries in meters
dbd_df['depth_min_m'] = dbd_df['depth_min'] / 100
dbd_df['depth_max_m'] = dbd_df['depth_max'] / 100
dbd_df['thickness_m'] = dbd_df['depth_max_m'] - dbd_df['depth_min_m']
dbd_df['segment_m'] = dbd_df['depth_max_m'].diff()
dbd_df.iloc[0, dbd_df.columns.get_loc('segment_m')] = dbd_df['depth_max_m'].iloc[0]  # assumes core top = 0


dbd_df['depth'] = (dbd_df['depth_max'] + dbd_df['depth_min']) / 2
dbd_df.index = dbd_df['depth'] / 100

#dbd_df = dbd_df[[c for c in dbd_df.columns if c.startswith('D') or c.startswith('depth') or c.startswith('depth')]]

fig, axs = plt.subplots(nrows = 3, ncols = 3, sharex=True, sharey=True)
axs = axs.flatten()

dfs = []

i = 0
for _, auger_id in enumerate(dbd_df.columns):
    if auger_id.startswith('D'):
        # load auger
        auger_comp = augers_comp_dbds[auger_id]

        # interpolate compaction percentages
        auger_sorted = auger_comp.sort_values('I')
        compaction_perc = np.interp(dbd_df.index, auger_comp['I'], 100 * auger_comp['compaction'])

        # correct DBD
        dbd_comp = 1000*dbd_df[auger_id] * (compaction_perc/100)

        # decompact each interval using the sample SPACING, not the thin slice thickness
        dbd_segment_corrected = dbd_df['segment_m'] / (compaction_perc / 100)
        dbd_depth_bottom = dbd_segment_corrected.cumsum()  # true depth at bottom of each represented interval
        dbd_depth = dbd_depth_bottom - dbd_segment_corrected / 2  # true depth at interval midpoint, for plotting

        # create new dataframe
        new_df_index = [auger_id+f"-{i}" for i in 1+np.arange(len(dbd_df[auger_id].dropna())) ]
        dbd_comp = pd.DataFrame({'dbd_comp': dbd_comp.values, 'comp_per': compaction_perc}, index = dbd_depth)
        corr_dbd = dbd_comp.dropna()
        new_df = pd.DataFrame({"depth_corr": corr_dbd.index,
                                     "DBD_corr": corr_dbd['dbd_comp'].values,
                                     "comp_per":corr_dbd['comp_per'].values},
                                    index = new_df_index)
        dfs.append(new_df)

        # plot
        ax_i = ax[ax_id[auger_id]]
        ax_i.plot(compaction_perc, -1 * dbd_df.index, marker='o', c='C1')

        ax_i_dbd = axs[i]
        ax_i_dbd.set_xlim(0, 1000)
        ax_i_dbd.scatter(1000 * dbd_df[auger_id], -1 * dbd_df.index, marker='D', c='C1', alpha=0.25,
                         label='without correction')

        ax_i_dbd.plot(corr_dbd.values, -1 * corr_dbd.index, marker='D', c='C1', label='with correction')
        ax_i_dbd.set_title(auger_id)

        real_depth = -1 * df[df['auger'] == auger_id]['D'].max()
        ax_i_dbd.hlines(real_depth, xmin=0, xmax=1000, ls='--', color='w', label='real depth')
        if i == 8:
            ax_i_dbd.legend()
        i += 1

[axs[i].set_xlabel(r"dry bulk density [$kg/m^3$]") for i in range(6,9)]
[axs[i].set_ylabel("depth [m]") for i in[0,3,6]]

new_df = pd.concat(dfs)
new_df.to_csv('/Users/ignace/Documents/WETCOAST/Data/Blackwater/Mona/corrected_DBD.csv')

#%% update raw
fn = "/Users/ignace/Documents/WETCOAST/Data/Blackwater/Mona/raw.csv"
df_raw = pd.read_csv(fn, sep = ";")
df_raw.index = df_raw['Sample']

df_raw_updated = pd.merge(df_raw, new_df, how='left', left_index=True, right_index=True).dropna()

df_raw_updated.to_csv('/Users/ignace/Documents/WETCOAST/Data/Blackwater/Mona/all_data.csv', index=False)