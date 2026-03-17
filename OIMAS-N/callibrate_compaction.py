import os
os.chdir('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import sys
sys.path.append('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import pandas as pd
from scipy.stats import linregress
import scipy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cmasher as cm

from rmse import rmse

from OIMAS import OIMAS_N

matplotlib.style.use('ip01')

#%% define function to load observations data and calculate void ratios

def calc_void(fn):
    df = pd.read_csv(fn, skiprows=1, index_col=0)
    df['depth'] = [float(d.split('-')[-1][:-2]) - 2.5 for d in df["Depth"]]
    df['auger'] = [i[-1] for i in df.index]

    rho_min = 2600
    rho_om = 1300

    # using OCC = 0.5208 * SOM + 1.17 (Ouyang & Lee, 2020)
    # diameter auger is 6 cm
    df['mass_m2']           = 0.05 * 1000 * df['DBD']
    df['mass']              = 0.05 * np.pi * 0.03**2 * 1000 * df['DBD']
    df['Cmass_m2']          = df['mass_m2'] * df['C_percentage'] / 100
    df['om_percentage']     = df['C_percentage']/.52
    df['om']                = df['mass_m2'] * df['om_percentage'] / 100
    df['min']               = df['mass_m2'] - df['om']
    df['volume']            = 0.05 * np.pi * 0.03**2
    df['rho_solid']         = (df['min'] * rho_min + df['om'] * rho_om) / (df['mass_m2'])
    df['solid_volume']      = df['mass'] / df['rho_solid']
    df['void_ratio']        =( df['volume'] - df['solid_volume']) / df['solid_volume']
    df_top                  = df[df['depth'] == 2.5]
    E0                      = np.median(df_top['void_ratio'])
    C0                      = np.median(df_top['C_percentage'])
    DBD0                    = 1000 * np.median(df_top['DBD'])

    return {'df': df, 'df_top': df_top, 'E0': E0, 'C0': C0, 'DBD0': DBD0}


#%% load carbon data
fn_0y = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/soil_carbon/S0y.csv"
S0y = calc_void(fn_0y)

fn_10y = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/soil_carbon/S10y.csv"
S10y = calc_void(fn_10y)

fn_20y = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/soil_carbon/S20y.csv"
S20y = calc_void(fn_20y)

fn_40y = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/soil_carbon/S40y.csv"
S40y = calc_void(fn_40y)

#%% calculate E0_min and E0_om


# calculate linear regression for
# x: organic matter fraction (in terms of mass)
# y: the void ratio of the top layer (E0)
x = np.concatenate([S0y['df_top']['om_percentage'],
                    S10y['df_top']['om_percentage'],
                    S20y['df_top']['om_percentage'],
                    S40y['df_top']['om_percentage']])/100

y = np.concatenate([S0y['df_top']['void_ratio'],
                    S10y['df_top']['void_ratio'],
                    S20y['df_top']['void_ratio'],
                    S40y['df_top']['void_ratio']])
res = linregress(x,y)
a = res.slope
b = res.intercept
R2 = res.rvalue**2

fig, ax = plt.subplots(figsize = (4,3))
ax.scatter(100*x, y)
ax.plot([0,10],[b, a * 0.1 + b])
ax.set_xlabel(r'$p_{om}$ $\%$')
ax.set_ylabel(r'top layer void ratio ($E_0$)')

E0_min = b
E0_om = a + b
fig.tight_layout()
#%%
fig, axs = plt.subplots(ncols = 4)

fig_extra, axs_extra = plt.subplots(ncols = 3)

# latin  hyper cube
param_bounds = {'CI_min': (0.000, 0.2),
                'CI_om': (0.000, 2),}

lhc_sampler = scipy.stats.qmc.LatinHypercube(d=2)
lhc_samples = lhc_sampler.random(n=10000)
lhc_samples = scipy.stats.qmc.scale(lhc_samples,
                                    [param_bounds[c][0] for c in param_bounds.keys()],
                                    [param_bounds[c][1] for c in param_bounds.keys()])

RMSE_Ci = []



for i, site in enumerate([S0y, S10y, S20y, S40y]):

    RMSE_per_zone = []

    median_mass = site['df'].groupby('depth').agg({'mass_m2': lambda x: x.median(skipna=True)})
    median_om   = site['df'].groupby('depth').agg({'om_percentage': lambda x: x.median(skipna=True)})
    median_dbd   = site['df'].groupby('depth').agg({'DBD': lambda x: x.median(skipna=True)})
    median_silt   = site['df'].groupby('depth').agg({'%Silt': lambda x: x.median(skipna=True)})
    median_sand   = site['df'].groupby('depth').agg({'%Sand': lambda x: x.median(skipna=True)})

    om_mass     = (median_mass * median_om.values / 100).dropna().values[:,0]
    min_mass     = (median_mass * (100 - median_om.values) / 100).dropna().values[:,0]
    median_silt = median_silt.dropna()
    median_sand = median_sand.dropna()
    median_dbd  = median_dbd.dropna()

    # plot the dry bulk density of each sample
    for group_i, group in site['df'].groupby('auger'):
        axs[i].scatter(1000 * group['DBD'], -.01 * group['depth'], 4,  alpha = .25, c = 'orange')

    # plot observed medians
    axs[i].scatter(1000 * median_dbd,
                   -.01 * median_dbd.index,
                   8, alpha=.75, c='orange')

    for CI_om_i, (CI_min, CI_om) in enumerate(lhc_samples):

        oim = OIMAS_N(n_layers=len(om_mass), dt=30,
                      sigma_ref_min= 'top', sigma_ref_om= 'top',
                      CI_min=CI_min, CI_om=CI_om,
                      E0_min=E0_min, E0_om=E0_om,
                      Kla0=0, Kre0=0,
                      max_layer_thickness=.10)

        oim.initialize_layers(init_min_mass=min_mass,
                              init_om_mass=om_mass)

        # get and plot dry bulk density
        dbd_sim = oim.get_dbd()


        # interpolate observed dbd to model depths
        dbd_obs = np.interp(oim.d, median_dbd.index/100, 1000 * median_dbd.values[:,0])
        RMSE = rmse(dbd_obs, dbd_sim)

        RMSE_per_zone.append(RMSE)

    # run model for best parameter set
    best_param_set = np.argmin(RMSE_per_zone)

    oim = OIMAS_N(n_layers=len(om_mass), dt=30,
                  sigma_ref_min='top', sigma_ref_om='top',
                  CI_min=lhc_samples[best_param_set][0], CI_om=lhc_samples[best_param_set][1],
                  E0_min=E0_min, E0_om=E0_om,
                  Kla0=0, Kre0=0,
                  max_layer_thickness=.10)

    oim.initialize_layers(init_min_mass=min_mass,
                          init_om_mass=om_mass)

    # get and plot dry bulk density
    dbd_sim = oim.get_dbd()

    # plot
    axs[i].plot(dbd_sim, -1 * oim.d, 4, alpha=.25, c='blue')
    axs[i].set_ylim(-1.1, 0)
    title_lab = r'$CI_{min}$ = %.4f, $CI_{om}$ = %.2f' % (lhc_samples[best_param_set][0], lhc_samples[best_param_set][1])
    axs[i].set_title(title_lab, fontsize = 10)
    RMSE_Ci.append(RMSE_per_zone)


if False:
    #%% calculate average RMSE per CI_min
    CI_min_RMSE = np.array(CI_min_RMSE)

    avg_RMSE = np.mean(CI_min_RMSE, axis = 0)
    std_RMSE = np.std(CI_min_RMSE, axis = 0)

    ax_extra.scatter(CI_min_range, avg_RMSE, label = 'average RMSE', c = 'k', marker = 's', alpha = .5)
    # plot error bar
    ax_extra.errorbar(CI_min_range, avg_RMSE, yerr = std_RMSE, label = 'average RMSE', c = 'k', marker = 's', alpha = .5)

    opt_CI_i = np.argmin(avg_RMSE)

    ax_extra.scatter(CI_min_range[opt_CI_i], avg_RMSE[opt_CI_i], 150, marker = 'o' ,c = [0,0,0,0], ec = 'orange', lw = 2)


