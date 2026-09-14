import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import scipy
from rmse import rmse
from tqdm import tqdm
from scipy.optimize import curve_fit
matplotlib.style.use("ip02")

BASE_MODEL_DIR = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N"
sys.path.append(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR + '/Blackwater/')

from OIMAS import OIMAS_N
from load_data_Blackwater import *
matplotlib.use('MacOSX')

# surpress warnings
import warnings
warnings.filterwarnings("ignore")

#%% Load data
fn = "/Users/ignace/Documents/WETCOAST/Data/Blackwater/Mona/all_data.csv"
df_raw = pd.read_csv(fn, index_col=0)

rho_min = 1990
rho_om = 850
rho_root = 1500

df = pd.DataFrame()
df['depth'] = df_raw['depth_corr']
df['auger'] = df_raw['Core']

# diameter auger is 10 cm
# segments represent a depth of 2.6 cm
df['DBD']               = df_raw['DBD_corr']
df['C_percentage']      = df_raw['%C_sediment']
df['mass']              = 0.026 / (df_raw['comp_per']/100) * df['DBD']
df['mass_segm']         = 0.026 / (df_raw['comp_per']/100) * np.pi * 0.05**2 * df['DBD']
df['Cmass_m2']          = df['mass'] * df['C_percentage'] / 100

df['om_percentage']     = df['C_percentage'] / 0.44
df['roots_percentage'] = 100 * df_raw['BD_root_gcm3'] / df_raw['BD_total_gcm3']
df['min_percentage']    = 100 - df['om_percentage'] - df['roots_percentage']

df['om_mass']    = df['mass'] * df['om_percentage'] / 100      # non-root organic (SOM)
df['root_mass']  = df['mass'] * df['roots_percentage'] / 100
df['min_mass']   = df['mass'] - df['om_mass'] - df['root_mass']

df['om']                = df['mass'] * df['om_percentage'] / 100
df['min']               = df['mass'] - df['om']
df['volume']            = 0.026 * np.pi * 0.05**2 / (df_raw['comp_per']/100)

f_om   = df['om_mass']   / df['mass']
f_min  = df['min_mass']  / df['mass']
f_root = df['root_mass'] / df['mass']

df['rho_solid'] = 1 / (f_om/rho_om + f_min/rho_min + f_root/rho_root)

df['solid_volume']      = df['mass_segm'] / df['rho_solid']

df['ups_om'] = (f_om/rho_om) / ((f_om/rho_om) + (f_min/rho_min) + (f_root/rho_root))      # unit weight of organic matter
df['ups_min'] = (f_min/rho_min) / ((f_om/rho_om) + (f_min/rho_min) + (f_root/rho_root))
df['ups_root'] = (f_root/rho_root) / ((f_om/rho_om) + (f_min/rho_min) + (f_root/rho_root))

df['void_ratio']        = df['rho_solid'] / df['DBD'] - 1

df_top                  = df[df['depth'] < 0.03]
df_bottom                  = df[df['depth'] == 0.30550]


#%% establish emperical relation to predict DBD
x = df_top['om_percentage']/100
y = df_top['void_ratio']
z = df_top['depth']

plt.scatter(x, y, )


def exp_endpoint(x, E0_min, E0_om):
    return E0_min*(E0_om/E0_min)**x

# fit curve
popt, pcov = curve_fit(exp_endpoint, x, y)


plt.plot(np.sort(x), exp_endpoint(np.sort(x), *popt))
plt.title(r'$E0 = E0_{min} * (\frac{E0_{min}}{E0_{om}})^{om}$ with $E0_{min} = %.2f$ and $E0_{om} = %.2f$' % (popt[0], popt[1]))



#%% run model for best parameter set

fig, axs = plt.subplots(nrows = 3, ncols = 3, sharex=True, sharey=True)
axs = axs.flatten()

for i, auger in tqdm(enumerate(np.unique(df.auger))):

    df_auger = df[df['auger'] == auger]

    # plot observed
    sc = axs[i].scatter(df_auger['DBD'],
                        -1 * df_auger['depth'],
                        8, df_auger['roots_percentage'],
                        vmax=50, vmin=0, cmap='YlGn')

    e = exp_endpoint(df_auger['om_percentage']/100, *popt)
    dbd = df_auger['rho_solid'] / (e + 1)

    axs[i].scatter(dbd,
                  -1 * df_auger['depth'],
                  8, c =  'gold', alpha = .5)


    axs[i].set_title(auger)

    oim = OIMAS_N(n_layers=len(df_auger['min']),
                  compaction_method = 'Gutierrez',
                  E0_om = 587, E0_min = 0.7,
                  CI_om = 0, CI_min = 0,
                  Kla0=0, Kre0=0,
                  Bmax = 0,
                  rho_om = rho_om, rho_min = rho_min,
                  max_layer_thickness=.10)

    oim.initialize_layers(init_min_mass=df_auger['min'].values,
                          init_om_mass=df_auger['om'].values,
                          set_bbg=True, bbg = df_auger['root_mass'].values)

    # get and plot dry bulk density
    dbd_sim = oim.get_dbd()

    # plot
    axs[i].plot(dbd_sim, -1 * oim.d, 4, alpha=.75, c='orange', marker = 'o', markersize = 3)
    axs[i].set_ylim(-0.75, 0)


axs[3].set_ylabel('depth (m)')
axs[7].set_xlabel(r'dry bulk density [$kg/m^3$]')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.10, 0.025, 0.78])
fig.colorbar(sc, cax=cbar_ax, label = 'organic matter [%]')

#%%
plt.scatter(df_auger['om_percentage'], 100*oim.Pom)
plt.scatter(df_auger['void_ratio'], oim.E)
plt.scatter(df_auger['void_ratio'], e)