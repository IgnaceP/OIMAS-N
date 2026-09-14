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



#%% run model for best parameter set

fig, axs = plt.subplots(nrows = 3, ncols = 3, sharex=True, sharey=True)
axs = axs.flatten()

for i, auger in tqdm(enumerate(np.unique(df.auger))):

    df_auger = df[df['auger'] == auger]

    # plot observed
    sc = axs[i].scatter(df_auger['C_percentage'],
                        -1 * df_auger['depth'],
                        8, df_auger['roots_percentage'],
                        vmax=50, vmin=0, cmap='YlGn')

    axs[i].set_title(auger)

axs[3].set_ylabel('depth (m)')
axs[7].set_xlabel(r'C [%]')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.10, 0.025, 0.78])
fig.colorbar(sc, cax=cbar_ax, label = 'roots [%]')

