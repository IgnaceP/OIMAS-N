import os
os.chdir('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import sys
sys.path.append('/Users/ignace/Documents/WETCOAST/model/OIMAS-N/')
import pandas as pd
import scipy
from scipy.stats import linregress

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

#%% Get all the arrays
om = np.concatenate([S0y['df_top']['om_percentage'],
                    S10y['df_top']['om_percentage'],
                    S20y['df_top']['om_percentage'],
                    S40y['df_top']['om_percentage']])/100

E = (1 - om) * np.concatenate([S0y['df_top']['void_ratio'],
                    S10y['df_top']['void_ratio'],
                    S20y['df_top']['void_ratio'],
                    S40y['df_top']['void_ratio']])

sand = (1 - om) * np.concatenate([S0y['df_top']['%Sand'],
                    S10y['df_top']['%Sand'],
                    S20y['df_top']['%Sand'],
                    S40y['df_top']['%Sand']])/100

silt = (1 - om) * np.concatenate([S0y['df_top']['%Silt'],
                       S10y['df_top']['%Silt'],
                       S20y['df_top']['%Silt'],
                       S40y['df_top']['%Silt']])/100

clay = (1 - om) * np.concatenate([S0y['df_top']['%Clay'],
                       S10y['df_top']['%Clay'],
                       S20y['df_top']['%Clay'],
                       S40y['df_top']['%Clay']])/100

dbd = np.concatenate([S0y['df_top']['DBD'],
                       S10y['df_top']['DBD'],
                       S20y['df_top']['DBD'],
                       S40y['df_top']['DBD']])


#%% add a Latin Hypercube to fit E0

# prepare a Latin Hypercube for sensitivity analysis for K labile
param_bounds = {'E0_om': (8, 20),
                'E0_sand': (0, 20),
                'E0_silt': (0, 20),
                'E0_clay': (0, 20),}

lhc_sampler = scipy.stats.qmc.LatinHypercube(d=4)
lhc_samples = lhc_sampler.random(n=50000)
lhc_samples = scipy.stats.qmc.scale(lhc_samples,
                                    [param_bounds[e][0] for e in param_bounds.keys()],
                                    [param_bounds[e][1] for e in param_bounds.keys()],)
RMSE = []
for i in range(lhc_samples.shape[0]):
    E_sim = (lhc_samples[i, 0] * om +
             (1 - om)* (
             lhc_samples[i, 1] * sand +
             lhc_samples[i, 2] * silt,
             lhc_samples[i, 3] * clay))

    rmse = np.sqrt(np.nanmean((E - E_sim)**2))
    RMSE.append(rmse)

print(np.min(RMSE))
print(lhc_samples[np.argmin(RMSE)])