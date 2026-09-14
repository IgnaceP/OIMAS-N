import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import scipy

matplotlib.style.use("ip02")

BASE_MODEL_DIR = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N"
sys.path.append(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR + '/Blackwater/')

from OIMAS import OIMAS_N
from load_data_Blackwater import *
matplotlib.use('MacOSX')


#%% Load data
fn = "/Users/ignace/Documents/WETCOAST/Data/Blackwater/Mona/raw.csv"
df_raw = pd.read_csv(fn, sep = ";")
df_raw = df_raw[df_raw['Site'] == 'Least degraded']
rho_min = 1990
rho_om = 850

df = pd.DataFrame()
df['depth'] = df_raw['Depth_avg_cm']/100
df['auger'] = df_raw['Core']

# diameter auger is 10 cm
# segments represent a depth of 2.6 cm
df['DBD']               = df_raw['BD_total_gcm3']*1000
df['root_percentage_mass'] = 100 * df_raw['BD_root_gcm3'] / df_raw['BD_total_gcm3']
df['C_percentage']      = df_raw['%C_sediment']
df['mass_m2']           = 0.026 * df['DBD']
df['mass']              = 0.026 * np.pi * 0.05**2 * df['DBD']
df['Cmass_m2']          = df['mass_m2'] * df['C_percentage'] / 100
df['om_percentage']     = df['C_percentage'] / 0.44
df['om']                = df['mass_m2'] * df['om_percentage'] / 100
df['min']               = df['mass_m2'] - df['om']
df['volume']            = 0.026 * np.pi * 0.05**2

df['rho_solid']         = (df['min'] * rho_min + df['om'] * rho_om) / (df['mass_m2'])
df['rho_solid']         = ((df['om_percentage']/100)/rho_om + (1-df['om_percentage']/100)/rho_min)**-1

df['solid_volume']      = df['mass'] / df['rho_solid']
df['void_ratio']        =( df['volume'] - df['solid_volume']) / df['solid_volume']
df_top                  = df[df['depth'] == 0.0195]

E0                      = np.median(df_top['void_ratio'])
C0                      = np.median(df_top['C_percentage'])
DBD0                    = 1000 * np.median(df_top['DBD'])

#%% linear regression between om percentage and E0 to estimate E0_min and E0_om
x = df_top['om_percentage']/100
y = df_top['void_ratio']
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

def line(x, a, b):
    return a + b * x   # a = intercept, b = slope

# bounds = ([lower_a, lower_b], [upper_a, upper_b])
# here: intercept >= 0, slope unbounded
popt, pcov = scipy.optimize.curve_fit(
    line, x, y,
    bounds=([0, -np.inf], [np.inf, np.inf])
)

intercept, slope = popt

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, x*slope+intercept, 'r')
ax.set_xlabel('OM percentage')
ax.set_ylabel('E0')

E0_min = intercept
E0_om = slope

#%% estimate Cl_min and Cl_om
