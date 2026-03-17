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

matplotlib.style.use('ip02')
import cmasher as cm

from rmse import rmse

from OIMAS import OIMAS_N
from read_C_obs_data import read_observation_data


#%% load carbon data
fn_0y = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/soil_carbon/S0y.csv"
S0y = read_observation_data(fn_0y)

fn_10y = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/soil_carbon/S10y.csv"
S10y = read_observation_data(fn_10y)

fn_20y = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/soil_carbon/S20y.csv"
S20y = read_observation_data(fn_20y)

fn_40y = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/soil_carbon/S40y.csv"
S40y = read_observation_data(fn_40y)

# load below-ground biomass
fn_bgb = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/biomass/BGB2025.csv"
bgb = pd.read_csv(fn_bgb, skiprows=1, index_col=1)
footprint_bgb = .05**2 * np.pi
bgb['mass_m2'] =  bgb['mass'] / footprint_bgb / 1000
bgb = bgb[bgb['primary_plant'] != 'bare']
bgb['depth'] = [(float(d.split('-')[-1][:-2]) - 5)/100 for d in bgb["Depth"]]

#%% plot carbon content

fig, axs = plt.subplots(ncols = 4, sharey=True, sharex=True, figsize = (15,8))


for i, site in enumerate([S0y, S10y, S20y, S40y]):

    # ---- bottom x-axis: carbon content ----
    for group_i, group in site.groupby('auger'):
        axs[i].scatter(group['C_percentage'], -.01 * group['depth'], 4,  alpha = .25, c = 'orange')

    median_C = site.groupby('depth').agg({'C_percentage': lambda x: x.median(skipna=True)})
    axs[i].plot(median_C, -.01 * median_C.index, ms = 4, alpha=.75, c='orange', marker = 'o', label = 'C [%]')

    # set axis settings
    axs[i].set_xlabel(r'C [%]')

    #axs[i].set_title(["S0y", "S10y", "S20y", "S40y"][i])


axs[0].set_ylabel(r"depth [$m$]")
axs[0].set_xlim(0, 9)
axs[0].set_ylim(-.9, 0)


###################################################################################################
#%% After 0 years: model initiation
###################################################################################################

median_mass     = S0y.groupby('depth').agg({'mass_m2': lambda x: x.median(skipna=True)})
median_om       = S0y.groupby('depth').agg({'om_percentage': lambda x: x.median(skipna=True)})
median_dbd      = S0y.groupby('depth').agg({'DBD': lambda x: x.median(skipna=True)})
om_mass         = (median_mass * median_om.values / 100).dropna().values[:, 0]
min_mass        = (median_mass * (100 - median_om.values) / 100).dropna().values[:, 0]

n_layers        = 20
om_init         = np.zeros(n_layers)
om_init[:len(om_mass)] = om_mass
om_init[len(om_mass):] = om_mass[-1]

min_init         = np.zeros(n_layers)
min_init[:len(min_mass)] = min_mass
min_init[len(min_mass):] = min_mass[-1]


oim             = OIMAS_N(n_layers=n_layers, dt=30,
                    sigma_ref_min='top', sigma_ref_om='top',
                    theta_bg = 0, Dmbm = .5,
                    Bmax=2.,
                    f_C = 0.5208,
                    CI_min=.1, CI_om=1.,
                    E0_min = 0.861, E0_om = 27.545,
                    Kla0 = 0.00125, Kre0 = 0.000,
                    chi_la = .85, chi_re = .15,
                    max_layer_thickness= .07,
                    gamma = .15, kappa = .15, lamda = .15)

oim.initialize_layers(init_min_mass=min_init,
                      init_om_mass=om_init)

S0y_C           = 100 * oim.get_C() / oim.mass
axs[0].plot(S0y_C, -1 * oim.d)

sedimentation = {}
sedimentation['S0y'] = oim.surface
axs[0].set_title(axs[0].get_title() + '\n' + 'surface: %.2f m' % oim.surface)

###################################################################################################
#%% After 10 years: model
###################################################################################################

months = 40 * 12

biomass_0y          = .99
biomass_10y         = .5 * 1.34 + .5 * 1.25

root_shoot_0y       = .11
root_shoot_10y      = .375 * .82 + .625 * 1.15

# set maximum above ground biomass and root-shoot ratio
biomass             = np.linspace(biomass_0y, biomass_10y, months+1)
root_shoot          = np.linspace(root_shoot_0y, root_shoot_10y, months+1)

Bag = np.zeros(months+1)
Bbg = np.zeros(months+1)
bbg_sim = []
Depth = []
Thickness = []

# sedimentation
# 0.032 m/yr in first 10 years
t_elev = [10,20,30,50]
elev = [2.13, 2.45, 2.56, 2.72]
a1, a2 = np.polyfit(np.log(t_elev), elev, deg = 1)
x_elev = np.arange(10,10+months/12,1/12)
y_elev = a1 * np.log(x_elev) + a2
sed = np.zeros(months+1)
sed[:months-1] = np.diff(y_elev)/np.diff(x_elev)

total_sed_mass = sed * 700 # per year
sed_om = .22 * total_sed_mass / 12 # per month
sed_min = .78 * total_sed_mass / 12 # per month

axi = 0
for month in range(months+1):

    if month in [0,120,240,480]:
        C = 100 * oim.get_C() / oim.mass
        axs[axi].plot(C, -1 * oim.d)
        axs[axi].set_title(f'{month} months \n surface: %.2f' % (oim.surface))
        axi += 1

    oim.Bmax        = biomass[month]
    oim.Dmbm        = root_shoot[month]
    oim.Kla0        = .005
    oim.Kre0        = .00005

    oim.biomass()
    oim.organic_carbon_decay()
    oim.sedimentation(sedimentation_om = sed_om[month], sedimentation_min = sed_min[month],
                      f_Cla = 0.85)
    oim.update_layers()
















