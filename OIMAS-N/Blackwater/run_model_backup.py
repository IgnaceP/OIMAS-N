import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

matplotlib.style.use("ip02")

BASE_MODEL_DIR = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N"
sys.path.append(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR + '/Blackwater/')

from OIMAS import OIMAS_N
from load_data_Blackwater import *
matplotlib.use('MacOSX')

# =============================================================================
#%% SET PARAMETERS HERE
# =============================================================================

# OIMAS parameters

dt          = 1                 # years
auger_ID    = "DB4"

Kla         = 0.051539
Kre         = 0.001163

# MARSED parameters
k           = 0.01947

ws          = 2e-4
sed_om_frac = 0.10

chi_re = 0.3
chi_la = 0.7

# general parameters
n_years     = 100

init_elev = 0
init_om_frac = 0.05

# =============================================================================

# polynomial coefficient to estimate biomass in function of inundation depth at mhwl
biomass_coeff = [ 102.09192788, -302.56185477,  324.44520351, -151.47311919, 27.34594944, 0.0]

# root-shoot ratio at moment of maximum biomass
# turnover rate (root + rhizome) yr^-1
# gamma, lamda and kappa
veg_params = {
    'cynusuroides':    [1.101,  0.985,  .27],
    'americanus':      [1.789,  1.314,  .27],
    'alterniflora':     [1.101,  0.985,  .27],
}

# relative weights in function of mhwl depth
def veg_weights(d):
    if d < 0.12:
        return 1,0,0
    elif d < 0.16:
        w_am = (d - 0.12)/(0.16 - 0.12)
        w_cyn = 1 - w_am
        return w_cyn, w_am, 0
    elif d < 0.19:
        return 0,1,0
    elif d < 0.23:
        w_alt = (d - 0.19)/(0.23 - 0.19)
        w_am = 1 - w_alt
        return 0, w_am, w_alt
    else:
        return 0,0,1
# -----------------------------------------------------------------------------
#%% Load data
# -----------------------------------------------------------------------------

soil = load_soil_carbon()
auger_soil = soil.loc[auger_ID]
rtk, sar = load_rtk_data()

hwl_df = load_hwls()
avg_tide = load_avg_tide()

# -----------------------------------------------------------------------------
#%% Model initialisation
# -----------------------------------------------------------------------------

n_layers = 50
om_init, min_init = np.full(n_layers, init_om_frac * 7.5), np.full(n_layers, 7.5)

sar = sar.loc[auger_ID[:-1]]['SAR']
#n_years = int(auger_soil.depth.max() / sar)

oim = OIMAS_N(
        n_layers = n_layers,
        dt= dt,
        f_C = 0.44,
        rho_min = 1990, rho_om = 850,
        sigma_ref_min=1e5, sigma_ref_om=1e4,
        CI_min = 0, CI_om = 0,
        E0_min = 0.7, E0_om = 587,
        Kla0 = Kla, Kre0 = Kre,
        chi_la = chi_la, chi_re = chi_re,
        max_layer_thickness = 0.07,
        )

oim.initialize_layers(
    init_min_mass    = min_init,
    init_om_mass     = om_init,
    f_Cla            = 0.0,
    initial_surface  = init_elev,
)

oim.update_geometry(surface=init_elev)

# -----------------------------------------------------------------------------
#%% Prepare plot
# -----------------------------------------------------------------------------

fig, (axs) = plt.subplots(ncols=1, figsize=(7, 8))
axs_inset = axs.inset_axes([.825, 0.1, 0.15, 0.4], facecolor = [0.1,0.15,0.15, 0.85])
axs_inset2 = axs.inset_axes([.1, 0.1, 0.25, 0.25], facecolor = [0.1,0.15,0.15, 0.85])
axs_inset3 = axs.inset_axes([.45, 0.1, 0.25, 0.25], facecolor = [0.1,0.15,0.15, 0.85])


plot_carbon_profiles([axs], [auger_soil])
plot_dbd_profiles([axs_inset], [auger_soil])

C = 100 * oim.get_C() / oim.mass

# -----------------------------------------------------------------------------
#%% Run model
# -----------------------------------------------------------------------------

time_steps  = n_years
year = 2023-n_years
years = range(2023 - n_years + 1, 2023 + 1)

C_tot = []
M_tot = []
mhwl_d_tot = []
AGB_tot = []
spec = []

for year in years:
    print('t = ', year)

    delta_msl = 0.47623*np.exp(0.0161643*(year - 2014)) - 0.476
    hwls = hwl_df.values.flatten() + delta_msl
    mhwl = np.mean(hwls)

    # set model parameters
    z          = oim.surface
    mhwl_d     = max(mhwl - z, 0)

    oim.Bmax   = np.polyval(biomass_coeff, mhwl_d)


    oim.Kla0   = Kla
    oim.Kre0   = Kre
    oim.chi_la = chi_la
    oim.chi_re = chi_re

    w_cyn, w_am, w_alt = veg_weights(mhwl_d)

    R = w_cyn * veg_params["cynusuroides"][0] + w_am * veg_params["americanus"][0] + w_alt * veg_params["alterniflora"][0]
    TO = w_cyn * veg_params["cynusuroides"][1] + w_am * veg_params["americanus"][1] + w_alt * veg_params["alterniflora"][1]
    gamma = w_cyn * veg_params["cynusuroides"][2] + w_am * veg_params["americanus"][2] + w_alt * veg_params["alterniflora"][2]
    spec.append([w_cyn, w_am, w_alt])

    oim.gamma = oim.kappa = oim.lamda = gamma
    oim.root_to_shoot = R
    oim.turnover = TO

    oim.biomass()
    oim.organic_carbon_decay()
    oim.marsed(hwls, avg_tide.index, avg_tide.avg_H, k = k, ws = ws, sed_om_frac = sed_om_frac, use_Julia = True, f_Cla = chi_la)
    oim.update_layers()

    C = 100 * oim.get_C() / oim.mass

    C_kg = oim.get_C()

    C_tot.append(np.sum(C_kg))
    M_tot.append(np.sum(oim.Mbg_int))
    mhwl_d_tot.append(mhwl_d)
    AGB_tot.append(oim.Agb)


print("inundation depth at mean high water level: %.2f m" % mhwl_d)

# -----------------------------------------------------------------------------
#%% Plot
# -----------------------------------------------------------------------------

mask = oim.d < sar*n_years
axs.plot((100 * oim.Cre / oim.mass)[mask], -oim.d[mask], ls='--', color='darkgreen', alpha = .5)
axs.plot(C[mask], -oim.d[mask], ls='--', color='palegreen', marker = 'o')

axs_inset.plot(oim.get_dbd(), -oim.d, ls=':', color='palegreen')
axs_inset2.plot(years, mhwl_d_tot, c ='C1')

axs_inset3.scatter(years, AGB_tot, 5, c = spec)
axs_inset3.legend(handles = [matplotlib.lines.Line2D([0], [0], marker='o', lw = 0, color=c, markersize=5) for c in [[1,0,0], [0,1,0], [0,0,1]]], labels = ['S. cynosuroides', 'S. americanus', 'S. alterniflora'], frameon = False, fontsize = 8)

axs_inset2.set_title(r'$d_{mhwl}$ [m]')
axs_inset3.set_title(r'$agb$ [kg]')
axs_inset2.set_ylim(-0.1, mhwl_d_tot[-1] + 0.1)
leg = axs_inset2.legend(frameon = False, fontsize = 8); [l.set_color('gray') for l in leg.get_texts()]

text = [
    r'$K_{la} = %.4f$'  % Kla,
    r'$K_{re} = %.5f$'  % Kre,
    r'$k = %.4f$'     % k + '\n',
    r'$z_{sim} = %.2f$' % oim.surface,
    r'$z_{obs} = %.2f$' % rtk.loc[auger_ID].iloc[0],
]
axs.text(0.95, 0.975, '\n'.join(text), ha='right', va='top', transform=axs.transAxes, fontsize=10)
axs.set_xlim(0, 35)
#axs.legend()

# -----------------------------------------------------------------------------
#%% Evaluate
# -----------------------------------------------------------------------------

# interpolate the simulated C densities to the observed densities
C_sim = np.interp(auger_soil["depth"], oim.d, C)
mask = auger_soil["depth"] < sar*n_years

# calculate the root mean square error
rmse_C = np.sqrt(np.mean(np.square(C_sim[mask] - auger_soil["C_percentage"][mask])))

axs.set_title(f'{auger_ID} - RMSE = {rmse_C:.2f} %')

sar_sim,_ = np.polyfit(np.arange(25), mhwl_d_tot[-25:], deg = 1)
print('simulated sediment accretion rates: %.1f mm/yr' % (1000*sar_sim))
print('observed sediment accretion rates: %.1f mm/yr' % (1000*sar))
