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
sys.path.append(BASE_MODEL_DIR + '/Saefthinge/')

from OIMAS import OIMAS_N
from read_C_obs_data import read_observation_data
from load_data import *
matplotlib.use('MacOSX')

# =============================================================================
#%% SET PARAMETERS HERE
# =============================================================================

# OIMAS parameters
dt          = 1                 # years
auger_ID    = "S40y1"
Kla         = 0.05
Kre         = 0.0029

# MARSED parameters
k           = 0.09
ws          = 1.1e-4
sed_om_frac = 0.049

# =============================================================================

zone        = int(auger_ID[1:3])
init_elev   = 4.5


# root-shoot ratio at moment of maximum biomass
# turnover rate (root + rhizome) yr^-1
# gamma, lamda and kappa
veg_params = {
    'Tripolium':     [.82,  0.5,  .1],
    'Atriplex':      [.9,   1,    .17],
    'Bolboschoenus': [1.99, 0.8,  .2],
    'Elytrigia':     [0.7,  0.8,  .2],
}

# -----------------------------------------------------------------------------
#%% Load data
# -----------------------------------------------------------------------------

soil = load_soil_carbon([0, zone])
bgb = load_bgb_data()
rtk = load_rtk_data()
sar = load_sar_data()
rtk = rtk.join(sar['SAR'], how='inner')
avg_tide = load_avg_tide()
hwl_df = load_hwls()

# -----------------------------------------------------------------------------
#%% Model initialisation
# -----------------------------------------------------------------------------

n_layers = 10
om_init, min_init = compute_initial_masses(soil[0], n_layers)
C_init = soil[0].groupby("depth")["C_percentage"].median()

oim = OIMAS_N(
        n_layers = n_layers,
        dt= dt,
        sigma_ref_min="top", sigma_ref_om="top",
        f_C = 0.5208,
        CI_min = 0.025, CI_om = 1.0,
        E0_min = 0.861, E0_om = 27.545,
        Kla0 = Kla, Kre0 = Kre,
        chi_la = 0.40, chi_re = 0.60,
        max_layer_thickness = 0.07,
        )

oim.initialize_layers(
    init_min_mass    = min_init,
    init_om_mass     = om_init,
    f_Cla            = 0.4,
    initial_surface  = init_elev,
)


# -----------------------------------------------------------------------------
#%% Prepare plot
# -----------------------------------------------------------------------------

fig, (axs) = plt.subplots(ncols=1, figsize=(7, 8))
axs_inset = axs.inset_axes([.7, 0.1, 0.25, 0.5])
#axs_inset.set_facecolor([.2, .2, .2, .75])

auger_soil = soil[zone].loc[auger_ID]
plot_carbon_profiles([axs], [auger_soil])
plot_dbd_profiles([axs_inset], [auger_soil])


# -----------------------------------------------------------------------------
#%% Run model
# -----------------------------------------------------------------------------

time_steps  = zone

C_tot = []
M_tot = []

for time_step in range(time_steps + 1):
    # set years
    t0 = datetime(2026 - time_steps + (time_step-1), 1, 1)
    t1 = datetime(2026 - time_steps + (time_step-1), 12, 31)

    # set model parameters
    oim.Bmax   = 1.45 * oim.surface - 5.58
    oim.Kla0   = Kla
    oim.Kre0   = Kre
    oim.chi_la = .8
    oim.chi_re = .2

    if oim.surface > 5.05:
        R, TO, gamma = veg_params["Elytrigia"]
    elif oim.surface > 4.8:
        R, TO, gamma = veg_params["Bolboschoenus"]
    else:
        R, TO, gamma = veg_params["Tripolium"]

    oim.gamma = oim.kappa = oim.lamda = gamma
    oim.root_to_shoot = R
    oim.turnover_rate = TO

    oim.biomass()
    oim.organic_carbon_decay()
    hwls = hwl_df.loc[t0:t1].values.flatten()
    oim.marsed(hwls, avg_tide.index, avg_tide.avg_H, k = k, ws = ws, sed_om_frac = sed_om_frac, use_Julia = True)
    #oim.sedimentation(sedimentation_om=.35, sedimentation_min=8, f_Cla=.40)
    oim.update_layers()

    C = 100 * oim.get_C() / oim.mass
    C_kg = oim.get_C()
    #axs.plot(C, -oim.d, ls='--', color=matplotlib.colormaps['viridis'](time_step / time_steps), label=time_step, zorder=10)
    #axs_kg.plot(C_kg, -oim.d, ls='--', color=matplotlib.colormaps['viridis'](time_step / time_steps), label=time_step, zorder=10)

    C_tot.append(np.sum(C_kg))
    M_tot.append(np.sum(oim.Mbg_int))

# -----------------------------------------------------------------------------
#%% Plot
# -----------------------------------------------------------------------------

axs.plot(C, -oim.d, ls='--', color='white')
axs_inset.plot(oim.get_dbd(), -oim.d, ls=':', color='C0')

text = [
    r'$K_{la} = %.4f$'  % Kla,
    r'$K_{re} = %.5f$'  % Kre,
    r'$k = %.2f$'     % k + '\n',
    r'$z_{sim} = %.2f$' % oim.surface,
    r'$z_{obs} = %.2f$' % rtk.loc[auger_ID].iloc[0],
]
axs.text(0.95, 0.975, '\n'.join(text), ha='right', va='top', transform=axs.transAxes, fontsize=10)
axs.set_title(auger_ID)
#axs.legend()

if False:
    age_horizons = oim.get_age_horizons()
    for age in age_horizons:
        if age['t'] % 365 == 0:
            # plot age horizon as horizontal line
            axs.hlines(age['z']-oim.surface, xmin = 0, xmax = 10, ls=':', color='grey', zorder=10)
            axs.text(0.5, age['z']-oim.surface + 0.01, ' %.1f years' % (time_steps - int(age['t']/365)), ha='center', va='center', fontsize=8, c = 'grey', transform=axs.transData, zorder=10)
            print(' %.1f years' % (time_steps/12 - int(age['t']/365)))
print(f"z_sim = {oim.surface:.3f} m  |  z_obs = {rtk.loc[auger_ID].iloc[0]:.3f} m  |  Δz = {oim.surface - rtk.loc[auger_ID].iloc[0]:+.3f} m")

