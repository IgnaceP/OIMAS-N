import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.style.use("ip02")

BASE_MODEL_DIR = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N"
os.chdir(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR)

from OIMAS import OIMAS_N
from read_C_obs_data import read_observation_data
from functions import (
    load_soil_carbon,
    load_bgb_data,
    load_biomass_data,
    load_elevation_data,
    plot_carbon_profiles,
    plot_dbd_profiles,
    compute_initial_masses,
    create_oimas_model,
)

matplotlib.use('MacOSX')

# =============================================================================
# SET PARAMETERS HERE
# =============================================================================

auger_ID    = "S40y2"
Kla         = 0.0005
Kre         = 0.00006
sed         = .91

# =============================================================================

zone        = int(auger_ID[1:3])
init_elev   = 4.5
sed_om_frac = 0.049

veg_params = {
    'Tripolium':     [.82,  0.5,  .1],
    'Atriplex':      [.9,   0,    .17],
    'Bolboschoenus': [1.99, .75,  .2],
    'Elytrigia':     [0.7,  0.9,  .2],
}

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

soil    = load_soil_carbon([0, zone], read_observation_data)
bgb     = load_bgb_data()
biomass = load_biomass_data()
rtk     = pd.read_csv(
    '/Users/ignace/Documents/WETCOAST/Data/Saefthinge/RTK/sample_locations_RTK.csv',
    index_col=1
)[["z_TAW"]]

auger_soil = soil[zone].loc[auger_ID]

# -----------------------------------------------------------------------------
# Model initialisation
# -----------------------------------------------------------------------------

n_layers = 20
om_init, min_init = compute_initial_masses(soil[0], n_layers)

oim = create_oimas_model(OIMAS_N)
oim.initialize_layers(
    init_min_mass    = min_init,
    init_om_mass     = om_init,
    f_Cla            = 0.4,
    initial_surface  = init_elev,
)

# -----------------------------------------------------------------------------
# Run model
# -----------------------------------------------------------------------------

months  = zone * 12
agb     = np.linspace(0.99, biomass.loc[auger_ID]["mass_m2agb"], months + 1)
sed_om  = sed_om_frac * sed
sed_min = (1 - sed_om_frac) * sed

for month in range(months + 1):
    oim.Bmax   = agb[month]
    oim.Kla0   = Kla
    oim.Kre0   = Kre
    oim.chi_la = .8
    oim.chi_re = .2

    if oim.surface > 5.05:
        R, m, gamma = veg_params["Elytrigia"]
    elif oim.surface > 4.8:
        R, m, gamma = veg_params["Bolboschoenus"]
    else:
        R, m, gamma = veg_params["Tripolium"]

    day_peak = 244
    rs = lambda d: .5*(R - m*R) * np.cos(2*np.pi*(d - day_peak)/365) + .5*(R - m*R) + m*R

    oim.gamma = oim.kappa = oim.lamda = gamma
    oim.root_to_shoot = rs

    oim.biomass()
    oim.organic_carbon_decay()
    oim.sedimentation(sedimentation_om=sed_om, sedimentation_min=sed_min, f_Cla=.40)
    oim.update_layers()

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

fig, axs = plt.subplots(ncols=1, figsize=(7, 8))
axs_inset = axs.inset_axes([.7, 0.1, 0.25, 0.5])
axs_inset.set_facecolor([.2, .2, .2, .75])

plot_carbon_profiles([axs], [auger_soil])
plot_dbd_profiles([axs_inset], [auger_soil])

C = 100 * oim.get_C() / oim.mass
axs.plot(C, -oim.d, ls='--', color='w', label='simulated C [%]', zorder=10)
axs_inset.plot(oim.get_dbd(), -oim.d, ls=':', color='C0')

text = [
    r'$K_{la} = %.4f$'  % Kla,
    r'$K_{re} = %.5f$'  % Kre,
    r'$sed = %.2f$'     % sed + '\n',
    r'$z_{sim} = %.2f$' % oim.surface,
    r'$z_{obs} = %.2f$' % rtk.loc[auger_ID].iloc[0],
]
axs.text(0.95, 0.95, '\n'.join(text), ha='right', va='top', transform=axs.transAxes, fontsize=10)
axs.set_title(auger_ID)
axs.legend(loc=3)

age_horizons = oim.get_age_horizons()
for age in age_horizons:
    if age['t'] % 365 == 0:
        # plot age horizon as horizontal line
        axs.hlines(age['z']-oim.surface, xmin = 0, xmax = 10, ls=':', color='grey', zorder=10)
        axs.text(0.5, age['z']-oim.surface + 0.01, str(months/12 - int(age['t']/365)) + ' years', ha='center', va='center', fontsize=8, c = 'grey', transform=axs.transData, zorder=10)

print(f"z_sim = {oim.surface:.3f} m  |  z_obs = {rtk.loc[auger_ID].iloc[0]:.3f} m  |  Δz = {oim.surface - rtk.loc[auger_ID].iloc[0]:+.3f} m")