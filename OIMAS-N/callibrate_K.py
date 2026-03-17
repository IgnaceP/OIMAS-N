import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import pandas as pd
from tqdm import tqdm

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
    plot_belowground_biomass,
    compute_initial_masses,
    create_oimas_model,
    pareto_front_2d
)

matplotlib.use('Agg')

#%% loop over all auger IDs in a zone

#auger_IDs = [f"S20y{i}" for i in [1,2,3,4,5,7,8]]
auger_IDs = [f"S40y{i}" for i in [1,2,3,4,5,6,7,8]]
#auger_IDs = [f"S10y{i}" for i in [1,2,3,4,5,6,7,8]]

# initialize dataframe to store callibrated values
call_df = pd.DataFrame({"Kla0": np.empty(len(auger_IDs)),
                     "Kre0": np.empty(len(auger_IDs)),
                     "sed_opt": np.empty(len(auger_IDs)),
                     },
                                        index=auger_IDs)

for auger_ID in auger_IDs:

    zone = int(auger_ID[1:3])

    # -----------------------------------------------------------------------------
    #%% Load data
    # -----------------------------------------------------------------------------

    soil = load_soil_carbon([0, zone], read_observation_data)
    bgb = load_bgb_data()
    biomass = load_biomass_data()
    elev = load_elevation_data([zone])

    # -----------------------------------------------------------------------------
    #%% Plot observations
    # -----------------------------------------------------------------------------

    fig, axs = plt.subplots(ncols= 1 , sharex=True, sharey=True, figsize=(7, 8))

    auger_soil = soil[zone].loc[auger_ID]
    auger_bgb = bgb.loc[auger_ID]

    plot_carbon_profiles([axs], [auger_soil])

    # -----------------------------------------------------------------------------
    #%% Model initialization (S0y)
    # -----------------------------------------------------------------------------

    n_layers = 20
    om_init, min_init = compute_initial_masses(soil[0], n_layers)

    oim = create_oimas_model(OIMAS_N)
    oim.initialize_layers(
        init_min_mass=min_init,
        init_om_mass=om_init,
        f_Cla = 0.4
    )

    C0 = 100 * oim.get_C() / oim.mass

    # -----------------------------------------------------------------------------
    #%% Simulation setup (0–XX years)
    # -----------------------------------------------------------------------------

    # total number of months to simulate
    years                   = zone
    months                  = years * 12

    # aboveground biomass from S0y to SXXy, stepwise
    agb                     = np.linspace(0.99, biomass.loc[auger_ID]["mass_m2agb"], months + 1)
    # root-shoot ratio from S0y to SXXy, stepwise
    root_shoot              = np.linspace(0.11, biomass.loc[auger_ID]["root-shoot"], months + 1)

    # Sedimentation based on LiDAR measurements
    sed_om_frac = 0.049
    dbd_top_layer = soil[zone].loc[auger_ID].iloc[0]["DBD"]
    sed_mass_max = 2 * elev[zone].max().iloc[0] * 1000 * dbd_top_layer / 12
    sed_mass_min = .5 * elev[zone].min().iloc[0] * 1000 * dbd_top_layer / 12

    # -----------------------------------------------------------------------------
    #%% Run simulation for callibration
    # -----------------------------------------------------------------------------

    # prepare inset axes
    axs_inset       = axs.inset_axes([.7, 0.1, 0.25, 0.5])
    axs_inset.set_facecolor([.2, .2, .2, .75])
    plot_dbd_profiles([axs_inset], [auger_soil])


    # prepare a Latin Hypercube for sensitivity analysis for K labile
    param_bounds    = { 'Kla0': (0.000, 0.2),
                        'Kre0': (0.000, 0.0003),
                        'sed': (sed_mass_min, sed_mass_max)}

    lhc_sampler     = scipy.stats.qmc.LatinHypercube(d=3)
    lhc_samples     = lhc_sampler.random(n=1000)
    lhc_samples     = scipy.stats.qmc.scale(lhc_samples,
                            [param_bounds['Kla0'][0], param_bounds['Kre0'][0], param_bounds['sed'][0],],
                            [param_bounds['Kla0'][1], param_bounds['Kre0'][1], param_bounds['sed'][1],])
    lhc_rmse_C      = np.zeros(lhc_samples.shape[0])
    lhc_rmse_DBD    = np.zeros(lhc_samples.shape[0])
    lhc_rmse_Z      = np.zeros(lhc_samples.shape[0])

    # callibrate K labile
    for i, (Kla, Kre, total_sed_mass) in tqdm(enumerate(lhc_samples), total=lhc_samples.shape[0]):
        # copy the oim instance to not overwrite the original
        oim_call            = oim.copy()

        # sedimentation, based on Latin Hypercube sampling
        sed_om = sed_om_frac * total_sed_mass
        sed_min = (1 - sed_om_frac) * total_sed_mass

        for month in range(months + 1):

            oim_call.Bmax   = agb[month]
            oim_call.Dmbm   = root_shoot[month]
            oim_call.Kla0   = Kla
            oim_call.Kre0   = Kre
            oim.mu_la = 1e6
            oim.mu_re = 1e6
            oim_call.chi_la = .8
            oim_call.chi_re = .2

            oim_call.biomass()
            oim_call.organic_carbon_decay()
            oim_call.sedimentation(
                sedimentation_om    = sed_om,
                sedimentation_min   = sed_min,
                f_Cla               = .40,
            )
            oim_call.update_layers()

            #print('surface at %.2f m' % oim_call.surface)

        # retrieve organic carbon
        C                   = 100 * oim_call.get_C() / oim_call.mass

        # interpolate the simulated C densities to the observed densities
        C_sim               = np.interp(auger_soil["depth"] / 100, oim_call.d, C)

        # calculate the root mean square error
        lhc_rmse_C[i]       = np.sqrt(np.mean(np.square(C_sim - auger_soil["C_percentage"])))

        # retrieve dry bulk density
        dbd_sim             = np.interp(auger_soil["depth"] / 100, oim_call.d, oim_call.get_dbd())

        # calculate the root mean square error
        lhc_rmse_DBD[i]     = np.sqrt(np.mean(np.square(dbd_sim - 1000 * auger_soil["DBD"])))

        # calculate the root mean square error on the elevation
        lhc_rmse_Z[i]     = np.sqrt(np.mean(np.square(oim_call.surface - years*elev[zone].loc[auger_ID])))



   #%% find optimal Kla, Kre and sedimentation using Bayesian likelihood maximization

    # choose observational error scales (example values)
    sigma_C = 1.0  # %C units
    sigma_DBD = 100.0  # kg m-3 (since you used 1000*DBD)
    sigma_Z = 0.15  # m

    logL = (
            -0.5 * (lhc_rmse_C / sigma_C) ** 2
            - 0.5 * (lhc_rmse_DBD / sigma_DBD) ** 2
            - 0.5 * (lhc_rmse_Z / sigma_Z) ** 2
    )

    L = np.exp(logL)

    # plot L in function of Kla, Kre and sedimentation
    fig_L, axs_L = plt.subplots(ncols = 1, nrows = 2, figsize = (5,8))

    hb1 = axs_L[0].hexbin(lhc_samples[:, 0], lhc_samples[:, 2], gridsize = 25, C = L, cmap = 'viridis', vmin = 0, vmax = np.max(L))
    hb2 = axs_L[1].hexbin(lhc_samples[:, 1], lhc_samples[:, 2], gridsize = 25, C = L, cmap = 'viridis', vmin = 0, vmax = np.max(L))
    axs_L[0].set_xlabel(r'$K_{la}$')
    axs_L[1].set_xlabel(r'$K_{re}$')
    axs_L[0].set_ylabel(r'sedimentation $[kg \; m^{-2} \; month^{-1}]$')
    axs_L[1].set_ylabel(r'sedimentation $[kg \; m^{-2} \; month^{-1}]$')
    fig_L.colorbar(hb1, ax = axs_L[0], label = r'Bayesian likelihood')
    fig_L.colorbar(hb2, ax = axs_L[1], label = r'Bayesian likelihood')

    Kla_opt, Kre_opt, sed_opt = lhc_samples[np.argmax(L)]

    axs_L[0].scatter(Kla_opt, sed_opt, 150, marker = '+' ,c = [0,0,0,0], ec = 'w', lw = 2)
    axs_L[1].scatter(Kre_opt, sed_opt, 150, marker = '+' ,c = [0,0,0,0], ec = 'w', lw = 2)

    fig_L.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/{auger_ID}_call_L.png')

    #%% run model with optimal Kla and Kre

    # sedimentation, based on Latin Hypercube sampling
    sed_om = sed_om_frac * sed_opt
    sed_min = (1 - sed_om_frac) * sed_opt

    for month in range(months + 1):

        oim.Bmax   = agb[month]
        oim.Dmbm   = root_shoot[month]
        oim.Kla0   = lhc_samples[np.argmin(lhc_rmse_C), 0]
        oim.Kre0   = lhc_samples[np.argmin(lhc_rmse_C), 1]
        oim.mu_la   = 1e6
        oim.mu_re   = 1e6
        oim.chi_la = .8
        oim.chi_re = .2

        oim.biomass()
        oim.organic_carbon_decay()
        oim.sedimentation(
            sedimentation_om    = sed_om,
            sedimentation_min   = sed_min,
            f_Cla               = .40,
        )
        oim.update_layers()

        print(oim.surface)


    # plot organic carbon
    C                           = 100 * oim.get_C() / oim.mass
    axs.plot(C, -oim.d, ls = ':', c = 'C0', label = 'simulated C [%]')

    # plot dry bulk density
    axs_inset.plot(oim.get_dbd(), -oim.d, ls = ':', c = 'C0')

    # set title
    text = [r'$K_{la} = %.4f$' % Kla_opt,
                   r'$K_{re} = %.5f$' % Kre_opt + '\n',
                   r'$sed_{sim} = %.2f$' % oim.surface,
                   r'$sed_{obs} = %.2f$' % (years*elev[zone].loc[auger_ID])]
    axs.text(0.95, 0.95, '\n'.join(text), ha = 'right', va = 'top', transform=axs.transAxes, fontsize = 10)
    axs.set_title(auger_ID)

    axs.legend(loc = 3)
    #%% add calibrated values to dataframe

    call_df.loc[auger_ID, "Kla0"]             = Kla_opt
    call_df.loc[auger_ID, "Kre0"]             = Kre_opt
    call_df.loc[auger_ID, "sed_opt"]          = sed_opt

    #%% save plots

    #fig.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/{auger_ID}_call_sedcomp_125per.png')
    fig.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/{auger_ID}_call.png')

call_df.to_csv(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/S{zone}y_call.csv')