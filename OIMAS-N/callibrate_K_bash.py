import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

matplotlib.use('agg')

def parse_args():
    parser = argparse.ArgumentParser(description="Run OIMAS calibration for a given auger ID")
    parser.add_argument(
        "--auger_id",
        type=str,
        required=True,
        help="Auger ID (e.g., S10y1)"
    )

    parser.add_argument(
        "--lhs_n",
        type=int,
        required=False,
        help="number of latin hypercub samples",
        default=10000
    )
    return parser.parse_args()

def main(auger_ID, lhs_n = 10000):
    #%% loop over all auger IDs in a zone

    zone = int(auger_ID[1:3])

    init_elev = 4.5

    # -----------------------------------------------------------------------------
    #%% Load data
    # -----------------------------------------------------------------------------

    soil = load_soil_carbon([0, zone], read_observation_data)
    bgb = load_bgb_data()
    biomass = load_biomass_data()
    elev = load_elevation_data([zone])
    rtk = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/RTK/sample_locations_RTK.csv', index_col=1)[["z_TAW"]]

    # -----------------------------------------------------------------------------
    #%% set vegetation parameters
    # -----------------------------------------------------------------------------

    # root-shoot ratio at moment of maximum biomass
    # ratio min to max shoot-root ratio
    # gamma, lamda and kappa

    veg_params = {
        'Tripolium': [.82, 0.5, .1],
        'Atriplex': [.9, 0, .17],
        'Bolboschoenus': [1.99, .75, .2],
        'Elytrigia': [0.7, 0.9, .2]
    }

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
        f_Cla = 0.4,
        initial_surface = init_elev
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
    sed_mass_min = max(.5 * elev[zone].min().iloc[0] * 1000 * dbd_top_layer / 12, 0)

    # -----------------------------------------------------------------------------
    #%% Run simulation for callibration
    # -----------------------------------------------------------------------------

    # prepare inset axes
    axs_inset       = axs.inset_axes([.7, 0.1, 0.25, 0.5])
    axs_inset.set_facecolor([.2, .2, .2, .75])
    plot_dbd_profiles([axs_inset], [auger_soil])

    # prepare a Latin Hypercube for sensitivity analysis for K labile
    param_bounds = {'Kla0': (-10, 0),
                    'Kre0': (0, 0.0003),
                    'sed': (sed_mass_min, sed_mass_max)}

    lhc_sampler     = scipy.stats.qmc.LatinHypercube(d = 3)
    lhc_samples     = lhc_sampler.random(n = lhs_n)
    lhc_samples     = scipy.stats.qmc.scale(lhc_samples,
                            [param_bounds['Kla0'][0], param_bounds['Kre0'][0], param_bounds['sed'][0],],
                            [param_bounds['Kla0'][1], param_bounds['Kre0'][1], param_bounds['sed'][1],])
    lhc_rmse_C      = np.zeros(lhc_samples.shape[0])
    lhc_rmse_DBD    = np.zeros(lhc_samples.shape[0])
    lhc_rmse_Z      = np.zeros(lhc_samples.shape[0])

    # callibrate K labile
    for i, (Kla_exp, Kre, total_sed_mass) in tqdm(enumerate(lhc_samples), total=lhc_samples.shape[0]):
        # copy the oim instance to not overwrite the original
        oim_call            = oim.copy()

        # sedimentation, based on Latin Hypercube sampling
        sed_om = sed_om_frac * total_sed_mass
        sed_min = (1 - sed_om_frac) * total_sed_mass

        for month in range(months + 1):

            # update max biomass
            oim_call.Bmax   = agb[month]

            # set decay coefficients
            oim_call.Kla0   = np.exp(Kla_exp)
            oim_call.Kre0   = Kre

            # set fraction of labile and recalcitrant carbon
            oim_call.chi_la = .8
            oim_call.chi_re = .2

            # set vegetation parameters depending on the surface
            if oim_call.surface > 5.05:
                R, m, gamma = veg_params["Elytrigia"]
            elif oim_call.surface > 4.8:
                R, m, gamma = veg_params["Bolboschoenus"]
            else:
                R, m, gamma = veg_params["Tripolium"]

            day_peak = 244
            rs = lambda d: .5 * (R - m * R) * np.cos(2 * np.pi * (d - day_peak) / 365) + .5 * (R - m * R) + m * R

            oim_call.gamma = oim_call.kappa = oim_call.lamda = gamma
            oim_call.root_to_shoot = rs

            # run model
            oim_call.biomass()
            oim_call.organic_carbon_decay()
            oim_call.sedimentation(
                sedimentation_om    = sed_om,
                sedimentation_min   = sed_min,
                f_Cla               = .40,
            )
            oim_call.update_layers()


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
        lhc_rmse_Z[i]     = np.sqrt(np.mean(np.square(oim_call.surface - rtk.loc[auger_ID]['z_TAW'])))

        # plot organic carbon for all callibration runs
        # axs.plot(C, -oim_call.d, ls='-', c=matplotlib.cm.plasma(Kre/.0003), alpha = .1, zorder = 0)


    """# plot callibration results
    cmap = matplotlib.cm.plasma
    norm = matplotlib.colors.Normalize(vmin=0, vmax=.0003)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = axs.inset_axes([.05, .1, .4, .025], transform = axs.transAxes)
    cbar = plt.colorbar(sm, cax=cax, orientation = 'horizontal')
    cbar.set_label(r'$K_{re}$', fontsize = 10)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    axs.set_ylim(-0.95,.09)"""


    #%% find optimal Kla, Kre and sedimentation using Bayesian likelihood maximization

    # normalize errors
    lhc_rmse_C_n        = lhc_rmse_C / np.std(lhc_rmse_C)
    lhc_rmse_DBD_n      = lhc_rmse_DBD / np.std(lhc_rmse_DBD)
    lhc_rmse_Z_n        = lhc_rmse_Z / np.std(lhc_rmse_Z)

    logL = -0.5 * (
            2.0 * lhc_rmse_C_n ** 2 +
            1.0 * lhc_rmse_DBD_n ** 2 +
            1.0 * lhc_rmse_Z_n ** 2
    )

    L = np.exp(logL)

    # plot L in function of Kla, Kre and sedimentation
    fig_L, axs_L = plt.subplots(ncols = 1, nrows = 2, figsize = (5,8))

    hb1 = axs_L[0].hexbin(lhc_samples[:, 0], lhc_samples[:, 2], gridsize=35, C=L, cmap='viridis', vmin=0,
                          vmax=np.max(L))
    hb2 = axs_L[1].hexbin(lhc_samples[:, 0], lhc_samples[:, 1], gridsize=35, C=L, cmap='viridis', vmin=0,
                          vmax=np.max(L))

    axs_L[0].set_xlabel(r'log($K_{la}$)')
    axs_L[1].set_xlabel(r'log($K_{la}$)')
    axs_L[0].set_ylabel(r'sedimentation $[kg \; m^{-2} \; month^{-1}]$')
    axs_L[1].set_ylabel(r'log($K_{re}$)')

    fig_L.colorbar(hb1, ax=axs_L[0], label=r'Bayesian likelihood')
    fig_L.colorbar(hb2, ax=axs_L[1], label=r'Bayesian likelihood')

    Kla_opt_exp, Kre_opt, sed_opt = lhc_samples[np.argmax(L)]
    Kla_opt = np.exp(Kla_opt_exp)

    axs_L[0].scatter(Kla_opt_exp, sed_opt, 150, marker='+', color=[0, 0, 0, 0], ec='w', lw=2)
    axs_L[1].scatter(Kla_opt_exp, Kre_opt, 150, marker='+', color=[0, 0, 0, 0], ec='w', lw=2)

    fig_L.tight_layout()
    fig_L.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/{auger_ID}_call_L.png')

    # create dataframe with callibration results
    call_df = pd.DataFrame(lhc_samples, columns=["Kla0", "Kre0", "sed_opt"])
    call_df['RMSE_C'] = lhc_rmse_C
    call_df['RMSE_DBD'] = lhc_rmse_DBD
    call_df['RMSE_Z'] = lhc_rmse_Z
    call_df['Likelihood'] = L
    call_df.to_csv(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/{auger_ID}_call_df.csv')


    #%% run model with optimal Kla and Kre

    # sedimentation, based on Latin Hypercube sampling
    sed_om = sed_om_frac * sed_opt
    sed_min = (1 - sed_om_frac) * sed_opt

    for month in range(months + 1):

        # update max biomass
        oim.Bmax = agb[month]

        # set decay coefficients
        oim.Kla0   = Kla_opt
        oim.Kre0   = Kre_opt

        # set fraction of labile and recalcitrant carbon
        oim.chi_la = .8
        oim.chi_re = .2

        # set vegetation parameters depending on the surface
        if oim.surface > 5.05:
            R, m, gamma = veg_params["Elytrigia"]
        elif oim.surface > 4.8:
            R, m, gamma = veg_params["Bolboschoenus"]
        else:
            R, m, gamma = veg_params["Tripolium"]

        day_peak = 244
        rs = lambda d: .5 * (R - m * R) * np.cos(2 * np.pi * (d - day_peak) / 365) + .5 * (R - m * R) + m * R

        oim.gamma = oim.kappa = oim.lamda = gamma
        oim.root_to_shoot = rs

        # run model
        oim.biomass()
        oim.organic_carbon_decay()
        oim.sedimentation(
            sedimentation_om    = sed_om,
            sedimentation_min   = sed_min,
            f_Cla               = .40,
        )
        oim.update_layers()



    # plot organic carbon
    C                           = 100 * oim.get_C() / oim.mass
    axs.plot(C, -oim.d, ls = '--', color = 'w', label = 'simulated C [%]', zorder = 10)

    # plot dry bulk density
    axs_inset.plot(oim.get_dbd(), -oim.d, ls = ':', color = 'C0')

    # set title
    text = [r'$K_{la} = %.4f$' % Kla_opt,
                   r'$K_{re} = %.5f$' % Kre_opt,
                   r'$sed = %.2f$' % sed_opt + '\n',
                   r'$z_{sim} = %.2f$' % oim.surface,
                   r'$z_{obs} = %.2f$' % (rtk.loc[auger_ID].iloc[0])]
    axs.text(0.95, 0.95, '\n'.join(text), ha = 'right', va = 'top', transform=axs.transAxes, fontsize = 10)
    axs.set_title(auger_ID)

    axs.legend(loc = 3)
    #%% add calibrated values to dataframe
    """
    call_df.loc[auger_ID, "Kla0"]             = Kla_opt
    call_df.loc[auger_ID, "Kre0"]             = Kre_opt
    call_df.loc[auger_ID, "sed_opt"]          = sed_opt
    """
    #%% save plots

    #fig.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/{auger_ID}_call_sedcomp_125per.png')
    fig.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/{auger_ID}_call.png')

    cal_path = f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/callibrated_params.csv'
    df = pd.read_csv(cal_path, index_col=0)
    df.loc[auger_ID] = [Kla_opt, Kre_opt, sed_opt]
    df.to_csv(cal_path)

if __name__ == "__main__":
    args = parse_args()
    main(args.auger_id, lhs_n = args.lhs_n)