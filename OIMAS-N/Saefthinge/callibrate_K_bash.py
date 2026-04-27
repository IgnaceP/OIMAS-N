import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scipy
import pandas as pd
from datetime import datetime

from tqdm import tqdm
from callibrate_K_bash_functions import *
import gc

matplotlib.style.use("ip02")

BASE_MODEL_DIR = "/Users/ignace/Documents/WETCOAST/model/OIMAS-N"
os.chdir(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR)

from OIMAS import OIMAS_N
from read_C_obs_data import read_observation_data

from functions import (
    load_soil_carbon,
    load_bgb_data,
    load_elevation_data,
    plot_carbon_profiles,
    plot_dbd_profiles,
    compute_initial_masses,
)

matplotlib.use('agg')

def parse_args():
    parser = argparse.ArgumentParser(description="Run OIMAS calibration for a given auger ID")
    parser.add_argument(
        "--auger_id",
        type=str,
        required=False,
        help="Auger ID (e.g., S10y1)",
        default = "S10y1"
    )

    parser.add_argument(
        "--lhs_n",
        type=int,
        required=False,
        help="number of latin hypercub samples",
        default=100000
    )
    return parser.parse_args()

def main(auger_ID, lhs_n = 100000):
    #%% loop over all auger IDs in a zone

    zone        = int(auger_ID[1:3])

    # -----------------------------------------------------------------------------
    # %% set model parameters
    # -----------------------------------------------------------------------------

    # numerics
    dt = 1
    n_layers = 20

    # initial soil surface
    init_elev = 4.5

    # root-shoot ratio at moment of maximum biomass
    # ratio min to max shoot-root ratio
    # gamma, lamda and kappa

    veg_params = {
        'Tripolium': [0.82, 0.5, .10],
        'Atriplex': [0.90, 1.0, .17],
        'Bolboschoenus': [1.99, 0.8, .20],
        'Elytrigia': [0.70, 0.8, .20]
    }

    # MARSED parameters
    ws = 1.1e-4
    sed_om_frac = 0.049

    # -----------------------------------------------------------------------------
    #%% Load data
    # -----------------------------------------------------------------------------

    soil        = load_soil_carbon([0, zone], read_observation_data)
    bgb         = load_bgb_data()
    elev        = load_elevation_data([zone])
    rtk         = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/RTK/sample_locations_RTK.csv', index_col=1)[["z_TAW"]]
    sar         = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/LiDAR/surface_elevation_accumulation.csv', index_col=0)
    rtk         = rtk.join(sar['SAR'], how = 'inner')

    avg_tide = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/Getij/Kloosterzande_avg_H.csv',
                           index_col=0)
    hwl_df = pd.read_csv('/Users/ignace/Documents/WETCOAST/Data/Saefthinge/Getij/Kloosterzande_HWLs_1986-2025.csv',
                         skiprows=1, index_col=0, parse_dates=True)

    # -----------------------------------------------------------------------------
    #%% Plot observations
    # -----------------------------------------------------------------------------

    fig, axs     = plt.subplots(ncols= 1 , sharex=True, sharey=True, figsize=(7, 8))

    auger_soil  = soil[zone].loc[auger_ID]
    auger_bgb   = bgb.loc[auger_ID]

    plot_carbon_profiles([axs], [auger_soil])

    # -----------------------------------------------------------------------------
    #%% Model initialization (S0y)
    # -----------------------------------------------------------------------------

    # initial elevation
    init_elev = rtk.loc[auger_ID]["z_TAW"] - zone * rtk.loc[auger_ID]["SAR"]
    print(f'{auger_ID} initial elevation: {init_elev:.3f} m')

    # create OIMAS instance
    oim = OIMAS_N(
        n_layers = n_layers,
        dt= dt,
        sigma_ref_min="top", sigma_ref_om="top",
        f_C = 0.5208,
        CI_min = 0.025, CI_om = 1.0,
        E0_min = 0.861, E0_om = 27.545,
        chi_la = 0.80, chi_re = 0.20,
        max_layer_thickness = 0.07,
        gamma = 0.15, kappa = 0.15, lamda = 0.15)

    # initialize layers
    om_init, min_init = compute_initial_masses(soil[0], n_layers)
    oim.initialize_layers(
        init_min_mass=min_init,
        init_om_mass=om_init,
        f_Cla = 0.4,
        initial_surface = init_elev
    )

    # -----------------------------------------------------------------------------
    #%% Simulation setup (0–XX years)
    # -----------------------------------------------------------------------------

    # total number of timesteps to simulate
    years           = zone
    timesteps       = years

    # -----------------------------------------------------------------------------
    #%% Run simulation for callibration
    # -----------------------------------------------------------------------------

    # prepare inset axes
    axs_inset       = axs.inset_axes([.7, 0.1, 0.25, 0.5])
    axs_inset.set_facecolor([.2, .2, .2, .75])
    plot_dbd_profiles([axs_inset], [auger_soil])

    # prepare a Latin Hypercube for sensitivity analysis for K labile
    param_bounds    = {'Kla0': (0, 0.25),
                       'Kre0': (0, 0.1),
                       'k': (0, .5)}

    lhc_sampler     = scipy.stats.qmc.LatinHypercube(d = 3)
    lhc_samples     = lhc_sampler.random(n = lhs_n)
    lhc_samples     = scipy.stats.qmc.scale(lhc_samples,
                            [param_bounds['Kla0'][0], param_bounds['Kre0'][0], param_bounds['k'][0],],
                            [param_bounds['Kla0'][1], param_bounds['Kre0'][1], param_bounds['k'][1],])

    # mask out where Kla < Kre
    lhc_samples     = lhc_samples[lhc_samples[:, 0] > lhc_samples[:, 1]]

    # prepare arrays to store performance results
    lhc_rmse_C      = np.zeros(lhc_samples.shape[0])
    lhc_rmse_DBD    = np.zeros(lhc_samples.shape[0])
    lhc_rmse_Z      = np.zeros(lhc_samples.shape[0])

    # callibrate K labile
    for i, (Kla, Kre, k) in tqdm(enumerate(lhc_samples), total=lhc_samples.shape[0]):

        # copy the oim instance to not overwrite the original
        oim_call    = oim.copy()

        for timestep in range(timesteps + 1):
            # set years
            t0 = datetime(2026 - timesteps + (timestep - 1), 1, 1)
            t1 = datetime(2026 - timesteps + (timestep - 1), 12, 31)

            # update max biomass
            oim_call.Bmax   = 1.45 * oim.surface - 5.58

            # set decay coefficients
            oim_call.Kla0   = Kla
            oim_call.Kre0   = Kre

            # set vegetation parameters depending on the surface
            if oim_call.surface > 5.05:
                rootshoot, turnover, gamma = veg_params["Elytrigia"]
            elif oim_call.surface > 4.8:
                rootshoot, turnover, gamma = veg_params["Bolboschoenus"]
            else:
                rootshoot, turnover, gamma = veg_params["Tripolium"]
            oim_call.gamma = oim_call.kappa = oim_call.lamda = gamma
            oim_call.root_to_shoot = rootshoot
            oim_call.turnover = turnover

            # run model
            oim_call.biomass()
            oim_call.organic_carbon_decay()
            oim_call.marsed(
                hwls        = hwl_df.loc[t0:t1].values.flatten(),
                avg_tide_t  = avg_tide.index,
                avg_tide_h  = avg_tide.avg_H,
                k           = k,
                sed_om_frac =sed_om_frac,
                f_Cla       = .40,
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

        #if i % 1000:
        #    gc.collect()



    #%% find optimal Kla, Kre and sedimentation using Bayesian likelihood maximization

    # normalize errors
    lhc_rmse_C_n        = lhc_rmse_C / np.std(lhc_rmse_C)
    lhc_rmse_DBD_n      = lhc_rmse_DBD / np.std(lhc_rmse_DBD)
    lhc_rmse_Z_n        = lhc_rmse_Z / np.std(lhc_rmse_Z)

    logL = -0.5 * (
            1.0 * lhc_rmse_C_n ** 2 +
            1.0 * lhc_rmse_DBD_n ** 2 +
            1.0 * lhc_rmse_Z_n ** 2
    )

    L = np.exp(logL)

    # plot L in function of Kla, Kre and sedimentation
    fig_L, axs_L     = plt.subplots(ncols = 1, nrows = 2, figsize = (5,8))

    # sort lhc_samples based on L
    lhc_samples     = lhc_samples[np.argsort(L),:]
    L               = L[np.argsort(L)]

    # create colormap from [0,0,0,0] to maroon
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('black to golden', [(0,0,0,0.1), (1.0, 0.65, 0.0, .5), (1.0, 0.85, 0.0, 1.),(1,1,1,1)])
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list('white to maroon', [(1,1,1,0), (0.5, 0, 0, .5),(0,0,0,1)])
    hb1 = axs_L[0].scatter(lhc_samples[:, 0], lhc_samples[:, 2], c=L, cmap=cmap, vmin=0, vmax=np.max(L), s = 5)
    hb2 = axs_L[1].scatter(lhc_samples[:, 0], lhc_samples[:, 1], c=L, cmap=cmap, vmin=0, vmax=np.max(L), s = 5)

    axs_L[0].set_xlabel(r'$K_{la}$')
    axs_L[1].set_xlabel(r'$K_{la}$')
    axs_L[0].set_ylabel(r'$k (MARSED)$')
    axs_L[1].set_ylabel(r'$K_{re}$')

    fig_L.colorbar(hb1, ax=axs_L[0], label=r'Bayesian likelihood')
    fig_L.colorbar(hb2, ax=axs_L[1], label=r'Bayesian likelihood')

    Kla_opt, Kre_opt, k_opt = lhc_samples[np.argmax(L)]
    # Kla_opt = np.exp(Kla_opt_exp)
    # Kre_opt = np.exp(Kre_opt_exp)

    axs_L[0].scatter(Kla_opt, k_opt, 150, marker='o', color=[0, 0, 0, 0], ec='k', lw=2)
    axs_L[1].scatter(Kla_opt, Kre_opt, 150, marker='o', color=[0, 0, 0, 0], ec='k', lw=2)

    fig_L.tight_layout()
    fig_L.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Saefthinge/callibration_output/{auger_ID}_call_L.png')

    #%% run model with optimal Kla and Kre

    for timestep in range(timesteps + 1):
        # set years
        t0 = datetime(2026 - timesteps + (timestep - 1), 1, 1)
        t1 = datetime(2026 - timesteps + (timestep - 1), 12, 31)

        # update max biomass
        oim.Bmax    = 1.45 * oim.surface - 5.58

        # set decay coefficients
        oim.Kla0   = Kla_opt
        oim.Kre0   = Kre_opt

        # set vegetation parameters depending on the surface
        if oim_call.surface > 5.05:
            rootshoot, turnover, gamma = veg_params["Elytrigia"]
        elif oim_call.surface > 4.8:
            rootshoot, turnover, gamma = veg_params["Bolboschoenus"]
        else:
            rootshoot, turnover, gamma = veg_params["Tripolium"]
        oim_call.gamma = oim_call.kappa = oim_call.lamda = gamma
        oim_call.root_to_shoot = rootshoot
        oim_call.turnover = turnover

        # run model
        oim.biomass()
        oim.organic_carbon_decay()
        oim.marsed(
            hwls=hwl_df.loc[t0:t1].values.flatten(),
            avg_tide_t=avg_tide.index,
            avg_tide_h=avg_tide.avg_H,
            k=k_opt,
            sed_om_frac=sed_om_frac,
            f_Cla=.40,
        )
        oim.update_layers()



    # plot organic carbon
    C       = 100 * oim.get_C() / oim.mass
    axs.plot(C, -oim.d, ls = '--', color = 'w', label = 'simulated C [%]', zorder = 10)

    # plot dry bulk density
    axs_inset.plot(oim.get_dbd(), -oim.d, ls = ':', color = 'C0')

    # set title
    text = [r'$K_{la} = %.4f$' % Kla_opt,
                   r'$K_{re} = %.5f$' % Kre_opt,
                   r'$k = %.2f$' % k_opt + '\n',
                   r'$z_{sim} = %.2f$' % oim.surface,
                   r'$z_{obs} = %.2f$' % (rtk.loc[auger_ID].iloc[0])]
    axs.text(0.95, 0.95, '\n'.join(text), ha = 'right', va = 'top', transform=axs.transAxes, fontsize = 10)
    axs.set_title(auger_ID)

    axs.legend(loc = 3)

    #%% save plots

    #fig.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/callibration_output/{auger_ID}_call_sedcomp_125per.png')
    fig.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Saefthinge/callibration_output/{auger_ID}_call.png')

    cal_path = f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Saefthinge/callibration_output/callibrated_params.csv'
    df = pd.read_csv(cal_path, index_col=0)
    df.loc[auger_ID] = [Kla_opt, Kre_opt, k_opt]
    df.to_csv(cal_path)

if __name__ == "__main__":
    args = parse_args()
    main(args.auger_id, lhs_n = args.lhs_n)