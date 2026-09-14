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
sys.path.append(BASE_MODEL_DIR)
sys.path.append(BASE_MODEL_DIR + '/Blackwater/')
from OIMAS import OIMAS_N

from load_data_Blackwater import *

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

    # OIMAS parameters
    # Kla, Kre and k will be callibrated

    dt = 1

    ws = 2e-4
    sed_om_frac = 0.10
    init_om_frac = 0.05
    n_years = 100

    chi_re = 0.3
    chi_la = 0.7

    # -----------------------------------------------------------------------------
    #%% Load data
    # -----------------------------------------------------------------------------

    soil = load_soil_carbon()
    auger_soil = soil.loc[auger_ID]
    rtk, sar = load_rtk_data()

    hwl_df = load_hwls()
    avg_tide = load_avg_tide()

    # -----------------------------------------------------------------------------
    #%% set vegetation parameters
    # -----------------------------------------------------------------------------

    # polynomial coefficient to estimate biomass in funtion of inundation depth at mhwl
    biomass_coeff = [102.09192788, -302.56185477, 324.44520351, -151.47311919, 27.34594944, 0.0]

    # root-shoot ratio at moment of maximum biomass
    # ratio min to max shoot-root ratio
    # gamma = lamda = kappa
    veg_params = {
        'cynusuroides': [1.101, 0.985, .27],
        'americanus': [1.789, 1.314, .27],
        'alterniflora': [1.101, 0.985, .27],
    }

    # relative weights in function of mhwl depth
    def veg_weights(d):
        if d < 0.12:
            return 1, 0, 0
        elif d < 0.16:
            w_am = (d - 0.12) / (0.16 - 0.12)
            w_cyn = 1 - w_am
            return w_cyn, w_am, 0
        elif d < 0.19:
            return 0, 1, 0
        elif d < 0.23:
            w_alt = (d - 0.19) / (0.23 - 0.19)
            w_am = 1 - w_alt
            return 0, w_am, w_alt
        else:
            return 0, 0, 1

    # -----------------------------------------------------------------------------
    #%% Prepare figure and plot observations
    # -----------------------------------------------------------------------------

    fig, axs = plt.subplots(ncols= 1 , sharex=True, sharey=True, figsize=(7, 8))
    axs_inset = axs.inset_axes([.55, 0.05, 0.35, 0.25], facecolor=[0.1, 0.15, 0.15, 0.85])
    plot_carbon_profiles([axs], [auger_soil])

    # -----------------------------------------------------------------------------
    #%% Model initialization
    # -----------------------------------------------------------------------------

    years = range(2023 - n_years + 1, 2023 + 1)
    n_layers = 50
    om_init, min_init = np.full(n_layers, init_om_frac * 7.5), np.full(n_layers, 7.5)

    sar = sar.loc[auger_ID[:-1]]['SAR']
    init_elev = 0

    oim = OIMAS_N(
        n_layers=n_layers,
        dt=dt,
        f_C=0.44,
        rho_min=1990, rho_om=850,
        sigma_ref_min=1e5, sigma_ref_om=1e4,
        CI_min=0, CI_om=0,
        E0_min=0.7, E0_om=587,
        Kla0=0, Kre0=0,
        chi_la=chi_la, chi_re=1-chi_la,
        max_layer_thickness=0.07,
    )

    oim.initialize_layers(
        init_min_mass=min_init,
        init_om_mass=om_init,
        f_Cla=0.00,
        initial_surface=init_elev,
    )

    oim.update_geometry(surface=init_elev)



    # -----------------------------------------------------------------------------
    #%% Run simulation for callibration
    # -----------------------------------------------------------------------------

    # prepare a Latin Hypercube for sensitivity analysis for K labile
    param_bounds = {'Kla0': (-4, -1),
                    'Kre0': (-9, -5),
                    'k':    (0, 0.05)}

    lhc_sampler     = scipy.stats.qmc.LatinHypercube(d = 3)
    lhc_samples     = lhc_sampler.random(n = lhs_n)
    lhc_samples     = scipy.stats.qmc.scale(lhc_samples,
                            [param_bounds['Kla0'][0], param_bounds['Kre0'][0], param_bounds['k'][0],],
                            [param_bounds['Kla0'][1], param_bounds['Kre0'][1], param_bounds['k'][1],])
    lhc_samples     = lhc_samples[(lhc_samples[:,0] > lhc_samples[:,1]),:]

    lhc_rmse_C      = np.zeros(lhc_samples.shape[0])
    lhc_rmse_DBD    = np.zeros(lhc_samples.shape[0])
    lhc_rmse_Z      = np.zeros(lhc_samples.shape[0])
    lhc_rmse_sar    = np.zeros(lhc_samples.shape[0])


    # callibrate K labile
    for i, (Kla_exp, Kre_exp, k) in tqdm(enumerate(lhc_samples), total=lhc_samples.shape[0]):
        # copy the oim instance to not overwrite the original
        oim_call            = oim.copy()

        surface = []

        for year in years:

            # get sea level rise
            delta_msl = 0.47623 * np.exp(0.0161643 * (year - 2014)) - 0.476
            hwls = hwl_df.values.flatten() + delta_msl
            mhwl = np.mean(hwls)

            # set biomass model parameters
            z = oim_call.surface
            mhwl_d = max(mhwl - z, 0)
            oim_call.Bmax = np.polyval(biomass_coeff, mhwl_d)

            # set decay coefficients
            oim_call.Kla0   = np.exp(Kla_exp)
            oim_call.Kre0   = np.exp(Kre_exp)

            # set fraction of labile and recalcitrant carbon
            oim_call.chi_la = chi_la
            oim_call.chi_re = 1 - chi_la

            # set vegetation parameters depending on the surface
            w_cyn, w_am, w_alt = veg_weights(mhwl_d)
            R = w_cyn * veg_params["cynusuroides"][0] + w_am * veg_params["americanus"][0] + w_alt * \
                veg_params["alterniflora"][0]
            TO = w_cyn * veg_params["cynusuroides"][1] + w_am * veg_params["americanus"][1] + w_alt * \
                 veg_params["alterniflora"][1]
            gamma = w_cyn * veg_params["cynusuroides"][2] + w_am * veg_params["americanus"][2] + w_alt * \
                    veg_params["alterniflora"][2]

            oim_call.gamma = oim_call.kappa = oim_call.lamda = gamma
            oim_call.root_to_shoot = R
            oim_call.turnover = TO

            # run model
            oim_call.biomass()
            oim_call.organic_carbon_decay()
            hwls = hwl_df.values.flatten() + delta_msl
            oim_call.marsed(hwls, avg_tide.index, avg_tide.avg_H, k=k, ws=ws, sed_om_frac=sed_om_frac, use_Julia=True, f_Cla=chi_la)
            oim_call.update_layers()

            # save surface
            surface.append(oim_call.surface)

        # retrieve organic carbon
        C                   = 100 * oim_call.get_C() / oim_call.mass

        # interpolate the simulated C densities to the observed densities
        C_sim               = np.interp(auger_soil["depth"] , oim_call.d, C)
        mask = auger_soil["depth"] < sar * n_years

        # calculate the root mean square error
        lhc_rmse_C[i]       = np.sqrt(np.mean(np.square(C_sim[mask] - auger_soil["C_percentage"][mask])))

        # calculate the root mean square error on the elevation
        lhc_rmse_Z[i]     = np.sqrt(np.mean(np.square(oim_call.surface - rtk.loc[auger_ID].iloc[0])))

        # calculate RMSE on sar
        sarr_sim, _ = np.polyfit(np.arange(25), surface[-25:], deg=1)
        lhc_rmse_sar[i]     = np.sqrt(np.mean(np.square(sarr_sim - sar)))

    #%% find optimal Kla, Kre and sedimentation using Bayesian likelihood maximization

    print(lhc_rmse_C)
    print(lhc_rmse_Z)

    # normalize errors
    lhc_rmse_C_n        = lhc_rmse_C / np.std(lhc_rmse_C)
    lhc_rmse_DBD_n      = lhc_rmse_DBD / np.std(lhc_rmse_DBD)
    lhc_rmse_Z_n        = lhc_rmse_Z / np.std(lhc_rmse_Z)

    logL = -0.5 * (
            1.0 * lhc_rmse_C_n ** 2 +
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
    hb1 = axs_L[0].scatter(np.exp(lhc_samples[:, 0]), lhc_samples[:, 2], c=L, cmap=cmap, vmin= np.min(L), vmax=np.max(L), s = 5)
    hb2 = axs_L[1].scatter(np.exp(lhc_samples[:, 0]), np.exp(lhc_samples[:, 1]), c=L, cmap=cmap, vmin= np.min(L), vmax=np.max(L), s = 5)

    axs_L[0].set_xlabel(r'$K_{la}$')
    axs_L[1].set_xlabel(r'$K_{la}$')
    axs_L[0].set_ylabel(r'$k_{MARSED}$')
    axs_L[1].set_ylabel(r'$K_{re}$')

    fig_L.colorbar(hb1, ax=axs_L[0], label=r'Bayesian likelihood')
    fig_L.colorbar(hb2, ax=axs_L[1], label=r'Bayesian likelihood')

    Kla_opt_exp, Kre_opt_exp, k_opt = lhc_samples[-1,:]

    Kla_opt = np.exp(Kla_opt_exp)
    Kre_opt = np.exp(Kre_opt_exp)

    axs_L[0].scatter(Kla_opt, k_opt, 150, marker='o', color=[0, 0, 0, 0], ec='k', lw=2)
    axs_L[1].scatter(Kla_opt, Kre_opt, 150, marker='o', color=[0, 0, 0, 0], ec='k', lw=2)

    fig_L.tight_layout()
    axs_L[0].set_xscale("log")
    axs_L[1].set_xscale("log")
    axs_L[1].set_yscale("log")
    fig_L.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Blackwater/callibration_output/{auger_ID}_call_L.png')

    # create dataframe with callibration results
    call_df = pd.DataFrame(lhc_samples, columns=["Kla0", "Kre0", "sed_opt"])
    call_df['RMSE_C'] = lhc_rmse_C
    call_df['RMSE_DBD'] = lhc_rmse_DBD
    call_df['RMSE_Z'] = lhc_rmse_Z
    call_df['Likelihood'] = L
    call_df.to_csv(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Blackwater/callibration_output/{auger_ID}_call_df.csv')



    #%% run model with optimal Kla and Kre

    agb_series = []
    mhwl_d_series = []
    z_series = []
    species_series = []

    for year in years:

        # get sea level rise
        delta_msl = 0.47623 * np.exp(0.0161643 * (year - 2014)) - 0.476
        hwls = hwl_df.values.flatten() + delta_msl
        mhwl = np.mean(hwls)

        # set biomass model parameters
        z = oim.surface
        mhwl_d = max(mhwl - z, 0)
        oim.Bmax = np.polyval(biomass_coeff, mhwl_d)

        # set decay coefficients
        oim.Kla0   = Kla_opt
        oim.Kre0   = Kre_opt


        # set fraction of labile and recalcitrant carbon
        oim.chi_la = chi_la
        oim.chi_re = 1 - chi_la

        # set vegetation parameters depending on the surface
        w_cyn, w_am, w_alt = veg_weights(mhwl_d)


        R = w_cyn * veg_params["cynusuroides"][0] + w_am * veg_params["americanus"][0] + w_alt * \
            veg_params["alterniflora"][0]
        TO = w_cyn * veg_params["cynusuroides"][1] + w_am * veg_params["americanus"][1] + w_alt * \
             veg_params["alterniflora"][1]
        gamma = w_cyn * veg_params["cynusuroides"][2] + w_am * veg_params["americanus"][2] + w_alt * \
                veg_params["alterniflora"][2]

        oim.gamma = oim.kappa = oim.lamda = gamma
        oim.root_to_shoot = R
        oim.turnover = TO

        # run model
        oim.biomass()
        oim.organic_carbon_decay()
        hwls = hwl_df.values.flatten() + delta_msl
        oim.marsed(hwls, avg_tide.index, avg_tide.avg_H, k=k_opt, ws=ws, sed_om_frac=sed_om_frac, use_Julia=True, f_Cla= chi_la)

        oim.update_layers()

        # save biomass and depth
        agb_series.append(oim.Agb)
        mhwl_d_series.append(mhwl_d)
        z_series.append(oim.surface - delta_msl)
        species_series.append([w_cyn, w_am, w_alt])

    # plot organic carbon
    C                           = 100 * oim.get_C() / oim.mass

    mask = oim.d < sar * n_years
    axs.plot(C[mask], -oim.d[mask], ls = '--', color = 'w', label = 'simulated C [%]', zorder = 10)

    # plot dry bulk density
    sc = axs_inset.scatter(years, agb_series, 5, color = species_series)
    axs_inset_right = axs_inset.twinx()
    axs_inset_right.plot(years, mhwl_d_series, ls = ':', color = 'C1')

    axs_inset.set_ylabel('AGB [kg m$^{-2}$]')
    axs_inset_right.set_ylabel('z [m MSL]', color = 'C1')
    axs_inset_right.set_ylim(-0.2,0.4)
    axs_inset_right.tick_params(axis='y', labelcolor='C1')
    axs_inset.grid(False)
    axs_inset.legend(handles=[matplotlib.lines.Line2D([0], [0], marker='o', lw=0, color=c, markersize=5) for c in
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                      labels=['S. cynosuroides', 'S. americanus', 'S. alterniflora'], frameon=False, fontsize=8)

    # set title
    text = [r'$K_{la} = %.7f$' % Kla_opt,
                   r'$K_{re} = %.7f$' % Kre_opt,
                   r'$k = %.7f$' % k_opt + '\n',
                   r'$z_{sim} = %.2f$' % oim.surface,
                   r'$z_{obs} = %.2f$' % (rtk.loc[auger_ID].iloc[0])]
    axs.text(0.95, 0.95, '\n'.join(text), ha = 'right', va = 'top', transform=axs.transAxes, fontsize = 10)

    axs.set_xlim(0, 35)
    axs.legend(loc = 3)

    #%% add evaluation metric

    # interpolate the simulated C densities to the observed densities
    C_sim = np.interp(auger_soil["depth"], oim.d, C)

    # calculate the root mean square error
    rmse_C = np.sqrt(np.mean(np.square(C_sim - auger_soil["C_percentage"])))

    axs.set_title(f'{auger_ID} - RMSE = {rmse_C:.2f} %')

    #%% save plots
    fig.savefig(f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Blackwater/callibration_output/{auger_ID}_call.png')

    cal_path = f'/Users/ignace/Documents/WETCOAST/model/OIMAS-N/Blackwater/callibration_output/callibrated_params.csv'
    df = pd.read_csv(cal_path, index_col=0)
    df.loc[auger_ID] = [Kla_opt, Kre_opt, k_opt]
    df.to_csv(cal_path)

if __name__ == "__main__":
    args = parse_args()
    main(args.auger_id, lhs_n = args.lhs_n)