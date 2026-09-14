# blackwater_functions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OIMAS import OIMAS_N
from read_C_obs_data import read_observation_data


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = "/Users/ignace/Documents/WETCOAST"
DATA_DIR = f"{BASE_DIR}/Data/Saefthinge"
SOIL_CARBON_DIR = f"{DATA_DIR}/soil_carbon"
BIOMASS_DIR = f"{DATA_DIR}/biomass"


# =============================================================================
# Data loading
# ======================== =====================================================


def oimas_run(oim, years, hwl_df, avg_tide, biomass_coeff, veg_params, veg_weights, Kla, Kre, k, chi_la, chi_re, ws, sed_om_frac):
    C_tot = []
    M_tot = []
    mhwl_d_tot = []
    AGB_tot = []
    spec = []
    surface = []

    for year in years:

        delta_msl = 0.47623 * np.exp(0.0161643 * (year - 2014)) - 0.476
        hwls = hwl_df.values.flatten() + delta_msl
        mhwl = np.mean(hwls)

        # set model parameters
        z = oim.surface
        mhwl_d = max(mhwl - z, 0)

        oim.Bmax = np.polyval(biomass_coeff, mhwl_d)

        oim.Kla0 = Kla
        oim.Kre0 = Kre
        oim.chi_la = chi_la
        oim.chi_re = chi_re

        w_cyn, w_am, w_alt = veg_weights(mhwl_d)

        R = w_cyn * veg_params["cynusuroides"][0] + w_am * veg_params["americanus"][0] + w_alt * \
            veg_params["alterniflora"][0]
        TO = w_cyn * veg_params["cynusuroides"][1] + w_am * veg_params["americanus"][1] + w_alt * \
             veg_params["alterniflora"][1]
        gamma = w_cyn * veg_params["cynusuroides"][2] + w_am * veg_params["americanus"][2] + w_alt * \
                veg_params["alterniflora"][2]
        spec.append([w_cyn, w_am, w_alt])

        oim.gamma = oim.kappa = oim.lamda = gamma
        oim.root_to_shoot = R
        oim.turnover = TO

        oim.biomass()
        oim.organic_carbon_decay()
        oim.marsed(hwls, avg_tide.index, avg_tide.avg_H, k=k, ws=ws, sed_om_frac=sed_om_frac, use_Julia=True,
                   f_Cla=chi_la)
        oim.update_layers()

        C = 100 * oim.get_C() / oim.mass

        C_kg = oim.get_C()

        C_tot.append(np.sum(C_kg))
        M_tot.append(np.sum(oim.Mbg_int))
        mhwl_d_tot.append(mhwl_d)
        AGB_tot.append(oim.Agb)
        surface.append(oim.surface)

    return oim, C, C_tot, M_tot, mhwl_d_tot, AGB_tot, spec, surface