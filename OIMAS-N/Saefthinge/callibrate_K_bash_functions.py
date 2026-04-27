from joblib import Parallel, delayed
from datetime import datetime
import numpy as np


def run_single_iteration(sample, oim_template, timesteps, veg_params, hwl_df, avg_tide, sed_om_frac, auger_soil):
    Kla, Kre, k = sample
    # Every worker gets its own fresh copy
    oim_call = oim_template.copy()

    for timestep in range(timesteps + 1):
        # Localize dates
        t0 = datetime(2026 - timesteps + (timestep - 1), 1, 1)
        t1 = datetime(2026 - timesteps + (timestep - 1), 12, 31)

        oim_call.Bmax = 1.45 * oim_call.surface - 5.58
        oim_call.Kla0, oim_call.Kre0 = Kla, Kre

        # Veg params logic
        if oim_call.surface > 5.05:
            rootshoot, turnover, gamma = veg_params["Elytrigia"]
        elif oim_call.surface > 4.8:
            rootshoot, turnover, gamma = veg_params["Bolboschoenus"]
        else:
            rootshoot, turnover, gamma = veg_params["Tripolium"]

        oim_call.gamma = oim_call.kappa = oim_call.lamda = gamma
        oim_call.root_to_shoot, oim_call.turnover = rootshoot, turnover

        oim_call.biomass()
        oim_call.organic_carbon_decay()
        oim_call.marsed(
            hwls=hwl_df.loc[t0:t1].values.flatten(),
            avg_tide_t=avg_tide.index,
            avg_tide_h=avg_tide.avg_H,
            k=k,
            sed_om_frac=sed_om_frac,
            f_Cla=.40,
        )
        oim_call.update_layers()

    # Calculations
    C = 100 * oim_call.get_C() / oim_call.mass
    C_sim = np.interp(auger_soil["depth"] / 100, oim_call.d, C)
    rmse_C = np.sqrt(np.mean(np.square(C_sim - auger_soil["C_percentage"])))

    dbd_sim = np.interp(auger_soil["depth"] / 100, oim_call.d, oim_call.get_dbd())
    rmse_DBD = np.sqrt(np.mean(np.square(dbd_sim - 1000 * auger_soil["DBD"])))

    return rmse_C, rmse_DBD
