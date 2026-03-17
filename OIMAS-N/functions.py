# functions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = "/Users/ignace/Documents/WETCOAST"
DATA_DIR = f"{BASE_DIR}/Data/Saefthinge"
SOIL_CARBON_DIR = f"{DATA_DIR}/soil_carbon"
BIOMASS_DIR = f"{DATA_DIR}/biomass"


# =============================================================================
# Data loading
# =============================================================================

def load_soil_carbon(years, read_observation_data):
    """
    Load soil carbon datasets for multiple years.
    """
    return {
        y: read_observation_data(f"{SOIL_CARBON_DIR}/S{y}y.csv")
        for y in years
    }


def load_biomass_data():
    fn_biomass = f"{BIOMASS_DIR}/summary.csv"

    return pd.read_csv(fn_biomass, skiprows=0, index_col=0)
def load_bgb_data():
    """
    Load and preprocess below-ground biomass data.
    """
    fn_bgb = f"{BIOMASS_DIR}/BGB2025.csv"
    bgb = pd.read_csv(fn_bgb, skiprows=1, index_col=1)

    footprint = np.pi * 0.05**2
    bgb["mass_m2"] = bgb["mass"] / footprint / 1000
    bgb = bgb[bgb["primary_plant"] != "bare"]

    bgb["depth"] = [
        (float(d.split("-")[-1][:-2]) - 5) / 100
        for d in bgb["Depth"]
    ]

    bgb['auger'] = bgb.index.copy()

    return bgb

def load_elevation_data(years):
    """
    Load elevation data
    """
    elev_data = {
        y: pd.read_csv(f"{DATA_DIR}/LiDAR/ElevationOverTime_tables/Original_Resolution/S{y}y_elevation_rates.csv",
            skiprows=0, index_col=0)
        for y in years
    }

    return elev_data

# =============================================================================
# Plotting
# =============================================================================

def plot_carbon_profiles(axs, soil_data):
    """
    Plot observed carbon profiles for multiple sites.
    """
    for i, site in enumerate(soil_data):
        for _, group in site.groupby("auger"):
            axs[i].scatter(
                group["C_percentage"],
                -0.01 * group["depth"],
                s=4,
                alpha=0.25,
                c="orange"
            )


        median_C = site.groupby("depth")["C_percentage"].median()
        axs[i].plot(
            median_C,
            -0.01 * median_C.index,
            marker="o",
            c="orange",
            alpha=0.75,
            label="observed C [%]"
        )

        axs[i].set_xlabel("C [%]")

    axs[0].set_ylabel("depth [m]")
    axs[0].set_xlim(0, 9)
    axs[0].set_ylim(-0.9, 0)

def plot_dbd_profiles(axs, soil_data):
    """
    Plot observed DBD profiles for multiple sites.
    """
    for i, site in enumerate(soil_data):
        for _, group in site.groupby("auger"):
            axs[i].scatter(
                1000 * group["DBD"],
                -0.01 * group["depth"],
                s=4,
                alpha=0.25,
                c="violet"
            )

        median_dbd = site.groupby("depth")["DBD"].median()
        axs[i].plot(
            1000 * median_dbd,
            -0.01 * median_dbd.index,
            marker="o",
            c="violet",
            alpha=0.75,
            label="DBD"
        )

        axs[i].set_xlabel(r"dry bulk density [$g/cm^3$]")

    axs[0].set_ylabel("depth [m]")
    axs[0].set_xlim(0, 1950)
    axs[0].set_ylim(-0.9, 0)

def plot_belowground_biomass(axs, biomass_data):
    """
    Plot observed belowground biomass profiles for multiple sites.
    """
    for i, site in enumerate(biomass_data):
        for _, group in site.groupby("auger"):
            axs[i].scatter(
                group["mass_m2"],
                -1 * group["depth"],
                s=4,
                alpha=0.25,
                c="green"
            )
            print(group["mass_m2"])

        median_bgb = site.groupby("depth")["mass_m2"].median()
        axs[i].plot(
            median_bgb,
            -1. * median_bgb.index,
            marker="o",
            c="green",
            alpha=0.75,
            label="Belowground Biomass"
        )

        axs[i].set_xlabel("Belowground Biomass [kg/m^2]")
        axs[i].set_ylabel("")

    axs[0].set_xlim(0, 1.25 * max([max(site["mass_m2"]) for site in biomass_data]))

# =============================================================================
# Model setup
# =============================================================================

def compute_initial_masses(S0y, n_layers):
    """
    Compute initial OM and mineral mass profiles from observations.
    """
    median_mass = S0y.groupby("depth")["mass_m2"].median()
    median_om = S0y.groupby("depth")["om_percentage"].median()

    om_mass = (median_mass * median_om / 100).dropna().values
    min_mass = (median_mass * (100 - median_om) / 100).dropna().values

    om_init = np.zeros(n_layers)
    min_init = np.zeros(n_layers)

    om_init[:len(om_mass)] = om_mass
    om_init[len(om_mass):] = om_mass[-1]

    min_init[:len(min_mass)] = min_mass
    min_init[len(min_mass):] = min_mass[-1]

    return om_init, min_init


def create_oimas_model(OIMAS_N):
    """
    Instantiate the OIMAS-N model with fixed parameters.
    """
    return OIMAS_N(
        n_layers=20,
        dt=30,
        sigma_ref_min="top",
        sigma_ref_om="top",
        theta_bg=0,
        Dmbm=0.5,
        Bmax=2.0,
        f_C=0.5208,
        CI_min=0.025,
        CI_om=1.0,
        E0_min=0.861,
        E0_om=27.545,
        Kla0=0.00125,
        Kre0=0.000,
        chi_la=0.40,
        chi_re=0.60,
        max_layer_thickness=0.07,
        gamma=0.15,
        kappa=0.15,
        lamda=0.15,
    )

def pareto_front_2d(f1, f2):
    """
    Return indices of Pareto-optimal points for a 2D minimization problem.
    f1, f2: 1D arrays (same length)
    """
    points = np.vstack([f1, f2]).T
    is_pareto = np.ones(points.shape[0], dtype=bool)

    for i, p in enumerate(points):
        if not is_pareto[i]:
            continue
        # any point that is strictly better in both objectives is dominating
        dominates = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
        is_pareto[dominates] = False
        is_pareto[i] = True

    return np.where(is_pareto)[0]


