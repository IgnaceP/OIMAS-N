import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ip02')
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import scipy

########################################################################################################################
#%% 1. parameters ######################################################################################################
########################################################################################################################

# === General similation parameters ===
N_YEARS     = 20
d_dx        = 0.005        # depth resolution for computing below-ground distributions (m)
depths      = np.arange(-1, 0, d_dx)  # depth intervals (m)
n_layers    = len(depths)
output_doy  = [60, 150, 240, 330]

# === Above-ground biomass parameters ===

Bmax        = 2.5         # maximum above-ground biomass at optimal elevation (kg m^-2)
theta_Bmin  = 0           # proportional minimum above-ground biomass at low elevation (fraction of Bp)

ups_Gmax    = 0.0138      # max above-ground growth rate coefficient (kg m^-2 day^-1 per biomass unit)
nu_Gmin     = 0.0         # min above-ground growth coefficient (can be 0)
phase       = 56          # phase shift between biomass accumulation and growth curves (days)

# === Below-ground biomass parameters ===

theta_bg    = -6.8        # linear coefficient of the relation between the roots:shoots ratio and the depth below MHHW
Dmbm        = 4.8         # vertical biomass distribution constant
gamma       = 0.11        # scale depth for below-ground biomass decay (m)
kappa       = gamma       # scale depth for mortality decay profile (m)
lamda       = gamma       # scale depth for growth decay profile (m)

# === Elevation/depth relative to tides ===

D           = .3          # marsh platform depth relative to MHHW (m)
Dmin        = 0           # elevation lower bound (m)
Dmax        = .55         # elevation upper bound (m)

# === Time parameters ===

day_peak    = 244         # peak biomass day (DOY = mid-August)
P           = 365         # annual cycle period (days)

# === Carbon cycling parameters ===

Kl0         = 0.2         # surface decay constant for labile pool (day^-1)
mu_fa       = 1e6         # attenuation length-scale for labile-fast decay (m)
mu_sl       = 1e6         # attenuation length-scale for labile-slow decay (m)
chi_lfa     = 0.32        # fraction of mortality routed to labile-fast carbon pool
chi_lsl     = 0.50        # fraction of mortality routed to labile-slow carbon pool
chi_ref     = 0.16        # fraction of mortality routed to refractory pool


# === Sedimentation rates ===

S           = 0.01       # sedimentation rate (m yr^-1)
Call_ref    = 0.0        # carbon densities in deposited sediments for recalcitrant fraction (kg C m^-3)
Call_l      = 0.0        # carbon densities in deposited sediments for labile fraction (kg C m^-3)

########################################################################################################################
#%% 2. simulation ######################################################################################################
########################################################################################################################

C = {}
Cl          = np.zeros((n_layers, P))
Cref        = np.zeros((n_layers, P))

for year in range(N_YEARS):

    # 2.1 Above-ground vegetation

    Bp          = Bmax * (D - Dmin) / (Dmax - Dmin)     # peak above-ground biomass (kg m^-2)
    day         = np.arange(1,366)                      # days of the year (day)

    Bmin        = theta_Bmin * Bp
    Bag         = .5 * (Bmin + Bp + (Bp - Bmin) *       # above-ground biomass over time (kg m^-2)
                    np.cos(2 * np.pi * (day - day_peak)/P))

    Gmax        = ups_Gmax * Bp                         # maximum growth grate (kg m^-2 day^-1)
    Gmin        = nu_Gmin * Bp                          # minimum growth rate (kg m^-2 day^-1)
    Gag         = .5 * (Gmin + Gmax + (Gmax - Gmin) *   # above-ground growth rate (kg m^-2 day^-1)
                    np.cos(2 * np.pi * (day - day_peak + phase)/P))

    Bag_evol    = np.concatenate([Bag[1:] - Bag[:-1], [Bag[0] - Bag[-1]]]) # above-ground biomass over time (kg m^-2 day^-1)
    Mag         = Bag_evol - Gag                        # above-ground mortality rate (kg m^-2 day^-1)

    # 2.2 Below-ground vegetation

    Bbg         = Bag * (theta_bg * D + Dmbm)           # total below-ground biomass (kg m^-2)
    Bbg_evol    = np.concatenate([Bbg[1:] - Bbg[:-1],   # below-ground biomass over time (kg m^-2 day^-1)
                                  [Bbg[0] - Bbg[-1]]])
    b0          = Bbg / gamma                           # below-ground biomass per unit volume at the surface of the marsh (kg m^-3)
    Mbg = Mag * (theta_bg * D + Dmbm)  # below-ground mortality rate (kg m^-2 day^-1)
    m0 = Mbg / kappa  # below-ground biomass mortality at the surface of the marsh (kg m^-3 day^-1)

    bbg         = np.zeros((n_layers, P))            # below-ground biomass per depth interval (kg m^-3)
    mbg         = np.zeros((n_layers, P))            # below-ground biomass mortality per depth interval (kg m^-3 day^-1)
    gbg         = np.zeros((n_layers, P))            # below-ground biomass growth rate per depth interval (kg m^-3 day^-1)
    for di, d in enumerate(depths):
      bbg[di,:] = b0 * np.exp(d / gamma)
      mbg[di,:] = m0 * np.exp(d / kappa)
      gbg[di,:] = m0 * np.exp(d / lamda)

    # 2.3 Organic carbon decay

    Kl          = Kl0 * np.exp( -d / mu_fa)             # decay coefficient for labile fast pool

    for i in range(1, P):
      Cl_elov   = (-Kl * Cl[:,i-1] +                    # evolution of labile fast carbon pool (kg m^-3)
                   -1 * mbg[:, i] * chi_lfa)
      Cl[:,i]   = Cl[:,i-1] + Cl_elov
      Cl[:,i][Cl[:,i] < 0] = 0

      Cref_elov = -1 * mbg[:, i] * chi_ref              # evolution of recalcitrant carbon pool (kg m^-3)
      Cref[:,i] = Cref[:,i-1] + Cref_elov
      Cref[:,i][Cref[:,i] < 0] = 0

    C[year] = {'Cl': Cl[:,output_doy],
               'Cref': Cref[:,output_doy]}

    # 2.5 update sedimentation procesess
    D            -= S
    new_depths   = np.concatenate((depths - S, [0]))
    Cref_f = scipy.interpolate.interp1d(new_depths, np.concatenate((Cref[:,-1], [Call_ref])), fill_value="extrapolate")
    Cl_f = scipy.interpolate.interp1d(new_depths, np.concatenate((Cl[:,-1], [Call_l])), fill_value="extrapolate")
    Cref[:,0] = Cref_f(depths)
    Cl[:,0] = Cl_f(depths)


########################################################################################################################
#%% 4. Plot growth rates################################################################################################
########################################################################################################################

fig, axs = plt.subplots(nrows = 3)
axs[0].plot(day/30.5, Bag, label = 'above-ground')
axs[0].plot(day/30.5, Bbg, label = 'below-ground')
axs[0].set_ylabel(r'biomass [$kg\;m^{-2}$]')
axs[0].legend()

axs[1].plot(day/30.5, Bag_evol, c = 'C0', label = 'dB/dt')
axs[1].plot(day/30.5, Gag, c = 'C0', ls = ':', label = 'growth')
axs[1].plot(day/30.5, Mag, c = 'C0', ls = '--', label = 'mortality')
axs[1].set_ylabel(r'$\partial B_{ag}\;/\;\partial t$ [$kg\;m^{-2}\;day^{-1}$]')
axs[1].legend()

axs[2].plot(day/30.5, Bbg_evol, c = 'C1', label = 'dB/dt')
axs[2].plot(day/30.5, Bbg_evol- Mbg, c = 'C1', ls = ':', label = 'growth')
axs[2].plot(day/30.5, Mbg, c = 'C1', ls = '--', label = 'mortality')
axs[2].set_ylabel(r'$\partial B_{bg}\;/\;\partial t$ [$kg\;m^{-2}\;day^{-1}$]')
axs[2].legend()

########################################################################################################################
#%% 4. Plot below-ground profiles #######################################################################################
########################################################################################################################
fig, axs = plt.subplots(ncols = 4, figsize = (10,6))

Bag_min = np.argmin(Bag)

for i in [Bag_min + j for j in [0, 90, 180, 270]]:
    lab = (datetime(1, 1, 1) + timedelta(days = int(i))).strftime("%b %d")
    axs[0].plot(bbg[:,i], depths, label = lab)
    axs[1].plot(-1 * mbg[:,i], depths, label = lab)
    #axs[2].plot(Cref[:,i], depths, label = lab)
    axs[2].plot(Cl[:,i], depths, label = lab)
    axs[3].plot(Cref[:,i], depths, label = lab)

axs[0].legend()