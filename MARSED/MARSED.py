import numpy as np

def marsed(HWLs, avg_tide_t, avg_tide_h, E0 = 4.5, k = 0.606, rho = 560, ws = 1.1e-4, dt = 300):
    """
    Method to execute the MARSED model by Temmerman et al. (2003, 2004)
    :param HWLs: (1D numpy array) high water levels, this also determines the amount of tidal events considered [m]
    :param avg_tide_t: (1D numpy array) times (seconds before and after high tide) [s]
    :param avg_tide_h: (1D numpy array) average tidal water levels [m]
    :param E0: (float)initial elevation [m]
    :param k: (float) factor to determine the incoming sediment in function of the incoming high water level C0 = k * (HWL - E0)
    :param rho: (float) bulk density of the sediment [kg/m^3]
    :param ws: (float) sediment settling velocity [m/s]
    :param dt:  (int) time step [s]
    :return: updated elevation
    """

    # compute number of tides
    tides_n = len(HWLs)

    # compute time steps to calculate within a tidal cycle
    tide_times = np.arange(0, 21600, dt) - 21600 / 2

    # initialize arrays
    E   = np.zeros(tides_n + 1); E[0] = E0
    sed  = np.zeros(tides_n + 1)

    # loop over tides
    for tide in range(1, tides_n + 1):
        hwl = HWLs[tide - 1]

        if hwl > E[tide - 1]:
            # interpolate high water level
            ht = np.interp(tide_times, avg_tide_t, avg_tide_h - avg_tide_h.max() + hwl)

            # reset C
            C = np.zeros(len(tide_times))

            # loop over a single tide
            for i in range(1, len(tide_times) - 1):

                # compute water level
                h = np.round(ht[i], 3)

                # only proceed if water level is above E
                if np.round(h - E[tide - 1], 3) > 0:

                    # compute dh/dt
                    dhdt = (ht[i] - ht[i - 1]) / dt

                    # depth

                    # flood
                    if dhdt > 0:
                        c0 = k * (hwl - E[tide - 1])

                    # ebb
                    else:
                        c0 = C[i]

                    water_depth = np.round((h - E[tide - 1]), 3)
                    dC_dt = (-ws * C[i] + (c0 - C[i]) * dhdt) / water_depth

                    # compute C
                    C[i + 1] = max(C[i] + dC_dt * dt, 0)

            # compute E
            sed[tide] = np.trapz(ws * C , dx=dt)

            E[tide] = E[tide - 1] + sed[tide] / rho


        else:
            E[tide] = E[tide - 1]

    return E[-1]


