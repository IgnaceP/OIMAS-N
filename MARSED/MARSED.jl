#=
MARSED:
- Julia version: 
- Author: ignace
- Date: 2026-04-24
=#

using Interpolations

function marsed(HWLs::Vector{Float64}, avg_tide_t::Vector{Float64}, avg_tide_h::Vector{Float64};
                E0=4.5, k=0.606, rho=560.0, ws=1.1e-4, dt=300)
    #=
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
    =#

    # compute number of tides
    tides_n = length(HWLs)

    # compute time steps to calculate within a tidal cycle (21600 seconds = 6 hours)
    tide_times = range(0, 21600, step = dt) .- 21600 / 2
    tide_times_n = length(tide_times)

    # initialize arrays
    E = zeros(Float64, tides_n + 1) # array to keep track of the elevation
    E[1] = E0
    sed = zeros(Float64, tides_n + 1) # array to keep track of the sediment (kg/m^2)

    # loop over tides
    for tide in 1:tides_n

        # get high water level
        hwl = HWLs[tide]

        if hwl > E[tide]

            # interpolate water level
            ht_interp = LinearInterpolation(avg_tide_t, avg_tide_h .- maximum(avg_tide_h) .+ hwl)
            ht = ht_interp(tide_times)

            # initialize/reset C
            C = zeros(Float64, tide_times_n)

            for i in 1:tide_times_n-1

                # compute water level and depth
                h = round(ht[i], digits = 3)
                d = round(h - E[tide], digits = 3)

                if d > 0

                    # compute dh/dt
                    dhdt = (ht[i + 1] - ht[i]) / dt

                    # in case of flood
                    if dhdt > 0
                        # incoming sediment
                        c0 = k * (hwl - E[tide])
                    else
                        c0 = C[i]
                    end

                    # compute sediment concentration
                    dC_dt = (-ws * C[i] + (max(c0,0) - C[i]) * dhdt) / d
                    C[i+1] = max(C[i] + dC_dt * dt,0)

                end
            end

            # compute sedimentation
            flux = ws * C
            sed[tide] = sum(flux[1:end-1] .+ flux[2:end]) * dt / 2

        end

        # update elevation (in case of no inundation, sed[tide] equals 0)
        E[tide + 1] = E[tide] + sed[tide] / rho

    end

    return E[end]
    end