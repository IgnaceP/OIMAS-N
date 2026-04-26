#=
run:
- Julia version: 
- Author: ignace
- Date: 2026-04-26
=#

using CSV
using DataFrames
using Dates
using PyPlot

#matplotlib.style.use("ip02")

include("MARSED.jl")
# or, if MARSED is a module:
# using .MARSED: marsed

# load average tidal cycle
df_fn = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/Getij/Kloosterzande_avg_H.csv"
df = CSV.read(df_fn, DataFrame, header=true)

# If the first column is the index / time column, keep it as a vector
# Adjust the column names below to match your CSV.
avg_tide_t = Float64.(df.minutes_to_nearest_peak)
avg_tide_h = Float64.(df.avg_H)

# distribution of high water levels
hwl_fn = "/Users/ignace/Documents/WETCOAST/Data/Saefthinge/Getij/Kloosterzande_HWLs_1987-2025.csv"
HWLs_df = CSV.read(hwl_fn, DataFrame, header=true)
HWLs = HWLs_df.H_TAW

E = Float64[4.5]
dt = 365 * 2

for t in 1:dt:length(HWLs)
    stopidx = min(t + dt - 1, length(HWLs))
    push!(E, marsed(HWLs[t:stopidx], avg_tide_t, avg_tide_h; E0=E[end]))
end

 years = 0:dt:(dt * (length(E) - 2))

fig, ax = subplots(1, 1)
ax.plot(collect(years), E[2:end], marker="o")
ax.set_xlabel("Tide index")
ax.set_ylabel("Elevation")
ax.set_title("MARSED elevation evolution")

show()