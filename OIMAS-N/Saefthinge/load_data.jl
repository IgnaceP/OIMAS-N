#=
load_data:
- Julia version: 
- Author: ignace
- Date: 2026-04-28
=#

using CSV, DataFrames, Dates

# =============================================================================
# Configuration
# =============================================================================
const BASE_DIR = "/Users/ignace/Documents/WETCOAST/Data"
const DATA_DIR = joinpath(BASE_DIR, "Saefthinge")
const SOIL_CARBON_DIR = joinpath(DATA_DIR, "soil_carbon")
const BIOMASS_DIR = joinpath(DATA_DIR, "biomass")
const RTK_DIR = joinpath(DATA_DIR, "rtk")
const LIDAR_DIR = joinpath(DATA_DIR, "LiDAR")
const TIDES_DIR = joinpath(DATA_DIR, "tides")

# =============================================================================
# Data loading
# =============================================================================

function load_soil_carbon(year::Int64)

    fn = joinpath(SOIL_CARBON_DIR, "S$(year)y.csv")
    df = CSV.read(fn, DataFrame; header=2, missingstring="NA")
    select!(df, Not(r"%Clay",r"%Sand",r"%Silt"))
    dropmissing!(df)
    df[!, :DBD] = parse.(Float64, string.(df[!, :DBD]))
    df[!, :C_percentage] = parse.(Float64, string.(df[!, :C_percentage]))

    # Parse depths: "0-10" → 7.5cm → 0.075m
    df[!, :depth] = parse.(Float64,
        [split(d, "-")[2][1:end-2] for d in string.(df[!, :Depth])] ) / 100 .- 0.025

    df[!, :auger] = string.(df[!, :Point])

    # Mass calculations (6cm auger diameter)
    df[!, :mass_m2] = 0.05 * 1000 * df[!, :DBD]
    df[!, :mass] = 0.05 * π * 0.03^2 * 1000 * df[!, :DBD]

    df[!, :Cmass_m2] = df[!, :mass_m2] .* df[!, :C_percentage] ./ 100
    df[!, :om_percentage] = df[!, :C_percentage] ./ 0.5208
    df[!, :om] = df[!, :mass_m2] .* df[!, :om_percentage] ./ 100
    df[!, :min] = df[!, :mass_m2] .- df[!, :om]

    return df
end

function load_rtk_data()
    fn = joinpath(RTK_DIR, "sample_locations_RTK.csv")
    df = CSV.read(fn, DataFrame)
    df[!, :Point] = string.(df[!, "Point name"])
    return df[!, ["Point","z_TAW"]]
end

function load_sar_data()
    fn = joinpath(LIDAR_DIR, "surface_elevation_accumulation.csv")
    return CSV.read(fn, DataFrame)[!,["Point","SAR"]]
end

function load_avg_tide()
    fn = joinpath(TIDES_DIR, "Kloosterzande_avg_H.csv")
    return CSV.read(fn, DataFrame)
end

function load_hwls()
    fn = joinpath(TIDES_DIR, "Kloosterzande_HWLs_1986-2025.csv")
    return CSV.read(fn, DataFrame; header=2, dateformat="yyyy-mm-dd")
end

function load_biomass_data()
    fn = joinpath(BIOMASS_DIR, "summary.csv")
    return CSV.read(fn, DataFrame)
end

function load_elevation_data(years::Vector{Int})
    res = Dict{Int, DataFrame}()
    for y in years
        fn = joinpath(DATA_DIR, "LiDAR", "ElevationOverTime_tables",
                     "Original_Resolution", "S$(y)y_elevation_rates.csv")
        res[y] = CSV.read(fn, DataFrame)
    end
    return res
end
