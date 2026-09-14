#=
run_model:
- Julia version: 
- Author: ignace
- Date: 2026-04-28
=#

using GLMakie, Dates, Statistics, DelimitedFiles
using CSV, DataFrames  # For data loading
GLMakie.activate!(inline=false)
set_theme!(theme_dark())

# Include your OIMAS
include("../OIMAS.jl")
include("load_data.jl")

# =============================================================================
# SET PARAMETERS HERE
# =============================================================================

# OIMAS parameters
dt                = 1                 # years
auger_ID         = "S40y1"
Kla             = 0.05
Kre             = 0.0029
init_elev       = 4.5

# MARSED parameters
k           = 0.09
ws          = 1.1e-4
sed_om_frac = 0.049

# Zone from auger_ID (e.g. "S10y1" → 10)
zone = parse(Int, auger_ID[2:3])

# Vegetation parameters: [root_shoot, turnover, gamma]
veg_params = Dict(
    "Tripolium"     => [0.82,  0.5,  0.1],
    "Atriplex"      => [0.9,   1.0,  0.17],
    "Bolboschoenus" => [1.99,  0.8,  0.2],
    "Elytrigia"     => [0.7,   0.8,  0.2],
)

# =============================================================================
# LOAD DATA
# =============================================================================

soil0       = load_soil_carbon(0)
soil        = load_soil_carbon(zone)
rtk         = load_rtk_data()
sar         = load_sar_data()
rtk         = innerjoin(rtk, sar, on = :Point)  # Or whatever your join key is

avg_tide    = load_avg_tide()
hwl_df      = load_hwls()

# Group by the :Point column and calculate the median of :C_percentage
soil0_C     = combine(groupby(soil0, :depth), :C_percentage => median => :C)
soil0_DBD   = combine(groupby(soil0, :depth), :DBD => median => :DBD)
soil0_mass  = combine(groupby(soil0, :depth), :mass_m2 => median => :mass_m2)
soil0_om    = combine(groupby(soil0, :depth), :om_percentage => median => :om_percentage)
soil0_n     = size(soil0_C,1)

# get the specific auger soild dataframe
auger = soil[soil.auger .== auger_ID,:]

# =============================================================================
# INITIALISE OIMAS
# =============================================================================

n_layers                 = 100

s0_om_mass                      = zeros(n_layers)
s0_min_mass                     = zeros(n_layers)
s0_mass                         = zeros(n_layers)

# expand the initial layers to n_layers and extrapolate downwards
s0_depth                        = range(soil0[1,:depth], n_layers * soil0[1,:depth], step = 2*soil0[1,:depth])
@. s0_mass[1:soil0_n]           = soil0_mass.mass_m2
@. s0_om_mass[1:soil0_n]        = (soil0_mass.mass_m2 * soil0_om.om_percentage / 100)
@. s0_min_mass[1:soil0_n]       = (soil0_mass.mass_m2 * (100 - soil0_om.om_percentage) / 100)

# expand the initial layers to n_layers and extrapolate downwards
@. s0_mass[soil0_n+1:end]       = s0_mass[soil0_n]
@. s0_om_mass[soil0_n+1:end]    = s0_om_mass[soil0_n]
@. s0_min_mass[soil0_n+1:end]   = s0_min_mass[soil0_n]

# =============================================================================
# Initialize data
# =============================================================================

# create instance of OIMAS and initialise
oim = oimas_init(n_layers = n_layers, dt = dt,
                f_C = 0.5208,
                Kla = Kla, Kre = Kre,
                chi_la = 0.40, chi_re = 0.60,
                E0_min = 0.861, E0_om = 27.545,
                CI_min = 0.025, CI_om = 1.0,)

initialize_layers!(oim, s0_min_mass, s0_om_mass, initial_surface = init_elev)



# get initial percentage of C
C0 = 100 * oim.C ./ oim.mass

# =============================================================================
# run model
# =============================================================================

for t in 1:10

    # set model parameters
    oim.Bmax   = 1.45 * oim.surface - 5.58

    sedimentation!(oim, .35, 8., f_Cla=.40)
    biomass!(oim)
    decay!(oim)
    update_geometry!(oim)

    println(oim.surface)

end

# get percentage of C
C = 100 * oim.C ./ oim.mass


# =============================================================================
# Plot DATA
# =============================================================================

fig = Figure(figsize=(10,7))
ax1 = Axis(fig[1, 1], xlabel = "C content", ylabel = "depth [m]", limits = (nothing, (-1, 0)))

# plot observations

scatter!(ax1, C0, -1 .* oim.d, color = :orange, label = "modelled C at t = 0 [%]")
lines!(ax1, soil0_C[!, :C], -1 .* soil0_C[!,:depth], color = :orange, label = "observed C [%]")

lines!(ax1, C, -1 .* oim.d, color = :red, label = "modelled C [%]")
scatter!(ax1, auger[!, :C_percentage], -1 .* auger[!,:depth], color = :red, label = "observed C [%]")

xlims!(ax1, 0, 9)

#ax2 = Axis(fig[1, 2], xlabel = "DBD", ylabel = "-depth")
#scatter!(ax2, soil0_DBD[!, :DBD], -1 .* soil0_DBD[!,:depth])
#lines!(ax2, soil0_DBD[!, :DBD], -1 .* soil0_DBD[!,:depth])
axislegend(ax1)
display(fig)

#readline()

