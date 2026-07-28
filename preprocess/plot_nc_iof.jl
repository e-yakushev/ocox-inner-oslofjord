
using Oceananigans
using NCDatasets
using NetCDF
using Printf
using Oceananigans.Units
using Oceananigans.Utils: maybe_int
using CairoMakie
using Statistics

# Put the helper code in a separate file:
include("plot_nc_stuff.jl")
using .plot_nc_stuff
import .plot_nc_stuff: prettydays, replace_zeros_with_NaN!, get_interior,
       plot_ztime, plot_param_transect, plot_bottom_depth_map!,
       compute_transect, compute_pressure_from_depth,
       compute_oxygen_saturation_fields, compute_bottom_index_from_O2,
       vert_transect_slice, make_bottom_depth_and_transect_figure,
       plot_tracer_subplot_map!, bottom_slices_at_day, oxygen_saturation,
       record_horizontal_tracer, record_bottom_tracer,
       get_365_days_indices, plot_six_animations_365_days,
       prepare_six_animations

# ===================== MAIN CODE STARTS HERE =====================
base_dir = dirname(@__DIR__)
folder = joinpath(base_dir, "data", "output")
filename = joinpath(folder, "snapshots_ocean")

# ===================== PLOTTING CONFIGURATION =====================
# Map plotting parameters for each tracer (also used for bottom maps and animations)
const PLOT = Dict(
    :T       => (label="T [°C]",      colorrange=(0, 20),    colormap=Reverse(:RdYlBu),           whiteline=0.0),
    :S       => (label="S [psu]",     colorrange=(15, 35),   colormap=:viridis,                    whiteline=0.0),
    :e       => (label="e [m²/s²]",   colorrange=nothing,    colormap=:viridis,                    whiteline=1.0),
    :O2      => (label="O₂ [μM]",     colorrange=(0, 350),   colormap=:turbo,                      whiteline=67.0),
    :O2sat   => (label="O₂ [%]",      colorrange=(0, 150),   colormap=:gist_stern,                 whiteline=100.0),
    :P       => (label="PHY [μM N]",  colorrange=(0, 5),     colormap=Reverse(:cubehelix),         whiteline=0.0),
    :HET     => (label="HET [μM N]",  colorrange=(0, 5),     colormap=Reverse(:afmhot),            whiteline=0.0),
    :NUT     => (label="NUT [μM N]",  colorrange=(0, 40),    colormap=Reverse(:cherry),            whiteline=0.0),
    :DOM     => (label="DOM [μM N]",  colorrange=(0, 15),    colormap=Reverse(:CMRmap),            whiteline=0.0),
    :POM     => (label="POM [μM N]",  colorrange=(0, 5),     colormap=Reverse(:greenbrownterrain), whiteline=0.0),
    :u       => (label="u [m/s]",     colorrange=(-0.2, 0.2),colormap=Reverse(:seismic),           whiteline=0.0),
    :v       => (label="v [m/s]",     colorrange=(-0.2, 0.2),colormap=Reverse(:seismic),           whiteline=0.0),
    :Ci_free => (label="Ci_free [mg/m³]",  colorrange=(0, 1.5),   colormap=:turbo,                 whiteline=0.0),
    :Ci_DOM  => (label="Ci_DOM [mg/m³]",   colorrange=(0, 0.1),   colormap=:turbo,                 whiteline=0.0),
    :Ci_part => (label="Ci_part [mg/m³]",  colorrange=(0, 0.01),  colormap=:turbo,                 whiteline=0.0),
    :Ci_PHY  => (label="Ci_PHY [mg/m³]",   colorrange=(0, 0.01),  colormap=:turbo,                 whiteline=0.0),
    :Ci_HET  => (label="Ci_HET [mg/m³]",   colorrange=(0, 0.01),  colormap=:turbo,                 whiteline=0.0),
    :Ci_POM  => (label="Ci_POM [mg/m³]",   colorrange=(0, 0.01),  colormap=:turbo,                 whiteline=0.0),
    :Ci_float => (label="Ci_float [mg/m³]",  colorrange=(0, 1.5),   colormap=:turbo,                 whiteline=0.0),
)

# Transect-specific overrides (different labels/colormaps/whitelines from maps)
"""
const TRANSECT = Dict(
    :O2  => (label="O₂ [μM]",   colorrange=(0, 350), colormap=:turbo,   whiteline=20.0), #67 
    :NUT => (label="NUT [μM]",  colorrange=(0, 50),  colormap=:viridis, whiteline=10.0),
    :e   => (label="e [m²/s²]", colorrange=nothing,  colormap=:viridis, whiteline=1.0),
    :S   => (label="S [psu]",   colorrange=(10, 25), colormap=:viridis, whiteline=15.0),
    :P   => (label="P [μM]",    colorrange=(0, 5),   colormap=:viridis, whiteline=5.0),
    :POM => (label="POM [μM]",  colorrange=(0, 3),   colormap=:viridis, whiteline=1.0),
    :DOM => (label="DOM [μM]",  colorrange=(0, 10),   colormap=:viridis, whiteline=3.0),
    :Ci_free => (label="Ci_free [μM]", colorrange=(0, 2),    colormap=:turbo, whiteline=0.01),
    :Ci_DOM  => (label="Ci_DOM [μM]",  colorrange=(0, 0.1), colormap=:turbo, whiteline=0.01),
    :Ci_float => (label="Ci_float [μM]", colorrange=(0, 2),    colormap=:turbo, whiteline=0.01),
    :Ci_PHY  => (label="Ci_PHY [μM]",  colorrange=(0, 0.01), colormap=:turbo, whiteline=0.01),
    :Ci_HET  => (label="Ci_HET [μM]",  colorrange=(0, 0.01), colormap=:turbo, whiteline=0.01),
    :Ci_POM  => (label="Ci_POM [μM]",  colorrange=(0, 0.1), colormap=:turbo, whiteline=0.01),
    :Ci_part => (label="Ci_part [μM]", colorrange=(0, 0.01), colormap=:turbo, whiteline=0.01),
)
"""
ds       = NCDataset("$filename.nc", "r")


###################################################
# flags to control which plots to make:
BGC_             = true     # to include BGC fields and related plots
Ci_              = false     # to include Ci fields and related plots
maps_animations_ = true     # to prepare individual tracer animations 
six_animations_  = false    # to prepare 6-maps-tracer animations 
# general settings for plots:
plot_dates     = [25, 70, 115, 160, 205, 250, 295, 350] # for maps and transects 
#plot_dates     = [40, 100, 160, 220, 280, 340] # for maps and transects 
animation_speed = 10.0  # model days per second of animation
#plot_dates     = [36, 72, 108, 144, 180, 226, 262, 298, 334]

###################################################
# ----- Explore the contents of the NetCDF file ---

println("1D Variables in $filename.nc:")
for (varname, var) in ds
    if ndims(var) == 1
        println("$varname, Dimensions: ", dimnames(var), ", Sizes: ", size(var))
    end
end

println("2D Variables in $filename.nc:")
for (varname, var) in ds
    if ndims(var) == 2
        println("$varname, Dimensions: ", dimnames(var), ", Sizes: ", size(var))
    end
end

println("3D Variables in $filename.nc:")
for (varname, var) in ds
    if ndims(var) == 3
        println("$varname, Dimensions: ", dimnames(var), ", Sizes: ", size(var))
    end
end

println("4D Variables in $filename.nc:")
for (varname, var) in ds
    if ndims(var) == 4
        println("$varname, Dimensions: ", dimnames(var), ", Sizes: ", size(var))
    end
end

println("Groups in file:")
for (name, grp) in ds.group
    println(" - ", name)
end

Nx = size(ds["λ_caa"], 1)
Ny = size(ds["φ_aca"], 1)
Nz = size(ds["z_aac"], 1)
println("Grid dimensions: Nx=$Nx, ", typeof(Nx), ", Ny=$Ny, ", typeof(Ny), ", Nz=$Nz, ", typeof(Nz))

# ------------------ Extract fields from NetCDF ------------------

times = ds["time"][:]
depth = ds["z_aac"][:]
println("Time stats (in seconds) — min: ", minimum(times), ", max: ", maximum(times))

T   = ds["T"][:, :, :, :]
println("T stats — min: ", minimum(T), ", max: ", maximum(T))

S   = ds["S"][:, :, :, :]
println("S stats — min: ", minimum(S), ", max: ", maximum(S))

e = ds["e"][:, :, :, :]
println("e stats — min: ", minimum(e), ", max: ", maximum(e))

u = ds["u"][:, :, :, :]
println("u stats — min: ", minimum(u), ", max: ", maximum(u))

v = ds["v"][:, :, :, :]
println("v stats — min: ", minimum(v), ", max: ", maximum(v))

if BGC_
    P   = ds["P"][:, :, :, :]
    println("P stats — min: ", minimum(P), ", max: ", maximum(P))

    HET = ds["HET"][:, :, :, :]
    println("HET stats — min: ", minimum(HET), ", max: ", maximum(HET))

    NUT = ds["NUT"][:, :, :, :]
    println("NUT stats — min: ", minimum(NUT), ", max: ", maximum(NUT))

    POM = ds["POM"][:, :, :, :]
    println("POM stats — min: ", minimum(POM), ", max: ", maximum(POM))

    DOM = ds["DOM"][:, :, :, :]
    println("DOM stats — min: ", minimum(DOM), ", max: ", maximum(DOM))

    O₂  = ds["O₂"][:, :, :, :]
    println("O₂ stats — min: ", minimum(O₂), ", max: ", maximum(O₂))
end

if Ci_
    Ci_free = ds["Ci_free"][:, :, :, :]
    println("Ci_free stats — min: ", minimum(Ci_free), ", max: ", maximum(Ci_free))

    Ci_DOM  = ds["Ci_DOM"][:, :, :, :]
    println("Ci_DOM stats — min: ", minimum(Ci_DOM), ", max: ", maximum(Ci_DOM))

    Ci_PHY  = ds["Ci_PHY"][:, :, :, :]
    println("Ci_PHY stats — min: ", minimum(Ci_PHY), ", max: ", maximum(Ci_PHY))

    Ci_HET  = ds["Ci_HET"][:, :, :, :]
    println("Ci_HET stats — min: ", minimum(Ci_HET), ", max: ", maximum(Ci_HET))

    Ci_POM  = ds["Ci_POM"][:, :, :, :]
    println("Ci_POM stats — min: ", minimum(Ci_POM), ", max: ", maximum(Ci_POM))
end

@info "all arrays extracted from NetCDF file. Starting to make plots ..."

##########################################
# ------------------ Coordinates for plotting ------------------

oslo_fjord_lon_min = 10.23
oslo_fjord_lon_max = 10.45
oslo_fjord_lat_min = 59.585
oslo_fjord_lat_max = 59.755

# Use actual data dimensions (may differ from grid coordinate variables due to staggering)
data_Nx = size(T, 1)
data_Ny = size(T, 2)
real_lon = range(oslo_fjord_lon_min, oslo_fjord_lon_max, length=data_Nx)
real_lat = range(oslo_fjord_lat_min, oslo_fjord_lat_max, length=data_Ny)

##########################################
# -----------Compute additional variables ------------------
# oxygen saturation and bottom depth from O₂
if BGC_
    Pressure = compute_pressure_from_depth(T, depth, Nz)
    O₂_sat_val, O₂_sat = compute_oxygen_saturation_fields(T, S, Pressure, O₂)
    println("O₂_sat_val stats — min: ", minimum(O₂_sat_val), ", max: ", maximum(O₂_sat_val))
    println("O₂_sat % stats — min: ", minimum(O₂_sat), ", max: ", maximum(O₂_sat))

    bottom_z = compute_bottom_index_from_O2(O₂, Nz)

    # Build 3D water mask from bottom_z: water cells are k >= bottom_z[i,j]
    water_mask = falses(data_Nx, data_Ny, Nz)
    for i in 1:data_Nx, j in 1:data_Ny
        bz = bottom_z[i, j]
        # Only mark as water if the bottom cell actually has non-zero O₂
        if O₂[i, j, bz, 1] != 0
            water_mask[i, j, bz:Nz] .= true
        end
    end
    n_water = sum(water_mask)
    println("Water mask: $n_water water cells out of $(data_Nx * data_Ny * Nz) total")
end

# compute coordinated of the transect line (i, j) points along the 
# deepest bottom, for each column in the horizontal plane
transect = compute_transect(bottom_z, Nz)
#transect = compute_transect(bottom_z)
i_vals   = [t[2] for t in transect]
j_vals   = [t[3] for t in transect]

##########################################
# Bottom depth map
fig_depth_map0 = Figure(size=(1200, 1000))
plot_bottom_depth_map!(fig_depth_map0, (1, 1), bottom_z, depth;
                       title_str="Bottom depth (m)", use_abs=true,
                       colormap=Reverse(:oslo25), whiteline=0.0)
save(joinpath(folder, "Bottom_depth_map.png"), fig_depth_map0)
println("Saved: Bottom_depth_map.png")

##########################################
# -- Bottom map with transect line -------
make_bottom_depth_and_transect_figure(folder, bottom_z, depth, real_lon, real_lat, i_vals, j_vals)

##########################################
# - Vertical-time plots of BGC variables -
plot_ztime(NUT, O₂, O₂_sat, P, HET, T, DOM, POM, S, 16, 22, times, depth, folder)
plot_ztime(NUT, O₂, O₂_sat, P, HET, T, DOM, POM, S, 10, 44, times, depth, folder)


##########################################
# AVERAGED VALUES OVER THE VOLUME
#    days = ds["times"][:] ./ 4.0   # seconds → days
   nt = length(times)
   dt_days = 6 / 24     # 0.25  
   times_days = (0:nt-1) .* dt_days
#    day_index = plot_day * round(Int, length(times) / 365)   

    # Allocate arrays
    Int_S   = zeros(nt)
    Int_T   = zeros(nt)
    Int_e   = zeros(nt)
    if BGC_    
        Int_NUT = zeros(nt)
        Int_P   = zeros(nt)
        Int_HET = zeros(nt)
        Int_POM = zeros(nt)
        Int_DOM = zeros(nt)
        Int_O2  = zeros(nt)
    end
    if Ci_
        Int_Ci_free = zeros(nt)
        Int_Ci_PHY  = zeros(nt)
        Int_Ci_HET = zeros(nt)
        Int_Ci_POM = zeros(nt)
        Int_Ci_DOM = zeros(nt)
    end
    # Helper function: NaN-safe mean
    nanmean(A) = mean(filter(!isnan, vec(A)))
    # Average only over water cells identified by bottom_z
    for t in 1:nt
        Int_S[t]  = mean(ds["S"][:, :, :, t][water_mask])
        Int_T[t]  = mean(ds["T"][:, :, :, t][water_mask])
        Int_e[t]  = mean(ds["e"][:, :, :, t][water_mask])
        if BGC_
            Int_NUT[t] = mean(ds["NUT"][:, :, :, t][water_mask])
            Int_P[t]   = mean(ds["P"][:, :, :, t][water_mask])
            Int_HET[t] = mean(ds["HET"][:, :, :, t][water_mask])
            Int_POM[t] = mean(ds["POM"][:, :, :, t][water_mask])
            Int_DOM[t] = mean(ds["DOM"][:, :, :, t][water_mask])
            Int_O2[t]  = mean(ds["O₂"][:, :, :, t][water_mask])
        end
        if Ci_
            Int_Ci_free[t] = mean(ds["Ci_free"][:, :, :, t][water_mask])
            Int_Ci_PHY[t]  = mean(ds["Ci_PHY"][:, :, :, t][water_mask])
            Int_Ci_HET[t] = mean(ds["Ci_HET"][:, :, :, t][water_mask])
            Int_Ci_POM[t] = mean(ds["Ci_POM"][:, :, :, t][water_mask])
            Int_Ci_DOM[t] = mean(ds["Ci_DOM"][:, :, :, t][water_mask])
        end
    end

# ---------------- Monthly & annual averaged table ----------------
# Assign each timestep to a month (1–12) assuming 365-day year starting Jan 1
month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # days per month
month_cumdays = cumsum(month_days)  # cumulative end-of-month day

function day_to_month(d)
    d_mod = mod(d, 365)  # wrap for multi-year
    for m in 1:12
        if d_mod < month_cumdays[m]
            return m
        end
    end
    return 12
end

# Map each timestep to its month
timestep_month = [day_to_month(times_days[t]) for t in 1:nt]

# Collect all variable names and their arrays
avg_vars = Pair{String, Vector{Float64}}[]
push!(avg_vars, "T" => Int_T)
push!(avg_vars, "S" => Int_S)
push!(avg_vars, "e*1e6" => Int_e .* 1e6)
if BGC_
    push!(avg_vars, "NUT" => Int_NUT)
    push!(avg_vars, "P"   => Int_P)
    push!(avg_vars, "HET" => Int_HET)
    push!(avg_vars, "POM" => Int_POM)
    push!(avg_vars, "DOM" => Int_DOM)
    push!(avg_vars, "O2"  => Int_O2)
end
if Ci_
    push!(avg_vars, "Ci_free" => Int_Ci_free)
    push!(avg_vars, "Ci_PHY"  => Int_Ci_PHY)
    push!(avg_vars, "Ci_HET"  => Int_Ci_HET)
    push!(avg_vars, "Ci_POM"  => Int_Ci_POM)
    push!(avg_vars, "Ci_DOM"  => Int_Ci_DOM)
    push!(avg_vars, "Ci_tot"  => Int_Ci_free .+ Int_Ci_PHY .+ Int_Ci_HET .+ Int_Ci_POM .+ Int_Ci_DOM)
end

# Compute monthly and annual means
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Print header
global header = @sprintf("%-10s", "Variable") * join([@sprintf("%10s", mn) for mn in month_names]) * @sprintf("%10s", "Annual")
println("\n", "="^length(header))
println("Volume-averaged monthly and annual means")
println("="^length(header))
println(header)
println("-"^length(header))

for (varname, vardata) in avg_vars
    row = @sprintf("%-10s", varname)
    for m in 1:12
        idx = findall(timestep_month .== m)
        if isempty(idx)
            row *= @sprintf("%10s", "N/A")
        else
            row *= @sprintf("%10.5f", mean(vardata[idx]))
        end
    end
    row *= @sprintf("%10.5f", mean(vardata))
    println(row)
end
println("="^length(header))

# Also save the table to a text file
table_file = joinpath(folder, "monthly_annual_averages.txt")
open(table_file, "w") do io
    println(io, "Volume-averaged monthly and annual means")
    println(io, "="^length(header))
    println(io, header)
    println(io, "-"^length(header))
    for (varname, vardata) in avg_vars
        row = @sprintf("%-10s", varname)
        for m in 1:12
            idx = findall(timestep_month .== m)
            if isempty(idx)
                row *= @sprintf("%10s", "N/A")
            else
                row *= @sprintf("%10.5f", mean(vardata[idx]))
            end
        end
        row *= @sprintf("%10.5f", mean(vardata))
        println(io, row)
    end
    println(io, "="^length(header))
end
println("Monthly/annual averages table saved to: ", table_file)

# ---------------- Plot averaged values over time ---------------- 
fig = Figure(size = (800, 800))

ax1 = Axis(fig[1, 1], xlabel = "Time (days)", ylabel = "T")    
ax2 = Axis(fig[1, 2], xlabel = "Time (days)", ylabel = "S")
ax3 = Axis(fig[1, 3], xlabel = "Time (days)", ylabel = "e")
lines!(ax1, times_days, Int_T, linewidth = 2, color = :cyan)
lines!(ax2, times_days, Int_S, linewidth = 2, color = :magenta)
lines!(ax3, times_days, Int_e, linewidth = 2, color = :blue)

if BGC_
    ax4 = Axis(fig[2, 1], xlabel = "Time (days)", ylabel = "P")    
    ax5 = Axis(fig[2, 2], xlabel = "Time (days)", ylabel = "HET")
    ax6 = Axis(fig[2, 3], xlabel = "Time (days)", ylabel = "POM")
    ax7 = Axis(fig[3, 1], xlabel = "Time (days)", ylabel = "DOM")
    ax8 = Axis(fig[3, 2], xlabel = "Time (days)", ylabel = "NUT")
    ax9 = Axis(fig[3, 3], xlabel = "Time (days)", ylabel = "N-total")
lines!(ax4, times_days, Int_P,   linewidth = 2, color = :green)
lines!(ax5, times_days, Int_HET, linewidth = 2, color = :orange)
lines!(ax6, times_days, Int_POM, linewidth = 2, color = :blue)
lines!(ax7, times_days, Int_DOM, linewidth = 2, color = :purple)
lines!(ax8, times_days, Int_NUT, linewidth = 2, color = :red)
lines!(ax9, times_days, Int_NUT .+ Int_P .+ Int_HET .+ Int_POM .+ Int_DOM,  linewidth = 2, color = :black)
lines!(ax9, times_days, Int_NUT, linewidth = 2, color = :red)
lines!(ax9, times_days, Int_NUT .+ Int_P,  linewidth = 2, color = :green)
lines!(ax9, times_days, Int_NUT .+ Int_P .+ Int_HET,  linewidth = 2, color = :orange)
lines!(ax9, times_days, Int_NUT .+ Int_P .+ Int_HET .+ Int_POM,  linewidth = 2, color = :blue)
end

    CairoMakie.ylims!(ax1, nothing, nothing)
    CairoMakie.ylims!(ax2, nothing, nothing)
    CairoMakie.ylims!(ax3, 0, nothing)
if BGC_    
    CairoMakie.ylims!(ax4, 0, nothing)
    CairoMakie.ylims!(ax5, 0, nothing)
    CairoMakie.ylims!(ax6, 0, nothing)
    CairoMakie.ylims!(ax7, 0, nothing)
    CairoMakie.ylims!(ax8, 0, nothing)
    CairoMakie.ylims!(ax9, 0, nothing)
end
filename1 = joinpath(folder, "vol_averaged_timeseries.png")
save(filename1, fig)
println("averaged values figure saved to: ", filename1)
# ---
if Ci_
fig11 = Figure(size = (800, 800))

ax11 = Axis(fig11[1, 1], xlabel = "Time (days)", ylabel = "Ci_tot")    
ax12 = Axis(fig11[1, 2], xlabel = "Time (days)", ylabel = "Ci_float")
ax13 = Axis(fig11[1, 3], xlabel = "Time (days)", ylabel = "Ci_part")
lines!(ax11, times_days, Int_Ci_free .+ Int_Ci_PHY .+ Int_Ci_HET .+ Int_Ci_POM .+ Int_Ci_DOM, linewidth = 2, color = :cyan)
lines!(ax12, times_days, Int_Ci_free .+ Int_Ci_DOM, linewidth = 2, color = :magenta)
lines!(ax13, times_days, Int_Ci_PHY .+ Int_Ci_HET .+ Int_Ci_POM, linewidth = 2, color = :blue)

    ax14 = Axis(fig11[2, 1], xlabel = "Time (days)", ylabel = "Ci_PHY")    
    ax15 = Axis(fig11[2, 2], xlabel = "Time (days)", ylabel = "Ci_HET")
    ax16 = Axis(fig11[2, 3], xlabel = "Time (days)", ylabel = "Ci_POM")
    ax17 = Axis(fig11[3, 1], xlabel = "Time (days)", ylabel = "Ci_DOM")
    ax18 = Axis(fig11[3, 2], xlabel = "Time (days)", ylabel = "Ci_free")
    ax19 = Axis(fig11[3, 3], xlabel = "Time (days)", ylabel = "Ci_tot")
lines!(ax14, times_days, Int_Ci_PHY,   linewidth = 2, color = :green)
lines!(ax15, times_days, Int_Ci_HET, linewidth = 2, color = :orange)
lines!(ax16, times_days, Int_Ci_POM, linewidth = 2, color = :blue)
lines!(ax17, times_days, Int_Ci_DOM, linewidth = 2, color = :purple)
lines!(ax18, times_days, Int_Ci_free, linewidth = 2, color = :red)
lines!(ax19, times_days, Int_Ci_free .+ Int_Ci_PHY .+ Int_Ci_HET .+ Int_Ci_POM .+ Int_Ci_DOM,  linewidth = 2, color = :black)
lines!(ax19, times_days, Int_Ci_free, linewidth = 2, color = :red)
lines!(ax19, times_days, Int_Ci_free .+ Int_Ci_PHY,  linewidth = 2, color = :green)
lines!(ax19, times_days, Int_Ci_free .+ Int_Ci_PHY .+ Int_Ci_HET,  linewidth = 2, color = :orange)
lines!(ax19, times_days, Int_Ci_free .+ Int_Ci_PHY .+ Int_Ci_HET .+ Int_Ci_POM,  linewidth = 2, color = :blue)
lines!(ax19, times_days, Int_Ci_free .+ Int_Ci_PHY .+ Int_Ci_HET .+ Int_Ci_POM .+ Int_Ci_DOM,  linewidth = 2, color = :purple)


    CairoMakie.ylims!(ax11, 0, 0.4)
    CairoMakie.ylims!(ax12, 0, nothing)
    CairoMakie.ylims!(ax13, 0, nothing)
    CairoMakie.ylims!(ax14, 0, nothing)

    CairoMakie.ylims!(ax15, 0, nothing)
    CairoMakie.ylims!(ax16, 0, nothing)
    CairoMakie.ylims!(ax17, 0, nothing)
    CairoMakie.ylims!(ax18, 0, nothing)
    CairoMakie.ylims!(ax19, 0, 0.4)

filename11 = joinpath(folder, "vol_averaged_Ci_timeseries.png")
save(filename11, fig11)
println("averaged Ci_ values figure saved to: ", filename11)

end

##########################################
# - Maps and transects at selected dates -

bottom_layer   = Nz
depth_indexes  = [Nz]
fig_width      = 1000
fig_height     = 1150

for plot_day in plot_dates
    day_index = plot_day * round(Int, length(times) / 365)

    for depth_index in depth_indexes
        println("Plotting full map figure for day $plot_day ...")
        ####################
        # vertical TRANSECTS
        O2_slice = vert_transect_slice(O₂, transect, day_index, Nz)
        _ = plot_param_transect(O2_slice, PLOT[:O2].label, depth, transect, folder;
                                colormap=PLOT[:O2].colormap, day_index=plot_day, whiteline=PLOT[:O2].whiteline, colorrange=PLOT[:O2].colorrange)
        NUT_slice = vert_transect_slice(NUT, transect, day_index, Nz)
        _ = plot_param_transect(NUT_slice, PLOT[:NUT].label, depth, transect, folder;
                                colormap=PLOT[:NUT].colormap, day_index=plot_day, whiteline=PLOT[:NUT].whiteline, colorrange=PLOT[:NUT].colorrange)
        e_slice = vert_transect_slice(e, transect, day_index, Nz)
        _ = plot_param_transect(e_slice, PLOT[:e].label, depth, transect, folder;
                                colormap=PLOT[:e].colormap, day_index=plot_day, whiteline=PLOT[:e].whiteline)
        S_slice = vert_transect_slice(S, transect, day_index, Nz)
        _ = plot_param_transect(S_slice, PLOT[:S].label, depth, transect, folder;
                                colormap=PLOT[:S].colormap, day_index=plot_day, whiteline=PLOT[:S].whiteline, colorrange=PLOT[:S].colorrange)
        P_slice = vert_transect_slice(P, transect, day_index, Nz)
        _ = plot_param_transect(P_slice, PLOT[:P].label, depth, transect, folder;
                                colormap=PLOT[:P].colormap, day_index=plot_day, whiteline=PLOT[:P].whiteline, colorrange=PLOT[:P].colorrange)
        POM_slice = vert_transect_slice(POM, transect, day_index, Nz)
        _ = plot_param_transect(POM_slice, PLOT[:POM].label, depth, transect, folder;
                                colormap=PLOT[:POM].colormap, day_index=plot_day, whiteline=PLOT[:POM].whiteline, colorrange=PLOT[:POM].colorrange)
        DOM_slice = vert_transect_slice(DOM, transect, day_index, Nz)
        _ = plot_param_transect(DOM_slice, PLOT[:DOM].label, depth, transect, folder;
                                colormap=PLOT[:DOM].colormap, day_index=plot_day, whiteline=PLOT[:DOM].whiteline, colorrange=PLOT[:DOM].colorrange)
           
    if Ci_
        Ci_free_slice = vert_transect_slice(Ci_free, transect, day_index, Nz)
        _ = plot_param_transect(Ci_free_slice, PLOT[:Ci_free].label, depth, transect, folder;
                                colormap=PLOT[:Ci_free].colormap, day_index=plot_day, whiteline=PLOT[:Ci_free].whiteline, colorrange=PLOT[:Ci_free].colorrange)
        Ci_DOM_slice = vert_transect_slice(Ci_DOM, transect, day_index, Nz)
        _ = plot_param_transect(Ci_DOM_slice, PLOT[:Ci_DOM].label, depth, transect, folder;
                                colormap=PLOT[:Ci_DOM].colormap, day_index=plot_day, whiteline=PLOT[:Ci_DOM].whiteline, colorrange=PLOT[:Ci_DOM].colorrange)
        Ci_diss_slice = Ci_free_slice .+ Ci_DOM_slice
        _ = plot_param_transect(Ci_diss_slice, PLOT[:Ci_float].label, depth, transect, folder;
                                colormap=PLOT[:Ci_float].colormap, day_index=plot_day, whiteline=PLOT[:Ci_float].whiteline, colorrange=PLOT[:Ci_float].colorrange)
        Ci_PHY_slice = vert_transect_slice(Ci_PHY, transect, day_index, Nz)
        _ = plot_param_transect(Ci_PHY_slice, PLOT[:Ci_PHY].label, depth, transect, folder;
                                colormap=PLOT[:Ci_PHY].colormap, day_index=plot_day, whiteline=PLOT[:Ci_PHY].whiteline, colorrange=PLOT[:Ci_PHY].colorrange)
        Ci_HET_slice = vert_transect_slice(Ci_HET, transect, day_index, Nz)
        _ = plot_param_transect(Ci_HET_slice, PLOT[:Ci_HET].label, depth, transect, folder;
                                colormap=PLOT[:Ci_HET].colormap, day_index=plot_day, whiteline=PLOT[:Ci_HET].whiteline, colorrange=PLOT[:Ci_HET].colorrange)
        Ci_POM_slice = vert_transect_slice(Ci_POM, transect, day_index, Nz)
        _ = plot_param_transect(Ci_POM_slice, PLOT[:Ci_POM].label, depth, transect, folder;
                                colormap=PLOT[:Ci_POM].colormap, day_index=plot_day, whiteline=PLOT[:Ci_POM].whiteline, colorrange=PLOT[:Ci_POM].colorrange)
        Ci_part_slice = Ci_PHY_slice .+ Ci_HET_slice .+ Ci_POM_slice
        _ = plot_param_transect(Ci_part_slice, PLOT[:Ci_part].label, depth, transect, folder;
                                colormap=PLOT[:Ci_part].colormap, day_index=plot_day, whiteline=PLOT[:Ci_part].whiteline, colorrange=PLOT[:Ci_part].colorrange)
        end

        ####################
        # horizontal slices for maps
        T_slice      = replace_zeros_with_NaN!(T, depth_index, day_index)
        S_slice      = replace_zeros_with_NaN!(S, depth_index, day_index)
        O₂_slice     = replace_zeros_with_NaN!(O₂, depth_index, day_index)
        NUT_slice    = replace_zeros_with_NaN!(NUT, depth_index, day_index)
        P_slice      = replace_zeros_with_NaN!(P, depth_index, day_index)
        HET_slice    = replace_zeros_with_NaN!(HET, depth_index, day_index)
        DOM_slice    = replace_zeros_with_NaN!(DOM, depth_index, day_index)
        POM_slice    = replace_zeros_with_NaN!(POM, depth_index, day_index)
        O₂_sat_slice = replace_zeros_with_NaN!(O₂_sat, depth_index, day_index)

        if Ci_
            Ci_free_slice = replace_zeros_with_NaN!(Ci_free, depth_index, day_index)
            Ci_DOM_slice  = replace_zeros_with_NaN!(Ci_DOM, depth_index, day_index)
            Ci_PHY_slice  = replace_zeros_with_NaN!(Ci_PHY, depth_index, day_index)
            Ci_POM_slice  = replace_zeros_with_NaN!(Ci_POM, depth_index, day_index)
            Ci_HET_slice  = replace_zeros_with_NaN!(Ci_HET, depth_index, day_index)
        end

        v_slice = replace_zeros_with_NaN!(v, depth_index, day_index)
        u_slice = replace_zeros_with_NaN!(u, depth_index, day_index)

        ####################
        # HORIZONTAL MAPS
        println("Creating maps for day $plot_day at depth index $depth_index ...")
        fig = Figure(size=(fig_width, fig_height))
        plot_tracer_subplot_map!(fig, (1, 1), T_slice,       PLOT[:T].label,      real_lon, real_lat;
                                 colorrange=PLOT[:T].colorrange, colormap=PLOT[:T].colormap, whiteline=PLOT[:T].whiteline)
        plot_tracer_subplot_map!(fig, (1, 3), S_slice,       PLOT[:S].label,      real_lon, real_lat;
                                 colorrange=PLOT[:S].colorrange, colormap=PLOT[:S].colormap, whiteline=PLOT[:S].whiteline)
        plot_tracer_subplot_map!(fig, (1, 5), O₂_slice,      PLOT[:O2].label,     real_lon, real_lat;
                                 colorrange=PLOT[:O2].colorrange, colormap=PLOT[:O2].colormap, whiteline=PLOT[:O2].whiteline)

        plot_tracer_subplot_map!(fig, (2, 1), P_slice,       PLOT[:P].label,      real_lon, real_lat;
                                 colorrange=PLOT[:P].colorrange, colormap=PLOT[:P].colormap, whiteline=PLOT[:P].whiteline)
        plot_tracer_subplot_map!(fig, (2, 3), HET_slice,     PLOT[:HET].label,    real_lon, real_lat;
                                 colorrange=PLOT[:HET].colorrange, colormap=PLOT[:HET].colormap, whiteline=PLOT[:HET].whiteline)
        plot_tracer_subplot_map!(fig, (2, 5), O₂_sat_slice,  PLOT[:O2sat].label,  real_lon, real_lat;
                                 colorrange=PLOT[:O2sat].colorrange, colormap=PLOT[:O2sat].colormap, whiteline=PLOT[:O2sat].whiteline)

        plot_tracer_subplot_map!(fig, (3, 1), DOM_slice,     PLOT[:DOM].label,    real_lon, real_lat;
                                 colorrange=PLOT[:DOM].colorrange, colormap=PLOT[:DOM].colormap, whiteline=PLOT[:DOM].whiteline)
        plot_tracer_subplot_map!(fig, (3, 3), POM_slice,     PLOT[:POM].label,    real_lon, real_lat;
                                 colorrange=PLOT[:POM].colorrange, colormap=PLOT[:POM].colormap, whiteline=PLOT[:POM].whiteline)
        plot_tracer_subplot_map!(fig, (3, 5), NUT_slice,     PLOT[:NUT].label,    real_lon, real_lat;
                                 colorrange=PLOT[:NUT].colorrange, colormap=PLOT[:NUT].colormap, whiteline=PLOT[:NUT].whiteline)

        save(joinpath(folder, "map_iz_$(depth_index)_day_$(plot_day).png"), fig)
        @info "Saved: map_iz_$(depth_index)_day_$(plot_day).png"

        if Ci_
            fig2 = Figure(size=(fig_width, fig_height))

            plot_tracer_subplot_map!(fig2, (1, 1), Ci_free_slice, PLOT[:Ci_free].label, real_lon, real_lat;
                                     colorrange=PLOT[:Ci_free].colorrange, colormap=PLOT[:Ci_free].colormap, whiteline=PLOT[:Ci_free].whiteline)
            plot_tracer_subplot_map!(fig2, (1, 3), Ci_DOM_slice,  PLOT[:Ci_DOM].label,  real_lon, real_lat;
                                     colorrange=PLOT[:Ci_DOM].colorrange, colormap=PLOT[:Ci_DOM].colormap, whiteline=PLOT[:Ci_DOM].whiteline)
            Ci_part_slice = Ci_PHY_slice .+ Ci_HET_slice .+ Ci_POM_slice
            plot_tracer_subplot_map!(fig2, (1, 5), Ci_part_slice, PLOT[:Ci_part].label, real_lon, real_lat;
                                     colorrange=PLOT[:Ci_part].colorrange, colormap=PLOT[:Ci_part].colormap, whiteline=PLOT[:Ci_part].whiteline)
            plot_tracer_subplot_map!(fig2, (2, 1), Ci_PHY_slice,  PLOT[:Ci_PHY].label,  real_lon, real_lat;
                                     colorrange=PLOT[:Ci_PHY].colorrange, colormap=PLOT[:Ci_PHY].colormap, whiteline=PLOT[:Ci_PHY].whiteline)
            plot_tracer_subplot_map!(fig2, (2, 3), Ci_HET_slice,  PLOT[:Ci_HET].label,  real_lon, real_lat;
                                     colorrange=PLOT[:Ci_HET].colorrange, colormap=PLOT[:Ci_HET].colormap, whiteline=PLOT[:Ci_HET].whiteline)
            plot_tracer_subplot_map!(fig2, (2, 5), Ci_POM_slice,  PLOT[:Ci_POM].label,  real_lon, real_lat;
                                     colorrange=PLOT[:Ci_POM].colorrange, colormap=PLOT[:Ci_POM].colormap, whiteline=PLOT[:Ci_POM].whiteline)
            plot_tracer_subplot_map!(fig2, (3, 1), u_slice,       PLOT[:u].label,       real_lon, real_lat;
                                     colorrange=PLOT[:u].colorrange, colormap=PLOT[:u].colormap, whiteline=PLOT[:u].whiteline)
            plot_tracer_subplot_map!(fig2, (3, 3), v_slice,       PLOT[:v].label,       real_lon, real_lat;
                                     colorrange=PLOT[:v].colorrange, colormap=PLOT[:v].colormap, whiteline=PLOT[:v].whiteline)
             Ci_diss_slice = Ci_free_slice .+ Ci_DOM_slice
            plot_tracer_subplot_map!(fig2, (3, 5), Ci_diss_slice, PLOT[:Ci_float].label, real_lon, real_lat;
                                     colorrange=PLOT[:Ci_float].colorrange, colormap=PLOT[:Ci_float].colormap, whiteline=PLOT[:Ci_float].whiteline)
            save(joinpath(folder, "map2_iz_$(depth_index)_day_$(plot_day).png"), fig2)
            @info "Saved: map2_iz_$(depth_index)_day_$(plot_day).png"
        end
    end

    # bottom maps for this day
    fig_b = Figure(size=(fig_width, fig_height))

    T_slice_bot, S_slice_bot, O₂_slice_bot,
    P_slice_bot, HET_slice_bot, O₂_sat_slice_bot,
    DOM_slice_bot, POM_slice_bot, NUT_slice_bot =
        bottom_slices_at_day(T, S, O₂, P, HET, O₂_sat, DOM, POM, NUT, bottom_z, day_index)

    plot_tracer_subplot_map!(fig_b, (1, 1), T_slice_bot, PLOT[:T].label,      real_lon, real_lat;
                             colorrange=PLOT[:T].colorrange, colormap=PLOT[:T].colormap, whiteline=PLOT[:T].whiteline)
    plot_tracer_subplot_map!(fig_b, (1, 3), S_slice_bot, PLOT[:S].label,      real_lon, real_lat;
                             colorrange=PLOT[:S].colorrange, colormap=PLOT[:S].colormap, whiteline=PLOT[:S].whiteline)
    plot_tracer_subplot_map!(fig_b, (1, 5), O₂_slice_bot, PLOT[:O2].label,    real_lon, real_lat;
                             colorrange=PLOT[:O2].colorrange, colormap=PLOT[:O2].colormap, whiteline=PLOT[:O2].whiteline)

    plot_tracer_subplot_map!(fig_b, (2, 1), P_slice_bot, PLOT[:P].label,      real_lon, real_lat;
                             colorrange=PLOT[:P].colorrange, colormap=PLOT[:P].colormap, whiteline=PLOT[:P].whiteline)
    plot_tracer_subplot_map!(fig_b, (2, 3), HET_slice_bot, PLOT[:HET].label,  real_lon, real_lat;
                             colorrange=PLOT[:HET].colorrange, colormap=PLOT[:HET].colormap, whiteline=PLOT[:HET].whiteline)
    plot_tracer_subplot_map!(fig_b, (2, 5), O₂_sat_slice_bot, PLOT[:O2sat].label, real_lon, real_lat;
                             colorrange=PLOT[:O2sat].colorrange, colormap=PLOT[:O2sat].colormap, whiteline=PLOT[:O2sat].whiteline)

    plot_tracer_subplot_map!(fig_b, (3, 1), DOM_slice_bot, PLOT[:DOM].label,   real_lon, real_lat;
                             colorrange=PLOT[:DOM].colorrange, colormap=PLOT[:DOM].colormap, whiteline=PLOT[:DOM].whiteline)
    plot_tracer_subplot_map!(fig_b, (3, 3), POM_slice_bot, PLOT[:POM].label,   real_lon, real_lat;
                             colorrange=PLOT[:POM].colorrange, colormap=PLOT[:POM].colormap, whiteline=PLOT[:POM].whiteline)
    plot_tracer_subplot_map!(fig_b, (3, 5), NUT_slice_bot, PLOT[:NUT].label,   real_lon, real_lat;
                             colorrange=PLOT[:NUT].colorrange, colormap=PLOT[:NUT].colormap, whiteline=PLOT[:NUT].whiteline)

    save(joinpath(folder, "map_bottom_day_$(plot_day).png"), fig_b)
    @info "Saved: map_bottom_day_$(plot_day).png"
end

##########################################
#        record_horizontal_tracer(Ci_free, times, folder, "Ci_free", "Ci_free [μM N]",
#                                real_lon, real_lat;
#                                colorrange=(0, 1), colormap=Reverse(:cherry), iz=Nz)


if maps_animations_ 
# ------------------ Movies ------------------
    record_horizontal_tracer(S,   times, folder, "Ssurf",    PLOT[:S].label,
                            real_lon, real_lat;
                            colorrange=(0, 35), colormap=PLOT[:S].colormap, iz=Nz, speed=animation_speed)
    record_horizontal_tracer(NUT, times, folder, "NUTsurf",  PLOT[:NUT].label,
                            real_lon, real_lat;
                            colorrange=PLOT[:NUT].colorrange, colormap=PLOT[:NUT].colormap, iz=Nz, speed=animation_speed)
    record_horizontal_tracer(DOM, times, folder, "DOMsurf",  PLOT[:DOM].label,
                            real_lon, real_lat;
                            colorrange=(0, 5), colormap=Reverse(:cherry), iz=Nz, speed=animation_speed)
    record_horizontal_tracer(O₂,  times, folder, "O2surf",   PLOT[:O2].label,
                            real_lon, real_lat;
                            colorrange=PLOT[:O2].colorrange, colormap=PLOT[:O2].colormap, iz=Nz, speed=animation_speed)
    record_horizontal_tracer(P,   times, folder, "PHYsurf",  PLOT[:P].label,
                            real_lon, real_lat;
                            colorrange=PLOT[:P].colorrange, colormap=PLOT[:P].colormap, iz=Nz, speed=animation_speed)

    if Ci_
        record_horizontal_tracer(Ci_free .+ Ci_DOM, times, folder, "Ci_float", PLOT[:Ci_float].label,
                                real_lon, real_lat;
                                colorrange=PLOT[:Ci_float].colorrange, colormap=PLOT[:Ci_float].colormap, iz=Nz, speed=animation_speed)
        record_horizontal_tracer(Ci_free, times, folder, "Ci_free", PLOT[:Ci_free].label,
                                real_lon, real_lat;
                                colorrange=PLOT[:Ci_free].colorrange, colormap=PLOT[:Ci_free].colormap, iz=Nz, speed=animation_speed)
        record_horizontal_tracer(Ci_PHY,  times, folder, "Ci_PHY",  PLOT[:Ci_PHY].label,
                                real_lon, real_lat;
                                colorrange=PLOT[:Ci_PHY].colorrange, colormap=PLOT[:Ci_PHY].colormap, iz=Nz, speed=animation_speed)
        record_horizontal_tracer(Ci_HET,  times, folder, "Ci_HET",  PLOT[:Ci_HET].label,
                                real_lon, real_lat;
                                colorrange=PLOT[:Ci_HET].colorrange, colormap=PLOT[:Ci_HET].colormap, iz=Nz, speed=animation_speed)
        record_horizontal_tracer(Ci_POM,  times, folder, "Ci_POM",  PLOT[:Ci_POM].label,
                                real_lon, real_lat;
                                colorrange=PLOT[:Ci_POM].colorrange, colormap=PLOT[:Ci_POM].colormap, iz=Nz, speed=animation_speed)
        record_horizontal_tracer(Ci_DOM,  times, folder, "Ci_DOM",  PLOT[:Ci_DOM].label,
                                real_lon, real_lat;
                                colorrange=PLOT[:Ci_DOM].colorrange, colormap=PLOT[:Ci_DOM].colormap, iz=Nz, speed=animation_speed)
    end

    record_bottom_tracer(O₂, "O2_bottom", Nz, times, folder,
                        real_lon, real_lat;
                        colorrange=(-1, 350), colormap=PLOT[:O2].colormap, figsize=(400, 550), speed=animation_speed)
end
##########################################
#-------Six animation in 1 figure---------
if six_animations_
    prepare_six_animations(tracers=[NUT, O₂, P, HET, POM, DOM],
                           times=times, folder=folder,
                           real_lon=real_lon, real_lat=real_lat,
                           Nz=Nz)
end

close(ds)
@info "Script completed!"
