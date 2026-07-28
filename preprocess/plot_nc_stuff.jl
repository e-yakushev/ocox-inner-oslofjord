module plot_nc_stuff

using CairoMakie
using Oceananigans
using Oceananigans.Utils: maybe_int

# Number of seconds in a day (for time conversion)
const days = 24 * 60 * 60

export prettydays,
       replace_zeros_with_NaN!,
       get_interior,
       plot_ztime,
       plot_param_transect,
       plot_bottom_depth_map!,
       compute_transect,
       compute_pressure_from_depth,
       compute_oxygen_saturation_fields,
       compute_bottom_index_from_O2,
       vert_transect_slice,
       make_bottom_depth_and_transect_figure,
       plot_tracer_subplot_map!,
       bottom_slices_at_day,
       oxygen_saturation,
       record_horizontal_tracer,
       record_bottom_tracer,
       get_365_days_indices,
       plot_six_animations_365_days,
       prepare_six_animations

# ----------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------

function prettydays(time_seconds)
    days = round(Int, time_seconds / (24 * 3600))
    return "day $days"
end

function replace_zeros_with_NaN!(A, depth_index, day_index)
    slice = Float64.(view(A, :, :, depth_index, day_index))
    @. slice = ifelse(slice == 0, NaN, slice)
    return slice
end

# Helper for interior vs plain arrays
get_interior(A, inds...) =
    hasmethod(interior, Tuple{typeof(A),Vararg{Any}}) ? interior(A, inds...) : A[inds...]

# ----------------------------------------------------------
# Plot: vertical distributions at a point
# ----------------------------------------------------------

function plot_ztime(NUT, O₂, O₂_sat, PHY, HET, T, DOM, POM, S, i, j, times, z, folder)
    fig = Figure(size=(1500, 1000), fontsize=20)

    axis_kwargs = (
        xlabel = "Time (days)",
        ylabel = "z (m)",
        xticks = (0:50:times[end]),
        xtickformat = "{:.0f}",
    )

    axNUT = Axis(fig[1, 1]; title="NUT [μM N]", axis_kwargs...)
    hmNUT = heatmap!(times / days, z, get_interior(NUT, i, j, :, :)',
                     colormap=Reverse(:cherry))
    Colorbar(fig[1, 2], hmNUT)

    axOXY = Axis(fig[1, 3]; title="O₂ [μM]", axis_kwargs...)
    O₂_slice = get_interior(O₂, i, j, :, :)'
    hmOXY = heatmap!(times / days, z, O₂_slice, colormap=:turbo)
    contour!(times / days, z, O₂_slice; levels=[67], color=:white,
             linewidth=4, linestyle=:dot)
    Colorbar(fig[1, 4], hmOXY)

    axOXY_rel = Axis(fig[1, 5]; title="O₂ saturation [%]", axis_kwargs...)
    O₂_sat_slice = get_interior(O₂_sat, i, j, :, :)'
    hmOXY_rel = heatmap!(times / days, z, O₂_sat_slice, colormap=:gist_stern)
    contour!(times / days, z, O₂_sat_slice; levels=[100], color=:white,
             linewidth=4, linestyle=:dot)
    Colorbar(fig[1, 6], hmOXY_rel)

    axPHY = Axis(fig[2, 1]; title="PHY [μM N]", axis_kwargs...)
    hmPHY = heatmap!(times / days, z, get_interior(PHY, i, j, :, :)',
                     colormap=Reverse(:cubehelix))
    Colorbar(fig[2, 2], hmPHY)

    axHET = Axis(fig[2, 3]; title="HET [μM N]", axis_kwargs...)
    hmHET = heatmap!(times / days, z, get_interior(HET, i, j, :, :)',
                     colormap=Reverse(:afmhot))
    Colorbar(fig[2, 4], hmHET)

    axT = Axis(fig[2, 5]; title="T [°C]", axis_kwargs...)
    hmT = heatmap!(times / days, z, get_interior(T, i, j, :, :)',
                   colormap=Reverse(:RdYlBu))
    Colorbar(fig[2, 6], hmT)

    axDOM = Axis(fig[3, 1]; title="DOM [μM N]", axis_kwargs...)
    hmDOM = heatmap!(times / days, z, get_interior(DOM, i, j, :, :)',
                     colormap=Reverse(:CMRmap))
    Colorbar(fig[3, 2], hmDOM)

    axPOM = Axis(fig[3, 3]; title="POM [μM N]", axis_kwargs...)
    hmPOM = heatmap!(times / days, z, get_interior(POM, i, j, :, :)',
                     colormap=Reverse(:greenbrownterrain))
    Colorbar(fig[3, 4], hmPOM)

    axS = Axis(fig[3, 5]; title="S [psu]", axis_kwargs...)
    hmS = heatmap!(times / days, z, get_interior(S, i, j, :, :)',
                   colormap=:viridis)
    Colorbar(fig[3, 6], hmS)

    save(joinpath(folder, "ztime_$(i)_$(j).png"), fig)
    @info "Saved ztime_$(i)_$(j) plot in $folder"
end

# ----------------------------------------------------------
# Transect plots
# ----------------------------------------------------------

function plot_param_transect(Par_transect_slice, title_str, depth, transect, folder;
                             colormap=:turbo, day_index=1, whiteline=1.0, colorrange=nothing)

    depth = vec(depth)
    nmax  = length(transect)

    dist = zeros(Float64, nmax)
    for n in 2:nmax
        (_, i1, j1, _) = transect[n-1]
        (_, i2, j2, _) = transect[n]
        dist[n] = dist[n-1] + hypot(i2 - i1, j2 - j1) * 0.2
    end

    fig = Figure(size=(900, 450), fontsize=24)
    ax  = Axis(fig[1, 1];
               xlabel="Distance along transect (km)",
               ylabel="Depth (m)",
               title="$title_str transect for day: $day_index",
               yreversed=false)

    data_plot = copy(Par_transect_slice)
    nan_mask  = isnan.(data_plot)

    if any(.!nan_mask)
        valid_data = data_plot[.!nan_mask]
        vmin, vmax = extrema(valid_data)
        offset = 0.1 * abs(vmax - vmin)
        data_plot[nan_mask] .= vmin - offset

        cr = isnothing(colorrange) ? (vmin, vmax) : colorrange
        hm = heatmap!(ax, dist, depth, data_plot;
                      colorrange=cr,
                      colormap=colormap,
                      interpolate=false,
                      nan_color=:silver)
    else
        data_plot .= 0.0
        hm = heatmap!(ax, dist, depth, data_plot;
                      colorrange=(0.0, 1.0),
                      colormap=[:silver],
                      interpolate=false)
    end

    if whiteline != 0.0
        contour_data = copy(Par_transect_slice)
        if any(nan_mask)
            contour_data[nan_mask] .= whiteline + 1000.0
        end
        contour!(ax, dist, depth, contour_data;
                 levels=[whiteline], color=:white,
                 linewidth=4, linestyle=:dot)
    end

    if any(.!nan_mask)
        Colorbar(fig[1, 2], hm, label=title_str, width=25)
    end

    title_short = title_str[1:something(findfirst('[', title_str), length(title_str)+1)-1]

    filename = joinpath(folder, "transect_$(title_short)_day_$(day_index).png")
    save(filename, fig, px_per_unit=2)

    @info "Saved plot to $filename"
    println("NaN pixels $(sum(nan_mask)) of $(length(nan_mask))")

    return fig
end

# ----------------------------------------------------------
# Bottom depth map
# ----------------------------------------------------------

function plot_bottom_depth_map!(fig, pos, bottom_z::AbstractMatrix{<:Integer}, z_vals::AbstractVector;
                                title_str="Bottom depth [m]", use_abs=true,
                                colormap=:viridis, whiteline=0.0)
    depth_map = [z_vals[bottom_z[i, j]] for i in 1:size(bottom_z, 1), j in 1:size(bottom_z, 2)]
    if use_abs
        depth_map = abs.(depth_map)
    end

    @. depth_map = ifelse(bottom_z == 0, NaN, depth_map)

    finite_vals = depth_map[isfinite.(depth_map)]
    colorrange = isempty(finite_vals) ? (0.0, 1.0) : (minimum(finite_vals), maximum(finite_vals))

    ax = Axis(fig[pos...]; title=title_str,
              width=200, height=300,
              xlabel="", ylabel="")
    hm = heatmap!(ax, depth_map; colorrange=colorrange,
                  colormap=colormap, nan_color=:silver)
    if whiteline != 0.0
        contour!(ax, depth_map; levels=[whiteline], color=:white,
                 linewidth=4, linestyle=:dot)
    end
    Colorbar(fig[pos[1], pos[2]+1], hm, vertical=true)
end

# ----------------------------------------------------------
# Bottom_z, transect, pressure, oxygen saturation
# ----------------------------------------------------------

function compute_transect(bottom_z, Nz)
    transect = Vector{Tuple{Int,Int,Int,Int}}()
    num   = 1
    lier  = 0
    drammen = 1
    oslo  = 0

    if drammen == 1
        println("Computing transect for Drammen fjord...")
        for j in 1:66
            max_depth_index = Nz
            for i in 1:17
                if max_depth_index < bottom_z[i+1, j]
                    push!(transect, (num, i, j, max_depth_index))
                    num += 1
                    break
                end
                max_depth_index = bottom_z[i+1, j]
            end
        end
    end

    if lier == 1
        println("Computing transect for Lier bay fjord...")
        for j in 1:14
            max_depth_index = Nz
            for i in 1:11
                if max_depth_index < bottom_z[i+1, j]
                    push!(transect, (num, i, j, max_depth_index))
                    num += 1
                    break
                end
                max_depth_index = bottom_z[i+1, j]
            end
        end
    end

    if oslo == 1
        println("Computing transect for Oslo fjord...")
        for j in 1:71
            max_depth_index = 12
            for i in 27:-1:14
                if max_depth_index < bottom_z[i-1, j]
                    push!(transect, (num, i, j, max_depth_index))
                    num += 1
                    break
                end
                max_depth_index = bottom_z[i-1, j]
            end
        end
        for j in 71:-1:34
            max_depth_index = 12
            for i in 27:44
                if max_depth_index < bottom_z[i+1, j]
                    push!(transect, (num, i, j, max_depth_index))
                    num += 1
                    break
                end
                max_depth_index = bottom_z[i+1, j]
            end
        end
    end

    println("✅ Total transect points found: ", length(transect))
    return transect
end

function compute_pressure_from_depth(T, depth, Nz)
    Pressure = similar(T[:, :, :, 1])
    for k in 1:Nz
        Pressure[:, :, k] .= 1. + 0.0992 * (-depth[k])
    end
    return Pressure
end

function oxygen_saturation(T::Float64, S::Float64, P::Float64)::Float64
    T_kelvin = T + 273.15

    ln_O2_sat =
        -173.4292 +
        249.6339 * (100 / T_kelvin) +
        143.3483 * log(T_kelvin / 100) +
        -21.8492 * T_kelvin / 100 +
        -0.033096 * (T_kelvin / 100)^2 +
        0.014259 * (T_kelvin / 100)^3 +
        S * (-0.035274 + 0.001429 * (T_kelvin / 100) +
             -0.00007292 * (T_kelvin / 100)^2) +
        0.0000826 * S^2

    O2_sat = exp(ln_O2_sat) * 44.66
    P_corr = 1.0 + P * (5.6e-6 + 2.0e-11 * P)

    return O2_sat * P_corr
end

function compute_oxygen_saturation_fields(T, S, Pressure, O₂)
    O₂_sat_val = similar(T)
    O₂_sat     = similar(T)
    ϵ          = eps(Float32)

    for i = 1:size(O₂, 1), j = 1:size(O₂, 2),
        k = 1:size(O₂, 3), it = 1:size(O₂, 4)

        O₂_sat_val[i, j, k, it] = oxygen_saturation(
            Float64(T[i, j, k, it]),
            Float64(S[i, j, k, it]),
            Float64(Pressure[i, j, k])
        )
        denom = O₂_sat_val[i, j, k, it] == 0f0 ? ϵ : O₂_sat_val[i, j, k, it]
        O₂_sat[i, j, k, it] = 100f0 * O₂[i, j, k, it] / denom
    end

    return O₂_sat_val, O₂_sat
end

function compute_bottom_index_from_O2(O₂, Nz)
    bottom_z = ones(Int, size(O₂, 1), size(O₂, 2))
    for i = 1:size(O₂, 1)
        for j = 1:size(O₂, 2)
            for k = 1:size(O₂, 3)
                if O₂[i, j, k, 1] != 0
                    bottom_z[i, j] = k
                    break
                end
                if k == Nz
                    bottom_z[i, j] = Nz
                end
            end
        end
    end
    return bottom_z
end

# ----------------------------------------------------------
# Vertical transect slice
# ----------------------------------------------------------

function vert_transect_slice(Param, transect, t, Nz)
    nmax = length(transect)
    Param_slice = Array{Float64}(undef, nmax, Nz)
    for (n, (_, i, j, _)) in enumerate(transect)
        @inbounds Param_slice[n, :] = Param[i, j, 1:Nz, t]
    end
    return Param_slice
end

# ----------------------------------------------------------
# Depth map + transect figure
# ----------------------------------------------------------

function make_bottom_depth_and_transect_figure(folder, bottom_z, depth, real_lon, real_lat, i_vals, j_vals)
    fig_depth_map = Figure(size=(1200, 1000))

    ax_depth = Axis(fig_depth_map[1, 1], title="Bottom depth (m)",
                    xlabel="Longitude (°E)",
                    ylabel="Latitude (°N)")

    lon_vals = real_lon[i_vals]
    lat_vals = real_lat[j_vals]

    depth_map = [depth[bottom_z[i, j]] for i in 1:size(bottom_z, 1), j in 1:size(bottom_z, 2)]
    depth_map = abs.(depth_map)
    @. depth_map = ifelse(bottom_z == 0, NaN, depth_map)

    hm = heatmap!(ax_depth, real_lon, real_lat, depth_map; colormap=Reverse(:oslo25))
    Colorbar(fig_depth_map[1, 2], hm, label="Depth (m)")

    CairoMakie.lines!(ax_depth, lon_vals, lat_vals;
                      color=:white,
                      linewidth=2.5,
                      linestyle=:solid)

    save(joinpath(folder, "Bottom_depth_and_transect_map.png"), fig_depth_map)
    println("Saved: Bottom_depth_and_transect_map.png")
end

# ----------------------------------------------------------
# Map subplot with lon/lat
# ----------------------------------------------------------

function plot_tracer_subplot_map!(fig, pos, data, title_str, longitudes, latitudes;
                                  colorrange=(0, 1), colormap=:viridis, whiteline=1.0)
    ni, nj = size(data)
    # Re-interpolate coordinate ranges when data dimensions differ (staggered u/v grids)
    if length(longitudes) != ni
        longitudes = range(first(longitudes), last(longitudes), length=ni)
    end
    if length(latitudes) != nj
        latitudes = range(first(latitudes), last(latitudes), length=nj)
    end

    ax = Axis(fig[pos...]; title=title_str,
              width=180, height=300,
              xlabel="Longitude (°E)", ylabel="Latitude (°N)")

    hm = heatmap!(ax, longitudes, latitudes, data;
                  colorrange=colorrange, colormap=colormap, nan_color=:silver)

    if whiteline != 0.0
        contour!(ax, longitudes, latitudes, data;
                 levels=[whiteline], color=:white,
                 linewidth=4, linestyle=:dot)
    end
    Colorbar(fig[pos[1], pos[2]+1], hm, vertical=true)
end

# ----------------------------------------------------------
# Bottom slices for all tracers at one day
# ----------------------------------------------------------

function bottom_slices_at_day(T, S, O₂, P, HET, O₂_sat, DOM, POM, NUT, bottom_z, day_index)
    nx, ny, _, _ = size(O₂)

    T_slice_bot      = Array{Float64}(undef, nx, ny)
    S_slice_bot      = Array{Float64}(undef, nx, ny)
    O₂_slice_bot     = Array{Float64}(undef, nx, ny)
    P_slice_bot      = Array{Float64}(undef, nx, ny)
    HET_slice_bot    = Array{Float64}(undef, nx, ny)
    O₂_sat_slice_bot = Array{Float64}(undef, nx, ny)
    DOM_slice_bot    = Array{Float64}(undef, nx, ny)
    POM_slice_bot    = Array{Float64}(undef, nx, ny)
    NUT_slice_bot    = Array{Float64}(undef, nx, ny)

    for i in 1:nx, j in 1:ny
        z = bottom_z[i, j]
        T_slice_bot[i, j]      = Float64(T[i, j, z, day_index])
        S_slice_bot[i, j]      = Float64(S[i, j, z, day_index])
        O₂_slice_bot[i, j]     = Float64(O₂[i, j, z, day_index])
        P_slice_bot[i, j]      = Float64(P[i, j, z, day_index])
        HET_slice_bot[i, j]    = Float64(HET[i, j, z, day_index])
        O₂_sat_slice_bot[i, j] = Float64(O₂_sat[i, j, z, day_index])
        DOM_slice_bot[i, j]    = Float64(DOM[i, j, z, day_index])
        POM_slice_bot[i, j]    = Float64(POM[i, j, z, day_index])
        NUT_slice_bot[i, j]    = Float64(NUT[i, j, z, day_index])
    end

    @. T_slice_bot      = ifelse(T_slice_bot == 0, NaN, T_slice_bot)
    @. S_slice_bot      = ifelse(S_slice_bot == 0, NaN, S_slice_bot)
    @. O₂_slice_bot     = ifelse(O₂_slice_bot == 0, NaN, O₂_slice_bot)
    @. P_slice_bot      = ifelse(P_slice_bot == 0, NaN, P_slice_bot)
    @. HET_slice_bot    = ifelse(HET_slice_bot == 0, NaN, HET_slice_bot)
    @. O₂_sat_slice_bot = ifelse(O₂_sat_slice_bot == 0, NaN, O₂_sat_slice_bot)
    @. DOM_slice_bot    = ifelse(DOM_slice_bot == 0, NaN, DOM_slice_bot)
    @. POM_slice_bot    = ifelse(POM_slice_bot == 0, NaN, POM_slice_bot)
    @. NUT_slice_bot    = ifelse(NUT_slice_bot == 0, NaN, NUT_slice_bot)

    return T_slice_bot, S_slice_bot, O₂_slice_bot,
           P_slice_bot, HET_slice_bot, O₂_sat_slice_bot,
           DOM_slice_bot, POM_slice_bot, NUT_slice_bot
end

# ----------------------------------------------------------
# Movies: horizontal and bottom
# ----------------------------------------------------------

function record_horizontal_tracer(tracer, times, folder, name, label,
                                  longitudes, latitudes;
                                  colorrange=(-1, 30), colormap=:magma,
                                  iz=10, speed=3.0) # speed in model days per second
    dt_days = (times[2] - times[1]) / 86400.0  # snapshot interval in days
    framerate = max(1, round(Int, speed / dt_days))
    Nt = length(times)
    iter = Observable(1)
    Ti = @lift begin
        if tracer isa AbstractArray
            Ti = tracer[:, :, iz, $iter]
        elseif tracer isa FieldTimeSeries
            Ti = interior(tracer[$iter], :, :, iz)
        else
            error("Unsupported tracer type: $(typeof(tracer))")
        end
        Ti[Ti .== 0] .= NaN
        Ti
    end
    title = @lift "$label at $(prettydays(times[$iter]))"

    fig = Figure(size=(400, 550), fontsize=20)
    ax  = Axis(fig[1, 1]; title=title,
               xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    hm  = heatmap!(ax, longitudes, latitudes, Ti,
                   colorrange=colorrange, colormap=colormap,
                   nan_color=:silver)
    Colorbar(fig[0, 1], hm, vertical=false)

    record(fig, joinpath(folder, "movie_$(name)_iz_$iz.gif"), 1:Nt,
           framerate=framerate) do i
        iter[] = i
    end
    @info "movie_$(name)_iz_$iz record made"
end

function record_bottom_tracer(variable, var_name, Nz, times, folder,
                              longitudes, latitudes;
                              colorrange=(-1, 350), colormap=:turbo,
                              figsize=(1000, 400), speed=3.0)
    dt_days = (times[2] - times[1]) / 86400.0  # snapshot interval in days
    framerate = max(1, round(Int, speed / dt_days))
    bottom_z = ones(Int, size(variable, 1), size(variable, 2))
    for i = 1:size(variable, 1)
        for j = 1:size(variable, 2)
            for k = size(variable, 3):-1:1
                if variable[i, j, k, 1] == 0
                    bottom_z[i, j] = k
                    if k != Nz
                        bottom_z[i, j] = k + 1
                    end
                    break
                end
            end
        end
    end

    iter = Observable(1)
    f = @lift begin
        x = [variable[i, j, bottom_z[i, j], $iter]
             for i in 1:size(variable, 1), j in 1:size(variable, 2)]
        x[x .== 0] .= NaN
        x
    end

    title = @lift "bottom $(var_name), μM at " * prettydays(times[$iter])
    fig = Figure(size=figsize)
    ax  = Axis(fig[1, 1]; title=title,
               xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    hm  = heatmap!(ax, longitudes, latitudes, f,
                   colorrange=colorrange, colormap=colormap)
    Colorbar(fig[0, 1], hm, vertical=false, label="$(var_name), μM")

    Nt = length(times)
    record(fig, joinpath(folder, "movie_$(var_name).gif"), 1:Nt,
           framerate=framerate) do i
        iter[] = i
    end
end

# ----------------------------------------------------------
# 365-day helpers (six animations)
# ----------------------------------------------------------

function get_365_days_indices(times_seconds, total_days=365)
    times_days = times_seconds ./ (24 * 3600)
    max_days   = maximum(times_days)

    if max_days < total_days
        @warn "Available data only for $max_days days, but requested $total_days days"
        total_days = Int(floor(max_days))
    end

    target_times = range(0, total_days, length=total_days)
    indices = Int[]
    for target_day in target_times
        idx = argmin(abs.(times_days .- target_day))
        push!(indices, idx)
    end
    unique_indices = unique(indices)

    @info "Selected $(length(unique_indices)) frames for $total_days days"
    @info "Time range: day 0 to day $total_days"
    return unique_indices
end

function plot_six_animations_365_days(tracers, times_seconds, folder, labels, longitudes, latitudes;
                                      colorranges, colormaps, iz, framerate=1,
                                      figsize=(1800, 1200), fontsize=24)
    day_indices = get_365_days_indices(times_seconds, 365)
    Nt          = length(day_indices)

    day_iter      = Observable(1)
    current_frame = Observable(1)

    times_days = times_seconds ./ (24 * 3600)

    fig = Figure(size=figsize, fontsize=fontsize)
    axes_vec     = Any[]
    heatmaps_vec = Any[]

    for i in 1:6
        row = ((i - 1) ÷ 3) + 1
        col = ((i - 1) % 3) + 1

        ax = Axis(fig[row, col*2-1],
                  xlabel=(row == 2 ? "Longitude (°E)" : ""),
                  ylabel=(col == 1 ? "Latitude (°N)" : ""),
                  title=labels[i])

        Ti = @lift begin
            tracer_data = tracers[i]
            frame_idx   = day_indices[$day_iter]
            if ndims(tracer_data) == 4
                Ti_val = tracer_data[:, :, iz, frame_idx]
            else
                error("Expected 4D array, got $(ndims(tracer_data))D")
            end
            Ti_val = Float64.(Ti_val)
            Ti_val[Ti_val .== 0] .= NaN
            Ti_val
        end

        hm = heatmap!(ax, longitudes, latitudes, Ti,
                      colorrange=colorranges[i],
                      colormap=colormaps[i],
                      nan_color=:white)
        Colorbar(fig[row, col*2], hm, width=20, label="")
        push!(axes_vec, ax)
        push!(heatmaps_vec, hm)
    end

    super_title = @lift begin
        day_idx = day_indices[$day_iter]
        current_day = round(Int, times_days[day_idx])
        "Day $current_day - Surface"
    end
    Label(fig[0, :], super_title, fontsize=24, font=:bold)

    rowgap!(fig.layout, 15)
    colgap!(fig.layout, 10)

    output_file = joinpath(folder, "six_animations_365days_iz_$iz.gif")
    @info "Starting 365-day animation recording with $Nt frames..."
    @info "Output file: $output_file"
    @info "Frame rate: $framerate fps"

    record(fig, output_file, 1:Nt, framerate=framerate) do i
        day_iter[]      = i
        current_frame[] = day_indices[i]
        if i % 30 == 0 || i <= 5 || i >= Nt - 5
            day_idx = day_indices[i]
            current_day = round(Int, times_days[day_idx])
            @info "Processing: frame $i/$Nt (day $current_day)"
        end
    end

    @info "365-day animation completed: $output_file"
    return fig
end

function prepare_six_animations(; tracers, times, folder, real_lon, real_lat, Nz)
    labels = ["NUT [μM N]", "O₂ [μM N]", "PHY [μM N]",
              "HET [μM N]", "POM [μM N]", "DOM [μM N]"]

    colorranges = [
        (0.0, 40.0),
        (0.0, 350.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (0.0, 20.0),
    ]

    colormaps = [
        :viridis,
        :turbo,
        :plasma,
        :hot,
        :rainbow,
        :jet,
    ]

    times_days = times ./ (24 * 3600)
    max_days   = maximum(times_days)
    @info "Available data: $(length(times)) time steps, up to day $(round(max_days, digits=1))"

    if max_days >= 365
        @info "Enough data for 365-day animation"
        _ = plot_six_animations_365_days(tracers, times, folder, labels,
                                         real_lon, real_lat;
                                         colorranges=colorranges,
                                         colormaps=colormaps,
                                         iz=Nz, framerate=1)
    else
        @info "Only <365 days available. Implement daily version if needed."
        # You can re-hook your older plot_six_animations_daily here if needed.
    end
end

end # module
