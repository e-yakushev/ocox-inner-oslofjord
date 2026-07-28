using NCDatasets
using XLSX
using Plots
using Statistics
using Dates

base_dir = dirname(@__DIR__)
function export_river_data(file_path::String, output_file::String, selected_year::Int,
                           river_index::Int, s_rho_index::Int, param_name::String = "river_N3_n")
    ds = NCDataset(file_path, "r")
    
    param = ds[param_name]  # dimensions: (river, s_rho, river_time)
    time_var = ds["river_time"]
    water_discharge = ds["river_transport"]
   
    river_times_raw = time_var[:]                 # assumed to be DateTime array
    river_values = param[river_index, s_rho_index, :]
    river_discharge = water_discharge[river_index, :]

# to check names of 2D (or 3D) variables in .nc file
#    println("2D Variables in $file_path:")
#    for (varname, var) in ds
#        if ndims(var) == 2
#            println("Variable: $varname")
#            println("  Dimensions: ", dimnames(var))
#            println("  Sizes: ", size(var))
#        end
#    end

    # Pre-allocate filtered data
    selected_days = Int[]
    selected_times = typeof(river_times_raw[1])[]
    selected_values = Float64[]
    selected_discharge = Float64[]

    for (i, t) in enumerate(river_times_raw)
        d = Date(t)  # strip time part for year/day filtering
        if year(d) == selected_year
            push!(selected_days, dayofyear(d))
            push!(selected_times, t)
            push!(selected_values, river_values[i])
            push!(selected_discharge, river_discharge[i])
        end
    end

     # Output diagnostics
    println("📏 Dimensions of 'array': ", dimnames(param))
    println("📏 Sizes of 'array': ", size(param))
    println("📅 Selected year: $selected_year → ", length(selected_values), " entries")
    println("📏 ",param_name," values — min: ", minimum(selected_values), 
        ", mean: ", mean(selected_values), 
        ", max: ", maximum(selected_values))

    close(ds)
#=
    # Write to Excel
    XLSX.openxlsx(output_file * ".xlsx", mode="w") do xf
        sheet = xf[1]
        sheet["A1"] = "DayOfYear"
        sheet["B1"] = "flux,m3/s"
        sheet["C1"] = "con,mmol/m3" 
        sheet["D1"] = "dis,mmol/s" 
        for i in 1:length(selected_values)
            sheet["A$(i+1)"] = selected_days[i]
            sheet["B$(i+1)"] = selected_discharge[i]
            sheet["C$(i+1)"] = selected_values[i]
            sheet["D$(i+1)"] = (selected_values[i] * selected_discharge[i])
        end
    end
=#
    open(output_file * ".csv", "w") do io
        # Write header
        println(io, "DayOfYear;flux_m3/s;c_mmol/m3;dis_mmol/s")
        
        # Write rows
        for i in 1:length(selected_values)
            println(io, "$(selected_days[i]);$(selected_discharge[i]);$(selected_values[i]);$(selected_values[i] * selected_discharge[i])")
        end
    end

    println("📤 Data for river $selected_river for year $selected_year written to: $output_file")
end

# Example usage
selected_year = 2020
selected_river = 18
selected_depth = 1
param_names    =     ["river_N3_n","river_R2_n","river_PON0"]
param_names_oxydep = ["NUT",       "DOM",       "POM"]
param_multiply =     [1.0,          1.0,        1.0]   # multiply parameters
param_smooth =       [true,         true,       true] # whether to smooth parameters 
disch_smooth = true  # whether to smooth water discharge
folder = joinpath(base_dir, "data", "input", "Rivers")
#file = joinpath(folder, "of800_rivers_13_22.nc")
#file = joinpath(folder, "of800_rivers_v9_1990_2022_RA1.nc")
file = joinpath(folder, "of_rivers.nc")

##########################################################################
for selected_river in 1:25
##########################################################################    
    # Plot figures for each selected river (one page per river, 3 subplots for param_names)
    plots_list = []

    # Top subplot: water discharge
    ds = NCDataset(file, "r")
    time_var = ds["river_time"]
    river_times_raw = time_var[:]
    water_discharge = ds["river_transport"]
    river_disch = water_discharge[selected_river, :]
    close(ds)
    disch_days = Int[]
    disch_vals = Float64[]
    for (i, t) in enumerate(river_times_raw)
        d = Date(t)
        if year(d) == selected_year
            push!(disch_days, dayofyear(d))
            push!(disch_vals, river_disch[i])
        end
    end
    p_disch = plot(disch_days, disch_vals, ylabel="m³/s",
                   title="Water discharge", legend=false, linewidth=3,
                   guidefontsize=12, titlefontsize=12, tickfontsize=10,
                   left_margin=15Plots.mm, right_margin=5Plots.mm,
                   bottom_margin=2Plots.mm, top_margin=2Plots.mm)
    push!(plots_list, p_disch)

    for ip in 1:length(param_names)
        ds = NCDataset(file, "r")
        param = ds[param_names[ip]]
        time_var = ds["river_time"]
        river_times_raw = time_var[:]
        river_values = param[selected_river, selected_depth, :]
        close(ds)

        days = Int[]
        vals = Float64[]
        for (i, t) in enumerate(river_times_raw)
            d = Date(t)
            if year(d) == selected_year
                push!(days, dayofyear(d))
                push!(vals, river_values[i])
            end
        end
        p = plot(days, vals, ylabel="mmol/m³",
                 title="$(param_names_oxydep[ip]) ($(param_names[ip]))", legend=false, linewidth=3,
                 guidefontsize=12, titlefontsize=12, tickfontsize=10,
                 left_margin=15Plots.mm, right_margin=5Plots.mm,
                 bottom_margin=2Plots.mm, top_margin=2Plots.mm,
                 ylims=(0, Inf))
        push!(plots_list, p)
    end
    fig = plot(plots_list..., layout=(4, 1), size=(800, 1000),
              plot_title="River $selected_river — Year $selected_year",
              plot_titlefontsize=12)
    savefig(fig, joinpath(folder, "river_$(selected_river)_year_$(selected_year)_ini.png"))
    println("📊 Plot saved for river $selected_river")

    for ip in 1:length(param_names)
        output = joinpath(folder, "$(param_names[ip])_ini_from_$(selected_river)_year_$(selected_year)")
    #    output = joinpath("$(param_names[ip])_ini_from_$(selected_river)_year_$(selected_year)")
        export_river_data(file, output, selected_year, selected_river, selected_depth, param_names[ip])
    end

end
#----------------------------------------------------------
# Now repeat with multiplied parameters + more constant contaminant

river_S = 0.3 #concentrtaion of added contaminant, i.e. MP=0.785164118 mg/m3
river_Ci_ = 0.785164118 #concentrtaion of added contaminant, i.e. MP=0.785164118 mg/m3

# Repeat with multiplied parameters + 4th constant contaminant
param_names_mod = [param_names..., "river_S", "river_Ci"]
param_multiply_mod = [param_multiply..., 1.0, 1.0]

for selected_river in 1:25
    plots_list = []
    all_days = Int[]  # to reuse for constant Ci_ plot

    # Top subplot: water discharge
    ds = NCDataset(file, "r")
    time_var = ds["river_time"]
    river_times_raw = time_var[:]
    water_discharge = ds["river_transport"]
    river_disch_all = water_discharge[selected_river, :]
    close(ds)
    disch_days = Int[]
    disch_vals = Float64[]
    for (i, t) in enumerate(river_times_raw)
        d = Date(t)
        if year(d) == selected_year
            push!(disch_days, dayofyear(d))
            push!(disch_vals, river_disch_all[i])
        end
    end
    if disch_smooth
        n = length(disch_vals)
        smoothed = similar(disch_vals)
        for j in 1:n
            lo = max(1, j - 15)
            hi = min(n, j + 15)
            smoothed[j] = mean(disch_vals[lo:hi])
        end
        disch_vals = smoothed
    end
    all_days = copy(disch_days)
    p_disch = plot(disch_days, disch_vals, ylabel="m³/s",
                   title="Water discharge", legend=false, linewidth=3,
                   guidefontsize=12, titlefontsize=12, tickfontsize=10,
                   left_margin=15Plots.mm, right_margin=5Plots.mm,
                   bottom_margin=2Plots.mm, top_margin=2Plots.mm)
    push!(plots_list, p_disch)

    for ip in 1:length(param_names)
        ds = NCDataset(file, "r")
        param = ds[param_names[ip]]
        time_var = ds["river_time"]
        river_times_raw = time_var[:]
        river_values = param[selected_river, selected_depth, :]
        close(ds)

        days = Int[]
        vals = Float64[]
        for (i, t) in enumerate(river_times_raw)
            d = Date(t)
            if year(d) == selected_year
                push!(days, dayofyear(d))
                push!(vals, river_values[i] * param_multiply[ip])
            end
        end
        if param_smooth[ip]
            n = length(vals)
            smoothed = similar(vals)
            for j in 1:n
                lo = max(1, j - 15)
                hi = min(n, j + 15)
                smoothed[j] = mean(vals[lo:hi])
            end
            vals = smoothed
        end
        if isempty(all_days)
            all_days = copy(days)
        end
        p = plot(days, vals, ylabel="mmol/m³",
                 title="$(param_names_oxydep[ip]) ($(param_names[ip]) × $(param_multiply[ip]))", legend=false, linewidth=3,
                 guidefontsize=12, titlefontsize=12, tickfontsize=10,
                 left_margin=15Plots.mm, right_margin=5Plots.mm,
                 bottom_margin=2Plots.mm, top_margin=2Plots.mm,
                 ylims=(0, Inf))
        push!(plots_list, p)
    end
    # 4th plot: constant river_S
    p_s = plot(all_days, fill(river_S, length(all_days)), ylabel="PSU",
               title="S (salinity)", legend=false, linewidth=3,
               guidefontsize=12, titlefontsize=12, tickfontsize=10,
               left_margin=15Plots.mm, right_margin=5Plots.mm,
               bottom_margin=2Plots.mm, top_margin=2Plots.mm,
               ylims=(0, Inf))
    push!(plots_list, p_s)

    # 5th plot: constant river_Ci_
    p_ci = plot(all_days, fill(river_Ci_, length(all_days)), ylabel="mg/m³",
                title="Ci (contaminant)", legend=false, linewidth=3,
                guidefontsize=12, titlefontsize=12, tickfontsize=10,
                left_margin=15Plots.mm, right_margin=5Plots.mm,
                bottom_margin=2Plots.mm, top_margin=2Plots.mm,
                ylims=(0, Inf))
    push!(plots_list, p_ci)

    fig = plot(plots_list..., layout=(6, 1), size=(800, 1000),
              plot_title="River $selected_river — Year $selected_year (modified)",
              plot_titlefontsize=12)
    savefig(fig, joinpath(folder, "river_$(selected_river)_year_$(selected_year).png"))
    println("📊 Modified plot saved for river $selected_river")

    for ip in 1:length(param_names)
        output = joinpath(folder, "$(param_names[ip])_from_$(selected_river)_year_$(selected_year)")
        export_river_data(file, output, selected_year, selected_river, selected_depth, param_names[ip])
    end
    # Output constant Ci_ as CSV (with discharge column like other params)
    ds = NCDataset(file, "r")
    water_discharge = ds["river_transport"]
    river_times_raw = ds["river_time"][:]
    river_disch = water_discharge[selected_river, :]
    close(ds)
    open(joinpath(folder, "river_Ci_from_$(selected_river)_year_$(selected_year).csv"), "w") do io
        println(io, "DayOfYear;flux_m3/s;c_mg/m3;dis_mg/s")
        idx = 1
        for (i, t) in enumerate(river_times_raw)
            d = Date(t)
            if year(d) == selected_year
                println(io, "$(dayofyear(d));$(river_disch[i]);$river_Ci_;$(river_Ci_ * river_disch[i])")
            end
        end
    end

    # Output constant river_S as CSV
    open(joinpath(folder, "river_S_from_$(selected_river)_year_$(selected_year).csv"), "w") do io
        println(io, "DayOfYear;flux_m3/s;S_PSU;dis_PSU*m3/s")
        for (i, t) in enumerate(river_times_raw)
            d = Date(t)
            if year(d) == selected_year
                println(io, "$(dayofyear(d));$(river_disch[i]);$river_S;$(river_S * river_disch[i])")
            end
        end
    end

end


