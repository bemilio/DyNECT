
using StatsPlots
using Serialization
using DataFrames
using LaTeXStrings

# Load the results DataFrame
results_df = deserialize("results_df.jls")


# Helper: collect all benchmarked metric values for each row, tagging with parameter value
function collect_metric_by_param(df, param::Symbol, metric::Symbol)
    vals = []
    params = []
    for row in eachrow(df)
        for t in row[metric]
            push!(vals, t)
            push!(params, row[param])
        end
    end
    return (params, vals)
end


# Plot all metrics for each parameter
params_list = [(:T_hor, raw"$T_{\mathrm{hor}}$"), (:N, raw"$ \#~\mathrm{agents}$"), (:nx, raw"$n_x$"), (:mx, raw"$\#~\mathrm{state}~ \mathrm{constr.}$")]
metric_fields = [
    (symbol=:build_explicit_times, value_name=raw"$\mathrm{Build~time (s)}$", color=:skyblue),
    (symbol=:explicit_evaluation_times, value_name=raw"$\mathrm{Explicit eval time (s)}$", color=:orange),
    (symbol=:implicit_solution_times, value_name=raw"$\mathrm{Implicit solve time (s)}$", color=:green),
    (symbol=:num_CRs, value_name=raw"$\#~\mathrm{Regions}$", color=:red),
    (symbol=:explicit_residual, value_name=raw"$\mathrm{Explicit solution residual}$", color=:teal),
    (symbol=:implicit_residual, value_name=raw"$\mathrm{Implicit solution residual}$", color=:magenta)
]

# 4x2 grid: rows=parameters, columns=benchmark metric
all_plots = []
for (i, (param, param_label)) in enumerate(params_list)
    #### 1. Explicit build time
    params, vals = collect_metric_by_param(results_df, param, metric_fields[1][:symbol])
    # Filter NaNs
    keep = .!isnan.(vals)
    params = params[keep]
    vals = vals[keep]
    push!(all_plots, boxplot(params, vals, xticks=false, xlabel="", ylabel=(i == 1 ? metric_fields[1][:value_name] : ""), yguidefontsize=18, legend=false, color=metric_fields[1][:color], yaxis=:log10))

    ##### 2. Evaluation time: explicit vs implicit side-by-side (clear and robust)
    params_exp, vals_exp = collect_metric_by_param(results_df, param, metric_fields[2][:symbol])
    params_imp, vals_imp = collect_metric_by_param(results_df, param, metric_fields[3][:symbol])

    # Filter NaNs
    keep = .!isnan.(vals_exp)
    params_exp = params_exp[keep]
    vals_exp = vals_exp[keep]
    keep = .!isnan.(vals_imp)
    params_imp = params_imp[keep]
    vals_imp = vals_imp[keep]

    using CategoricalArrays
    all_param_vals = sort(unique(vcat(params_exp, params_imp)))
    eval_times = Vector{Float64}()
    eval_param = Vector{eltype(all_param_vals)}()
    eval_group = String[]
    for p in all_param_vals
        # Explicit
        for v in vals_exp[findall(==(p), params_exp)]
            push!(eval_times, v)
            push!(eval_param, p)
            push!(eval_group, raw"$\mathrm{Explicit (ours)}$")
        end
        # Implicit
        for v in vals_imp[findall(==(p), params_imp)]
            push!(eval_times, v)
            push!(eval_param, p)
            push!(eval_group, raw"$\mathrm{Douglas-Rachford}$")
        end
    end
    eval_param_cat = categorical(string.(eval_param), ordered=true, levels=string.(all_param_vals))
    plot_eval = groupedboxplot(
        eval_param_cat, eval_times,
        group=eval_group,
        xlabel=latexstring(param_label),
        ylabel=(i == 1 ? raw"$\mathrm{Eval~time (s)}$" : ""),
        xguidefontsize=18,
        yguidefontsize=18,
        legendfontsize=12,
        legend=:topright,
        color=[metric_fields[2][:color] metric_fields[3][:color]],
        yaxis=:log10
    )
    push!(all_plots, plot_eval)

    ##### 3. Number of critical regions
    params, vals = collect_metric_by_param(results_df, param, metric_fields[4][:symbol])
    # Filter NaNs
    keep = .!isnan.(vals)
    params = params[keep]
    vals = vals[keep]
    push!(all_plots, boxplot(params, vals, xticks=false, xlabel="", ylabel=(i == 1 ? metric_fields[4][:value_name] : ""), yguidefontsize=18, legend=false, color=metric_fields[4][:color], yaxis=:log10))

    ##### 4. Residual of solution
    params_exp, vals_exp = collect_metric_by_param(results_df, param, metric_fields[5][:symbol])
    params_imp, vals_imp = collect_metric_by_param(results_df, param, metric_fields[6][:symbol])

    # Filter NaNs
    keep = .!isnan.(vals_exp)
    params_exp = params_exp[keep]
    vals_exp = vals_exp[keep]
    keep = .!isnan.(vals_imp)
    params_imp = params_imp[keep]
    vals_imp = vals_imp[keep]

    using CategoricalArrays
    all_param_vals = sort(unique(vcat(params_exp, params_imp)))
    eval_times = Vector{Float64}()
    eval_param = Vector{eltype(all_param_vals)}()
    eval_group = String[]
    for p in all_param_vals
        # Explicit
        for v in vals_exp[findall(==(p), params_exp)]
            push!(eval_times, v)
            push!(eval_param, p)
            push!(eval_group, raw"$\mathrm{Explicit (ours)}$")
        end
        # Implicit
        for v in vals_imp[findall(==(p), params_imp)]
            push!(eval_times, v)
            push!(eval_param, p)
            push!(eval_group, raw"$\mathrm{Douglas-Rachford}$")
        end
    end
    eval_param_cat = categorical(string.(eval_param), ordered=true, levels=string.(all_param_vals))
    plot_eval = groupedboxplot(
        eval_param_cat, eval_times,
        group=eval_group,
        xlabel=latexstring(param_label),
        ylabel=(i == 1 ? raw"$\mathrm{Residual}$" : ""),
        xguidefontsize=18,
        yguidefontsize=18,
        legendfontsize=12,
        legend=:topright,
        color=[metric_fields[5][:color] metric_fields[6][:color]],
        yaxis=:log10
    )
    push!(all_plots, plot_eval)

end

layout44 = @layout [a b c d; e f g h; i j k l; m n o p]
# Rearrangement for 4x4 layout: each row is a parameter, columns are metrics (build, eval, num_CR, residual)
reordered_plots = [
    all_plots[1], all_plots[5], all_plots[9], all_plots[13],
    all_plots[2], all_plots[6], all_plots[10], all_plots[14],
    all_plots[3], all_plots[7], all_plots[11], all_plots[15],
    all_plots[4], all_plots[8], all_plots[12], all_plots[16]
]
combined = plot(reordered_plots..., layout=layout44, size=(2400, 2400), left_margin=10Plots.mm)
display(combined)
savefig(combined, "boxplots_all_metrics.png")