
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
    (:build_explicit_times, "Explicit build time (s)", :skyblue),
    (:explicit_evaluation_times, "Explicit eval time (s)", :orange),
    (:implicit_solution_times, "Implicit solve time (s)", :green),
    (:diffs_exp_imp, "diff. explicit-implicit solutions", :purple)
]


# 4x2 grid: rows=parameters, columns=benchmark metric
all_plots = []
for (i, (param, param_label)) in enumerate(params_list)
    # 1. Explicit build time
    params, vals = collect_metric_by_param(results_df, param, :build_explicit_times)
    push!(all_plots, boxplot(params, vals, xticks=false, xlabel="", ylabel=(i == 1 ? "Build time (s)" : ""), yguidefontsize=18, legend=false, color=:skyblue, yaxis=:log10))

    # 2. Evaluation time: explicit vs implicit side-by-side (clear and robust)
    params_exp, vals_exp = collect_metric_by_param(results_df, param, :explicit_evaluation_times)
    params_imp, vals_imp = collect_metric_by_param(results_df, param, :implicit_solution_times)
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
            push!(eval_group, "Explicit (ours)")
        end
        # Implicit
        for v in vals_imp[findall(==(p), params_imp)]
            push!(eval_times, v)
            push!(eval_param, p)
            push!(eval_group, "Douglas-Rachford")
        end
    end
    eval_param_cat = categorical(string.(eval_param), ordered=true, levels=string.(all_param_vals))
    plot_eval = groupedboxplot(
        eval_param_cat, eval_times,
        group=eval_group,
        xlabel=latexstring(param_label),
        ylabel=(i == 1 ? "Eval time (s)" : ""),
        xguidefontsize=18,
        yguidefontsize=18,
        legendfontsize=12,
        legend=:topright,
        color=[:orange :green],
        yaxis=:log10
    )
    push!(all_plots, plot_eval)

    # 3. Difference between explicit and implicit
    # params, vals = collect_metric_by_param(results_df, param, :diffs_exp_imp)
    # keep = .!isnan.(vals)
    # params = params[keep]
    # vals = vals[keep]
    # push!(all_plots, boxplot(params, vals, xlabel=param_label, ylabel="|Exp-Imp|", title="|Exp-Imp| vs $(param_label)", legend=false, color=:purple, yaxis=:log10))
end

layout24 = @layout [a b c d; e f g h]
combined = plot(all_plots[[1, 3, 5, 7, 2, 4, 6, 8]]..., layout=layout24, size=(1800, 1600), left_margin=10Plots.mm)
display(combined)
savefig(combined, "boxplots_all_metrics.png")