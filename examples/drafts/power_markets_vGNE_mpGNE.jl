import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DyNECT, CommonSolve, Plots, LinearAlgebra, Statistics, Printf

# ============================================================
# STYLIZED ENERGY MARKET RISK GAME
# ============================================================

game_risk_2 = StaticGNEGame(
    N = 2, n = [1, 1],

    Q = [
        [fill(1.2, 1, 1), fill(0.4, 1, 1)],
        [fill(0.4, 1, 1), fill(1.2, 1, 1)]
    ],

    q = [[-1.0], [-1.0]],

    # local bounds: 0 ≤ xi ≤ 1
    A_loc = [
        [-1.0; 1.0;;],
        [-1.0; 1.0;;]
    ],

    b_loc = [
        [0.0, 1.0],
        [0.0, 1.0]
    ],

    # shared exposure capacity: x1 + x2 ≤ 1
    A_sh = [
        fill(1.0, 1, 1),
        fill(1.0, 1, 1)
    ],

    b_sh = [1.0]
)

# ============================================================
# APPROACH 1: PARAMETRIC VI / mpGNE MAP
# ============================================================

println("="^70)
println("APPROACH 1: PARAMETRIC VI / FULL GNE MAP")
println("="^70)

mpvi_risk_2 = StaticGNE2mpAVI(game_risk_2)

mpvi_risk_2_bounded = DyNECT.setParameterSpace(
    mpvi_risk_2,
    lb = [0.0],
    ub = [1.0]
)

t_param_start = time()
sol_risk_2 = CommonSolve.solve(mpvi_risk_2_bounded, DyNECT.ParametricDAQPSolver)
time_parametric_raw = time() - t_param_start

filter_gne_crs!(sol_risk_2, game_risk_2)

println("Solved parametric problem in $(round(time_parametric_raw, digits=6)) seconds")
println("Valid filtered GNE critical regions: $(length(sol_risk_2.CRs))")

for (k, cr) in enumerate(sol_risk_2.CRs)
    println("\n====== Valid CR $k ======")
    println("Ath = ", cr.Ath)
    println("bth = ", cr.bth)
    println("z   = ", cr.z)
end

# ============================================================
# APPROACH 2: DIRECT VI FOR FIXED θ VALUES
# ============================================================

println("\n" * "="^70)
println("APPROACH 2: DIRECT FIXED-θ VI")
println("="^70)

θ_test_vals = [0.2, 0.5, 0.8]

for θval in θ_test_vals
    println("\n--- θ = $θval ---")

    avi_theta = AVI(mpvi_risk_2, [θval])

    # Douglas-Rachford
    t_dr_start = time()
    sol_dr = CommonSolve.solve(
        avi_theta,
        DyNECT.DouglasRachford;
        params = IterativeSolverParams(
            max_iter = 1000,
            tol = 1e-6,
            verbose = false
        )
    )
    time_dr = time() - t_dr_start

    println("DouglasRachford:")
    println("  time     = $(round(time_dr, digits=6)) sec")
    println("  status   = $(sol_dr.status)")
    println("  residual = $(sol_dr.residual)")
    println("  x        = $(sol_dr.x)")

    # Optional Monviso
    try
        t_mv_start = time()
        sol_mv = CommonSolve.solve(
            avi_theta,
            DyNECT.MonvisoSolver;
            method = :pg,
            params = IterativeSolverParams(
                max_iter = 1000,
                tol = 1e-6,
                stepsize = 0.01,
                verbose = false
            )
        )
        time_mv = time() - t_mv_start

        println("MonvisoSolver:")
        println("  time     = $(round(time_mv, digits=6)) sec")
        println("  status   = $(sol_mv.status)")
        println("  residual = $(sol_mv.residual)")
        println("  x        = $(sol_mv.x)")
    catch err
        println("MonvisoSolver skipped/failed:")
        println("  ", err)
    end

    # Filtered parametric GNE
    x_parametric = DyNECT.evaluatePWA(sol_risk_2, [θval])

    println("Filtered parametric GNE:")
    println("  x = $x_parametric")

    if x_parametric !== nothing
        err_dr = norm(sol_dr.x - x_parametric)
        println("  error DouglasRachford vs filtered GNE = $err_dr")
    else
        println("  no valid filtered GNE at this θ")
    end
end

# ============================================================
# FULL PARAMETER SWEEP
# ============================================================

println("\n" * "="^70)
println("FULL PARAMETER SWEEP")
println("="^70)

θ_vals = collect(0.0:0.02:1.0)

x_param_vals = Vector{Vector{Float64}}()
x_dr_vals = Vector{Vector{Float64}}()
times_dr = Float64[]
errors_dr = Float64[]
valid_flags = Bool[]

for θval in θ_vals
    # parametric filtered GNE
    x_param = DyNECT.evaluatePWA(sol_risk_2, [θval])

    if x_param === nothing
        push!(x_param_vals, [NaN, NaN])
        push!(valid_flags, false)
    else
        push!(x_param_vals, Vector{Float64}(x_param))
        push!(valid_flags, true)
    end

    # direct fixed-θ VI
    avi_theta = AVI(mpvi_risk_2, [θval])

    t_dr_start = time()
    sol_dr = CommonSolve.solve(
        avi_theta,
        DyNECT.DouglasRachford;
        params = IterativeSolverParams(
            max_iter = 1000,
            tol = 1e-6,
            verbose = false
        )
    )
    push!(times_dr, time() - t_dr_start)
    push!(x_dr_vals, Vector{Float64}(sol_dr.x))

    # error only meaningful when filtered GNE exists
    if x_param === nothing
        push!(errors_dr, NaN)
    else
        push!(errors_dr, norm(sol_dr.x - x_param))
    end
end

valid_idx = findall(valid_flags)

println("Number of θ samples: $(length(θ_vals))")
println("Valid filtered GNE samples: $(length(valid_idx))")
println("Parametric full-space solve time: $(round(time_parametric_raw, digits=6)) sec")
println("Average direct VI time per θ: $(round(mean(times_dr), digits=6)) sec")

if !isempty(valid_idx)
    valid_errors = errors_dr[valid_idx]
    println("Max error on valid GNE samples: $(maximum(valid_errors))")
    println("Mean error on valid GNE samples: $(mean(valid_errors))")
end

# ============================================================
# PLOTTING
# ============================================================

theme(:wong2)

x1_param = [x[1] for x in x_param_vals]
x2_param = [x[2] for x in x_param_vals]
x1_dr = [x[1] for x in x_dr_vals]
x2_dr = [x[2] for x in x_dr_vals]

p1 = plot(
    θ_vals,
    x1_param,
    label = "Parametric filtered GNE",
    linewidth = 3,
    color = :royalblue,
    xlabel = "θ: market-risk budget split",
    ylabel = "Agent 1 exposure",
    title = "Agent 1",
    titlefont = font(11, :bold),
    guidefont = font(10),
    tickfont = font(8),
    legendfont = font(8),
    grid = false,
    legend = :topright
)

scatter!(
    p1,
    θ_vals[valid_idx],
    x1_dr[valid_idx],
    label = "Direct VI at valid θ",
    color = :black,
    markersize = 3,
    markerstrokewidth = 0
)

p2 = plot(
    θ_vals,
    x2_param,
    label = "Parametric filtered GNE",
    linewidth = 3,
    color = :firebrick,
    xlabel = "θ: market-risk budget split",
    ylabel = "Agent 2 exposure",
    title = "Agent 2",
    titlefont = font(11, :bold),
    guidefont = font(10),
    tickfont = font(8),
    legendfont = font(8),
    grid = false,
    legend = :topright
)

scatter!(
    p2,
    θ_vals[valid_idx],
    x2_dr[valid_idx],
    label = "Direct VI at valid θ",
    color = :black,
    markersize = 3,
    markerstrokewidth = 0
)

p3 = plot(
    θ_vals,
    times_dr,
    label = "Direct VI per θ",
    linewidth = 2.5,
    color = :purple,
    xlabel = "θ: market-risk budget split",
    ylabel = "Time [sec]",
    title = "Computation Time",
    titlefont = font(11, :bold),
    guidefont = font(10),
    tickfont = font(8),
    legendfont = font(8),
    grid = false,
    legend = :topright
)

hline!(
    p3,
    [time_parametric_raw / length(θ_vals)],
    label = "Parametric amortized",
    linestyle = :dash,
    color = :gray,
    linewidth = 2
)

p4 = plot(
    θ_vals,
    errors_dr,
    label = "Error on valid θ",
    linewidth = 2.5,
    color = :darkorange,
    xlabel = "θ: market-risk budget split",
    ylabel = "L2 error",
    title = "Direct VI vs Parametric GNE",
    titlefont = font(11, :bold),
    guidefont = font(10),
    tickfont = font(8),
    legendfont = font(8),
    grid = false,
    legend = :topright
)

hline!(
    p4,
    [1e-6],
    label = "Tolerance",
    linestyle = :dash,
    color = :gray,
    linewidth = 1.5
)

fig = plot(
    p1, p2, p3, p4,
    layout = (2, 2),
    size = (1200, 850),
    plot_title = "Parametric GNE Map vs Fixed-θ VI Solver",
    plot_titlefont = font(14, :bold)
)

