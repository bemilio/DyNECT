import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DyNECT, CommonSolve, Plots, Printf, Polyhedra, LinearAlgebra, Statistics

game_risk_2 = StaticGNEGame(
    N = 2, n = [1, 1],

    Q = [
        [fill(1.2, 1, 1), fill(0.4, 1, 1)],
        [fill(0.4, 1, 1), fill(1.2, 1, 1)]
    ],

    # stronger incentive to take risky RT exposure
    q = [[-1.0], [-1.0]],

    A_loc = [
        [-1.0; 1.0;;],
        [-1.0; 1.0;;]
    ],
    b_loc = [
        [0.0, 1.0],
        [0.0, 1.0]
    ],

    A_sh = [
        fill(1.0, 1, 1),
        fill(1.0, 1, 1)
    ],
    b_sh = [1.0]
)

mpvi_risk_2 = StaticGNE2mpAVI(game_risk_2)

mpvi_risk_2_bounded = DyNECT.setParameterSpace(
    mpvi_risk_2,
    lb = [0.0],
    ub = [1.0]
)

sol_risk_2 = CommonSolve.solve(mpvi_risk_2_bounded, DyNECT.ParametricDAQPSolver)

filter_gne_crs!(sol_risk_2, game_risk_2)

println("Number of valid GNE CRs = ", length(sol_risk_2.CRs))

for (k, cr) in enumerate(sol_risk_2.CRs)
    println("====== CR $k ======")
    println("Ath = ", cr.Ath)
    println("bth = ", cr.bth)
    println("z = ", cr.z)
end

for θval in 0.0:0.1:1.0
    xθ = DyNECT.evaluatePWA(sol_risk_2, [θval])

    if xθ === nothing
        println("θ = $θval -> no valid GNE")
    else
        println("θ = $θval -> x = $xθ, sum = $(sum(xθ))")
    end
end


θ_vals = collect(0.0:0.01:1.0)

x1_vals = Float64[]
x2_vals = Float64[]
agg_vals = Float64[]

for θval in θ_vals
    xθ = DyNECT.evaluatePWA(sol_risk_2, [θval])

    if xθ === nothing
        push!(x1_vals, NaN)
        push!(x2_vals, NaN)
        push!(agg_vals, NaN)
    else
        push!(x1_vals, xθ[1])
        push!(x2_vals, xθ[2])
        push!(agg_vals, sum(xθ))
    end
end

theme(:wong2)   # minimal, clean colors

plot(
    θ_vals, x1_vals,
    label = "Agent 1 equilibrium exposure",
    linewidth = 2,
    color=:royalblue,
    xlabel = "θ: market-risk budget split",
    ylabel = "equilibrium exposure",
    title = "Equilibrium Risk Allocation under Shared Market Capacity",
    titlefont = font(9, :bold),
    guidefont = font(6, :bold),
    tickfont = font(6, :bold),
    legendfont = font(6, :bold),
    grid = false
    
)

plot!(
    θ_vals, x2_vals,
    label = "Agent 2 equilibrium exposure",
    linewidth = 2,
    color=:darkgreen   
)

# shade valid region
vspan!([0.25, 0.75], alpha=0.07, color=:grey)
#vline!([0.25, 0.75], linestyle=:dash, alpha=1.0, color=:grey, label="GNE boundaries")