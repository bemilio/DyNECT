import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DyNECT, CommonSolve, Plots, Printf, Polyhedra, LinearAlgebra, Statistics

println("=== GAME 1: ABUNDANT WATER (W=20) ===")
game1 = StaticGNEGame(
    N = 3, n = [1, 1, 1],
    Q = [
        [fill(1.0, 1, 1),   fill(0.20, 1, 1), fill(0.10, 1, 1)],
        [fill(0.15, 1, 1),  fill(1.5, 1, 1),  fill(0.12, 1, 1)],
        [fill(0.18, 1, 1),  fill(0.22, 1, 1), fill(1.2, 1, 1)]
    ],
    q = [[-6.0], [-7.5], [-5.0]],
    A_loc = [
        [-1.0; 1.0;;],
        [-1.0; 1.0;;],
        [-1.0; 1.0;;]
    ],
    b_loc = [
        [0.0, 8.0],
        [0.0, 8.0],
        [0.0, 8.0]
    ],
    A_sh = [
        fill(1.0, 1, 1),
        fill(1.0, 1, 1),
        fill(1.0, 1, 1)
    ],
    b_sh = [20.0] #testing
)

mpvi1 = StaticGNE2mpAVI(game1)
#mpvi_bounded = DyNECT.setParameterSpace(mpvi1, lb=[-10.0, -10.0], ub=[0.0, 0.0])
mpvi1_bounded = DyNECT.setParameterSpace(mpvi1, 
    lb = [0.0, 0.0],           # θ₁, θ₂ ≥ 0
    ub = [20.0, 20.0],         # θ₁, θ₂ ≤ 20 max
    C = [1.0 1.0],             # θ₁ + θ₂ ≤ 20 (simplex constraint)
    d = [20.0]
)
sol1 = CommonSolve.solve(mpvi1_bounded, DyNECT.ParametricDAQPSolver)
filter_gne_crs!(sol1, game1)

x1 = DyNECT.evaluatePWA(sol1, [5.0; 5.0])
println("At θ=[5,5]: x = $x1, sum = $(sum(x1))\n")

println("=== GAME 2: SCARCE WATER (W=8) ===")
game2 = StaticGNEGame(
    N = 3, n = [1, 1, 1],
    Q = [
        [fill(1.0, 1, 1),   fill(0.20, 1, 1), fill(0.10, 1, 1)],
        [fill(0.15, 1, 1),  fill(1.5, 1, 1),  fill(0.12, 1, 1)],
        [fill(0.18, 1, 1),  fill(0.22, 1, 1), fill(1.2, 1, 1)]
    ],
    q = [[-6.0], [-7.5], [-5.0]],
    A_loc = [
        [-1.0; 1.0;;],
        [-1.0; 1.0;;],
        [-1.0; 1.0;;]
    ],
    b_loc = [
        [0.0, 8.0],
        [0.0, 8.0],
        [0.0, 8.0]
    ],
    A_sh = [
        fill(1.0, 1, 1),
        fill(1.0, 1, 1),
        fill(1.0, 1, 1)
    ],
    b_sh = [8.0] #testing
)

mpvi2 = StaticGNE2mpAVI(game2)
#mpvi_bounded = DyNECT.setParameterSpace(mpvi2, lb=[-10.0, -10.0], ub=[0.0, 0.0])
mpvi2_bounded = DyNECT.setParameterSpace(mpvi2, 
    lb = [0.0, 0.0],           # θ₁, θ₂ ≥ 0
    ub = [8.0, 8.0],           # θ₁, θ₂ ≤ 8 max
    C = [1.0 1.0],             # θ₁ + θ₂ ≤ 8 (simplex constraint)
    d = [8.0]
)
sol2 = CommonSolve.solve(mpvi2_bounded, DyNECT.ParametricDAQPSolver)
filter_gne_crs!(sol2, game2)

x2 = DyNECT.evaluatePWA(sol2, [4.0; 4.0])
println("At θ=[4,4]: x = $x2, sum = $(sum(x2))\n")


println("=== COMPARISON ===")
println("Abundant: $x1")
println("Scarce:   $x2")

##using DyNECT, CommonSolve, Plots
function plot_solution_trajectory(sol, θlb, θub, title_str)
    x_values     = []
    theta_values = []
    
    # Handle 1D parameter space
    if length(θlb) == 1
        for θ in θlb[1]:0.02:θub[1]
            x = DyNECT.evaluatePWA(sol, [θ])
            if x !== nothing
                push!(x_values, x)
                push!(theta_values, θ)
            end
        end
    
    # Handle 2D parameter space - sample along diagonal
    elseif length(θlb) == 2
        for t in 0:0.02:1
            θ = [θlb[1] + t*(θub[1]-θlb[1]), 
                 θlb[2] + t*(θub[2]-θlb[2])]
            x = DyNECT.evaluatePWA(sol, θ)
            if x !== nothing
                push!(x_values, x)
                push!(theta_values, LinearAlgebra.norm(θ))  # Use norm as scalar colorbar
            end
        end
    end
    
    if isempty(x_values)
        println("⚠️  No valid solutions found for $title_str")
        return nothing
    end
    
    println("✓ Found $(length(x_values)) valid solutions")
    
    x_array = hcat(x_values...)'
    
    # For 2-agent: x1 vs x2
    if size(x_array, 2) == 2
        fig = scatter(x_array[:, 1], x_array[:, 2], marker_z=theta_values,
            xlabel="x₁", ylabel="x₂", label="", 
            colorbar_title="||θ||", title=title_str, 
            right_margin=5Plots.mm, markersize=4)
    # For 3-agent: multiple 2D projections
    elseif size(x_array, 2) == 3
        fig = @layout [a b; c d]
        p1 = scatter(x_array[:, 1], x_array[:, 2], marker_z=theta_values,
            xlabel="x₁", ylabel="x₂", label="", legend=false, markersize=3)
        p2 = scatter(x_array[:, 1], x_array[:, 3], marker_z=theta_values,
            xlabel="x₁", ylabel="x₃", label="", legend=false, markersize=3)
        p3 = scatter(x_array[:, 2], x_array[:, 3], marker_z=theta_values,
            xlabel="x₂", ylabel="x₃", label="", colorbar_title="||θ||", markersize=3)
        p4 = plot(theta_values, x_array[:, 1], label="x₁", legend=:topleft)
        plot!(theta_values, x_array[:, 2], label="x₂")
        plot!(theta_values, x_array[:, 3], label="x₃", xlabel="||θ||", ylabel="Extraction")
        
        fig = plot(p1, p2, p3, p4, layout=fig, title=title_str)
    end
    
    return fig
end

# 2-AGENT case
println("\n=== 2-AGENT WATER GAME ===")
game_2 = StaticGNEGame(
    N = 2, n = [1, 1],
    Q = [
        [fill(1.0, 1, 1), fill(0.20, 1, 1)],
        [fill(0.15, 1, 1), fill(1.5, 1, 1)]
    ],
    q = [[-6.0], [-7.5]],
    A_loc = [[-1.0; 1.0;;], [-1.0; 1.0;;]],
    b_loc = [[0.0, 8.0], [0.0, 8.0]],
    A_sh = [fill(1.0, 1, 1), fill(1.0, 1, 1)],
    b_sh = [15.0]
)

mpvi_2 = StaticGNE2mpAVI(game_2)
mpvi_2_bounded = DyNECT.setParameterSpace(mpvi_2, lb = [0.0], ub = [15.0])
sol_2 = CommonSolve.solve(mpvi_2_bounded, DyNECT.ParametricDAQPSolver)
filter_gne_crs!(sol_2, game_2)

println("\n=== ANALYZING CRITICAL REGIONS ===")

println("\nSOL2 (Scarce W=8) CRs BEFORE filter:")
println("Total CRs: $(length(sol2.CRs))")
for (i, cr) in enumerate(sol2.CRs)
    println("CR #$i:")
    println("  Active constraints (AS): $(cr.AS)")
    println("  th: $(cr.th)")
end

println("\nSOL1 (Abundant W=20) CRs BEFORE filter:")
println("Total CRs: $(length(sol1.CRs))")
for (i, cr) in enumerate(sol1.CRs)
    println("CR #$i:")
    println("  Active constraints (AS): $(cr.AS)")
    println("  th: $(cr.th)")
end

# Dense sampling
println("\n=== DENSE PARAMETER SAMPLING (Sol2) ===")
local hit_count = 0
local total_count = 0
for t in 0:0.05:1
    θ = [t*8.0, t*8.0]
    x = DyNECT.evaluatePWA(sol2, θ)
    global total_count = total_count + 1
    if x !== nothing
        global hit_count = hit_count + 1
        println("θ=$(round.(θ, digits=2)) → ✓")
    else
        println("θ=$(round.(θ, digits=2)) → ✗")
    end
end
println("Coverage: $hit_count / $total_count")

