using Pkg
Pkg.activate("..")         

include("../src/Static_mpGNE.jl")
using .Static_mpGNE
include("../src/Solver.jl")

using CairoMakie
using LinearAlgebra
using ParametricDAQP       

game = GameBuilder(N=2)

@player game 1 n=2
@player game 2 n=2

@param game Q1 dims=(2,2)
@param game Q2 dims=(2,2)
@param game R  dims=(2,2)
@param game c1 dims=(2,)
@param game c2 dims=(2,)

@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game [
    x_1_1 + x_1_2 + x_2_1 + x_2_2,
    x_1_1 + x_2_2
] <= [3.0, 2.0]

assign_params!(game, Dict(
    :Q1 => [2.0 0.0; 0.0 1.0],
    :Q2 => [3.0 0.0; 0.0 2.0],
    :R  => [0.0 1.0; 1.0 0.0],   # testing coupling
    :c1 => [-1.0, 0.0],
    :c2 => [0.0, -1.0]
))

mpvi = build_mpvi(game)
sol  = solve_gne(mpvi)

println("Critical regions: $(length(sol.CRs))")

# ─────────────────────────────────────────────
# HELPERS (testing version)
# ─────────────────────────────────────────────

function in_theta(mpvi, β)
    all(Float64.(mpvi.C) * β .<= mpvi.e .+ 1e-8)
end

function safe_heatmap!(ax, X, Y, M; kwargs...)
    if all(isnan, M)
        println("Skipping empty plot")
        return
    end
    heatmap!(ax, X, Y, Float32.(M); kwargs...)
end

# GRID
lb, ub = _extract_theta_bounds(mpvi)

β1 = range(0.01, ub[1]-0.01, length=120)
β2 = range(0.01, ub[2]-0.01, length=120)

n1, n2 = length(β1), length(β2)

# COMPUTE x*(β)
x1_map = fill(NaN, n2, n1)
x2_map = similar(x1_map)
x3_map = similar(x1_map)
x4_map = similar(x1_map)

for i in 1:n2, j in 1:n1
    β = [β1[j], β2[i]]

    if !in_theta(mpvi, β)
        continue
    end

    x = evaluate_gne(sol, β)

    x1_map[i,j] = x[1]
    x2_map[i,j] = x[2]
    x3_map[i,j] = x[3]
    x4_map[i,j] = x[4]
end

# PLOT
fig = Figure(size=(1000,800))

ax1 = Axis(fig[1,1], title="x₁₁*(β)")
safe_heatmap!(ax1, β1, β2, x1_map, colormap=:viridis)

ax2 = Axis(fig[1,2], title="x₁₂*(β)")
safe_heatmap!(ax2, β1, β2, x2_map, colormap=:viridis)

ax3 = Axis(fig[2,1], title="x₂₁*(β)")
safe_heatmap!(ax3, β1, β2, x3_map, colormap=:viridis)

ax4 = Axis(fig[2,2], title="x₂₂*(β)")
safe_heatmap!(ax4, β1, β2, x4_map, colormap=:viridis)

save("../output/pwa_xstar.png", fig)
println("Saved → output/pwa_xstar.png")