using DyNECT
using LinearAlgebra
using ParametricDAQP
using BlockArrays
using MatrixEquations
using CommonSolve
using Random
using Plots


# Jᵢ = |xᵢ-10|²
# x₁ + x₂ ≤ 1; 
# 4x₁ + x₁ ≤ 2

Q = [[[1.;;], [0.;;]], # Quadratic terms of agent 1 (Q₁₁, Q₁₂)
     [[0.;;], [1.;;]]] # Quadratic terms of agent 1 (Q₂₁, Q₂₂)
q = [[-10.], 
     [-10.]]
# x1 ≥ 0, x2 ≥ 0
A_loc = [[-1.;;], [-1.;;]] 
b_loc = [[0.], [0.]]
# x1 + x2 ≤ 1; 4x1 + x2 ≤ 2 
A_sh = [[1.; 
         4.0;;], 
        [1.0; 
         1.0;;]]
b_sh = [1.; 
        2.]

gnep = DyNECT.StaticGNEGame(Q, q, A_loc, b_loc, A_sh, b_sh)

# restrict parameter space for reparametrization
θub = [5.0; 5.0]
θlb = [-5.0; -5.0]
mpavi = DyNECT.NabetaniParametrization(gnep, θub=θub, θlb=θlb)

sol = CommonSolve.solve(gnep, DyNECT.NabetaniParametrizationSolver, θub = θub, θlb = θlb)

x_sol = Vector{Vector{Float64}}()

n_grid = 1000
θ1_grid = range(θlb[1], θub[1]; length=n_grid)
θ2_grid = range(θlb[2], θub[2]; length=n_grid)

for θ1 in θ1_grid, θ2 in θ2_grid
    θ = [θ1, θ2]
    x = DyNECT.evaluatePWA(sol, θ)
    if !isnothing(x)
        push!(x_sol, collect(x))
    end
end

if !isempty(x_sol)
    x1 = [x[1] for x in x_sol]
    x2 = [x[2] for x in x_sol]
    plt = scatter(x1, x2; xlabel="x₁", ylabel="x₂", title="x_sol over θ grid ($(n_grid)×$(n_grid))", label="")
    display(plt)
    savefig(plt, "congestion_game_scatter.png")
end

# Test solution: x₂ = min(1 - x₁, 2 - 4x₁) for x₁ ∈ [0, .5]
tol = 1e-6
all_solutions_found = true
for x in x_sol
    global all_solutions_found
    x2_sol = min(1 - x[1], 2 - 4 * x[1])
    all_solutions_found = all_solutions_found & (norm(x[2] - x2_sol) < tol)
    all_solutions_found = all_solutions_found & (0.0 <= x[1] <= 0.5)
end
println("all solutions found = $all_solutions_found")


# Test optimal selection
function closest_point_on_segment(P, A, B)
    d = B - A
    t = clamp(dot(P - A, d) / dot(d, d), 0.0, 1.0)
    A + t * d
end

all_optima_found = true
x_des_all = Vector{Vector{Float64}}()
x_res_all = Vector{Vector{Float64}}()
x_exp_all = Vector{Vector{Float64}}()
for test in 1:50
    global all_optima_found
    x_des = rand(2) .* [.5;2.0] # 0 <= x_des <= .5
    ϕ(x) = sum(abs2, x - x_des) # |x-x_des|²
    opt_gnep = OptimalGNEP(gnep, ϕ)
    result = CommonSolve.solve(opt_gnep, DyNECT.PWAConvexOptSolver)

    # Candidate optima: closest point on each segment of x₂ = min(1-x₁, 2-4x₁), x₁ ∈ [0, .5]
    proj1 = closest_point_on_segment(x_des, [0.0, 1.0], [1 / 3, 2 / 3])   # x2 = 1 - x1,   x1 ∈ [0, 1/3]
    proj2 = closest_point_on_segment(x_des, [1 / 3, 2 / 3], [0.5, 0.0])  # x2 = 2 - 4x1,  x1 ∈ [1/3, .5]
    x_expected = norm(proj1 - x_des) <= norm(proj2 - x_des) ? proj1 : proj2
    all_optima_found = all_optima_found & (norm(result.x - x_expected) < tol)
    println(result.x)

    push!(x_des_all, x_des)
    push!(x_res_all, collect(result.x))
    push!(x_exp_all, collect(x_expected))
end
println("all optima found = $all_optima_found")

plt_opt = scatter([x[1] for x in x_des_all], [x[2] for x in x_des_all]; color=:orange, label="x_des")
scatter!(plt_opt, [x[1] for x in x_res_all], [x[2] for x in x_res_all]; color=:blue, label="result.x")
scatter!(plt_opt, [x[1] for x in x_exp_all], [x[2] for x in x_exp_all]; color=:green, label="x_expected")
quiver!(plt_opt, [x[1] for x in x_des_all], [x[2] for x in x_des_all];
    quiver=([x_res_all[i][1] - x_des_all[i][1] for i in 1:length(x_des_all)],
            [x_res_all[i][2] - x_des_all[i][2] for i in 1:length(x_des_all)]),
    color=:gray, label="")
plot!(plt_opt; xlabel="x₁", ylabel="x₂", title="x_des vs optimal GNE", aspect_ratio=:equal)
display(plt_opt)
savefig(plt_opt, "optimal_gne_arrows.png")