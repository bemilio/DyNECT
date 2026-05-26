using DyNECT
using CommonSolve
using LinearAlgebra
using ParametricDAQP
using Random
using Plots

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
mpavi = StaticGNE2mpAVI(gnep, θub=θub, θlb=θlb)
println("$mpavi")
sol = CommonSolve.solve(gnep, DyNECT.StaticGNEpDAQPSolver, θub = θub, θlb = θlb)

for cr in sol.CRs
println("$cr")
end

x_sol = Vector{Vector{Float64}}()

for _ in 1:100000
    θ = rand(2) .* (θub .- θlb) .+ θlb
    x = DyNECT.evaluatePWA(sol, θ)
    if !isnothing(x)
        push!(x_sol, collect(x))
    end
end

if !isempty(x_sol)
    x1 = [x[1] for x in x_sol]
    x2 = [x[2] for x in x_sol]
    plt = scatter(x1, x2; xlabel="x₁", ylabel="x₂", title="x_sol for random θ (1000 samples)", label="")
    display(plt)
    savefig(plt, "congestion_game_scatter.png")
end



