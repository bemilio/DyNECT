using DyNECT
using CommonSolve
using LinearAlgebra
using ParametricDAQP
using Plots

Q = [[[1.;;], [-1.;;]], # Quadratic terms of agent 1 (Q₁₁, Q₁₂)
    [[1.;;], [2.;;]]] # Quadratic terms of agent 1 (Q₂₁, Q₂₂)
q = [[0.], 
[0.]]
A_loc = [zeros(0, 1), zeros(0, 1)] # No local constraints
b_loc = [[], []] # No local constraints
A_sh = [[-1;;], [-1;;]]
b_sh = [-1.]

gnep = DyNECT.StaticGNEGame(Q, q, A_loc, b_loc, A_sh, b_sh)

# restrict parameter space for reparametrization
θub = [5.0]
θlb = [-5.0]
sol = CommonSolve.solve(gnep, DyNECT.NabetaniParametrizationSolver; θub = θub, θlb = θlb)

for cr in sol.CRs
println("$cr")
end

tol = 1e-6
test_interval = θlb[1]:0.1:θub[1]
x_sol = []

for θ in test_interval
    x = DyNECT.evaluatePWA(sol, [θ])
    if !isnothing(x)
        push!(x_sol, x)
    end
end
# solution expected: x1 = 1-x2,
# -1<x2<1/2
all_test_ok = all(norm(xk[1] - (1 - xk[2])) < tol && xk[2] <= 0.5 && xk[2] >= -1 for xk in x_sol)


println("All test_ok are true: ", all_test_ok)

if !isempty(x_sol)
    x1 = [x[1] for x in x_sol]
    x2 = [x[2] for x in x_sol]
    plt = scatter(x1, x2; xlabel="x₁", ylabel="x₂", title="x_sol", label="")
    display(plt)
    savefig(plt, "congestion_game_scatter.png")
end

