using DyNECT, CommonSolve, LinearAlgebra, Statistics

# Setup (same as Rosens example)
Q = [[[1.;;], [-1.;;]], [[1.;;], [2.;;]]]
q = [[0.], [0.]]
A_loc = [zeros(0, 1), zeros(0, 1)]
b_loc = [[], []]
A_sh = [[-1;;], [-1;;]]
b_sh = [-1.]

gnep = StaticGNEGame(Q, q, A_loc, b_loc, A_sh, b_sh)
sol = CommonSolve.solve(gnep, StaticGNEpDAQPSolver, θub=[5.0], θlb=[-5.0])

#testing
metrics = [
    (name="Welfare (L2 norm)", f = u -> 0.5 * sum(u.^2)),
    (name="Fairness (variance)", f = u -> sum((u .- mean(u)).^2)),
    (name="Weighted asymmetric", f = u -> 0.5 * (2*u[1]^2 + u[2]^2)),
]

println("Exploring performance metrics")

for metric in metrics
    println("Metric: $(metric.name)")
    result = select_optimal_gne!(sol, metric.f)
    
    println("  θ*: $(round.(result.θ_star; digits=6))")
    println("  u*: $(round.(result.u_star; digits=6))")
    println("  φ*: $(round(result.φ_star; digits=8))")
    println("  region: $(result.region_id)\n")
end