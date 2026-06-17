"""
DyNECT Quick Start: Solving Multiparametric Nash Games

Two agents optimizing their own goals, but their decisions affect each other.
Solve the entire equilibrium manifold as parameters vary.
"""

using DyNECT, CommonSolve, LinearAlgebra, ParametricDAQP

println("\e[34m" * "█" * " DyNECT: Multiparametric GNE Games\n" * "\e[0m")

# ============================================================================
# The Problem
# ============================================================================
println("Two agents sharing a resource (e.g., bandwidth, energy).")
println("Each agent wants to minimize its own cost.")
println("Using more resource makes it harder for the other agent.")
println("What is the equilibrium? And how does it change with demand?\n")

# ============================================================================
# Step 1: Define the game
# ============================================================================
println("\e[33m" * "─ Define the Nash Game - API: mpGNE(N,n,Q,q,A_loc,b_loc,A_sh,b_sh)\n" * "\e[0m")

game = DyNECT.StaticGNEGame(
     # Cost functions (each agent wants to minimize deviation from desired point)
    Q = [[[2.0 -1.0; -1.0 2.0], zeros(2, 2)],
        [zeros(2, 2), [2.0 -1.0; -1.0 2.0]]],
    q = [[-2.0; -1.0],
        [-1.0; -2.0]],

    # Each agent's local limits (e.g., max power output)
    A_loc = [[-1.0 0.0; 0.0 -1.0], [-1.0 0.0; 0.0 -1.0]],
    b_loc = [[1.0, 1.0], [1.0, 1.0]],

    # Shared resource constraint (e.g., total available power)
    A_sh = [[1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]],
    b_sh = [5.0, 5.0],
)

#Parameter space for reparametrization (e.g., demand variations)
θub = [5.0, 5.0]
θlb = [-5.0, -5.0]

println("Game defined: N agents of dimension n, competing for shared resource A_sh, b_sh\n")

# ============================================================================
# Step 2: Solve parametrically
# ============================================================================
println("─ Find Equilibrium Manifold:\n")

sol = CommonSolve.solve(game, DyNECT.mpGNESolver; θub = θub, θlb = θlb, params=DyNECT.IterativeSolverParams(warmstart=:UnconstrainedSolution))
println("Equilibrium PWA mapping found!\n")

# ============================================================================
# Step 3: Interpret the landscape
# ============================================================================
println("$(length(sol.CRs)) filtered active regions found\n")
for (i, region) in enumerate(sol.CRs[1:min(2, length(sol.CRs))])
    println("Region $i: u* = $(round.(region.z[:, 1]; digits=3))")
end
if length(sol.CRs) > 2
    println("... $(length(sol.CRs) - 2) more regions")
end

# Select globally optimal equilibrium
println("\n\e[33m─ Optimal Equilibrium Selection\e[0m")
φ(u) = sum(abs.(u))
optimal = DyNECT.select_optimal_gne!(sol, φ)

println("θ* = $(round.(optimal.θ_star; digits=3))")
println("u* = $(round.(optimal.u_star; digits=3))")
println("φ* = $(round(optimal.φ_star; digits=6))")
println("Region: $(optimal.region_id)\n")

println("\e[34m" * "Next: See examples/...jl for full application\n" * "\e[0m")