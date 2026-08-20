using DyNECT
using BlockDiagonals
using LinearAlgebra
using CommonSolve
using Plots
using Random
Random.seed!(1234)

T_hor = 5

# Dynamics
A = [1.0 1.0; 
     0   1.0]
B = [[0; 1.;;],
    [0; 1.;;]]

# Objectives
Q = [[1. 0; 0 0],
    [0 0; 0 1.]]
R = [[[5.;;], [0.;;]], [[0.;;], [8.;;]]]

# No state/input constraints
C_x = zeros(0, 2)
b_x = zeros(0)
C_loc = [zeros(0, 1), zeros(0, 1)]
b_loc = [zeros(0), zeros(0)]
C_u = [zeros(0, 1), zeros(0, 1)]
b_u = zeros(0)

# Construct game
game = DynLQGame(
    A=A,
    B=B,
    Q=Q,
    R=R,
    C_x=C_x,
    b_x=b_x,
    C_loc_vec=C_loc,
    b_loc_vec=b_loc,
    C_u_vec=C_u,
    b_u=b_u)

# Convert dynamic game to AVI
mpvi = DynLQGame2mpAVI(game, T_hor)

# Define initial state
x0 = 2 .* randn(game.nx)

# Define AVI for the given initial state
avi = DyNECT.AVI(mpvi, x0)

# Define parameters for the iterative solvers
params = DyNECT.IterativeSolverParams(verbose=true, time_limit=10.0)

# Solve the AVI via iterative solvers
DR_sol = solve(avi, DyNECT.DouglasRachford; params=params)
println("Solution residual DR = $(DR_sol.residual)")

# ============================================================================
# SLS solution and rollout
# ============================================================================
# NOTE: SLSSolver's achievability constraint is known to have unresolved
# correctness issues (it can report an objective value below the provable
# H2-optimal cost for the same weights), so ϕ_x/ϕ_u are not guaranteed to
# correspond to a physically achievable closed-loop trajectory. The rollout
# below is still computed and compared, but should be read with that caveat.

sls_solver = CommonSolve.init(game, DyNECT.SLSSolver, T_hor)
sls_sol = CommonSolve.solve!(sls_solver)
println("SLS status = $(sls_sol.status), residual = $(sls_sol.residual)")

# Given causal (block lower-triangular) gains K[j] such that u_j = K[j]*x̄, the closed-loop
# trajectory from x0 solves (Z̅ - A_K)x̄ = w̄, with A_K = A̅ + Σⱼ B̅[j]K[j] and w̄ = [x0;0;...;0]
# (the same relation SLSSolver itself uses to certify K[i] = ϕ_u_i/ϕ_x, generalized to the
# closed loop across all agents at once). Works for any causal K[j], including constant
# (memoryless) gains lifted to block form via kron(I(T_hor), K_i).
function rollout(game::DynLQGame, K::Vector{<:AbstractMatrix}, x0::Vector{Float64}, T_hor::Int64)
    A̅ = kron(I(T_hor), game.A)
    B̅ = [kron(I(T_hor), game.B[j]) for j in 1:game.N]
    Z̅ = kron(diagm(-1 => ones(T_hor - 1)), I(game.nx))
    A_K = A̅ + sum(B̅[j] * K[j] for j in 1:game.N)
    w̄ = zeros(T_hor * game.nx)
    w̄[1:game.nx] = x0
    x̄ = (Z̅ - A_K) \ w̄
    ū = vcat([K[j] * x̄ for j in 1:game.N]...)
    return x̄, ū
end

x_sls, u_sls = rollout(game, sls_solver.K, x0, T_hor)

println("Comparison: SLS rollout vs AVI (DR) solution")
println("  ||u_sls - u_avi|| = $(norm(u_sls - DR_sol.x))")
u_sls_seq = DyNECT.arrange_vector_as_time_seq(u_sls, game.nu, game.N, T_hor)
u_avi_seq = DyNECT.arrange_vector_as_time_seq(DR_sol.x, game.nu, game.N, T_hor)

# ============================================================================
# Infinite-horizon CLNE / OLNE rollouts, from the same x0
# ============================================================================

_, K_clne, res_clne = DyNECT.solveCLNE(game)
_, K_olne, res_olne = DyNECT.solveOLNE(game)
println("CLNE residual = $res_clne")
println("OLNE residual = $res_olne")

_, u_clne = rollout(game, [kron(I(T_hor), K_clne[j]) for j in 1:game.N], x0, T_hor)
_, u_olne = rollout(game, [kron(I(T_hor), K_olne[j]) for j in 1:game.N], x0, T_hor)

u_clne_seq = DyNECT.arrange_vector_as_time_seq(u_clne, game.nu, game.N, T_hor)
u_olne_seq = DyNECT.arrange_vector_as_time_seq(u_olne, game.nu, game.N, T_hor)

# ============================================================================
# Plot: SLS rollout vs. CLNE / OLNE rollouts, per agent
# ============================================================================

plt = plot(layout=(game.N, 1), size=(700, 300 * game.N))
for i in 1:game.N
    plot!(plt[i], 1:T_hor, [u_sls_seq[t][i][1] for t in 1:T_hor]; label="SLS", marker=:circle, xlabel="t", ylabel="u_$i", title="Agent $i input")
    plot!(plt[i], 1:T_hor, [u_clne_seq[t][i][1] for t in 1:T_hor]; label="CLNE", marker=:square)
    plot!(plt[i], 1:T_hor, [u_olne_seq[t][i][1] for t in 1:T_hor]; label="OLNE", marker=:diamond)
end
display(plt)
savefig(plt, "examples/SLS_sol_dyn_game_inputs.png")

plt_diff = plot(layout=(game.N, 1), size=(700, 300 * game.N))
for i in 1:game.N
    diff_clne = [u_sls_seq[t][i][1] - u_clne_seq[t][i][1] for t in 1:T_hor]
    diff_olne = [u_sls_seq[t][i][1] - u_olne_seq[t][i][1] for t in 1:T_hor]
    plot!(plt_diff[i], 1:T_hor, diff_clne; label="SLS - CLNE", marker=:square, xlabel="t", ylabel="Δu_$i", title="Agent $i input difference")
    plot!(plt_diff[i], 1:T_hor, diff_olne; label="SLS - OLNE", marker=:diamond)
end
display(plt_diff)
savefig(plt_diff, "examples/SLS_sol_dyn_game_diff.png")

println("||u_sls - u_clne_rollout|| = $(norm(u_sls - u_clne))")
println("||u_sls - u_olne_rollout|| = $(norm(u_sls - u_olne))")
