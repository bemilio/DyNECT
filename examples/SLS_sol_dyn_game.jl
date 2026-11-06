using DyNECT
using BlockDiagonals
using LinearAlgebra
using CommonSolve
using Plots
using Random
Random.seed!(1234)

T_hor = 50

# Dynamics
A = [1.0 0.0; 
     0.0   1.0]
B = [[1.0; 0.0;;],
    [0.0; 1.0;;]]

# Objectives
Q = [[0.0 0.0; 
    0.0 1.0],
    [1.0 -1.0; 
    -1.0 1.0]]
R = [[[1.;;], [0.;;]], [[0.;;], [1.;;]]]

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

# Define solver parameters
params = DyNECT.IterativeSolverParams(verbose=true, time_limit=10.0)

# Solve the AVI with DAQP's native AVI solver (https://darnstrom.github.io/daqp/start/advanced/avi)
avi_sol = solve(avi, DyNECT.DAQPSolver; params=params)
println("Solution residual DAQP = $(avi_sol.residual)")

# ============================================================================
# SLS solution and rollout
# ============================================================================
# Best response w.r.t. the other agents' controllers K
sls_solver = CommonSolve.init(game, DyNECT.SLS_BR_K, T_hor)
sls_sol = CommonSolve.solve!(sls_solver)
println("SLS (BR K) status = $(sls_sol.status), residual = $(sls_sol.residual)")

# Best response w.r.t. the other agents' disturbance-to-input maps ϕ_u
sls_ϕ_solver = CommonSolve.init(game, DyNECT.SLS_BR_ϕ, T_hor)
sls_ϕ_sol = CommonSolve.solve!(sls_ϕ_solver)
println("SLS (BR ϕ) status = $(sls_ϕ_sol.status), residual = $(sls_ϕ_sol.residual)")

# Closed-loop rollout of the causal controllers K:
#   u_{i,t} = ∑_{τ≤t} K_i[t,τ] x_τ,   x_{t+1} = A x_t + ∑ B_i u_{i,t}
# Returns x_seq = [x_0, ..., x_T] and u_seq[t][i] = u_{i,t-1}
function rollout(game::DynLQGame, K::Vector{<:AbstractMatrix}, x0::Vector{Float64}, T_hor::Int64)
    nx, nu = game.nx, game.nu
    x_seq = [x0]
    u_seq = Vector{Vector{Vector{Float64}}}()
    for t in 1:T_hor
        x_past = vcat(x_seq...) # [x_0; ...; x_{t-1}]
        u_t = [K[i][(t-1)*nu[i]+1:t*nu[i], 1:t*nx] * x_past for i in 1:game.N]
        push!(u_seq, u_t)
        push!(x_seq, game.A * x_seq[end] + sum(game.B[i] * u_t[i] for i in 1:game.N))
    end
    return x_seq, u_seq
end

x_sls_seq, u_sls_seq = rollout(game, sls_solver.K, x0, T_hor)
x_sls_ϕ_seq, u_sls_ϕ_seq = rollout(game, sls_ϕ_solver.K, x0, T_hor)
stack_inputs(u_seq) = vcat([vcat([u_seq[t][i] for t in 1:T_hor]...) for i in 1:game.N]...) # stacked per agent, as in the AVI
u_sls = stack_inputs(u_sls_seq)
u_sls_ϕ = stack_inputs(u_sls_ϕ_seq)

println("Comparison of input sequences:")
println("  ||u_sls(BR K) - u_avi||        = $(norm(u_sls - avi_sol.x))")
println("  ||u_sls(BR ϕ) - u_avi||        = $(norm(u_sls_ϕ - avi_sol.x))")
println("  ||u_sls(BR K) - u_sls(BR ϕ)||  = $(norm(u_sls - u_sls_ϕ))")
u_avi_seq = DyNECT.arrange_vector_as_time_seq(avi_sol.x, game.nu, game.N, T_hor)

# ============================================================================
# State trajectories from the same x0
# ============================================================================

# Simulate x_{t+1} = A x_t + ∑ B_i u_{i,t} from x0; returns [x_0, ..., x_T]
function simulate_states(game::DynLQGame, u_seq, x0::Vector{Float64}, T_hor::Int64)
    x = [x0]
    for t in 1:T_hor
        push!(x, game.A * x[end] + sum(game.B[i] * u_seq[t][i] for i in 1:game.N))
    end
    return x
end

x_avi_seq = simulate_states(game, u_avi_seq, x0, T_hor)

# Infinite-horizon CLNE: closed loop x_{t+1} = (A + ∑ B_i K_i) x_t, with u_i = K_i x
_, K_clne, res_clne = DyNECT.solveCLNE(game)
println("CLNE residual = $res_clne")
A_clne = game.A + sum(game.B[i] * K_clne[i] for i in 1:game.N)
x_clne_seq = [A_clne^t * x0 for t in 0:T_hor]

# ============================================================================
# Plot: states over time for SLS (BR K and BR ϕ), AVI and CLNE
# ============================================================================

plt_x = plot(layout=(game.nx, 1), size=(700, 300 * game.nx))
for k in 1:game.nx
    plot!(plt_x[k], 0:T_hor, [x[k] for x in x_sls_seq]; label="SLS (BR K)", marker=:circle, xlabel="t", ylabel="x_$k", title="State $k")
    plot!(plt_x[k], 0:T_hor, [x[k] for x in x_sls_ϕ_seq]; label="SLS (BR ϕ)", marker=:utriangle)
    plot!(plt_x[k], 0:T_hor, [x[k] for x in x_avi_seq]; label="AVI (DAQP)", marker=:square)
    plot!(plt_x[k], 0:T_hor, [x[k] for x in x_clne_seq]; label="CLNE", marker=:diamond)
end
display(plt_x)
savefig(plt_x, "examples/SLS_sol_dyn_game_states.png")
