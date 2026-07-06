using CommonSolve
using DyNECT
using BlockDiagonals
using LinearAlgebra
using Random
Random.seed!(1234)

T_hor = 3
u_max = [0.1, 0.1]
u_min = [-0.1, -0.2]

# Dynamics
A = [0.7 1.; 0 0.5]
B = [[0; 1.;;],
    [0; 1.;;]]

# Objectives
Q = [[1. 0; 0 0],
    [0 0; 0 1.]]
R = [[[5.;;], [0.;;]], [[0.;;], [8.;;]]]

# State constraints
C_x = [Matrix{Float64}(I(2)); -Matrix{Float64}(I(2))]
b_x = [5 * ones(2); 5 * ones(2)]

# Local input constraints
C_loc = [[1.; -1.;;], [1.; -1.;;]]
b_loc = [[u_max[1]; -u_min[1]], [u_max[2]; -u_min[2]]]

# Shared input constraints
C_u = [zeros(0, 1), zeros(0, 1)]
b_u = zeros(0)

# Construct game
game = DynLQGame(
    A=A,
    Bvec=B,
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

# Solve multi-parametric problem for all initial states
# Range of initial states 
nx = size(A, 1)
mpvi.ub[:] = 5. * ones(nx)
mpvi.lb[:] = -5. * ones(nx)
mpvi_sol = CommonSolve.solve(mpvi, DyNECT.ParametricDAQPSolver)

# Define initial state
x0 = 2 .* randn(game.nx)

# Define AVI for the given initial state
avi = DyNECT.AVI(mpvi, x0)

# Retrieve multi-parametric solution for the given state
pDAQP_sol = DyNECT.evaluatePWA(mpvi_sol, x0)
pDAQP_res = DyNECT.compute_residual(avi, pDAQP_sol)
println("Solution residual pDAQP (explicit) = $pDAQP_res")

# Define parameters for the iterative solvers
params = DyNECT.IterativeSolverParams(verbose=true, time_limit=10.0)

# Solve the AVI via iterative solvers

DR_sol = CommonSolve.solve(avi, DyNECT.DouglasRachford; params=params)
println("Solution residual DR = $(DR_sol.residual)")

ADMM_sol = CommonSolve.solve(game, DyNECT.ADMMCLQGSolver; x0=x0, T_hor=T_hor, params=params)
println("Solution residual ADMM = $(ADMM_sol.residual)")

DGSQP_sol = CommonSolve.solve(avi, DyNECT.DGSQPSolver; params=params)
println("Solution residual DGSQP = $(DGSQP_sol.residual)")

monviso_sol = CommonSolve.solve(avi, DyNECT.MonvisoSolver; method=:pg, params=params)
println("Solution residual monviso = $(monviso_sol.residual)")
