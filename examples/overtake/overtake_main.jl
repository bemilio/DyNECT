using Revise
using DyNECT
using BlockDiagonals
using ParametricDAQP
using LinearAlgebra
using DynamicalSystems
using Plots
include("overtake_utils.jl")

# Parameters

solve_explicit = false
solve_implicit = true

T_hor = 4
tau = 0.1 #sampling time
v_ref = [20., 25.] # reference speed for each agent 
v_min = 10.
v_max = 30.
d_ref = 10. # reference distance
d_min = 5. # safety distance
a_max = 5. # max acceleration
a_min = -5.
angle_max = pi / 8
angle_min = -pi / 8
l_ref = [0.5, -0.5] #reference lateral position for normal and overtake lane 
gain = 0. # Gain of pre-stabilizing controller. Can be 0.
T_sim = 100

## Case 1: platooning
# states:
# 1) v1 - v_ref_1
# 2) l1 - lref (lateral position)
# 3) p2 - p1 + d_ref (longitudinal pos)
# 4) v2 - v1   /// The second car ignores its reference velocity and just follows the leading vehicle
# 5) l2 - lref
#
# Inputs:
# 1) a1
# 2) alpha1 (angle)
# 3) a2 
# 4) alpha2

A = [1. 0 0 0 0;
    0 1. 0 0 0;
    0 0 1. tau 0;
    0 0 0 1. 0;
    0 0 0 0 1.]

B = [[tau 0;
        0 tau*v_ref[1];
        -tau^2/2 0;
        -tau 0;
        0 0],
    [0 0;
        0 0;
        tau^2/2 0;
        tau 0;
        0 tau*v_ref[1]]]

# # Include pre-stabilizing controller
# K_pre = [
#     -gain * [1. 0. 0. 0. 0.;
#         0. 1. 0. 0. 0.],
#     -gain * [0. 0. 1. 1. 0.;
#         0. 0. 0. 0. 1.]]
# A = A + B[1] * K_pre[1] + B[2] * K_pre[2]
# if (maximum(abs.(eigvals(A))) >= 1)
#     @warn "The pre-stabilizing controller is not stabilizing"
# end

# Objectives
Q = [Matrix{Float64}(BlockDiagonal([Matrix{Float64}(I(2)), zeros(3, 3)])),
    Matrix{Float64}(BlockDiagonal([zeros(2, 2), Matrix{Float64}(I(3))]))]
R = [3 * Matrix{Float64}(I(2)),
    2 * Matrix{Float64}(I(2))]

C_x = [0 0 1. 0 0; # Safety distance
    1. 0 0 0 0; # top speed agent 1
    -1. 0 0 0 0; # min speed agent 1
    1. 0 0 1. 0; # top speed agent 2
    -1. 0 0 -1. 0] # min speed agent 2

b_x = [d_ref - d_min;
    v_max - v_ref[1];
    v_ref[1] - v_min;
    v_max - v_ref[1];
    v_ref[1] - v_min]

C_loc = [[1. 0;
        -1. 0;
        0 1;
        0 -1],
    [1. 0;
        -1. 0;
        0 1;
        0 -1]]

b_loc = [[a_max;
        -a_min;
        angle_max;
        -angle_min],
    [a_max;
        -a_min;
        angle_max;
        -angle_min]]

C_u = [zeros(0, 2), zeros(0, 2)]
b_u = zeros(0)

# Range of initial conditions
nx = size(A, 1)
Theta = (A=C_x', b=b_x, ub=10. * ones(nx), lb=-10. * ones(nx))

game = DyNEP(
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

mpVI = generate_mpVI(game, T_hor)

P, K = DyNECT.solveOLNE(game)
K1 = K[1]
K2 = K[2]
println("K1 = $K1; K2 = $K2")
game.P[:] = P[:]
mpVI = generate_mpVI(game, T_hor)


## Case 2: Initiate overtake
# states:
# 1) v1 - v1_ref 
# 2) l1 - lref (lateral position)
# 3) p2 - p1 (longitudinal pos)
# 4) v2 - v2_ref 
# 5) l2 - lref
#
# Inputs:
# 1) a1
# 2) alpha1 (angle)
# 3) a2 
# 4) alpha2



## Case 3: Perform overtake

## Case 4: Complete overtake

## Case 5: Alone on the road



# Simulate implicit MPC 
n_pv = 6 # two unicycles
if (solve_implicit)
    pv0 = [0., (v_max + v_min) / 2, l_ref[1], -15., (v_max + v_min) / 2, l_ref[1]]
    pv_imp = zeros(6, T_sim)
    u_imp = zeros(4, T_sim)
    agent1 = CoupledODEs(unicycle!, pv0[1:3], zeros(2)) #initialize system with zero input
    agent2 = CoupledODEs(unicycle!, pv0[4:6], zeros(2))
    pv_imp[:, 1] = pv0
    for t in 1:T_sim-1
        pv_now = pv_imp[:, t]
        println("[t=$t] posvel= $pv_now")
        x = posvel_to_state(pv_imp[:, t], Platooning)
        println("[t=$t] x= $x")
        useq, _ = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x; 1], mpVI.A, mpVI.B' * [x; 1])
        solution_found = !isnothing(useq)
        if solution_found
            u_imp[:, t] = vcat(DyNECT.first_input_of_sequence(useq, game.nu, game.N, T_hor)...)
        else
            @warn "[implicit VI solution] Infeasible problem"
        end
        println("Timestep: $t, solution found: $solution_found")
        u_now = u_imp[:, t]
        println("[t=$t] u= $u_now")

        # Evolve dynamic system
        set_parameters!(agent1, u_imp[1:2, t])
        set_parameters!(agent2, u_imp[3:4, t])
        step!(agent1, tau, true) # progress for tau units of time
        step!(agent2, tau, true) # progress for tau units of time
        pv_imp[1:3, t+1] = current_state(agent1)
        pv_imp[4:6, t+1] = current_state(agent2)

        # Evolve linear system
        x_pred = game.A * x + game.B * u_now
        println("[t=$t] x_pred= $x_pred")
        pv_pred = state_to_posvel(x_pred, Platooning)
        println("[t=$t] pv_pred= $pv_pred")

    end
end

# Solve and simulate explicit MPC
if (solve_explicit)
    sol, _ = ParametricDAQP.mpsolve(mpVI, Theta)
    x_exp = Vector{Vector{Float64}}(undef, T_sim)
    u_exp = Vector{Vector{Float64}}(undef, T_sim)
    x_exp[1] = x0
    for t in 1:T_sim-1
        u_exp[t] = vcat(MPC_control(x_exp[t], sol, game.nu, game.N, T_hor)...)
        x_exp[t+1] = game.A * x_exp[t] + game.B * u_exp[t]
    end
    diff = norm(x_exp - x_imp)
    println("Difference explicit-implicit MPC trajectory = $diff")
    ## Test solution
    tol = 10^(-5)

    for i in 1:100
        x0_test = ones(nx) - 2 * rand(nx)
        ind = find_CR(x0_test, sol) # Find  CR corresponding to x0_test
        # Extract primal solution
        if !isnothing(ind)
            usol = sol.CRs[ind].z' * [x0_test; 1]
        end
        uref, _ = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x0_test; 1], mpVI.A, mpVI.B' * [x0_test; 1], tol=tol)
        if isnothing(ind) && !isnothing(uref)
            @warn "The explicit solution is infeasible, while the implicit solution is feasible"
        end
        if !isnothing(ind) && isnothing(uref)
            @warn "The Implicit solution is infeasible, while the Explicit solution is feasible"
        end
        if isnothing(ind) && isnothing(uref)
            println("Both explicit and implicit solution are infeasible")
        end
        if !isnothing(ind) && !isnothing(uref) && norm(uref - usol) > tol
            @warn "Explicit and implicit solutions are different"
        end
        if !isnothing(ind) && !isnothing(uref) && norm(uref - usol) <= tol
            println("Explicit and implicit solutions are equal")
        end
    end
end

## Plot 
# Save position and velocity values to CSV
using CSV
using DataFrames

# Prepare DataFrame for implicit MPC simulation
pv_imp_df = DataFrame(time=collect(0:T_sim-1) .* tau,
    p1=pv_imp[1, :],
    v1=pv_imp[2, :],
    l1=pv_imp[3, :],
    p2=pv_imp[4, :],
    v2=pv_imp[5, :],
    l2=pv_imp[6, :])
CSV.write("pv_imp_results.csv", pv_imp_df)

include("overtake_plot.jl")



