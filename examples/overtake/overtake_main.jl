using Revise
using DyNECT
using BlockDiagonals
using ParametricDAQP
using LinearAlgebra
using DynamicalSystems
using Plots
# ###### FOR DEBUG ######
using Infiltrator

# ######################

include("overtake_utils.jl")

# Parameters

solve_explicit = true
solve_implicit = true

T_hor = 2
Δt = 0.1 #sampling time
v_ref = [20., 25.] # reference speed for each agent 
v_min = 10.
v_max = 35.
d_ref = zeros(length(instances(Case))) # A reference distance for every operating mode of the controller
d_ref[Int(Platooning)] = 10.
d_ref[Int(BeginOvertake)] = 5.
d_ref[Int(PerformOvertake)] = -5.
d_ref[Int(CompleteOvertake)] = -10.
tol = 0.1 # Distance in meters from reference position at which the controller is allowed to switch mode
d_min = 5. # safety distance
a_max = 5. # max acceleration
a_min = -5.
angle_max = pi / 16
angle_min = -pi / 16
l_ref = [0.5, -0.5] #reference lateral position for normal and overtake lane 
l_min = 0.5 # Safety lateral distance

gain = 0. # Gain of pre-stabilizing controller. Can be 0.
T_sim = 100

# Create vectors to hold 5 games and 5 mpVI objects
games = Vector{DyNEP}(undef, length(instances(Case)))
mpVIs = Vector{ParametricDAQP.MPVI}(undef, length(instances(Case)))
explicit_sol = Vector{ParametricDAQP.Solution}(undef, length(instances(Case)))

## Define dynamics 
# States:
# 1) v1
# 2) l1 (lateral position)
# 3) p2 - p1 (longitudinal pos)
# 4) v2   
# 5) l2
#
# Inputs:
# 1) a1
# 2) alpha1 (angle)
# 3) a2 
# 4) alpha2
A = [1. 0 0 0 0;
    0 1. 0 0 0;
    -Δt 0 1. Δt 0;
    0 0 0 1. 0;
    0 0 0 0 1.]
B = [[Δt 0;
        0 Δt*v_ref[1];
        -Δt^2/2 0;
        0 0;
        0 0],
    [0 0;
        0 0;
        Δt^2/2 0;
        Δt 0;
        0 Δt*v_ref[2]]]

# Range of states for which explicit solution is computed
nx = size(A, 1)
Theta = Vector{Any}(undef, length(instances(Case)))
for case in instances(Case)
    if case == Platooning
        ub = vcat(v_max, 1., -d_min, v_max, 1.)
        lb = vcat(v_min, 0., -2 * d_ref[Int(Platooning)], v_min, 0.)
        Theta[Int(case)] = (ub=ub, lb=lb)
    elseif case == BeginOvertake
        ub = vcat(v_max, 1., -d_ref[Int(BeginOvertake)] + tol, v_max, 1.)
        lb = vcat(v_min, 0., -d_ref[Int(Platooning)] - tol, v_min, -1.)
        Theta[Int(case)] = (ub=ub, lb=lb)
    elseif case == PerformOvertake
        ub = vcat(v_max, 1., -d_ref[Int(PerformOvertake)] + tol, v_max, 0.)
        lb = vcat(v_min, 0., -d_ref[Int(BeginOvertake)] - tol, v_min, -1.)
        Theta[Int(case)] = (ub=ub, lb=lb)
    elseif case == CompleteOvertake
        ub = vcat(v_max, 1., -d_ref[Int(CompleteOvertake)] + tol, v_max, 1.)
        lb = vcat(v_min, 0., -d_ref[Int(PerformOvertake)] - tol, v_min, -1.)
        Theta[Int(case)] = (ub=ub, lb=lb)
    elseif case == NormalOperation
        ub = vcat(v_max, 1., v_max, 1.)
        lb = vcat(v_min, 0., v_min, 0.)
        Theta[Int(case)] = (ub=ub, lb=lb)
    end
end

## Case 1: platooning

# Objectives
# J₁ = ‖v₁ - vᵈᵉˢ‖² + ‖l₁ - lᵈᵉˢ‖²
# J₂ = ‖p₂ - p₁ - dᵈᵉˢ‖² + ‖v₂ - v₁‖² + ‖l₂ - lᵈᵉˢ‖²

Q = [Matrix{Float64}(Diagonal([1., 1., 0., 0., 0.])),
    Matrix{Float64}([
        1 0 0 -1 0;
        0 0 0 0 0;
        0 0 1 0 0;
        -1 0 0 1 0;
        0 0 0 0 1])]
R = [[Matrix{Float64}(Diagonal([5., 20.,])), zeros(2, 2)],
    [zeros(2, 2), Matrix{Float64}(Diagonal([5., 20.]))]]
q = [[-v_ref[1]; -l_ref[1]; 0.; 0.; 0.],
    [0.; 0.; d_ref[Int(Platooning)]; 0.; -l_ref[1]]]

C_x = zeros(0, 5)
b_x = zeros(0)

# C_x = [0 0 1. 0 0; # Safety distance
#     1. 0 0 0 0; # top speed agent 1
#     -1. 0 0 0 0; # min speed agent 1
#     0 0 0 1. 0; # top speed agent 2
#     0 0 0 -1. 0] # min speed agent 2

# b_x = [-d_min;
#     v_max
#     -v_min;
#     v_max;
#     -v_min]

C_loc = [zeros(0, 2), zeros(0, 2)]
b_loc = [zeros(0), zeros(0)]
# C_loc = [
#     [1. 0; # max acceleration
#         -1. 0; # min acceleration
#         0 1; # max angle
#         0 -1], #min angle
#     [1. 0;
#         -1. 0;
#         0 1;
#         0 -1]]

# b_loc = [[a_max;
#         -a_min;
#         angle_max;
#         -angle_min],
#     [a_max;
#         -a_min;
#         angle_max;
#         -angle_min]]

C_u = [zeros(0, 2), zeros(0, 2)] # No shared input constraints
b_u = zeros(0)

games[Int(Platooning)] = DyNEP(
    A=A,
    Bvec=B,
    Q=Q,
    R=R,
    P=Q,
    q=q,
    p=q,
    C_x=C_x,
    b_x=b_x,
    C_loc_vec=C_loc,
    b_loc_vec=b_loc,
    C_u_vec=C_u,
    b_u=b_u)

mpVIs[Int(Platooning)] = generate_mpVI(games[Int(Platooning)], T_hor)

##### Case 2: Initiate overtake #####
# Objectives
# J₁ = ‖v₁ - vᵈᵉˢ₁‖² + ‖l₁ - lᵈᵉˢ₂‖²
# J₂ = ‖v₂ - vᵈᵉˢ₂‖² + ‖l₂ - lᵈᵉˢ₂‖²

Q = [Matrix{Float64}(Diagonal([1., 1., 0., 0., 0.])),
    Matrix{Float64}(Diagonal([0., 0., 0., 1., 1.]))]
R = [[Matrix{Float64}(Diagonal([5., 20.,])), zeros(2, 2)],
    [zeros(2, 2), Matrix{Float64}(Diagonal([5., 20.]))]]
q = [[-v_ref[1]; -l_ref[1]; 0.; 0.; 0.],
    [0.; 0.; 0.; -v_ref[2]; -l_ref[2]]]

# Safety distance: (p2-p1) <= - α (l2-l1) - d_min 
α = 2.
C_x = [0 -α 1. 0 α; # Safety distance:
    1. 0 0 0 0; # top speed agent 1
    -1. 0 0 0 0; # min speed agent 1
    0. 0 0 1. 0; # top speed agent 2
    0. 0 0 -1. 0] # min speed agent 2

b_x = [-d_min;
    v_max
    - v_min
    v_max
    - v_min]

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

games[Int(BeginOvertake)] = DyNEP(
    A=A,
    Bvec=B,
    Q=Q,
    R=R,
    P=Q,
    q=q,
    p=q,
    C_x=C_x,
    b_x=b_x,
    C_loc_vec=C_loc,
    b_loc_vec=b_loc,
    C_u_vec=C_u,
    b_u=b_u)

mpVIs[Int(BeginOvertake)] = generate_mpVI(games[Int(BeginOvertake)], T_hor)

## Case 3: Perform overtake

# Objectives
# J₁ = ‖v₁ - vᵈᵉˢ₁‖² + ‖l₁ - lᵈᵉˢ₂‖²
# J₂ = ‖v₂ - vᵈᵉˢ₂‖² + ‖l₂ - lᵈᵉˢ₂‖²
Q = [Matrix{Float64}(Diagonal([1., 1., 0., 0., 0.])),
    Matrix{Float64}(Diagonal([0., 0., 0., 1., 1.]))]
R = [[Matrix{Float64}(Diagonal([5., 20.,])), zeros(2, 2)],
    [zeros(2, 2), Matrix{Float64}(Diagonal([5., 20.]))]]
q = [[-v_ref[1]; -l_ref[1]; 0.; 0.; 0.],
    [0.; 0.; 0.; -v_ref[2]; -l_ref[2]]]

# Constraints 
C_x = [0 -1. 0 0 1.; # Safety lateral distance: (l1-l2) >= l_min 
    1. 0 0 0 0; # top speed agent 1
    -1. 0 0 0 0; # min speed agent 1
    0. 0 0 1. 0; # top speed agent 2
    0. 0 0 -1. 0] # min speed agent 2

b_x = [-l_min;
    v_max
    - v_min
    v_max
    - v_min]

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

games[Int(PerformOvertake)] = DyNEP(
    A=A,
    Bvec=B,
    Q=Q,
    R=R,
    P=Q,
    q=q,
    p=q,
    C_x=C_x,
    b_x=b_x,
    C_loc_vec=C_loc,
    b_loc_vec=b_loc,
    C_u_vec=C_u,
    b_u=b_u)

mpVIs[Int(PerformOvertake)] = generate_mpVI(games[Int(PerformOvertake)], T_hor)

#### Case 4: Complete overtake ####
# Objectives
# J₁ = ‖v₁ - vᵈᵉˢ₁‖² + ‖l₁ - lᵈᵉˢ₂‖²
# J₂ = ‖v₂ - vᵈᵉˢ₂‖² + ‖l₂ - lᵈᵉˢ₂‖²
Q = [Matrix{Float64}(Diagonal([1., 1., 0., 0., 0.])),
    Matrix{Float64}(Diagonal([0., 0., 0., 1., 1.]))]
R = [[Matrix{Float64}(Diagonal([5., 20.,])), zeros(2, 2)],
    [zeros(2, 2), Matrix{Float64}(Diagonal([5., 20.]))]]
q = [[-v_ref[1]; -l_ref[1]; 0.; 0.; 0.],
    [0.; 0.; 0.; -v_ref[2]; -l_ref[1]]]

# Constraints 
# Safety distance: (p2-p1) >= α * (l2-l1) + d_min 
C_x = [0 -α -1. 0 α;
    1. 0 0 0 0; # top speed agent 1
    -1. 0 0 0 0; # min speed agent 1
    0. 0 0 1. 0; # top speed agent 2
    0. 0 0 -1. 0] # min speed agent 2

b_x = [-d_min;
    v_max
    - v_min
    v_max
    - v_min]

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

games[Int(CompleteOvertake)] = DyNEP(
    A=A,
    Bvec=B,
    Q=Q,
    R=R,
    q=q,
    P=Q,
    p=q,
    C_x=C_x,
    b_x=b_x,
    C_loc_vec=C_loc,
    b_loc_vec=b_loc,
    C_u_vec=C_u,
    b_u=b_u)

mpVIs[Int(CompleteOvertake)] = generate_mpVI(games[Int(CompleteOvertake)], T_hor)

## Case 5: Normal operation
# States:
# 1) v1
# 2) l1 (lateral position)
# 4) v2   
# 5) l2
#
# Inputs:
# 1) a1
# 2) alpha1 (angle)
# 3) a2 
# 4) alpha2
A_red = [1. 0 0 0;
    0 1. 0 0;
    0 0 1. 0;
    0 0 0 1.]
B = [[Δt 0;
        0 Δt*v_ref[1];
        0 0;
        0 0],
    [0 0;
        0 0;
        Δt 0;
        0 Δt*v_ref[2]]]

# J₁ = ‖v₁ - vᵈᵉˢ₁‖² + ‖l₁ - lᵈᵉˢ₂‖²
# J₂ = ‖v₂ - vᵈᵉˢ₂‖² + ‖l₂ - lᵈᵉˢ₂‖²


Q = [Matrix{Float64}(Diagonal([1., 1., 0., 0.])),
    Matrix{Float64}(Diagonal([0., 0., 1., 1.]))]
R = [[Matrix{Float64}(Diagonal([5., 20.,])), zeros(2, 2)],
    [zeros(2, 2), Matrix{Float64}(Diagonal([5., 20.]))]]
q = [[-v_ref[1]; -l_ref[1]; 0.; 0.],
    [0.; 0.; -v_ref[2]; -l_ref[1]]]

# Constraints 
C_x = [1. 0 0 0; # top speed agent 1
    -1. 0 0 0; # min speed agent 1
    0. 0 1. 0; # top speed agent 2
    0. 0 -1. 0] # min speed agent 2

b_x = [v_max
 - v_min
    v_max
    - v_min]

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

games[Int(NormalOperation)] = DyNEP(
    A=A_red,
    Bvec=B,
    Q=Q,
    R=R,
    q=q,
    P=Q,
    p=q,
    C_x=C_x,
    b_x=b_x,
    C_loc_vec=C_loc,
    b_loc_vec=b_loc,
    C_u_vec=C_u,
    b_u=b_u)

mpVIs[Int(NormalOperation)] = generate_mpVI(games[Int(NormalOperation)], T_hor)

#### Simulations
pv0 = [0., (v_max + v_min) / 2, l_ref[1], -15., (v_max + v_min) / 2, l_ref[1]]
n_pv = 6 # two unicycles
agent1 = CoupledODEs(unicycle!, pv0[1:3], zeros(2)) #initialize system with zero input
agent2 = CoupledODEs(unicycle!, pv0[4:6], zeros(2))

## Simulation: Iterative solution
if (solve_implicit)
    println("Simulating iterative-based controller...")
    pv_imp = zeros(6, T_sim)
    u_imp = zeros(4, T_sim)
    pv_imp[:, 1] = pv0
    local case = Platooning
    for t in 1:T_sim-1
        case = choose_controller(pv_imp[:, t], v_ref, d_ref, l_ref, tol, case)
        x = posvel_to_state(pv_imp[:, t], case, v_ref, l_ref, d_ref)
        mpVI = mpVIs[Int(case)]
        useq, _ = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x; 1], mpVI.A, mpVI.B' * [x; 1])
        solution_found = !isnothing(useq)
        game = games[Int(case)]
        if solution_found
            u_imp[:, t] = vcat(DyNECT.first_input_of_sequence(useq, game.nu, game.N, T_hor)...)
        else
            @warn "[Iterative VI solution] Infeasible problem: time = $t; controller case = $case"
        end
        u_now = u_imp[:, t]
        # Evolve dynamic system
        set_parameters!(agent1, u_imp[1:2, t])
        set_parameters!(agent2, u_imp[3:4, t])
        step!(agent1, Δt, true) # progress for Δt units of time
        step!(agent2, Δt, true) # progress for Δt units of time
        pv_imp[1:3, t+1] = current_state(agent1)
        pv_imp[4:6, t+1] = current_state(agent2)

        println("[t=$t] x= $x")
        pv_now = pv_imp[:, t]
        println("[t=$t] pv= $pv_now")
    end
end

## Simulation: Explicit MPC
pv_exp = zeros(6, T_sim)
pv_exp[:, 1] = pv0

# x0 = posvel_to_state(pv0, Platooning, v_ref, l_ref, d_ref)
# expl_sol_1, _ = ParametricDAQP.mpsolve(mpVIs[1], Theta[1])
# index_CR = DyNECT.find_CR(x0, expl_sol_1)
if (solve_explicit)
    for case in instances(Case)
        println("[overtake_main] Computing explicit solution for case $case ...")
        explicit_sol[Int(case)], _ = ParametricDAQP.mpsolve(mpVIs[Int(case)], Theta[Int(case)])
        println("[overtake_main] Explicit solution COMPUTED! Case $case")
    end
    println("Simulating explicit controller...")
    agent1 = CoupledODEs(unicycle!, pv0[1:3], zeros(2)) #initialize system with zero input
    agent2 = CoupledODEs(unicycle!, pv0[4:6], zeros(2))

    u_exp = zeros(4, T_sim)
    local case = Platooning
    for t in 1:T_sim-1
        case = choose_controller(pv_exp[:, t], v_ref, d_ref, l_ref, tol, case)
        x = posvel_to_state(pv_exp[:, t], case, v_ref, l_ref, d_ref)
        # Extract primal solution
        u_seq = evaluate_solution(explicit_sol[Int(case)], x)
        if !isnothing(u_seq)
            u_exp[:, t] = vcat(DyNECT.first_input_of_sequence(u_seq, games[Int(case)].nu, games[Int(case)].N, T_hor)...)
        else
            @warn "[Explicit VI solution] Infeasible problem: time = $t; controller case = $case"
        end
        # Evolve dynamic system
        set_parameters!(agent1, u_exp[1:2, t])
        set_parameters!(agent2, u_exp[3:4, t])
        step!(agent1, Δt, true) # progress for Δt units of time
        step!(agent2, Δt, true) # progress for Δt units of time
        pv_exp[1:3, t+1] = current_state(agent1)
        pv_exp[4:6, t+1] = current_state(agent2)
        println("[t=$t] x= $x")
        pv_now = pv_exp[:, t]
        println("[t=$t] pv= $pv_now")
    end
end
if solve_explicit && solve_implicit
    diff = norm(pv_imp - pv_exp)
    if diff > tol
        @warn("The explicit and the implicit solution appear to be different")
    end
    println("Difference explicit-implicit MPC trajectory = $diff")
end

## Plot 
# Save position and velocity values to CSV
using CSV
using DataFrames

# Prepare DataFrame for implicit MPC simulation
if solve_implicit
    pv_imp_df = DataFrame(time=collect(0:T_sim-1) .* Δt,
        p1=pv_imp[1, :],
        v1=pv_imp[2, :],
        l1=pv_imp[3, :],
        p2=pv_imp[4, :],
        v2=pv_imp[5, :],
        l2=pv_imp[6, :])
    CSV.write("pos_vel_implicit_results.csv", pv_imp_df)
end
if solve_explicit
    pv_exp_df = DataFrame(time=collect(0:T_sim-1) .* Δt,
        p1=pv_exp[1, :],
        v1=pv_exp[2, :],
        l1=pv_exp[3, :],
        p2=pv_exp[4, :],
        v2=pv_exp[5, :],
        l2=pv_exp[6, :])
    CSV.write("pos_vel_explicit_results.csv", pv_exp_df)
end
if solve_explicit || solve_implicit
    include("overtake_plot.jl")
end


