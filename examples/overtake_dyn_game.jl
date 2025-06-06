using Revise
using DyNECT
using BlockDiagonals
using ParametricDAQP
using LinearAlgebra


function find_CR(x0::Vector{Float64}, sol; eps_gap=1e-6)
    #Find the critical region for x0
    contained_in = Int64[]
    for (ind, region) in enumerate(sol.CRs)
        try
            violation = minimum(region.bth - region.Ath' * x0)
            if (violation >= -eps_gap)
                push!(contained_in, ind)
            end
        catch err
            println("Error: ", err)
        end
    end
    return isempty(contained_in) ? nothing : contained_in[1]
end

function first_input_of_sequence(u::Vector{Float64}, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    u_first = Vector{Vector{Float64}}(undef, N)
    start_row = 1
    for i = 1:N
        u_first[i] = u[start_row:start_row+nu[i]-1]
        start_row += nu[i] * T_hor
    end
    return u_first
end

function MPC_control(x0::Vector{Float64}, sol, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    ind = find_CR(x0, sol)
    # Extract primal solution
    u = sol.CRs[ind].z' * [x0; 1]
    # Extract first input of each agent's sequence
    return first_input_of_sequence(u, nu, N, T_hor)
end

# 2-vehicles platooning
# states:
# 1) v1 - v_ref 
# 2) l1 - lref (lateral position)
# 3) p1 - p2 - pref (longitudinal pos)
# 4) v1 - v2 
# 5) l2 - lref
#
# Inputs:
# 1) a1
# 2) alpha1 (angle)
# 3) a2 
# 4) alpha2

T_hor = 5
tau = 0.1 #sampling time
v_ref = 20. # reference speed 
v_min = 10.
v_max = 30.
d_ref = 10. # reference distance
d_min = 5. # safety distance
a_max = 10. # max acceleration
a_min = -10.
angle_max = pi / 8
angle_min = -pi / 8

A = [1. 0 0;
    0 1. tau;
    0 0 1]
B = [[
        tau;
        tau^2/2;
        tau;;
    ],
    [
        0;
        -tau^2/2;
        -tau;;
    ]]

# Objectives
Q = [Matrix{Float64}(BlockDiagonal([[1.;;], zeros(2, 2)])),
    Matrix{Float64}(BlockDiagonal([[0.;;], Matrix{Float64}(I(2))]))]
R = [3 * [1.;;],
    2 * [1.;;]]
P = Q

C_x = [0 -1. 0; # Safety distance
    1. 0 0; # top speed agent 1
    -1. 0 0; # min speed agent 1
    1. 0 1.; # top speed agent 2
    -1 0 -1] # min speed agent 2

b_x = [d_ref - d_min;
    v_max - v_ref;
    v_ref - v_min;
    v_max - v_ref;
    v_ref - v_min]

C_loc = [[1.; -1.;;], [1.; -1.;;]]

b_loc = [[a_max; -a_min], [a_max; -a_min]]

C_u = [zeros(0, 2), zeros(0, 2)]
b_u = zeros(0)

# A = [1. 0 0 0 0;
#     0 1. 0 0 0;
#     0 0 1. tau 0;
#     0 0 0 1 0;
#     0 0 0 0 1]
# B = [[tau 0;
#         0 tau*v_ref;
#         tau^2/2 0;
#         tau 0;
#         0 0],
#     [0 0;
#         0 0;
#         -tau^2/2 0;
#         -tau 0;
#         0 tau*v_ref]]


# # Objectives
# Q = [Matrix{Float64}(BlockDiagonal([Matrix{Float64}(I(2)), zeros(3, 3)])),
#     Matrix{Float64}(BlockDiagonal([zeros(2, 2), Matrix{Float64}(I(3))]))]
# R = [3 * Matrix{Float64}(I(2)),
#     2 * Matrix{Float64}(I(2))]
# P = Q

# C_x = [0 0 -1. 0 0; # Safety distance
#     1. 0 0 0 0; # top speed agent 1
#     -1. 0 0 0 0; # min speed agent 1
#     1. 0 0 1. 0; # top speed agent 2
#     -1 0 0 -1 0] # min speed agent 2

# b_x = [d_ref - d_min;
#     v_max - v_ref;
#     v_ref - v_min;
#     v_max - v_ref;
#     v_ref - v_min]

# C_loc = [[1. 0;
#         -1. 0;
#         0 1;
#         0 -1],
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

# C_u = [zeros(0, 2), zeros(0, 2)]
# b_u = zeros(0)

# Range of initial conditions
nx = size(A, 1)
Theta = (A=C_x', b=b_x, ub=10. * ones(nx), lb=-10. * ones(nx))

game = DyNEP(
    A=A,
    Bvec=B,
    Q=Q,
    R=R,
    P=P,
    C_x=C_x,
    b_x=b_x,
    C_loc_vec=C_loc,
    b_loc_vec=b_loc,
    C_u_vec=C_u,
    b_u=b_u)
mpVI = generate_mpVI(game, T_hor)

# Simulate implicit MPC 

x0 = ones(nx) - 2 * rand(nx)
T_sim = 20
x_imp = Vector{Vector{Float64}}(undef, T_sim)
u_imp = Vector{Vector{Float64}}(undef, T_sim)
x_imp[1] = x0
for t in 1:T_sim-1
    useq, _ = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x0; 1], mpVI.A, mpVI.B' * [x0; 1])
    if !isnothing(useq)
        u_imp[t] = vcat(first_input_of_sequence(useq, game.nu, game.N, T_hor)...)
    end
    x_imp[t+1] = game.A * x_imp[t] + game.B * u_imp[t]
end

# Solve and simulate explicit MPC
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

## Plot 

