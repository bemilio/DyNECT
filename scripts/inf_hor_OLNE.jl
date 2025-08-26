using DyNECT
using LinearAlgebra
using ParametricDAQP

T_hor = 5
nx = 2
nu = [1, 1]

A = [0.7 1.; 0 0.5]

B = [[0; 1.;;],
    [0; 1.;;]]

# Objectives
Q = [[1. 0; 0 0],
    [0 0; 0 1.]]
R = [[5.;;], [8.;;]]

C_x = zeros(0, 2)

b_x = zeros(0)

C_loc = [zeros(0, 1), zeros(0, 1)]

b_loc = [zeros(0), zeros(0)]

C_u = [zeros(0, 1), zeros(0, 1)]
b_u = zeros(0)

prob = DyNEP(
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

x = 10 * rand(nx)

# Solve infinite horizon problem 
P, K = DyNECT.solveOLNE(prob)
prob.P[:] = P[:]
u_inf = vcat(K...) * x

# Test solution
P_ext, K_ext = DyNECT.solveExtendedARE(prob, K)
for i = 1:prob.N
    err_P = norm(P_ext[i][1:prob.nx, 1:prob.nx] + P_ext[i][1:prob.nx, prob.nx+1:end] - P[i])
    println("Error on P at agent $i = $err_P")
end

# Generate and solve mpVI
mpVI = generate_mpVI(prob, T_hor)
useq, _ = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x; 1], mpVI.A, mpVI.B' * [x; 1])
solution_found = !isnothing(useq)
if solution_found
    u = vcat(DyNECT.first_input_of_sequence(useq, prob.nu, prob.N, T_hor)...)
else
    @error "VI solution not found"
end

err = norm(u_inf - u)

println("Err = $err")