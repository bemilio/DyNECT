using Revise
using DyNECT
using BlockDiagonals
using ParametricDAQP
using LinearAlgebra

T_hor = 4
u_max = [5., 5.]
u_min = [-3., -5.]

A = [1. 1.; 0 1.]

B = [[0; 1.;;],
    [0; 1.;;]]

# Objectives
Q = [[1. 0; 0 0],
    [0 0; 0 1.]]
R = [3 * Matrix{Float64}(I(2)),
    2 * Matrix{Float64}(I(2))]

C_x = zeros(0, 2)

b_x = zeros(0)

C_loc = [[1.; -1.;;], [1.; -1.;;]]

b_loc = [[u_max[1]; -u_min[1]], [u_max[2]; -u_min[2]]]

C_u = [zeros(0, 1), zeros(0, 1)]
b_u = zeros(0)

# Range of initial states 
nx = size(A, 1)
Theta = (ub=5. * ones(nx), lb=-5. * ones(nx))

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

game.P, K = DyNECT.solveOLNE(game)

mpVI = generate_mpVI(game, T_hor)
sol, _ = ParametricDAQP.mpsolve(mpVI, Theta)