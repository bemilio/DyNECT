using Revise
using DyNECT
using BlockDiagonals
using ParametricDAQP
using LinearAlgebra
using Pkg
Pkg.precompile()

T_hor = 5
u_max = [1., 0.5]
u_min = [-0.5, -2.]

A = [0.7 1.; 0 0.5]

B = [[0; 1.;;],
    [0; 1.;;]]

# Objectives
Q = [[1. 0; 0 0],
    [0 0; 0 1.]]
R = [[5.;;], [8.;;]]

C_x = [Matrix{Float64}(I(2)); -Matrix{Float64}(I(2))]

b_x = [ones(2); ones(2)]

C_loc = [[1.; -1.;;], [1.; -1.;;]]

b_loc = [[u_max[1]; -u_min[1]], [u_max[2]; -u_min[2]]]

C_u = [zeros(0, 1), zeros(0, 1)]
b_u = zeros(0)

# Range of initial states 
nx = size(A, 1)
Theta = (ub=1. * ones(nx), lb=-1. * ones(nx))

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

P, K = DyNECT.solveOLNE(game)
game.P[:] = P[:]

mpVI = generate_mpVI(game, T_hor)
@time begin
    sol, _ = ParametricDAQP.mpsolve(mpVI, Theta)
end

## Test solution
tol = 10^(-5)
all_good = true
for i in 1:100
    x0_test = ones(game.nx) - 2 * rand(game.nx)
    # x0_test = [-0.7896149115496409, -0.5772141696821653]
    ind = DyNECT.find_CR(x0_test, sol) # Find  CR corresponding to x0_test
    # Extract primal solution
    if !isnothing(ind)
        usol = sol.CRs[ind].z' * [x0_test; 1]
    end
    uref, res = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x0_test; 1], mpVI.A, mpVI.B' * [x0_test; 1], tol=tol)
    if isnothing(ind) && !isnothing(uref)
        @warn "The explicit solution is infeasible, while the implicit solution is feasible"
        println("Residual implicit = $res")
        println("x0 = $x0_test")
        global all_good = false
    end
    if !isnothing(ind) && isnothing(uref)
        @warn "The Implicit solution is infeasible, while the Explicit solution is feasible"
        global all_good = false
    end
    if isnothing(ind) && isnothing(uref)
        println("Both explicit and implicit solution are infeasible")
    end
    if !isnothing(ind) && !isnothing(uref) && norm(uref - usol) > tol
        @warn "Explicit and implicit solutions are different"
        global all_good = false
    end


end
if all_good
    println("Explicit and implicit solutions are equal in all tested cases")
end

# Plot partitions

display(ParametricDAQP.plot_regions(sol))