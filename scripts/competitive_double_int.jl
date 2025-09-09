using Revise
using DyNECT
using BlockDiagonals
using ParametricDAQP
using LinearAlgebra
using Pkg
using Infiltrator
using BenchmarkTools
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
R = [[[5.;;], [0.;;]], [[0.;;], [8.;;]]]

C_x = [Matrix{Float64}(I(2)); -Matrix{Float64}(I(2))]

b_x = [5 * ones(2); 5 * ones(2)]

C_loc = [[1.; -1.;;], [1.; -1.;;]]

b_loc = [[u_max[1]; -u_min[1]], [u_max[2]; -u_min[2]]]

C_u = [zeros(0, 1), zeros(0, 1)]
b_u = zeros(0)

# Range of initial states 
nx = size(A, 1)
Θ = (A=C_x', b=b_x, ub=6. * ones(nx), lb=-6. * ones(nx))

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
sol, _ = ParametricDAQP.mpsolve(mpVI, Θ)

## Test solution
tol = 10^(-5)
all_good = true
for i in 1:100
    x0_test = 2 * ones(game.nx) - rand(game.nx)
    # Check if x0_test is in Θ
    if any(Θ.A' * x0_test .> Θ.b) || any(x0_test .> Θ.ub) || any(x0_test .< Θ.lb)
        @warn "x0_test is outside of Θ"
        continue
    end
    usol = evaluate_solution(sol, x0_test)
    # θ_normalized = (x0_test - sol.translation) .* sol.scaling
    # all_CRs_indexes = ParametricDAQP.pointlocation(θ_normalized, sol.CRs)
    # if !isempty(all_CRs_indexes)
    #     CR = sol.CRs[all_CRs_indexes[1]]
    #     usol = CR.z' * [θ_normalized; 1]
    # end
    # @infiltrate
    uref, res, solved_implicit = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x0_test; 1], mpVI.A, mpVI.B' * [x0_test; 1], tol=tol)
    if isnothing(usol) && solved_implicit
        @warn "The explicit solution is infeasible, while the implicit solution is feasible"
        println("Residual implicit = $res")
        println("x0 = $x0_test")
        global all_good = false
    end
    if !isnothing(usol) && !solved_implicit
        @warn "The Implicit solution is infeasible, while the Explicit solution is feasible"
        global all_good = false
    end
    if isnothing(usol) && !solved_implicit
        println("Both explicit and implicit solution are infeasible")
    end
    if !isnothing(usol) && solved_implicit && norm(uref - usol) > tol
        @warn "Explicit and implicit solutions are different"
        global all_good = false
    end


end
if all_good
    println("Explicit and implicit solutions are equal in all tested cases")
end

# Plot partitions

display(ParametricDAQP.plot_regions(sol))