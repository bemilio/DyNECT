include("../compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
include("../Solver.jl")
using Plots
using LinearAlgebra
using ParametricDAQP

t_start = time()
println("=== Test Name: Vector game 2-player ===")

Q1 = [2.0 0.0; 0.0 1.0]
Q2 = [3.0 0.0; 0.0 2.0]
R  = [1.0 0.0; 0.0 1.0]
c1 = [-1.0, 0.0]
c2 = [0.0, -1.0]

game = GameBuilder(N=2)
@player game 1 n=2
@player game 2 n=2

@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  [x_1_1 + x_1_2 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0]

mpvi = build_mpvi(game)
# show_mpvi(mpvi)

θub = [3.0, 2.0]
θlb = [0.0, 0.0]
mpvi_dynect = to_dynect_mpAVI(mpvi)
mpvi_dynect = DyNECT.setParameterSpace(mpvi_dynect,
    C = [Matrix{Float64}(I, 2, 2); -Matrix{Float64}(I, 2, 2)],
    d = [θub; -θlb],
    ub = θub, lb = θlb)
sol = CommonSolve.solve(mpvi_dynect, DyNECT.ParametricDAQPSolver)

filter_gne_crs!(sol, mpvi)

println("total: $(round(time() - t_start, digits=2))s")