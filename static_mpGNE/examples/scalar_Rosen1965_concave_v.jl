isdefined(Main, :Fast_mpGNE) || include("../compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
isdefined(Main, :solve_gne) || include("../Solver.jl")
using Plots
using ParametricDAQP
using CommonSolve

t_start = time()
game = Fast_mpGNE.GameBuilder(N=2)
@player game 1 n=1
@player game 2 n=1
@cost game 1  0.5*x1^2 - x1*x2
@cost game 2  x2^2 + x1*x2
@constraint game  x1 + x2 >= 1

mpvi = build_mpvi(game)
show_mpvi(mpvi)

θub = [5.0]
θlb = -[5.0]
mpvi_dynect = to_dynect_mpAVI(mpvi)
mpvi_dynect = DyNECT.setParameterSpace(mpvi_dynect, C=[1.0;-1.0;;], d=[θub;-θlb], ub=θub, lb=θlb)

result = solve_gne_both(mpvi_dynect, mpvi)

plot_gne_solution(result.unfiltered, θlb, θub, "Unfiltered")
plot_gne_solution(result.filtered, θlb, θub, "Filtered")

println("total: $(round(time() - t_start, digits=2))s")
readline()