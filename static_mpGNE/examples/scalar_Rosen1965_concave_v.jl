include("../compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
include("../Solver.jl")

game = GameBuilder(N=2)
@player game 1 n=1
@player game 2 n=1

@cost game 1  0.5*x1^2 - x1*x2
@cost game 2  0.5*x2^2 + x1*x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  x1 + x2 >= 1

mpvi = build_mpvi(game)
show_mpvi(mpvi)

lb, ub = _extract_theta_bounds(mpvi)
println("lb = $lb,  ub = $ub")

sol = solve_gne(mpvi)
show_solution(sol, mpvi)