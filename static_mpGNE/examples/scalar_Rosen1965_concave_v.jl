# scalar_Rosen1965_concave.jl — Rosen (1965) scalar concave game
# 2-player scalar game, non-monotone operator
# Fast pipeline: numeric params defined upfront

include("../compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
include("../Solver.jl")

# Define game structure
game = GameBuilder(N=2)
@player game 1 n=1
@player game 2 n=1

@cost game 1  x1^2 - x1*x2 - x1
@cost game 2  -2*x2^2 - x1*x2 - x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  x1 + x2 >= 1

# Assemble and solvej
mpvi = build_mpvi(game)
#sol  = solve_gne(mpvi)
#show_solution(sol, mpvi)