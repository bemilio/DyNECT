# vector_game.jl — 2-player vector quadratic game
# Non-diagonal shared constraint, numeric params upfront
# Fast pipeline

include("../compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
include("../Solver.jl")

# Define game parameters
Q1 = [2.0 0.0; 0.0 1.0]
Q2 = [3.0 0.0; 0.0 2.0]
R  = [1.0 0.0; 0.0 1.0]
c1 = [-1.0, 0.0]
c2 = [0.0, -1.0]

# Define game structure
game = GameBuilder(N=2)
@player game 1 n=2
@player game 2 n=2

@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  [x_1_1 + x_1_2 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0]

# Assemble and solve
mpvi = build_mpvi(game)
sol  = solve_gne(mpvi)
show_solution(sol, mpvi)