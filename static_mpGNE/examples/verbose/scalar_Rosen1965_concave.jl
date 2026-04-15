include("../../verbose/src/Static_mpGNE.jl")
using .Static_mpGNE
include("../../Solver.jl")

println("=== Test Name: Rosen scalar game ===")

## Game definition
game = GameBuilder(N=2)

@player game 1 n=1
@player game 2 n=1

@cost game 1  0.5*x1^2 - x1*x2
@cost game 2  0.5*x2^2 + x1*x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  x1 + x2 >= 1

## Display
println()
#validate_game(game)
#show_operator(game)
#show_shared_constraints(game)
#show_local_constraints(game)
#show_feasible_set(game)
#show_parametric_constraints(game)
#show_theta_set(game)

## Build and solve
println()
mpvi = build_mpvi(game)
show_mpvi(mpvi)

println()
sol = solve_gne(mpvi)
show_solution(sol, mpvi)
show_pwa_map(sol)