include("../src/Static_mpGNE.jl")
using .Static_mpGNE
include("../src/Solver.jl")   

println("=== Test Name: Rosen scalar game ===")

## User-defined game
game = GameBuilder(N=2)

@player game 1 n=1 #xᵢ ∈ ℝ, thus n=1
@player game 2 n=1

@cost game 1  x1^2 - x1*x2 - x1
@cost game 2  -2*x2^2 - x1*x2 - x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  x1 + x2 >= 1

## Display
println()
validate_game(game) 
show_operator(game)
show_shared_constraints(game)
show_local_constraints(game)
show_feasible_set(game)
show_parametric_constraints(game)
show_theta_set(game)

println()
mpvi = build_mpvi(game)
show_mpvi(mpvi)

println()
#sol = solve_gne(mpvi) 
#show_solution(sol, mpvi)