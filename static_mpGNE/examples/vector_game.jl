include("../src/Static_mpGNE.jl")
using .Static_mpGNE
include("../src/Solver.jl")   

println("=== Test Name: Vector game ===")
println()
## User-defined game
game = GameBuilder(N=2)

@player game 1 n=2
@player game 2 n=2

@param game Q1 dims=(2,2)
@param game Q2 dims=(2,2)
@param game R  dims=(2,2)
@param game c1 dims=(2,)
@param game c2 dims=(2,)

@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
#@constraint game  [x_1_1 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0]
@constraint game  [x_1_1 + x_1_2 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0] #non diagonal case

# optionally assign numerical values here:
assign_params!(game, Dict(
    :Q1 => [2.0 0.0; 0.0 1.0],
    :Q2 => [3.0 0.0; 0.0 2.0],
    :R  => [1.0 0.0; 0.0 1.0],
    :c1 => [-1.0, 0.0],
    :c2 => [0.0, -1.0]
))

## Display
println()
#validate_game(game) # testing 
#show_operator(game)
#show_shared_constraints(game)
#show_local_constraints(game)
#show_feasible_set(game)
#show_parametric_constraints(game)
#show_theta_set(game)


mpvi = build_mpvi(game)
show_mpvi(mpvi)

## Solve
sol = solve_gne(mpvi)

# === Evaluate — three presentation strategies (pick one) ===

# Option A: automatic — show_solution samples corners of Θ
show_solution(sol, mpvi)

# Option B: explicit betas passed in
# show_solution(sol, mpvi, betas=[[1.0, 1.0], [0.5, 0.5], [2.0, 1.5]])

# Option C: scenario dict (if physical scenarios)
# scenario = load_scenario("scenarios/energy_2player.toml")
# show_solution(sol, mpvi, betas=scenario[:betas])