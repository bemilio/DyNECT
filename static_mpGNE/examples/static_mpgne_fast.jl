include("Fast_mpGNE.jl")
using .Fast_mpGNE
include("../Solver.jl")

#Define game
Q1 = [2.0 0.0; 0.0 1.0]
Q2 = [3.0 0.0; 0.0 2.0]
Q3 = [4.0 0.0; 0.0 3.0]
R  = [1.0 0.0; 0.0 1.0]
c1 = [-1.0, 0.0]
c2 = [0.0, -1.0]
c3 = [0.0, 0.0]

game = GameBuilder(N=3)
@player game 1 n=2
@player game 2 n=2
@player game 3 n=2
@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2
@cost game 3  0.5*x3'*Q3*x3 + x3'*R'*x1 + c3'*x3
@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  -x3 <= 0
@constraint game  [x_1_1 + x_1_2 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0]
@constraint game  [x_1_1 + x_2_1 + x_3_1, x_1_2 + x_2_2 + x_3_2] <= [5.0, 3.0]

#Solving:
mpvi = build_mpvi(game)
sol = solve_gne(mpvi)
show_solution(sol, mpvi)
