using DyNECT
# using DynamicalSystems
# using DifferentialEquations
# using SciMLSensitivity
using Zygote

using Random
Random.seed!(1234)


function unicycle_dynamics(x,u)
    return [
        x[2] * cos(u[2]),
        u[1],
        x[2] * sin(u[2])
    ]
end

T_hor = 4
Δt = 0.1 #sampling time
v_ref = [20., 25.] # reference speed for each agent 
v_min = 10.
v_max = 35.
d_min = 5. # safety distance
a_max = 5. # max acceleration
a_min = -5.
angle_max = pi / 16
angle_min = -pi / 16
l_ref = [0.5, -0.5] #reference lateral position for normal and overtake lane 

x0 = [0., (v_max + v_min) / 2, l_ref[1], -15., (v_max + v_min) / 2, l_ref[1]]
# x0 = [0., (v_max + v_min) / 2, 0.]
u0=[0.0, 0.0]

# Dynamics: x = [p₁, v₁, l₁, p₂, v₂, l₂]

function f(x, u1, u2) # discretized unicycles
    dx = vcat(unicycle_dynamics(x[1:3],u1), unicycle_dynamics(x[4:6],u2))
    return x .+ Δt .* dx
end

# J = ‖v₁ - vᵈᵉˢ‖² + ‖l₁ - lᵈᵉˢ‖² + 5 * a² + 20 * γ²
J1(x, u1, u2) = (x[2] - v_ref[1])^2 + (x[3] - l_ref[1])^2 + 5 * u1[1]^2 + 20 * u1[2]^2
J2(x, u1, u2) = (x[5] - v_ref[2])^2 + (x[6] - l_ref[1])^2 + 5 * u2[1]^2 + 20 * u2[2]^2
J = [J1, J2]

gx(x) = [
    d_min - (x[1] - x[4])^2; # Safety distance
    x[2] - v_min;  # min speed agent 1 
    v_max - x[2];  # Max speed agent 1
    x[5] - v_min;  # min speed agent 1 
    v_max - x[5];  # Max speed agent 1
]

gloc1(u1) = [
        u1[1] - a_min;
        a_max - u1[1];
        u1[2] - angle_min;
        angle_max - u1[2] ] # Input constraints agent 1
gloc2(u2) = [
        u2[1] - a_min;
        a_max - u2[1];
        u2[2] - angle_min;
        angle_max - u2[2] ] # Input constraints agent 2
gloc = [gloc1, gloc2]

gu(u1,u2) = -1.0 # Dummy

game = DyNECT.DynGame(f, J, gx, gu, gloc, 6, [2,2], 5, 1, [4,4], 2)

u = [[ones(2), ones(2)] for _ in 1:T_hor] 

x_next = f(x0,u[1]...)

lq_game = DyNECT.LQapprox(game, u, x0, T_hor)

mpavi = DyNECT.DynLQGame2mpAVI(lq_game)

avi = DyNECT.AVI(mpavi, x0)