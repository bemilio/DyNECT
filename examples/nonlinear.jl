using DyNECT
# using DynamicalSystems
# using DifferentialEquations
# using SciMLSensitivity
using Zygote
using DAQPBase
# using BenchmarkTools
using LinearAlgebra
# using DynamicalSystems
# using Plots
# using CSV
# using DataFrames
# using CommonSolve
using DynamicalSystems
using Random
Random.seed!(1234)

include("overtake_utils.jl")

# Parameters
T_sim = 200 # Simulation length
T_hor = 3
Δt = 0.1 #sampling time
v_ref = [20., 25.] # reference speed for each agent 
v_min = 10.
v_max = 35.
d_overtake = 10. # Distance at which overtake is initiated
a_max = 5. # max acceleration
a_min = -5.
angle_max = pi / 32
angle_min = -pi / 32
l_ref = [0.5, -0.5] #reference lateral position for normal and overtake lane 

dx_min = 5. # safety longitudinal distance
dl_min = 0.7 # safety lateral distance

# Fixed values
N = 2
nu = [2,2]
nx = 6

## Initialization

# State:
x0 = [0., (v_max + v_min) / 2, l_ref[1], -15., (v_max + v_min) / 2, l_ref[1]]
u0=[zeros(2), zeros(2)]

# Dynamics
function unicycle_dynamics(x,u)
    return [
        x[2] * cos(u[2]),
        u[1],
        x[2] * sin(u[2])
    ]
end
function f(x, u1, u2) # discretized unicycles
    dx = vcat(unicycle_dynamics(x[1:3],u1), unicycle_dynamics(x[4:6],u2))
    return x .+ Δt .* dx
end

# Continuous-time simulator
function unicycle_dynamics!(dx::AbstractVector, x::AbstractVector, u::AbstractVector, t::Float64) # Compatible function with DynamicalSystems.jl
    dx .= unicycle_dynamics(x,u)
    return nothing
end
agent1 = CoupledODEs(unicycle_dynamics!, x0[1:3], zeros(2)) #initialize system with zero input
agent2 = CoupledODEs(unicycle_dynamics!, x0[4:6], zeros(2))

# Define objectives
# J = ‖v₁ - vᵈᵉˢ‖² + ‖l₁ - lᵈᵉˢ‖² + 5 * a² + 20 * γ²
J1_platoon(x, u1, u2) = (x[2] - v_ref[1])^2 + (x[3] - l_ref[1])^2 + 5 * u1[1]^2 + 50 * u1[2]^2
J2_platoon(x, u1, u2) = (x[5] - v_ref[2])^2 + (x[6] - l_ref[1])^2 + 5 * u2[1]^2 + 50 * u2[2]^2
J_platoon = [J1_platoon, J2_platoon]

J1_overtake(x, u1, u2) = (x[2] - v_ref[1])^2 + (x[3] - l_ref[1])^2 + 5 * u1[1]^2 + 50 * u1[2]^2
J2_overtake(x, u1, u2) = (x[5] - v_ref[2])^2 + (x[6] - l_ref[2])^2 + 5 * u2[1]^2 + 50 * u2[2]^2
J_overtake = [J1_overtake, J2_overtake]

# State constraints
gx(x) = [
    1 - ( (x[1] - x[4])^2 / dx_min^2 + (x[3] - x[6])^2 / dl_min^2 ); # Safety distance: Ellipse
    v_min - x[2];  # min speed agent 1 
    x[2] - v_max;  # Max speed agent 1
    v_min - x[5];  # min speed agent 2
    x[5] - v_max;  # Max speed agent 2
]

gloc1(u1) = [
        a_min - u1[1];
        u1[1] - a_max;
        angle_min - u1[2];
        u1[2] - angle_max ] # Input constraints agent 1
gloc2(u2) = [
        a_min - u2[1];
        u2[1] - a_max;
        angle_min - u2[2];
        u2[2] - angle_max ] # Input constraints agent 2
gloc = [gloc1, gloc2]

gu(u1,u2) = -1.0 # Dummy

# Define games
game = Dict(
    :Platoon  => DyNECT.DynGame(f, J_platoon,  gx, gu, gloc, nx, nu, 5, 1, [4,4], N),
    :Overtake => DyNECT.DynGame(f, J_overtake, gx, gu, gloc, nx, nu, 5, 1, [4,4], N),
)


#### Simulations ####

pv0 = [0., (v_max + v_min) / 2, l_ref[1], -15., (v_max + v_min) / 2, l_ref[1]]

println("Simulating...")
# Initialize storing vectors
# x = zeros(6, T_sim) # position and velocity of two unicycles
# u = zeros(4, T_sim) # input of two unicycles
# elapsed_time = zeros(T_sim - 1)
# residual = zeros(T_sim - 1)
# Initialize values
xt = x0 # state at time T
xsim = [zeros(nx) for t in 1:Tsim] # stores state along simulation
xsim[0] .= x0
useq = [[zeros(nui) for nui in nu] for t in 1:T_hor] # Stores MPC-predicted sequence of inputs
ut = [zeros(nui) for nui in nu] # input at time t
case = :Platoon

# Main loop
for t in 1:T_sim
    global case
    # avi = DyNECT.AVI(mpVIs[Int(case)], x)
    lq_game = DyNECT.LQapprox(game[case], useq, xt, T_hor) # state/input are δx,δu: deviation from reference input/state
    mpavi = DyNECT.DynLQGame2mpAVI(lq_game)
    avi = DyNECT.AVI(mpavi, zeros(nx)) 
    sol = @timed DAQPBase.avi(avi.H, avi.f, avi.A, avi.b, -Inf.*ones(length(avi.b)))
    sol.time
    exitflag = sol.value[3]
    exitflag <0 && println("infeasible!, t=$t")
    exitflag >0 && println("OK, t=$t")

    δuseq = DyNECT.arrange_vector_as_time_seq(sol.value[1], nu, N, T_hor)
    # Apply bias
    useq .= useq .+ δuseq
    # Apply control
    ut .= useq[1] + δuseq[1]
    # Shift control sequence
    useq[1:end-1] = useq[2:end]
    useq[end] = [zeros(nui) for nui in nu] 
    # TODO: compute residual
    # sol_res[t] = DyNECT.compute_residual

    # Evolve dynamic system
    set_parameters!(agent1, ut[1])
    set_parameters!(agent2, ut[2])
    step!(agent1, Δt, true) # progress for Δt units of time
    step!(agent2, Δt, true) # progress for Δt units of time
    xt[1:3] = current_state(agent1)
    xt[4:6] = current_state(agent2)
    xsim[t] .= xt
    println("[t=$t] xt= $xt")

    # Overtake logic
    d = xt[1] - xt[4]
    if d < d_overtake && d > 0. && case == :Platoon
        println("Initiate overtake!")
        case = :Overtake
    elseif d < 0 && case == :Overtake # Overtake completed
        println("Overtake complete")
        case = :Platoon
    end

end


# Prepare DataFrame
pv_df = DataFrame(time=collect(0:T_sim-1) .* Δt,
    p1=vstack([xsim[t][1] for t in 1:T_sim]...),
    v1=vstack([xsim[t][2] for t in 1:T_sim]...),
    l1=vstack([xsim[t][3] for t in 1:T_sim]...),
    p2=vstack([xsim[t][4] for t in 1:T_sim]...),
    v2=vstack([xsim[t][5] for t in 1:T_sim]...),
    l2=vstack([xsim[t][6] for t in 1:T_sim]...)
    )
CSV.write("pos_vel_DataFrame.csv", pv_df)



