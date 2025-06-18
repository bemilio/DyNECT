using Revise
using DyNECT
using BlockDiagonals
using ParametricDAQP
using Plots
using LinearAlgebra
using Random
Random.seed!(1)
using MAT


function find_CR(x0::Vector{Float64}, sol; eps_gap=1e-6)
    #Find the critical region for x0
    contained_in = Int64[]
    for (ind, region) in enumerate(sol.CRs)
        try
            violation = minimum(region.bth - region.Ath' * x0)
            if (violation >= -eps_gap)
                push!(contained_in, ind)
            end
        catch err
            println("Error: ", err)
        end
    end
    return isempty(contained_in) ? nothing : contained_in[1]
end

function first_input_of_sequence(u::Vector{Float64}, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    u_first = Vector{Vector{Float64}}(undef, N)
    start_row = 1
    for i = 1:N
        u_first[i] = u[start_row:start_row+nu[i]-1]
        start_row += nu[i] * T_hor
    end
    return u_first
end

function MPC_control(x0::Vector{Float64}, sol, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    ind = find_CR(x0, sol)
    # Extract primal solution
    u = sol.CRs[ind].z' * [x0; 1]
    # Extract first input of each agent's sequence
    return first_input_of_sequence(u, nu, N, T_hor)
end

# 2-vehicles platooning
# states:
# 1) v1 - v_ref 
# 2) l1 - lref (lateral position)
# 3) p1 - p2 - pref (longitudinal pos)
# 4) v1 - v2 
# 5) l2 - lref
#
# Inputs:
# 1) a1
# 2) alpha1 (angle)
# 3) a2 
# 4) alpha2

T_hor = 4
tau = 0.1 #sampling time
v_ref = 20. # reference speed 
v_min = 10.
v_max = 30.
d_ref = 10. # reference distance
d_min = 5. # safety distance
a_max = 10. # max acceleration
a_min = -10.
l_ref = 1.
angle_max = pi / 8
angle_min = -pi / 8

# A = [1. 0 0;
#     0 1. tau;
#     0 0 1]
# B = [[
#         tau;
#         tau^2/2;
#         tau;;
#     ],
#     [
#         0;
#         -tau^2/2;
#         -tau;;
#     ]]

# # Objectives
# Q = [Matrix{Float64}(BlockDiagonal([[1.;;], zeros(2, 2)])),
#     Matrix{Float64}(BlockDiagonal([[0.;;], Matrix{Float64}(I(2))]))]
# R = [3 * [1.;;],
#     2 * [1.;;]]
# P = Q

# C_x = [0 -1. 0; # Safety distance
#     1. 0 0; # top speed agent 1
#     -1. 0 0; # min speed agent 1
#     1. 0 1.; # top speed agent 2
#     -1 0 -1] # min speed agent 2

# b_x = [d_ref - d_min;
#     v_max - v_ref;
#     v_ref - v_min;
#     v_max - v_ref;
#     v_ref - v_min]

# C_loc = [[1.; -1.;;], [1.; -1.;;]]

# b_loc = [[a_max; -a_min], [a_max; -a_min]]

# C_u = [zeros(0, 2), zeros(0, 2)]
# b_u = zeros(0)

A = [1. 0 0 0 0;
    0 1. 0 0 0;
    0 0 1. tau 0;
    0 0 0 1 0;
    0 0 0 0 1]
B = [[tau 0;
        0 tau*v_ref;
        tau^2/2 0;
        tau 0;
        0 0],
    [0 0;
        0 0;
        -tau^2/2 0;
        -tau 0;
        0 tau*v_ref]]


# Objectives
Q = [Matrix{Float64}(BlockDiagonal([Matrix{Float64}(I(2)), zeros(3, 3)])),
    Matrix{Float64}(BlockDiagonal([zeros(2, 2), Matrix{Float64}(I(3))]))]
R = [3 * Matrix{Float64}(I(2)),
    2 * Matrix{Float64}(I(2))]
P = Q

C_x = [0 0 -1. 0 0; # Safety distance
    1. 0 0 0 0; # top speed agent 1
    -1. 0 0 0 0; # min speed agent 1
    1. 0 0 1. 0; # top speed agent 2
    -1 0 0 -1 0] # min speed agent 2

b_x = [d_ref - d_min;
    v_max - v_ref;
    v_ref - v_min;
    v_max - v_ref;
    v_ref - v_min]

C_loc = [[1. 0;
        -1. 0;
        0 1;
        0 -1],
    [1. 0;
        -1. 0;
        0 1;
        0 -1]]

b_loc = [[a_max;
        -a_min;
        angle_max;
        -angle_min],
    [a_max;
        -a_min;
        angle_max;
        -angle_min]]

C_u = [zeros(0, 2), zeros(0, 2)]
b_u = zeros(0)

# Range of initial state of MPC problem (VI parameter)
nx = size(A, 1)
# Theta = (A=C_x', b=b_x, ub=10. * ones(nx), lb=-10. * ones(nx)) ### This appears broken!!
Theta = (ub=5. * ones(nx), lb=-5. * ones(nx))

game = DyNEP(
    A=A,
    Bvec=B,
    Q=Q,
    R=R,
    C_x=C_x,
    b_x=b_x,
    C_loc_vec=C_loc,
    b_loc_vec=b_loc,
    C_u_vec=C_u,
    b_u=b_u)
P, K = DyNECT.solveOLNE(game)
game.P[:] = P[:]
opts = ParametricDAQP.Settings()
opts.early_stop = true
mpVI = generate_mpVI(game, T_hor)
sol, _ = ParametricDAQP.mpsolve(mpVI, Theta; opts)



## Test solution
tol = 10^(-5)

for i in 1:100
    # x0_test = [-0.397653367382937; -0.25652948068500336; -0.8298580073256627]
    x0_test = ones(nx) - 2 * rand(nx)
    ind = find_CR(x0_test, sol) # Find  CR corresponding to x0_test
    # Extract primal solution
    if !isnothing(ind)
        usol = sol.CRs[ind].z' * [x0_test; 1]
    end
    uref, _ = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x0_test; 1], mpVI.A, mpVI.B' * [x0_test; 1], tol=tol)
    if isnothing(ind) && !isnothing(uref)
        println("pause...")
        @warn "The explicit solution is infeasible, while the implicit solution is feasible"
    end
    if !isnothing(ind) && isnothing(uref)
        @warn "The Implicit solution is infeasible, while the Explicit solution is feasible"
    end
    if isnothing(ind) && isnothing(uref)
        println("Both explicit and implicit solution are infeasible")
    end
    if !isnothing(ind) && !isnothing(uref) && norm(uref - usol) > tol
        @warn "Explicit and implicit solutions are different"
    end
    if !isnothing(ind) && !isnothing(uref) && norm(uref - usol) <= tol
        println("Explicit and implicit solutions are equal")
    end
end



# Simulate implicit MPC 

x0 = ones(nx) - 2 * rand(nx)
T_sim = 200
x_imp = Vector{Vector{Float64}}(undef, T_sim)
u_imp = Vector{Vector{Float64}}(undef, T_sim)
x_imp[1] = x0
for t in 1:T_sim-1
    useq, _ = ParametricDAQP.AVIsolve(mpVI.H, mpVI.F' * [x_imp[t]; 1], mpVI.A, mpVI.B' * [x_imp[t]; 1])
    if !isnothing(useq)
        u_imp[t] = vcat(first_input_of_sequence(useq, game.nu, game.N, T_hor)...)
    end
    x_imp[t+1] = game.A * x_imp[t] + game.B * u_imp[t]
end

# Simulate explicit MPC
x_exp = Vector{Vector{Float64}}(undef, T_sim)
u_exp = Vector{Vector{Float64}}(undef, T_sim - 1)
x_exp[1] = x0
for t in 1:T_sim-1
    u_exp[t] = vcat(MPC_control(x_exp[t], sol, game.nu, game.N, T_hor)...)
    x_exp[t+1] = game.A * x_exp[t] + game.B * u_exp[t]
end
diff = norm(x_exp - x_imp)
println("Difference explicit-implicit MPC trajectory = $diff")


## Plot 

t = 1:T_sim

# Animation of car positions over time
p1 = Vector{Float64}(undef, T_sim)
v1 = Vector{Float64}(undef, T_sim)
l1 = Vector{Float64}(undef, T_sim)

p2 = Vector{Float64}(undef, T_sim)
v2 = Vector{Float64}(undef, T_sim)
l2 = Vector{Float64}(undef, T_sim)

anim = @animate for t in 1:T_sim-1
    # Car 1 position and lateral position
    v1[t] = x_exp[t][1] + v_ref
    if t == 1
        p1[t] = 0
        p1[t+1] = tau * v1[t]
    else
        p1[t+1] = p1[t] + tau * v1[t]  # position of car 1 (integral of velocity)
    end
    l1[t] = x_exp[t][2] + l_ref                 # lateral position of car 1

    # Car 2 position and lateral position
    p2[t] = -x_exp[t][3] + p1[t] - d_ref          # position of car 2
    l2[t] = x_exp[t][5] + l_ref                       # lateral position of car 2

    scatter([p1[t], p2[t]], [l1[t], l2[t]],
        xlim=(-20, 20), ylim=(-5, 5),
        xlabel="Longitudinal Position", ylabel="Lateral Position",
        legend=false, title="Car positions at t = $t",
        markershape=:rect, markersize=8, color=[:blue :red])

    annotate!(p1[t+1], l1[t], text("Car 1", :left, 10))
    annotate!(p2[t], l2[t], text("Car 2", :right, 10))
end

matwrite("data.mat", Dict("x_exp" => x_exp, "u_exp" => u_exp, "p1" => p1, "p2" => p2, "v1" => v1, "v2" => v2, "l1" => l1, "l2" => l2))

# gif(anim, "car_positions.gif", fps=5)

# Time vector
# t = 1:T_sim-1

# # Create subplots
# plot1 = plot(t, p1, label="Car 1", xlabel="Time", ylabel="Longitudinal Position", title="Position")
# plot!(plot1, t, p2, label="Car 2")

# plot2 = plot(t, v1, label="Car 1", xlabel="Time", ylabel="Velocity", title="Velocity")
# plot!(plot2, t, v2, label="Car 2")

# plot3 = plot(t, l1, label="Car 1", xlabel="Time", ylabel="Lateral Position", title="Lateral Position")
# plot!(plot3, t, l2, label="Car 2")

# # Combine plots into a layout
# final_plot = plot(plot1, plot2, plot3, layout=(3, 1), legend=:topright)

# # Save or display
# # savefig(final_plot, "car_trajectories.png")
# display(final_plot)


# Plot for implicit MPC
# plt1 = plot(layout=(2, 1), size=(800, 600), title="Implicit MPC")
# plot!(plt1[1], t, hcat(x_imp...)', xlabel="Time", ylabel="States", label=["x₁" "x₂" "x₃"])
# plot!(plt1[2], t, hcat(u_imp...)', xlabel="Time", ylabel="Inputs", label=["u₁" "u₂"])

# # Plot for explicit MPC
# plt2 = plot(layout=(2, 1), size=(800, 600), title="Explicit MPC")
# plot!(plt2[1], t, hcat(x_exp...)', xlabel="Time", ylabel="States", label=["x₁" "x₂" "x₃"])
# plot!(plt2[2], t, hcat(u_exp...)', xlabel="Time", ylabel="Inputs", label=["u₁" "u₂"])

# display(plt1)
# display(plt2)


