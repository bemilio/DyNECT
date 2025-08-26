using Plots
using CSV
using DataFrames
include("overtake_utils.jl")
# Parameters

road_width = 10.0
# Ellipse parameters
a = 100.0   # semi-major axis
b = 70.0   # semi-minor axis

# Import positions and velocities
# Import the CSV and recreate p1, v1, l1, p2, v2, l2 variables
if solve_explicit
    println("Plotting results from explicit MPC...")
    pv_loaded = CSV.read("pos_vel_explicit_results.csv", DataFrame)
else
    println("Plotting results from iterative-solution MPC...")
    pv_loaded = CSV.read("pos_vel_implicit_results.csv", DataFrame)
end
p1 = pv_loaded.p1
v1 = pv_loaded.v1
l1 = pv_loaded.l1
p2 = pv_loaded.p2
v2 = pv_loaded.v2
l2 = pv_loaded.l2
# Make p1, p2 positive
pmin = min(minimum(p1), minimum(p2))
if pmin < 0.
    p1 = p1 .- pmin
    p2 = p2 .- pmin
end

θ = range(0, 2π, length=200)
# Parametric equations for the outer ellipse
x = a * cos.(θ)
y = b * sin.(θ)

normal_vecs, tangent_vecs = compute_normal_tangent(x, y)

pos_vehicle_1, tangent_vehicle_1 = pos_in_road_to_abs_position(x, y, tangent_vecs, normal_vecs, p1, l1, road_width)
pos_vehicle_2, tangent_vehicle_2 = pos_in_road_to_abs_position(x, y, tangent_vecs, normal_vecs, p2, l2, road_width)
angle_vehicle_1 = atan.(tangent_vehicle_1[:, 2], tangent_vehicle_1[:, 1])
angle_vehicle_2 = atan.(tangent_vehicle_2[:, 2], tangent_vehicle_2[:, 1])

plt = plot()  # Create a new plot object
drawRoad!(plt, x, y, normal_vecs, road_width, :grey)

# Parameters for vehicle rectangle
vehicle_width = 3
vehicle_length = vehicle_width * 3.0 / 2.0  # Adjust as needed

anim = @animate for t in 1:length(p1)
    plt = plot()
    drawRoad!(plt, x, y, normal_vecs, road_width, :grey)
    drawRectangle!(plt, pos_vehicle_1[t, 1], pos_vehicle_1[t, 2], angle_vehicle_1[t], vehicle_length, vehicle_width, :blue)
    drawRectangle!(plt, pos_vehicle_2[t, 1], pos_vehicle_2[t, 2], angle_vehicle_2[t], vehicle_length, vehicle_width, :red)

end

gif(anim, "vehicle_animation.gif", fps=10)




display(plt)
readline()


