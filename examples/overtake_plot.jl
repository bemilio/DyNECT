using Plots
using CSV
using DataFrames
include("overtake_utils.jl")
# Parameters

function plot_overtake()
    road_width = 10.0
    road_length = 1000.0
    vehicle_width = 2.
    vehicle_length = vehicle_width * 3.0 / 2.0  # Adjust as needed
    length_plotted_road_section = 8 * vehicle_length

    # Ellipse parameters

    # Import positions and velocities
    # Import the CSV and recreate p1, v1, l1, p2, v2, l2 variables
    println("Plotting results from DR...")
    pv_loaded = CSV.read("pos_vel_DataFrame.csv", DataFrame)
    
    p1 = pv_loaded.p1
    v1 = pv_loaded.v1
    l1 = pv_loaded.l1
    p2 = pv_loaded.p2
    v2 = pv_loaded.v2
    l2 = pv_loaded.l2
    dx_min = pv_loaded.dx_min[1]
    dl_min = pv_loaded.dl_min[1]
    # Make p1, p2 positive
    pmin = min(minimum(p1), minimum(p2), -length_plotted_road_section)
    if pmin < 0.
        p1 = p1 .- pmin
        p2 = p2 .- pmin
    end

    # Scale p by vehicle length and l by road width
    p1 *= vehicle_length
    l1 *= road_width/2
    p2 *= vehicle_length
    l2 *= road_width/2

    resolution = 10000
    x = zeros(resolution)
    y = Vector{Float64}(range(0., road_length, length=resolution))
    normal_vecs, tangent_vecs, angles = compute_normal_tangent(x, y)

    # pos_vehicle_1, tangent_vehicle_1 = pos_in_road_to_abs_position(x, y, tangent_vecs, normal_vecs, p1, l1, road_width)
    # pos_vehicle_2, tangent_vehicle_2 = pos_in_road_to_abs_position(x, y, tangent_vecs, normal_vecs, p2, l2, road_width)
    # angle_vehicle_1 = atan.(tangent_vehicle_1[:, 2], tangent_vehicle_1[:, 1])
    # angle_vehicle_2 = atan.(tangent_vehicle_2[:, 2], tangent_vehicle_2[:, 1])

    anim = @animate for t in 1:length(p1)
        plt = plot(; aspect_ratio=:equal,
            xlim=(-2 * road_width, 2 * road_width),
            ylim=(-length_plotted_road_section + p1[t], length_plotted_road_section + p1[t]),
            xlabel=raw"$X\ (\mathrm{m})$",
            ylabel=raw"$Y\ (\mathrm{m})$",
            dpi=300
        )
        # plt = plot(; aspect_ratio=:equal)
        drawRoad!(plt, x, y, normal_vecs, road_width, :grey)
        println("p1 = $(p1[t])")
        println("l1 = $(l1[t])")
        scatter!(plt, [l1[t]], [p1[t]], color=:blue, markersize=2)
        drawRectangle!(plt, l1[t], p1[t], pi/2, vehicle_length, vehicle_width, :blue)
        scatter!(plt, [l2[t]], [p2[t]], color=:red, markersize=2)
        drawRectangle!(plt, l2[t], p2[t], pi/2, vehicle_length, vehicle_width, :red)
        drawEllipse!(plt, l1[t], p1[t], (road_width/2) * dl_min, vehicle_length * dx_min, :orange)

    end


    gif(anim, "vehicle_animation.gif", fps=10)
    mp4(anim, "vehicle_animation.mp4", fps=10)

end
plot_overtake()



