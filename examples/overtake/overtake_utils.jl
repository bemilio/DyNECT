@enum Case begin
    Platooning
    BeginOvertake
    PerformOvertake
    CompleteOvertake
    NormalOperation
end

# Unicycle model
function unicycle!(dx, x, u, t)
    # x = (pos, vel, lat_pos)
    # u = (acc, steer_angle)
    dx[1] = x[2] * cos(u[2])
    dx[2] = u[1]
    dx[3] = x[2] * sin(u[2])
    return nothing
end


function state_to_posvel(x, case::Case)
    # Switch/case over all possible cases of Case
    if case == Platooning
        p1 = 0.0 # Leading agent is the reference position
        v1 = x[1] + v_ref[1]
        l1 = x[2] + l_ref[1] # normal lane
        p2 = x[3] + p1 - d_ref
        v2 = x[4] + v1
        l2 = x[5] + l_ref[1] # normal lane
    elseif case == BeginOvertake
        # TODO
        p1 = 0.0 # Leading agent is the reference position
        v1 = x[1] + v_ref
        l1 = x[2] + 0.0 # assuming lref = 0
        p2 = x[3] + p1 - d_ref
        v2 = x[4] + v1
        l2 = x[5] + 0.0 # assuming lref = 0
    elseif case == PerformOvertake
        # TODO 
        p1 = 0.0 # Leading agent is the reference position
        v1 = x[1] + v_ref
        l1 = x[2] + 0.0 # assuming lref = 0
        p2 = x[3] + p1 - d_ref
        v2 = x[4] + v1
        l2 = x[5] + 0.0 # assuming lref = 0

    elseif case == CompleteOvertake
        # TODO
        p1 = 0.0 # Leading agent is the reference position
        v1 = x[1] + v_ref
        l1 = x[2] + 0.0 # assuming lref = 0
        p2 = x[3] + p1 - d_ref
        v2 = x[4] + v1
        l2 = x[5] + 0.0 # assuming lref = 0

    elseif case == NormalOperation
        # TODO
        p1 = 0.0 # Leading agent is the reference position
        v1 = x[1] + v_ref
        l1 = x[2] + 0.0 # assuming lref = 0
        p2 = x[3] + p1 - d_ref
        v2 = x[4] + v1
        l2 = x[5] + 0.0 # assuming lref = 0

    else
        error("Unknown control_case")
    end
    posvel = [p1, v1, l1, p2, v2, l2]
    return posvel
end

function posvel_to_state(posvel, case)
    #posvel = (pos_1, vel_1, lat_pos_1, pos_2, vel_2, lat_pos_2)
    if case == Platooning
        # 1) v1 - v_ref_1
        # 2) l1 - lref (lateral position)
        # 3) p2 - p1 + d_ref (longitudinal pos)
        # 4) v2 - v1   /// The second car ignores its reference velocity and just follows the leading vehicle
        # 5) l2 - lref
        x = zeros(5)
        x[1] = posvel[2] - v_ref[1]
        x[2] = posvel[3] - l_ref[1] # normal lane
        x[3] = posvel[4] - posvel[1] + d_ref
        x[4] = posvel[5] - posvel[2]
        x[5] = posvel[6] - l_ref[1] # normal lane
    elseif case == BeginOvertake
        # Placeholder for BeginOvertake
    elseif case == PerformOvertake
        # Placeholder for PerformOvertake
    elseif case == CompleteOvertake
        # Placeholder for CompleteOvertake
    elseif case == NormalOperation
        # Placeholder for NormalOperation
    else
        error("posvel_to_state not implemented for this control_case")
    end
    return x
end

### Plotting utilities ###

function compute_normal_tangent(x, y)

    dx = diff(x)
    dy = diff(y)

    angles = atan.(dy, dx)

    # Normal vectors
    normal_vecs = hcat(-dy, dx)
    normal_vecs = normal_vecs ./ sqrt.(sum(normal_vecs .^ 2, dims=2))

    # Tangent vectors
    tangent_vecs = hcat(dx, dy)
    tangent_vecs = tangent_vecs ./ sqrt.(sum(tangent_vecs .^ 2, dims=2))

    # Repeat last vector to match the dimension of x, y 
    normal_vecs = [normal_vecs; normal_vecs[end, :]']
    tangent_vecs = [tangent_vecs; tangent_vecs[end, :]']
    angles = [angles; angles[end]]

    return normal_vecs, tangent_vecs, angles
end

function drawRoad!(plt, x, y, normal_vecs, width::Real, color)
    # Extend road points to create a filled polygon
    leftEdge = hcat(x, y) .+ (width / 2.) * normal_vecs
    rightEdge = hcat(x, y) .- (width / 2.) * normal_vecs
    roadPatch = vcat(leftEdge, reverse(rightEdge, dims=1))

    # Draw the filled road patch
    plot!(plt, roadPatch[:, 1], roadPatch[:, 2], fill=(true, color), linecolor=:transparent, legend=false)
    # Add dashed centerline
    dash_length = width * 0.2
    gap_length = width * 0.2
    dash_width = width * 0.05
    plot!(plt, x, y, line=:dash, color=:white, linewidth=dash_width, linestyle=:dash, dashes=(dash_length, gap_length))

    return plt
end


function pos_in_road_to_abs_position(x, y, tangent_vecs, normal_vecs, position, lat_offset, road_width::Real)
    ### given a position and lateral offsed in curve-coordinates, find the actual position
    dx = diff([x; x[1]])
    dy = diff([y; y[1]])
    distance = cumsum(sqrt.(dx .^ 2 + dy .^ 2))
    circuit_length = distance[end]
    vehicle_pos = zeros(length(position), 2)
    vehicle_tangent = zeros(length(position), 2)
    for t in 1:length(position)
        # try
        pos_modulo_lap = position[t] % circuit_length
        # Find the first index in distance such that distance[i] <= position
        idx = searchsortedfirst(distance, pos_modulo_lap)
        # Compute the progress since index: 
        ## If progress_from_idx=0, then pos_modulo_lap = distance[idx]
        ## If progress_from_idx=1, then pos_modulo_lap = distance[idx+1]
        progress_from_idx = (pos_modulo_lap - distance[idx]) / (distance[idx+1] - distance[idx])
        vehicle_pos[t, :] = (1. - progress_from_idx) * [x[idx]; y[idx]] + progress_from_idx * [x[idx+1]; y[idx+1]]
        vehicle_pos[t, :] = vehicle_pos[t, :] - normal_vecs[idx, :] * lat_offset[t] * (road_width / 2)
        vehicle_tangent[t, :] = tangent_vecs[idx, :]
        # catch
        #     println(" eeeee ")
        # end
    end
    return vehicle_pos, vehicle_tangent
end

function drawRectangle!(plt, x::Real, y::Real, angle::Real, height::Real, width::Real, color)
    # Rectangle corners in local coordinates (centered at origin)
    half_h = height / 2
    half_w = width / 2
    corners = [
        -half_h -half_w;
        half_h -half_w;
        half_h half_w;
        -half_h half_w
    ]

    # Rotation matrix
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)]

    # Rotate and translate corners
    rotated_corners = (R * corners')'
    translated_corners = rotated_corners .+ [x; y]'

    # Close the rectangle for plotting
    rect_x = vcat(translated_corners[:, 1], translated_corners[1, 1])
    rect_y = vcat(translated_corners[:, 2], translated_corners[1, 2])

    # Plot the rectangle
    plot!(plt, rect_x, rect_y, color=color, fillalpha=0.2)
end