
# ParametricDAQP Interface

function find_CR(x0::Vector{Float64}, sol; eps_gap=1e-5)
    #Find the critical region for x0
    contained_in = Int64[]
    for (ind, region) in enumerate(sol.CRs)
        violation = minimum(region.bth - region.Ath' * x0)
        if (violation >= -eps_gap)
            push!(contained_in, ind)
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

function MPC_control(x0::Vector{Float64}, sol::MPVI, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    ind = find_CR(x0, sol)
    # Extract primal solution
    u = sol.CRs[ind].z' * [x0; 1]
    # Extract first input of each agent's sequence
    return first_input_of_sequence(u, nu, N, T_hor)
end