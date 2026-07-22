
# ParametricDAQP Interface

"""
    find_CR(x0, sol; eps_gap=1e-5) -> region or nothing

Find the critical region in `sol.CRs` that contains the (normalised) parameter `x0`.
Returns the matching region, or `nothing` if `x0` is outside all stored regions.
"""
function find_CR(x0::Vector{Float64}, sol; eps_gap=1e-5)
    for (ind, region) in enumerate(sol.CRs)
        violation = minimum(region.bth - region.Ath' * x0)
        if (violation >= -eps_gap)
            return region
        end
    end
    return nothing
end

"""
    first_input_of_sequence(u, nu, N, T_hor) -> Vector{Vector{Float64}}

Extract the first time-step input for each agent from the stacked input sequence `u`.

The sequence is laid out as `[u₁[0:T-1]; u₂[0:T-1]; …]`, where each block has length
`nu[i] * T_hor`. Returns a length-`N` vector, one `nu[i]`-vector per agent.
"""
function first_input_of_sequence(u::Vector{Float64}, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    u_first = Vector{Vector{Float64}}(undef, N)
    start_row = 1
    for i = 1:N
        u_first[i] = u[start_row:start_row+nu[i]-1]
        start_row += nu[i] * T_hor
    end
    return u_first
end

"""
    evaluatePWA(sol::ParametricDAQP.Solution, θ) -> Vector or nothing

Evaluate the piecewise-affine (PWA) solution at parameter `θ`.
Normalises `θ` using `sol.translation` and `sol.scaling`, locates the corresponding
critical region, and returns the affine evaluation `z' * [θ_norm; 1]`.
Returns `nothing` if `θ` falls outside all stored critical regions.
"""
function arrange_vector_as_time_seq(u::Vector{Float64}, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    start_row_agent = cumsum(vcat(1,nu.*T_hor))
    useq = [[u[start_row_agent[i] + (t-1)*nu[i] : start_row_agent[i] - 1 + t*nu[i]] for i=1:N] for t=1:T_hor]
    return useq
end

function evaluatePWA(sol::ParametricDAQP.Solution, θ::Vector{Float64})
    θ_normalized = (θ - sol.translation) .* sol.scaling
    CR = find_CR(θ_normalized, sol)
    if !isnothing(CR)
        return CR.z' * [θ_normalized; 1]
    else
        return nothing
    end
end

"""
    MPC_control(sol, x0, nu, N, T_hor) -> Vector{Vector{Float64}}

Retrieve the first MPC control input for each agent at state `x0` from a pre-computed
`ParametricDAQP.Solution`. Combines `evaluatePWA` and `first_input_of_sequence`.
"""
function MPC_control(sol::ParametricDAQP.Solution, x0::Vector{Float64}, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    u = evaluatePWA(sol::ParametricDAQP.Solution, x0::Vector{Float64})
    return first_input_of_sequence(u, nu, N, T_hor)
end

"""
    extract_input_at_timestep(u, nu, N, T_hor, t) -> Vector{Vector{Float64}}

Extract the inputs for all agents at timestep `t` (1-indexed) from the stacked input
sequence `u`. Layout: `[u₁[1:T]; u₂[1:T]; …]`, each block of length `nu[i] * T_hor`.
"""
function extract_input_at_timestep(u::AbstractVector, nu::Vector{Int64}, N::Int64, T_hor::Int64, t::Int64)
    u_t = Vector{Vector{Float64}}(undef, N)
    agent_start = 1
    for i in 1:N
        offset = (t - 1) * nu[i]
        u_t[i] = u[agent_start + offset : agent_start + offset + nu[i] - 1]
        agent_start += nu[i] * T_hor
    end
    return u_t
end