
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
    evaluatePWA(sol::ParametricDAQP.Solution, θ) -> Vector or nothing

Evaluate the piecewise-affine (PWA) solution at parameter `θ`.
Normalises `θ` using `sol.translation` and `sol.scaling`, locates the corresponding
critical region, and returns the affine evaluation `z' * [θ_norm; 1]`.
Returns `nothing` if `θ` falls outside all stored critical regions.
"""
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
    arrange_vector_as_time_seq(u, nu, N, T_hor; n_slack=0) -> Vector{Vector{Vector{Float64}}}

Split the stacked input sequence `u` into a per-timestep, per-agent input sequence:
`useq[t][i]` is agent `i`'s input at timestep `t`. `u`'s layout and the optional `n_slack`
trailing slack entries per agent (e.g. when `u` comes from an `LQGNEP` built with
`soften_state_constraints=true`, see [`DynLQGame2LQGNEP`](@ref)) match
[`extract_input_at_timestep`](@ref), which this function calls once per timestep.
"""
function arrange_vector_as_time_seq(u::AbstractVector, nu::Vector{Int64}, N::Int64, T_hor::Int64;
    n_slack::Int64=0)
    return [extract_input_at_timestep(u, nu, N, T_hor, t; n_slack=n_slack) for t in 1:T_hor]
end

"""
    MPC_control(sol, x0, nu, N, T_hor; n_slack=0) -> Vector{Vector{Float64}}

Retrieve the first MPC control input for each agent at state `x0` from a pre-computed
`ParametricDAQP.Solution`. Combines `evaluatePWA` and `extract_input_at_timestep`.

Set `n_slack` to the number of trailing slack entries per agent (`T_hor*prob.m_x`, one per
state-constraint row per timestep) if `sol` was built from an `LQGNEP` with
`soften_state_constraints=true` (see [`DynLQGame2LQGNEP`](@ref)), so they are excluded.
"""
function MPC_control(sol::ParametricDAQP.Solution, x0::Vector{Float64}, nu::Vector{Int64}, N::Int64, T_hor::Int64;
    n_slack::Int64=0)
    u = evaluatePWA(sol::ParametricDAQP.Solution, x0::Vector{Float64})
    return extract_input_at_timestep(u, nu, N, T_hor, 1; n_slack=n_slack)
end

"""
    extract_input_at_timestep(u, nu, N, T_hor, t; n_slack=0) -> Vector{Vector{Float64}}

Extract the inputs for all agents at timestep `t` (1-indexed) from the stacked input
sequence `u`. Layout: `[u₁[1:T]; u₂[1:T]; …]`, each block of length `nu[i] * T_hor`
(plus `n_slack` trailing slack entries per agent, e.g. when `u` comes from an `LQGNEP`
built with `soften_state_constraints=true`, see [`DynLQGame2LQGNEP`](@ref), where
`n_slack = T_hor*prob.m_x`). Only the `nu[i]`-long input part of each agent's block is
returned; any trailing slack entries are never included.
"""
function extract_input_at_timestep(u::AbstractVector, nu::Vector{Int64}, N::Int64, T_hor::Int64, t::Int64;
    n_slack::Int64=0)
    u_t = Vector{Vector{Float64}}(undef, N)
    agent_start = 1
    for i in 1:N
        offset = (t - 1) * nu[i]
        u_t[i] = u[agent_start + offset : agent_start + offset + nu[i] - 1]
        agent_start += nu[i] * T_hor + n_slack
    end
    return u_t
end