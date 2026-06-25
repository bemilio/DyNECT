
# ParametricDAQP Interface

function find_CR(x0::Vector{Float64}, sol; eps_gap=1e-5)
    #Find the critical region for x0
    for (ind, region) in enumerate(sol.CRs)
        violation = minimum(region.bth - region.Ath' * x0)
        if (violation >= -eps_gap)
            return region
        end
    end
    return nothing
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

function evaluatePWA(sol::ParametricDAQP.Solution, θ::Vector{Float64})
    θ_normalized = (θ - sol.translation) .* sol.scaling
    CR = find_CR(θ_normalized, sol)
    if !isnothing(CR)
        return CR.z' * [θ_normalized; 1]
    else
        return nothing
    end
end

function MPC_control(sol::ParametricDAQP.Solution, x0::Vector{Float64}, nu::Vector{Int64}, N::Int64, T_hor::Int64)
    u = evaluatePWA(sol::ParametricDAQP.Solution, x0::Vector{Float64})
    return first_input_of_sequence(u, nu, N, T_hor)
end