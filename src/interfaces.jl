using ForwardDiff, SparseArrays

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

#added v_static_mpGNE
function filter_gne_crs!(sol::ParametricDAQP.Solution, game::StaticGNEGame) #testing
    N = game.N
    m_sh = length(game.b_sh)
    n_local = sum(size(game.A_loc[i], 1) for i in 1:N)
    shared_start = n_local + 1
    
    filter!(sol.CRs) do cr
        for i in 1:m_sh
            player_row_indices = [shared_start + (p - 1) * m_sh + (i - 1) for p in 1:N]
            active_status = [row_idx ∈ cr.AS for row_idx in player_row_indices]
            all_active = all(active_status)
            none_active = !any(active_status)
            if !(all_active || none_active)
                return false
            end
        end
        return true
    end
    return length(sol.CRs)
end

#added Optimal GNE selection
function select_optimal_gne!(
    sol::ParametricDAQP.Solution,
    φ::Function
)::OptimalGNEResult
    
    candidates = []
    
    for k in 1:length(sol.CRs)
        CR_k = sol.CRs[k]
        
        # Extract region geometry
        # Ath has shape n_θ × m_k; constraint is Ath'θ ≤ bth (see find_CR)
        A_k = CR_k.Ath'  # m_k × n_θ
        b_k = CR_k.bth
        n_θ = size(A_k, 2)
        
        # Extract affine map: u = M_k θ + d_k from CR.z' * [θ; 1]
        # CR.z is (n_θ+1) × n_u, so:
        M_k = CR_k.z[1:end-1, :]'  # n_u × n_θ
        d_k = vec(CR_k.z[end, :])   # n_u
        
        try
            # Define composite function: ψ(θ) = φ(M_k θ + d_k)
            ψ(θ::Vector) = φ(M_k * θ + d_k)
            
            # Compute Hessian and gradient via AD
            H_θ = ForwardDiff.hessian(ψ, zeros(n_θ))
            g_θ = ForwardDiff.gradient(ψ, zeros(n_θ))
            
            # Solve convex QP: min_θ 0.5*θ'*H_θ*θ + g_θ'*θ  s.t. A_k θ ≤ b_k
            qp = Clarabel.Solver()
            settings = Clarabel.Settings(verbose=false)
            cone = [Clarabel.NonnegativeConeT(size(A_k, 1))]
            
            Clarabel.setup!(qp, SparseMatrixCSC(H_θ), g_θ, 
                           SparseMatrixCSC(A_k), b_k, cone, settings)
            result = Clarabel.solve!(qp)
            
            if result.status == Clarabel.SOLVED
                θ_k = result.x[1:n_θ]
                u_k = M_k * θ_k + d_k
                φ_k = φ(u_k)
                push!(candidates, (θ=θ_k, u=u_k, φ=φ_k, region=k))
            end
            
        catch e
            @warn "[select_optimal_gne!] Region $k skipped: $e"
            continue
        end
    end
    
    if isempty(candidates)
        error("[select_optimal_gne!] No regions solved successfully.")
    end
    
    # Global selection: pick best φ
    φ_values = [c.φ for c in candidates]
    k_best = argmin(φ_values)
    best = candidates[k_best]
    
    return OptimalGNEResult(best.θ, best.u, best.φ, best.region, candidates)
end