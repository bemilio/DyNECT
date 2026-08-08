####### Type conversion functions #####

@doc raw"""
    DynLQGame2LQGNEP(prob::DynLQGame, T_hor::Int64)

Constructs the [`LQGNEP`](@ref) followers' game for a dynamic Nash equilibrium
problem (`DynLQGame`) over a finite prediction horizon, with the initial state ``x_0`` as
the external parameter ``\gamma``.

# Arguments
- `prob::DynLQGame`: Dynamic game structure containing system dynamics, cost, and constraints.
- `T_hor::Int64`: Prediction horizon.

# Returns
- `LQGNEP`: the static GNEP over the stacked input sequence ``u``, parametrized by
  ``\gamma = x_0``, equivalent to the [`mpAVI`](@ref) produced by [`DynLQGame2mpAVI`](@ref).
"""
function DynLQGame2LQGNEP(prob::DynLQGame, T_hor::Int64)
    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    Γ, Γi, Θ, c̅ = generate_prediction_model(prob.A, prob.B, T_hor; c=prob.c)

    # Define Q̅[i] = blkdg(Q[1][i],..., Q[T][i], P[i])
    Q̅ = [BlockDiagonal([kron(I(T_hor - 1), prob.Q[i]), prob.P[i]]) for i in 1:prob.N]

    Q_static = Matrix{Matrix{Float64}}(undef, prob.N, prob.N)
    for i in 1:prob.N
        QiΓ = Γi[i]' * Q̅[i]
        for j in 1:prob.N
            Q_static[i, j] = Matrix{Float64}(QiΓ * Γi[j] + kron(I(T_hor), prob.R[i][j]))
        end
    end
    Q = [[Q_static[i, j] for j in 1:prob.N] for i in 1:prob.N]

    # Parametric affine influence of x0 on the objective (γ = x0)
    Q_qγ = [Matrix{Float64}(Γi[i]' * Q̅[i] * Θ) for i in 1:prob.N]

    # define f (affine, γ=0 part of the VI mapping)
    q_static = Vector{Vector{Float64}}(undef, prob.N)
    for i in 1:prob.N
        q̅i = vcat(kron(ones(T_hor - 1), prob.q[i]), prob.p[i])
        r̅i = kron(ones(T_hor), prob.r[i])
        q_static[i] = Γi[i]' * (Q̅[i] * c̅ + q̅i) + r̅i
    end

    ## Constraints
    # Cₓ*x[t] ≤ bₓ ∀ t ==> C̅ₓΓu̅ <= b̅ₓ -C̅ₓΘx₀ - C̅ₓc̅
    # where C̅ₓ = I ⊗ Cₓ
    C̅_x = kron(I(T_hor), prob.C_x)

    # Shared constraints (input + state), one column block per agent, same rows for all agents
    A_sh = [Matrix{Float64}([kron(I(T_hor), prob.C_u_i[i]); C̅_x * Γi[i]]) for i in 1:prob.N]
    b_sh = vcat(kron(ones(T_hor), prob.b_u), kron(ones(T_hor), prob.b_x) - C̅_x * c̅)
    # Parametric affine influence of x0 on the shared constraints (γ = x0)
    B_sh_γ = Matrix{Float64}([zeros(T_hor * prob.m_u, prob.nx); -1 * C̅_x * Θ])

    # Local constraints, per agent (independent of x0)
    A_loc = [Matrix{Float64}(kron(I(T_hor), prob.C_loc_i[i])) for i in 1:prob.N]
    b_loc = [kron(ones(T_hor), prob.b_loc_i[i]) for i in 1:prob.N]
    B_loc_γ = [zeros(length(b_loc[i]), prob.nx) for i in 1:prob.N]

    return LQGNEP(Q, q_static, A_loc, b_loc, A_sh, b_sh;
        Q_qγ=Q_qγ, B_loc_γ=B_loc_γ, B_sh_γ=B_sh_γ)
end

@doc raw"""
    DynLQGame2LQGNEP(prob::DynLQGameTV)

Constructs the [`LQGNEP`](@ref) followers' game for a time-varying dynamic Nash
equilibrium problem (`DynLQGameTV`) over its (fixed) prediction horizon `prob.Thor`, with
the initial state ``x_0`` as the external parameter ``\gamma``.

# Arguments
- `prob::DynLQGameTV`: Time-varying dynamic game structure containing system dynamics, cost,
  and constraints.

# Returns
- `LQGNEP`: the static GNEP over the stacked input sequence ``u``, parametrized by
  ``\gamma = x_0``, equivalent to the [`mpAVI`](@ref) produced by [`DynLQGame2mpAVI`](@ref).
"""
function DynLQGame2LQGNEP(prob::DynLQGameTV)
    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    Γ, Γi, Θ, c̅ = generate_prediction_model(prob.A, prob.B, prob.Thor; c=prob.c)

    # Define Q̅[i] = blkdg(Q[1][i],..., Q[T][i], P[i])
    Q̅ = [
        BlockDiagonal(vcat([prob.Q[t][i] for t in 1:prob.Thor-1], [prob.P[i]]))
        for i in 1:prob.N
    ]

    Q_static = Matrix{Matrix{Float64}}(undef, prob.N, prob.N)
    for i in 1:prob.N
        QiΓ = Γi[i]' * Q̅[i]
        for j in 1:prob.N
            Q_static[i, j] = Matrix{Float64}(QiΓ * Γi[j] + BlockDiagonal([prob.R[t][i][j] for t = 1:prob.Thor]))
        end
    end
    Q = [[Q_static[i, j] for j in 1:prob.N] for i in 1:prob.N]

    # Parametric affine influence of x0 on the objective (γ = x0)
    Q_qγ = [Matrix{Float64}(Γi[i]' * Q̅[i] * Θ) for i in 1:prob.N]

    # define f (affine, γ=0 part of the VI mapping)
    q_static = Vector{Vector{Float64}}(undef, prob.N)
    for i in 1:prob.N
        q̅i = vcat([prob.q[t][i] for t in 1:prob.Thor-1]..., prob.p[i])
        r̅i = vcat([prob.r[t][i] for t in 1:prob.Thor]...)
        q_static[i] = Γi[i]' * (Q̅[i] * c̅ + q̅i) + r̅i
    end

    ## Constraints
    # Cᵗₓ*x[t] ≤ bᵗₓ ∀ t ==> C̅ₓΓu̅ <= b̅ₓ -C̅ₓΘx₀ - C̅ₓc̅
    # where C̅ₓ = blkdiag(C¹ₓ, ...,Cᵗₓ)
    C̅_x = BlockDiagonal(prob.C_x)

    # Shared constraints (input + state), one column block per agent, same rows for all agents
    A_sh = [Matrix{Float64}([BlockDiagonal([prob.C_u[t][i] for t in 1:prob.Thor]); C̅_x * Γi[i]]) for i in 1:prob.N]
    b_sh = vcat(vcat(prob.b_u...), vcat(prob.b_x...) - C̅_x * c̅)
    # Parametric affine influence of x0 on the shared constraints (γ = x0)
    B_sh_γ = Matrix{Float64}([zeros(prob.Thor * prob.m_u, prob.nx); -1 * C̅_x * Θ])

    # Local constraints, per agent (independent of x0)
    A_loc = [Matrix{Float64}(BlockDiagonal([prob.C_loc[t][i] for t in 1:prob.Thor])) for i in 1:prob.N]
    b_loc = [vcat([prob.b_loc[t][i] for t in 1:prob.Thor]...) for i in 1:prob.N]
    B_loc_γ = [zeros(length(b_loc[i]), prob.nx) for i in 1:prob.N]

    return LQGNEP(Q, q_static, A_loc, b_loc, A_sh, b_sh;
        Q_qγ=Q_qγ, B_loc_γ=B_loc_γ, B_sh_γ=B_sh_γ)
end

@doc raw"""
    DynLQGame2mpAVI(prob::DynLQGame, T_hor::Int64)

Constructs the parametric variational inequality ([`mpAVI`](@ref)) for a dynamic Nash
equilibrium problem (`DynLQGame`) over a finite prediction horizon, by building the
equivalent [`LQGNEP`](@ref) via [`DynLQGame2LQGNEP`](@ref) and
converting it with `mpAVI(::LQGNEP)`.

# Arguments
- `prob::DynLQGame`: Dynamic game structure containing system dynamics, cost, and constraints.
- `T_hor::Int64`: Prediction horizon.

# Returns
- `mpAVI`: the parametric variational inequality
  ``\mathrm{VI}(Hu + Fx_0 + f,\ Au \leq Bx_0 + b)``, where ``u`` is the stacked input
  sequence for all agents and ``x_0`` is the initial state.
"""
function DynLQGame2mpAVI(prob::DynLQGame, T_hor::Int64)
    return mpAVI(DynLQGame2LQGNEP(prob, T_hor))
end

@doc raw"""
    DynLQGame2mpAVI(prob::DynLQGameTV)

Constructs the parametric variational inequality ([`mpAVI`](@ref)) for a time-varying
dynamic Nash equilibrium problem (`DynLQGameTV`) over its (fixed) prediction horizon
`prob.Thor`, by building the equivalent [`LQGNEP`](@ref) via
[`DynLQGame2LQGNEP`](@ref) and converting it with `mpAVI(::LQGNEP)`.

# Arguments
- `prob::DynLQGameTV`: Time-varying dynamic game structure containing system dynamics, cost,
  and constraints.

# Returns
- `mpAVI`: the parametric variational inequality
  ``\mathrm{VI}(Hu + Fx_0 + f,\ Au \leq Bx_0 + b)``, where ``u`` is the stacked input
  sequence for all agents and ``x_0`` is the initial state.
"""
function DynLQGame2mpAVI(prob::DynLQGameTV)
    return mpAVI(DynLQGame2LQGNEP(prob))
end

@doc raw"""
    NabetaniParametrization(game::LQGNEP; θub=nothing, θlb=nothing)

Assemble a (possibly parametric) GNE game into a multi-parametric variational inequality (mpAVI).

The Nabetani-Tseng-Fukushima reparametrization transforms shared constraints into parameter-dependent bounds:
- Agent 1: ``A_{\text{sh},1} x_1 \leq \theta_1``
- Agent i (i>1): ``A_{\text{sh},i} x_i \leq -\theta_{i-1} + b_{\text{sh}}``

The Nabetani parameter ``\theta`` is stacked together with `game`'s own parameter ``\gamma``
(zero-dimensional if `game` is not parametric) into a single combined mpAVI parameter.

Returns: ``\text{VI}(H x + F [\theta;\gamma] + f, A x \leq B [\theta;\gamma] + b)`` where
``\theta \in [\text{lb}, \text{ub}]`` and ``C\gamma \leq d``.
"""
function NabetaniParametrization(game::LQGNEP; θub::Union{Vector{Float64},Nothing}=nothing, θlb::Union{Vector{Float64},Nothing}=nothing)
    # Infer dimensions
    N = game.N
    n = game.n
    n_total = sum(n)
    m_sh = length(game.b_sh)
    n_param = (N - 1) * m_sh + game.n_γ
    # Check size of bounds, default to ±100 
    θub = isnothing(θub) ? 100 .* ones((N - 1) * m_sh) : θub
    θlb = isnothing(θlb) ? -100 .* ones((N - 1) * m_sh) : θlb
    @assert length(θub)==(game.N-1) * m_sh "# of par. upper bounds is $((game.N-1) * m_sh), got $(length(θub))"
    @assert length(θlb)==(game.N-1) * m_sh "# of par. low bounds is $((game.N-1) * m_sh), got $(length(θlb)) "

    # Assemble Hessian (H) from Q blocks 
    H = BlockArray{Float64}(undef_blocks, n, n)
    for i in 1:N
        for j in 1:N
            H[Block(i, j)] = game.Q[i][j]
        end
    end
    H = Matrix(H)
    
    # Assemble linear cost (f) from q vectors
    f = vcat(game.q...)
    
    # Assemble local constraints
    A_loc = BlockDiagonal(game.A_loc)
    b_loc = vcat(game.b_loc...)

    # Assemble Nabetani reparametrization
    # A_hat = blkdiag(A_sh[1], A_sh[2], ..., A_sh[N])
    A_hat = BlockDiagonal(game.A_sh)
    A_hat = Matrix(A_hat)
    
    # B_g structure (correct for N ≥ 2):
    # Shape: (N*m_sh) × ((N-1)*m_sh)
    # Top block: I_{(N-1)*m_sh}    (agents 1,...,N-1 get explicit allocation θ)
    # Bottom block: -ones(m_sh, (N-1)*m_sh)  (agent N gets remainder)
    B_g_top = hcat(Matrix{Float64}(I((N - 1) * m_sh)), zeros((N - 1) * m_sh, game.n_γ))
    # [-I, ..., -I, game.B_sh_γ]
    B_g_bottom = hcat(kron(-1 .* ones(1, N-1), Matrix(I, m_sh, m_sh)), game.B_sh_γ)
    B_g = vcat(B_g_top, B_g_bottom)

    # d_g = [zeros((N-1)*m_sh); b_sh]
    d_g = vcat(zeros((N - 1) * m_sh), game.b_sh)

    # Stack all constraints
    A = vcat(A_loc, A_hat)

    # B matrix: local constraints have no theta dependence (zeros), shared constraints have B_g
    B_loc = hcat(zeros(size(A_loc, 1), (N - 1) * m_sh), vcat(game.B_loc_γ...))
    B = vcat(B_loc, B_g)

    b = vcat(b_loc, d_g)

    # Assemble F (γ-sensitivity of the objective), zero-padded for the Nabetani-θ columns
    # (F and B must share the same n_param columns, since both are multiplied by the combined [θ; γ])
    F = hcat(zeros(n_total, (N - 1) * m_sh), vcat(game.Q_qγ...))

    # γ polytope constraint (Cγ ≤ d), zero-padded for the Nabetani-θ columns
    C = hcat(zeros(size(game.C, 1), (N - 1) * m_sh), game.C)
    d = game.d

    # Return mpAVI
    ub = vcat(θub, game.ub)
    lb = vcat(θlb, game.lb)
    @assert length(ub) == n_param && length(lb) == n_param "[NabetaniParametrization] internal size mismatch: expected $n_param parameters, got ub=$(length(ub)), lb=$(length(lb))"

    return mpAVI(H, F, f, A, B, b, C=C, d=d, ub=ub, lb=lb)
end
####### END Type conversion functions #######

####### Helper functions for optimal GNE selection ##########
function filter_gne_crs!(sol::ParametricDAQP.Solution, game::LQGNEP)
    N = game.N
    m_sh = length(game.b_sh)
    n_local = sum(size(game.A_loc[i], 1) for i in 1:N)
    shared_start = n_local + 1
    
    filter!(sol.CRs) do cr
        for i in 1:m_sh
            # Collect all rows of the reformulated constraints associated to the same shared constraints
            player_row_indices = [shared_start + (p - 1) * m_sh + (i - 1) for p in 1:N]
            # Check if they are all active / inactive
            active_status = [row_idx ∈ cr.AS for row_idx in player_row_indices]
            all_active = all(active_status)
            none_active = !any(active_status)
            # If they are neither all active or inactive, the region does not correspond to a set of GNEs
            if !(all_active || none_active)
                return false
            end
        end
        return true
    end
    return length(sol.CRs)
end

function select_optimal_gne(
    sol::ParametricDAQP.Solution,
    ϕ::Function,
    is_quadratic::Bool,
    optimizer = Clarabel.Optimizer
)::OptimalGNEResult
    if !is_quadratic && optimizer == Clarabel.Optimizer
        @warn "[select_optimal_gne!] Non-quadratic problem with Clarabel not supported. Switching to IPOPT."
        optimizer = Ipopt.Optimizer
    end
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
        n_u = size(M_k, 1)

        # Construct optimization problem
        if is_quadratic
            # Quadratic problem: objective built by straightforward composition
            model = Model(optimizer)
            set_silent(model)
            @variable(model, θ[1:n_θ])
            @objective(model, Min, ϕ(M_k * θ .+ d_k))
            @constraint(model, A_k * θ .<= b_k)
        else
            # Otherwise: add linear equality constraint
            model = Model(optimizer)
            set_silent(model)
            @variable(model, θ[1:n_θ])
            @variable(model, y[1:n_u])
            @constraint(model, y .== M_k * θ .+ d_k)
            @constraint(model, A_k * θ .<= b_k)
            @operator(model, op_ϕ, m, (ys...) -> ϕ(collect(ys)))
            @objective(model, Min, op_ϕ(y...))
        end
        
        optimize!(model)
        # Extract results only when optimal
        if termination_status(model) == OPTIMAL
            θ_opt = value.(θ)
            obj_opt = objective_value(model)
            push!(candidates, (θ=θ_opt, u=M_k * θ_opt + d_k, φ=obj_opt, region=k))
        else
            @warn "[select_optimal_gne!] Region $k skipped: termination status $(termination_status(model))"
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

####### END Helper functions for optimal GNE selection ##########

####### Dynamics functions #######

@doc raw"""
    generate_prediction_model(A, Bi, T_hor)

Constructs the prediction model matrices, ``\Gamma_i``, and ``\Theta``, and a vector ``\bar{c}`` for a discrete-time affine system
```math 
x^+ = Ax + \sum_{i=1}^N (B_i u_i) + c
```
Prediction model:
```math
\bar{x} = \Theta x_0 + \sum_{i=1}^N (\Gamma_i\bar{u}_i) + \bar{c}
```
where
```math
\bar{x}=\text{col}(x[1], ... , x[T]), \quad \bar{u}_i = \text{col}(u_i[0], ..., u_i[T-1])
```
Specifically, the matrices are
```math
\Gamma_i = \begin{bmatrix}
	B_i & 0 & 0 & ... & 0 \\
	AB_i & B_i & 0 & ... & 0 \\
	\vdots & & \ddots & & \\
	A^{T-1}B_i & A^{T-2}B_i & ... &  & B_i
	\end{bmatrix} \quad \Theta = \begin{bmatrix}
		A \\ A^2 \\ ... \\ A^T
	\end{bmatrix} \quad \bar{c} = \begin{bmatrix}
	I & 0 & 0 & ... & 0 \\
	A & I & 0 & ... & 0 \\
	\vdots & & \ddots & & \\
	A^{T-1} & A^{T-2} & ... &  & I
	\end{bmatrix} (1_{T} \otimes c)
```
# Arguments
- `A`: State transition matrix.
- `B`: Vector of input matrices for each agent or subsystem.
- `T_hor`: Prediction horizon (number of time steps).
- `c`: affine part of the dynamics. If omitted, defaults to a vector of zeros.

# Returns
- `Γ`: Matrix mapping the stacked input sequences to the stacked state sequence.
- `Γi`: Vector of views into `Γ`, each corresponding to the input sequence of an agent.
- `Θ`: Matrix mapping the initial state to the stacked state sequence.
- `c̅`: Affine additive vector.
"""
function generate_prediction_model(A::Matrix{Float64}, B::Vector{Matrix{Float64}}, T_hor::Int; c::Vector{Float64}=zeros(size(A, 1)))
    # system: x⁺=Ax + ∑(Bᵢuᵢ) + c
    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    n_x = size(A, 1)
    n_u = map(x -> size(x, 2), B)
    N = length(B)
    Γ = zeros(n_x * T_hor, sum(n_u) * T_hor) # Maps the column stack of input sequences to state sequence
    Γi = Vector{SubArray}(undef, N) # Views into Γ
    start_col = 1
    for i in 1:N
        Γi[i] = @view Γ[:, start_col:start_col+n_u[i]*T_hor-1]
        start_col += n_u[i] * T_hor
    end
    # Define the Γ matrices
    for i in 1:N
        Γi[i][1:n_x, 1:n_u[i]] .= B[i]
        for t in 1:T_hor-1
            current_rows = n_x*t+1:n_x*(t+1)
            previous_rows = n_x*(t-1)+1:n_x*t
            Γi[i][current_rows, n_u[i]+1:end] .= Γi[i][previous_rows, 1:end-n_u[i]] # Shift the previous block row by n_u
            Γi[i][current_rows, 1:n_u[i]] .= A * Γi[i][previous_rows, 1:n_u[i]] # Multiply the leftmost block of the previous row by A
        end
    end

    Θ = vcat([A^t for t in 1:T_hor]...) # Maps the initial state to state sequence

    # Affine part
    Γc = zeros(n_x * T_hor, n_x * T_hor)
    Γc[1:n_x, 1:n_x] .= Matrix{Float64}(I(n_x))
    for t in 1:T_hor-1
        current_rows = n_x*t+1:n_x*(t+1)
        previous_rows = n_x*(t-1)+1:n_x*t
        Γc[current_rows, n_x+1:end] .= Γc[previous_rows, 1:end-n_x] # Shift the previous block row by n_x
        Γc[current_rows, 1:n_x] .= A * Γc[previous_rows, 1:n_x] # Multiply the leftmost block of the previous row by A
    end
    c̅ = Γc * kron(ones(T_hor), c)

    return Γ, Γi, Θ, c̅
end

# For time-varying systems
function generate_prediction_model(A::Vector{Matrix{Float64}}, 
        B::Vector{Vector{Matrix{Float64}}},     
        T_hor::Int; 
        c::Vector{Vector{Float64}}=[zeros(size(A[1], 1)) for _ in 1:T_hor])
    # system: x⁺=Aᵗx + ∑(Bᵢᵗuᵢ) + cᵗ
    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    n_x = size(A[1], 1)
    n_u = map(x -> size(x, 2), B[1])
    N = length(B[1])
    Γ = zeros(n_x * T_hor, sum(n_u) * T_hor) # Maps the column stack of input sequences to state sequence
    Γi = Vector{SubArray}(undef, N) # Views into Γ
    start_col = 1
    for i in 1:N
        Γi[i] = @view Γ[:, start_col:start_col+n_u[i]*T_hor-1]
        start_col += n_u[i] * T_hor
    end
    # Define the Γ matrices
    # Γᵢ = [ Bᵢ¹      0    0  ... 0;
    #       A²Bᵢ¹    Bᵢ²   0  ... 0;
    #       A³A²Bᵢ¹  A³Bᵢ² Bᵢ³ ... 0; 
    # ...
    #]
    for i in 1:N
        Γi[i][1:n_x, 1:n_u[i]] .= B[1][i] # Place Bᵢ[1] on the top left block
        for t in 2:T_hor
            current_rows = n_x*(t-1)+1:n_x*(t)
            previous_rows = n_x*(t-2)+1:n_x*(t-1)
            Γi[i][current_rows, :] .= Γi[i][previous_rows, :] # copy the previous block row to the current one
            Γi[i][current_rows, :] .= A[t] * Γi[i][current_rows, :] # Premultiply by A[t]
            Γi[i][current_rows, n_u[i]*(t-1)+1:n_u[i]*t] .= B[t][i] # Place Bᵢ[t] on the diagonal block
        end
    end
    # Define the Θ matrix (multiplying x_0)
    # Θ = [ A¹    ;
    #       A²A¹  ;
    #       A³A²A¹; 
    #      ...]
    Θ = zeros(n_x * T_hor, n_x)
    Θ[1:n_x, :] = A[1]
    for t in 2:T_hor
        current_rows = n_x*(t-1)+1:n_x*(t)
        previous_rows = n_x*(t-2)+1:n_x*(t-1)
        Θ[current_rows,:] = A[t] * Θ[previous_rows,:]
    end

    # Affine part
    # Define the Γ_c  matrix
    # Γ_c = [ I     0     0 ... 0;
    #       A²    I     0 ... 0;
    #       A³A²  A³A²  I ... 0; 
    # ...]
    # Then, the affine part of the prediction model will be
    # c̄ = Γ_c * [c₁; c₂, ....]

    Γc = zeros(n_x * T_hor, n_x * T_hor)
    Γc[1:n_x, 1:n_x] .= Matrix{Float64}(I(n_x)) # Place I on the top left block
    for t in 2:T_hor
        current_rows = n_x*(t-1)+1:n_x*(t)
        previous_rows = n_x*(t-2)+1:n_x*(t-1)
        Γc[current_rows, :] .= Γc[previous_rows, :] # copy the previous block row to the current one
        Γc[current_rows, :] .= A[t] * Γc[current_rows, :] # Premultiply by A[t]
        Γc[current_rows, n_x*(t-1)+1:n_x*t] .= Matrix{Float64}(I(n_x)) # Place I on the diagonal block
    end
    c̅ = Γc * vcat(c...)

    return Γ, Γi, Θ, c̅
end


"""
    is_stabilizable(A, B) -> Bool

Return `true` if the pair `(A, B)` is stabilizable, i.e. every unstable mode of `A`
(eigenvalue with |λ| ≥ 1) is controllable. Uses the Hautus lemma.
"""
function is_stabilizable(A, B)
    eigv_A = eigvals(A)
    n_x = size(A, 1)
    for λ in eigv_A
        if abs(λ) ≥ 1
            M = hcat(λ * I(n_x) - A, B)
            if rank(M) < n_x
                return false
            end
        end
    end
    return true
end

"""
    is_detectable(A, C) -> Bool

Return `true` if the pair `(A, C)` is detectable, i.e. every unstable mode of `A`
(eigenvalue with |λ| ≥ 1) is observable. Uses the Hautus lemma.
"""
function is_detectable(A, C)
    eigv_A = eigvals(A)
    n_x = size(A, 1)
    for λ in eigv_A
        if abs(λ) ≥ 1
            M = vcat(λ * I(n_x) - A, C)
            if rank(M) < n_x
                return false
            end
        end
    end
    return true
end
####### END Dynamics functions #######

"""
    compute_residual(prob::AVI, x::AbstractVector) -> Float64

Compute the natural residual of the AVI at point `x`: the distance from `x` to its
projection onto the feasible set along the VI mapping direction.
"""
function compute_residual(prob::AVI, x::AbstractVector)
    y = x - (prob.H * x + prob.f)
    proj = DAQP.Model()
    DAQP.setup(proj, Matrix{Float64}(I, prob.n, prob.n), -y, Matrix{Float64}(prob.A), prob.b, Float64[], zeros(Cint, prob.m))
    x_transf, _, _, _ = DAQP.solve(proj)
    return norm(x - x_transf)
end

@doc raw"""
    setParameterSpace(mpavi::mpAVI; C, d, ub, lb) -> mpAVI

Return a new `mpAVI` with updated parameter-space bounds. Any argument left as `nothing`
retains the value from `mpavi`.

# Keyword Arguments
- `C`, `d`: Polytope constraint matrix and right-hand side (`Cθ ≤ d`).
- `ub`, `lb`: Upper and lower box bounds on the parameter `θ`.
"""
function setParameterSpace(mpavi::mpAVI;
    C::Union{AbstractMatrix{Float64},Nothing}=nothing,
    d::Union{AbstractVector{Float64},Nothing}=nothing,
    ub::Union{AbstractVector{Float64},Nothing}=nothing,
    lb::Union{AbstractVector{Float64},Nothing}=nothing)
    C = isnothing(C) ? mpavi.C : C
    d = isnothing(d) ? mpavi.d : d
    ub = isnothing(ub) ? mpavi.ub : ub
    lb = isnothing(lb) ? mpavi.lb : lb
    return mpAVI(mpavi.H, mpavi.F, mpavi.f, mpavi.A, mpavi.B, mpavi.b; C, d, ub, lb)
end


# Computes the hessian of a multi-input function
function block_hessian(f, x...)
    # Vectorize x
    v = vcat(x...)
    # Create selection ranges on v that correspond to the elements of x
    lengths = length.(x)
    offsets = cumsum((1, lengths[1:end-1]...))
    ranges = [offsets[i] : offsets[i] + lengths[i] - 1 for i in eachindex(x)]

    # Vectorize f 
    f_vec(v) = f(ntuple(i -> # The ntuple is better understood by Zygote than generators, apparently
        @view(v[offsets[i]:offsets[i] + lengths[i] - 1]),
        length(x)
    )...)

    H = Zygote.hessian(f_vec, v)

    blocks = Dict(
    (i, j) => @view H[
        ranges[i],
        ranges[j]
    ]
    for i in eachindex(x),
        j in eachindex(x)
    )
    return H, blocks
end


@doc raw"""
    LQapprox(game, u, x0, T_hor, γ=nothing)

Linearize the dynamics and quadratize the objectives of `game` (a [`DynGame`](@ref))
along the horizon `T_hor`, around the input sequence `u` and initial state `x0`,
returning the resulting [`DynLQGameTV`](@ref) time-varying LQ game.

# Arguments
- `game::DynGame`: the (possibly nonlinear) dynamic game to approximate.
- `u::AbstractVector{<:AbstractVector{<:AbstractVector}}`: input sequence to linearize
  around; `u[t][i]` is the input of agent `i` at time `t`.
- `x0::AbstractVector`: initial state.
- `T_hor::Int64`: horizon length.
- `γ::Union{Nothing,AbstractVector{<:AbstractVector}}=nothing`: exogenous, possibly
  time-varying parameter passed to `game.f`, `game.J`, `game.gx`, `game.gu` and
  `game.gloc` at every timestep, with `γ[t]` the parameter at time `t` (e.g. a reference
  trajectory or an external disturbance). If `nothing` (the default), `game`'s functions
  are called without a trailing `γ` argument, so existing games are unaffected. If a
  `γ` is passed, `game`'s functions must all accept the trailing `γ` argument. See
  [`DynGame`](@ref).
"""
function LQapprox(game::DynGame,
    u::AbstractVector{<:AbstractVector{<:AbstractVector}},
    x0::AbstractVector,
    T_hor::Int64, # Could possibly be inferred as length of u
    γ::Union{Nothing,AbstractVector{<:AbstractVector}}=nothing # Parameters / exhogenous inputs for J, f, gx, gu
    )

    xt = copy(x0)
    has_param = γ !== nothing # If false, game's functions are not called with a trailing γ argument

    A =  [Matrix{Float64}(undef, game.nx, game.nx) for _ in 1:T_hor]
    B = [[Matrix{Float64}(undef, game.nx, nui) for nui in game.nu] for _ in 1:T_hor]
    Cx = [Matrix{Float64}(undef, game.mx, game.nx) for _ in 1:T_hor]
    bx = [Vector{Float64}(undef, game.mx) for _ in 1:T_hor]
    Cu = [[Matrix{Float64}(undef, game.mu, nui) for nui in game.nu] for _ in 1:T_hor]
    bu = [Vector{Float64}(undef, game.mu) for _ in 1:T_hor]
    Cloc = [[Matrix{Float64}(undef, game.mloc[i], game.nu[i]) for i in 1:game.N] for _ in 1:T_hor]
    bloc = [[Vector{Float64}(undef, mi) for mi in game.mloc] for _ in 1:T_hor]
    Q = [[Matrix{Float64}(undef, game.nx, game.nx) for i in 1:game.N] for _ in 1:T_hor-1]
    R = [[[Matrix{Float64}(undef, game.nu[i], game.nu[j]) for j in 1:game.N] for i in 1:game.N] for _ in 1:T_hor]
    q = [[Vector{Float64}(undef, game.nx) for _ in 1:game.N] for _ in 1:T_hor-1]
    r = [[Vector{Float64}(undef, game.nu[i]) for i in 1:game.N] for _ in 1:T_hor]


    for t=1:T_hor
        γt = has_param ? (γ[t],) : () # Trailing γ argument, empty tuple if game has no parameter

        # Linearize dynamics
        Jac_f_t = Zygote.jacobian(game.f, xt, u[t]..., γt...)
        A[t] .= Jac_f_t[1]
        B[t] .= Jac_f_t[2:game.N+1]

        # Linearize state constraints
        Cx[t] .= Zygote.jacobian(game.gx, xt, γt...)[1]
        bx[t] .= -game.gx(xt, γt...)

        # Linearize input constraints
        Cu[t] .= Zygote.jacobian(game.gu, u[t]..., γt...)[1:game.N]
        bu[t] .= -game.gu(u[t]..., γt...)

        # Linearize local input constraints

        Cloc[t] .= [Zygote.jacobian(game.gloc[i], u[t][i], γt...)[1] for i in 1:game.N]
        bloc[t] .= -[game.gloc[i](u[t][i], γt...) for i in 1:game.N]

        # Quadratize objectives
        for i=1:game.N
            H, Hij = block_hessian(game.J[i], xt, u[t]..., γt...)
            # t>=2 && println("size of Qti = $(Q[t-1][i])")
            if t>=2
                Q[t-1][i] .= Hij[(1,1)]
            end
            R[t][i] .= [Hij[(1+i, 1+j)] for j in 1:game.N] # Hessian with respect to ui, uj

            grad_Ji = Zygote.gradient(game.J[i], xt, u[t]..., γt...)
            if t>=2
                q[t-1][i] .= grad_Ji[1]
            end
            r[t][i] = grad_Ji[i+1]
        end

        # # Propagate dynamics
        xt = game.f(xt, u[t]..., γt...)
    end

    LQgame = DynLQGameTV(
        A=A,
        B=B,
        Q=Q,
        R=R,
        q=q,
        r=r,
        C_x=Cx,
        b_x=bx,
        C_loc=Cloc,
        b_loc=bloc,
        C_u=Cu,
        b_u=bu
    )     
    return LQgame
end

function LQ_bilevel_approx(game::DynGame,
    u::AbstractVector{<:AbstractVector{<:AbstractVector}},
    x0::AbstractVector,
    T_hor::Int64, # Could possibly be inferred as length of u
    γ::Union{Nothing,AbstractVector{<:AbstractVector}}=nothing; # Parameters / exhogenous inputs for J, f, gx, gu,
    leader::Int64=1
    )
    _, H_block = block_hessian(game.J[leader], xt, u[t]..., γt...)
    Qx = H_block[(1,1)]
    if t>=2
        Ql[t-1] .= Hlj[(1,1)]
    end
    R[t][i] .= [Hij[(1+i, 1+j)] for j in 1:game.N] # Hessian with respect to ui, uj

    grad_Ji = Zygote.gradient(game.J[i], xt, u[t]..., γt...)
    if t>=2
        q[t-1][i] .= grad_Ji[1]
    end
    r[t][i] = grad_Ji[i+1]

    # # Propagate dynamics
    xt = game.f(xt, u[t]..., γt...)
    # Quadratic approximation of leader objective

    Q0
end