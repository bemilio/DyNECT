@doc raw"""
    DynLQGame2mpAVI(prob::DynLQGameTI, T_hor::Int64)
    Constructs the parametric variational inequality (mpVI) for a dynamic Nash equilibrium problem (DynLQGameTI) over a finite prediction horizon.

# Arguments
- `prob::DynLQGameTI`: Dynamic game structure containing system dynamics, cost, and constraints.
- `T_hor::Int64`: Prediction horizon.

# Returns
- `MPVI`: An instance of `ParametricDAQP.MPVI` representing the parametric variational inequality:
    - `H`: Block matrix for the quadratic part of the VI mapping.
    - `F`: Matrix mapping the initial state to the affine part of the VI mapping.
    - `f`: Constant affine vector in the VI mapping.
    - `D`: Constraint matrix for the stacked input sequence.
    - `E`: Matrix mapping the initial state to the affine part of the constraints.
    - `d`: Constant affine vector in the constraints.

The VI is of the form:  
``H u + F x_0 + f``,  subject to  ``D u \leq E x_0 + d``
where ``u`` is the stacked input sequence for all agents.
"""
function DynLQGame2mpAVI(prob::DynLQGameTI, T_hor::Int64)
    # Matrices defined as in Baghdalhorani, Benenati, Grammatico - Arxiv 2025

    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    Γ, Γi, Θ, c̅ = generate_prediction_model(prob.A, prob.B, T_hor; c=prob.c)

    # Define Q̅[i] = blkdg(I ⊗ Q[i], Pi)
    Q̅ = [BlockDiagonal([kron(I(T_hor - 1), prob.Q[i]), prob.P[i]]) for i in 1:prob.N]

    # Define R̅ as a block matrix where each block is R̅ᵢⱼ = I ⊗ Rᵢⱼ
    R̅ = BlockArray{Float64}(undef_blocks, [T_hor * prob.nu[i] for i = 1:prob.N], [T_hor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            R̅[Block(i, j)] = kron(I(T_hor), prob.R[i][j])
        end
    end

    # Define H (linear part of the VI mapping)
    H = BlockArray{Float64}(undef_blocks, [T_hor * prob.nu[i] for i = 1:prob.N], [T_hor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            H[Block(i, j)] = Γi[i]' * Q̅[i] * Γi[j]
        end
    end
    H = H + R̅

    # Define F (maps from x0 to the affine part of the VI mapping)
    F = vcat([Γi[i]' * Q̅[i] * Θ for i in 1:prob.N]...)

    # define f (affine part of the VI mapping)
    q̅ = [vcat(kron(ones(T_hor - 1), prob.q[i]), prob.p[i]) for i in 1:prob.N]
    r̅ = [kron(ones(T_hor), prob.r[i]) for i in 1:prob.N]
    f = vcat([Γi[i]' * (Q̅[i] * c̅ + q̅[i]) + r̅[i] for i in 1:prob.N]...)

    ## Constraints
    # Cₓ*x[t] ≤ bₓ ∀ t ==> C̅ₓΓu̅ <= b̅ₓ -C̅ₓΘx₀ - C̅ₓc̅
    # where C̅ₓ = I ⊗ Cₓ
    C̅_x = kron(I(T_hor), prob.C_x)
    # D_shar = row_stack([I ⊗ Du_i ; (I ⊗ Dx)*Γi ] )
    D_shar = hcat([[kron(I(T_hor), prob.C_u_i[i]);
        C̅_x * Γi[i]] for i in 1:prob.N]...)
    # Append local constraints 
    D = [D_shar;
        BlockDiagonal([kron(I(T_hor), prob.C_loc_i[i]) for i = 1:prob.N])]

    # Define E (maps from x0 to constraints)
    E = [zeros(T_hor * prob.m_u, prob.nx)        # Shared input constraints
        -1 * C̅_x * Θ;                            # State constraints
        zeros(sum(prob.m_loc) * T_hor, prob.nx)] # Local input constraints

    # Affine part of the constraints
    d = [kron(ones(T_hor), prob.b_u);            # Shared input constraints
        kron(ones(T_hor), prob.b_x) - C̅_x * c̅    # State constraints
        vcat([kron(ones(T_hor), prob.b_loc_i[i]) for i in 1:prob.N]...)] # Local input constraints
    # VI(H*x + F*x0 + f, D*x <= E*x0 + d)
    return mpAVI(Matrix{Float64}(H), F, f, D, E, d)
end

function DynLQGame2mpAVI(prob::DynLQGameTV) # time varying
    # prediction model: x̅ = Θx₀+(∑ Γᵢu̅ᵢ) + c̅
    Γ, Γi, Θ, c̅ = generate_prediction_model(prob.A, prob.B, prob.Thor; c=prob.c)

    # Define Q̅[i] = blkdg(Q[1][i],..., Q[T][i], P[i])
    Q̅ = [
        BlockDiagonal(vcat([prob.Q[t][i] for t in 1:prob.Thor-1], [prob.P[i]]))
        for i in 1:prob.N
    ]

    # Define R̅ as a block matrix where each block is R̅ᵢⱼ = blkdg(R[1][i][j],..., R[T][i][j])
    R̅ = BlockArray{Float64}(undef_blocks, [prob.Thor * prob.nu[i] for i = 1:prob.N], [prob.Thor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            R̅[Block(i, j)] = BlockDiagonal([prob.R[t][i][j] for t=1:prob.Thor])
        end
    end

    # Define H (linear part of the VI mapping)
    H = BlockArray{Float64}(undef_blocks, [prob.Thor * prob.nu[i] for i = 1:prob.N], [prob.Thor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            H[Block(i, j)] = Γi[i]' * Q̅[i] * Γi[j]
        end
    end
    H = H + R̅

    # Define F (maps from x0 to the affine part of the VI mapping)
    F = vcat([Γi[i]' * Q̅[i] * Θ for i in 1:prob.N]...)

    # define f (affine part of the VI mapping)
    q̅ = [vcat([prob.q[t][i] for t in 1:prob.Thor-1]..., prob.p[i]) for i in 1:prob.N]
    r̅ = [vcat([prob.r[t][i] for t in 1:prob.Thor]...) for i in 1:prob.N]
    f = vcat([Γi[i]' * (Q̅[i] * c̅ + q̅[i]) + r̅[i] for i in 1:prob.N]...)

    ## Constraints
    # Cᵗₓ*x[t] ≤ bᵗₓ ∀ t ==> C̅ₓΓu̅ <= b̅ₓ -C̅ₓΘx₀ - C̅ₓc̅
    # where C̅ₓ = blkdiag(C¹ₓ, ...,Cᵗₓ)
    C̅_x = BlockDiagonal(prob.C_x)
    # Collect both shared input constraints and state constraints:
    # D_shar = row_stack ([I ⊗ Cu[i]; 
    #                     (I ⊗ C̅ₓ)*Γ[i] ] )
    D_shar = hcat([[BlockDiagonal([prob.C_u[t][i] for t in 1:prob.Thor]);
        C̅_x * Γi[i]] for i in 1:prob.N]...)
    # Append local constraints 
    D = [D_shar;
        BlockDiagonal([ BlockDiagonal([prob.C_loc[t][i] for t in 1:prob.Thor] ) for i = 1:prob.N])]

    # Define E (maps from x0 to constraints)
    E = [zeros(prob.Thor * prob.m_u, prob.nx)        # Shared input constraints
        -1 * C̅_x * Θ;                            # State constraints
        zeros(sum(prob.m_loc) * prob.Thor, prob.nx)] # Local input constraints

    # Affine part of the constraints
    d = [vcat(prob.b_u...);            # Shared input constraints
        vcat(prob.b_x...) - C̅_x * c̅    # State constraints
        vcat([vcat([prob.b_loc[t][i] for t in 1:prob.Thor]...) for i in 1:prob.N]...)] # Local input constraints
    # VI(H*x + F*x0 + f, D*x <= E*x0 + d)
    return mpAVI(Matrix{Float64}(H), F, f, D, E, d)
end

@doc raw"""
    generate_prediction_model(A, B, T_hor)

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


function is_stabilizable(A, B)
    # Hautus lemma for stabilizability
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

function is_detectable(A, C)
    # Hautus lemma for stabilizability
    eigv_A = eigvals(A)
    n_x = size(A, 1)
    is_stbl = true
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

function compute_residual(prob::AVI, x::AbstractVector)
    y = x - (prob.H * x + prob.f)
    #TODO: Switch to Clarabel
    proj = DAQP.Model()
    DAQP.setup(proj, Matrix{Float64}(I, prob.n, prob.n), -y, Matrix{Float64}(prob.A), prob.b, Float64[], zeros(Cint, prob.m))
    x_transf, _, _, _ = DAQP.solve(proj)
    r = norm(x - x_transf)
    # qp = Clarabel.Solver()
    # eye = spdiagm(0 => ones(Float64, prob.n))
    # settings = Clarabel.Settings(verbose=false)
    # cone = [Clarabel.NonnegativeConeT(size(prob.A, 1))] # Sets all constraints to inequalities
    # Clarabel.setup!(qp, eye, -y, prob.A, prob.b, cone, settings)
    # results = solveQPRobust(qp)
    # r = norm(x - results.x)
    # if results.status != :Solved
    #     @warn "[compute_residual] QP solver returned $(results.status)"
    # end
    return r
end

```
Set the limits of the parameter space to an mpAVI
```
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