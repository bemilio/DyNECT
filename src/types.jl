@doc raw"""
    DynGame

Description of a general (possibly nonlinear) dynamic Nash equilibrium problem. Used as
the starting point for the iterative LQ approximation performed by [`LQapprox`](@ref),
which linearizes the dynamics and quadratizes the objectives around a given input/state
trajectory.

By default, every user-supplied function has the signature below (no `γ`):
- `f(x, u_1, ..., u_N)`: dynamics, `x⁺ = f(x, u)`.
- `J[i](x, u_1, ..., u_N)`: objective of agent `i`.
- `gx(x)`: state constraints, `gx(x) ≤ 0`.
- `gu(u_1, ..., u_N)`: shared input constraints, `gu(u) ≤ 0`.
- `gloc[i](u_i)`: local input constraints of agent `i`, `gloc[i](u_i) ≤ 0`.

Optionally, all of `f`, `J[i]`, `gx`, `gu`, `gloc[i]` can instead accept a trailing
argument `γ` (in that case *all* of them must, consistently), an exogenous, possibly
time-varying parameter that is not optimized over (e.g. a reference trajectory, a
disturbance, or another agent's prediction). This second calling convention is only used
when `γ` is explicitly passed to [`LQapprox`](@ref); games that don't need a
time-varying parameter can ignore this and keep the signatures above unchanged.

# Fields
- `f::Function`: dynamics.
- `J::Vector{Function}`: objective of each agent.
- `gx::Function`: state constraints.
- `gu::Function`: shared input constraints.
- `gloc::Vector{Function}`: local input constraints of each agent.
- `nx::Int64`: number of states.
- `nu::Vector{Int64}`: number of inputs per agent.
- `mx::Int64`: number of state constraints.
- `mu::Int64`: number of shared input constraints.
- `mloc::Vector{Int64}`: number of local input constraints per agent.
- `N::Int64`: number of agents.
"""
struct DynGame
    ## Control problem quantities
    f::Function # dynamics
    J::Vector{Function} # Objective per each agent
    gx::Function # State constraints gₓ(x)≤0
    gu::Function # Shared input constraints gᵤ(u)≤0
    gloc::Vector{Function} # Local input constraints gᵢ(uᵢ)≤0

    # Helper quantities. TODO: Can these be inferred?
    nx::Int64 #number of states
    nu::Vector{Int64} #number of inputs
    mx::Int64 #number of input constraints
    mu::Int64 #number of shared constraints
    mloc::Vector{Int64} #number of local constraints
    N::Int64 #Number of agents

end




@doc raw"""

DynLQGame

Time-invariant tynamic Nash Equilibrium Problem.
The linear dynamics are given by the equation:
```math
    x^+ = A x + \sum_i(B_i u_i) + c
```
The objective for agent i is 
```math
    J_i = \|x[T]\|^2_{P_i} + \sum_t \left\{\frac{1}{2}\|x[t]\|^2_{Q_i} + \frac{1}{2}\|u_i[t]\|^2_{R_{ii}} + 
    \sum_{j\neq i}\left(\langle u_i[t], R_{ij}u_j[t]\rangle\right) + \langle x[t], q_i\rangle + \langle u_i[t], r_i\rangle \right\}
``` 
# Constructor
`DynLQGame(;
    A::Matrix{Float64},
    Bvec::Vector{Matrix{Float64}},
    c::Vector{Float64},
    Q::Vector{Matrix{Float64}},
    R::Vector{Vector{Matrix{Float64}}}
    P::Union{Nothing,Vector{Matrix{Float64}}}=nothing,
    q::Vector{Vector{Float64}},
    r::Vector{Vector{Float64}},
    C_x::Matrix{Float64},
    b_x::Vector{Float64},
    C_loc_vec::Vector{Matrix{Float64}},
    b_loc_vec::Vector{Vector{Float64}},
    C_u_vec::Vector{Matrix{Float64}},
    b_u::Vector{Float64}
    )`
# Fields
- `nx::Int64`: Number of states.
- `nu::Vector{Int64}`: Number of inputs per agent.
- `N::Int64`: Number of agents.
- `A::Matrix{Float64}`: State transition matrix.
- `B::Matrix{Float64}`: Input matrix (horizontally concatenated for all agents).
- `B::Vector{SubArray{Float64,2}}`: Views on `B` for each agent.
- `Q::Vector{Matrix{Float64}}`: State cost matrices for each agent, size nₓ×nₓ.
- `R::Vector{Vector{Matrix{Float64}}}`: Input cost matrix, the i,j-th element is size nᵤⁱ×nᵤʲ.
- `P::Vector{Matrix{Float64}}`: Terminal cost matrices for each agent, size nₓ×nₓ.
- `C_x::Matrix{Float64}`: State constraint matrix, size mₓ×nₓ.
- `b_x::Vector{Float64}`: State constraint bounds, vector size mₓ.
- `m_x::Int64`: Number of state constraints.
- `C_loc::Matrix{Float64}`: Local input constraint matrix (block diagonal), size (∑mᵤⁱ)×(∑nᵤⁱ) .
- `b_loc::Vector{Float64}`: Local input constraint bounds, vector size ∑mᵤⁱ
- `C_loc_i::Vector{SubArray{Float64,2}}`: Views on `C_loc` for each agent, each size mᵤⁱ×nᵤⁱ
- `b_loc_i::Vector{SubArray{Float64,1}}`: Views on `b_loc` for each agent, eqch vector size mᵤⁱ
- `m_loc::Vector{Int64}`: Number of local constraints per agent.
- `C_u::Matrix{Float64}`: Shared input constraint matrix: size mˢʰ⨯(∑nᵤⁱ).
- `b_u::Vector{Float64}`: Shared input constraint bounds, vector size mˢʰ.
- `C_u_i::Vector{SubArray{Float64,2}}`: Views on `C_u` for each agent, each size mˢʰ⨯nᵤⁱ
- `m_u::Int64`: Number of shared input constraints.
"""
struct DynLQGame # Dynamic time-invariant Nash equilibrium problem
    nx::Int64
    nu::Vector{Int64}
    N::Int64
    ## Control problem quantities
    A::Matrix{Float64}
    B::Vector{Matrix{Float64}} 
    c::Vector{Float64}
    Q::Vector{Matrix{Float64}}
    R::Vector{Vector{Matrix{Float64}}}
    P::Vector{Matrix{Float64}}
    q::Vector{Vector{Float64}}
    r::Vector{Vector{Float64}}
    p::Vector{Vector{Float64}}

    ## Constraints
    # State constraints in the form C_ux*x <= b_x
    C_x::Matrix{Float64}
    b_x::Vector{Float64}
    m_x::Int64 # number of state constraints
    # Local input constraints in the form C_loc*u <= b_loc
    C_loc::Matrix{Float64} # C_loc = diag(C_loc_i[1], C_loc_i[2], ...)
    b_loc::Vector{Float64} # b_loc = [b_loc_i[1], b_loc_i[2], ...] 
    C_loc_i::Vector{SubArray{Float64,2}} # Views on C_loc
    b_loc_i::Vector{SubArray{Float64,1}} # Views on b_loc
    m_loc::Vector{Int64} # Number of local constraints per each agent
    # Shared input constraints in the form C_u * u <= b_u
    C_u::Matrix{Float64} # C_u = [C_u_i[1], C_u_i[2], ...]
    b_u::Vector{Float64}
    C_u_i::Vector{SubArray{Float64,2}} # Views on C_u
    m_u::Int64 # number of input constraints

    function DynLQGame(; # Constructor for time-invariant game
        A::Matrix{Float64},
        B::Vector{Matrix{Float64}},
        c::Union{Nothing,Vector{Float64}}=nothing, # Defaults to 0
        Q::Vector{Matrix{Float64}},
        R::Vector{Vector{Matrix{Float64}}},
        P::Union{Nothing,Vector{Matrix{Float64}}}=nothing, # Defaults to 0
        q::Union{Nothing,Vector{Vector{Float64}}}=nothing, # Defaults to 0
        r::Union{Nothing,Vector{Vector{Float64}}}=nothing, #Defaults to 0
        p::Union{Nothing,Vector{Vector{Float64}}}=nothing, #Defaults to 0
        C_x::Matrix{Float64},
        b_x::Vector{Float64},
        C_loc_vec::Vector{Matrix{Float64}},
        b_loc_vec::Vector{Vector{Float64}},
        C_u_vec::Vector{Matrix{Float64}},
        b_u::Vector{Float64}
    )

        ## Extract sizes
        nx = size(A, 1)
        nu = map(x -> size(x, 2), B)
        N = length(B)
        m_x = size(C_x, 1)
        m_loc = map(x -> size(x, 1), C_loc_vec)

        ## Dimensionality checks
        @assert size(A, 1) == size(A, 2) "A must be square (nx × nx)"
        @assert all(size(B_i, 1) == nx for B_i in B) "Each matrix B_i in B must have nx rows"
        @assert length(Q) == N "Q must have one matrix per agent"
        @assert all(size(Qi, 1) == nx && size(Qi, 2) == nx for Qi in Q) "Each Q[i] must be nx × nx"
        @assert length(R) == N "R must be a N-long list of N matrices, where N is the # of agents (one weight R[i][j] for each agent i,j)"
        @assert all(length(Ri) == N for Ri in R) "Each R[i] must be a list of length N (one weight R[i][j] for each agent i,j)"
        @assert all(size(R[i][j], 1) == nu[i] && size(R[i][j], 2) == nu[j] for i in 1:N for j in 1:N) "Each R[i][j] must be nu[i] × nu[j]"
        if !isnothing(P)
            @assert length(P) == N "P must have one matrix per agent"
            @assert all(size(Pi, 1) == nx && size(Pi, 2) == nx for Pi in P) "Each P[i] must be nx × nx"
        else
            P = [zeros(nx, nx) for _ in 1:N]
        end
        @assert size(C_x, 2) == nx "C_x must have nx columns"
        @assert size(C_x, 1) == length(b_x) "Number of C_x rows must match length of b_x"
        @assert length(C_loc_vec) == N "C_loc_vec must have one matrix per agent"
        @assert length(b_loc_vec) == N "b_loc_vec must have one vector per agent"
        @assert all(size(C_loc_vec[i], 2) == nu[i] for i in 1:length(B)) "Each C_loc_vec[i] must have nu[i] columns"
        @assert all(size(C_loc_vec[i], 1) == length(b_loc_vec[i]) for i in 1:length(B)) "Each C_loc_vec[i] rows must match length of b_loc_vec[i]"
        @assert length(C_u_vec) == N "C_u_vec must have one matrix per agent"
        @assert all(size(C_u_vec[i], 2) == nu[i] for i in 1:length(B)) "Each C_u_vec[i] must have nu[i] columns"
        @assert size(b_u, 1) == size(C_u_vec[1], 1) "b_u length must match number of shared input constraints (rows of C_u_vec[1])"
        @assert all(size(C_u_vec[i], 1) == size(C_u_vec[1], 1) for i in 1:N) "All C_u_vec[i] must have the same number of rows"
        if !isnothing(c)
            @assert length(c) == nx "c must have length nx"
        else
            c = zeros(nx)
        end
        if !isnothing(q)
            @assert length(q) == N "q must have one vector per agent"
            @assert all(length(qi) == nx for qi in q) "Each q[i] must have length nx"
        else
            q = [zeros(nx) for _ in 1:N]
        end
        if !isnothing(r)
            @assert length(r) == N "r must have one vector per agent"
            @assert all(length(ri) == nu[i] for (i, ri) in enumerate(r)) "Each r[i] must have length nu[i]"
        else
            r = [zeros(nu[i]) for i in 1:N]
        end
        if !isnothing(p)
            @assert length(p) == N "p must have one vector per agent"
            @assert all(length(pi) == nx for pi in p) "Each p[i] must have length nx"
        else
            p = [zeros(nx) for _ in 1:N]
        end

        ## Construct matrices
      

        ## Constraints
        C_loc = Matrix(BlockDiagonal(C_loc_vec))
        b_loc = vcat(b_loc_vec...)
        # Create C_loc_i, b_loc_i as views on C_loc, b_loc
        start_row = 1
        start_col = 1
        C_loc_i = Vector{SubArray}(undef, N)
        b_loc_i = Vector{SubArray}(undef, N)
        for i in 1:N
            n_rows = m_loc[i]
            n_cols = nu[i]
            C_loc_i[i] = @view C_loc[start_row:start_row+n_rows-1, start_col:start_col+n_cols-1]
            b_loc_i[i] = @view b_loc[start_row:start_row+n_rows-1]
            start_row += n_rows
            start_col += n_cols
        end

        C_u = hcat(C_u_vec...)
        m_u = size(C_u, 1)

        # Create C_u_i as views on C_u
        start_col = 1
        C_u_i = Vector{SubArray}(undef, N)
        for i in 1:N
            n_cols = nu[i]
            C_u_i[i] = @view C_u[:, start_col:start_col+n_cols-1]
            start_col += n_cols
        end

        new(nx, nu, N, A, B, c, Q, R, P, q, r, p, C_x, b_x, m_x, C_loc, b_loc, C_loc_i, b_loc_i, m_loc, C_u, b_u, C_u_i, m_u)
    end
end

# TODO: Fix documentation
@doc raw""" 
DynLQGameTV

Time-varying tynamic Nash Equilibrium Problem.
The linear dynamics are given by the equation:
```math
    x^+ = Aᵗ x + \sum_i(B_i u_i) + c
```
The objective for agent i is 
```math
    J_i = \|x[T]\|^2_{P_i} + \sum_t \left\{\frac{1}{2}\|x[t]\|^2_{Q_i} + \frac{1}{2}\|u_i[t]\|^2_{R_{ii}} + 
    \sum_{j\neq i}\left(\langle u_i[t], R_{ij}u_j[t]\rangle\right) + \langle x[t], q_i\rangle + \langle u_i[t], r_i\rangle \right\}
``` 
# Constructor
`DynLQGameTV(;
    A::Matrix{Float64},
    Bvec::Vector{Matrix{Float64}},
    c::Vector{Float64},
    Q::Vector{Matrix{Float64}},
    R::Vector{Vector{Matrix{Float64}}}
    P::Union{Nothing,Vector{Matrix{Float64}}}=nothing,
    q::Vector{Vector{Float64}},
    r::Vector{Vector{Float64}},
    C_x::Matrix{Float64},
    b_x::Vector{Float64},
    C_loc_vec::Vector{Matrix{Float64}},
    b_loc_vec::Vector{Vector{Float64}},
    C_u_vec::Vector{Matrix{Float64}},
    b_u::Vector{Float64}
    )`
# Fields
- `nx::Int64`: Number of states.
- `nu::Vector{Int64}`: Number of inputs per agent.
- `N::Int64`: Number of agents.
- `A::Matrix{Float64}`: State transition matrix.
- `B::Matrix{Float64}`: Input matrix (horizontally concatenated for all agents).
- `B::Vector{SubArray{Float64,2}}`: Views on `B` for each agent.
- `Q::Vector{Matrix{Float64}}`: State cost matrices for each agent, size nₓ×nₓ.
- `R::Vector{Vector{Matrix{Float64}}}`: Input cost matrix, the i,j-th element is size nᵤⁱ×nᵤʲ.
- `P::Vector{Matrix{Float64}}`: Terminal cost matrices for each agent, size nₓ×nₓ.
- `C_x::Matrix{Float64}`: State constraint matrix, size mₓ×nₓ.
- `b_x::Vector{Float64}`: State constraint bounds, vector size mₓ.
- `m_x::Int64`: Number of state constraints.
- `C_loc::Matrix{Float64}`: Local input constraint matrix (block diagonal), size (∑mᵤⁱ)×(∑nᵤⁱ) .
- `b_loc::Vector{Float64}`: Local input constraint bounds, vector size ∑mᵤⁱ
- `C_loc_i::Vector{SubArray{Float64,2}}`: Views on `C_loc` for each agent, each size mᵤⁱ×nᵤⁱ
- `b_loc_i::Vector{SubArray{Float64,1}}`: Views on `b_loc` for each agent, eqch vector size mᵤⁱ
- `m_loc::Vector{Int64}`: Number of local constraints per agent.
- `C_u::Matrix{Float64}`: Shared input constraint matrix: size mˢʰ⨯(∑nᵤⁱ).
- `b_u::Vector{Float64}`: Shared input constraint bounds, vector size mˢʰ.
- `C_u_i::Vector{SubArray{Float64,2}}`: Views on `C_u` for each agent, each size mˢʰ⨯nᵤⁱ
- `m_u::Int64`: Number of shared input constraints.
"""
struct DynLQGameTV # Dynamic time-Variant Nash equilibrium problem
    nx::Int64
    nu::Vector{Int64}
    N::Int64
    Thor::Int64
    m_x::Int64 # number of state constraints
    m_loc::Vector{Int64} # Number of local constraints per each agent
    m_u::Int64 # number of shared input constraints

    ## Control problem quantities
    A::Vector{Matrix{Float64}}
    B::Vector{Vector{Matrix{Float64}}} # B[t] = [Bᵗ₁, ... Bᵗₙ] for each t ≤ Thor
    c::Vector{Vector{Float64}}
    Q::Vector{Vector{Matrix{Float64}}}
    R::Vector{Vector{Vector{Matrix{Float64}}}} # R[t][i] = [Rᵗᵢ₁, ... Rᵗᵢₙ] that is, for each t, i: List of input cost couplings between agents
    P::Vector{Matrix{Float64}} # Terminal quadratic cost
    q::Vector{Vector{Vector{Float64}}}
    r::Vector{Vector{Vector{Float64}}}
    p::Vector{Vector{Float64}} # Terminal linear cost

    ## Constraints
    # State constraints in the form Cᵗₓ*x ≤ bᵗₓ
    C_x::Vector{Matrix{Float64}}
    b_x::Vector{Vector{Float64}}
    # Local input constraints in the form Cₗᵗ * u ≤ bᵗₗ , ∀i
    C_loc::Vector{Vector{Matrix{Float64}}} 
    b_loc::Vector{Vector{Vector{Float64}}}
    # Shared input constraints in the form ∑ᵢ Cᵗᵢ * u ≤ bᵗ
    C_u::Vector{Vector{Matrix{Float64}}}
    b_u::Vector{Vector{Float64}}

    function DynLQGameTV(; # Constructor for time-invariant game
        A::Vector{Matrix{Float64}},
        B::Vector{Vector{Matrix{Float64}}},
        c::Union{Nothing,Vector{Vector{Float64}}}=nothing, # Defaults to 0
        Q::Vector{Vector{Matrix{Float64}}},
        R::Vector{Vector{Vector{Matrix{Float64}}}},
        P::Union{Nothing,Vector{Matrix{Float64}}}=nothing, # Defaults to 0
        q::Union{Nothing,Vector{Vector{Vector{Float64}}}}=nothing, # Defaults to 0
        r::Union{Nothing,Vector{Vector{Vector{Float64}}}}=nothing, #Defaults to 0
        p::Union{Nothing,Vector{Vector{Float64}}}=nothing, #Defaults to 0
        C_x::Vector{Matrix{Float64}},
        b_x::Vector{Vector{Float64}},
        C_loc::Vector{Vector{Matrix{Float64}}},
        b_loc::Vector{Vector{Vector{Float64}}},
        C_u::Vector{Vector{Matrix{Float64}}},
        b_u::Vector{Vector{Float64}}
    )

        ## Extract sizes
        Thor = length(A)
        nx = size(A[1], 1)
        nu = map(x -> size(x, 2), B[1])
        N = length(B[1])
        m_x = size(C_x[1], 1)
        m_loc = map(x -> size(x, 1), C_loc[1])
        m_u = size(C_u[1][1],1)

        # helpers
        all_t = 1:Thor
        all_ag = 1:N

        ## Dimensionality checks
        @assert all(size(A[t], 1) == size(A[t], 2) for t in all_t ) "A must be square (nx × nx)"
        @assert all(size(B[t][i], 1) == nx for i in all_ag, t in all_t) "Each matrix B_i in B must have nx rows"
        @assert length(Q) == Thor-1 "Q must have length T_hor-1"
        @assert all(length(Qt) == N for Qt in Q) "Qt must have one matrix per agent"
        @assert all(size(Q[t][i], 1) == nx && size(Q[t][i], 2) == nx for i in all_ag,t in 1:Thor-1 ) "Each Q[i] must be nx × nx"
        @assert all(length(Rt) == N for Rt in R) "R[t] must be a N-long list of N matrices, where N is the # of agents "
        @assert all(length(R[t][i]) == N  for i in all_ag, t in all_t) "Each R[t][i] must be a list of length N (one weight R[i][j] for each agent i,j)"
        @assert all(size(R[t][i][j], 1) == nu[i] && size(R[t][i][j], 2) == nu[j] for t in all_t, i in all_ag, j in all_ag) "Each R[i][j] must be nu[i] × nu[j]"
        if !isnothing(P)
            @assert length(P) == N "P must have one matrix per agent"
            @assert all(size(Pi, 1) == nx && size(Pi, 2) == nx for Pi in P) "Each P[i] must be nx × nx"
        else
            P = [zeros(nx, nx) for _ in 1:N]
        end
        @assert all(size(C_xt, 2) == nx for C_xt in C_x) "C_x must have nx columns"
        @assert all(size(C_x[t], 1) == length(b_x[t]) for t in all_t) "Number of C_x rows must match length of b_x"
        @assert length(C_loc) == Thor "C_loc must be a list long as the horizon length"
        @assert all(length(C_loc_t) == N for C_loc_t in C_loc) "C_loc must have one matrix per agent"
        @assert length(b_loc) == Thor "b_loc must be a list long as the horizon length"
        @assert all(length(b_loc_t) == N for b_loc_t in b_loc) "b_loc[t] must have one vector per agent"
        @assert all(size(C_loc[t][i], 2) == nu[i] for i in all_ag, t in all_t) "Each C_loc_vec[t][i] must have nu[i] columns"
        @assert all(size(C_loc[t][i], 1) == length(b_loc[t][i]) for i in all_ag, t in all_t) "Each C_loc[t][i] rows must match length of b_loc[t][i]"
        @assert length(C_u) == Thor "C_u_vec must have one element per timestep"
        @assert all(length(C_u[t]) == N for t in all_t) "C_u[t] must have one matrix per agent"

        @assert all(size(C_u[t][i], 2) == nu[i] for i in all_ag, t in all_t) "Each C_u[t][i] must have nu[i] columns"
        @assert all(size(b_u[t], 1) == m_u for t in all_t) "b_u[t] length must match number of shared input constraints"
        @assert all(size(C_u[t][i], 1) == m_u for i in all_ag, t in all_t) "All C_u_vec[i] must have the same number of rows"
        if !isnothing(c)
            @assert all(length(c[t]) == nx for t in all_t) "c[t] must have length nx"
        else
            c = [zeros(nx) for _ in all_t]
        end
        if !isnothing(q)
            @assert length(q) == Thor-1 "q must be of length T_hor - 1"
            @assert all(length(q[t]) == N for t in 1:Thor-1) "q[t] must have one vector per agent"
            @assert all(length(q[t][i]) == nx for i in all_ag, t in 1:Thor-1) "Each q[t][i] must have length nx"
        else
            q = [[zeros(nx) for _ in all_ag] for _ in 1:Thor-1]
        end
        if !isnothing(r)
            @assert all(length(r[t]) == N for t in all_t) "r[t] must have one vector per agent"
            @assert all(length(r[t][i]) == nu[i] for i in all_ag, t in all_t) "Each r[t][i] must have length nu[i]"
        else
            r = [[zeros(nu[i]) for i in 1:N] for _ in all_t]
        end
        if !isnothing(p)
            @assert length(p) == N "p must have one vector per agent"
            @assert all(length(pi) == nx for pi in p) "Each p[i] must have length nx"
        else
            p = [zeros(nx) for _ in 1:N]
        end

        new(nx, nu, N, Thor, m_x, m_loc, m_u, A, B, c, Q, R, P, q, r, p, C_x, b_x, C_loc, b_loc, C_u, b_u)
    end
end

@doc raw"""
    LQGNEP

Linear-quadratic generalized Nash equilibrium problem (GNEP). Agent ``i``'s decision
``x_i \in \mathbb{R}^{n_i}`` solves
```math
\min_{x_i} \quad \frac{1}{2}x_i^\top Q_{ii} x_i + \sum_{j\neq i} x_i^\top Q_{ij} x_j + q_i^\top x_i \\
\mathrm{s.t.} \qquad A_{\mathrm{loc},i} x_i \leq b_{\mathrm{loc},i} \\
\qquad \qquad \textstyle\sum_{j=1}^N A_{\mathrm{sh},j} x_j \leq b_\mathrm{sh}
```
i.e. each agent has its own quadratic objective (coupled to the others through the
off-diagonal blocks of `Q`) and its own local constraints, and all agents share a common
set of shared/coupling constraints.

Optionally, the game's data can depend affinely on an external parameter
``\gamma \in \mathbb{R}^{n_\gamma}``:
```math
q_i(\gamma) = q_i + Q_{q_i\gamma}\gamma, \qquad
b_{\mathrm{loc},i}(\gamma) = b_{\mathrm{loc},i} + B_{\mathrm{loc},i,\gamma}\gamma, \qquad
b_\mathrm{sh}(\gamma) = b_\mathrm{sh} + B_{\mathrm{sh},\gamma}\gamma.
```
The fields above (`N`, `n`, `Q`, `q`, `A_loc`, `b_loc`, `A_sh`, `b_sh`) hold the nominal
(``\gamma=0``) game data. Setting `n_γ = 0` (the default, obtained by omitting `Q_qγ`,
`B_loc_γ` and `B_sh_γ` in the constructor) removes the ``\gamma``-dependence entirely, giving
a plain (non-parametric) GNEP. ``\gamma`` is an external parameter, not one of the agents'
decision variables: a parametric `LQGNEP` is meant to be used as the followers' game in a
[`BilevelGame`](@ref), whose leader chooses ``\gamma``.

As in [`mpAVI`](@ref), ``\gamma`` is restricted to a set ``\{C\gamma \leq d\} \cap \{lb \leq \gamma \leq ub\}``.
The box bounds `ub`, `lb` default to ``\pm 100`` per component, and the polytope `C`, `d` default to
empty (no polytope constraints) when not given explicitly.

# Fields
- `N::Int`: number of agents.
- `n::Vector{Int}`: decision-variable dimension of each agent.
- `Q::Vector{Vector{Matrix{Float64}}}`: quadratic cost terms; `Q[i][j]` is the block coupling
  agent `i`'s cost to agent `j`'s decision, size `n[i] × n[j]`.
- `q::Vector{Vector{Float64}}`: linear cost terms, `q[i]` has length `n[i]`.
- `A_loc::Vector{Matrix{Float64}}`: local constraint matrices, `A_loc[i]` has size `m_loc[i] × n[i]`.
- `b_loc::Vector{Vector{Float64}}`: local constraint bounds, `b_loc[i]` has length `m_loc[i]`.
- `A_sh::Vector{Matrix{Float64}}`: shared constraint matrices, `A_sh[i]` has size `m_sh × n[i]`.
- `b_sh::Vector{Float64}`: shared constraint bounds, length `m_sh`.
- `n_γ::Int`: number of external parameters ``\gamma`` (`0` if there is no ``\gamma``-dependence).
- `Q_qγ::Vector{Matrix{Float64}}`: sensitivity of `q[i]` to ``\gamma``, size `n[i] × n_γ`.
- `B_loc_γ::Vector{Matrix{Float64}}`: sensitivity of `b_loc[i]` to ``\gamma``, size `m_loc[i] × n_γ`.
- `B_sh_γ::Matrix{Float64}`: sensitivity of `b_sh` to ``\gamma``, size `m_sh × n_γ`.
- `ub::Vector{Float64}`, `lb::Vector{Float64}`: box bounds on ``\gamma``, length `n_γ`. Default to ±100.
- `C::Matrix{Float64}`, `d::Vector{Float64}`: polytope constraint on ``\gamma``, `C\gamma \leq d`.
  `C` has size `m_γ × n_γ`, `d` has length `m_γ`. Default to no constraints (`m_γ = 0`).
"""
struct LQGNEP
    N::Int
    n::Vector{Int}
    Q::Vector{Vector{Matrix{Float64}}}
    q::Vector{Vector{Float64}}
    A_loc::Vector{Matrix{Float64}}
    b_loc::Vector{Vector{Float64}}
    A_sh::Vector{Matrix{Float64}}
    b_sh::Vector{Float64}
    n_γ::Int
    Q_qγ::Vector{Matrix{Float64}}
    B_loc_γ::Vector{Matrix{Float64}}
    B_sh_γ::Matrix{Float64}
    ub::Vector{Float64}
    lb::Vector{Float64}
    C::Matrix{Float64}
    d::Vector{Float64}

    function LQGNEP(
        Q::Vector{Vector{Matrix{Float64}}},
        q::AbstractVector,
        A_loc::AbstractVector,
        b_loc::AbstractVector,
        A_sh::AbstractVector,
        b_sh::AbstractVector{<:Real};
        Q_qγ::Union{AbstractVector,Nothing}=nothing,
        B_loc_γ::Union{AbstractVector,Nothing}=nothing,
        B_sh_γ::Union{AbstractMatrix,Nothing}=nothing,
        ub::Union{AbstractVector,Nothing}=nothing,
        lb::Union{AbstractVector,Nothing}=nothing,
        C::Union{AbstractMatrix,Nothing}=nothing,
        d::Union{AbstractVector,Nothing}=nothing,
    )
        # Infer number of agents
        N = length(Q)
        # Infer size of decision variables
        n = [size(Q[i][i], 1) for i in 1:N]
        @assert all(n .> 0) "All player dimensions n[i] must be positive"
        @assert all(length(Qi) == N for Qi in Q) "Each element of Q must have length $(N) (one matrix per agent)"
        for i in 1:N
            for j in 1:N
                @assert size(Q[i][j], 1) == n[i] "Q[$i][$j] must have n[$i]=$(n[i]) rows, got $(size(Q[i][j], 1))"
                @assert size(Q[i][j], 2) == n[j] "Q[$i][$j] must have n[$j]=$(n[j]) columns, got $(size(Q[i][j], 2))"
            end
        end

        @assert length(q) == N "q must have length N (one vector per player), got $(length(q))"
        for i in 1:N
            @assert length(q[i]) == n[i] "q[$i] must have length n[$i]=$(n[i]), got $(length(q[i]))"
        end

        @assert length(A_loc) == N "A_loc must have length N (one matrix per agent)"
        @assert length(b_loc) == N "b_loc must have length N (one matrix per agent)"
        for i in 1:N
            @assert size(A_loc[i], 2) == n[i] "A_loc[$i] must have $(n[i]) columns, got $(size(A_loc[i], 2))"
            @assert size(A_loc[i], 1) == length(b_loc[i]) "A_loc[$i] has $(size(A_loc[i], 1)) rows but b_loc[$i] has length $(length(b_loc[i]))"
        end

        @assert length(A_sh) == N "A_sh must have length N (one block per player)"
        m_sh = size(A_sh[1], 1)
        for i in 1:N
            @assert size(A_sh[i], 1) == m_sh "All A_sh[i] must have m_sh=$m_sh rows. A_sh[$i] has $(size(A_sh[i], 1))"
            @assert size(A_sh[i], 2) == n[i] "A_sh[$i] must have n[$i]=$(n[i]) columns, got $(size(A_sh[i], 2))"
        end

        @assert length(b_sh) == m_sh "b_sh must have m_sh=$m_sh elements, got $(length(b_sh))"

        # Infer n_γ from whichever γ-sensitivity argument is given (0 if none are given)
        n_γ = 0
        if !isnothing(Q_qγ) && !isempty(Q_qγ)
            n_γ = size(Q_qγ[1], 2)
        elseif !isnothing(B_loc_γ) && !isempty(B_loc_γ)
            n_γ = size(B_loc_γ[1], 2)
        elseif !isnothing(B_sh_γ)
            n_γ = size(B_sh_γ, 2)
        end

        Q_qγ = isnothing(Q_qγ) ? [zeros(n[i], n_γ) for i in 1:N] : Matrix{Float64}.(Q_qγ)
        B_loc_γ = isnothing(B_loc_γ) ? [zeros(length(b_loc[i]), n_γ) for i in 1:N] : Matrix{Float64}.(B_loc_γ)
        B_sh_γ = isnothing(B_sh_γ) ? zeros(m_sh, n_γ) : Matrix{Float64}(B_sh_γ)
        ub = isnothing(ub) ? 100 .* ones(n_γ) : Vector{Float64}(ub)
        lb = isnothing(lb) ? -100 .* ones(n_γ) : Vector{Float64}(lb)
        C = isnothing(C) ? zeros(0, n_γ) : Matrix{Float64}(C)
        d = isnothing(d) ? zeros(0) : Vector{Float64}(d)

        @assert length(Q_qγ) == N "[LQGNEP constructor] Q_qγ must have one matrix per agent"
        for i in 1:N
            @assert size(Q_qγ[i]) == (n[i], n_γ) "[LQGNEP constructor] Q_qγ[$i] must be n[$i] × n_γ = $(n[i]) × $n_γ, got $(size(Q_qγ[i]))"
        end
        @assert length(B_loc_γ) == N "[LQGNEP constructor] B_loc_γ must have one matrix per agent"
        for i in 1:N
            @assert size(B_loc_γ[i]) == (length(b_loc[i]), n_γ) "[LQGNEP constructor] B_loc_γ[$i] must be m_loc[$i] × n_γ = $(length(b_loc[i])) × $n_γ, got $(size(B_loc_γ[i]))"
        end
        @assert size(B_sh_γ) == (m_sh, n_γ) "[LQGNEP constructor] B_sh_γ must be m_sh × n_γ = $m_sh × $n_γ, got $(size(B_sh_γ))"
        @assert length(ub) == n_γ "[LQGNEP constructor] ub must have length n_γ = $n_γ, got $(length(ub))"
        @assert length(lb) == n_γ "[LQGNEP constructor] lb must have length n_γ = $n_γ, got $(length(lb))"
        @assert all(lb .<= ub) "[LQGNEP constructor] lb must be componentwise ≤ ub"
        @assert size(C, 2) == n_γ "[LQGNEP constructor] C must have n_γ = $n_γ columns, got $(size(C, 2))"
        @assert size(C, 1) == length(d) "[LQGNEP constructor] C must have as many rows as length(d) = $(length(d)), got $(size(C, 1))"

        return new(N, n, Q, q, A_loc, b_loc, A_sh, b_sh, n_γ, Q_qγ, B_loc_γ, B_sh_γ, ub, lb, C, d)
    end

    function LQGNEP(; Q, q, A_loc, b_loc, A_sh, b_sh,
        Q_qγ=nothing, B_loc_γ=nothing, B_sh_γ=nothing, ub=nothing, lb=nothing, C=nothing, d=nothing)
        LQGNEP(Q, q, A_loc, b_loc, A_sh, b_sh;
            Q_qγ=Q_qγ, B_loc_γ=B_loc_γ, B_sh_γ=B_sh_γ, ub=ub, lb=lb, C=C, d=d)
    end
end #

# Returns an LQGNEP with n_γ = 0, evaluated at a given parameter vector γ
function LQGNEP(pg::LQGNEP, γ::AbstractVector)
    q = [pg.q[i] + pg.Q_qγ[i] * γ for i in 1:pg.N]
    b_loc = [pg.b_loc[i] + pg.B_loc_γ[i] * γ for i in 1:pg.N]
    b_sh = pg.b_sh + pg.B_sh_γ * γ
    return LQGNEP(pg.Q, q, pg.A_loc, b_loc, pg.A_sh, b_sh)
end

@doc raw"""
    mpAVI

Multi-parametric Affine Variational Inequality of the form
```math
\mathrm{VI}(Hx + F\theta + f,\ Ax \leq B\theta + b)
```
with parameter set ``\theta \in \{C\theta \leq d\} \cap \{lb \leq \theta \leq ub\}``.

# Fields
- `H`: Mapping matrix, size `n × n`.
- `F`: Parameter-to-mapping matrix, size `n × n_θ`.
- `f`: Constant affine term, length `n`.
- `A`: Constraint matrix, size `m × n`.
- `B`: Parameter-to-constraint matrix, size `m × n_θ`.
- `b`: Constraint right-hand side, length `m`.
- `C`: Parameter polytope constraint matrix.
- `d`: Parameter polytope right-hand side.
- `ub`, `lb`: Box bounds on the parameter `θ`, length `n_θ`. Defaults to ±100.
- `n`: Number of decision variables.
- `m`: Number of constraints.
- `n_θ`: Number of parameters.
"""
struct mpAVI #src usage
    # VI(Hx + Fθ + f, Ax ≤ Bθ + b)
    # With θ ∈ { Cθ ≤ d } ∩ { lb ≤ θ ≤ ub }
    H::AbstractMatrix # size = n_x * n_x
    F::AbstractMatrix # size =  n_x * n_θ
    f::AbstractVector # size =  n_x
    A::AbstractMatrix # size = n_constr * n_x
    B::AbstractMatrix # size = n_constr * n_θ 
    b::AbstractVector # size = n_constr
    C::AbstractMatrix
    d::AbstractVector
    ub::AbstractVector
    lb::AbstractVector
    n::Int
    m::Int
    n_θ::Int

    function mpAVI(
        H::AbstractMatrix{Float64},
        F::AbstractMatrix{Float64},
        f::AbstractVector{Float64},
        A::AbstractMatrix{Float64},
        B::AbstractMatrix{Float64},
        b::AbstractVector{Float64};
        C::Union{AbstractMatrix{Float64},Nothing}=nothing,
        d::Union{AbstractVector{Float64},Nothing}=nothing,
        ub::Union{AbstractVector{Float64},Nothing}=nothing,
        lb::Union{AbstractVector{Float64},Nothing}=nothing,
    )
        n_θ = size(F, 2)
        n = size(H, 1)
        m = size(A, 1)  # Number of constraints

        if isnothing(C) || isnothing(d)
            # Default to infinitely large box constraints
            C = zeros(0, n_θ)
            d = zeros(0)
        end
        if isnothing(ub)
            ub = 100 .* ones(n_θ)
        end
        if isnothing(lb)
            lb = -100 .* ones(n_θ)
        end

        # Sanity checks
        @assert size(A, 2) == n "[MPVI constructor] Columns of A ($(size(A, 2))) must equal number of decision variables ($n)"
        @assert size(B, 1) == m "[MPVI constructor] Rows of B ($(size(B, 1))) must match rows of A ($m)"
        @assert length(b) == m "[MPVI constructor] Length of b ($(length(b))) must match rows of A ($m)"
        @assert size(F, 1) == n "[MPVI constructor] F has $(size(F, 1)) rows. It must be equal to the number of decision variables ($n)"
        @assert size(H, 2) == n "[MPVI constructor] H must be square"
        @assert length(f) == n "[MPVI constructor] Length of f ($(length(f))) must match number of decision variables ($n)"
        @assert size(C, 2) == n_θ "[MPVI constructor] C has $(size(C, 2)) columns. It must be equal to the number of parameters ($n_θ)"
        @assert length(d) == size(C, 1) "[MPVI constructor] C has $(size(C, 1)) rows. It must be equal to the size of d ($d)"

        return new(H, F, f, A, B, b, C, d, ub, lb, n, m, n_θ)
    end
end #

function mpAVI(pgame::LQGNEP)
    game_no_par = LQGNEP(pgame, zeros(pgame.n_γ))
    avi = AVI(game_no_par)
    F = vcat(pgame.Q_qγ...)
    B = vcat(pgame.B_sh_γ, pgame.B_loc_γ...)
    return mpAVI(avi.H, F, avi.f, avi.A, B, avi.b; C=pgame.C, d=pgame.d, ub=pgame.ub, lb=pgame.lb)
end

@doc raw"""
    AVI

Affine Variational Inequality of the form
```math
\mathrm{VI}(Hx + f,\ Ax \leq b)
```
Find ``x`` such that ``\langle Hx + f,\, y - x \rangle \geq 0`` for all feasible ``y``.

# Fields
- `H`: Mapping matrix, size `n × n`.
- `f`: Affine term, length `n`.
- `A`: Constraint matrix, size `m × n`.
- `b`: Constraint right-hand side, length `m`.
- `n`: Number of decision variables.
- `m`: Number of constraints.

Can be constructed directly or from an `mpAVI` at a given parameter value via `AVI(mpAVI, θ)`.
"""
struct AVI
    # VI(Hx + f, Ax <= b)
    # where f, b are the last rows of F,B, respect.
    H::AbstractMatrix # size = n_x * n_x
    f::AbstractVector # size =  n_x
    A::AbstractMatrix # size = n_constr * n_x
    b::AbstractVector # size = n_constr
    n::Int
    m::Int

    function AVI(
        H::AbstractMatrix{Float64},
        f::AbstractVector{Float64},
        A::AbstractMatrix{Float64},
        b::AbstractVector{Float64}
    )
        n = size(H, 1)
        m = size(A, 1)  # Number of constraints

        # Sanity checks
        @assert size(A, 2) == n "[AVI constructor] Columns of A must equal number of decision variables"
        @assert length(b) == m "[AVI constructor] Length of b must match rows of A"
        @assert size(H, 2) == n "[AVI constructor] H must be square"
        @assert length(f) == n "[AVI constructor] Length of f must match number of decision variables"

        return new(H, f, A, b, n, m)
    end
end

function AVI(mpAVI::mpAVI, θ::AbstractVector)
    return AVI(mpAVI.H, mpAVI.F * θ + mpAVI.f, mpAVI.A, mpAVI.B * θ + mpAVI.b)
end

function AVI(game::LQGNEP)
    n = game.n
    N = game.N
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
    # Assemble shared constraints
    A_sh = hcat(game.A_sh...)
    A = vcat(A_sh, Matrix(A_loc))
    b = vcat(game.b_sh, b_loc)

    return AVI(Matrix{Float64}(H), f, Matrix{Float64}(A), b)
end

@doc raw"""
    BilevelGame

Bilevel optimization problem, subject to the variable `x` being a generalized Nash equilibrium:
```math
\min_\gamma \phi(\gamma, x^*) \\
\mathrm{s.t.} \qquad A_\gamma \gamma \leq b_\gamma \\
 \qquad x^*\in \mathrm{GNE}(\gamma)
```
The set `\mathrm{GNE}(\gamma)` is the Nash equilibrium set of a [`LQGNEP`](@ref) at the given
``\gamma`` — see that type's docstring for how ``\gamma`` enters its cost and constraints.

`BilevelGame` is constructed from a [`LQGNEP`](@ref). Passing one with `n_γ = 0` makes the
`BilevelGame` reduce to the problem of selecting the point in the (unparametrized)
  Nash equilibrium set of `LowLevelGNEP` that minimizes `ϕ`.

If `\phi` is a quadratic function, its defining matrices and vectors are stored such that
```math
\phi(\gamma, x) = \frac{1}{2}\gamma^\top Q_\gamma \gamma + \gamma^\top Q_{\gamma x} x + \frac{1}{2} x^\top Q_x x + x^\top q_x + \gamma ^\top q_\gamma.
```

# Fields
- `LowLevelGNEP::LQGNEP`: followers game.
- `A_γ::Matrix{Float64}`: leader constraint matrix, size `m_γ × n_γ`.
- `b_γ::Vector{Float64}`: leader constraint bounds, length `m_γ`.
- `n_γ::Int`: number of leader decision variables (`0` when `LowLevelGNEP.n_γ == 0`).
- `ϕ::Function`: leader's objective, called as `ϕ(γ, x)`.
- `is_quadratic::Bool`: whether `ϕ` is quadratic.
- `Qγ::Union{Matrix{Float64},Nothing}`: leader-leader quadratic weight, size `n_γ × n_γ` (if `is_quadratic`).
- `Qγx::Union{Matrix{Float64},Nothing}`: leader-follower coupling, size `n_γ × sum(n)` (if `is_quadratic`).
- `Qx::Union{Matrix{Float64},Nothing}`: follower-follower quadratic weight, size `sum(n) × sum(n)` (if `is_quadratic`).
- `qγ::Union{Vector{Float64},Nothing}`: leader linear term, length `n_γ` (if `is_quadratic`).
- `qx::Union{Vector{Float64},Nothing}`: follower linear term, length `sum(n)` (if `is_quadratic`).
"""
struct BilevelGame
    LowLevelGNEP::LQGNEP
    A_γ::Matrix{Float64}
    b_γ::Vector{Float64}
    n_γ::Int
    ϕ::Function # Objective of the leader, ϕ(γ, x)
    is_quadratic::Bool
    Qγ::Union{Matrix{Float64},Nothing}
    Qγx::Union{Matrix{Float64},Nothing}
    Qx::Union{Matrix{Float64},Nothing}
    qγ::Union{Vector{Float64},Nothing}
    qx::Union{Vector{Float64},Nothing}

    function BilevelGame(
        LowLevelGNEP::LQGNEP,
        Qx::AbstractMatrix,
        qx::AbstractVector;
        A_γ::Union{AbstractMatrix,Nothing}=nothing,
        b_γ::Union{AbstractVector,Nothing}=nothing,
        Qγ::Union{AbstractMatrix,Nothing}=nothing,
        Qγx::Union{AbstractMatrix,Nothing}=nothing,
        qγ::Union{AbstractVector,Nothing}=nothing,
    )
        n_γ, n_tot = _retrieve_bilevel_n_γ_and_n_tot(LowLevelGNEP)

        A_γ, b_γ = _retrieve_bilevel_leader_constraints(n_γ, A_γ, b_γ)
        Qγ = isnothing(Qγ) ? zeros(n_γ, n_γ) : Matrix{Float64}(Qγ)
        Qγx = isnothing(Qγx) ? zeros(n_γ, n_tot) : Matrix{Float64}(Qγx)
        qγ = isnothing(qγ) ? zeros(n_γ) : Vector{Float64}(qγ)

        @assert size(Qx, 1) == n_tot && size(Qx, 2) == n_tot "[BilevelGame constructor] Qx must be square with size sum(n)"
        @assert length(qx) == n_tot "[BilevelGame constructor] qx must have length sum(n)"
        @assert issymmetric(Qx) "[BilevelGame constructor] Qx must be symmetric"
        @assert size(Qγ) == (n_γ, n_γ) "[BilevelGame constructor] Qγ must be n_γ × n_γ"
        @assert size(Qγx) == (n_γ, n_tot) "[BilevelGame constructor] Qγx must be n_γ × sum(n)"
        @assert length(qγ) == n_γ "[BilevelGame constructor] qγ must have length n_γ"

        full_Q = Symmetric([Qγ Qγx; Qγx' Qx])
        @assert isposdef(full_Q) "[BilevelGame constructor] the combined quadratic form [Qγ Qγx; Qγx' Qx] must be positive definite"

        ϕ = (γ, x) -> 0.5 * γ' * Qγ * γ + γ' * Qγx * x + 0.5 * x' * Qx * x + x' * qx + γ' * qγ

        return new(LowLevelGNEP, A_γ, b_γ, n_γ, ϕ, true, Qγ, Qγx, Qx, qγ, qx)
    end

    function BilevelGame(
        LowLevelGNEP::LQGNEP,
        ϕ::Function;
        A_γ::Union{AbstractMatrix,Nothing}=nothing,
        b_γ::Union{AbstractVector,Nothing}=nothing,
    )
        n_γ, n_tot = _retrieve_bilevel_n_γ_and_n_tot(LowLevelGNEP)

        A_γ, b_γ = _retrieve_bilevel_leader_constraints(n_γ, A_γ, b_γ)

        # Check if ϕ is quadratic
        dummy_model = Model()
        @variable(dummy_model, γ_test[1:n_γ])
        @variable(dummy_model, x_test[1:n_tot])
        result = ϕ(γ_test, x_test)
        result isa Union{Number,AffExpr,QuadExpr} || throw(ErrorException("[BilevelGame constructor] ϕ must return a scalar"))
        is_quadratic = result isa Union{Number,AffExpr,QuadExpr}

        return new(LowLevelGNEP, A_γ, b_γ, n_γ, ϕ, is_quadratic, nothing, nothing, nothing, nothing, nothing)
    end
end

# Returns (n_γ, n_tot) for a LQGNEP LowLevelGNEP.
_retrieve_bilevel_n_γ_and_n_tot(LowLevelGNEP::LQGNEP) = (LowLevelGNEP.n_γ, sum(LowLevelGNEP.n))

# Validates/defaults the leader's own constraint on γ (A_γ γ ≤ b_γ), shared by the BilevelGame constructors.
function _retrieve_bilevel_leader_constraints(
    n_γ::Int,
    A_γ::Union{AbstractMatrix,Nothing},
    b_γ::Union{AbstractVector,Nothing},
)
    if isnothing(A_γ) && isnothing(b_γ)
        return zeros(0, n_γ), zeros(0)
    end
    @assert !isnothing(A_γ) && !isnothing(b_γ) "[BilevelGame constructor] A_γ and b_γ must be given together"
    @assert size(A_γ, 2) == n_γ "[BilevelGame constructor] A_γ must have n_γ=$(n_γ) columns"
    @assert size(A_γ, 1) == length(b_γ) "[BilevelGame constructor] A_γ rows must match length of b_γ"
    return Matrix{Float64}(A_γ), Vector{Float64}(b_γ)
end

@doc raw"""
    IterativeSolverParams

Configuration for iterative AVI/VI solvers.

# Fields
- `max_iter`: Maximum number of iterations (default: `10000`).
- `stepsize`: Step size; `nothing` lets each solver pick its own default.
- `tol`: Convergence tolerance on the VI residual (default: `1e-6`).
- `warmstart`: Warm-start strategy — `:NoWarmStart` (zeros) or `:UnconstrainedSolution`.
- `verbose`: Print progress every 1000 iterations when `true` (default: `false`).
- `time_limit`: Wall-clock time limit in seconds (default: `100.0`).
"""
mutable struct IterativeSolverParams
    max_iter::Int
    stepsize::Union{Float64,Nothing}
    tol::Float64
    warmstart::Symbol
    verbose::Bool
    time_limit::Float64
end

function IterativeSolverParams(; max_iter::Int=10000,
    stepsize::Union{Float64,Nothing}=nothing,
    tol::Float64=1e-6,
    warmstart::Symbol=:NoWarmStart,
    verbose::Bool=false,
    time_limit::Float64=1e2)
    return IterativeSolverParams(max_iter, stepsize, tol, warmstart, verbose, time_limit)
end

#added optimal GNE selection #usage
struct OptimalGNEResult
    θ_star::Vector{Float64} #Optimal parameter value
    u_star::Vector{Float64} #Optimal equilibrium (GNE)  
    φ_star::Float64         #Performance metric value
    region_id::Int          #Which critical region contained optimum
    all_candidates::Vector  #All regional candidates for analysis
end

function Base.show(io::IO, result::OptimalGNEResult)
    println(io, "OptimalGNEResult")
    println(io, "  θ*: $(round.(result.θ_star; digits=6))")
    println(io, "  u*: $(round.(result.u_star; digits=6))")
    println(io, "  φ*: $(round(result.φ_star; digits=8))")
    println(io, "  region: $(result.region_id)")
    println(io, "  candidates: $(length(result.all_candidates))")
end