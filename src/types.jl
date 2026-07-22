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

function LQapprox(game::DynGame, u::Vector{Vector{Vector{Float64}}}, x0::Vector{Float64}, T_hor::Int64)
    xt = copy(x0)
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
        # Linearize dynamics
        Jac_f_t = Zygote.jacobian(game.f, xt, u[t]...)
        A[t] .= Jac_f_t[1]
        B[t] .= Jac_f_t[2:end]
        
        # Linearize state constraints
        Cx[t] .= Zygote.jacobian(game.gx, xt)[1]
        bx[t] .= -game.gx(xt)

        # Linearize input constraints
        Cu[t] .= Zygote.jacobian(game.gu, u[t]...)
        bu[t] .= -game.gu(u[t]...)

        # Linearize local input constraints
        Cloc[t] .= [Zygote.jacobian(game.gloc[i], u[t][i])[1] for i in 1:game.N] 
        bloc[t] .= -[game.gloc[i](u[t][i]) for i in 1:game.N]

        # Quadratize objectives
        for i=1:game.N
            H, Hij = block_hessian(game.J[i], xt, u[t]...)
            # t>=2 && println("size of Qti = $(Q[t-1][i])")
            if t>=2
                Q[t-1][i] .= Hij[(1,1)]
            end
            R[t][i] .= [Hij[(1+i, 1+j)] for j in 1:game.N] # Hessian with respect to ui, uj

            grad_Ji = Zygote.gradient(game.J[i], xt, u[t]...) 
            if t>=2
                q[t-1][i] .= grad_Ji[1]
            end
            r[t][i] = grad_Ji[i+1]
        end

        # # Propagate dynamics
        xt = game.f(xt, u[t]...)
    end

    LQgame = DynLQGameTV(
        A=A,
        B=B,
        Q=Q,
        R=R,
        q=q, # Defaults to 0
        r=r, #Defaults to 0
        C_x=Cx,
        b_x=bx,
        C_loc=Cloc,
        b_loc=bloc,
        C_u=Cu,
        b_u=bu
    )     
    return LQgame
end


@doc raw"""
DynLQGameTI

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
`DynLQGameTI(;
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
struct DynLQGameTI # Dynamic time-invariant Nash equilibrium problem
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

    function DynLQGameTI(; # Constructor for time-invariant game
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

#added v_static_mpGNE
struct StaticGNEGame #
    N::Int
    n::Vector{Int}
    Q::Vector{Vector{Matrix{Float64}}}
    q::Vector{Vector{Float64}}
    A_loc::Vector{Matrix{Float64}}
    b_loc::Vector{Vector{Float64}}
    A_sh::Vector{Matrix{Float64}}
    b_sh::Vector{Float64}
 
    function StaticGNEGame(
        Q::Vector{Vector{Matrix{Float64}}},
        q::AbstractVector,
        A_loc::AbstractVector,
        b_loc::AbstractVector,
        A_sh::AbstractVector,
        b_sh::AbstractVector{<:Real}
    )
        # Infer number of agents
        N = length(Q)
        # Infer size of decision variables
        n = [size(Q[i][i], 1) for i in 1:N]
        @assert N >= 2 "N must be at least 2 (Nash equilibrium requires multiple players)"
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

        return new(N, n, Q, q, A_loc, b_loc, A_sh, b_sh) 
    end 

    function StaticGNEGame(; Q, q, A_loc, b_loc, A_sh, b_sh) 
        StaticGNEGame(Q, q, A_loc, b_loc, A_sh, b_sh) 
    end 
end #
 
struct mpAVI #src usage
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

struct OptimalGNEP
    GNEP::StaticGNEGame
    ϕ::Function
    is_quadratic::Bool
    function OptimalGNEP(
        GNEP::StaticGNEGame,
        Q::AbstractMatrix,
        q::AbstractVector
    )
        n_tot = sum(GNEP.n)

        @assert size(Q, 1) == n_tot && size(Q, 2) == n_tot "[OptimalGNEP constructor] Q must be square with size sum(n)"
        @assert length(q) == n_tot "[OptimalGNEP constructor] q must have length sum(n)"

        @assert issymmetric(Q) "[OptimalGNEP constructor] Q must be symmetric"
        @assert isposdef(Q) "[OptimalGNEP constructor] Q must be positive definite"

        ϕ = x -> 0.5 * x' * Q * x + x' * q

        return new(GNEP, ϕ, true)
    end
    function OptimalGNEP(
        GNEP::StaticGNEGame,
        ϕ::Function
    )   
        # Check if ϕ is quadratic
        dummy_model = Model()
        @variable(dummy_model, y_test[1:sum(GNEP.n)])
        result = ϕ(y_test)
        result isa Union{Number,AffExpr,QuadExpr} || throw(ErrorException("[OptimalGNEP constructor] ϕ must return a scalar"))
        is_quadratic = result isa Union{Number, AffExpr, QuadExpr}
        
        return new(GNEP, ϕ, is_quadratic)
    end
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