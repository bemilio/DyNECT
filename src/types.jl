struct DyNEP # Dynamic Nash equilibrium problem
    nx::Int64
    nu::Vector{Int64}
    N::Int64
    ## Control problem quantities
    A::Matrix{Float64}
    B::Matrix{Float64} # B = [Bi[1], Bi[2], ...]
    Bi::Vector{SubArray{Float64,2}} # Views on B
    Q::Vector{Matrix{Float64}}
    R::Vector{Matrix{Float64}}
    P::Vector{Matrix{Float64}}
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

    function DyNEP(;
        A::Matrix{Float64},
        Bvec::Vector{Matrix{Float64}},
        Q::Vector{Matrix{Float64}},
        R::Vector{Matrix{Float64}},
        P::Union{Nothing,Vector{Matrix{Float64}}}=nothing,
        C_x::Matrix{Float64},
        b_x::Vector{Float64},
        C_loc_vec::Vector{Matrix{Float64}},
        b_loc_vec::Vector{Vector{Float64}},
        C_u_vec::Vector{Matrix{Float64}},
        b_u::Vector{Float64}
    )

        B = hcat(Bvec...)
        nx = size(A, 1)
        nu = map(x -> size(x, 2), Bvec)
        N = length(Bvec)
        # Create Bi as views on B
        Bi = Vector{SubArray}(undef, N)
        start_col = 1
        for i in 1:N
            n_cols = nu[i]
            Bi[i] = @view B[:, start_col:start_col+n_cols-1]
            start_col += n_cols
        end

        ## Constraints

        m_x = size(C_x, 1)

        C_loc = Matrix(BlockDiagonal(C_loc_vec))
        b_loc = vcat(b_loc_vec...)
        m_loc = map(x -> size(x, 1), C_loc_vec)
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
        end

        if isnothing(P)
            P = [zeros(nx, nx) for _ in 1:N]
        end

        new(nx, nu, N, A, B, Bi, Q, R, P, C_x, b_x, m_x, C_loc, b_loc, C_loc_i, b_loc_i, m_loc, C_u, b_u, C_u_i, m_u)
    end
end

