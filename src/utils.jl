using ParametricDAQP # For the mpVI type
using BlockArrays
using BlockDiagonals
function generate_mpVI(prob::DyNEP, T_hor::Int64)
    # Matrices defined as in Baghdalhorani, Benenati, Grammatico - CDC 2025

    Γ, Γi, Θ = generate_prediction_model(prob.A, prob.Bi, T_hor)

    # Define Qi = blkdg(I ⊗ prob.Qi, Pi)
    Q = [BlockDiagonal([kron(I(T_hor - 1), prob.Q[i]), prob.P[i]]) for i in 1:prob.N]

    # Define Ri = I ⊗ prob.Ri
    R = map(Ri -> kron(I(T_hor), Ri), prob.R)

    ## VI Mapping

    # Define H 
    H = BlockArray{Float64}(undef_blocks, [T_hor * prob.nu[i] for i = 1:prob.N], [T_hor * prob.nu[i] for i = 1:prob.N])
    for i in 1:prob.N
        for j in 1:prob.N
            H[Block(i, j)] = Γi[i]' * Q[i] * Γi[j]
        end
    end
    sum!(H, BlockDiagonal(R))

    # Define F (maps from x0 to VI mapping)
    F = vcat([Γi[i]' * Q[i] * Θ for i in 1:prob.N]...)

    # define f (affine part of the VI mapping)
    f = zeros(sum(prob.nu) * T_hor)

    ## Constraints
    # D_shar = row_stack([I ⊗ Du_i ; (I ⊗ Dx)*Γi ] )
    D_shar = hcat([[kron(I(T_hor), prob.C_u_i[i]);
        kron(I(T_hor), prob.C_x) * Γi[i]] for i in 1:prob.N]...)
    # Append local constraints 
    D = [D_shar;
        BlockDiagonal([kron(I(T_hor), prob.C_loc_i[i]) for i = 1:prob.N])]

    # Define E (maps from x0 to constraints)
    E = [zeros(T_hor * prob.m_u, prob.nx);       # Shared input constraints
        kron(I(T_hor), prob.C_x) * Θ;            # State constraints
        zeros(sum(prob.m_loc) * T_hor, prob.nx)] # Local input constraints

    # Affine part of the constraints
    d = [kron(ones(T_hor), prob.b_u);            # Shared input constraints
        kron(ones(T_hor), prob.b_x);             # State constraints
        vcat([kron(ones(T_hor), prob.b_loc_i[i]) for i in 1:prob.N]...)] # Local input constraints

    # VI(H*x + F*x0 + f, D*x <= E*x0 + d)
    mpVI = MPVI(H, F, f, D, E, d) #TODO: Define constructor
end

function generate_prediction_model(A::Matrix{Float64}, Bi::Vector{<:AbstractMatrix{Float64}}, T_hor::Int)
    n_x = size(A, 1)
    n_u = map(x -> size(x, 2), Bi)
    N = length(Bi)
    Γ = zeros(n_x * T_hor, sum(n_u) * T_hor) # Maps the column stack of input sequences to state sequence
    Γi = Vector{SubArray}(undef, N) # Views into Γ
    start_col = 1
    for i in 1:N
        Γi[i] = @view Γ[:, start_col:start_col+n_u[i]*T_hor-1]
        start_col += n_u[i] * T_hor
    end
    Θ = vcat([A^t for t in 1:T_hor]...) # Maps the initial state to state sequence
    for i in 1:N
        Γi[i][1:n_x, 1:n_u[i]] .= Bi[i]
        for t in 1:T_hor-1
            current_rows = n_x*t+1:n_x*(t+1)
            previous_rows = n_x*(t-1)+1:n_x*t
            Γi[i][current_rows, n_u[i]+1:end] .= Γi[i][previous_rows, 1:end-n_u[i]] # Shift the previous block row by n_u
            Γi[i][current_rows, 1:n_u[i]] .= A * Γi[i][previous_rows, 1:n_u[i]] # Multiply the leftmost block of the previous row by A
        end
    end
    return Γ, Γi, Θ
end
