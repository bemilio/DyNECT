@enum method STEIN = 1

function solveOLNE(game::DynLQGame; method=STEIN, max_iter=1000, stepsize=0.1)
    if any(norm(qi) > 1e-5 for qi in game.q) ||
       any(norm(ri) > 1e-5 for ri in game.r) ||
       norm(game.c) > 1e-5
        @warn "Affine part in objective or dynamics is not supported for the infinite-horizon OLNE solution and will be ignored."
    end

    if any(norm(game.R[i][j]) > 1e-5 && i != j for i in 1:game.N for j in 1:game.N)
        @warn "Cross-weights between input objectives (off-diagonal elements in R) are not supported for the infinite-horizon OLNE solution and will be ignored."
    end

    #  Check if basic assumptions are satisfied
    eps_err = 1e-5
    if minimum(abs.(eigvals(game.A))) < eps_err
        @warn "[solveInfHorOL] The matrix A is singular"
    end
    for i in 1:game.N
        if !is_stabilizable(game.A, game.Bi[i])
            @warn "[solveInfHorOL] The system is not stabilizable for agent $i"
        end
        if !is_detectable(game.A, game.Q[i])
            @warn "[solveInfHorOL] The system is not detectable for agent $i"
        end
    end
    P = [zeros(game.nx, game.nx) for _ in 1:game.N]
    Rinv = BlockDiagonal([inv(game.R[i][i]) for i in 1:game.N])
    RinvB = Rinv * BlockDiagonal(game.Bi)'
    # Matrix S: multiply by col(P_i) to get \sum B_i * R_i_inv * B_i' * P_i 
    S = game.B * RinvB # dim. (n_x,  N *nx)

    # Initialize to cooperative optimum
    R_all = BlockDiagonal([game.R[i][i] for i in 1:game.N])
    P_0, _, K, _, _ = ared(game.A, game.B, R_all, sum(game.Q), zeros(game.nx, sum(game.nu)))
    K = -K # Controller convention
    for k = 1:max_iter
        A_cl = game.A + game.B * K
        for i = 1:game.N
            try
                P[i] = (1 - stepsize) * P[i] + stepsize * sylvd(-game.A', A_cl, game.Q[i]) # Solves P[i] - A' * P[i] * A_cl = Q[i]
            catch e
                disp("[solveInfHorOL] An error occurred while solving the Sylvester equation: " + e.message)
                return nothing
                break
            end
        end
        A_cl = (I(game.nx) + S * vcat(P...)) \ game.A # Closed loop matrix
        K = -RinvB * vcat(P...) * A_cl # K_i = Ri_inv * Bi'* Pi * A_cl
        if mod(k, 10) == 0
            # Test solution: Check if (9) [Freiling-Jank-Kandil '99] is satisfied
            err = 0
            A_cl = (I(game.nx) + S * vcat(P...)) \ game.A # Closed loop matrix
            for i = 1:game.N
                err = err + norm(game.Q[i] - P[i] + game.A' * P[i] * A_cl)
            end
            # println("[solveOLNE] Error = $err; Iter = $k")
            if err < eps_err
                if maximum(abs.(eigvals(A_cl))) > 1.0
                    @warn "[solveInfHorOL] The solution is found, but it is not stabilizing"
                end
                break
            end
        end
    end
    s = cumsum([0; game.nu]) # gives [0, nu[1], nu[1]+nu[2], ..., sum(nu)]
    # Separate controllers into list
    Ki = [K[s[i]+1:s[i+1], :] for i in 1:game.N]
    return P, Ki
end

function solveExtendedARE(game::DynLQGame, K::Vector{<:AbstractMatrix})
    nx = game.nx
    P_ext = [zeros(2 * nx, 2 * nx) for _ in 1:game.N]
    K_ext = [zeros(nu, 2 * nx) for nu in game.nu]
    for i = 1:game.N
        others = [j for j in 1:game.N if j != i]
        BjKj = sum([game.Bi[j] * K[j] for j in others])
        A_ol = game.A + game.B * vcat(K...)
        A_ext = [game.A BjKj;
            zeros(nx, nx) A_ol]
        B_ext = [game.Bi[i]; zeros(nx, game.nu[i])]
        Q_ext = [game.Q[i] zeros(nx, nx); zeros(nx, nx) zeros(nx, nx)]
        R_ext = game.R[i][i]
        P_ext[i], _, K_ext[i], _, _ = ared(A_ext, B_ext, R_ext, Q_ext, zeros(2 * nx, game.nu[i]))
    end
    return P_ext, K_ext
end

function solveCLNE(game::DynLQGame; max_iter=1000, stepsize=0.1)
    if any(norm(qi) > 1e-5 for qi in game.q) ||
       any(norm(ri) > 1e-5 for ri in game.r) ||
       norm(game.c) > 1e-5
        @warn "Affine part in objective or dynamics is not supported for the infinite-horizon CLNE solution and will be ignored."
    end

    if any(norm(game.R[i][j]) > 1e-5 && i != j for i in 1:game.N for j in 1:game.N)
        @warn "Cross-weights between input objectives (off-diagonal elements in R) are not supported for the infinite-horizon CLNE solution and will be ignored."
    end

    # Initialize to collaborative LQR
    _, _, K_collab, _,_ = ared(game.A, hcat(game.Bi...), Matrix{Float64}(I, sum(game.nu), sum(game.nu)), Matrix(I, game.nx, game.nx), zeros(game.nx, sum(game.nu)))
    K_collab = - K_collab
    eps_err = 1e-5

    idx = cumsum([1; game.nu])
    K = [K_collab[idx[i]:(idx[i+1]-1), :] for i in 1:game.N]
    P = [zeros(game.nx, game.nx) for _ in 1:game.N]
    
    for k=1:max_iter
        for i in 1:game.N
            # println("Solving are, iter. $k, agent $i")
            Acl_i = game.A + sum(game.Bi[j] * K[j] for j in 1:game.N) - game.Bi[i] * K[i]
            P[i], _, K_ared, _, _ = ared(Acl_i, game.Bi[i], game.R[i][i], game.Q[i], zeros(game.nx, game.nu[i]))
            K_ared = -K_ared
            K[i] = (1 - stepsize) * K[i] + stepsize * K_ared
        end
        # Test solution: Check residual from simultaneous ARE solution
        riccati_residual = 0.0

        for i in 1:game.N
            Acl_i = game.A + sum(game.Bi[j] * K[j] for j in 1:game.N) - game.Bi[i] * K[i]
            _, _, K_ared, _, _ = ared(Acl_i, game.Bi[i], game.R[i][i], game.Q[i], zeros(game.nx, game.nu[i]))
            K_ared = -K_ared
            riccati_residual = riccati_residual + norm(K[i] - K_ared)
        end

        if riccati_residual < eps_err
            break
        end
    end

    return P, K

end