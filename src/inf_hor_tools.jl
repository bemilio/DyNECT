using MatrixEquations
using BlockDiagonals
@enum method STEIN = 1

function solveOLNE(game::DyNEP; method=STEIN, max_iter=1000)
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
    Rinv = BlockDiagonal([inv(Ri) for Ri in game.R])
    RinvB = Rinv * BlockDiagonal(game.Bi)'
    # Matrix S: multiply by col(P_i) to get \sum B_i * R_i_inv * B_i' * P_i 
    S = game.B * RinvB # dim. (n_x,  N *nx)

    # Initialize to cooperative optimum
    R_all = BlockDiagonal(game.R)
    P_0, _, K, _, _ = ared(game.A, game.B, R_all, sum(game.Q), zeros(game.nx, game.nx))
    K = -K # Controller convention

    for k = 1:max_iter
        A_cl = game.A + game.B * K
        for i = 1:game.N
            try
                P[i] = sylvd(-game.A', A_cl, game.Q[i]) # Solves P[i] - A' * P[i] * A_cl = Q[i]
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
            println("[solveOLNE] Error = $err; Iter = $k")
            if err < eps_err
                break
            end
        end
    end
    s = cumsum([0; game.nu]) # gives [0, nu[1], nu[1]+nu[2], ..., sum(nu)]
    # Separate controllers into list
    Ki = [K[s[i]+1:s[i+1], :] for i in 1:game.N]
    return P, Ki
end
