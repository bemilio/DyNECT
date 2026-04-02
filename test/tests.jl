using DyNECT
using Test
using LinearAlgebra
using ParametricDAQP
using BlockArrays
using MatrixEquations
using CommonSolve


@testset "DouglasRachford.jl" begin
    n = 2
    # [ 1 1      [-1  
    #  -1 1] x +   1] = 0
    # solution: [1;0]
    H = [1. 1.; -1. 1.]
    f = [-1., 1.]
    A = zeros(0, n)
    b = zeros(0)
    avi = DyNECT.AVI(H, f, A, b)
    params = DyNECT.IterativeSolverParams(verbose=true)
    solution = CommonSolve.solve(avi, DyNECT.DouglasRachford; params=params)
    tol = 1e-4
    @test norm(solution.x - [1., 0.]) < tol

end


@testset "generate_prediction_model.jl" begin
    using Random
    Random.seed!(1)

    nx = 3
    N = 3
    nu = [2, 3, 4]
    A = rand(nx, nx)
    B = [rand(nx, nu[j]) for j in 1:N]
    T_hor = 4
    Γ, Γi, Θ = DyNECT.generate_prediction_model(A, B, T_hor)
    tol = 1e-6
    @test norm(Γ - hcat(Γi...)) < tol
    j = 2 # test on matrix associated to agent 2, arbitrary choice
    @test norm(Γi[j][1:nx, 1:nu[j]] - B[j]) < tol
    @test norm(Γi[j][end-nx+1:end, 1:nu[j]] - A^(T_hor - 1) * B[j]) < tol
end

@testset "DynLQGame2mpAVI.jl" begin
    using Random
    Random.seed!(1)

    nx = 3
    N = 3
    nu = [2, 3, 4]
    T_hor = 4

    A = rand(nx, nx)
    B = [rand(nx, nu[j]) for j in 1:N]

    Q = [rand(nx, nx) for _ in 1:N]
    P = [rand(nx, nx) for _ in 1:N]
    # Define R_i with only non-zero element (i,i)
    R = [[zeros(nu[i], nu[j]) for j in 1:N] for i in 1:N]
    for i in 1:N
        R[i][i] .= 100.0 * I(nu[i])
    end

    # Constraints
    mx = 2
    C_x = rand(mx, nx)
    b_x = rand(mx)

    mloc = [1, 2, 1]
    C_loc_vec = [rand(mloc[i], nu[i]) for i in 1:N]
    b_loc_vec = [rand(mloc[i]) for i in 1:N]

    mu = 2
    C_u_vec = [rand(mu, nu[i]) for i in 1:N]
    b_u = rand(mu)

    prob = DynLQGameTI(
        A=A,
        B=B,
        Q=Q,
        R=R,
        P=P,
        C_x=C_x,
        b_x=b_x,
        C_loc_vec=C_loc_vec,
        b_loc_vec=b_loc_vec,
        C_u_vec=C_u_vec,
        b_u=b_u)

    mpVI = DynLQGame2mpAVI(prob, T_hor)

end

@testset "infinite_horizon_OLNE.jl" begin
    using Random
    Random.seed!(1)

    nx = 3
    N = 3
    nu = [2, 3, 4]
    T_hor = 2

    A = rand(nx, nx)
    B = [rand(nx, nu[j]) for j in 1:N]
    Q = Vector{Matrix{Float64}}(undef, N)
    for i = 1:N
        Q[i] = 0.1 * rand(nx, nx) + Matrix{Float64}(I(nx))
        Q[i] = Q[i] + Q[i]'
    end
    # Define R_i with only non-zero element (i,i)
    R = [[zeros(nu[i], nu[j]) for j in 1:N] for i in 1:N]
    for i in 1:N
        R[i][i] .= Matrix{Float64}(1.0 * I(nu[i]))
    end

    # No Constraints
    C_x = zeros(0, nx)
    b_x = zeros(0)

    C_loc_vec = [zeros(0, nu[i]) for i in 1:N]
    b_loc_vec = [zeros(0) for i in 1:N]

    C_u_vec = [zeros(0, nu[i]) for i in 1:N]
    b_u = zeros(0)

    prob = DynLQGameTI(
        A=A,
        B=B,
        Q=Q,
        R=R,
        C_x=C_x,
        b_x=b_x,
        C_loc_vec=C_loc_vec,
        b_loc_vec=b_loc_vec,
        C_u_vec=C_u_vec,
        b_u=b_u)

    x = 10 * rand(nx)

    # Solve infinite horizon problem 
    P, K = DyNECT.solveOLNE(prob)
    prob.P[:] = P[:]
    u_inf = vcat(K...) * x

    # Test solution
    P_ext, K_ext = DyNECT.solveExtendedARE(prob, K)
    err_P = 0.
    for i = 1:prob.N
        err_P = max(err_P, norm(P_ext[i][1:prob.nx, 1:prob.nx] + P_ext[i][1:prob.nx, prob.nx+1:end] - P[i]))
    end
    @test err_P < 1e-5

    # Generate and solve mpVI
    mpVI = DynLQGame2mpAVI(prob, T_hor)
    avi = DyNECT.AVI(mpVI, x)
    params = DyNECT.IterativeSolverParams(verbose=true)
    solution = CommonSolve.solve(avi, DyNECT.DouglasRachford; params=params)
    if solution.status == :Solved
        u = vcat(DyNECT.first_input_of_sequence(solution.x, prob.nu, prob.N, T_hor)...)
    else
        @error "VI solution not found"
    end

    @test norm(u_inf - u) < 1e-5
end


@testset "time_varying_LQ_game.jl" begin
    using Random
    Random.seed!(1)

    nx = 2
    N = 2
    nu = [2, 2, 2]
    T_hor = 2
    x0 = randn(nx)

    A = rand(nx, nx)
    B = [rand(nx, nu[j]) for j in 1:N]

    Q = [rand(nx, nx) for _ in 1:N]
    # Define R_i with only non-zero element (i,i)
    R = [[zeros(nu[i], nu[j]) for j in 1:N] for i in 1:N]
    for i in 1:N
        R[i][i] .= 100.0 * I(nu[i])
    end

    # Constraints
    mx = 2
    C_x = rand(mx, nx)
    b_x = rand(mx)

    mloc = [1, 2, 1]
    C_loc_vec = [rand(mloc[i], nu[i]) for i in 1:N]
    b_loc_vec = [rand(mloc[i]) for i in 1:N]

    mu = 1
    C_u_vec = [rand(mu, nu[i]) for i in 1:N]
    b_u = rand(mu)

    TIgame = DynLQGameTI(
        A=A,
        B=B,
        Q=Q,
        R=R,
        C_x=C_x,
        b_x=b_x,
        C_loc_vec=C_loc_vec,
        b_loc_vec=b_loc_vec,
        C_u_vec=C_u_vec,
        b_u=b_u)

    # Create AVI from time-invariant game
    TImpvi = DynLQGame2mpAVI(TIgame, T_hor)
    TIvi = DyNECT.AVI(TImpvi, x0)

    # Create AVI as time-varying game, where all quantities do not change over time
    TVgame = DynLQGameTV(
        A=[A for t in 1:T_hor],
        B=[B for t in 1:T_hor],
        Q=[Q for t in 1:T_hor-1],
        R=[R for t in 1:T_hor],
        C_x=[C_x for t in 1:T_hor],
        b_x=[b_x for t in 1:T_hor],
        C_loc=[C_loc_vec for t in 1:T_hor],
        b_loc=[b_loc_vec for t in 1:T_hor],
        C_u=[C_u_vec for t in 1:T_hor],
        b_u=[b_u for t in 1:T_hor])

    TVmpvi = DynLQGame2mpAVI(TVgame)
    TVvi = DyNECT.AVI(TVmpvi, x0)

    # The prediction model should be equal in both cases
    _, TIΓi, TIΘ, _ = DyNECT.generate_prediction_model(TIgame.A, TIgame.B, T_hor)
    _, TVΓi, TVΘ, _ = DyNECT.generate_prediction_model(TVgame.A, TVgame.B, T_hor)
    @test norm(TIΓi - TVΓi) < 1e-5
    @test norm(TIΘ - TVΘ) < 1e-5

    # The time-varying and time-invariant AVIs should be equal
    @test norm(TVvi.H - TIvi.H) < 1e-5
    @test norm(TVvi.f - TIvi.f) < 1e-5
    @test norm(TVmpvi.A - TImpvi.A) < 1e-5
    @test norm(TVmpvi.B - TImpvi.B) < 1e-5

    @test norm(TVvi.A - TIvi.A) < 1e-5
    @test norm(TVvi.b - TIvi.b) < 1e-5


end

@testset "nonlinear_game.jl" begin
    using Random
    Random.seed!(1)

    nx = 3
    N = 3
    nu = [2, 3, 4]
    T_hor = 4
    x0 = randn(nx)

    A = rand(nx, nx)
    B = [rand(nx, nu[j]) for j in 1:N]

    Q = [rand(nx, nx) for _ in 1:N]
    # Make symmetric
    foreach(i -> Q[i] .+= Q[i]', 1:N)

    R = [[rand(nu[i], nu[j]) for j in 1:N] for i in 1:N]
    # make symmetric
    foreach(i ->  R[i][i] .+= R[i][i]', 1:N)

    # Constraints
    mx = 2
    C_x = rand(mx, nx)
    b_x = rand(mx)

    mloc = [1, 2, 1]
    C_loc_vec = [rand(mloc[i], nu[i]) for i in 1:N]
    b_loc_vec = [rand(mloc[i]) for i in 1:N]

    mu = 2
    C_u_vec = [rand(mu, nu[i]) for i in 1:N]
    b_u = rand(mu)

    prob = DynLQGameTI(
        A=A,
        B=B,
        Q=Q,
        R=R,
        C_x=C_x,
        b_x=b_x,
        C_loc_vec=C_loc_vec,
        b_loc_vec=b_loc_vec,
        C_u_vec=C_u_vec,
        b_u=b_u)

    # Create AVI from time-invariant game
    mpvi = DynLQGame2mpAVI(prob, T_hor)
    vi = DyNECT.AVI(mpvi, x0)

    # Define non-LQ game object equal to the linear game
    f(x,u...) = A*x + sum(B[i] * u[i] for i=1:N)
    J = Vector{Function}(undef, N)
    for i in 1:N
        J[i] = let i = i
            (x, u...) -> 0.5 * x' * Q[i] * x +
                        sum(u[i]' * R[i][j] * u[j] for j in 1:N) -
                        0.5 * u[i]' * R[i][i] * u[i]
        end
    end
    gx(x) = C_x * x - b_x
    gloc = [ui -> C_loc_vec[i] * ui - b_loc_vec[i] for i in 1:N] 
    gu(u...) = sum([C_u_vec[i] * u[i] for i in 1:N ]) - b_u
    nonlinear_game = DyNECT.DynGame(f, J, gx, gu, gloc, nx, nu, mx, mu, mloc, N)
    # Create AVI as approximation of the nonlinear game at a generic point
    useq = [[zeros(nu[i]) for i in 1:N] for t in 1:T_hor]
    LQ_approx = DyNECT.LQapprox(nonlinear_game, useq, zeros(nx), T_hor)
    mpvi_approx = DyNECT.DynLQGame2mpAVI(LQ_approx)
    vi_approx = DyNECT.AVI(mpvi_approx, x0)

    # Test approximation of LQ game
    A_diff = maximum(norm.([At - A for At in LQ_approx.A]))
    @test A_diff < 1e-5
    B_diff = maximum(norm.([LQ_approx.B[t][i] - B[i] for i in 1:N, t in 1:T_hor]))
    @test B_diff < 1e-5
    Q_diff = maximum(norm.([LQ_approx.Q[t][i] - Q[i] for i in 1:N, t in 1:T_hor-1]))
    @test Q_diff < 1e-5
    R_diff = maximum(norm.([LQ_approx.R[t][i][j] - R[i][j] for i in 1:N, j in 1:N, t in 1:T_hor]))
    @test R_diff < 1e-5

    # The approximation and the original AVI should be equal
    @test norm(vi.H - vi_approx.H) < 1e-5
    @test norm(vi.f - vi_approx.f) < 1e-5
    @test norm(vi.A - vi_approx.A) < 1e-5
    @test norm(vi.b - vi_approx.b) < 1e-5

end