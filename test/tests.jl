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
    Bi = [rand(nx, nu[j]) for j in 1:N]
    T_hor = 4
    Γ, Γi, Θ = DyNECT.generate_prediction_model(A, Bi, T_hor)
    tol = 1e-6
    @test norm(Γ - hcat(Γi...)) < tol
    j = 2 # test on matrix associated to agent 2, arbitrary choice
    @test norm(Γi[j][1:nx, 1:nu[j]] - Bi[j]) < tol
    @test norm(Γi[j][end-nx+1:end, 1:nu[j]] - A^(T_hor - 1) * Bi[j]) < tol
end

@testset "DynLQGame2mpAVI.jl" begin
    using Random
    Random.seed!(1)

    nx = 3
    N = 3
    nu = [2, 3, 4]
    T_hor = 4

    A = rand(nx, nx)
    Bvec = [rand(nx, nu[j]) for j in 1:N]

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

    prob = DynLQGame(
        A=A,
        Bvec=Bvec,
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
    Bvec = [rand(nx, nu[j]) for j in 1:N]
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

    prob = DynLQGame(
        A=A,
        Bvec=Bvec,
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
