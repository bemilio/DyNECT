using DyNECT
using Test
using LinearAlgebra

@testset "generate_prediction_model.jl" begin
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

@testset "generate_mpVI.jl" begin
    nx = 3
    N = 3
    nu = [2, 3, 4]
    T_hor = 4

    A = rand(nx, nx)
    Bvec = [rand(nx, nu[j]) for j in 1:N]

    Q = [rand(nx, nx) for _ in 1:N]
    P = [rand(nx, nx) for _ in 1:N]
    R = [rand(nu[j], nu[j]) for j in 1:N]
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

    prob = DyNEP(
        A::Matrix{Float64},
        Bvec::Vector{Matrix{Float64}},
        Q::Vector{Matrix{Float64}},
        R::Vector{Matrix{Float64}},
        P::Vector{Matrix{Float64}},
        C_x::Matrix{Float64},
        b_x::Vector{Float64},
        C_loc_vec::Vector{Matrix{Float64}},
        b_loc_vec::Vector{Vector{Float64}},
        C_u_vec::Vector{Matrix{Float64}},
        b_u::Vector{Float64}
    )

    mpVI = generate_mpVI(prob, T_hor)

end