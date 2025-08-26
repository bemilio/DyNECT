using Revise
using DyNECT
using ParametricDAQP
using LinearAlgebra
using Random
using Statistics
using BlockDiagonals
using Infiltrator

Random.seed!(1)

function make_random_prob(nx::Int, N::Int, nu::Vector{Int}, mx::Int, mu::Vector{Int}, T_hor::Int)
    A = randn(nx, nx)
    Bvec = [randn(nx, nu[i]) for i in 1:N]

    Q = [
        let M = randn(nx, nx)
            M' * M + 1e-3 * I(nx)
        end for _ in 1:N
    ]
    q = [randn(nx) for _ in 1:N]
    P = [zeros(nx, nx) for _ in 1:N]
    R = [[zeros(nu[i], nu[j]) for j in 1:N] for i in 1:N]

    # Define Rᵢᵢ = rᵢ*I, with rᵢ large enough to ensure strong monotonicity:
    # rᵢ > (∑ ‖Γᵢ'(Q̅ᵢ+Q̅ⱼ')½Γⱼ‖ - λ̲(Γᵢ'(Q̅ᵢ+Q̅ᵢ')½Γᵢ)) 
    _, Γ, _, _ = DyNECT.generate_prediction_model(A, Bvec, T_hor)
    Q̅ = [BlockDiagonal([kron(I(T_hor - 1), Q[i]), P[i]]) for i in 1:N]

    for i in 1:N
        sum_norm = 0.0
        for j in 1:N
            M_ij = Γ[i]' * (Q̅[i] + Q̅[j]') * Γ[j] ./ 2
            sum_norm += norm(M_ij)
        end
        M_ii = Γ[i]' * (Q̅[i] + Q̅[i]') * Γ[i] ./ 2
        lambda_min = minimum(eigvals(M_ii))
        r_i = sum_norm - lambda_min + 4. * rand(1)[1]
        R[i][i] .= r_i .* I(nu[i])
    end
    ub_x = 5 * rand(nx)
    lb_x = -5 * rand(nx)
    C_x = randn(mx, nx)
    b_x = ones(mx)
    C_x_all = [C_x;
        Matrix{Float64}(I(nx));
        -Matrix{Float64}(I(nx))]
    b_x_all = [b_x; ub_x; -lb_x]
    C_loc_vec = [randn(mu[i], nu[i]) for i in 1:N]
    b_loc_vec = [ones(mu[i]) for i in 1:N]
    C_u_vec = [zeros(0, nu[i]) for i in 1:N]
    b_u = zeros(0)

    prob = DyNEP(
        A=A,
        Bvec=Bvec,
        Q=Q,
        R=R,
        q=q,
        C_x=C_x_all,
        b_x=b_x_all,
        C_loc_vec=C_loc_vec,
        b_loc_vec=b_loc_vec,
        C_u_vec=C_u_vec,
        b_u=b_u
    )

    # Define set of parameters
    Theta = (A=C_x', b=b_x, ub=ub_x, lb=lb_x)
    return generate_mpVI(prob, T_hor), Theta
end

function benchmark_once(nx::Int, N::Int, nu::Vector{Int}, mx::Int, mu::Vector{Int}, T_hor::Int; n_queries::Int=50)
    mpVI, Theta = make_random_prob(nx, N, nu, mx, mu, T_hor)

    println("Building explicit solution...")
    opts = ParametricDAQP.Settings()
    t0 = time_ns()
    sol, _ = ParametricDAQP.mpsolve(mpVI, Theta; opts)
    t_explicit_build = (time_ns() - t0) / 1e9
    println("Explicit solution built in $(t_explicit_build) seconds.")
    xs = [rand(nx) .- 0.5 for _ in 1:n_queries]

    println("Evaluating explicit solution for each query...")
    explicit_solutions = Vector{Any}(undef, n_queries)
    explicit_eval_times = zeros(n_queries)
    for (i, x) in enumerate(xs)
        println("  Explicit evaluation for query $(i)/$n_queries")
        t1 = time_ns()
        try
            explicit_solutions[i] = evaluate_solution(sol, x)
        catch e
            @warn "Error during explicit evaluation" exception = (e, catch_backtrace())
            explicit_solutions[i] = nothing
        end
        explicit_eval_times[i] = (time_ns() - t1) / 1e9
    end
    println("Explicit evaluation done.")

    println("Solving implicitly for each query (AVIsolve)...")
    implicit_solutions = Vector{Any}(undef, n_queries)
    implicit_eval_times = zeros(n_queries)
    for (i, x) in enumerate(xs)
        println("  Implicit solve for query $(i)/$n_queries")
        t1 = time_ns()
        try
            implicit_solutions[i], _ = ParametricDAQP.AVIsolve(
                mpVI.H,
                mpVI.F' * [x; 1.0],
                mpVI.A,
                mpVI.B' * [x; 1.0];
                warmstart=ParametricDAQP.UnconstrainedSolution,
                max_iter=10^6,
                stepsize=0.5,
                tol=1e-5
            )
        catch e
            @warn "Error during implicit solve" exception = (e, catch_backtrace())
            implicit_solutions[i] = nothing
        end
        implicit_eval_times[i] = (time_ns() - t1) / 1e9
    end
    println("Implicit solves done.")

    # Benchmark difference between explicit and implicit solutions
    diffs_exp_imp = Float64[]
    for i in 1:n_queries
        ex = explicit_solutions[i]
        im = implicit_solutions[i]
        if ex !== nothing && im !== nothing
            try
                push!(diffs_exp_imp, norm(ex - im))
            catch e
                @warn "Could not compute norm between explicit and implicit solution for query $i" exception = (e, catch_backtrace())
                push!(diffs_exp_imp, NaN)
            end
        else
            if ex === nothing && im === nothing
                @warn "Both explicit and implicit solutions are infeasible for query $i"
            elseif ex === nothing
                @warn "Explicit solution is infeasible for query $i, while the Implicit solution is feasible"
            else
                @warn "Implicit solution is infeasible for query $i, while the explicit solution is feasible"
            end
            push!(diffs_exp_imp, NaN)
        end
    end

    return (
        t_explicit_build=t_explicit_build,
        explicit_eval_times=explicit_eval_times,
        implicit_eval_times=implicit_eval_times,
        diffs_exp_imp=diffs_exp_imp,
    )
end

function run_benchmarks(; n_instances::Int, nx::Int, N::Int, nu::Vector{Int}, mx::Int, mu::Vector{Int}, T_hor::Int, n_queries::Int)
    results = Vector{NamedTuple{(
            :t_explicit_build,
            :explicit_eval_times,
            :implicit_eval_times,
            :diffs_exp_imp),
        Tuple{Float64,Vector{Float64},Vector{Float64},Vector{Float64}}}}(undef, n_instances)
    for i in 1:n_instances
        println("Benchmark instance $i / $n_instances")
        res = benchmark_once(nx, N, nu, mx, mu, T_hor; n_queries=n_queries)
        results[i] = res
    end

    build_explicit_times = vcat([r.t_explicit_build for r in results]...)
    explicit_evaluation_times = vcat([r.explicit_eval_times for r in results]...)
    implicit_solution_times = vcat([r.implicit_eval_times for r in results]...)
    diffs_exp_imp = vcat([r.diffs_exp_imp for r in results]...)
    @infiltrate
    println("\nBenchmark summary over $n_instances instances (times in seconds):")
    println("Explicit build time: mean=$(mean(build_explicit_times)), median=$(median(build_explicit_times)), std=$(std(build_explicit_times))")
    println("Explicit eval time (per query): mean=$(mean(explicit_evaluation_times)), median=$(median(explicit_evaluation_times)), std=$(std(explicit_evaluation_times))")
    println("Implicit solve time (per query): mean=$(mean(implicit_solution_times)), median=$(median(implicit_solution_times)), std=$(std(implicit_solution_times))")
    println("difference between explicit and implicit solution: mean=$(mean(diffs_exp_imp)), median=$(median(diffs_exp_imp)), std=$(std(diffs_exp_imp))")

end

# Default benchmark parameters - tweak as needed
run_benchmarks(n_instances=2, nx=2, N=3, nu=[3, 2, 3], mx=3, mu=[2, 2, 3], T_hor=2, n_queries=10)


