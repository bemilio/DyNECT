### AVI Solvers ###

@doc raw"""
Douglas Rachford algorithm for the affine variational inequality
```math
\mathrm{VI}(Hx + f, Ax <= b)
```
Algorithm:
    ```math
    y^{(k)} = \mathrm{min_\omega: A\omega \leq b} \omega^\top(Q + M_1)\omega + q + (M_2 - Q)x^{(k)} \\
    x^{(k+1)} = (Q + M_2)^{-1}(Q(2\gamma y^{(k)} + (1 - 2\gamma)x^{(k)})+M_2x^{(k)}))
    ```
where ...
"""
struct DouglasRachford
    prob::AVI
    Q::SparseMatrixCSC #Regularization matrix
    # M1::SparseMatrixCSC
    M2::SparseMatrixCSC
    # M1plusQ::SparseMatrixCSC
    M2minusQ::SparseMatrixCSC
    M2plusQ::LinearAlgebra.Factorization # LU parametrization
    M2x_plus_f::AbstractVector
    proj::Clarabel.Solver
    y::AbstractVector
    x::Union{AbstractVector,Nothing}
    status::Ref{Symbol}

    stepsize::Float64
    params::IterativeSolverParams
    last_res::Ref{Float64}

end

function CommonSolve.init(prob::AVI, ::Type{DouglasRachford}; Q=nothing, γ::Float64=0.5, stepsize::Float64=0.5, params::IterativeSolverParams=IterativeSolverParams())
    n = prob.n
    # Regularization matrix
    if isnothing(Q)
        Q = γ .* I(n) + (1 - γ) * (prob.H + prob.H') ./ 2
    end
    # Compute warmstart point
    if params.warmstart == :NoWarmStart
        x = zeros(n)
    elseif params.warmstart == :UnconstrainedSolution
        x = -prob.H \ prob.f
    else
        @warn "[DouglasRachford initialization] Requested warmstart not implemented."
        x = zeros(n)
    end

    # check for trivial infeasibility: A[i,:] = 0, b[i]<0 for some i
    zero_rows = findall(row -> norm(row) < params.tol, eachrow(prob.A))
    for idx_row in zero_rows
        if prob.b[idx_row] < 0
            return DouglasRachford(prob, Q, M2, M2minusQ, M2plusQ, M2x_plus_f, proj, y, x, Ref(:Infeasible), stepsize, params, Ref(Inf))
        end
    end

    # Precompute auxiliary matrices
    H_sym = (prob.H + prob.H') ./ 2
    H_antisym = (prob.H - prob.H') ./ 2
    M1 = γ * H_sym
    M2 = (1 - γ) * H_sym + H_antisym
    M1plusQ = SparseMatrixCSC(triu(M1 + Q)) # Clarabel convention: upper-triangular matrix
    M2minusQ = SparseMatrixCSC(M2 - Q)
    M2plusQ = lu(M2 + Q)
    M2 = SparseMatrixCSC(M2)
    M2x_plus_f = zeros(n) # pre-allocation, updated in iterations
    y = zeros(n) # pre-allocation, updated in iterations

    # Compute projection operator
    proj = Clarabel.Solver()
    settings = Clarabel.Settings(verbose=false, presolve_enable=false)
    cone = [Clarabel.NonnegativeConeT(size(prob.A, 1))] # Sets all constraints to inequalities
    A = SparseMatrixCSC(prob.A)
    Clarabel.setup!(proj, M1plusQ, M2x_plus_f, A, prob.b, cone, settings)
    return DouglasRachford(prob, Q, M2, M2minusQ, M2plusQ, M2x_plus_f, proj, y, x, Ref(:Initialized), stepsize, params, Ref(Inf))
end

function CommonSolve.step!(DR::DouglasRachford)
    DR.M2x_plus_f[:] = DR.prob.f
    mul!(DR.M2x_plus_f, DR.M2minusQ, DR.x, 1.0, 1.0)
    Clarabel.update_q!(DR.proj, DR.M2x_plus_f)
    results = Clarabel.solve!(DR.proj)
    if results.status in (Clarabel.PRIMAL_INFEASIBLE,
        Clarabel.DUAL_INFEASIBLE)
        DR.x[:] = NaN .* ones(length(DR.x))
        DR.status[] = :Infeasible
        return DR
    end

    tmp = (2 * DR.stepsize) .* results.x + (1 - 2 * DR.stepsize) .* DR.x # Can this be optimized?
    mul!(DR.y, DR.Q, tmp)
    mul!(DR.y, DR.M2, DR.x, 1.0, 1.0)
    ldiv!(DR.x, DR.M2plusQ, DR.y)
end

function CommonSolve.solve!(DR::DouglasRachford)
    res = Inf
    n_iter = 0
    t0 = time()
    for k in 1:DR.params.max_iter
        CommonSolve.step!(DR)
        if DR.status[] == :Infeasible
            @warn "[DouglasRachford::solve] Infeasibility detected"
            break
        end
        res = compute_residual(DR.prob, DR.x)
        if res < DR.params.tol
            DR.status[] = :Solved
            n_iter = k
            break
        end
        if k % 1000 == 0
            if res > DR.last_res[] - DR.params.tol / 10
                DR.status[] = :NoImprovement
                @warn "[DouglasRachford::solve] Solution not improving, residual = $res"
                break
            end
            DR.last_res[] = res
            if DR.params.verbose
                println("Iteration $k, Residual = $res")
            end
        end
        if k == DR.params.max_iter
            DR.status[] = :MaximumIterationsReached
            @warn "[DouglasRachford::solve] Maximum iterations reached, residual = $res"
            n_iter = k
            break
        end
        if time() - t0 > DR.params.time_limit
            DR.status[] = :TimeLimitReached
            println("[DouglasRachford::solve] Time limit reached, residual = $res ")
            n_iter = k
            break
        end
    end
    solution = (x=DR.x, status=DR.status[], residual=res, iterations=n_iter)
    return solution
end

# Log-domain interior point method [Liu, Liao-McPherson 2025]
struct LogIPMSolver
    prob::AVI
    # Q::SparseMatrixCSC #Regularization matrix

    x::Union{AbstractVector,Nothing}
    status::Ref{Symbol}

    stepsize::Float64
    params::IterativeSolverParams

end

function CommonSolve.init(prob::AVI, ::Type{LogIPMSolver}; stepsize::Float64=0.5, params::IterativeSolverParams=IterativeSolverParams())


    return LogIPMSolver(prob, x, Ref(:Initialized), stepsize, params)
end

function CommonSolve.step!(solver::LogIPMSolver)

end

function CommonSolve.solve!(solver::LogIPMSolver)
    res = Inf
    n_iter = 0
    for k in 1:solver.params.max_iter
        CommonSolve.step!(solver)
        if solver.status[] == :Infeasible
            @warn "[LogIPMSolver::solve] Infeasibility detected"
            break
        end
        res = compute_residual(solver.prob, solver.x)
        if res < solver.params.tol
            solver.status[] = :Solved
            n_iter = k
            break
        end
        ((k % 1000 == 0) && solver.params.verbose) && println("Iteration $k, Residual = $res")
        if k == solver.params.max_iter
            solver.status[] = :MaximumIterationsReached
            @warn "[LogIPMSolver::solve] Maximum iterations reached, residual = $res"
            n_iter = k
            break
        end
    end
    solution = (x=solver.x, status=solver.status[], residual=res, iterations=n_iter)
    return solution
end


# Compatibility layer with Monviso

struct MonvisoSolver
    prob::AVI
    monviso_vi::Monviso.VI
    method::Symbol
    x::AbstractVector
    status::Ref{Symbol}
    params::IterativeSolverParams
end

function CommonSolve.init(prob::AVI, ::Type{MonvisoSolver}; method::Symbol=:pg, params::IterativeSolverParams=IterativeSolverParams())

    if isnothing(params.stepsize)
        method == :pg && (params.stepsize = 0.1)
        #TODO: add the other methods
    end

    # Compute warmstart point
    if params.warmstart == :NoWarmStart
        x = zeros(prob.n)
    elseif params.warmstart == :UnconstrainedSolution
        x = -prob.H \ prob.f
    else
        @warn "[DouglasRachford initialization] Requested warmstart not implemented."
        x = zeros(prob.n)
    end

    F(x) = prob.H * x + prob.f
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    y = @variable(model, [1:prob.n])
    @constraint(model, prob.A * y .<= prob.b)
    monviso_vi = Monviso.VI(F, y=y, model=model)
    x = zeros(prob.n)
    return MonvisoSolver(prob, monviso_vi, method, x, Ref(:Initialized), params)
end

function CommonSolve.step!(solver::MonvisoSolver)
    if solver.method == :pg
        solver.x .= Monviso.pg(solver.monviso_vi, solver.x, solver.params.stepsize)
        #TODO: add the other methods
        #TODO: check infeasibility
    end
end

function CommonSolve.solve!(solver::MonvisoSolver)
    res = Inf
    t0 = time()
    for k in 1:solver.params.max_iter
        CommonSolve.step!(solver)
        if solver.status == :Infeasible
            @warn "[MonvisoSolver] Infeasibility detected"
            break
        end
        res = compute_residual(solver.prob, solver.x)
        if res < solver.params.tol
            solver.status[] = :Solved
            break
        end
        ((k % 1000 == 0) && solver.params.verbose) && println("Iteration $k, Residual = $res")
        if k == solver.params.max_iter
            solver.status[] = :MaximumIterationsReached
            @warn "[MonvisoSolver] Maximum iterations reached, residual = $res"
            break
        end
        if time() - t0 > solver.params.time_limit
            solver.status[] = :TimeLimitReached
            println("[MonvisoSolver::solve] Time limit reached, residual = $res ")
            break
        end
    end
    solution = (x=solver.x, status=solver.status[], residual=res)
    return solution
end

# DGSQP
struct DGSQPSolver
    prob::AVI
    x::AbstractVector{Float64}
    λ::AbstractVector{Float64}
    x_last::AbstractVector{Float64}
    λ_last::AbstractVector{Float64}
    iter::Ref{Int}
    iter_last::Ref{Int}
    merit_last::Ref{Float64}
    merit::Function
    use_relaxed_step::Ref{Bool}
    qp::Clarabel.Solver
    params::IterativeSolverParams
    max_watchdog_iter::Int
    max_safe_linesearch_iter::Int
    status::Ref{Symbol}
    last_res::Ref{Float64}
end

function CommonSolve.init(prob::AVI, ::Type{DGSQPSolver}; max_watchdog_iter::Int=5, max_safe_linesearch_iter::Int=20, params::IterativeSolverParams=IterativeSolverParams())
    regularization = 0.1
    B = SparseMatrixCSC((prob.H + prob.H') / 2 + regularization * I(prob.n))
    qp = Clarabel.Solver()
    settings = Clarabel.Settings(verbose=false, presolve_enable=false)

    cone = [Clarabel.NonnegativeConeT(size(prob.A, 1))] # Sets all constraints to inequalities
    A = SparseMatrixCSC(prob.A)

    if params.warmstart == :NoWarmStart
        x = zeros(prob.n)
    elseif params.warmstart == :UnconstrainedSolution
        x = -prob.H \ prob.f
    else
        @warn "[DGSQPSolver initialization] Requested warmstart not implemented."
        x = zeros(prob.n)
    end

    Clarabel.setup!(qp, B, zeros(prob.n), A, prob.b, cone, settings)
    merit = (x, λ, s, μ) -> 0.5 * norm(prob.H * x + prob.f + prob.A' * λ, 2)^2 + μ * norm(prob.A * x - prob.b - s, 1)
    return DGSQPSolver(prob, x, zeros(prob.m), x, zeros(prob.m), Ref(0), Ref(0), Ref(Inf), merit, Ref(true), qp, params, max_watchdog_iter, max_safe_linesearch_iter, Ref(:Initialized), Ref(Inf))
end

function CommonSolve.step!(solver::DGSQPSolver)
    #Primal update: QP solution
    h = solver.prob.H * solver.x + solver.prob.f
    Clarabel.update_q!(solver.qp, h)
    C = solver.prob.A * solver.x - solver.prob.b # C(x) = Ax - b
    Clarabel.update_b!(solver.qp, -C)
    results = Clarabel.solve!(solver.qp)
    p_x = results.x
    if results.status in (Clarabel.PRIMAL_INFEASIBLE, Clarabel.DUAL_INFEASIBLE)
        solver.x .= NaN .* ones(solver.prob.n)
        solver.status[] = :Infeasible
        return solver
    end
    # Dual update
    d = results.z # Dual variable of the QP
    p_λ = d .- solver.λ
    # Aux variable update
    s = min.(zeros(solver.prob.m), C) # s = min(0, Ax - b - s)
    p_s = solver.prob.A * p_x + C - s

    # Watchdog line search
    eps = 0.0001
    if norm(C - s) > eps
        ∇γ = [solver.prob.H' * (solver.prob.H * solver.x + solver.prob.f + solver.prob.A' * solver.λ);
            solver.prob.A * (solver.prob.H * solver.x + solver.prob.f + solver.prob.A' * solver.λ)]
        ρ = 0.5
        μ = ∇γ' * [p_x; p_λ] / ((1 - ρ) * norm(C - s, 1))
    else
        μ = 0.
    end
    x_plus = zeros(solver.prob.n)
    λ_plus = zeros(solver.prob.m)
    merit_x_plus = 0.
    if solver.use_relaxed_step[]
        x_plus .= solver.x + p_x
        λ_plus .= solver.λ + p_λ
        s_plus = s + p_s
        merit_x_plus = solver.merit(x_plus, λ_plus, s_plus, μ)
    else
        #safe line search
        alpha = 1.0
        for k in 1:solver.max_safe_linesearch_iter
            x_plus .= solver.x + alpha * p_x
            λ_plus .= solver.λ + alpha * p_λ
            s_plus = s + alpha * p_s
            merit_x_plus = solver.merit(x_plus, λ_plus, s_plus, μ)
            merit_x = solver.merit(solver.x, solver.λ, s, μ)
            if merit_x_plus <= merit_x
                break
            else
                alpha .* 0.5
            end
        end
    end
    if merit_x_plus <= solver.merit_last[]
        solver.x .= x_plus
        solver.iter_last[] = solver.iter[]
        solver.merit_last[] = merit_x_plus[]
        solver.use_relaxed_step[] = true
    else
        solver.use_relaxed_step[] = false
        if solver.iter[] - solver.iter_last[] >= solver.max_watchdog_iter
            # reset to point of last decrease
            solver.x .= solver.x_last
            solver.λ .= solver.λ_last
            solver.iter_last[] = solver.iter[]
        end
    end
    solver.iter[] = solver.iter[] + 1
end

function CommonSolve.solve!(solver::DGSQPSolver)
    res = Inf
    t0 = time()
    for k in 1:solver.params.max_iter
        CommonSolve.step!(solver)
        if solver.status[] == :Infeasible
            @warn "[DGSQPSolver::solve] Infeasibility detected"
            break
        end
        res = compute_residual(solver.prob, solver.x)
        if k % 1000 == 0
            if res > solver.last_res[] - solver.params.tol / 10
                solver.status[] = :NoImprovement
                @warn "[DGSQPSolver::solve] Solution not improving, residual = $res"
                break
            end
            solver.last_res[] = res
            if solver.params.verbose
                println("Iteration $k, Residual = $res")
            end
        end
        if res < solver.params.tol
            solver.status[] = :Solved
            break
        end
        ((k % 1000 == 0) && solver.params.verbose) && println("[DGSQPSolver] Iteration $k, Residual = $res")
        if k == solver.params.max_iter
            solver.status[] = :MaximumIterationsReached
            @warn "[DGSQPSolver::solve] Maximum iterations reached, residual = $res"
            break
        end
        if time() - t0 > solver.params.time_limit
            solver.status[] = :TimeLimitReached
            println("[DGSQPSolver::solve] Time limit reached, residual = $res ")
            break
        end
    end
    solution = (x=solver.x, status=solver.status[], residual=res)
    return solution
end


#### DynLQGame solvers ####

# ADMM-CLQG
struct ADMMCLQGSolver
    # Stored quantities
    prob::DynLQGame
    vi::AVI
    mpvi::mpAVI
    mpvi_reg::mpAVI
    predmod::NamedTuple
    x0::Vector{Float64}
    luH::LinearAlgebra.Factorization # LU parametrization
    f::Vector{Float64}
    proj_X::Clarabel.Solver
    proj_U::Clarabel.Solver

    # Primal vars
    x::Vector{Float64}
    u::Vector{Float64}
    # Dual vars
    λ::Vector{Float64}
    μ::Vector{Float64}
    # Aux vars
    z::Vector{Float64}
    ω::Vector{Float64}

    # Parameters
    ρ::Float64 # Regularization factor

    params::IterativeSolverParams
    status::Ref{Symbol}
    last_res::Ref{Float64}
end

function update_x0!(solver::ADMMCLQGSolver, x0::Vector{Float64})
    @assert length(x0) == solver.prob.nx
    vi = AVI(solver.mpvi, x0)
    solver.vi.f[:] = vi.f[:]
    solver.vi.b[:] = vi.b[:]
    f[:] = AVI(solver.mpvi_reg, x0).f
end

function CommonSolve.init(prob::DynLQGame, ::Type{ADMMCLQGSolver};
    T_hor::Union{Int,Nothing}=nothing,
    x0::Union{AbstractVector,Nothing}=nothing,
    ρ::Float64=0.5,
    params::IterativeSolverParams=IterativeSolverParams())

    # Compute regularized VI
    game_reg = deepcopy(prob)
    # add regularizer
    map!(Qi -> Qi + ρ .* I(game_reg.nx), game_reg.Q, game_reg.Q)
    map!(Pi -> Pi + ρ .* I(game_reg.nx), game_reg.P, game_reg.P)
    for i = 1:prob.N
        game_reg.R[i][i] = game_reg.R[i][i] + ρ .* I(prob.nu[i])
    end
    if isnothing(T_hor)
        @error "[ADMMCLQGSolver] T_hor cannot be unspecified"
    else
        mpvi_reg = DynLQGame2mpAVI(game_reg, T_hor)
        mpvi = DynLQGame2mpAVI(prob, T_hor) # Used to compute residual
    end
    if isnothing(x0)
        @error "[ADMMCLQGSolver] x_0 cannot be unspecified"
    else
        vi = AVI(mpvi, x0)
        vi_reg = AVI(mpvi_reg, x0)
        luH = lu(vi_reg.H)
        f = vi_reg.f
    end

    # Initialize projection on state space
    proj_X = Clarabel.Solver()
    settings = Clarabel.Settings(verbose=false, presolve_enable=false)
    Cx = SparseMatrixCSC(kron(I(T_hor), prob.C_x))
    dx = kron(ones(T_hor), prob.b_x)
    cone = [Clarabel.NonnegativeConeT(size(Cx, 1))] # Sets all constraints to inequalities
    Q = spdiagm(0 => ones(Float64, prob.nx * T_hor)) # Sparse identity
    q = zeros(prob.nx * T_hor)
    Clarabel.setup!(proj_X, Q, q, Cx, dx, cone, settings)

    # Initialize projection on input space
    proj_U = Clarabel.Solver()
    settings = Clarabel.Settings(verbose=false, presolve_enable=false)
    Cu_loc = BlockDiagonal(map(Cu_i -> kron(I(T_hor), Cu_i), prob.C_loc_i))
    du_loc = vcat(map(bu_i -> kron(ones(T_hor), bu_i), prob.b_loc_i)...)

    Cu_sh = hcat(map(Cu_i -> kron(I(T_hor), Cu_i), prob.C_u_i)...)
    du_sh = kron(ones(T_hor), prob.b_u)
    Cu = SparseMatrixCSC([Cu_loc; Cu_sh])
    du = [du_loc; du_sh]
    cone = [Clarabel.NonnegativeConeT(size(Cu, 1))] # Sets all constraints to inequalities
    Q = spdiagm(0 => ones(Float64, sum(prob.nu) * T_hor))
    q = zeros(sum(prob.nu) * T_hor)
    Clarabel.setup!(proj_U, Q, q, Cu, du, cone, settings)

    # Store prediction model for the linear system
    Γ, _, Θ, k = generate_prediction_model(prob.A, prob.Bi, T_hor)
    predmod = (Γ=SparseMatrixCSC(Γ), Θ=SparseMatrixCSC(Θ), k=k)

    # check for trivial infeasibility: A[i,:] = 0, b[i]<0 for some i
    # zero_rows = findall(row -> norm(row) < params.tol, eachrow(prob.A))
    # for idx_row in zero_rows
    #     if prob.b[idx_row] < 0
    #         #TODO: fix
    #         return ADMMCLQGSolver(prob, Q, M2, M2minusQ, M2plusQ, M2x_plus_f, proj, y, x, Ref(:Infeasible), params)
    #     end
    # end

    return ADMMCLQGSolver(prob, vi, mpvi, mpvi_reg, predmod, x0, luH, f, proj_X, proj_U, # Stored quantities
        zeros(prob.nx * T_hor), zeros(sum(prob.nu) * T_hor), # Primal vars init
        zeros(prob.nx * T_hor), zeros(sum(prob.nu) * T_hor), # Dual vars init
        zeros(prob.nx * T_hor), zeros(sum(prob.nu) * T_hor), # Aux vars init
        ρ, params, Ref(:Initialized), Ref(Inf))
end

function CommonSolve.step!(solver::ADMMCLQGSolver)
    # Update affine part of the regularized game
    f = solver.f + solver.predmod.Γ' * (solver.λ - solver.ρ * solver.z) + (solver.μ - solver.ρ * solver.ω)

    # Solve unconstrained game
    solver.u[:] = solver.luH \ (-f)
    mul!(solver.x, solver.predmod.Γ, solver.u)
    mul!(solver.x, solver.predmod.Θ, solver.x0, 1.0, 1.0)
    solver.x .+= solver.predmod.k

    # Slack variable update
    Clarabel.update_q!(solver.proj_X, -(solver.x + solver.λ ./ solver.ρ))
    results = Clarabel.solve!(solver.proj_X)
    solver.z[:] = results.x

    if results.status in (Clarabel.PRIMAL_INFEASIBLE, Clarabel.DUAL_INFEASIBLE)
        solver.u[:] = fill(NaN, length(solver.u))
        solver.status[] = :Infeasible
        return solver
    end

    Clarabel.update_q!(solver.proj_U, -(solver.u + solver.μ ./ solver.ρ))
    results = Clarabel.solve!(solver.proj_U)
    solver.ω[:] = results.x

    if results.status in (Clarabel.PRIMAL_INFEASIBLE, Clarabel.DUAL_INFEASIBLE)
        solver.x = NaN .* ones(solver.prob.n)
        solver.status[] = :Infeasible
        return solver
    end

    # Dual update 
    solver.λ .+= solver.ρ .* (solver.x - solver.z)
    solver.μ .+= solver.ρ .* (solver.u - solver.ω)

    return solver
end


function CommonSolve.solve!(solver::ADMMCLQGSolver)
    res = Inf
    t0 = time()
    for k in 1:solver.params.max_iter
        CommonSolve.step!(solver)
        if solver.status[] == :Infeasible
            @warn "[ADMMCLQGSolver] Infeasibility detected"
            break
        end
        res = compute_residual(solver.vi, solver.u)
        if k % 1000 == 0
            if res > solver.last_res[] - solver.params.tol / 10
                solver.status[] = :NoImprovement
                @warn "[ADMMCLQGSolver::solve] Solution not improving, residual = $res"
                break
            end
            solver.last_res[] = res
            if solver.params.verbose
                println("Iteration $k, Residual = $res")
            end
        end
        if res < solver.params.tol
            solver.status[] = :Solved
            break
        end
        if k == solver.params.max_iter
            solver.status[] = :MaximumIterationsReached
            @warn "[ADMMCLQGSolver] Maximum iterations reached, residual = $res"
            break
        end
        if time() - t0 > solver.params.time_limit
            solver.status[] = :TimeLimitReached
            println("[ADMMCLQGSolver::solve] Time limit reached, residual = $res ")
            break
        end
    end
    solution = (u=solver.u, status=solver.status[], residual=res)
    return solution
end

##### mpAVI solvers #####

# Compatibility layer with ParametricDAQP

struct ParametricDAQPSolver
    mpAVI::mpAVI
    options::ParametricDAQP.Settings
    status::Ref{Symbol}
    AS0::Vector{Int64}
end

function CommonSolve.init(prob::mpAVI, ::Type{ParametricDAQPSolver};
    eps_zero::Float64=1e-12,
    verbose::Bool=true,
    store_AS::Bool=true,
    store_points::Bool=true,
    store_regions::Bool=true,
    store_dual::Bool=false,
    remove_redundant::Bool=true,
    time_limit::Int64=10000,
    chunk_size::Int64=1000,
    factorization::Symbol=:chol,
    postcheck_rank::Bool=true,
    lowdim_tol::Float64=0.,
    early_stop::Bool=false)

    options = ParametricDAQP.Settings(eps_zero,
        verbose,
        store_AS,
        store_points,
        store_regions,
        store_dual,
        remove_redundant,
        time_limit,
        chunk_size,
        factorization,
        postcheck_rank,
        lowdim_tol,
        early_stop)


    # Solve feasibility problem in (x,θ)-space to find initial active set:
    # Find (x,θ) s.t. Cθ ≤ d ; lb ≤ θ ≤ ub ; Ax ≤ Bθ + b
    qp = Clarabel.Solver()
    settings = Clarabel.Settings(verbose=false)
    A_θ = [prob.C; Matrix{Float64}(I(prob.n_θ)); -Matrix{Float64}(I(prob.n_θ))]
    b_θ = [prob.d; prob.ub; -prob.lb]
    A_θ_x = SparseMatrixCSC([-prob.B prob.A; A_θ zeros(size(A_θ, 1), prob.n)])
    b_θ_x = [prob.b; b_θ]
    cone = [Clarabel.NonnegativeConeT(size(A_θ_x, 1))] # Sets all constraints to inequalities
    H_θ_x = SparseMatrixCSC(Matrix{Float64}(I(prob.n_θ + prob.n)))
    f_θ_x = zeros(prob.n_θ + prob.n)
    Clarabel.setup!(qp, H_θ_x, f_θ_x, A_θ_x, b_θ_x, cone, settings)
    results = Clarabel.solve!(qp)

    if results.status == :Infeasible
        @warn("[ParametricDAQPSolver::init] Could not find a θ for which the mpAVI is feasible")
        return ParametricDAQPSolver(prob, options, Ref(:Infeasible), Vector{Int64}())
    end

    θ = results.x[1:prob.n_θ]

    # Solve AVI for the given θ
    tol = 1e-4
    avi = AVI(prob, θ)
    avi_sol = CommonSolve.solve(avi, DGSQPSolver; params=IterativeSolverParams(warmstart=:UnconstrainedSolution, tol=tol))
    x = avi_sol.x
    if avi_sol.status == :Infeasible
        @warn("[ParametricDAQPSolver::init] Could not solve AVI to find the initial active set.")
        return ParametricDAQPSolver(prob, options, Ref(:Infeasible), Vector{Int64}())
    end
    if avi_sol.status == :MaximumIterationsReached
        @warn("[ParametricDAQPSolver::init] Maximum iterations reached while solving AVI for initial active set. Residual = $(avi_sol.residual)")
    end

    # Retrieve initial active set
    AS0 = findall(prob.A * x .>= prob.B * θ + prob.b .- tol)
    return ParametricDAQPSolver(prob, options, Ref(:Initialized), AS0)

end

function CommonSolve.solve!(pDAQP::ParametricDAQPSolver)
    # Convert problem into ParametricDAQP format
    prob = ParametricDAQP.MPVI(pDAQP.mpAVI.H, pDAQP.mpAVI.F, pDAQP.mpAVI.f, pDAQP.mpAVI.A, pDAQP.mpAVI.B, pDAQP.mpAVI.b)
    n_θ = size(pDAQP.mpAVI.C, 2)
    Θ = (A=pDAQP.mpAVI.C', b=pDAQP.mpAVI.d, ub=pDAQP.mpAVI.ub, lb=pDAQP.mpAVI.lb)
    (sol, info) = ParametricDAQP.mpsolve(prob, Θ; opts=pDAQP.options, AS0=pDAQP.AS0)
    return sol
end



