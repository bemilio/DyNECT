### AVI Solvers ###


function AVIsolver_common_init()
end

function compute_residual(prob::AVI, x::AbstractVector)
    y = x - (prob.H * x + prob.f)
    #TODO: Switch to Clarabel
    proj = DAQP.Model()
    DAQP.setup(proj, Matrix{Float64}(I, prob.n, prob.n), -y, prob.A, prob.b, Float64[], zeros(Cint, prob.m))
    x_transf, _, _, _ = DAQP.solve(proj)
    r = norm(x - x_transf)
    return r
end

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
    params::IterativeSolverParams

end

function CommonSolve.init(prob::AVI, ::Type{DouglasRachford}; Q=nothing, params::IterativeSolverParams=IterativeSolverParams())
    n = size(prob.H, 1)
    # Regularization matrix
    if Q === nothing
        Q = I(n)
    end

    if isnothing(params.stepsize)
        params.stepsize = 0.5
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
            return DouglasRachford(prob, Q, M2, M2minusQ, M2plusQ, M2x_plus_f, proj, y, x, Ref(:Infeasible), params)
        end
    end

    # Precompute auxiliary matrices
    M1 = (prob.H + prob.H') / 4
    M2 = SparseMatrixCSC(M1 + (prob.H - prob.H') / 2)
    M1plusQ = SparseMatrixCSC(triu(M1 + Q)) # Clarabel convention: upper-triangular matrix
    M2minusQ = SparseMatrixCSC(M2 - Q)
    M2plusQ = lu(M2 + Q)
    M2x_plus_f = zeros(n) # pre-allocation, updated in iterations
    y = zeros(n) # pre-allocation, updated in iterations

    # Compute projection operator
    proj = Clarabel.Solver()
    settings = Clarabel.Settings(verbose=false, presolve_enable=false)
    cone = [Clarabel.NonnegativeConeT(size(prob.A, 1))] # Sets all constraints to inequalities
    A = SparseMatrixCSC(prob.A)
    Clarabel.setup!(proj, M1plusQ, M2x_plus_f, A, prob.b, cone, settings)

    return DouglasRachford(prob, Q, M2, M2minusQ, M2plusQ, M2x_plus_f, proj, y, x, Ref(:Initialized), params)
end

function CommonSolve.step!(DR::DouglasRachford)
    DR.M2x_plus_f[:] = DR.prob.f
    mul!(DR.M2x_plus_f, DR.M2minusQ, DR.x, 1.0, 1.0)
    Clarabel.update_q!(DR.proj, DR.M2x_plus_f)
    results = Clarabel.solve!(DR.proj)
    DR.y[:] = results.x
    if results.status != Clarabel.SOLVED && results.status != Clarabel.ALMOST_SOLVED
        DR.x = nothing
        DR.status[] = :Infeasible
        return DR
    end
    DR.y[:] = 2 * DR.params.stepsize * DR.y + (1 - 2 * DR.params.stepsize) * DR.x
    mul!(DR.y, DR.M2, DR.x, 1.0, 1.0)
    ldiv!(DR.x, DR.M2plusQ, DR.y)
end

function CommonSolve.solve!(DR::DouglasRachford)
    res = Inf
    for k in 1:DR.params.max_iter
        CommonSolve.step!(DR)
        if DR.status[] == :Infeasible
            @warn "[DouglasRachford::solve] Infeasibility detected"
            break
        end
        if mod(k, 10) == 0
            res = compute_residual(DR.prob, DR.x)
            if res < DR.params.tol
                DR.status[] = :Solved
                break
            end
        end
        if k == DR.params.max_iter
            DR.status[] = :MaximumIterationsReached
            @warn "[DouglasRachford::solve] Maximum iterations reached, residual = $res"
            break
        end
    end
    solution = (x=DR.x, status=DR.status[], residual=res)
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
    for k in 1:solver.params.max_iter
        CommonSolve.step!(solver)
        if solver.status == :Infeasible
            @warn "[MonvisoSolver] Infeasibility detected"
            break
        end
        if mod(k, 10) == 0
            res = compute_residual(solver.prob, solver.x)
            if res < solver.params.tol
                solver.status[] = :Solved
                break
            end
        end
        if k == solver.params.max_iter
            solver.status[] = :MaximumIterationsReached
            @warn "[MonvisoSolver] Maximum iterations reached, residual = $res"
            break
        end
    end
    solution = (x=solver.x, status=solver.status[], residual=res)
    return solution
end

#### DynLQGame solvers ####

# Compatibility layer with DGSQP
#TODO
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
    return DGSQPSolver(prob, x, zeros(prob.m), x, zeros(prob.m), Ref(0), Ref(0), Ref(Inf), merit, Ref(true), qp, params, max_watchdog_iter, max_safe_linesearch_iter, Ref(:Initialized))
end

function CommonSolve.step!(solver::DGSQPSolver)
    #Primal update: QP solution
    h = solver.prob.H * solver.x + solver.prob.f
    Clarabel.update_q!(solver.qp, h)
    C = solver.prob.A * solver.x - solver.prob.b # C(x) = Ax - b
    Clarabel.update_b!(solver.qp, -C)
    results = Clarabel.solve!(solver.qp)
    p_x = results.x
    if results.status != Clarabel.SOLVED && results.status != Clarabel.ALMOST_SOLVED
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
        ∇γ = [solver.prob.H' * (solver.prob.H * x + solver.prob.f + solver.prob.A'λ);
            solver.prob.A * (solver.prob.H * x + solver.prob.f + solver.prob.A'λ)]
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
    for k in 1:solver.params.max_iter
        CommonSolve.step!(solver)
        if solver.status[] == :Infeasible
            @warn "[DGSQPSolver::solve] Infeasibility detected"
            break
        end
        if mod(k, 10) == 0
            res = compute_residual(solver.prob, solver.x)
            if solver.params.verbose
                println("[DGSQPSolver] Residual = $res")
            end
            if res < solver.params.tol
                solver.status[] = :Solved
                break
            end
        end
        if k == solver.params.max_iter
            solver.status[] = :MaximumIterationsReached
            @warn "[DGSQPSolver::solve] Maximum iterations reached, residual = $res"
            break
        end
    end
    solution = (x=solver.x, status=solver.status[], residual=res)
    return solution
end


##### mpAVI solvers #####

# Compatibility layer with ParametricDAQP

struct ParametricDAQPSolver
    mpAVI::mpAVI
    options::ParametricDAQP.Settings
end

function CommonSolve.init(prob::mpAVI, ::Type{ParametricDAQPSolver};
    eps_zero::Float64=1e-12,
    verbose::Int64=1,
    store_AS::Bool=true,
    store_points::Bool=true,
    store_regions::Bool=true,
    store_dual::Bool=false,
    remove_redundant::Bool=true,
    time_limit::Int64=100000,
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

    return ParametricDAQPSolver(prob, options)

end

function CommonSolve.solve!(pDAQP::ParametricDAQPSolver)
    # Convert problem into ParametricDAQP format
    prob = ParametricDAQP.MPVI(pDAQP.mpAVI.H, pDAQP.mpAVI.F, pDAQP.mpAVI.f, pDAQP.mpAVI.A, pDAQP.mpAVI.B, pDAQP.mpAVI.b)
    n_θ = size(pDAQP.mpAVI.C, 2)
    Θ = (A=pDAQP.mpAVI.C', b=pDAQP.mpAVI.d, ub=pDAQP.mpAVI.ub, lb=pDAQP.mpAVI.lb)

    ParametricDAQP.mpsolve(prob, Θ; opts=pDAQP.options)
end


##### UnconstrainedDynLQGame solvers #####
#TODO: Dynamic programming solver, see " ADMM-iCLQG: A Fast Solver of Constrained Dynamic Games for  Planning Multi-Vehicle Feedback Trajectory"
# struct HJBSolver

# end

# function CommonSolve.step!(solver::HJBSolver)

# end


