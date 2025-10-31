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
end

function CommonSolve.init(prob::AVI, ::Type{DGSQPSolver}; params::IterativeSolverParams=IterativeSolverParams())
    B = (prob.H + prob.H') / 2 + param.regularization * eye(prob.n)
    qp = Clarabel.Solver()
    settings = Clarabel.Settings(verbose=false, presolve_enable=false)
    cone = [Clarabel.NonnegativeConeT(size(prob.A, 1))] # Sets all constraints to inequalities
    h = zeros(n)
    Clarabel.setup!(qp, B, h, A, prob.b, cone, settings)
end

function CommonSolve.step!(solver::DGSQPSolver)
    # Compute projection operator
    qp = Clarabel.Solver()
    settings = Clarabel.Settings(verbose=false, presolve_enable=false)
    cone = [Clarabel.NonnegativeConeT(size(prob.A, 1))] # Sets all constraints to inequalities
    A = SparseMatrixCSC(prob.A)
    Clarabel.setup!(proj, M1plusQ, M2x_plus_f, A, prob.b, cone, settings)

end

function CommonSolve.solve!(solver::DGSQPSolver)
    mul!(solver.h, solver.prob.H * solver.h)
    Clarabel.update_q!(solver.qp, solver.h)
    mul!(solver.C, prob.A, solver.x) # C(x) = Ax - b
    solver.C .= solver.C .- solver.prob.b
    solver.qp.update(q=solver.h, b=-solver.C)
    results = Clarabel.solve!(solver.qp)
    solver.p_x[:] = results.x
    if results.status != Clarabel.SOLVED && results.status != Clarabel.ALMOST_SOLVED
        solver.x = nothing
        solver.status[] = :Infeasible
        return solver
    end
    d = results.z # Dual variable of the QP
    solver.p_λ .= d .- solver.λ
    solver.s .= min.(zeros(solver.prob.m), solver.C) # s = min(0, Ax - b - s)
    solver.p_s .= solver.C .- solver.s
    mul!(solver.p_s, solver.prob.A, solver.p_x, 1.0, 1.0) # p_s = Ap + Ax - b - s
    #TODO: line search, understand what μ does
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


