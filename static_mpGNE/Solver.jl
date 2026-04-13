# Solver.jl — MPVIAssembly → DyNECT.mpAVI → pDAQP
#using .Static_mpGNE
using DyNECT
using CommonSolve
using LinearAlgebra

# ── lb/ub extraction: C/e encodes θ ≥ 0 (nonnegativity) and θ ≤ b (conservation)
# modified for larger dimensions c
function _extract_theta_bounds(mpvi) #testing 
    n_theta = size(mpvi.C, 2)
    lb = fill(-Inf, n_theta)   # ← was zeros, should start at -Inf
    ub = fill(Inf, n_theta)

    for i in 1:size(mpvi.C, 1)
        row = mpvi.C[i, :]
        nz  = findall(!=(0.0), row)
        length(nz) == 1 || continue
        col = nz[1]
        if row[col] < 0
            lb[col] = max(lb[col], -mpvi.e[i])
        else
            ub[col] = min(ub[col], mpvi.e[i])
        end
    end

    # handle conservation rows
    for i in 1:size(mpvi.C, 1)
        row = mpvi.C[i, :]
        nz  = findall(!=(0.0), row)
        length(nz) == 1 && continue
        for col in nz
            row[col] > 0 && (ub[col] = min(ub[col], mpvi.e[i]))
        end
    end

    return lb, ub
end

function to_dynect_mpAVI(mpvi)
    mat       = materialize(mpvi)
    lb, ub    = _extract_theta_bounds(mpvi)
    return DyNECT.mpAVI(mat.H, mat.Ftheta, mat.f, mat.A, mat.B, mat.d;
                        C=mpvi.C, d=mpvi.e, lb=lb, ub=ub)
end

function solve_gne(mpvi)
    return CommonSolve.solve(to_dynect_mpAVI(mpvi), DyNECT.ParametricDAQPSolver)
end

function evaluate_gne(sol, beta::Vector{Float64})
    return DyNECT.evaluatePWA(sol, beta)
end

n_critical_regions(sol) = length(sol.CRs)

function show_solution(sol, mpvi)
    lb, ub = _extract_theta_bounds(mpvi)
    mid    = (lb .+ ub) ./ 2
    betas  = [mid]
    for i in 1:length(mid)
        b = copy(mid); b[i] = lb[i] + 0.01; push!(betas, b)
        b = copy(mid); b[i] = ub[i] - 0.01; push!(betas, b)
    end
    println("=== GNE Solution — $(n_critical_regions(sol)) critical regions ===")
    println()
    for beta in betas
        x = evaluate_gne(sol, beta)
        if x === nothing
            println("  x*(β=$(round.(beta, digits=2))) = infeasible")
        else
            println("  x*(β=$(round.(beta, digits=2))) = $(round.(x, digits=4))")
        end
    end
    println()
end