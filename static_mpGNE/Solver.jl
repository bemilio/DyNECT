# Solver.jl — MPVIAssembly → DyNECT.mpAVI → pDAQP
using DyNECT
using CommonSolve
using LinearAlgebra

# lb/ub user-defined exploration bounds; C/e from game simplex structure
function to_dynect_mpAVI(mpvi, lb::Vector{Float64}, ub::Vector{Float64})
    mat = materialize(mpvi)
    return DyNECT.mpAVI(mat.H, mat.Ftheta, mat.f, mat.A, mat.B, mat.d;
                        C=mpvi.C, d=mpvi.e, lb=lb, ub=ub)
end

# no lb/ub — parameter space set externally via DyNECT.setParameterSpace
function to_dynect_mpAVI(mpvi)
    mat = materialize(mpvi)
    return DyNECT.mpAVI(mat.H, mat.Ftheta, mat.f, mat.A, mat.B, mat.d)
end

function solve_gne(mpvi, lb::Vector{Float64}, ub::Vector{Float64})
    sol = CommonSolve.solve(to_dynect_mpAVI(mpvi, lb, ub),
                            DyNECT.ParametricDAQPSolver)
    return (sol=sol, lb=lb, ub=ub)
end

evaluate_gne(sol, beta::Vector{Float64}) = DyNECT.evaluatePWA(sol, beta)
n_critical_regions(sol) = length(sol.CRs)

# Theorem 3.6 (Nabetani): for each shared group, each constraint dimension i
# must have all k player rows active or none — otherwise CR is spurious.
function filter_gne_crs!(sol, mpvi)
    for (group_idx, shared_range) in enumerate(mpvi.shared_row_ranges)
        rows      = collect(shared_range)
        n_theta_g = length(mpvi.theta_ranges[group_idx])
        m         = length(rows) - n_theta_g
        k         = length(rows) ÷ m
        filter!(sol.CRs) do cr
            for i in 1:m
                player_rows = [rows[(p-1)*m + i] for p in 1:k]
                active      = [r ∈ cr.AS for r in player_rows]
                (all(active) || !any(active)) || return false
            end
            return true
        end
    end
    println("  [filter] $(length(sol.CRs)) valid CRs after Theorem 3.6")
end

function show_solution(result, mpvi)
    sol, lb, ub = result.sol, result.lb, result.ub
    mid   = (lb .+ ub) ./ 2
    betas = [mid]
    for i in 1:length(mid)
        b = copy(mid); b[i] = lb[i] + 0.01; push!(betas, b)
        b = copy(mid); b[i] = ub[i] - 0.01; push!(betas, b)
    end
    betas = filter(b -> all(mpvi.C * b .<= mpvi.e .+ 1e-8), betas)
    println("=== GNE Solution — $(n_critical_regions(sol)) critical regions ===")
    println()
    for beta in betas
        x = evaluate_gne(sol, beta)
        x === nothing ?
            println("  x*(β=$(round.(beta, digits=2))) = infeasible") :
            println("  x*(β=$(round.(beta, digits=2))) = $(round.(x, digits=4))")
    end
    println()
end

function show_pwa_map(sol)
    s = sol.scaling isa Number ? sol.scaling : sol.scaling[1]
    t = sol.translation isa Number ? sol.translation : sol.translation[1]
    println("=== PWA map x*(θ) per critical region ===")
    for (i, cr) in enumerate(sol.CRs)
        c0 = cr.z[2, :]
        c1 = cr.z[1, :] .* s
        println("\n  CR $i  AS=$(cr.AS)")
        println("  x*(θ) = $(round.(c0 .- c1.*t, digits=4)) + $(round.(c1, digits=4))⋅θ")
    end
end

show_pwa_map(result::NamedTuple) = show_pwa_map(result.sol)