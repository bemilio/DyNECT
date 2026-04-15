# Solver.jl — MPVIAssembly → DyNECT.mpAVI → pDAQP
#using .Static_mpGNE
using DyNECT
using CommonSolve
using LinearAlgebra

function to_dynect_mpAVI(mpvi, lb::Vector{Float64}, ub::Vector{Float64})
    mat = materialize(mpvi)
    return DyNECT.mpAVI(mat.H, mat.Ftheta, mat.f, mat.A, mat.B, mat.d;
                        C=mpvi.C, d=mpvi.e, lb=lb, ub=ub)
end

function solve_gne(mpvi, lb::Vector{Float64}, ub::Vector{Float64})
    sol = CommonSolve.solve(to_dynect_mpAVI(mpvi, lb, ub),
                            DyNECT.ParametricDAQPSolver)
    return (sol=sol, lb=lb, ub=ub)
end

function evaluate_gne(sol, beta::Vector{Float64})
    return DyNECT.evaluatePWA(sol, beta)
end

n_critical_regions(sol) = length(sol.CRs)

function show_solution(result, mpvi)
    sol, lb, ub = result.sol, result.lb, result.ub
    mid   = (lb .+ ub) ./ 2
    betas = [mid]
    for i in 1:length(mid)
        b = copy(mid); b[i] = lb[i] + 0.01; push!(betas, b)
        b = copy(mid); b[i] = ub[i] - 0.01; push!(betas, b)
    end
    betas = filter(b -> all(mpvi.C * b .<= mpvi.e .+ 1e-8), betas)
    println("=== GNE Solution — $(n_critical_regions(result.sol)) critical regions ===")
    println()
    for beta in betas
        x = evaluate_gne(result.sol, beta)
        if x === nothing
            println("  x*(β=$(round.(beta, digits=2))) = infeasible")
        else
            println("  x*(β=$(round.(beta, digits=2))) = $(round.(x, digits=4))")
        end
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
        offset = -c1 .* t
        println("\n--- CR $i  (active set: rows $(cr.AS)) ---")
        println("  x*(θ) = $(round.(c0 .+ offset, digits=4)) + $(round.(c1, digits=4)) * θ")
    end
end

show_pwa_map(result::NamedTuple) = show_pwa_map(result.sol)