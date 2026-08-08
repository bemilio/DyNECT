# Scalar Rosen example using Nabetani reformulation + mpAVI solver,
# where the game itself depends on a parameter γ (parametric LQGNEP).

using DyNECT
using CommonSolve
using LinearAlgebra
using ParametricDAQP
using Plots

# ============================================================================
# Step 1: Define the parametric game
# ============================================================================
# J₁ = -.5(x₁)² + x₁x₂
# J₂ = - (x₂)² - x₁x₂
# x₁ + x₂ - γ ≥ 0;   γ is a parameter of the game

gnep = DyNECT.LQGNEP(
    Q = [[[1.;;], [-1.;;]],
         [[1.;;], [2.;;]]],
    q = [[0.], [0.]],
    A_loc = [zeros(0, 1), zeros(0, 1)],
    b_loc = [Float64[], Float64[]],
    A_sh = [[-1;;], [-1;;]],
    b_sh = [0.],
    B_sh_γ = [-1.;;]
)

# ============================================================================
# Step 2: Solve parametrically
# ============================================================================

# restrict parameter space for reparametrization
θub = [5.0]
θlb = [-5.0]

sol = CommonSolve.solve(gnep, DyNECT.NabetaniParametrizationSolver; θub = θub, θlb = θlb)

# ============================================================================
# Step 3: Check solution
# ============================================================================

# Test solution: x1 = γ - x2, -1 < x2 < 0.5, where γ is the parameter of the parametric game
# θ is the parameter of the nabetani parametrization that carachterizes all GNEs for a given γ
tol = 1e-6
function check_all_solutions(sol, tol)
    all_solutions_found = true
    for γ in 0:0.1:1
        for θ in -γ:0.1:γ/2
            x = DyNECT.evaluatePWA(sol, [θ; γ])
            if !isnothing(x)
                ok_eq = norm(x[1] - (γ - x[2])) < tol
                ok_bounds = (-γ - tol <= x[2] <= γ/2+tol)
                if !(ok_eq && ok_bounds)
                    println("FAIL: γ=$γ, θ=$θ, x=$x, ok_eq=$ok_eq, ok_bounds=$ok_bounds, residual=$(norm(x[1] - (γ - x[2])))")
                end
                all_solutions_found = all_solutions_found & ok_eq
                all_solutions_found = all_solutions_found & ok_bounds
            end
        end
    end
    return all_solutions_found
end
all_solutions_found = check_all_solutions(sol, tol)

if all_solutions_found
    println("The parametric solution is correct.")
else
    println("The parametric solution is not correct.")
end

# ============================================================================
# Step 4: Plot (solutions for a fixed γ)
# ============================================================================



if all_solutions_found
    println("all_solutions_found is true")
else
    println("all_solutions_found is FALSE")
end


γ_plot = .2
test_interval = θlb[1]:0.1:θub[1]
x_sol = []
for θ in test_interval
    x = DyNECT.evaluatePWA(sol, [θ; γ_plot])
    if !isnothing(x)
        push!(x_sol, x)
    end
end

if !isempty(x_sol)
    x1 = [x[1] for x in x_sol]
    x2 = [x[2] for x in x_sol]
    plt = scatter(x1, x2;
        xlabel="x₁",
        ylabel="x₂",
        xlims=(-3, 3),
        ylims=(-3, 3),
        aspect_ratio=:equal,
        label=""
    )
    x_line = collect(-5:0.1:5)
    y_line = γ_plot .- x_line
    plot!(plt, x_line, y_line; color=:red, linewidth=2, label="y = γ_plot - x")
    display(plt)
    savefig(plt, "examples/parametricLQGNEP_solution.png")
end
