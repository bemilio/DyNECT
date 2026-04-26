include("../compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
include("../Solver.jl")
using LinearAlgebra
using ParametricDAQP

t_start = time()
println("=== Test Name: Vector game 2-player — Assembly + Solution Verification ===")

Q1 = [2.0 0.0; 0.0 1.0]
Q2 = [3.0 0.0; 0.0 2.0]
R  = [1.0 0.0; 0.0 1.0]
c1 = [-1.0, 0.0]
c2 = [0.0, -1.0]

game = GameBuilder(N=2)
@player game 1 n=2
@player game 2 n=2

@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  [x_1_1 + x_1_2 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0]

mpvi = build_mpvi(game)
mat  = materialize(mpvi)

#Verifying assembly
println("\n=== Step 1: Assembly verification ===")
println("A ($(size(mat.A))):")
display(mat.A)
println("B ($(size(mat.B))):")
display(mat.B)
println("d: ", mat.d)
println("H ($(size(mat.H))):")
display(mat.H)
println("f: ", mat.f)

# Solving
θub = [3.0, 2.0]
θlb = [0.0, 0.0]
mpvi_dynect = to_dynect_mpAVI(mpvi)
mpvi_dynect = DyNECT.setParameterSpace(mpvi_dynect,
    C = [Matrix{Float64}(I, 2, 2); -Matrix{Float64}(I, 2, 2)],
    d = [θub; -θlb],
    ub = θub, lb = θlb)
sol = CommonSolve.solve(mpvi_dynect, DyNECT.ParametricDAQPSolver)

println("\n=== Step 2: CRs before filtering ===")
for (i, cr) in enumerate(sol.CRs)
    println("  CR $i — AS: $(cr.AS)")
end

filter_gne_crs!(sol, mpvi)

#Constraint violation check
println("\n=== Step 3: Constraint violation check ===")
beta_test = [1.5, 1.0]
x = DyNECT.evaluatePWA(sol, beta_test)
println("β = $beta_test")
println("x* = $x")
println()

if x !== nothing
    lhs = mat.A * x
    rhs = mat.B * beta_test + mat.d
    println("Row-by-row constraint check  (A*x ≤ B*β + d):")
    local any_violation = false
    for i in 1:length(lhs)
        violation = lhs[i] - rhs[i]
        status = violation > 1e-6 ? "  ← VIOLATED" : ""
        if violation > 1e-6; any_violation = true; end
        println("  row $i:  $(round(lhs[i], digits=4)) ≤ $(round(rhs[i], digits=4))$status")
    end
    println()

    # === Step 4: Active set consistency check ===
    println("=== Step 4: Active set consistency (CR AS vs solution) ===")
    cr = sol.CRs[1]
    println("Surviving CR AS: $(cr.AS)")
    println()
    for row in cr.AS
        slack = lhs[row] - rhs[row]
        println("  row $row in AS: A[$row]*x - (B[$row]*β + d[$row]) = $(round(slack, digits=6))")
        if abs(slack) > 1e-4
            println("    ← INCONSISTENCY: row $row is in AS but constraint is not tight")
        end
    end

    if !any_violation
        println("✅ No constraint violations — solution is feasible")
    else
        println("❌ Constraint violations detected")
        println("   Assembly verified correct (Step 1)")
        println("   Active set inconsistent with solution (Step 4)")
        println("   → Suspecting issue in pDAQP evaluatePWA")
    end
end

println("total: $(round(time() - t_start, digits=2))s")