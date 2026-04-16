include("../compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
include("../Solver.jl")
using Plots
using LinearAlgebra
using ParametricDAQP

t_start = time()
println("=== Test Name: Vector game 3-player ===")

Q1 = [2.0 0.0; 0.0 1.0]
Q2 = [3.0 0.0; 0.0 2.0]
Q3 = [4.0 0.0; 0.0 3.0]
R  = [1.0 0.0; 0.0 1.0]
c1 = [-1.0, 0.0]
c2 = [0.0, -1.0]
c3 = [0.0, 0.0]

game = GameBuilder(N=3)
@player game 1 n=2
@player game 2 n=2
@player game 3 n=2

@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2
@cost game 3  0.5*x3'*Q3*x3 + x3'*R'*x1 + c3'*x3

@constraint game  -x1 <= 0
@constraint game  -x2 <= 0
@constraint game  -x3 <= 0
@constraint game  [x_1_1 + x_1_2 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0]
@constraint game  [x_1_1 + x_2_1 + x_3_1, x_1_2 + x_2_2 + x_3_2] <= [5.0, 3.0]

mpvi = build_mpvi(game)
# show_mpvi(mpvi)

# n_theta = 6: [θ₁,θ₂] for group 1 (2 players, m=2), [θ₃,θ₄,θ₅,θ₆] for group 2 (3 players, m=2)
θub = [3.0, 2.0, 5.0, 3.0, 5.0, 3.0]
θlb = zeros(6)
mpvi_dynect = to_dynect_mpAVI(mpvi)
mpvi_dynect = DyNECT.setParameterSpace(mpvi_dynect,
    C = [Matrix{Float64}(I, 6, 6); -Matrix{Float64}(I, 6, 6)],
    d = [θub; -θlb],
    ub = θub, lb = θlb)
sol = CommonSolve.solve(mpvi_dynect, DyNECT.ParametricDAQPSolver)

filter_gne_crs!(sol, mpvi)

# --- Evaluate over θ grid (fix θ₃,θ₄,θ₅,θ₆ at midpoint, sweep θ₁,θ₂) ---
n1, n2 = 20, 20
θ_mid = (θlb .+ θub) ./ 2
θ1_range = range(θlb[1], θub[1], length=n1)
θ2_range = range(θlb[2], θub[2], length=n2)

x_values     = []
theta_values = []
for θ1 in θ1_range, θ2 in θ2_range
    β = copy(θ_mid); β[1] = θ1; β[2] = θ2
    x = DyNECT.evaluatePWA(sol, β)
    if x !== nothing
        push!(x_values, x)
        push!(theta_values, [θ1, θ2])
    end
end

x_array = hcat(x_values...)'
θ_array = hcat(theta_values...)'

labels = ["x_1_1" "x_1_2" "x_2_1" "x_2_2" "x_3_1" "x_3_2"]
plts = []
for j in 1:6
    p = scatter(θ_array[:, 1], θ_array[:, 2],
        marker_z = x_array[:, j],
        xlabel="θ₁", ylabel="θ₂",
        title=labels[j], label="",
        colorbar_title="value",
        right_margin=5Plots.mm)
    push!(plts, p)
end
display(plot(plts..., layout=(2,3), size=(1200,600)))

println("=== Surviving CRs ===")
for (i, cr) in enumerate(sol.CRs)
    println("CR $i — AS: $(cr.AS)")
    println("  Ath: ", cr.Ath)
    println("  bth: ", cr.bth)
    println("  z:   ", cr.z)
end
println("total: $(round(time() - t_start, digits=2))s")
readline()