include("../compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
include("../Solver.jl")
using Plots
using ParametricDAQP

t_start = time()
println("=== Test Name: Rosen scalar game ===")

game = GameBuilder(N=2)
@player game 1 n=1
@player game 2 n=1

@cost game 1  0.5*x1^2 - x1*x2
@cost game 2  x2^2 + x1*x2

@constraint game  x1 + x2 >= 1

# Assemble
mpvi = build_mpvi(game)
# show_mpvi(mpvi)

# Solve
θub = [5.0]
θlb = -[5.0]
mpvi_dynect = to_dynect_mpAVI(mpvi)
mpvi_dynect = DyNECT.setParameterSpace(mpvi_dynect, C=[1.0;-1.0;;], d=[θub;-θlb], ub=θub, lb=θlb)
sol = CommonSolve.solve(mpvi_dynect, DyNECT.ParametricDAQPSolver)
filter_gne_crs!(sol, mpvi)

x_values     = []
theta_values = []
for θ in θlb[1]:0.02:θub[1]
    x = DyNECT.evaluatePWA(sol, [θ])
    if x !== nothing
        push!(x_values, x)
        push!(theta_values, θ)
    end
end

x_array = hcat(x_values...)'
scatter(x_array[:, 1], x_array[:, 2], marker_z=theta_values,
    xlabel="x1", ylabel="x2", label="",
    colorbar_title="θ", right_margin=5Plots.mm)
display(plot!())

println("total: $(round(time() - t_start, digits=2))s")
readline()
