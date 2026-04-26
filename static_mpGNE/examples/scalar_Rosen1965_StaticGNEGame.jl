import Pkg
Pkg.activate("DyNECT-thesis")
using DyNECT, CommonSolve, Plots

t_start = time()

game = StaticGNEGame(
    N = 2, n = [1, 1],
    Q = [[fill(1.0,1,1), fill(-1.0,1,1)], [fill(1.0,1,1), fill(2.0,1,1)]],
    q = [[0.0], [0.0]],
    A_loc = [zeros(0,1), zeros(0,1)],
    b_loc = [Float64[], Float64[]],
    A_sh  = [fill(-1.0,1,1), fill(-1.0,1,1)],
    b_sh  = [-1.0]
)

mpvi         = StaticGNE2mpAVI(game)
mpvi_bounded = DyNECT.setParameterSpace(mpvi, lb=[-5.0], ub=[5.0], C=[1.0;-1.0;;], d=[0.0; 1.0])
sol          = CommonSolve.solve(mpvi_bounded, DyNECT.ParametricDAQPSolver)
sol_unfiltered = deepcopy(sol)
filter_gne_crs!(sol, game)

function collect_sol(sol, θrange=-5.0:0.02:5.0)
    xs, θs = [], []
    for θ in θrange
        x = DyNECT.evaluatePWA(sol, [θ])
        if x !== nothing; push!(xs, x); push!(θs, θ); end
    end
    hcat(xs...)', θs
end

arr1, θs1 = collect_sol(sol_unfiltered)
arr2, θs2 = collect_sol(sol)

p1 = scatter(arr1[:,1], arr1[:,2], marker_z=θs1, xlabel="x1", ylabel="x2",
             title="Unfiltered", label="", colorbar_title="θ", right_margin=5Plots.mm)
p2 = scatter(arr2[:,1], arr2[:,2], marker_z=θs2, xlabel="x1", ylabel="x2",
             title="Filtered", label="", colorbar_title="θ", right_margin=5Plots.mm)

display(plot(p1, p2, layout=(1,2), size=(900,400)))
println("total: $(round(time() - t_start, digits=2))s")
