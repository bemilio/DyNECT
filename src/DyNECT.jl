module DyNECT

# Basic dependencies
using LinearAlgebra, SparseArrays, MatrixEquations, BlockDiagonals, BlockArrays, CommonSolve

# External solvers
using DAQP, Clarabel, JuMP, Monviso, ParametricDAQP


# CommonSolve

include("types.jl")
include("utils.jl")
include("inf_hor_tools.jl")
include("interfaces.jl")

using CommonSolve
include("solvers.jl")
export DynLQGame, DynLQGame2mpAVI
export StaticGNEGame, StaticGNE2mpAVI, filter_gne_crs!, OptimalGNEResult, select_optimal_gne!, mpGNESolver #added v_static_mpGNE
export AVI, IterativeSolverParams
export solve

end
