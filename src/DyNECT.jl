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
import CommonSolve: solve  #Needed to override the 'solve' exported by DAQP
export solve

end