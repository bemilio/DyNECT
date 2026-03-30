module DyNECT

# Basic dependencies
using LinearAlgebra, SparseArrays, MatrixEquations, BlockDiagonals, BlockArrays, CommonSolve

# External solvers
using DAQP, Clarabel, JuMP, Monviso, ParametricDAQP

# Automatic differentiation
using Zygote


# CommonSolve

include("types.jl")
include("utils.jl")
include("inf_hor_tools.jl")
include("interfaces.jl")

using CommonSolve
include("solvers.jl")
export DynLQGameTI, DynLQGameTV, DynLQGame2mpAVI
export solve

end