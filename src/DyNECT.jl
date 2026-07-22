module DyNECT

using LinearAlgebra, SparseArrays, MatrixEquations, BlockDiagonals, BlockArrays, CommonSolve

# External solvers
using DAQP, Clarabel, Ipopt, JuMP, Monviso, ParametricDAQP

# Automatic differentiation
using Zygote

include("types.jl")
include("utils.jl")
include("inf_hor_tools.jl")
include("interfaces.jl")
include("solvers.jl")

export DynLQGame, DynLQGameTV, DynLQGame2mpAVI, AVI
export StaticGNEP, OptimalGNEP
import CommonSolve: solve  # Override the `solve` exported by DAQP
export solve

end
