# Static_mpGNE.jl
module Static_mpGNE

using Symbolics
using LinearAlgebra

include("Types.jl")
include("Interface.jl")
include("Validation.jl")
include("Operator.jl")
include("Constraints.jl")
include("Assembly.jl")

# types
export GameBuilder

# macros
export @player, @param, @cost, @constraint

# setup
export assign_params!

# validation
export validate_game

# characterization
export show_coupling
export check_monotonicity
export check_symmetry
export check_theta_feasibility

# display
export show_operator
export show_shared_constraints
export show_local_constraints
export show_feasible_set
export show_parametric_constraints
export show_theta_set
export show_mpvi

# pipeline
export build_mpvi
export materialize

end