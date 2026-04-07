# Types.jl — data structures for the GNE → mpVI pipeline

using Symbolics

const MatrixEntry = Union{Float64, Num}

struct PlayerRecord
    index::Int
    symbol::Symbol
    local_indices::UnitRange{Int}
    dim::Int
    global_cols::UnitRange{Int}
end

struct ParamRecord
    name::Symbol
    dims::Tuple{Vararg{Int}}
    symbolic_entries::Array{Num}
    value::Union{Array{Float64}, Nothing}  # nothing = still symbolic
end

struct CostRecord
    player::Int
    expression::Num
end

struct ConstraintRecord
    expression_lhs::Num
    rhs::Float64
    sense::Symbol        # always :leq after normalization
    active::Bool
    label::String        # row label:   constraint_3_row_1
    group_label::String  # group label: constraint_3
end

struct MPVIAssembly
    # solver inputs
    H::Matrix{MatrixEntry}
    Ftheta::Matrix{MatrixEntry}  # zero for now
    f::Vector{MatrixEntry}
    A::Matrix{MatrixEntry}       # A*x <= B*theta + d
    B::Matrix{MatrixEntry}
    d::Vector{Float64}
    C::Matrix{Float64}           # C*theta <= e
    e::Vector{Float64}
    # traceability
    player_records::Vector{PlayerRecord}
    player_ranges::Vector{UnitRange{Int}}
    local_row_ranges::Vector{UnitRange{Int}}
    shared_row_ranges::Vector{UnitRange{Int}}
    theta_ranges::Vector{UnitRange{Int}}
    active_local_labels::Vector{String}
    active_shared_labels::Vector{String}
end

mutable struct GameBuilder
    N::Int
    players::Vector{PlayerRecord}
    costs::Vector{CostRecord}
    constraints::Vector{ConstraintRecord}
    params::Vector{ParamRecord}
    next_auto_label_index::Int
end