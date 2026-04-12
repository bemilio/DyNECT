# Fast_mpGNE.jl
# Compact single-file GNE → mpVI pipeline
# No symbolic parameters — numeric matrices defined in caller scope
# Silent assembly, prints only "mpvi OK"

module Fast_mpGNE
using Symbolics
using LinearAlgebra

# TYPES
const MatrixEntry = Union{Float64, Num}

struct PlayerRecord
    index::Int
    symbol::Symbol
    local_indices::UnitRange{Int}
    dim::Int
    global_cols::UnitRange{Int}
end

struct CostRecord
    player::Int
    expression::Num
end

struct ConstraintRecord
    expression_lhs::Num
    rhs::Float64
    sense::Symbol
    active::Bool
    label::String
    group_label::String
end

struct MPVIAssembly
    H::Matrix{MatrixEntry}
    Ftheta::Matrix{MatrixEntry}
    f::Vector{MatrixEntry}
    A::Matrix{MatrixEntry}
    B::Matrix{MatrixEntry}
    d::Vector{Float64}
    C::Matrix{Float64}
    e::Vector{Float64}
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
    next_auto_label_index::Int
end

function GameBuilder(; N::Int)
    @assert N >= 2 "Need at least 2 players"
    return GameBuilder(N,
        Vector{PlayerRecord}(),
        Vector{CostRecord}(),
        Vector{ConstraintRecord}(),
        1)
end

# HELPERS
function _display_var(player_index::Int, j::Int, dim::Int)::String
    return dim == 1 ? "x_$(player_index)" : "x_$(player_index)_$(j)"
end

function _build_symbolic_vars(players::Vector{PlayerRecord})
    vars = Dict{Symbol, Num}()
    for p in players
        for j in 1:p.dim
            varname = Symbol(:x_, p.index, :_, j)
            vars[varname] = only(@variables $varname)
        end
    end
    return vars
end

# fast version: env contains only decision variables
# numeric params are already in caller scope, picked up by Core.eval
function _build_full_env(players::Vector{PlayerRecord})
    env = Dict{Symbol, Any}()
    for p in players
        vec = Vector{Num}(undef, p.dim)
        for j in 1:p.dim
            varname = Symbol(:x_, p.index, :_, j)
            var = only(@variables $varname)
            env[varname] = var
            vec[j] = var
        end
        env[Symbol(:x, p.index)] = p.dim == 1 ? vec[1] : vec
    end
    return env
end

function _inject_symbolic_vars(expr, env::Dict{Symbol, Any})
    if expr isa Symbol
        return haskey(env, expr) ? env[expr] : expr
    elseif expr isa Expr
        if expr.head == :ref && expr.args[1] isa Symbol
            sym = expr.args[1]
            if haskey(env, sym)
                return :($(env[sym])[$(expr.args[2])])
            end
        end
        return Expr(expr.head,
            [_inject_symbolic_vars(a, env) for a in expr.args]...)
    else
        return expr
    end
end

function _extract_coefficients(expr::Num, vars::Dict{Symbol, Num})
    coeffs = Dict{Symbol, Float64}()
    for (sym, var) in vars
        c = Symbolics.value(Symbolics.simplify(Symbolics.derivative(expr, var)))
        if c isa Number && c != 0
            coeffs[sym] = Float64(c)
        end
    end
    zero_sub = Dict(var => 0.0 for (_, var) in vars)
    constant_val = Float64(Symbolics.value(Symbolics.simplify(
        Symbolics.substitute(expr, zero_sub))))
    return coeffs, constant_val
end

function _format_constraint(coeffs::Dict{Symbol,Float64}, rhs::Float64,
                              players::Vector{PlayerRecord})
    terms = String[]
    for p in players
        for j in 1:p.dim
            sym = Symbol(:x_, p.index, :_, j)
            if haskey(coeffs, sym)
                push!(terms, "[$(coeffs[sym])]$(_display_var(p.index, j, p.dim))")
            end
        end
    end
    return "$(isempty(terms) ? "0" : join(terms, " + ")) <= [$rhs]"
end

function _players_in_constraint(coeffs::Dict{Symbol,Float64},
                                  players::Vector{PlayerRecord})
    involved = Int[]
    for p in players
        for j in 1:p.dim
            if haskey(coeffs, Symbol(:x_, p.index, :_, j)) && !(p.index in involved)
                push!(involved, p.index)
            end
        end
    end
    return sort(involved)
end

function group_constraints(constraints::Vector{ConstraintRecord})
    groups = Dict{String, Vector{ConstraintRecord}}()
    for c in constraints
        c.active || continue
        push!(get!(groups, c.group_label, ConstraintRecord[]), c)
    end
    return groups
end

# MACROS 
macro player(game, index, ndecl)
    @assert ndecl.head == :(=) && ndecl.args[1] == :n "Expected n=k"
    return quote
        local _index = $(esc(index))
        local _dim   = $(esc(ndecl.args[2]))
        for p in $(esc(game)).players
            @assert p.index != _index "Player $_index already registered"
        end
        local _col_start = 1
        for p in $(esc(game)).players
            _col_start += p.dim
        end
        local _global_cols = _col_start:(_col_start + _dim - 1)
        push!($(esc(game)).players, PlayerRecord(
            _index, Symbol(:x_, _index), 1:_dim, _dim, _global_cols))
    end
end

macro cost(game, player_index, expr)
    return quote
        local _env = Base.invokelatest(_build_full_env, $(esc(game)).players)
        local _result = Core.eval(@__MODULE__,
            Base.invokelatest(_inject_symbolic_vars, $(QuoteNode(expr)), _env))
        local _scalar = _result isa AbstractArray ? only(_result) : _result
        @assert _scalar isa Num "Cost must evaluate to a symbolic scalar. Got $(typeof(_scalar))"
        push!($(esc(game)).costs, CostRecord($(esc(player_index)), _scalar))
    end
end

macro constraint(game, expr)
    @assert expr.head == :call "Expected a comparison expression (<=, >=, ==)"
    op  = expr.args[1]
    lhs = expr.args[2]
    rhs = expr.args[3]
    return quote
        local _svars = Base.invokelatest(_build_symbolic_vars, $(esc(game)).players)
        local _env   = Base.invokelatest(_build_full_env, $(esc(game)).players)
        local _lhs_eval = Core.eval(@__MODULE__,
            Base.invokelatest(_inject_symbolic_vars, $(QuoteNode(lhs)), _env))
        local _rhs_eval = Core.eval(@__MODULE__,
            Base.invokelatest(_inject_symbolic_vars, $(QuoteNode(rhs)), _env))
        local _lhs_vec = _lhs_eval isa AbstractArray ? Num.(_lhs_eval) : [Num(_lhs_eval)]
        local _rhs_vec = _rhs_eval isa AbstractArray ? Num.(_rhs_eval) : [Num(_rhs_eval)]
        if length(_lhs_vec) == 1 && length(_rhs_vec) > 1
            _lhs_vec = fill(_lhs_vec[1], length(_rhs_vec))
        elseif length(_rhs_vec) == 1 && length(_lhs_vec) > 1
            _rhs_vec = fill(_rhs_vec[1], length(_lhs_vec))
        end
        @assert length(_lhs_vec) == length(_rhs_vec) "Constraint dimension mismatch"
        local _op = $(QuoteNode(op))
        local _group_label = "constraint_" * string($(esc(game)).next_auto_label_index)
        $(esc(game)).next_auto_label_index += 1
        local _row_counter = Ref(0)
        function _register_row(norm_lhs::Num, rhs_const::Float64)
            _row_counter[] += 1
            local _c, _k = Base.invokelatest(_extract_coefficients, norm_lhs, _svars)
            local _rv    = rhs_const - _k
            local _inv   = Base.invokelatest(_players_in_constraint, _c, $(esc(game)).players)
            local _lab   = _group_label * "_row_" * string(_row_counter[])
            push!($(esc(game)).constraints, ConstraintRecord(
                norm_lhs, rhs_const, :leq, true, _lab, _group_label))
        end
        for _i in 1:length(_lhs_vec)
            local _lhs_row   = _lhs_vec[_i]
            local _rhs_float = Float64(Symbolics.value(Symbolics.simplify(_rhs_vec[_i])))
            if _op == :(<=)
                _register_row(_lhs_row, _rhs_float)
            elseif _op == :(>=)
                _register_row(-_lhs_row, -_rhs_float)
            elseif _op == :(==)
                _register_row(_lhs_row,  _rhs_float)
                _register_row(-_lhs_row, -_rhs_float)
            else
                error("Constraint must use <=, >= or ==")
            end
        end
    end
end

# ASSEMBLY 
function _classify_constraints(game::GameBuilder)
    svars  = Base.invokelatest(_build_symbolic_vars, game.players)
    groups = Base.invokelatest(group_constraints, game.constraints)
    local_groups  = String[]
    shared_groups = String[]
    for (glabel, rows) in groups
        coeffs, _ = Base.invokelatest(_extract_coefficients, rows[1].expression_lhs, svars)
        involved  = Base.invokelatest(_players_in_constraint, coeffs, game.players)
        length(involved) == 1 ? push!(local_groups, glabel) : push!(shared_groups, glabel)
    end
    return groups, sort(local_groups), sort(shared_groups)
end

function _assemble_operator(game::GameBuilder)
    n_total = sum(p.dim for p in game.players)
    svars   = Base.invokelatest(_build_symbolic_vars, game.players)
    H = Matrix{MatrixEntry}(undef, n_total, n_total)
    f = Vector{MatrixEntry}(undef, n_total)
    for cost in game.costs
        pi   = game.players[findfirst(p -> p.index == cost.player, game.players)]
        expr = cost.expression   # already numeric — no substitute_params needed
        for j in 1:pi.dim
            row  = pi.global_cols[j]
            grad = Symbolics.simplify(Symbolics.derivative(expr,
                svars[Symbol(:x_, pi.index, :_, j)]))
            for pk in game.players
                for k in 1:pk.dim
                    coeff = Symbolics.simplify(Symbolics.derivative(grad,
                        svars[Symbol(:x_, pk.index, :_, k)]))
                    val   = Symbolics.value(coeff)
                    H[row, pk.global_cols[k]] = val isa Number ? Float64(val) : coeff
                end
            end
            zero_sub  = Dict(var => 0.0 for (_, var) in svars)
            const_val = Symbolics.value(Symbolics.simplify(
                Symbolics.substitute(grad, zero_sub)))
            f[row] = const_val isa Number ? Float64(const_val) : const_val
        end
    end
    return H, f
end

function _extract_constraint_blocks(rows::Vector{ConstraintRecord}, game::GameBuilder)
    svars = Base.invokelatest(_build_symbolic_vars, game.players)
    m     = length(rows)
    coeffs_first, _ = Base.invokelatest(_extract_coefficients, rows[1].expression_lhs, svars)
    players_involved = Base.invokelatest(_players_in_constraint, coeffs_first, game.players)
    A_blocks = Vector{Matrix{MatrixEntry}}()
    for pi in players_involved
        p = game.players[findfirst(pl -> pl.index == pi, game.players)]
        push!(A_blocks, Matrix{MatrixEntry}(zeros(Float64, m, p.dim)))
    end
    b = zeros(Float64, m)
    for (row_idx, row) in enumerate(rows)
        coeffs, _ = Base.invokelatest(_extract_coefficients, row.expression_lhs, svars)
        for (k, pi) in enumerate(players_involved)
            p = game.players[findfirst(pl -> pl.index == pi, game.players)]
            for j in 1:p.dim
                sym = Symbol(:x_, p.index, :_, j)
                haskey(coeffs, sym) && (A_blocks[k][row_idx, j] = coeffs[sym])
            end
        end
        b[row_idx] = row.rhs
    end
    return A_blocks, b, players_involved
end

function _assemble_shared_reparametrization(A_blocks, b, players_involved, game)
    k         = length(players_involved)
    m         = length(b)
    n_theta_g = (k - 1) * m
    n_total   = sum(p.dim for p in game.players)
    A_hat = Matrix{MatrixEntry}(zeros(Float64, k * m, n_total))
    for (pi_idx, pi) in enumerate(players_involved)
        p = game.players[findfirst(pl -> pl.index == pi, game.players)]
        A_hat[(pi_idx-1)*m+1:pi_idx*m, p.global_cols] .= A_blocks[pi_idx]
    end
    B_g = zeros(Float64, k * m, n_theta_g)
    for pi_idx in 1:(k-1)
        B_g[(pi_idx-1)*m+1:pi_idx*m, (pi_idx-1)*m+1:pi_idx*m] .= I(m)
    end
    for prev_idx in 1:(k-1)
        B_g[(k-1)*m+1:k*m, (prev_idx-1)*m+1:prev_idx*m] .= -I(m)
    end
    d_g = zeros(Float64, k * m)
    d_g[(k-1)*m+1:k*m] .= b
    return A_hat, B_g, d_g, n_theta_g
end

function _assemble_local_constraints(game::GameBuilder)
    groups, local_groups, _ = Base.invokelatest(_classify_constraints, game)
    n_total = sum(p.dim for p in game.players)
    svars   = Base.invokelatest(_build_symbolic_vars, game.players)
    A_rows  = Vector{Vector{Float64}}()
    d_vals  = Float64[]
    for glabel in local_groups
        for row in groups[glabel]
            coeffs, _ = Base.invokelatest(_extract_coefficients, row.expression_lhs, svars)
            A_row = zeros(Float64, n_total)
            for p in game.players
                for j in 1:p.dim
                    sym = Symbol(:x_, p.index, :_, j)
                    haskey(coeffs, sym) && (A_row[p.global_cols[j]] = coeffs[sym])
                end
            end
            push!(A_rows, A_row)
            push!(d_vals, row.rhs)
        end
    end
    isempty(A_rows) && return Matrix{Float64}(undef, 0, n_total), Float64[]
    return reduce(vcat, [r' for r in A_rows]), d_vals
end

function _assemble_feasible_set(game::GameBuilder)
    groups, local_groups, shared_groups = Base.invokelatest(_classify_constraints, game)
    n_total = sum(p.dim for p in game.players)
    A_local, d_local = Base.invokelatest(_assemble_local_constraints, game)
    n_local_rows = size(A_local, 1)
    A_shared_rows = Vector{Vector{MatrixEntry}}()
    B_shared_rows = Vector{Vector{Float64}}()
    d_shared_vals = Float64[]
    n_theta = 0
    for glabel in shared_groups
        A_blocks, b, involved = Base.invokelatest(
            _extract_constraint_blocks, groups[glabel], game)
        A_hat, B_g, d_g, n_theta_g = Base.invokelatest(
            _assemble_shared_reparametrization, A_blocks, b, involved, game)
        for i in 1:size(A_hat, 1); push!(A_shared_rows, vec(A_hat[i:i, :])); end
        for i in 1:size(B_g, 1);   push!(B_shared_rows, B_g[i, :]); end
        append!(d_shared_vals, d_g)
        n_theta += n_theta_g
    end
    n_shared_rows = length(A_shared_rows)
    if n_local_rows == 0 && n_shared_rows == 0
        A = Matrix{MatrixEntry}(undef, 0, n_total)
    elseif n_local_rows == 0
        A = Matrix{MatrixEntry}(reduce(vcat, [r' for r in A_shared_rows]))
    elseif n_shared_rows == 0
        A = Matrix{MatrixEntry}(A_local)
    else
        A = vcat(Matrix{MatrixEntry}(A_local),
                 reduce(vcat, [r' for r in A_shared_rows]))
    end
    B = zeros(Float64, size(A, 1), n_theta)
    if n_shared_rows > 0
        col_offset, row_offset = 0, n_local_rows
        for glabel in shared_groups
            A_blocks, b, involved = Base.invokelatest(
                _extract_constraint_blocks, groups[glabel], game)
            _, B_g, _, n_theta_g = Base.invokelatest(
                _assemble_shared_reparametrization, A_blocks, b, involved, game)
            n_g_rows = size(B_g, 1)
            B[row_offset+1:row_offset+n_g_rows, col_offset+1:col_offset+n_theta_g] .= B_g
            row_offset += n_g_rows
            col_offset += n_theta_g
        end
    end
    return A, B, vcat(d_local, d_shared_vals), n_theta
end

function _assemble_theta_set(game::GameBuilder)
    groups, _, shared_groups = Base.invokelatest(_classify_constraints, game)
    n_theta = 0
    group_info = Tuple{Int,Int,Vector{Float64}}[]
    for glabel in shared_groups
        A_blocks, b, involved = Base.invokelatest(
            _extract_constraint_blocks, groups[glabel], game)
        k = length(involved); m = length(b)
        n_theta += (k - 1) * m
        push!(group_info, (k, m, b))
    end
    n_theta == 0 && return Matrix{Float64}(undef, 0, 0), Float64[]
    C_rows = Vector{Vector{Float64}}()
    e_vals = Float64[]
    col_offset = 0
    for (k, m, b) in group_info
        n_theta_g = (k - 1) * m
        for i in 1:n_theta_g
            row = zeros(Float64, n_theta)
            r   = mod1(i, m)
            row[col_offset + i] = b[r] >= 0 ? -1.0 : 1.0
            push!(C_rows, row); push!(e_vals, 0.0)
        end
        for r in 1:m
            row = zeros(Float64, n_theta)
            sign = b[r] >= 0 ? 1.0 : -1.0
            for prev_idx in 1:(k-1)
                row[col_offset + (prev_idx-1)*m + r] = sign
            end
            push!(C_rows, row)
            push!(e_vals, b[r] >= 0 ? b[r] : -b[r])
        end
        col_offset += n_theta_g
    end
    return reduce(vcat, [r' for r in C_rows]), e_vals
end

# BUILD — main entry point
function build_mpvi(game::GameBuilder)
    H, f             = Base.invokelatest(_assemble_operator, game)
    n_total          = sum(p.dim for p in game.players)
    A, B, d, n_theta = Base.invokelatest(_assemble_feasible_set, game)
    C, e             = Base.invokelatest(_assemble_theta_set, game)
    Ftheta           = zeros(n_total, max(n_theta, 0))

    groups, local_groups, shared_groups = Base.invokelatest(_classify_constraints, game)

    local_row_ranges = UnitRange{Int}[]
    current_row = 1
    for glabel in local_groups
        n = length(groups[glabel])
        push!(local_row_ranges, current_row:(current_row + n - 1))
        current_row += n
    end

    shared_row_ranges = UnitRange{Int}[]
    theta_ranges      = UnitRange{Int}[]
    current_theta     = 1
    for glabel in shared_groups
        A_blocks, b, involved = Base.invokelatest(
            _extract_constraint_blocks, groups[glabel], game)
        k = length(involved); m = length(b)
        n_theta_g = (k - 1) * m
        n_rows_g  = k * m
        push!(shared_row_ranges, current_row:(current_row + n_rows_g - 1))
        push!(theta_ranges, current_theta:(current_theta + n_theta_g - 1))
        current_row   += n_rows_g
        current_theta += n_theta_g
    end

    mpvi = MPVIAssembly(
        H, Ftheta, f,
        A, B, d,
        C, e,
        game.players,
        [p.global_cols for p in game.players],
        local_row_ranges,
        shared_row_ranges,
        theta_ranges,
        local_groups,
        shared_groups
    )

    println("mpvi OK")
    return mpvi
end

# MATERIALIZE — convert symbolic entries to Float64
function materialize(mpvi::MPVIAssembly)
    to_f64_matrix(M) = Float64.(Symbolics.value.(M))
    to_f64_vector(v) = Float64.(Symbolics.value.(v))
    return (
        H      = to_f64_matrix(mpvi.H),
        Ftheta = to_f64_matrix(mpvi.Ftheta),
        f      = to_f64_vector(mpvi.f),
        A      = to_f64_matrix(mpvi.A),
        B      = to_f64_matrix(mpvi.B),
        d      = mpvi.d,
        C      = mpvi.C,
        e      = mpvi.e,
    )
end

# DISPLAY
function show_mpvi(mpvi::MPVIAssembly)
    players = mpvi.player_records
    A, B, d = mpvi.A, mpvi.B, mpvi.d
    n_theta  = size(B, 2)

    println("Reparametrization Ax ≤ Bθ + d")
    println("  dims: A$(size(A))  B$(size(B))  n_theta=$n_theta")
    println()
    for i in 1:size(A, 1)
        lhs_terms = String[]
        for p in players
            for j in 1:p.dim
                val = A[i, p.global_cols[j]]
                valv = val isa Number ? val : Symbolics.value(val)
                if !(valv isa Number && iszero(valv))
                    dvar = p.dim == 1 ? "x_$(p.index)" : "x_$(p.index)_$(j)"
                    push!(lhs_terms, "$(round(Float64(valv), digits=4))⋅$dvar")
                end
            end
        end
        rhs_terms = String[]
        !iszero(d[i]) && push!(rhs_terms, "$(round(d[i], digits=4))")
        for t in 1:n_theta
            !iszero(B[i,t]) && push!(rhs_terms, "$(round(B[i,t], digits=4))⋅θ$t")
        end
        lhs = isempty(lhs_terms) ? "0" : join(lhs_terms, " + ")
        rhs = isempty(rhs_terms) ? "0" : join(rhs_terms, " + ")
        println("  row $i:  $lhs  ≤  $rhs")
    end
    println()
end

# EXPORTS
export GameBuilder, MPVIAssembly
export @player, @cost, @constraint
export build_mpvi, materialize
export show_mpvi

end