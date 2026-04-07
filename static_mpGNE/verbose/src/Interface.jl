# Interface.jl — GameBuilder constructor, macros, and helpers

using Symbolics
using LinearAlgebra

function GameBuilder(; N::Int)
    @assert N >= 2 "Need at least 2 players"
    return GameBuilder(N,
        Vector{PlayerRecord}(),
        Vector{CostRecord}(),
        Vector{ConstraintRecord}(),
        Vector{ParamRecord}(),
        1)
end

# display name: x_1 for scalar players, x_1_1 x_1_2 for vector players
function _display_var(player_index::Int, j::Int, dim::Int)::String
    return dim == 1 ? "x_$(player_index)" : "x_$(player_index)_$(j)"
end

# flat dict of symbolic decision variables: :x_1_1 => x_1_1, etc.
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

# symbolic entries for a parameter matrix or vector
function _build_symbolic_param(name::Symbol, dims)
    if length(dims) == 1
        entries = Vector{Num}(undef, dims[1])
        for i in 1:dims[1]
            vname = Symbol(name, :_, i)
            entries[i] = only(@variables $vname)
        end
        return entries
    else
        r, c = dims[1], dims[2]
        entries = Matrix{Num}(undef, r, c)
        for i in 1:r, j in 1:c
            vname = Symbol(name, :_, i, :_, j)
            entries[i,j] = only(@variables $vname)
        end
        return entries
    end
end

# full symbolic environment: decision vars + param matrices
# scalar players: x1 → Num, vector players: x1 → Vector{Num}
function _build_full_env(players::Vector{PlayerRecord},
                          params::Vector{ParamRecord})
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
    for pr in params
        env[pr.name] = pr.symbolic_entries
    end
    return env
end

# walk expression AST replacing known symbols with symbolic objects
# two methods: Dict{Symbol,Any} for full env, Dict{Symbol,Num} for vars-only calls
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

function _inject_symbolic_vars(expr, vars::Dict{Symbol, Num})
    return _inject_symbolic_vars(expr, Dict{Symbol, Any}(k => v for (k,v) in vars))
end

# substitute numeric param values — leaves symbolic params untouched
function _substitute_params(expr::Num, params::Vector{ParamRecord})::Num
    sub_dict = Dict{Num, Float64}()
    for pr in params
        pr.value === nothing && continue
        for idx in eachindex(pr.symbolic_entries)
            sub_dict[pr.symbolic_entries[idx]] = Float64(pr.value[idx])
        end
    end
    isempty(sub_dict) && return expr
    return Symbolics.substitute(expr, sub_dict)
end

# extract affine coefficients and constant from a symbolic expression
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

# format normalized constraint as [coeff]x_i_j + ... <= [rhs]
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
    return "$(isempty(terms) ? "0" : join(terms, " + ")) <= [$(rhs)]"
end

# identify which players appear in a constraint by variable support
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

# group active ConstraintRecords by group_label
# lives here (not Constraints.jl) because Validation.jl needs it before Constraints.jl loads
function group_constraints(constraints::Vector{ConstraintRecord})
    groups = Dict{String, Vector{ConstraintRecord}}()
    for c in constraints
        c.active || continue
        push!(get!(groups, c.group_label, ConstraintRecord[]), c)
    end
    return groups
end

# Version 1 pretty print — scholastic cost display for new game debugging, used in @cost
# replaced by a one-liner in Version 2 (to be developed for speed performance)
function _is_nonzero(val)::Bool
    v = Symbolics.value(Symbolics.simplify(Num(val)))
    v isa Number && return !iszero(v)
    return true
end

function _fmt_coeff(val)::String
    return string(Symbolics.value(Symbolics.simplify(Num(val))))
end

function _pretty_print_cost(expr::Num, player::PlayerRecord,
                              players::Vector{PlayerRecord},
                              params::Vector{ParamRecord})::String
    terms = String[]
    all_dec_vars = Dict{Num, Float64}()
    for p in players
        for j in 1:p.dim
            vname = Symbol(:x_, p.index, :_, j)
            all_dec_vars[only(@variables $vname)] = 0.0
        end
    end

    # quadratic own terms
    for j in 1:player.dim
        for k in j:player.dim
            vj = Symbol(:x_, player.index, :_, j)
            vk = Symbol(:x_, player.index, :_, k)
            coeff = Symbolics.simplify(Symbolics.derivative(
                Symbolics.derivative(expr, only(@variables $vj)), only(@variables $vk)))
            Base.invokelatest(_is_nonzero, coeff) || continue
            if j == k
                c_str = Base.invokelatest(_fmt_coeff, Symbolics.simplify(coeff / 2))
                push!(terms, "($(c_str))$(_display_var(player.index, j, player.dim))²")
            else
                c_str = Base.invokelatest(_fmt_coeff, coeff)
                push!(terms, "($(c_str))$(_display_var(player.index, j, player.dim))⋅$(_display_var(player.index, k, player.dim))")
            end
        end
    end

    # cross terms
    for other in players
        other.index == player.index && continue
        for j in 1:player.dim
            for k in 1:other.dim
                vj = Symbol(:x_, player.index, :_, j)
                vk = Symbol(:x_, other.index, :_, k)
                coeff = Symbolics.simplify(Symbolics.derivative(
                    Symbolics.derivative(expr, only(@variables $vj)), only(@variables $vk)))
                Base.invokelatest(_is_nonzero, coeff) || continue
                c_str = Base.invokelatest(_fmt_coeff, coeff)
                push!(terms, "($(c_str))$(_display_var(player.index, j, player.dim))⋅$(_display_var(other.index, k, other.dim))")
            end
        end
    end

    # linear terms
    for j in 1:player.dim
        vj = Symbol(:x_, player.index, :_, j)
        d1 = Symbolics.derivative(expr, only(@variables $vj))
        lin = Symbolics.simplify(Symbolics.substitute(d1, all_dec_vars))
        Base.invokelatest(_is_nonzero, lin) || continue
        push!(terms, "($(Base.invokelatest(_fmt_coeff, lin)))$(_display_var(player.index, j, player.dim))")
    end

    # constant
    const_expr = Symbolics.simplify(Symbolics.substitute(expr, all_dec_vars))
    Base.invokelatest(_is_nonzero, const_expr) &&
        push!(terms, Base.invokelatest(_fmt_coeff, const_expr))

    return isempty(terms) ? "0" : join(terms, " + ")
end

# =============================================================================
# Macros
# note: @player registration order matters — players must be declared
# in ascending index order (1, 2, ..., N) for correct global column assignment
# =============================================================================

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
        println("  [player $_index]  x_$(_index) ∈ R^$(_dim)  →  global cols $(_global_cols)")
    end
end

macro param(game, name, dimsdecl)
    @assert dimsdecl.head == :(=) && dimsdecl.args[1] == :dims "Expected dims=(...)"
    return quote
        local _name = $(QuoteNode(name))
        local _raw  = $(esc(dimsdecl.args[2]))
        local _dims = _raw isa Int ? (_raw,) : _raw isa Tuple ? _raw : Tuple(_raw)
        local _entries = Base.invokelatest(_build_symbolic_param, _name, _dims)
        push!($(esc(game)).params, ParamRecord(_name, _dims, _entries, nothing))
        println("  [param $_name]  dims=$(_dims)")
    end
end

macro cost(game, player_index, expr)
    return quote
        local _env = Base.invokelatest(_build_full_env,
            $(esc(game)).players, $(esc(game)).params)
        local _result = Core.eval(@__MODULE__,
            Base.invokelatest(_inject_symbolic_vars, $(QuoteNode(expr)), _env))
        local _scalar = _result isa AbstractArray ? only(_result) : _result
        @assert _scalar isa Num "Cost must evaluate to a symbolic scalar. Got $(typeof(_scalar))"
        push!($(esc(game)).costs, CostRecord($(esc(player_index)), _scalar))
        local _p = $(esc(game)).players[findfirst(
            p -> p.index == $(esc(player_index)), $(esc(game)).players)]
        local _pretty = Base.invokelatest(_pretty_print_cost, _scalar, _p,
            $(esc(game)).players, $(esc(game)).params)
        println("  [cost player $($(esc(player_index)))]  J = $(_pretty)")
    end
end

macro constraint(game, expr)
    @assert expr.head == :call "Expected a comparison expression (<=, >=, ==)"
    op  = expr.args[1]
    lhs = expr.args[2]
    rhs = expr.args[3]
    return quote
        local _svars = Base.invokelatest(_build_symbolic_vars, $(esc(game)).players)
        local _env   = Base.invokelatest(_build_full_env,
            $(esc(game)).players, $(esc(game)).params)
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
            local _fmt   = Base.invokelatest(_format_constraint, _c, _rv, $(esc(game)).players)
            println("  [constraint $_lab]  $(_fmt)  players=$(_inv)")
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

function assign_params!(game::GameBuilder, assignments::Dict{Symbol, Array{Float64}})
    for (name, val) in assignments
        idx = findfirst(p -> p.name == name, game.params)
        if idx === nothing
            @warn "param :$name not declared — ignored"
            continue
        end
        pr = game.params[idx]
        @assert Tuple(size(val)) == pr.dims "param :$name expected dims=$(pr.dims), got $(Tuple(size(val)))"
        game.params[idx] = ParamRecord(pr.name, pr.dims, pr.symbolic_entries, val)
        println("  [param :$name]  numeric value assigned  dims=$(pr.dims)")
    end
end