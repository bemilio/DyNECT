# Operator.jl — assembles VI operator F(x) = H*x + f from quadratic costs

function assemble_operator(game::GameBuilder)
    n_total = sum(p.dim for p in game.players)
    svars   = Base.invokelatest(_build_symbolic_vars, game.players)
    H = Matrix{MatrixEntry}(undef, n_total, n_total)
    f = Vector{MatrixEntry}(undef, n_total)

    for cost in game.costs
        pi   = game.players[findfirst(p -> p.index == cost.player, game.players)]
        expr = Base.invokelatest(_substitute_params, cost.expression, game.params)

        for j in 1:pi.dim
            row  = pi.global_cols[j]
            grad = Symbolics.simplify(Symbolics.derivative(expr, svars[Symbol(:x_, pi.index, :_, j)]))

            for pk in game.players
                for k in 1:pk.dim
                    coeff = Symbolics.simplify(Symbolics.derivative(grad, svars[Symbol(:x_, pk.index, :_, k)]))
                    val   = Symbolics.value(coeff)
                    H[row, pk.global_cols[k]] = val isa Number ? Float64(val) : coeff
                end
            end

            zero_sub  = Dict(var => 0.0 for (_, var) in svars)
            const_val = Symbolics.value(Symbolics.simplify(Symbolics.substitute(grad, zero_sub)))
            f[row]    = const_val isa Number ? Float64(const_val) : const_val
        end
    end

    return H, f
end

function show_operator(game::GameBuilder)
    H, f    = Base.invokelatest(assemble_operator, game)
    players = game.players
    n_total = sum(p.dim for p in players)

    println("=== VI Operator:  F(x) = H*x + f ===")
    println()

    println("H ($n_total × $n_total):")
    for p in players
        for j in 1:p.dim
            row   = p.global_cols[j]
            dvar  = _display_var(p.index, j, p.dim)
            entries = [string(H[row, c]) for c in 1:n_total]
            println("$(rpad("  ∂J_$(p.index)/∂$(dvar)", 28))  [ $(join(entries, "  ")) ]")
        end
    end
    println()

    println("f ($n_total):")
    for p in players
        for j in 1:p.dim
            row  = p.global_cols[j]
            dvar = _display_var(p.index, j, p.dim)
            println("  f[$row] = $(f[row])   (∂J_$(p.index)/∂$(dvar) constant)")
        end
    end
    println()

    println("Explicit pseudogradient equations:")
    for p in players
        for j in 1:p.dim
            row   = p.global_cols[j]
            terms = String[]
            for pk in players
                for k in 1:pk.dim
                    h  = H[row, pk.global_cols[k]]
                    hv = h isa Number ? h : Symbolics.value(h)
                    if !(hv isa Number && iszero(hv))
                        push!(terms, "($(h))$(_display_var(pk.index, k, pk.dim))")
                    end
                end
            end
            fv = f[row]
            !(fv isa Number && iszero(fv)) && push!(terms, "($(fv))")
            println("  F_$(p.index)_$(j)(x) = $(isempty(terms) ? "0" : join(terms, " + "))")
        end
    end
    println()
end