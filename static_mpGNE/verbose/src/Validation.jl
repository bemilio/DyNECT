# Validation.jl — pre-assembly checks on GameBuilder
# check 1: registration, check 2: quadraticity, check 3: group consistency

function validate_game(game::GameBuilder)
    println("=== Validating game ===")
    Base.invokelatest(_check_registration, game)
    Base.invokelatest(_check_linearity, game)
    Base.invokelatest(_check_shared_groups, game)
    println("=== Validation passed ===")
    println()
end

function _check_registration(game::GameBuilder)
    N       = game.N
    n_total = sum(p.dim for p in game.players)

    @assert length(game.players) == N "Expected $N players, got $(length(game.players))"
    @assert sort([p.index for p in game.players]) == collect(1:N) "Player indices must be 1:$N"

    for p in game.players
        n_costs = count(c -> c.player == p.index, game.costs)
        @assert n_costs == 1 "Player $(p.index) must have exactly 1 cost, got $n_costs"
    end

    all_cols = reduce(vcat, [collect(p.global_cols) for p in game.players])
    @assert sort(all_cols) == collect(1:n_total) "Column ranges must cover 1:$n_total without gaps"

    println("  [✓ check 1]  registration  N=$N  n_total=$n_total")
end

# third derivative check — costs must be at most quadratic
function _check_linearity(game::GameBuilder)
    svars = Base.invokelatest(_build_symbolic_vars, game.players)

    for cost in game.costs
        p = game.players[findfirst(pl -> pl.index == cost.player, game.players)]
        for j in 1:p.dim
            grad = Symbolics.simplify(Symbolics.derivative(
                cost.expression, svars[Symbol(:x_, p.index, :_, j)]))
            for (_, var) in svars
                second = Symbolics.simplify(Symbolics.derivative(grad, var))
                if Symbolics.value(second) isa Num
                    for (sym2, var2) in svars
                        third = Symbolics.simplify(Symbolics.derivative(second, var2))
                        v3 = Symbolics.value(third)
                        @assert !(v3 isa Num) && iszero(v3) "Player $(cost.player) cost not quadratic: nonlinear term in $sym2"
                    end
                end
            end
        end
        println("  [✓ check 2]  player $(cost.player) cost is quadratic")
    end
end

# all rows in a group must involve the same player set
function _check_shared_groups(game::GameBuilder)
    svars  = Base.invokelatest(_build_symbolic_vars, game.players)
    groups = Base.invokelatest(group_constraints, game.constraints)

    for (glabel, rows) in sort(collect(groups), by = x -> x[1])
        player_sets = map(rows) do row
            coeffs, _ = Base.invokelatest(_extract_coefficients, row.expression_lhs, svars)
            Base.invokelatest(_players_in_constraint, coeffs, game.players)
        end

        first_set = player_sets[1]
        for (i, pset) in enumerate(player_sets)
            @assert pset == first_set "Group $glabel: row $i involves $pset but row 1 involves $first_set"
        end

        if length(first_set) > 1
            println("  [✓ check 3]  group $glabel  shared  players=$first_set  rows=$(length(rows))")
        else
            println("  [✓ check 3]  group $glabel  local   player=$first_set   rows=$(length(rows))")
        end
    end
end