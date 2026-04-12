# Validation.jl — pre-assembly checks and game characterization
 
# =============================================================================
# INTERNAL CHECKS — called by validate_game
# =============================================================================
 
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
    println("  [check 1]  registration  N=$N  n_total=$n_total")
end
 
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
        println("  [check 2]  player $(cost.player) cost is quadratic")
    end
end
 
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
            println("  [check 3]  group $glabel  shared  players=$first_set  rows=$(length(rows))")
        else
            println("  [check 3]  group $glabel  local   player=$first_set   rows=$(length(rows))")
        end
    end
end
 
# =============================================================================
# VALIDATE — main entry point
# =============================================================================
 
function validate_game(game::GameBuilder)
    println("=== Validating game ===")
    Base.invokelatest(_check_registration, game)
    Base.invokelatest(_check_linearity, game)
    Base.invokelatest(_check_shared_groups, game)
    println("=== Validation passed ===")
    println()
end
 
# =============================================================================
# GAME CHARACTERIZATION — call after assign_params!
# =============================================================================
 
# shared helper — converts H to Float64, errors if params unassigned
function _H_numeric(game::GameBuilder)
    H, _ = Base.invokelatest(assemble_operator, game)
    H_num = Matrix{Float64}(undef, size(H)...)
    for i in eachindex(H)
        v = Symbolics.value(H[i])
        v isa Number || error("Unassigned symbolic params — call assign_params! first")
        H_num[i] = Float64(v)
    end
    return H_num
end
 
function check_monotonicity(game::GameBuilder)
    H_num = Base.invokelatest(_H_numeric, game)
    eigs  = eigvals(Symmetric(H_num + H_num'))
    println("=== Monotonicity ===")
    println()
    println("  H + H' eigenvalues: $(round.(eigs, digits=4))")
    if all(eigs .> 1e-10)
        println("  strongly monotone")
    elseif all(eigs .>= -1e-10)
        println("  monotone (not strongly)")
    else
        println("  not monotone")
    end
    println()
end
 
function check_symmetry(game::GameBuilder)
    H_num = Base.invokelatest(_H_numeric, game)
    println("=== Symmetry ===")
    println()
    if isapprox(H_num, H_num', atol=1e-10)
        println("  H is symmetric  (potential game)")
    else
        println("  H is not symmetric  (not a potential game)")
        println("  ||H - H'|| = $(round(norm(H_num - H_num'), digits=6))")
    end
    println()
end
 
function check_theta_feasibility(game::GameBuilder)
    C, e = Base.invokelatest(assemble_theta_set, game)
    println("=== Theta Feasibility ===")
    println()
    if isempty(e)
        println("  no shared constraints — theta trivially feasible")
        println()
        return
    end
    n_theta = size(C, 2)
    lb = zeros(n_theta)
    ub = fill(Inf, n_theta)
    for i in 1:size(C, 1)
        row = C[i, :]
        nz  = findall(!=(0.0), row)
        length(nz) == 1 || continue
        col = nz[1]
        row[col] < 0 ? lb[col] = max(lb[col], -e[i]) : ub[col] = min(ub[col], e[i])
    end
    for i in 1:size(C, 1)
        row = C[i, :]
        nz  = findall(!=(0.0), row)
        length(nz) == 1 && continue
        for col in nz
            row[col] > 0 && (ub[col] = min(ub[col], e[i]))
        end
    end
    if all(lb .<= ub .+ 1e-10)
        println("  Theta set is non-empty")
        for i in 1:n_theta
            println("    θ_$i ∈ [$(round(lb[i], digits=4)), $(round(ub[i], digits=4))]")
        end
    else
        println("  Theta set is empty — check constraint RHS signs")
        for i in 1:n_theta
            tag = lb[i] > ub[i] + 1e-10 ? "  ← infeasible" : ""
            println("    θ_$i ∈ [$(round(lb[i], digits=4)), $(round(ub[i], digits=4))]$tag")
        end
    end
    println()
end
 
function show_coupling(game::GameBuilder)
    groups, _, shared_groups = Base.invokelatest(classify_constraints, game)
    H_num = Base.invokelatest(_H_numeric, game)
    println("=== Coupling Structure ===")
    println()
    print("  cost coupling:        ")
    pairs = String[]
    for p in game.players
        for pk in game.players
            pk.index <= p.index && continue
            coupled = any(
                !iszero(H_num[p.global_cols[j], pk.global_cols[k]]) ||
                !iszero(H_num[pk.global_cols[k], p.global_cols[j]])
                for j in 1:p.dim for k in 1:pk.dim)
            coupled && push!(pairs, "$(p.index)↔$(pk.index)")
        end
    end
    println(isempty(pairs) ? "none" : join(pairs, ", "))
    print("  constraint coupling:  ")
    svars = Base.invokelatest(_build_symbolic_vars, game.players)
    shared_strs = String[]
    for glabel in shared_groups
        rows = groups[glabel]
        coeffs, _ = Base.invokelatest(_extract_coefficients, rows[1].expression_lhs, svars)
        involved  = Base.invokelatest(_players_in_constraint, coeffs, game.players)
        push!(shared_strs, "{$(join(involved, ","))} $glabel")
    end
    println(isempty(shared_strs) ? "none" : join(shared_strs, ", "))
    println()
end