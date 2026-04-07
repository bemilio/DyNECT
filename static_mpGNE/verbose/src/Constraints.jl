# Constraints.jl — classifies, reparametrizes and assembles the parametric feasible set
# produces: A, B, d (feasible set) and C, e (theta admissibility)

function classify_constraints(game::GameBuilder)
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

function extract_constraint_blocks(rows::Vector{ConstraintRecord}, game::GameBuilder)
    svars = Base.invokelatest(_build_symbolic_vars, game.players)
    m     = length(rows)

    expr_first       = Base.invokelatest(_substitute_params, rows[1].expression_lhs, game.params)
    coeffs_first, _  = Base.invokelatest(_extract_coefficients, expr_first, svars)
    players_involved = Base.invokelatest(_players_in_constraint, coeffs_first, game.players)

    A_blocks = Vector{Matrix{MatrixEntry}}()
    for pi in players_involved
        p = game.players[findfirst(pl -> pl.index == pi, game.players)]
        push!(A_blocks, Matrix{MatrixEntry}(zeros(Float64, m, p.dim)))
    end
    b = zeros(Float64, m)

    for (row_idx, row) in enumerate(rows)
        expr   = Base.invokelatest(_substitute_params, row.expression_lhs, game.params)
        coeffs, _ = Base.invokelatest(_extract_coefficients, expr, svars)
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

# player i gets theta_i, last player gets b - sum of all previous thetas
function assemble_shared_reparametrization(A_blocks::Vector{Matrix{MatrixEntry}},
                                            b::Vector{Float64},
                                            players_involved::Vector{Int},
                                            game::GameBuilder)
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

function assemble_local_constraints(game::GameBuilder)
    groups, local_groups, _ = Base.invokelatest(classify_constraints, game)
    n_total = sum(p.dim for p in game.players)
    svars   = Base.invokelatest(_build_symbolic_vars, game.players)
    A_rows  = Vector{Vector{Float64}}()
    d_vals  = Float64[]

    for glabel in local_groups
        for row in groups[glabel]
            expr   = Base.invokelatest(_substitute_params, row.expression_lhs, game.params)
            coeffs, _ = Base.invokelatest(_extract_coefficients, expr, svars)
            A_row  = zeros(Float64, n_total)
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

function assemble_feasible_set(game::GameBuilder)
    groups, local_groups, shared_groups = Base.invokelatest(classify_constraints, game)
    n_total = sum(p.dim for p in game.players)

    A_local, d_local = Base.invokelatest(assemble_local_constraints, game)
    n_local_rows     = size(A_local, 1)

    A_shared_rows = Vector{Vector{MatrixEntry}}()
    B_shared_rows = Vector{Vector{Float64}}()
    d_shared_vals = Float64[]
    n_theta       = 0

    for glabel in shared_groups
        A_blocks, b, involved = Base.invokelatest(
            extract_constraint_blocks, groups[glabel], game)
        A_hat, B_g, d_g, n_theta_g = Base.invokelatest(
            assemble_shared_reparametrization, A_blocks, b, involved, game)
        for i in 1:size(A_hat, 1)
            push!(A_shared_rows, vec(A_hat[i:i, :]))
        end
        for i in 1:size(B_g, 1)
            push!(B_shared_rows, B_g[i, :])
        end
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
        col_offset = 0
        row_offset = n_local_rows
        for glabel in shared_groups
            A_blocks, b, involved = Base.invokelatest(
                extract_constraint_blocks, groups[glabel], game)
            _, B_g, _, n_theta_g = Base.invokelatest(
                assemble_shared_reparametrization, A_blocks, b, involved, game)
            n_g_rows = size(B_g, 1)
            B[row_offset+1:row_offset+n_g_rows, col_offset+1:col_offset+n_theta_g] .= B_g
            row_offset += n_g_rows
            col_offset += n_theta_g
        end
    end

    return A, B, vcat(d_local, d_shared_vals), n_theta
end

function assemble_theta_set(game::GameBuilder)
    groups, _, shared_groups = Base.invokelatest(classify_constraints, game)
    n_theta    = 0
    group_info = Tuple{Int,Int,Vector{Float64}}[]

    for glabel in shared_groups
        A_blocks, b, involved = Base.invokelatest(
            extract_constraint_blocks, groups[glabel], game)
        k = length(involved)
        m = length(b)
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
            # determine which resource dimension this theta component belongs to
            r = mod1(i, m)
            if b[r] >= 0
                # θ ≥ 0: encode as -θ ≤ 0
                row[col_offset + i] = -1.0
                push!(C_rows, row)
                push!(e_vals, 0.0)
            else
                # θ ≤ 0: encode as θ ≤ 0
                row[col_offset + i] = 1.0
                push!(C_rows, row)
                push!(e_vals, 0.0)
            end
        end

        # conservation: sum of shares bounded by b per resource dimension
        for r in 1:m
            row = zeros(Float64, n_theta)
            if b[r] >= 0
                # θ₁ + ... + θ_{k-1} ≤ b[r]
                for prev_idx in 1:(k-1)
                    row[col_offset + (prev_idx-1)*m + r] = 1.0
                end
                push!(C_rows, row)
                push!(e_vals, b[r])
            else
                # θ₁ + ... + θ_{k-1} ≥ b[r]  →  -(θ₁+...+θ_{k-1}) ≤ -b[r]
                for prev_idx in 1:(k-1)
                    row[col_offset + (prev_idx-1)*m + r] = -1.0
                end
                push!(C_rows, row)
                push!(e_vals, -b[r])
            end
        end

        col_offset += n_theta_g
    end

    return reduce(vcat, [r' for r in C_rows]), e_vals
end

function show_shared_constraints(game::GameBuilder)
    groups, _, shared_groups = Base.invokelatest(classify_constraints, game)
    println("=== Shared Constraint Blocks ===")
    println()
    if isempty(shared_groups)
        println("  no shared constraints")
        println()
        return
    end
    for glabel in shared_groups
        rows = groups[glabel]
        A_blocks, b, involved = Base.invokelatest(extract_constraint_blocks, rows, game)
        A_hat, B_g, d_g, n_theta_g = Base.invokelatest(
            assemble_shared_reparametrization, A_blocks, b, involved, game)
        println("  group=$glabel  players=$involved  m=$(length(b))  n_theta=$n_theta_g")
        println()
        for (k, pi) in enumerate(involved)
            println("  A_$pi ($(size(A_blocks[k],1)) × $(size(A_blocks[k],2))):")
            for i in 1:size(A_blocks[k], 1)
                println("    row $i:  [ $(join([string(A_blocks[k][i,j]) for j in 1:size(A_blocks[k],2)], "  ")) ]")
            end
        end
        println("  b = $b")
        println()
        println("  --- Reparametrized ---")
        println("  A_hat ($(size(A_hat,1)) × $(size(A_hat,2))):")
        for i in 1:size(A_hat, 1)
            println("    row $i:  [ $(join([string(A_hat[i,j]) for j in 1:size(A_hat,2)], "  ")) ]")
        end
        println()
        println("  B_g ($(size(B_g,1)) × $(size(B_g,2))):")
        for i in 1:size(B_g, 1)
            println("    row $i:  [ $(join([string(B_g[i,j]) for j in 1:size(B_g,2)], "  ")) ]")
        end
        println()
        println("  d_g ($(length(d_g)) × 1):")
        for i in 1:length(d_g)
            println("    row $i:  [ $(d_g[i]) ]")
        end
        println()
    end
end

function show_local_constraints(game::GameBuilder)
    A_local, d_local = Base.invokelatest(assemble_local_constraints, game)
    println("=== Local Constraints:  A_local * x <= d_local ===")
    println()
    if size(A_local, 1) == 0
        println("  no local constraints")
        println()
        return
    end
    println("A_local ($(size(A_local,1)) × $(size(A_local,2))):")
    for i in 1:size(A_local, 1)
        println("  row $i:  [ $(join([string(A_local[i,j]) for j in 1:size(A_local,2)], "  ")) ]  <= $(d_local[i])")
    end
    println()
end

function show_feasible_set(game::GameBuilder)
    A, B, d, n_theta = Base.invokelatest(assemble_feasible_set, game)
    n_rows, n_cols   = size(A)
    println("=== Parametric Feasible Set:  A*x <= B*theta + d ===")
    println()
    println("  n_theta = $n_theta")
    println()
    println("A ($n_rows × $n_cols):")
    for i in 1:n_rows
        println("  row $i:  [ $(join([string(A[i,j]) for j in 1:n_cols], "  ")) ]")
    end
    println()
    println("B ($n_rows × $(size(B,2))):")
    for i in 1:n_rows
        println("  row $i:  [ $(join([string(B[i,j]) for j in 1:size(B,2)], "  ")) ]")
    end
    println()
    println("d ($n_rows × 1):")
    for i in 1:n_rows
        println("  row $i:  [ $(d[i]) ]")
    end
    println()
end

function show_theta_set(game::GameBuilder)
    C, e = Base.invokelatest(assemble_theta_set, game)
    println("=== Admissible Parameter Set:  Θ = {θ : C*θ <= e} ===")
    println()
    if isempty(e)
        println("  no shared constraints — theta is empty")
        println()
        return
    end
    println("C ($(size(C,1)) × $(size(C,2))):")
    for i in 1:size(C,1)
        println("  row $i:  [ $(join([string(C[i,j]) for j in 1:size(C,2)], "  ")) ]  <= $(e[i])")
    end
    println()
end

function show_parametric_constraints(game::GameBuilder)
    A, B, d, n_theta = Base.invokelatest(assemble_feasible_set, game)
    players = game.players
    println("=== Stacked Constraint System:  A*x <= B*theta + d ===")
    println()
    for i in 1:size(A, 1)
        lhs_terms = String[]
        for p in players
            for j in 1:p.dim
                val  = A[i, p.global_cols[j]]
                valv = val isa Number ? val : Symbolics.value(val)
                if !(valv isa Number && iszero(valv))
                    push!(lhs_terms, "[$(val)]$(_display_var(p.index, j, p.dim))")
                end
            end
        end
        rhs_terms = String[]
        !iszero(d[i]) && push!(rhs_terms, "[$(d[i])]")
        for t in 1:n_theta
            !iszero(B[i,t]) && push!(rhs_terms, "[$(B[i,t])]θ_$(t)")
        end
        println("  row $i:  $(isempty(lhs_terms) ? "0" : join(lhs_terms, " + "))  <=  $(isempty(rhs_terms) ? "[0]" : join(rhs_terms, " + "))")
    end
    println()
end