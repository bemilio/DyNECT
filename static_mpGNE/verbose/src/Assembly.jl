# Assembly.jl — orchestrates the full GNE → mpVI pipeline

using LinearAlgebra

function build_mpvi(game::GameBuilder;
                    params::Dict{Symbol, Array{Float64}} = Dict{Symbol, Array{Float64}}())

    # assign numeric params if provided
    !isempty(params) && Base.invokelatest(assign_params!, game, params)

    H, f    = Base.invokelatest(assemble_operator, game)
    n_total = sum(p.dim for p in game.players)
    A, B, d, n_theta = Base.invokelatest(assemble_feasible_set, game)
    C, e    = Base.invokelatest(assemble_theta_set, game)
    Ftheta  = zeros(n_total, max(n_theta, 0))

    groups, local_groups, shared_groups = Base.invokelatest(classify_constraints, game)

    # local row ranges
    local_row_ranges = UnitRange{Int}[]
    current_row = 1
    for glabel in local_groups
        n = length(groups[glabel])
        push!(local_row_ranges, current_row:(current_row + n - 1))
        current_row += n
    end

    # shared row ranges and theta ranges
    shared_row_ranges = UnitRange{Int}[]
    theta_ranges      = UnitRange{Int}[]
    current_theta     = 1
    for glabel in shared_groups
        A_blocks, b, involved = Base.invokelatest(
            extract_constraint_blocks, groups[glabel], game)
        k         = length(involved)
        m         = length(b)
        n_theta_g = (k - 1) * m
        n_rows_g  = k * m
        push!(shared_row_ranges, current_row:(current_row + n_rows_g - 1))
        push!(theta_ranges, current_theta:(current_theta + n_theta_g - 1))
        current_row   += n_rows_g
        current_theta += n_theta_g
    end

    return MPVIAssembly(
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
end

function show_mpvi(mpvi::MPVIAssembly)
    println("=== mpVI Object Dimensions ===")
    println()
    println("  H:      $(size(mpvi.H,1)) × $(size(mpvi.H,2))")
    println("  Ftheta: $(size(mpvi.Ftheta,1)) × $(size(mpvi.Ftheta,2))  (zero)")
    println("  f:      $(length(mpvi.f))")
    println("  A:      $(size(mpvi.A,1)) × $(size(mpvi.A,2))")
    println("  B:      $(size(mpvi.B,1)) × $(size(mpvi.B,2))")
    println("  d:      $(length(mpvi.d))")
    println("  C:      $(size(mpvi.C,1)) × $(size(mpvi.C,2))")
    println("  e:      $(length(mpvi.e))")
    println()
    println("  n_total = $(size(mpvi.H,1))  n_theta = $(size(mpvi.B,2))")
    println()
end

#
function materialize(mpvi::MPVIAssembly)
    function to_f64_matrix(M::Matrix)
        out = Matrix{Float64}(undef, size(M)...)
        for i in eachindex(M)
            v = Symbolics.value(M[i])
            v isa Number || error("Symbolic entry remains at index $i — unassigned parameter")
            out[i] = Float64(v)
        end
        return out
    end

    function to_f64_vector(v::Vector)
        out = Vector{Float64}(undef, length(v))
        for i in eachindex(v)
            val = Symbolics.value(v[i])
            val isa Number || error("Symbolic entry remains at index $i — unassigned parameter")
            out[i] = Float64(val)
        end
        return out
    end

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