# static_mpGNE

Static GNE formulation layer for DyNECT. Converts N-player quadratic static games into a multi-parametric Variational Inequality (mpVI) for GNE selection via PWA map computation, based on the Nabetani-Tseng-Fukushima parametrization.

## Problem formulation

Given a static N-player quadratic game, each player $i$ solves

```math
\min_{x_i} \frac{1}{2} x_i^\top Q_i x_i + x_i^\top \sum_{j \neq i} R_{ij} x_j + c_i^\top x_i
```

subject to local constraints $A_i x_i \leq b_i$ and shared constraints $\sum_i A_i^s x_i \leq b^s$.

The game is reformulated as a parametric VI

```math
\text{find } x^* \text{ such that } F(x^*)^\top(x - x^*) \geq 0 \quad \forall x \in \mathcal{C}(\theta)
```

where $F(x) = Hx + f$ is the pseudogradient, and $\mathcal{C}(\theta) = \{Ax \leq B\theta + d\}$ is the reparametrized feasible set with $\theta \in \Theta$.

## Structure

| Path | Description |
|------|-------------|
| `compact/src/Fast_mpGNE.jl` | **Default.** Numeric params upfront, silent assembly, fast |
| `verbose/src/` | Symbolic pipeline, full display and game characterization |
| `Solver.jl` | DyNECT bridge: `solve_gne`, `show_solution`, `evaluate_gne` |
| `examples/` | Compact examples (default) and `verbose/` subfolder |

## Quick start

**Compact (default)** — parameters defined numerically upfront:

```julia
include("compact/src/Fast_mpGNE.jl")
using .Fast_mpGNE
include("Solver.jl")

Q1 = [2.0 0.0; 0.0 1.0];  Q2 = [3.0 0.0; 0.0 2.0]
R  = [1.0 0.0; 0.0 1.0];  c1 = [-1.0, 0.0];  c2 = [0.0, -1.0]

game = GameBuilder(N=2)
@player game 1 n=2;  @player game 2 n=2
@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2
@constraint game  -x1 <= 0;  @constraint game  -x2 <= 0
@constraint game  [x_1_1 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0]

mpvi = build_mpvi(game)   # prints "mpvi OK"
# show_mpvi(mpvi)         # uncomment to verify constraint system
sol  = solve_gne(mpvi)
show_solution(sol, mpvi)
```

**Verbose** — symbolic params, full characterization:

```julia
include("verbose/src/Static_mpGNE.jl")
using .Static_mpGNE
include("Solver.jl")

game = GameBuilder(N=2)
@player game 1 n=2;  @player game 2 n=2
@param game Q1 dims=(2,2);  @param game R dims=(2,2);  @param game c1 dims=(2,)
@cost game 1  0.5*x1'*Q1*x1 + x1'*R*x2 + c1'*x1
@cost game 2  0.5*x2'*Q2*x2 + x2'*R'*x1 + c2'*x2
@constraint game  -x1 <= 0;  @constraint game  -x2 <= 0
@constraint game  [x_1_1 + 2*x_2_1, x_1_2 + x_2_2] <= [3.0, 2.0]

assign_params!(game, Dict(:Q1 => [2.0 0.0; 0.0 1.0], :R => [1.0 0.0; 0.0 1.0], :c1 => [-1.0, 0.0]))

validate_game(game)
show_coupling(game);  check_monotonicity(game);  check_symmetry(game)
show_operator(game);  show_parametric_constraints(game);  show_theta_set(game)

mpvi = build_mpvi(game)
sol  = solve_gne(mpvi)
show_solution(sol, mpvi)
```

## Dependencies

Inherits DyNECT dependencies. Additionally requires:
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) (verbose version only)

**Authors:** Stephanie Matta — stmb@kth.se  
**Supervisors:** Emilio Benenati, Dániel Tihanyi.
