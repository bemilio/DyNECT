# static_mpGNE
Static game formulation layer for the GNE → mpVI pipeline, built on top of DyNECT 
and ParametricDAQP.jl. Converts N-player quadratic static games into a 
multi-parametric Variational Inequality (mpVI) ready for GNE selection.

**Authors:** Stephanie Matta — stmb@kth.se  
**Supervisors:** Emilio Benenati, Daniel T.

## Structure

### `verbose/` (End-to-end working version)
Symbolic step-by-step game characterization via Symbolics.jl.
Designed for building, understanding, and debugging a game definition.
Includes pretty printing, constraint inspection, and validation tools.
Supports scalar and vector quadratic players.
Current files: `Types.jl`, `Static_mpGNE.jl`, `Interface.jl`

### `compact/` (TBD for speed performance)
Fast numeric version for offline iterative computation and analysis.
Matrix-in → mpVI-out, no symbolic overhead, built for repeated runs
and integration with DyNECT workflows.

### `examples/`
Shared example scripts. Set the `VERSION` flag at the top of each
script to switch between verbose and compact.

### `Solver.jl`
Shared DyNECT bridge used by both versions.

## Dependencies
- [DyNECT](https://github.com/bemilio/DyNECT)
- [ParametricDAQP.jl](https://github.com/darnstrom/ParametricDAQP.jl)
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)