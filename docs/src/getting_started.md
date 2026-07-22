# Getting Started

## Installation

DyNECT depends on `ParametricDAQP.jl` and `Monviso.jl`, which are not in the General
registry. From the Julia REPL:

```sh
] add https://github.com/nicomignoni/Monviso.jl.git#vi-based
] add https://github.com/darnstrom/ParametricDAQP.jl
] add https://github.com/bemilio/DyNECT.git
```

## Examples

Example scripts are in `examples/` and use their own Julia environment. Run them with:

```sh
julia --project=examples/ examples/LQ_dyn_game.jl
```

To install example dependencies once:

```sh
julia --project=examples/ -e 'using Pkg; Pkg.instantiate()'
```

Additional examples can be found at
[this link](https://github.com/bemilio/scripts_for_explicit_LQGames_paper).
