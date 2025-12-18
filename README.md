<p align="center">
  <img src="assets/logo.png" alt="DyNECT logo" width="180"/>
</p>

[![License](https://img.shields.io/github/license/bemilio/DyNECT.svg)](https://github.com/bemilio/DyNECT/blob/main/LICENSE)  


**DyNECT** stands for _Dynamic Nash Equilibrium Control Toolbox_ - a Julia package for modeling and solving dynamic Nash equilibrium control problems

## Features

- Collection of state-of-the-art iterative and explicit solvers for open-loop dynamic games  
- Utility tools for streamlined implementation of game-theoretic MPC
- Automatic reformulation of LQ games as (multi-parametric) Variational Inequalities (VIs)
- Iterative solution of coupled Riccati equations arising in unconstrained infinite-horizon games  
- Integration with [Monviso](https://github.com/nicomignoni/Monviso.jl) for access to multiple VI solvers
- Integration with [pDAQP](https://github.com/darnstrom/ParametricDAQP.jl/pull/19) for offline computation of state-to-solution map, enabling fast online control  control
- Unified solver access through the  [CommonSolve.jl](https://github.com/SciML/CommonSolve.jl) interface

## Installation

DyNECT depends on a development fork of 'ParametricDAQP.jl'. From the Julia REPL:

```sh
] add https://github.com/bemilio/ParametricDAQP.jl#mpVI
] add https://github.com/bemilio/DyNECT.git
```

## Examples

The example in `examples/solve_LQGame_as_VI.jl`:
- Implements a basic LQ game
- Converts it into a multi-parametric VI
- Finds explicitely the explicit solution mapping for all initial states
- Given an initial state, converts the multi-parametric VI into a VI
- Solves the VI via several iterative algorithms

Additional examples can be found at [this link](https://github.com/bemilio/scripts_for_explicit_LQGames_paper/tree/main/examples). 

For details on the LQ game-to-VI conversion, as well as a performance comparison between some implemented solvers, see

[_The explicit game-theoretic linear quadratic
regulator for constrained multi-agent systems_ E. Benenati, G. Belgioioso, 2025](https://arxiv.org/pdf/2512.07749)