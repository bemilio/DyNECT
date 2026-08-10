```@meta
CurrentModule = DyNECT
```

# From `DynLQGame` to `LQGNEP`

This page derives, step by step, the algebra implemented in [`DynLQGame2LQGNEP`](@ref)
(`src/utils.jl`): how a dynamic LQ game (a [`DynLQGame`](@ref), or its time-varying
counterpart [`DynLQGameTV`](@ref)) over a finite horizon is reformulated into a *static*
[`LQGNEP`](@ref) over the stacked input sequence, by eliminating the state via a prediction
model. Variable names below match the code (`Θ`, `Γ`, `c̄`, `Q̅`, `C̅ₓ`, ...) so this page can
be read side by side with `src/utils.jl`.

## 1. Setup

A [`DynLQGame`](@ref) with `N` agents is described, for `i = 1,...,N`, by the dynamics

```math
x[t+1] = A x[t] + \sum_{j=1}^N B_j u_j[t] + c,
```

the cost

```math
J_i = \tfrac12 x[T]^\top P_i x[T] + p_i^\top x[T] + \sum_{t=0}^{T-1}\Big(
\tfrac12 x[t]^\top Q_i x[t] + \tfrac12 u_i[t]^\top R_{ii} u_i[t]
+ \sum_{j\neq i} u_i[t]^\top R_{ij} u_j[t] + q_i^\top x[t] + r_i^\top u_i[t] \Big),
```

and the constraints

```math
C_x x[t] \leq b_x, \qquad C^{\mathrm{loc}}_i u_i[t] \leq b^{\mathrm{loc}}_i, \qquad
\sum_j C^{\mathrm u}_j u_j[t] \leq b^{\mathrm u} \qquad \forall t = 0,\dots,T-1.
```

`T` is the prediction horizon (`T_hor` in the code). The goal is a static reformulation in
the stacked input sequences ``\bar u_i = \mathrm{col}(u_i[0],\dots,u_i[T-1])``, parametric in
the initial state ``x_0`` (which plays the role of the [`LQGNEP`](@ref)'s external parameter
``\gamma``).

## 2. Eliminating the state: the prediction model

[`generate_prediction_model`](@ref) unrolls the dynamics into

```math
\bar x = \Theta x_0 + \sum_{j=1}^N \Gamma_j \bar u_j + \bar c, \qquad
\bar x = \mathrm{col}(x[1],\dots,x[T]),
```

with ``\Theta = \mathrm{col}(A,A^2,\dots,A^T)``, ``\Gamma_j`` the block lower-triangular
Toeplitz matrix of ``B_j, AB_j, A^2B_j,\dots``, and ``\bar c`` the affine part induced by
``c``. This is the only place the dynamics enter — every quantity below is obtained by
substituting this expression for ``\bar x`` wherever it appears, so the state itself never
appears in the resulting static game.

## 3. Quadratic cost in the stacked input

Stack agent `i`'s running and terminal state cost into a single quadratic form. Define

```math
\bar Q_i = \mathrm{blkdiag}(\underbrace{Q_i,\dots,Q_i}_{T-1},\, P_i)
\quad\text{(code: } \bar Q_i \text{ = kron}(I_{T-1}, Q_i) \text{ block-diagonal with } P_i\text{)},
```

so that ``\sum_{t=0}^{T-1}\tfrac12 x[t]^\top Q_i x[t] + \tfrac12 x[T]^\top P_i x[T]
= \tfrac12 \bar x^\top \bar Q_i \bar x``. Substituting ``\bar x = \Theta x_0
+ \sum_j\Gamma_j\bar u_j + \bar c`` and expanding, the terms involving ``\bar u`` are:

```math
\tfrac12 \bar x^\top \bar Q_i \bar x = \tfrac12 \sum_{j,k} \bar u_j^\top \Gamma_j^\top \bar Q_i
\Gamma_k \bar u_k + \Big(\sum_j \bar u_j^\top \Gamma_j^\top \bar Q_i\Big)(\Theta x_0 + \bar c)
+ \text{(terms independent of } \bar u\text{)}.
```

Together with the running input cost, this gives the [`LQGNEP`](@ref) quadratic/linear cost
data for agent `i`:

```math
Q_{ij} = \Gamma_i^\top \bar Q_i \Gamma_j + I_T\otimes R_{ij}, \qquad
q_i = \Gamma_i^\top(\bar Q_i \bar c + \bar q_i) + \bar r_i, \qquad
Q_{q_i\gamma} = \Gamma_i^\top \bar Q_i \Theta,
```

where ``\bar q_i = \mathrm{col}(\underbrace{q_i,\dots,q_i}_{T-1},\,p_i)`` and
``\bar r_i = \mathrm{col}(r_i,\dots,r_i)`` (`T` copies). ``Q_{q_i\gamma}`` is exactly the
sensitivity field `Q_qγ[i]` of [`LQGNEP`](@ref): the ``x_0``-dependent part of the linear
cost that [`LQGNEP`](@ref) keeps separate as ``q_i(\gamma) = q_i + Q_{q_i\gamma}\gamma``. In
code (`DynLQGame2LQGNEP`, one iteration `i` of the main loop):

```julia
QiΓ = Γi[i]' * Q̅[i]
Qi  = [QiΓ * Γi[j] + kron(I(T_hor), prob.R[i][j]) for j in 1:prob.N]
q̅i  = vcat(kron(ones(T_hor - 1), prob.q[i]), prob.p[i])
r̅i  = kron(ones(T_hor), prob.r[i])
qi  = Γi[i]' * (Q̅[i] * c̅ + q̅i) + r̅i
Qqγ_i = QiΓ * Θ
```

## 4. Local input constraints

``C^{\mathrm{loc}}_i u_i[t] \leq b^{\mathrm{loc}}_i`` for every ``t`` stacks directly, with no
elimination needed (it never involves the state):

```math
A_{\mathrm{loc},i} = I_T \otimes C^{\mathrm{loc}}_i, \qquad
b_{\mathrm{loc},i} = \mathbf 1_T \otimes b^{\mathrm{loc}}_i.
```

## 5. Shared input constraints

Likewise, ``\sum_j C^{\mathrm u}_j u_j[t] \leq b^{\mathrm u}`` stacks to

```math
A^{\mathrm{sh}}_{\mathrm{input},j} = I_T \otimes C^{\mathrm u}_j, \qquad
b_{\mathrm{input}} = \mathbf 1_T \otimes b^{\mathrm u},
```

with no ``x_0``-dependence (`B_input_γ = 0`).

## 6. State constraints: elimination turns them into shared constraints

``C_x x[t] \leq b_x`` for every ``t`` stacks to ``\bar C_x \bar x \leq \mathbf 1_T\otimes b_x``
with ``\bar C_x = I_T \otimes C_x``. Substituting the prediction model for ``\bar x``:

```math
\bar C_x \Big(\Theta x_0 + \sum_j \Gamma_j \bar u_j + \bar c\Big) \leq \mathbf 1_T \otimes b_x
\quad\Longleftrightarrow\quad
\sum_j \big(\bar C_x \Gamma_j\big) \bar u_j \leq \underbrace{\mathbf 1_T\otimes b_x - \bar C_x\bar c}_{b_{\mathrm{state}}}
\underbrace{- \bar C_x \Theta}_{B_{\mathrm{state},\gamma}} x_0.
```

This is why state constraints — despite being local to the state, not the input — become
*shared* constraints once the state is eliminated: every agent's stacked input ``\bar u_j``
appears through its own contribution ``A^{\mathrm{sh}}_{\mathrm{state},j} = \bar C_x\Gamma_j``
to the same rows. In code:

```julia
A_sh_state = [C̅_x * Γi[i] for i in 1:prob.N]
b_state    = kron(ones(T_hor), prob.b_x) - C̅_x * c̅
B_state_γ  = -C̅_x * Θ
```

## 7. Assembling the `LQGNEP`

The hard shared constraint for agent `i` stacks the (state-independent) input block on top of
the (state-derived) state block:

```math
A_{\mathrm{sh},i} = \begin{bmatrix} A^{\mathrm{sh}}_{\mathrm{input},i} \\ A^{\mathrm{sh}}_{\mathrm{state},i} \end{bmatrix},
\qquad
b_{\mathrm{sh}} = \begin{bmatrix} b_{\mathrm{input}} \\ b_{\mathrm{state}} \end{bmatrix},
\qquad
B_{\mathrm{sh},\gamma} = \begin{bmatrix} 0 \\ B_{\mathrm{state},\gamma} \end{bmatrix},
```

(`A_sh[i] = vcat(A_sh_input[i], A_sh_state[i])` in code), which together with `Q`, `q`,
`A_loc`, `b_loc`, `Q_qγ`, `B_loc_γ` from Sections 3–4 fully determines the [`LQGNEP`](@ref)
returned by [`DynLQGame2LQGNEP`](@ref). Passing this game to [`mpAVI`](@ref) then produces the
same parametric VI that [`DynLQGame2mpAVI`](@ref) builds directly.

## 8. Time-varying games (`DynLQGameTV`)

Everything above carries over unchanged for [`DynLQGameTV`](@ref), replacing every
time-invariant matrix by its per-timestep counterpart and every Kronecker product by a block
diagonal stack of the (now different) per-timestep blocks:

```math
\bar Q_i = \mathrm{blkdiag}(Q^1_i,\dots,Q^{T-1}_i,\,P_i), \qquad
\bar C_x = \mathrm{blkdiag}(C_x^1,\dots,C_x^T),
```

and ``\Gamma_i, \Theta, \bar c`` come from the time-varying prediction model (second method of
[`generate_prediction_model`](@ref)). The rest of the derivation — cost substitution, local
and shared constraint stacking, state-constraint elimination — is identical.

## 9. Optional: softening the state constraints

[`DynLQGame2LQGNEP`](@ref) can optionally soften the state constraint per agent, per
constraint row, and per timestep (`soften_state_constraints=true`). Instead of the single
hard copy of Section 6, agent `i` gets its own relaxed copy

```math
C_x x[t] \leq b_x + s_{i,l,t}, \qquad s_{i,l,t} \geq 0, \qquad l = 1,\dots,m_x,\ t=1,\dots,T,
```

with a linear penalty ``k_i[l]\, s_{i,l,t}`` added to agent ``i``'s cost (same per-row cost at
every timestep). Writing ``s_i = \mathrm{col}(s_{i,1,1},\dots,s_{i,m_x,T}) \in
\mathbb R^{m_xT}`` and appending it to agent ``i``'s decision vector
``x_i = [\bar u_i;\, s_i]``, the state-constraint block of Section 6 becomes, for every agent
``i``'s own copy:

```math
\bar C_x \Gamma_i \bar u_i - I_{m_xT}\, s_i + \sum_{j\neq i} \bar C_x\Gamma_j \bar u_j
\leq b_{\mathrm{state}} - B_{\mathrm{state},\gamma}\, x_0,
```

i.e. one shared constraint *per agent* (``N`` relaxed copies of the same physical
constraint), each depending on ``s_i`` only through its own copy — every other copy gets a
zero coefficient on ``s_i``. Since the slack has no quadratic cost, it is appended to `Q` and
`A_loc` (via `≥0`) with only zero/`-I` blocks, and to `q` via the tiled cost
`kron(ones(T_hor), k[i])`. See [`DynLQGame2LQGNEP`](@ref)'s docstring and the
`soften_state_constraints` branch of its implementation for the exact matrices.

## 10. Worked example: the `SoftenedStateConstraints` test

This section derives, by hand, the closed-form solutions checked by the
`SoftenedStateConstraints` testset (`test/tests.jl`), as an end-to-end sanity check of
Sections 3, 6 and 9 above.

### Setup

``N=2`` agents, scalar state, single step (``T=1``, so ``\bar u_i = u_i``):

```math
x^1 = u_1 + u_2 \quad (A=1,\ B_i=1,\ c=0), \qquad
J_i = \tfrac12 P (x^1)^2 + p\,x^1 + \tfrac12 R\, u_i^2, \quad P=R=1,\ p=-2,
```

subject to the hard state constraint ``x^1 \leq 1`` (``C_x=1``, ``b_x=1``). There is no
running state cost (``Q_i=0``) or running/terminal linear input cost (``r_i=0``), so
``\bar Q_i = P = 1`` and ``\Gamma_i = B_i = 1``.

DyNECT solves for the *variational* equilibrium: the single shared multiplier ``\lambda``
associated with the AVI feasible set ``\mathcal C = \{A_{\mathrm{sh}} u \leq b_{\mathrm{sh}}\}``
(see the [home page](index.md)) is common to every agent's KKT system — this is why the same
``\lambda`` appears in both agents' stationarity conditions below, rather than one multiplier
per agent.

### Hard-constrained KKT

From Section 3, ``Q_{ii} = \Gamma_i^\top\bar Q_i\Gamma_i + R = 1\cdot1\cdot1+1 = 2``,
``Q_{ij} = \Gamma_i^\top\bar Q_i\Gamma_j = 1`` (``j\neq i``), and
``q_i = \Gamma_i^\top(\bar Q_i\bar c+\bar q_i) = p = -2``. From Section 6,
``A^{\mathrm{sh}}_{\mathrm{state},i} = \bar C_x\Gamma_i = 1``. Agent `i`'s stationarity
condition (``\partial J_i/\partial u_i + A^{\mathrm{sh}\top}_{\mathrm{state},i}\lambda = 0``)
is then

```math
Q_{ii}u_i + Q_{ij}u_j + q_i + \lambda = 0 \quad\Longleftrightarrow\quad 2u_i + u_j - 2 + \lambda = 0,
```

together with primal/dual feasibility and complementary slackness
``\lambda\geq0,\ x^1\leq1,\ \lambda(x^1-1)=0``. At ``\lambda=0``: ``u_1=u_2=2/3``,
``x^1=4/3>1`` — infeasible, so the constraint is active. Setting ``x^1=1``
(``u_1=u_2=u=0.5``) in the stationarity condition: ``3(0.5)-2+\lambda=0 \Rightarrow \lambda=0.5``,
which is ``\geq0`` and consistent. Hence

```math
u_1=u_2=0.5,\qquad \lambda = 0.5,
```

matching `u_hard` in the test.

### Softened KKT

With `soften_state_constraints=true` and per-agent cost `k[i]=[k]`, Section 9 gives two
relaxed copies of the constraint (one per agent), each with its own multiplier
``\lambda_1,\lambda_2\geq0``:

```math
x^1 - s_1 \leq 1 \quad(\lambda_1), \qquad x^1 - s_2 \leq 1 \quad(\lambda_2),
\qquad s_1,s_2\geq0 \ (\mu_1,\mu_2\geq0).
```

Both copies contribute to every agent's stationarity condition (each row's ``u_i``
coefficient is still ``A^{\mathrm{sh}}_{\mathrm{state},i}=1``, from Section 9), while a slack
``s_i`` only appears — with coefficient ``-1`` — in *its own* agent's copy:

```math
\underbrace{2u_i+u_j-2}_{\partial J_i/\partial u_i} + \lambda_1+\lambda_2 = 0,
\qquad
\underbrace{k}_{\partial J_i/\partial s_i} - \lambda_i - \mu_i = 0,
```

with complementary slackness ``\lambda_j(x^1-s_j-1)=0`` and ``\mu_i s_i = 0``. Guessing the
symmetric solution ``u_1=u_2=u``, ``s_1=s_2=s``, ``\lambda_1=\lambda_2=\lambda``, the
``u``-stationarity condition gives ``3u-2+2\lambda=0``, i.e. ``u=(2-2\lambda)/3`` (note the
factor of ``2\lambda``: *two* copies now push back on ``u_i``, versus one in the hard case).

- **If ``s>0``** (slack used): ``\mu_i=0\Rightarrow\lambda=k``, and the copies are active
  (``x^1-s=1\Rightarrow 2u-s=1``). Substituting ``u=(2-2k)/3``:
  ```math
  s = 2u-1 = \frac{1-4k}{3}, \qquad u = \frac{2-2k}{3},
  ```
  valid (``s\geq0``) for ``k\leq1/4``. At `k=0.01`: ``u=0.66``, ``s=0.32`` — matching
  `u_soft_small`/`slacks(sol_soft_small.x)` in the test.
- **If ``s=0``**: the copies collapse to the hard constraint, ``x^1=1\Rightarrow u=0.5``, and
  the stationarity equation gives ``\lambda=0.25``; dual feasibility ``\mu_i=k-\lambda\geq0``
  then requires ``k\geq1/4``. At `k=1e6`: ``u=0.5=`` `u_hard`, ``s=0`` — matching
  `u_soft_large`/`slacks(sol_soft_large.x)`.

The two regimes meet exactly at ``k=1/4``, consistent with `s → 0` as `k → 0.25` in the
formula above.

### Multiple constraint rows (`prob2`, `m_x=2`)

Duplicating the same scalar row (``C_x=\mathrm{col}(1,1)``, ``b_x=(1,1)``) gives agent `i` two
slacks ``s_{i,1},s_{i,2}\geq0``, one per row of *its own* copy, at per-row costs
``k_{i,1},k_{i,2}``. Both rows bound the same physical quantity, ``x^1\leq1+s_{i,1}`` and
``x^1\leq1+s_{i,2}``, so the tightest one binds: reaching any target ``x^{1*}>1`` costs agent
`i` at least ``k_{i,1}(x^{1*}-1)+k_{i,2}(x^{1*}-1)``, minimized by setting
``s_{i,1}=s_{i,2}=x^{1*}-1`` (any larger value on one row is wasted cost with no relaxation
benefit). This reduces exactly to the single-row problem above with effective cost
``k_{\mathrm{eff},i}=k_{i,1}+k_{i,2}`` — which is what `gnep2_soft_small` checks
(`k_eff = 0.005+0.005 = 0.01`, reusing the single-row closed form from `k=0.01` above, with
both `slacks2` rows equal to ``(1-4k_{\mathrm{eff}})/3``).
