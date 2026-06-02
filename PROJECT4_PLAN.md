# Project 4: Bias Collapse in Shallow ReLU Networks — 1–2 Week Plan

**Goal:** Numerically reproduce the bias collapse phenomenon, develop mathematical intuition for the open problems, and produce a written report with findings (and ideally partial proofs).

---

## Background Summary

**Model:** 1D shallow ReLU network with fixed slopes:

```
f(x) = Σ_j a_j · σ(x − b_j),   σ(z) = max(0,z),   x ∈ [−1,1]
```

**Training:** Gradient flow on continuous MSE loss `L = ½ ∫_{-1}^{1} (f(x) − f*(x))² dx`

**The Phenomenon:** Even with m = 100 neurons, biases b_j collapse to only 6–8 distinct cluster values matching the number of inflection points of the target f*.

**Gradient Flow ODEs:**
- Amplitudes: `ȧ_j = −∫_{-1}^{1} (f − f*) σ(x − b_j) dx`
- Biases: `ḃ_j = a_j ∫_{b_j}^{1} (f − f*) dx`

**Open Problems:**
- **4.1** — Prove the cluster count is bounded by inflection points of f*
- **4.2** — Generalize collapse to R^d (d ≥ 2)
- **4.3** — Prove the pruning guarantee: collapsed neurons can be removed safely

---

## Numerical Foundation

### Implement the Gradient Flow Simulator ✅

- [x] Set up Python environment (numpy, scipy, matplotlib)
- [x] Implement f(x) as a sum of ReLUs given {a_j, b_j}
- [x] Implement the two integral ODEs (amplitudes + biases) using numerical quadrature (vectorized trapezoid rule + cumulative right integral)
- [x] Use `scipy.integrate.solve_ivp` (RK45) to integrate the system forward in time
- [x] Reproduce the figure from slide 93:
  - Target: `f*(x) = sin(2πx) + 0.5·sin(4πx)`, m = 100
  - Plot 1: Sorted bias trajectories over time → clustering confirmed
  - Plot 2: Final fit (red) vs. target (dashed) with bias dots
  - Plot 3: MSE loss on log scale

**Deliverable:** `notebooks/01_gradient_flow_simulator.ipynb` — figures saved to `figures/slide93_reproduction.png`, `figures/clusters_vs_inflections.png`. Simulation state saved to `data/sol_y.npy`.

---

### Parameter Sweep & Cluster Count Experiment ✅

- [x] Test different values of m: {20, 50, 100} with same target
- [x] Test multiple target functions:
  - `f*(x) = sin(2πx)` — 3 numerical inflection pts
  - `f*(x) = sin(2πx) + 0.5·sin(4πx)` — 7 inflection pts
  - `f*(x) = sin(2πx) + sin(6πx)` — 11 inflection pts
  - `f*(x) = x³ − x` — 1 inflection pt
  - `f*(x) = |x|` — 1 kink (non-smooth)
- [x] Count distinct clusters at convergence (tolerance `|b_i − b_j| < 0.02`)
- [x] Compare to inflection point counts; build results table
- [x] Save to `data/cluster_count_sweep.csv`

**Key finding:** At T=60, cluster counts often exceed inflection counts — longer integration times needed for large m (convergence is slow). For smooth, low-frequency targets (sin(2πx)+0.5sin(4πx), m=20), the conjecture holds (6 clusters ≤ 7 inflections). The x³−x and |x| results confirm the analytical inflection-point counter needs improvement for non-smooth targets.

**Deliverable:** `notebooks/02_parameter_sweep.ipynb` — figures in `figures/cluster_count_sweep.png`, `figures/bias_trajectories_all_targets.png`.

---

### Synchronization Dynamics Deep Dive ✅

- [x] Track two initially nearby biases (|b_i − b_j| = 0.05) — separation decays from 0.05 to ~1e-10
- [x] Verified O(ε) first-term behavior: power-law fit of |ḃ_i − ḃ_j| vs ε gives slope ≈ 1.0 ✓
- [x] Amplitude sum a_i + a_j is approximately conserved during merging ✓
- [x] Visualized positive feedback loop: all three quantities (b_sep, a_sep, db_sep) decay together
- [x] Stability test: after perturbing a merged pair by δ=0.01, separation re-merges to ~1e-10 ✓

**Deliverable:** `notebooks/03_synchronization_analysis.ipynb` — figures in `figures/sync_*.png`.

---

## Theory & Extensions

### Attack Open Problem 4.1 (Cluster Count Bound)

**Goal:** Try to prove `C(m, f*) ≤ #{inflection points of f*}`.

**Strategy:**
1. Write out the Lyapunov-style argument: show that the number of distinct bias locations cannot increase over time (merging is irreversible by the stability argument on slide 92)
2. Consider the energy functional: what is minimized at each cluster location?
3. Look at the "optimal k-kink piecewise linear L² approximation" — characterize its kink locations and relate them to inflection points of f*
4. For the upper bound, try contradiction: assume k+1 clusters exist at convergence and derive a contradiction using the L² approximation optimality conditions

**Reading:** Look up results on best piecewise linear approximation (L² theory), e.g., de Boor's work on spline approximation.

- [ ] Write a 1–2 page proof sketch (even if incomplete, record the key step that's missing)
- [ ] Identify the hardest step and what additional tools (PDE theory, convex analysis) might help

**Deliverable:** Proof sketch for Open Problem 4.1 with identified gaps.

---

### Attack Open Problem 4.3 (Provable Pruning)

**Goal:** Prove: if intra-cluster diameter ≤ δ, then `‖f̃ − f‖ ≤ δ · Σ|a_j|`.

**Strategy:**
1. Write out `f(x) − f̃(x)` explicitly in terms of the bias perturbations within each cluster
2. Use the Lipschitz property of σ(x − b): `|σ(x−b) − σ(x−b')| ≤ |b−b'|` for all x
3. So `|a_j(σ(x−b_j) − σ(x−b̄))| ≤ |a_j|·δ` where b̄ is the cluster centroid
4. Sum over all j to get the bound

- [ ] Work through this argument carefully; it may already be provable with elementary estimates
- [ ] Numerically verify the bound: for various δ, measure actual `‖f̃ − f‖_∞` and `‖f̃ − f‖_L²` and compare to `δ · Σ|a_j|`
- [ ] Check if `Σ|a_j|` stays bounded during training (this is the hard part)

**Deliverable:** Proof of the pruning bound (modulo the amplitude bound) + numerical verification.

---

### Explore Open Problem 4.2 (Higher Dimensions, Optional)

If time permits — start with d = 2.

- [ ] Implement 2D version: `f(x) = Σ a_j · σ(w_j · x + b_j)`, x ∈ [−1,1]²
- [ ] Fix directions w_j = (1,0) (all neurons have same direction) → reduces to 1D case; confirm collapse still happens
- [ ] Try random initial directions w_j; check if directional collapse (w_i → w_j) occurs alongside bias collapse
- [ ] Visualize the hyperplane configurations at t=0 and t=T

**Deliverable:** Exploratory figures for 2D collapse.

---

### Mean-Field PDE Connection (Optional but High-Impact)

- [ ] Write the empirical measure `μ(t) = (1/m) Σ δ_{(a_j(t), b_j(t))}`
- [ ] Write the mean-field PDE for μ (continuity equation with the ODE velocity field)
- [ ] Observe that bias collapse = μ(t) concentrating onto discrete support (sum of Dirac masses)
- [ ] This is "measure-valued gradient flow" — search for relevant PDE literature (Wasserstein gradient flows, JKO scheme)

---

### Write-Up

Structure the report as follows:

1. **Introduction** — State the model, phenomenon, and open problems (½ page)
2. **Numerical Experiments** — Reproduce slides 93–94 with your own code; parameter sweep table (1–2 pages)
3. **Synchronization Analysis** — Present your Day 5 verification of the O(ε) mechanism (1 page)
4. **Open Problem 4.1** — Proof sketch + identified gap (1–2 pages)
5. **Open Problem 4.3** — Proof of pruning bound (modulo amplitude bound) + numerics (1 page)
6. **Open Problem 4.2** — Exploratory 2D figures if completed (½–1 page)
7. **Conclusions & Future Directions** — What remains open, what tools are needed (½ page)

- [ ] Write LaTeX or Markdown draft
- [ ] Include all figures (bias trajectories, loss curves, cluster count table, pruning verification)

---

### Polish & Submit

- [ ] Proofread and tighten all arguments
- [ ] Make sure all code is clean, commented, and reproducible
- [ ] Commit everything to the repo

---

## Priority Ordering (if time is short)

| Priority | Task | Expected Time |
|----------|------|---------------|
| 1 | Implement ODE simulator + reproduce slide 93 figure | 1–2 days |
| 2 | Parameter sweep + cluster count table | 1 day |
| 3 | Open Problem 4.3 proof (elementary, likely achievable) | 1 day |
| 4 | Open Problem 4.1 proof sketch | 1–2 days |
| 5 | Synchronization deep dive | ½ day |
| 6 | Write-up | 1–2 days |
| 7 | Open Problem 4.2 (2D, optional) | 1 day |
| 8 | Mean-field PDE connection (optional) | 1 day |

---

## Key Formulas to Keep Handy

**Network output:**
```
f(x) = Σ_{j=1}^{m} a_j · max(0, x − b_j)
```

**ODE system (amplitudes):**
```
ȧ_j = −∫_{-1}^{1} (f(x) − f*(x)) · max(0, x − b_j) dx
```

**ODE system (biases):**
```
ḃ_j = a_j · ∫_{b_j}^{1} (f(x) − f*(x)) dx
```

**Loss:**
```
L = ½ ∫_{-1}^{1} (f(x) − f*(x))² dx
```

**Pruning bound to prove:**
```
‖f̃ − f‖ ≤ δ · Σ_{j=1}^{m} |a_j|,   where δ = max intra-cluster diameter
```

---

## Suggested File Structure

```
MathProject4/
├── PROJECT4_PLAN.md          ← this file
├── code/
│   ├── simulate.py           ← ODE integrator
│   ├── targets.py            ← target functions + inflection points
│   ├── cluster_analysis.py   ← cluster detection + counting
│   ├── experiments.py        ← parameter sweeps
│   └── plots.py              ← figure generation
├── figures/
│   ├── bias_trajectories.png
│   ├── final_fit.png
│   ├── loss_curve.png
│   └── cluster_count_table.png
└── writeup/
    └── project4_report.tex   ← or .md
```
