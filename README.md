# NSF RTG: Project 4: Bias Collapse in Shallow ReLU Networks

## Overview

This project numerically investigates Open Problem 4.1 from the course slides: the bias collapse phenomenon in shallow ReLU networks trained by gradient flow. The central conjecture is:

$$\lim_{m \to \infty} C(m, f^*) = \#\{x \in [-1,1] : (f^*)''(x) = 0 \text{ and changes sign}\}$$

In plain terms: when you train a wide enough ReLU network on a smooth target function, the bias parameters spontaneously collapse into clusters. The number of clusters is conjectured to converge toward k, the number of inflection points of the target, as the network width m grows without bound. This is a statement about limiting behavior, not about equality at any finite m. For any fixed m, the cluster count may still be above k, and what we observe numerically is the convergence process itself. The goal of this project is to build empirical evidence for that convergence across many target functions and network widths, and to verify the mathematical conditions that must hold at a stationary point of the gradient flow.

---

## Research Goals

### Goal 1: Verify the Cluster Count Conjecture

Numerically confirm that C(m, f*) converges toward k as m grows large, and that this convergence stabilizes independently of m once m is large enough. If the cluster count fails to approach k even at large m, that is meaningful evidence against the conjecture and redirects the research.

### Goal 2: Show k+1 Cluster Configurations Are Unstable

Identify what breaks when the network has one more cluster than the optimal k. Show numerically that such overcomplete configurations are unstable, meaning small perturbations cause one cluster to dissolve. Instability is often easier to prove mathematically than convergence, so this gives a concrete mechanism that a future proof could formalize.

### Goal 3: Verify Stationary Point Conditions

Run the simulation to convergence, plug the final values of a_j and b_j back into the ODE expressions, and verify that both the amplitude velocity and the bias velocity are approximately zero. Also check that the integrated residual R_j from each final bias location to 1 is approximately zero for every active neuron, and that the number of active neurons never exceeds k. This connects the numerical simulation directly to the mathematical fixed point structure and builds evidence that the ODE framework is the right one for an eventual proof.

### Goal 4: Numerically Verify the Pruning Bound (Open Problem 4.3)

After convergence, check whether the collapsed network can be safely pruned to k neurons without significant performance loss. Specifically, verify the inequality:

$$\|\tilde{f} - f\|_{L^2} \leq \delta \cdot \sum_{j=1}^{m} |a_j|$$

where f is the converged full network, f_tilde is the pruned network with one neuron per cluster, and delta is the maximum intra-cluster bias diameter. This gives numerical evidence for Open Problem 4.3 across all target functions and network widths before any formal proof exists.

### Goal 5: Build Intuition for a Future Proof

Use the numerical evidence from the goals above to develop concrete intuition about why extra clusters are unstable and why the pruning bound holds. The goal is to identify which proof strategy to pursue, whether that is a Lyapunov argument, a fixed point analysis, or tools from approximation theory.

---

## Project Structure

```
MathProject4/
|
|-- MathProject Slides.pdf        Source slides defining the model, ODEs, and open problems
|
|-- simulate.py                   Primary simulation script, runs sequentially
|-- simulate_parallel.py          Extended simulation script using multiprocessing
|-- verify_pruning.py             Post-processing script verifying Open Problem 4.3 bound
|
|-- notebooks/
|   |-- 01_gradient_flow_simulator.ipynb    Exploratory notebook for early x^2 experiments
|
|-- figures/
|   |-- Replication data/         All simulation outputs organized by target, m, and T
|       |-- {target}/
|       |   |-- m={m}/
|       |       |-- T={T}/
|       |           |-- slide93_reproduction.png      Bias trajectories, final fit, loss curve
|       |           |-- clusters_vs_inflections.png   Cluster locations vs inflection points
|       |           |-- ode_verification.png          ODE velocities and R_j at convergence
|       |           |-- convergence_check.csv         Per neuron da, db, R_j, and active flag
|       |           |-- run_meta.csv                  Single row summary for restart recovery
|       |           |-- pruning_verification.png      Pruning bound check (from verify_pruning.py)
|       |
|       |-- run_summary.csv                  Aggregate results from simulate.py
|       |-- run_summary_parallel.csv         Aggregate results from simulate_parallel.py
|       |-- convergence_plot.png             C(m) vs m from simulate.py
|       |-- convergence_plot_parallel.png    C(m) vs m combining both scripts
|       |-- pruning_bound_results.csv        One row per run with all pruning bound metrics
|       |-- pruning_bound_summary.png        Summary plots across all runs for Open Problem 4.3
|
|-- data/                         Raw trajectory arrays from the exploratory notebook
    |-- sol_t.npy
    |-- sol_y.npy
    |-- losses.npy
```

---

## Scripts

### simulate.py: Sequential Baseline

**Purpose:** Establishes the baseline numerical evidence for the conjecture across six target functions with analytically known inflection point counts. Uses a range of network widths and integration times sufficient to show the beginning of the collapse phenomenon and identify which targets need longer integration.

**What it runs:**

| Targets | m values | T values |
|---|---|---|
| sin(pi x) k=1, x^3 k=1 | 50, 100, 250 | 200, 500, 1000 |
| sin(2pi x) k=3, x^5 minus 3x^3 k=3 | 500, 1000, 1500 | |
| sin(3pi x) k=5, sin(4pi x) k=7 | | |

**Key finding:** The conjecture is supported for simpler targets at these T values. sin(pi x) converges to exactly 1 cluster at m at or above 1000. sin(4pi x) reaches exactly 7 clusters at m=1500, T=1000. Harder targets including x^3, x^5 minus 3x^3, and high frequency sine functions are still in the process of collapsing at T=1000, which motivates the parallel script.

**Outputs:**
- Per run figures and convergence_check.csv in each run folder
- run_summary.csv containing one row per completed run
- convergence_plot.png showing C(m) vs m for all targets with a horizontal line at k

**Run with:** `python simulate.py`

---

### simulate_parallel.py: Parallel Extended Runs

**Purpose:** Addresses two limitations of simulate.py. First, integration times at or below T=1000 were insufficient for harder targets because the system was still mid collapse. Second, the sin(n pi x) family needed to be extended to k=9, 11, and 13 to test the conjecture at higher complexity. Running everything sequentially at T=5000 and T=10000 would take too long, so multiprocessing runs multiple combinations simultaneously across available CPU cores.

**What it runs:**

| Targets | m values | T values |
|---|---|---|
| All 6 from simulate.py | 50, 100, 250, 500 | 5000, 10000 |
| sin(5pi x) k=9 | 1000, 1500, 2000 | |
| sin(6pi x) k=11 | 3000, 5000 | |
| sin(7pi x) k=13 | | |

**Why T=5000 and T=10000 only:** simulate.py already demonstrated that T at or below 1000 does not produce convergence for harder targets. The new targets in this script inherit that established evidence and start directly at the T values where convergence is expected. Running low T values for the new targets would only reproduce an already known result.

**Why the new targets skip low T:** The existing targets provide the baseline showing low T is insufficient. The experimental question for the new targets is whether they converge at all, not whether they fail at low T, which is already established.

**Outputs:**
- Same per run figures and CSVs as simulate.py, written to the same folder structure
- run_summary_parallel.csv containing only T=5000 and T=10000 rows, with no overlap with simulate.py
- convergence_plot_parallel.png combining all T values from both scripts; existing targets show the full T=200 through T=10000 progression while new targets show T=5000 and T=10000 only

**Run with:** `python simulate_parallel.py`

---

### verify_pruning.py: Open Problem 4.3 Verification

**Purpose:** After the simulation scripts have completed, this script reads the saved b_j and a_j values from each run folder and numerically verifies the pruning bound from Open Problem 4.3. No re-simulation is required. It works with data from both simulate.py and simulate_parallel.py automatically.

**What it does per run:**
1. Reads b_j and a_j from convergence_check.csv and metadata from run_meta.csv
2. Groups neurons into clusters using the same gap tolerance as the simulation scripts
3. Computes delta, the maximum intra-cluster diameter across all clusters
4. Computes the sum of absolute amplitudes across all neurons
5. Constructs the pruned network f_tilde by replacing each cluster with one neuron at the centroid bias with summed amplitude
6. Computes the actual pruning error, the L2 norm of f_tilde minus f, on the quadrature grid
7. Computes the bound, delta times the amplitude sum
8. Checks whether the actual error is at or below the bound
9. Records the tightness ratio, which is the actual error divided by the bound, to show how close the bound is to being saturated
10. Saves a per run figure and appends to the global results CSV

**Why this matters:** The bound holds numerically across all runs completed so far (108 out of 108). The tightness values are consistently small (in the range 0.001 to 0.15), meaning the bound is satisfied but not tight. The key open challenge identified in the slides is whether the amplitude sum stays bounded as m grows, because the bound becomes vacuous if that sum blows up faster than delta shrinks. The summary figure in this script directly shows that behavior across all targets and m values.

**Outputs:**
- pruning_verification.png in each run folder showing the full network, pruned network, target, pointwise error, and bound check
- pruning_bound_results.csv with one row per run containing delta, amplitude sum, bound, actual error, bound_holds flag, and tightness
- pruning_bound_summary.png with four panels: actual error vs bound scatter, tightness vs m, amplitude sum vs m, and cluster diameter vs m

**Run with:** `python verify_pruning.py`

---

## Model

**Network:**

$$f(x) = \sum_{j=1}^{m} a_j \, \sigma(x - b_j), \quad \sigma(z) = \max(0, z), \quad x \in [-1, 1]$$

**Training:** Gradient flow on the continuous MSE loss:

$$\mathcal{L} = \frac{1}{2}\int_{-1}^{1}(f(x) - f^*(x))^2\,dx$$

**Gradient flow ODEs:**

$$\dot{a}_j = -\int_{-1}^{1}(f - f^*)\,\sigma(x - b_j)\,dx \qquad \dot{b}_j = a_j \int_{b_j}^{1}(f - f^*)\,dx$$

---

## Target Functions

| Key | Function | k (inflection pts) | Introduced in |
|---|---|---|---|
| sin_1pi | sin(pi x) | 1 | simulate.py |
| x_cubed | x^3 | 1 | simulate.py |
| sin_2pi | sin(2 pi x) | 3 | simulate.py |
| poly_k3 | x^5 minus 3x^3 | 3 | simulate.py |
| sin_3pi | sin(3 pi x) | 5 | simulate.py |
| sin_4pi | sin(4 pi x) | 7 | simulate.py |
| sin_5pi | sin(5 pi x) | 9 | simulate_parallel.py |
| sin_6pi | sin(6 pi x) | 11 | simulate_parallel.py |
| sin_7pi | sin(7 pi x) | 13 | simulate_parallel.py |

Inflection point counts for sin(n pi x): the second derivative is negative n squared pi squared times sin(n pi x), which has exactly 2n minus 1 sign changing zeros in the open interval from negative 1 to 1.

---

## Output Files Per Run

| File | Goal | Contents |
|---|---|---|
| slide93_reproduction.png | 1 | Sorted bias trajectories, final network fit vs target, MSE loss on log scale |
| clusters_vs_inflections.png | 1 | Final cluster locations in blue overlaid on analytically known inflection points in green |
| ode_verification.png | 3 | Amplitude velocity, bias velocity, and integrated residual R_j at the final time, all expected near zero |
| convergence_check.csv | 3 | Per neuron table of b_j, a_j, amplitude velocity, bias velocity, R_j, active flag, and R near zero flag |
| run_meta.csv | All | Single row summary of key metrics enabling Ctrl+C safe restart without repeating completed runs |
| pruning_verification.png | 4 | Full network vs pruned network vs target, pointwise pruning error, and bar chart comparing actual error to bound |
