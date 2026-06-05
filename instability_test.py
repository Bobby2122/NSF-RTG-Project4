"""
instability_test.py
===================
Goal 2: Show numerically that configurations with more than k clusters are
unstable, and explore behavior of configurations near k from both sides.

Background
----------
The slides establish that the k-cluster merged state is neutrally stable.
Goal 2 asks the converse: if you start with more than k clusters, does
gradient flow drive the system back to k? This script tests that, and
extends the test to runs that are near k but not exactly at k.

Qualification: a run qualifies if |n_clusters - k_true| <= NEAR_K_THRESHOLD.

Three run categories and their test types
-----------------------------------------
  exact_k  (n_clusters == k_true)
      Two injection tests: 'near' and 'isolated'.
      One extra neuron is added artificially to create a k+1 configuration.
      Tests whether gradient flow dissolves the injected cluster.

  above_k  (k_true < n_clusters <= k_true + NEAR_K_THRESHOLD)
      One test: 'natural'.
      No injection. The ODE is continued forward for T_PERTURB from the
      current state. Tests whether the naturally overcomplete configuration
      dissolves on its own toward k without any external perturbation.

  below_k  (k_true - NEAR_K_THRESHOLD <= n_clusters < k_true)
      Two injection tests: 'near' and 'isolated'.
      Same injection logic as exact_k. Tests whether an under-complete
      configuration accepts the injected neuron (final C rises toward k)
      or rejects it (final C stays below k or returns to original).

Injection strategies (for exact_k and below_k)
-----------------------------------------------
  near      -- inject just outside the boundary of an existing cluster.
               Tests whether proximity causes the neuron to merge in.
  isolated  -- inject at the point farthest from all cluster centers.
               Tests whether the amplitude decays with no nearby attractor.

Run order
---------
  1. python simulate.py
  2. python simulate_parallel.py
  3. python verify_pruning.py
  4. python instability_test.py   <-- this script

Outputs
-------
  Per qualifying run and test type (saved in the original run folder):
      goal2_near.png        -- 3-panel figure for near injection
      goal2_isolated.png    -- 3-panel figure for isolated injection
      goal2_natural.png     -- 3-panel figure for natural dissolution

  Global (saved in figures/Replication data/):
      goal2_results.csv     -- one row per (run, test_type)
      goal2_summary.png     -- heatmap and bar chart across all tested runs
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, csv, time
from multiprocessing import Pool, cpu_count

# =============================================================================
# Constants
# =============================================================================
N_QUAD           = 400
X_QUAD           = np.linspace(-1.0, 1.0, N_QUAD)
DX               = X_QUAD[1] - X_QUAD[0]
CLUSTER_TOL      = 0.02
T_PERTURB        = 1000
N_SAVE           = 300
A_INJECT_SCALE   = 0.1    # injected amplitude = scale * mean|a_j|
NEAR_K_THRESHOLD = 2      # |n_clusters - k_true| <= this qualifies a run

FIG_BASE    = os.path.join('figures', 'Replication data')
RESULTS_CSV = os.path.join(FIG_BASE, 'goal2_results.csv')
SUMMARY_PNG = os.path.join(FIG_BASE, 'goal2_summary.png')

RESULTS_FIELDS = [
    'target', 'm', 'T_original', 'k_true',
    'run_category',       # exact_k | above_k | below_k
    'initial_n_clusters', # C at start of test (before any perturbation)
    'test_type',          # near | isolated | natural
    'b_inject',           # injection location (N/A for natural)
    'a_inject_initial',   # injected amplitude at t=0 (N/A for natural)
    'a_inject_final',     # injected amplitude at t=T_PERTURB (N/A for natural)
    'a_decayed',          # 1 if |a_final| < 0.1*|a_initial| (N/A for natural)
    'b_inject_final',     # injected bias at t=T_PERTURB (N/A for natural)
    'b_merged',           # 1 if injected bias merged into original cluster (N/A for natural)
    'final_n_clusters',   # C after T_PERTURB
    'cluster_change',     # final_n_clusters - initial_n_clusters
    'returned_to_k',      # 1 if final_n_clusters == k_true
    'approached_k',       # 1 if moved closer to k during test
]

NA = 'N/A'

# =============================================================================
# Target functions (lambdas looked up inside worker after module re-import)
# =============================================================================
TARGETS = {
    'sin_1pi': (r'$\sin(\pi x)$',      1, lambda x: np.sin(    np.pi * x)),
    'x_cubed': (r'$x^3$',              1, lambda x: x**3),
    'sin_2pi': (r'$\sin(2\pi x)$',     3, lambda x: np.sin(2 * np.pi * x)),
    'poly_k3': (r'$x^5-3x^3$',         3, lambda x: x**5 - 3 * x**3),
    'sin_3pi': (r'$\sin(3\pi x)$',     5, lambda x: np.sin(3 * np.pi * x)),
    'sin_4pi': (r'$\sin(4\pi x)$',     7, lambda x: np.sin(4 * np.pi * x)),
    'sin_5pi': (r'$\sin(5\pi x)$',     9, lambda x: np.sin(5 * np.pi * x)),
    'sin_6pi': (r'$\sin(6\pi x)$',    11, lambda x: np.sin(6 * np.pi * x)),
    'sin_7pi': (r'$\sin(7\pi x)$',    13, lambda x: np.sin(7 * np.pi * x)),
}

# =============================================================================
# Core math
# =============================================================================

def relu(z):
    return np.maximum(0.0, z)

def network(x, a, b):
    return (a * relu(x[:, None] - b[None, :])).sum(axis=1)

def make_ode(m, f_star):
    fstar_vals = f_star(X_QUAD)
    def ode(t, y):
        a, b      = y[:m], y[m:]
        residual  = network(X_QUAD, a, b) - fstar_vals
        relu_mat  = relu(X_QUAD[:, None] - b[None, :])
        da        = -(residual[:, None] * relu_mat).sum(0) * DX
        cum_right = np.cumsum(residual[::-1])[::-1] * DX
        idx       = np.searchsorted(X_QUAD, b).clip(0, N_QUAD - 1)
        db        = a * cum_right[idx]
        return np.concatenate([da, db])
    return ode

def count_clusters(biases, tol=CLUSTER_TOL):
    s = np.sort(biases)
    return 1 + int(np.sum(np.diff(s) > tol))

def get_cluster_centers(b, a, tol=CLUSTER_TOL):
    sort_idx = np.argsort(b)
    bs = b[sort_idx]
    centers, group_b = [], [bs[0]]
    for i in range(1, len(bs)):
        if bs[i] - bs[i-1] > tol:
            centers.append(float(np.mean(group_b)))
            group_b = []
        group_b.append(bs[i])
    centers.append(float(np.mean(group_b)))
    return np.array(centers)

# =============================================================================
# Injection location strategies
# =============================================================================

def inject_near(cluster_centers):
    """Just outside the boundary of the nearest cluster to the largest gap."""
    pts  = np.concatenate([[-1.0], np.sort(cluster_centers), [1.0]])
    gaps = np.diff(pts)
    idx  = int(np.argmax(gaps))
    left, right  = pts[idx], pts[idx + 1]
    midpoint     = (left + right) / 2.0
    nearest      = cluster_centers[np.argmin(np.abs(cluster_centers - midpoint))]
    direction    = np.sign(midpoint - nearest) if midpoint != nearest else 1.0
    b_inject     = float(np.clip(nearest + direction * 3 * CLUSTER_TOL,
                                 -1.0 + 1e-4, 1.0 - 1e-4))
    return b_inject

def inject_isolated(cluster_centers):
    """Maximally distant location from all cluster centers."""
    candidates = np.linspace(-1.0, 1.0, 2000)
    min_dists  = np.array([np.min(np.abs(x - cluster_centers))
                           for x in candidates])
    return float(candidates[np.argmax(min_dists)])

# =============================================================================
# Load a qualifying run from a folder
# =============================================================================

def load_qualifying_run(run_dir):
    """
    Load run data if |n_clusters - k_true| <= NEAR_K_THRESHOLD.
    Returns dict with run_category added, or None if not qualifying.
    """
    f_csv  = os.path.join(run_dir, 'convergence_check.csv')
    f_meta = os.path.join(run_dir, 'run_meta.csv')
    if not (os.path.exists(f_csv) and os.path.exists(f_meta)):
        return None

    with open(f_meta, newline='') as mf:
        meta = next(csv.DictReader(mf))

    n_clusters = int(meta['n_clusters'])
    k_true     = int(meta['k_true'])
    diff       = n_clusters - k_true

    if abs(diff) > NEAR_K_THRESHOLD:
        return None

    if diff == 0:
        category = 'exact_k'
    elif diff > 0:
        category = 'above_k'
    else:
        category = 'below_k'

    b_vals, a_vals = [], []
    with open(f_csv, newline='') as cf:
        for row in csv.DictReader(cf):
            b_vals.append(float(row['b_j']))
            a_vals.append(float(row['a_j']))

    return {
        'target':       meta['target'],
        'm':            int(meta['m']),
        'T_original':   int(meta['T']),
        'k_true':       k_true,
        'n_clusters':   n_clusters,
        'run_category': category,
        'b':            np.array(b_vals),
        'a':            np.array(a_vals),
    }

# =============================================================================
# Per-test worker
# =============================================================================

def test_one(args):
    run_dir, run_data, test_type = args

    target_key  = run_data['target']
    m_orig      = run_data['m']
    T_original  = run_data['T_original']
    k_true      = run_data['k_true']
    b_orig      = run_data['b']
    a_orig      = run_data['a']
    run_category = run_data['run_category']
    initial_n   = run_data['n_clusters']

    if target_key not in TARGETS:
        return None
    target_label, _, f_star = TARGETS[target_key]

    out_fig = os.path.join(run_dir, f'goal2_{test_type}.png')
    cluster_centers = get_cluster_centers(b_orig, a_orig)

    # =========================================================================
    # Natural dissolution test (above_k only, no injection)
    # =========================================================================
    if test_type == 'natural':
        m_run = m_orig
        sol   = solve_ivp(
            make_ode(m_run, f_star),
            t_span=(0.0, T_PERTURB),
            y0=np.concatenate([a_orig, b_orig]),
            method='RK45',
            t_eval=np.linspace(0.0, T_PERTURB, N_SAVE),
            rtol=1e-4, atol=1e-6,
            max_step=max(0.1, T_PERTURB / 500),
        )

        cluster_counts = np.array([
            count_clusters(sol.y[m_run:, i]) for i in range(sol.y.shape[1])
        ])
        final_n       = int(cluster_counts[-1])
        cluster_change = final_n - initial_n
        returned_to_k  = int(final_n == k_true)
        approached_k   = int(final_n < initial_n)  # dissolved toward k

        if not os.path.exists(out_fig):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(
                f'Goal 2: Natural Dissolution (above k)\n'
                f'target={target_label},  m={m_orig},  T_orig={T_original},  k={k_true}'
                f'  |  initial C={initial_n},  final C={final_n},  returned to k: {bool(returned_to_k)}',
                fontsize=10)

            ax = axes[0]
            for j in range(m_run):
                ax.plot(sol.t, np.sort(sol.y[m_run:, :], axis=0)[j],
                        color='steelblue', alpha=min(0.3, 15.0/m_run), lw=0.5)
            for cc in cluster_centers:
                ax.axhline(cc, color='green', lw=0.8, linestyle='--', alpha=0.5)
            ax.set_xlabel('Time after test start')
            ax.set_ylabel('Bias location')
            ax.set_title('Bias Trajectories\n(green = original cluster centers)')
            ax.set_xlim([0, T_PERTURB])

            ax = axes[1]
            ax.plot(sol.t, cluster_counts, color='purple', lw=1.8)
            ax.axhline(k_true,    color='crimson', lw=1.5, linestyle='--',
                       label=f'k={k_true} (target)')
            ax.axhline(initial_n, color='gray',   lw=1.0, linestyle=':',
                       label=f'initial C={initial_n}')
            ax.set_xlabel('Time after test start')
            ax.set_ylabel('Cluster count $C(t)$')
            ax.set_title(f'Cluster Count\nfinal={final_n},  returned to k: {bool(returned_to_k)}')
            ax.legend(fontsize=8)
            ax.set_xlim([0, T_PERTURB])
            ax.grid(True, alpha=0.3)

            ax = axes[2]
            bars_vals  = [initial_n, final_n, k_true]
            bars_lbls  = [f'Initial C\n({initial_n})', f'Final C\n({final_n})',
                          f'Target k\n({k_true})']
            bar_colors = ['steelblue',
                          'limegreen' if returned_to_k else ('orange' if approached_k else 'crimson'),
                          'gray']
            ax.bar(bars_lbls, bars_vals, color=bar_colors, edgecolor='black', lw=0.8)
            ax.set_ylabel('Cluster count')
            ax.set_title('Dissolution Summary\n'
                         f'approached k: {bool(approached_k)},  returned: {bool(returned_to_k)}')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(out_fig, bbox_inches='tight')
            plt.close()

        return {
            'target':            target_key,
            'm':                 m_orig,
            'T_original':        T_original,
            'k_true':            k_true,
            'run_category':      run_category,
            'initial_n_clusters': initial_n,
            'test_type':         test_type,
            'b_inject':          NA,
            'a_inject_initial':  NA,
            'a_inject_final':    NA,
            'a_decayed':         NA,
            'b_inject_final':    NA,
            'b_merged':          NA,
            'final_n_clusters':  final_n,
            'cluster_change':    cluster_change,
            'returned_to_k':     returned_to_k,
            'approached_k':      approached_k,
        }

    # =========================================================================
    # Injection test (exact_k and below_k)
    # =========================================================================
    if test_type == 'near':
        b_inject = inject_near(cluster_centers)
    else:
        b_inject = inject_isolated(cluster_centers)

    a_inject_initial = max(float(A_INJECT_SCALE * np.abs(a_orig).mean()), 0.01)
    m_new  = m_orig + 1
    a_init = np.append(a_orig, a_inject_initial)
    b_init = np.append(b_orig, b_inject)

    sol = solve_ivp(
        make_ode(m_new, f_star),
        t_span=(0.0, T_PERTURB),
        y0=np.concatenate([a_init, b_init]),
        method='RK45',
        t_eval=np.linspace(0.0, T_PERTURB, N_SAVE),
        rtol=1e-4, atol=1e-6,
        max_step=max(0.1, T_PERTURB / 500),
    )

    a_inj_traj = sol.y[m_orig, :]
    b_inj_traj = sol.y[m_new + m_orig, :]
    cluster_counts = np.array([
        count_clusters(sol.y[m_new:, i]) for i in range(sol.y.shape[1])
    ])

    a_inject_final = float(a_inj_traj[-1])
    b_inject_final = float(b_inj_traj[-1])
    final_n        = int(cluster_counts[-1])
    cluster_change = final_n - initial_n
    a_decayed      = int(abs(a_inject_final) < 0.1 * abs(a_inject_initial))
    b_merged       = int(np.min(np.abs(b_inject_final - cluster_centers)) < CLUSTER_TOL)
    returned_to_k  = int(final_n == k_true)

    # approached_k: for exact_k (initial==k), injection raised to k+1, so
    # returned_to_k == approached_k. For below_k (initial<k), approached_k
    # means final moved closer to k (final > initial).
    if run_category == 'below_k':
        approached_k = int(final_n > initial_n)
    else:
        approached_k = returned_to_k

    if not os.path.exists(out_fig):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f'Goal 2: Instability Test ({test_type} injection)  [{run_category}]\n'
            f'target={target_label},  m={m_orig},  T_orig={T_original},  k={k_true}'
            f'  |  initial C={initial_n},  final C={final_n},  returned to k: {bool(returned_to_k)}',
            fontsize=10)

        ax = axes[0]
        for j in range(m_orig):
            ax.plot(sol.t, np.sort(sol.y[m_new:m_new+m_orig, :], axis=0)[j],
                    color='steelblue', alpha=min(0.3, 15.0/m_orig), lw=0.5)
        ax.plot(sol.t, b_inj_traj, color='crimson', lw=1.8, label='Injected neuron')
        for cc in cluster_centers:
            ax.axhline(cc, color='green', lw=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel('Time after injection')
        ax.set_ylabel('Bias location')
        ax.set_title('Bias Trajectories\n(red=injected, green=original clusters)')
        ax.legend(fontsize=8)
        ax.set_xlim([0, T_PERTURB])

        ax = axes[1]
        ax.plot(sol.t, np.abs(a_inj_traj), color='darkorange', lw=1.8)
        ax.axhline(0.1 * abs(a_inject_initial), color='k', lw=1,
                   linestyle='--', label='10% of initial (decay threshold)')
        ax.set_xlabel('Time after injection')
        ax.set_ylabel('$|a_{\\mathrm{inject}}(t)|$')
        ax.set_title(f'Injected Amplitude\ninitial={a_inject_initial:.3f},  '
                     f'final={abs(a_inject_final):.3f},  decayed: {bool(a_decayed)}')
        ax.legend(fontsize=8)
        ax.set_xlim([0, T_PERTURB])
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(sol.t, cluster_counts, color='purple', lw=1.8)
        ax.axhline(k_true,    color='crimson', lw=1.5, linestyle='--',
                   label=f'k={k_true}')
        ax.axhline(initial_n + 1, color='gray', lw=1.0, linestyle=':',
                   label=f'post-inject={initial_n+1}')
        ax.set_xlabel('Time after injection')
        ax.set_ylabel('Cluster count $C(t)$')
        ax.set_title(f'Cluster Count\nfinal={final_n},  returned to k: {bool(returned_to_k)}')
        ax.legend(fontsize=8)
        ax.set_xlim([0, T_PERTURB])
        ax.set_ylim([max(0, k_true - 1), initial_n + 3])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_fig, bbox_inches='tight')
        plt.close()

    return {
        'target':             target_key,
        'm':                  m_orig,
        'T_original':         T_original,
        'k_true':             k_true,
        'run_category':       run_category,
        'initial_n_clusters': initial_n,
        'test_type':          test_type,
        'b_inject':           f'{b_inject:.4f}',
        'a_inject_initial':   f'{a_inject_initial:.4f}',
        'a_inject_final':     f'{a_inject_final:.4f}',
        'a_decayed':          a_decayed,
        'b_inject_final':     f'{b_inject_final:.4f}',
        'b_merged':           b_merged,
        'final_n_clusters':   final_n,
        'cluster_change':     cluster_change,
        'returned_to_k':      returned_to_k,
        'approached_k':       approached_k,
    }

# =============================================================================
# Summary figure
# =============================================================================

def make_summary_figure(results):
    """
    Left:  heatmap of returned_to_k / approached_k for every (run, test_type).
           Columns: near, isolated, natural.
           Cells that do not apply to a run are shown in light gray.
    Right: bar chart of success rate per test_type.
    """
    test_types  = ['near', 'isolated', 'natural']
    categories  = ['exact_k', 'above_k', 'below_k']
    cat_colors  = {'exact_k': '#4a90d9', 'above_k': '#e67e22', 'below_k': '#27ae60'}

    # Build run label list preserving order
    run_labels = []
    run_cats   = []
    seen       = set()
    for r in results:
        lbl = f"{r['target']}  m={r['m']}  T={r['T_original']}  C={r['initial_n_clusters']}  [{r['run_category']}]"
        if lbl not in seen:
            run_labels.append(lbl)
            run_cats.append(r['run_category'])
            seen.add(lbl)

    data = {lbl: {} for lbl in run_labels}
    for r in results:
        lbl  = f"{r['target']}  m={r['m']}  T={r['T_original']}  C={r['initial_n_clusters']}  [{r['run_category']}]"
        val  = int(r['returned_to_k']) if r['returned_to_k'] != NA else int(r['approached_k'])
        data[lbl][r['test_type']] = val

    n_runs = len(run_labels)
    fig_h  = max(5, n_runs * 0.35 + 3)
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_h))
    fig.suptitle(
        f'Goal 2: Instability Near k  (threshold = ±{NEAR_K_THRESHOLD})',
        fontsize=13)

    # Left: heatmap
    ax = axes[0]
    CMAP_VALID = matplotlib.colors.ListedColormap(['crimson', 'limegreen'])

    mat = np.full((n_runs, len(test_types)), np.nan)
    for i, lbl in enumerate(run_labels):
        for j, tt in enumerate(test_types):
            if tt in data[lbl]:
                mat[i, j] = data[lbl][tt]

    masked = np.ma.masked_invalid(mat)
    ax.imshow(masked, cmap=CMAP_VALID, vmin=0, vmax=1, aspect='auto')
    # Gray for N/A cells
    na_overlay = np.zeros((n_runs, len(test_types), 4))
    for i in range(n_runs):
        for j in range(len(test_types)):
            if np.isnan(mat[i, j]):
                na_overlay[i, j] = [0.85, 0.85, 0.85, 1.0]
    ax.imshow(na_overlay, aspect='auto')

    ax.set_xticks(range(len(test_types)))
    ax.set_xticklabels(test_types, fontsize=10)
    ax.set_yticks(range(n_runs))
    ax.set_yticklabels(run_labels, fontsize=6)
    ax.set_title('Success: returned/approached k?\n(green=yes, red=no, gray=N/A)')

    for i in range(n_runs):
        for j, tt in enumerate(test_types):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, 'YES' if mat[i,j]==1 else 'NO',
                        ha='center', va='center', fontsize=6,
                        color='white', fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=6, color='#888888')

    # Right: bar chart per test_type
    ax = axes[1]
    bar_colors_tt = ['steelblue', 'darkorange', 'purple']
    for ji, tt in enumerate(test_types):
        vals = [r for r in results if r['test_type'] == tt]
        if not vals:
            continue
        successes = sum(int(r['returned_to_k']) if r['returned_to_k'] != NA
                        else int(r['approached_k']) for r in vals)
        rate = successes / len(vals)
        ax.bar(ji, rate, color=bar_colors_tt[ji], edgecolor='black',
               label=f'{tt}  ({successes}/{len(vals)})')

    ax.set_xticks(range(len(test_types)))
    ax.set_xticklabels(test_types)
    ax.set_ylabel('Success rate')
    ax.set_ylim([0, 1.1])
    ax.axhline(1.0, color='k', lw=1, linestyle='--', alpha=0.4)
    ax.set_title('Success Rate by Test Type')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'Summary figure -> {SUMMARY_PNG}')

# =============================================================================
# Worker wrapper
# =============================================================================

def _worker(args):
    return test_one(args)

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    t_start = time.time()

    # ── Discover and classify qualifying runs ──────────────────────────────────
    qualifying = []
    for target_key in os.listdir(FIG_BASE):
        target_path = os.path.join(FIG_BASE, target_key)
        if not os.path.isdir(target_path) or target_key not in TARGETS:
            continue
        for m_dir in os.listdir(target_path):
            m_path = os.path.join(target_path, m_dir)
            if not os.path.isdir(m_path):
                continue
            for t_dir in os.listdir(m_path):
                t_path = os.path.join(m_path, t_dir)
                if not os.path.isdir(t_path):
                    continue
                run_data = load_qualifying_run(t_path)
                if run_data is not None:
                    qualifying.append((t_path, run_data))

    qualifying.sort(key=lambda x: (x[1]['run_category'],
                                   x[1]['target'], x[1]['m'], x[1]['T_original']))

    counts = {'exact_k': 0, 'above_k': 0, 'below_k': 0}
    for _, rd in qualifying:
        counts[rd['run_category']] += 1

    print(f'Qualifying runs (threshold = ±{NEAR_K_THRESHOLD}):')
    print(f'  exact_k  (C == k)          : {counts["exact_k"]}')
    print(f'  above_k  (k < C <= k+{NEAR_K_THRESHOLD})    : {counts["above_k"]}')
    print(f'  below_k  (k-{NEAR_K_THRESHOLD} <= C < k)    : {counts["below_k"]}')
    print(f'  Total                      : {len(qualifying)}')

    # ── Determine test types per category ─────────────────────────────────────
    def test_types_for(category):
        if category == 'above_k':
            return ['natural']
        else:
            return ['near', 'isolated']

    # ── Load existing results for restart safety ───────────────────────────────
    existing = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['target'], int(row['m']),
                       int(row['T_original']), row['test_type'])
                existing[key] = row
        print(f'\nLoaded {len(existing)} existing results from {RESULTS_CSV}')

    # ── Build job list ─────────────────────────────────────────────────────────
    jobs       = []
    skip_count = 0

    for run_dir, run_data in qualifying:
        for tt in test_types_for(run_data['run_category']):
            key     = (run_data['target'], run_data['m'],
                       run_data['T_original'], tt)
            out_fig = os.path.join(run_dir, f'goal2_{tt}.png')
            if key in existing and os.path.exists(out_fig):
                skip_count += 1
                print(f'  SKIP  {run_data["target"]:<12}  '
                      f'm={run_data["m"]:<5}  T={run_data["T_original"]:<6}  '
                      f'{tt:<10}  [{run_data["run_category"]}]')
            else:
                jobs.append((run_dir, run_data, tt))

    # Cap workers conservatively: ODE integration for large m is memory-heavy.
    # Each spawned process carries ~100 MB Python runtime overhead on top of
    # numpy arrays, so too many workers at once can exhaust available RAM.
    n_workers = max(1, min(cpu_count() - 2, 6))
    print(f'\nJobs to run : {len(jobs)}')
    print(f'Skipped     : {skip_count}')
    print(f'Workers     : {n_workers}')

    # ── Run in parallel ────────────────────────────────────────────────────────
    all_results = dict(existing)
    new_count   = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_worker, jobs):
            if result is None:
                continue
            key = (result['target'], int(result['m']),
                   int(result['T_original']), result['test_type'])
            all_results[key] = result
            new_count += 1

            success = (int(result['returned_to_k']) if result['returned_to_k'] != NA
                       else int(result['approached_k']))
            label   = 'SUCCESS' if success else 'no change'
            print(f'  DONE  {result["target"]:<12}  '
                  f'm={result["m"]:<5}  T={result["T_original"]:<6}  '
                  f'{result["test_type"]:<10}  [{result["run_category"]}]  '
                  f'C: {result["initial_n_clusters"]} -> {result["final_n_clusters"]}'
                  f'  k={result["k_true"]}  [{label}]')

    # ── Write results CSV ──────────────────────────────────────────────────────
    sorted_results = sorted(
        all_results.values(),
        key=lambda r: (r['run_category'], r['target'],
                       int(r['m']), int(r['T_original']), r['test_type'])
    )
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow({k: r[k] for k in RESULTS_FIELDS})
    print(f'\nResults CSV -> {RESULTS_CSV}  ({len(sorted_results)} rows)')

    if sorted_results:
        make_summary_figure(sorted_results)

    wall = time.time() - t_start
    print(f'\nDone.  {new_count} new,  {skip_count} skipped,  '
          f'wall time {wall:.1f}s')
