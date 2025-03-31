import numpy as np
import cvxpy as cp

# =============================================================================
#  Detailed SSD Efficiency Test via LP (Discretizing the Return Distribution)
# =============================================================================


def detailed_ssd_efficiency_test(candidate_returns, benchmark_returns, num_bins=50, tol=1e-4):
    """
    Detailed LP test for second-order stochastic dominance (SSD) via discretization.

    The test approximates the SSD condition:
        ∫_{-∞}^{x} F_candidate(t) dt <= ∫_{-∞}^{x} F_benchmark(t) dt  for all x,
    by discretizing the return space and computing cumulative sums.

    We seek nonnegative adjustments δ (one per grid point) such that for each grid index i:
        sum_{j=0}^{i} δ_j >= sum_{j=0}^{i} [F_candidate(x_j) - F_benchmark(x_j)].

    The objective is to minimize the total adjustment, sum(δ). An optimal value near zero
    indicates that no adjustment is needed—in other words, the candidate portfolio is SSD efficient.

    Parameters:
    -----------
    candidate_returns : np.array
        1D array of candidate portfolio returns.
    benchmark_returns : np.array
        1D array of benchmark returns.
    num_bins : int
        Number of discretization bins (grid points) over the return range.
    tol : float
        Tolerance for deciding if the total adjustment is zero.

    Returns:
    --------
    status : str
        The solver status from cvxpy.
    delta_opt : np.array
        The optimal adjustment vector (one per grid point).
    total_adjustment : float
        The sum of the optimal adjustments.
    grid : np.array
        The grid (discretized return values) used.
    diff : np.array
        Cumulative difference [cumsum(F_candidate - F_benchmark)] at the grid points.
    is_ssd_efficient : bool
        True if total_adjustment is less than or equal to tol.
    """
    # --- 1. Discretize the Return Space ---
    all_returns = np.concatenate([candidate_returns, benchmark_returns])
    min_val = np.min(all_returns)
    max_val = np.max(all_returns)
    grid = np.linspace(min_val, max_val, num_bins)

    # --- 2. Compute Empirical CDFs on the Grid ---
    f_candidate = np.array([np.mean(candidate_returns <= x) for x in grid])
    f_benchmark = np.array([np.mean(benchmark_returns <= x) for x in grid])

    # --- 3. Compute the Integrated (Cumulative) Difference ---
    diff = np.cumsum(f_candidate - f_benchmark)

    # --- 4. Set Up the LP ---
    # Lower-triangular matrix L such that (L @ delta)[i] = sum_{j=0}^{i} delta_j.
    L = np.tril(np.ones((num_bins, num_bins)))

    # Decision variable: delta (vector of length num_bins)
    delta = cp.Variable(num_bins)

    # For each grid index i, require:
    #   sum_{j=0}^{i} delta_j >= diff[i]
    constraints = [L @ delta >= diff,
                   delta >= 0]

    # Minimize the total adjustment.
    objective = cp.Minimize(cp.sum(delta))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    total_adjustment = cp.sum(delta).value
    status = prob.status
    delta_opt = delta.value
    is_ssd_efficient = (total_adjustment is not None) and (
        total_adjustment <= tol)

    return status, delta_opt, total_adjustment, grid, diff, is_ssd_efficient


def bootstrap_detailed_ssd_test(candidate_returns, benchmark_returns, num_bins=50, num_samples=500, tol=1e-4):
    """
    Bootstrap the detailed SSD test to assess the probability that the candidate portfolio is SSD efficient.

    Parameters:
    -----------
    candidate_returns : np.array
        1D array of candidate portfolio returns.
    benchmark_returns : np.array
        1D array of benchmark returns.
    num_bins : int
        Number of bins for the discretization.
    num_samples : int
        Number of bootstrap samples.
    tol : float
        Tolerance for the LP adjustment.

    Returns:
    --------
    efficiency_prob : float
        Fraction of bootstrap samples for which the candidate portfolio is SSD efficient.
    """
    T = len(candidate_returns)
    success_count = 0
    for _ in range(num_samples):
        idx = np.random.choice(T, T, replace=True)
        cand_sample = candidate_returns[idx]
        bench_sample = benchmark_returns[idx]
        status, _, total_adj, _, _, is_ssd_eff = detailed_ssd_efficiency_test(
            cand_sample, bench_sample, num_bins, tol
        )
        if status == 'optimal' and total_adj <= tol:
            success_count += 1
    efficiency_prob = success_count / num_samples
    return efficiency_prob
