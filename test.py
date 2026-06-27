import numpy as np
import matplotlib.pyplot as plt


def make_stepsizes(K, lambda0=0.30, a=0.75):
    """
    lambda_k = lambda0 / (k+1)^a.

    This satisfies lambda in l2 but not l1 for a in (1/2, 1].
    """
    k = np.arange(K)
    lambdas = lambda0 / (k + 1.0) ** a

    if np.any(lambdas <= 0) or np.any(lambdas >= 1):
        raise ValueError("All lambda_k must lie in (0, 1). Reduce lambda0.")

    return lambdas


def make_plot_indices(K, n_plot_points=1500):
    """
    Log-spaced plotting points, plus the first few iterations.
    This avoids trying to draw 500,000 points for every curve.
    """
    early = np.arange(min(10, K))
    log_part = np.logspace(0, np.log10(K), num=n_plot_points).astype(int) - 1
    idx = np.unique(np.concatenate([early, log_part]))
    idx = idx[(idx >= 0) & (idx < K)]
    return idx


def simulate_many_random_rotations(
    n_angles=100,
    K=500_000,
    n_runs=100,
    x0=np.array([1.0, 1.0]),
    lambda0=0.30,
    a=0.75,
    seed=123,
    n_plot_points=1500,
    block_size=500,
):
    """
    Simulates SKM for 100 random rotations.

    Operators:
        T_0 = I
        T_1 = R_theta

    At each iteration, the algorithm samples I or R_theta with probability 1/2.

    The simulation is memory-efficient:
    it tracks ||x_k||^2 rather than full x_k, because rotations preserve norm
    and the residual depends only on ||x_k||.
    """
    rng = np.random.default_rng(seed)

    # Avoid exactly 0 or 2pi, because then Fix(T) is not {0}.
    angle_eps = 1e-3
    angles = rng.uniform(angle_eps, 2.0 * np.pi - angle_eps, size=n_angles)
    angle_degrees = np.degrees(angles)

    lambdas = make_stepsizes(K, lambda0=lambda0, a=a)
    eta = lambdas * (1.0 - lambdas)

    # Lambda_0 = 1, Lambda_{k+1} = Lambda_k / (1 + 8 lambda_k^2).
    Lambda = np.empty(K + 1)
    Lambda[0] = 1.0
    Lambda[1:] = np.cumprod(1.0 / (1.0 + 8.0 * lambdas**2))

    weights = Lambda[1:] * eta
    cum_eta = np.cumsum(eta)

    # For every nonzero rotation here, Fix(T) = {0}.
    dist0_sq = float(np.dot(x0, x0))

    # Since both I and R_theta fix 0, sigma_*^2 = 0.
    paper_bound_sq = dist0_sq / (Lambda[1:] * cum_eta)

    plot_indices = make_plot_indices(K, n_plot_points=n_plot_points)
    horizons = plot_indices + 1

    # For T = 0.5(I + R_theta),
    # ||T x - x||^2 = 0.5(1 - cos(theta)) ||x||^2.
    residual_coeff = 0.5 * (1.0 - np.cos(angles))
    one_minus_cos = 1.0 - np.cos(angles)

    # Track ||x_k||^2 for each angle and stochastic run.
    norm_sq = np.full((n_angles, n_runs), dist0_sq, dtype=float)

    empirical_random_iter_residual_sq = np.empty((n_angles, len(plot_indices)))

    cum_weighted_residual = np.zeros(n_angles, dtype=float)
    cum_weight = 0.0

    plot_ptr = 0

    for start in range(0, K, block_size):
        end = min(K, start + block_size)
        B = end - start

        eta_block = eta[start:end]
        weights_block = weights[start:end]

        # If rotation is sampled at iteration k, then
        # ||x_{k+1}||^2
        # = |(1-lambda_k) + lambda_k exp(i theta)|^2 ||x_k||^2
        # = [1 - 2 lambda_k(1-lambda_k)(1-cos(theta))] ||x_k||^2.
        rotation_factors = (
            1.0 - 2.0 * eta_block[:, None] * one_minus_cos[None, :]
        )

        # Bernoulli choices: True means use rotation, False means use identity.
        use_rotation = rng.random((B, n_angles, n_runs)) < 0.5

        multipliers = np.where(
            use_rotation,
            rotation_factors[:, :, None],
            1.0,
        )

        cum_after_update = np.cumprod(multipliers, axis=0)

        # Residual at iteration k is evaluated before the k-th update.
        pre_update_multiplier = np.empty_like(cum_after_update)
        pre_update_multiplier[0, :, :] = 1.0
        if B > 1:
            pre_update_multiplier[1:, :, :] = cum_after_update[:-1, :, :]

        mean_norm_sq_before_update = (
            pre_update_multiplier * norm_sq[None, :, :]
        ).mean(axis=2)

        mean_residual_sq_block = (
            residual_coeff[None, :] * mean_norm_sq_before_update
        )

        # Theorem-comparable empirical quantity:
        #
        # E ||T x_{N_K} - x_{N_K}||^2
        #
        # where P(N_K = k) is proportional to
        # Lambda_{k+1} lambda_k (1-lambda_k).
        block_cum_weighted_residual = np.cumsum(
            weights_block[:, None] * mean_residual_sq_block,
            axis=0,
        )
        block_cum_weight = np.cumsum(weights_block)

        while plot_ptr < len(plot_indices) and plot_indices[plot_ptr] < end:
            local = plot_indices[plot_ptr] - start

            empirical_random_iter_residual_sq[:, plot_ptr] = (
                cum_weighted_residual
                + block_cum_weighted_residual[local, :]
            ) / (
                cum_weight
                + block_cum_weight[local]
            )

            plot_ptr += 1

        cum_weighted_residual += block_cum_weighted_residual[-1, :]
        cum_weight += block_cum_weight[-1]

        # Advance ||x_k||^2 to the end of the block.
        norm_sq *= cum_after_update[-1, :, :]

    return {
        "horizons": horizons,
        "angles": angles,
        "angle_degrees": angle_degrees,
        "empirical_random_iter_residual_sq": empirical_random_iter_residual_sq,
        "paper_bound_sq": paper_bound_sq[plot_indices],
    }


def plot_all_empirical_against_bound(results):
    horizons = results["horizons"]
    angles_deg = results["angle_degrees"]
    empirical = results["empirical_random_iter_residual_sq"]
    bound = results["paper_bound_sq"]

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=360.0)

    for j in range(empirical.shape[0]):
        ax.loglog(
            horizons,
            empirical[j],
            linewidth=0.9,
            alpha=0.35,
            color=cmap(norm(angles_deg[j])),
        )

    ax.loglog(
        horizons,
        bound,
        linewidth=2.5,
        linestyle="--",
        color="black",
        label="Worst-case deterministic upper bound",
    )

    # Dummy line for empirical legend entry.
    ax.loglog(
        [],
        [],
        linewidth=1.2,
        alpha=0.5,
        color=cmap(norm(180.0)),
        label="Empirical curves, 100 random rotations",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Rotation angle, degrees")

    ax.set_xlabel("Iteration / horizon K")
    ax.set_ylabel(r"Squared residual")
    ax.set_title(
        "SKM empirical squared residuals for 100 random rotations\n"
        r"versus the deterministic worst-case bound"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = simulate_many_random_rotations(
        n_angles=100,
        K=500_000,
        n_runs=10,
        x0=np.array([1.0, 1.0]),
        lambda0=0.30,
        a=0.75,
        seed=123,
        n_plot_points=1500,
        block_size=500,
    )

    plot_all_empirical_against_bound(results)