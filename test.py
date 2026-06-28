import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_stepsizes(K, lambda0=0.30, a=0.75):
    k = np.arange(K)
    lambdas = lambda0 / (k + 1.0) ** a
    return lambdas


def make_plot_indices(K, n_plot_points=1500):
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
    block_size=500,
):
    rng = np.random.default_rng(seed)

    angles = np.linspace(0, np.pi, n_angles)
    angle_degrees = np.degrees(angles)

    # Compute lambda (step-sizes)
    lambdas = lambda0 / (np.arange(K) + 1.0) ** a

    # Compute eta = lambda * (1 - lambda)
    etas = lambdas * (1.0 - lambdas)
    cum_eta = np.cumsum(etas)

    # Compute Lambda (Auxiliary variable)
    Lambda = np.empty(K + 1)
    Lambda[0] = 1
    Lambda[1:] = np.cumprod(1 / (1 + 8 * lambdas ** 2))

    # Compute weights (both for upper bound and for probability density)
    weights = Lambda[1:] * etas
    cum_weights = np.cumsum(weights)

    # Compute initial distance squared
    dist0_sq = np.linalg.norm(x0 - np.zeros_like(x0)) ** 2

    # Since both I and R_theta fix 0, sigma_*^2 = 0.
    theory_bound = dist0_sq / cum_weights

    # For T = 0.5(I + R_theta), ||T x - x||^2 = 0.5(1 - cos(theta)) ||x||^2.
    residual_coeff = 0.5 * (1.0 - np.cos(angles))
    one_minus_cos = 1.0 - np.cos(angles)

    # Track |x_k|^2 for each angle and stochastic run
    norm_sq = np.full((n_angles, n_runs), dist0_sq, dtype=float)

    plot_indices = make_plot_indices(K, K // block_size)
    empirical_random_iter_residual = np.empty((n_angles, len(plot_indices)))

    cum_weighted_residual = np.zeros(n_angles, dtype=float)
    cum_weight = 0.0


    for idx, start in tqdm(enumerate(plot_indices)):
        end = plot_indices[idx + 1] if idx + 1 < len(plot_indices) else K
        B = end - start

        eta_block = etas[start:end]
        weights_block = weights[start:end]

        # If rotation is sampled at iteration k, then
        # ||x_{k+1}||^2 = [1 - 2 lambda_k(1-lambda_k)(1-cos(theta))] ||x_k||^2.
        rotation_factors = (1.0 - 2.0 * eta_block[:, None] * one_minus_cos[None, :])

        # Bernoulli choices: True means use rotation, False means use identity.
        use_rotation = rng.random((B, n_angles, n_runs)) < 0.5

        # Multipliers to execute iterations: rotation_factor if rotation, else 1 (do nothing)
        multipliers = np.where(use_rotation, rotation_factors[:, :, None], 1.0)

        cum_after_update = np.cumprod(multipliers, axis=0)

        # Residual at iteration k is evaluated before the k-th update.
        pre_update_multiplier = np.empty_like(cum_after_update)
        pre_update_multiplier[0, :, :] = 1.0
        if B > 1: pre_update_multiplier[1:, :, :] = cum_after_update[:-1, :, :]

        mean_norm_sq_before_update = (pre_update_multiplier * norm_sq[None, :, :]).mean(axis=2)

        mean_residual_sq_block = residual_coeff[None, :] * mean_norm_sq_before_update

        # Empirically compute E{|T x_{N_K} - x_{N_K}|^2} over the block
        block_cum_weighted_residual = np.cumsum(weights_block[:, None] * mean_residual_sq_block, axis=0)
        block_cum_weight = np.cumsum(weights_block)

        # Compute the empirical random residual at end of block
        empirical_random_iter_residual[:, idx] = (
            cum_weighted_residual + block_cum_weighted_residual[-1, :]
        ) / (cum_weight + block_cum_weight[-1])

        # Update cumulative weighted residuals and the cumulative weights
        cum_weighted_residual += block_cum_weighted_residual[-1, :]
        cum_weight += block_cum_weight[-1]

        # Update |x_k|^2 to the end of the block
        norm_sq *= cum_after_update[-1, :, :]

    return {
        "plot_indices": plot_indices + 1,
        "angles": angles,
        "angle_degrees": angle_degrees,
        "empirical_random_iter_residual_sq": empirical_random_iter_residual,
        "theory_bound": theory_bound[plot_indices],
    }


def plot_all_empirical_against_bound(results1, results2):
    plot_indices1 = results1["plot_indices"]
    angles_deg1 = results1["angle_degrees"]
    empirical1 = results1["empirical_random_iter_residual_sq"]
    bound1 = results1["theory_bound"]

    plot_indices2 = results2["plot_indices"]
    angles_deg2 = results2["angle_degrees"]
    empirical2 = results2["empirical_random_iter_residual_sq"]
    bound2 = results2["theory_bound"]

    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=360.0)

    # Plot 1

    for j in range(empirical1.shape[0]):
        axs[0].loglog(
            plot_indices1,
            empirical1[j],
            linewidth=0.9,
            alpha=0.55,
            color=cmap(norm(angles_deg1[j])),
        )

    axs[0].loglog(
        plot_indices1,
        bound1,
        linewidth=2.5,
        linestyle="--",
        color="black",
        label="Theorem 2.9",
    )

    # Dummy line for empirical legend entry.
    axs[0].loglog(
        [],
        [],
        linewidth=1.2,
        alpha=0.5,
        color=cmap(norm(180.0)),
        label="Empirical Curves",
    )

    # Plot 2

    for j in range(empirical2.shape[0]):
        axs[1].loglog(
            plot_indices2,
            empirical2[j],
            linewidth=0.9,
            alpha=0.55,
            color=cmap(norm(angles_deg2[j])),
        )

    axs[1].loglog(
        plot_indices2,
        bound2,
        linewidth=2.5,
        linestyle="--",
        color="black",
        label="Theorem 2.9",
    )

    # Dummy line for empirical legend entry.
    axs[1].loglog(
        [],
        [],
        linewidth=1.2,
        alpha=0.5,
        color=cmap(norm(180.0)),
        label="Empirical Curves",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[1])
    cbar.set_label("Rotation Angle (degrees)")

    axs[0].set_xlabel("Iteration Counter ($k$)")
    axs[0].set_ylabel(r"Empirical Random Squared Residual")
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend()

    axs[1].set_xlabel("Iteration Counter ($k$)")
    axs[1].grid(True, which="both", alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results1 = simulate_many_random_rotations(
        n_angles=36,
        K=500_0,
        n_runs=10,
        x0=np.array([1.0, 1.0]),
        lambda0=0.2,
        a=1,
        seed=123,
        block_size=5_0,
    )

    results2 = simulate_many_random_rotations(
        n_angles=36,
        K=500_0,
        n_runs=10,
        x0=np.array([1.0, 1.0]),
        lambda0=0.2,
        a=0.51,
        seed=123,
        block_size=5_0,
    )

    plot_all_empirical_against_bound(results1, results2)