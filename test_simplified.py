import numpy as np
import matplotlib.pyplot as plt


X0 = np.array([1.0, 1.0])
SEED = 123
N_PLOT_POINTS = 1000


def make_stepsizes(K, lambda0=0.30, a=0.75):
    k = np.arange(K)
    return lambda0 / (k + 1.0) ** a


def make_plot_indices(K, n_plot_points=N_PLOT_POINTS):
    early = np.arange(min(10, K))
    log_part = np.logspace(0, np.log10(K), num=n_plot_points).astype(int) - 1
    plot_indices = np.unique(np.concatenate([early, log_part]))
    return plot_indices[(0 <= plot_indices) & (plot_indices < K)]


def simulate_many_random_rotations(
    n_angles=100,
    K=500_000,
    n_runs=100,
    lambda0=0.30,
    a=0.75,
):
    rng = np.random.default_rng(SEED)

    # One experiment = two random rotations.
    # The first angle is in (0, pi), the second in (pi, 2pi).
    angles = np.column_stack([
        rng.uniform(1e-3, np.pi - 1e-3, size=n_angles),
        rng.uniform(np.pi + 1e-3, 2.0 * np.pi - 1e-3, size=n_angles),
    ])
    angle_degrees = np.degrees(angles)

    # Constants used by the algorithm and by the deterministic bound.
    lambdas = make_stepsizes(K, lambda0=lambda0, a=a)
    etas = lambdas * (1.0 - lambdas)

    Lambda = np.empty(K + 1)
    Lambda[0] = 1.0
    Lambda[1:] = np.cumprod(1.0 / (1.0 + 8.0 * lambdas**2))
    weights = Lambda[1:] * etas

    dist0_sq = float(np.dot(X0, X0))
    theory_bound = dist0_sq / np.cumsum(weights)

    plot_indices = make_plot_indices(K)
    horizons = plot_indices + 1

    # For T = 0.5(R_a + R_b), ||T x - x||^2 = residual_coeff * ||x||^2.
    average_rotation = 0.5 * (np.exp(1j * angles[:, 0]) + np.exp(1j * angles[:, 1]))
    residual_coeff = np.abs(average_rotation - 1.0) ** 2

    # If R_theta is selected at iteration k, then
    # ||x_{k+1}||^2 = [1 - 2 lambda_k(1-lambda_k)(1-cos(theta))] ||x_k||^2.
    one_minus_cos = 1.0 - np.cos(angles)

    empirical_running_min_residual_sq = np.zeros((n_angles, len(plot_indices)))

    for _ in range(n_runs):
        norm_sq = np.full(n_angles, dist0_sq)
        running_min_residual_sq = np.full(n_angles, np.inf)
        plot_ptr = 0

        for k in range(K):
            residual_sq = residual_coeff * norm_sq
            running_min_residual_sq = np.minimum(running_min_residual_sq, residual_sq)

            if plot_ptr < len(plot_indices) and k == plot_indices[plot_ptr]:
                empirical_running_min_residual_sq[:, plot_ptr] += running_min_residual_sq
                plot_ptr += 1

            use_second_rotation = rng.random(n_angles) < 0.5
            selected_one_minus_cos = np.where(
                use_second_rotation,
                one_minus_cos[:, 1],
                one_minus_cos[:, 0],
            )
            norm_sq *= 1.0 - 2.0 * etas[k] * selected_one_minus_cos

    empirical_running_min_residual_sq /= n_runs

    return {
        "horizons": horizons,
        "angles": angles,
        "angle_degrees": angle_degrees,
        "empirical_random_iter_residual_sq": empirical_running_min_residual_sq,
        "theory_bound": theory_bound[plot_indices],
    }


def plot_all_empirical_against_bound(results):
    horizons = results["horizons"]
    angle_degrees = results["angle_degrees"]
    empirical = results["empirical_random_iter_residual_sq"]
    bound = results["theory_bound"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # One color per angle pair, using the mean angle only for visualization.
    mean_angle_degrees = angle_degrees.mean(axis=1)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=360.0)

    for j in range(empirical.shape[0]):
        ax.loglog(
            horizons,
            empirical[j],
            linewidth=0.9,
            alpha=0.35,
            color=cmap(norm(mean_angle_degrees[j])),
        )

    ax.loglog(
        horizons,
        bound,
        linewidth=2.5,
        linestyle="--",
        color="black",
        label="Worst-case deterministic upper bound",
    )

    ax.loglog(
        [],
        [],
        linewidth=1.2,
        alpha=0.5,
        color=cmap(norm(180.0)),
        label=f"Empirical curves, {empirical.shape[0]} random rotation pairs",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Mean rotation angle, degrees")

    ax.set_xlabel("Iteration / horizon K")
    ax.set_ylabel(r"Squared residual")
    ax.set_title(
        "SKM empirical squared residuals for random rotation pairs\n"
        r"versus the deterministic worst-case bound"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = simulate_many_random_rotations(
        n_angles=10,
        K=5000,
        n_runs=100,
        lambda0=0.30,
        a=0.75,
    )

    plot_all_empirical_against_bound(results)
