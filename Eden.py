import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde


class FastEdenScorer:
    """Scores the similarity between two datasets via the overlap of their
    likelihood-density annuli (Eden score).

    New in this version:
      * bw_method is passed straight through to scipy's gaussian_kde, so the
        KDE bandwidth can be swept ('scott', 'silverman', or a scalar factor).
      * Per-annulus statistics (Jaccard per level, counts of real/syn data
        points, and Monte Carlo intersection/union sizes) are stored in
        GraphicResult so plots and analyses can reuse them verbatim.
    """

    def __init__(
        self,
        n_of_levels: int = 5,
        thresh: float = 0.05,
        size: int = 50000,
        seed: int | None = None,
        debug: bool = False,
        cut: int = 5,
        edge_exp: float = 0.05,
        bw_method: str | float | None = None,
    ) -> None:
        self.n_of_levels = n_of_levels
        self.thresh = thresh
        self.size = size
        self.seed = seed
        self.debug = debug
        self.cut = cut
        self.edge_exp = edge_exp
        self.bw_method = bw_method
        self.make_quantiles()

    def make_quantiles(self) -> None:
        array_of_quantiles = np.arange(
            self.thresh, 1, (1 - self.thresh) / (self.n_of_levels - 1)
        )
        array_of_quantiles = np.concatenate(([0], array_of_quantiles))
        self.array_of_quantiles = np.concatenate((array_of_quantiles, [1]))

    class DataDist:
        def __init__(
            self,
            data: pd.DataFrame,
            sampling_size: int,
            bw_method: str | float | None = None,
            seed=None,
        ) -> None:
            self.original_data = data
            self.data_no_na = self.original_data.dropna()
            # NOTE: bw_method now honors 'scott', 'silverman', or a scalar
            self.kernel = gaussian_kde(
                self.data_no_na.T, bw_method=bw_method
            )
            if seed is not None:
                np.random.seed(seed)
            self.resample = self.kernel.resample(size=sampling_size)
            self.likelihoods = self.kernel.evaluate(self.resample)
            # likelihood of the *actual* (non-resampled) data points, used for
            # the "number of data points in each annulus" diagnostic
            self.likelihoods_of_data = self.kernel.evaluate(
                self.data_no_na.to_numpy().T
            )

        def make_bins(self, quantiles: np.ndarray) -> None:
            _, self.likelihoods_bins = pd.qcut(
                self.likelihoods, quantiles, retbins=True, duplicates='drop'
            )

        def make_contours(self) -> None:
            self.outermost_contour = self.resample.T[
                np.isclose(
                    self.likelihoods,
                    self.likelihoods_bins[1],
                    atol=self.likelihoods_bins[1] / 100,
                )
            ]
            self.outermost_contour_min = np.array(
                [
                    np.min(self.outermost_contour[:, 0]),
                    np.min(self.outermost_contour[:, 1]),
                ]
            )
            self.outermost_contour_max = np.array(
                [
                    np.max(self.outermost_contour[:, 0]),
                    np.max(self.outermost_contour[:, 1]),
                ]
            )

        def make_uniform_sample_likelihoods(self, uniform_sample: np.ndarray) -> None:
            self.likelihoods_of_uniform_sample = self.kernel.evaluate(
                uniform_sample.T
            )

        # legacy (mis-spelled) alias, so old callers keep working
        def make_uniform_sammple_likelihoods(self, uniform_sample):
            self.make_uniform_sample_likelihoods(uniform_sample)

    class GraphicResult:
        def __init__(
            self,
            Eden_score: np.float64,
            outermost_contour_min: np.ndarray,
            outermost_contour_max: np.ndarray,
            real_likelihoods_bins: np.ndarray,
            syn_likelihoods_bins: np.ndarray,
            kernel_real: gaussian_kde,
            kernel_syn: gaussian_kde,
            jaccard_scores: list | None = None,
            real_annulus_counts: list | None = None,
            syn_annulus_counts: list | None = None,
            intersection_counts: list | None = None,
            union_counts: list | None = None,
            n_real_points: int | None = None,
            n_syn_points: int | None = None,
            bw_method: str | float | None = None,
        ) -> None:
            self.EdenScore = Eden_score
            self.outermost_contour_min = outermost_contour_min
            self.outermost_contour_max = outermost_contour_max
            self.real_likelihoods_bins = real_likelihoods_bins
            self.syn_likelihoods_bins = syn_likelihoods_bins
            self.kernel_real = kernel_real
            self.kernel_syn = kernel_syn
            # --- new diagnostics (all optional, backwards compatible) ---
            self.jaccard_scores = jaccard_scores or []
            self.real_annulus_counts = real_annulus_counts or []
            self.syn_annulus_counts = syn_annulus_counts or []
            self.intersection_counts = intersection_counts or []
            self.union_counts = union_counts or []
            self.n_real_points = n_real_points
            self.n_syn_points = n_syn_points
            self.bw_method = bw_method

    def score(
        self,
        real_data: pd.DataFrame,
        syn_data: pd.DataFrame,
        return_more: bool = False,
    ) -> GraphicResult | np.float64:
        if self.seed is not None:  # fixed: falsy seed=0 used to be ignored
            np.random.seed(self.seed)

        real = self.DataDist(real_data, self.size, self.bw_method, seed=self.seed)
        syn = self.DataDist(syn_data, self.size, self.bw_method, seed=self.seed)

        # Fail loudly on degenerate configurations instead of silently
        # producing garbage bounds. The caller (parallelized_function_for_pairs)
        # already catches exceptions and records NaN.
        for dist in (real, syn):
            dist.make_bins(self.array_of_quantiles)
            dist.make_contours()

        outermost_contour_min = np.minimum(
            real.outermost_contour_min, syn.outermost_contour_min
        )
        outermost_contour_max = np.maximum(
            real.outermost_contour_max, syn.outermost_contour_max
        )

        outermost_contour_range = outermost_contour_max - outermost_contour_min
        low_edge, high_edge = (
            outermost_contour_min - self.edge_exp * outermost_contour_range,
            outermost_contour_max + self.edge_exp * outermost_contour_range,
        )
        uniform_sample = np.random.uniform(
            low=low_edge, high=high_edge, size=(self.size, real_data.shape[1])
        )
        for dist in (real, syn):
            dist.make_uniform_sample_likelihoods(uniform_sample)

        jaccard_scores = []
        real_annulus_counts, syn_annulus_counts = [], []
        intersection_counts, union_counts = [], []

        for i in range(self.n_of_levels - 1):
            lower_real_bin_edge, upper_real_bin_edge = (
                real.likelihoods_bins[i + 1],
                real.likelihoods_bins[i + 2],
            )
            lower_syn_bin_edge, upper_syn_bin_edge = (
                syn.likelihoods_bins[i + 1],
                syn.likelihoods_bins[i + 2],
            )

            in_real_annulus = (
                (lower_real_bin_edge <= real.likelihoods_of_uniform_sample)
                & (real.likelihoods_of_uniform_sample <= upper_real_bin_edge)
            )
            in_syn_annulus = (
                (lower_syn_bin_edge <= syn.likelihoods_of_uniform_sample)
                & (syn.likelihoods_of_uniform_sample <= upper_syn_bin_edge)
            )

            union_criterion = in_real_annulus | in_syn_annulus
            intersection_criterion = in_real_annulus & in_syn_annulus

            union = uniform_sample[union_criterion]
            intersection = uniform_sample[intersection_criterion]

            if len(union) == 0:  # can happen at extreme thresh/level settings
                jaccard = np.nan
            else:
                jaccard = len(intersection) / len(union)

            jaccard_scores.append(jaccard)
            union_counts.append(len(union))
            intersection_counts.append(len(intersection))

            # actual (non-resampled) data points falling in each annulus
            real_annulus_counts.append(
                int(
                    (
                        (lower_real_bin_edge <= real.likelihoods_of_data)
                        & (real.likelihoods_of_data <= upper_real_bin_edge)
                    ).sum()
                )
            )
            syn_annulus_counts.append(
                int(
                    (
                        (lower_syn_bin_edge <= syn.likelihoods_of_data)
                        & (syn.likelihoods_of_data <= upper_syn_bin_edge)
                    ).sum()
                )
            )

        Eden_score = np.float64(np.nanmean(jaccard_scores))

        if return_more:
            return self.GraphicResult(
                Eden_score,
                outermost_contour_min,
                outermost_contour_max,
                real.likelihoods_bins,
                syn.likelihoods_bins,
                real.kernel,
                syn.kernel,
                jaccard_scores=jaccard_scores,
                real_annulus_counts=real_annulus_counts,
                syn_annulus_counts=syn_annulus_counts,
                intersection_counts=intersection_counts,
                union_counts=union_counts,
                n_real_points=len(real.data_no_na),
                n_syn_points=len(syn.data_no_na),
                bw_method=self.bw_method,
            )
        return Eden_score

def plot_eden_contours(
    result: FastEdenScorer.GraphicResult,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (7, 6),
    real_color: str = "tab:blue",
    synthetic_color: str = "tab:orange",
    real_pts: np.ndarray | pd.DataFrame | None = None,
    syn_pts: np.ndarray | pd.DataFrame | None = None,
    scatter_alpha: float = 0.25,
    save_path: str | Path | None = None,
) -> matplotlib.axes.Axes:
    """Overlays the KDE contours of the real data on those of the synthetic
    data, using the *exact same* KDE kernels, likelihood-bin edges, and
    outermost-contour bounds that produced the Eden score in `result`.

    The title reports the Eden score and, for each annulus, the number of
    real/synthetic data points falling in it, plus the Monte Carlo
    intersection/union sizes the Jaccard is computed from.
    Optionally scatters the raw real/synthetic points underneath.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    om_min, om_max = result.outermost_contour_min, result.outermost_contour_max
    x_grid = np.linspace(om_min[0], om_max[0], 100)
    y_grid = np.linspace(om_min[1], om_max[1], 100)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

    z_real = result.kernel_real(positions).reshape(x_mesh.shape)
    z_syn = result.kernel_syn(positions).reshape(x_mesh.shape)

    # contour levels == the same likelihood-bin edges used for scoring
    # (index 0 is 0-density / below-threshold tail mass, so start at 1)
    levels_real = result.real_likelihoods_bins[1:]
    levels_syn = result.syn_likelihoods_bins[1:]
    n_levels = len(levels_real)

    # # optional raw data underneath the contours
    # if real_pts is not None:
    #     arr = real_pts.to_numpy() if isinstance(real_pts, pd.DataFrame) else real_pts
    #     ax.scatter(arr[:, 0], arr[:, 1], c=real_color, s=6, alpha=scatter_alpha,
    #                edgecolors="none", zorder=1)
    # if syn_pts is not None:
    #     arr = syn_pts.to_numpy() if isinstance(syn_pts, pd.DataFrame) else syn_pts
    #     ax.scatter(arr[:, 0], arr[:, 1], c=synthetic_color, s=6,
    #                alpha=scatter_alpha, edgecolors="none", zorder=1)

    # one contour call per level so inner (higher-likelihood) levels stand out
    for k in range(n_levels):
        lw = 1.0 + 0.6 * (k / max(n_levels - 1, 1))
        ax.contour(x_mesh, y_mesh, z_real, levels=[levels_real[k]],
                   colors=real_color, linewidths=lw, zorder=2)
        ax.contour(x_mesh, y_mesh, z_syn, levels=[levels_syn[k]],
                   colors=synthetic_color, linewidths=lw,
                   linestyles="--" if k > 0 else "-", zorder=2)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # legend proxies
    ax.plot([], [], color=real_color, label="real KDE contours")
    ax.plot([], [], color=synthetic_color, linestyle="--",
            label="synthetic KDE contours")
    ax.legend(loc="upper right", fontsize="small")

    # ---- title: Eden score + per-annulus diagnostics ----
    bw_repr = (result.bw_method if result.bw_method is not None else "scott (default)")
    title_lines = [
        f"Eden score = {result.EdenScore:.3f}   "
        f"(bw={bw_repr}, n_real={result.n_real_points}, n_syn={result.n_syn_points})"
    ]
    for i in range(len(result.jaccard_scores)):
        j = result.jaccard_scores[i]
        j_str = f"{j:.2f}" if not np.isnan(j) else "nan"
        title_lines.append(
            f"Annulus {i + 1}: "
            f"N_real={result.real_annulus_counts[i]}, "
            f"N_syn={result.syn_annulus_counts[i]}   "
            f"|∩|={result.intersection_counts[i]}, "
            f"|∪|={result.union_counts[i]}  (J={j_str})"
        )
    ax.set_title("\n".join(title_lines), fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax