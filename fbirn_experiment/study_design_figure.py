"""Publication-style study design / workflow figure (no data required)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def plot_study_design(out_path: Path | str, *, dpi: int = 300) -> Path:
    """Draw overall workflow: inputs → FC → multiverse forks → H1/H2/H3 → outputs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 8.2), facecolor="white")
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    def box(
        xy: tuple[float, float],
        w: float,
        h: float,
        text: str,
        *,
        fc: str,
        ec: str = "#333333",
        fs: float = 8.4,
        lw: float = 0.9,
    ) -> FancyBboxPatch:
        x, y = xy
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            mutation_aspect=0.35,
        )
        ax.add_patch(p)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs, wrap=True,
            color="#1a1a1a", linespacing=1.15,
        )
        return p

    def arrow(
        p0: tuple[float, float],
        p1: tuple[float, float],
        *,
        rad: float = 0.0,
        color: str = "#444444",
        lw: float = 1.0,
    ) -> None:
        arr = FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
        ax.add_patch(arr)

    # Title
    ax.text(
        5.75, 7.85,
        "FBIRN ICN functional connectivity: study design and multiverse workflow",
        ha="center", va="top", fontsize=12.5, fontweight="600", color="#0d2137",
    )

    # Row 1 — inputs
    w_in, h_in = 2.35, 0.95
    y_in = 6.55
    box((0.35, y_in), w_in, h_in, "ICN time\ncourses", fc="#eceff4", ec="#4c566a")
    box((2.85, y_in), w_in, h_in, "Group labels\n(SCZ / HC)", fc="#eceff4", ec="#4c566a")
    box((5.35, y_in), w_in, h_in, "Confounds\n(age, sex, site, …)", fc="#eceff4", ec="#4c566a")
    box((7.85, y_in), w_in, h_in, "ICN → domain\nmask (D5)", fc="#eceff4", ec="#4c566a")

    y_fc = 5.35
    h_fc = 1.05
    box(
        (0.35, y_fc), 10.8, h_fc,
        "Functional connectivity (D1: Pearson z, Spearman, partial correlation, mutual information)\n"
        "→ subject × edge matrix; optional confound handling (D2: none, OLS, ComBat)",
        fc="#d8e8f5", ec="#2c5f8d", fs=8.8,
    )
    for cx in (1.525, 4.025, 6.525, 9.025):
        arrow((cx, y_in), (cx, y_fc + h_fc + 0.02), rad=0.0)

    # Multiverse strip
    y_mv = 3.95
    h_mv = 1.05
    box(
        (0.35, y_mv), 10.8, h_mv,
        "Multiverse specification (full factorial example: 4×3×5×4×2 = 480)\n"
        "D1 connectivity × D2 confound × D3 reduction × D4 classifier × D5 domain granularity",
        fc="#e8f2e4", ec="#3d6b4a", fs=8.8,
    )
    arrow((5.75, y_fc), (5.75, y_mv + h_mv), rad=0.0)

    # Three hypotheses
    y_h = 2.35
    h_h = 1.35
    w_h = 3.35
    x1, x2, x3 = 0.35, 4.1, 7.85
    box(
        (x1, y_h), w_h, h_h,
        "H1 — Classification\n"
        "Nested CV: all edges vs latent\n"
        "(D3, D4) → ΔAUC",
        fc="#fde8e8", ec="#8b3a3a", fs=8.2,
    )
    box(
        (x2, y_h), w_h, h_h,
        "H2 — Effect sizes\n"
        "Per-edge Cohen's d; between vs within\n"
        "domain → label permutation p",
        fc="#e8f8ec", ec="#2d6b3f", fs=8.2,
    )
    box(
        (x3, y_h), w_h, h_h,
        "H3 — Loading structure\n"
        "Latent |loadings|: between vs within\n"
        "→ Wilcoxon p across factors",
        fc="#f0e8f8", ec="#5c3d7a", fs=8.2,
    )
    arrow((2.0, y_mv), (2.0, y_h + h_h), rad=0.0)
    arrow((5.75, y_mv), (5.75, y_h + h_h), rad=0.0)
    arrow((9.5, y_mv), (9.5, y_h + h_h), rad=0.0)

    # Outputs
    y_out = 0.45
    h_out = 1.55
    box(
        (0.35, y_out), 10.8, h_out,
        "Outputs\n"
        "• Single pipeline: `cli run` → nested CV, H2 permutation figure, H3 loadings\n"
        "• Multiverse: `cli multiverse` → multiverse_results.csv; specification curves, forests,\n"
        "  robustness tables, joint binomial tests (Steegen / Simonsohn-style summaries)",
        fc="#fff5e0", ec="#b8860b", fs=8.0,
    )
    arrow((2.0, y_h), (2.0, y_out + h_out), rad=0.0)
    arrow((5.75, y_h), (5.75, y_out + h_out), rad=0.0)
    arrow((9.5, y_h), (9.5, y_out + h_out), rad=0.0)

    ax.text(
        5.75, 0.12,
        "Arrows indicate data / analytic flow (not all cross-links shown).",
        ha="center", va="bottom", fontsize=7.2, color="#666666", style="italic",
    )

    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")
    return out_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Write study design workflow PNG.")
    p.add_argument(
        "-o", "--out",
        type=Path,
        default=Path("figures/study_design_workflow.png"),
        help="Output PNG path.",
    )
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    plot_study_design(args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
