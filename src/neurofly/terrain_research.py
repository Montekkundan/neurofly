from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy.stats import mannwhitneyu

from neurofly.terrain_benchmark import (
    BenchmarkConfig,
    build_spawn_positions,
    ensure_output_dirs,
    load_config,
    run_single_controller,
)

app = typer.Typer(add_completion=False)


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(float(np.mean(sample)))
    boot_arr = np.array(boot)
    return (
        float(np.mean(values)),
        float(np.quantile(boot_arr, alpha / 2)),
        float(np.quantile(boot_arr, 1 - alpha / 2)),
    )


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    comparisons = np.subtract.outer(x, y)
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / (len(x) * len(y)))


def summarize_with_ci(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = results_df.groupby(["terrain", "controller"], dropna=False)
    for (terrain, controller), group in grouped:
        speed_mean, speed_lo, speed_hi = bootstrap_mean_ci(group["average_speed"].to_numpy())
        dist_mean, dist_lo, dist_hi = bootstrap_mean_ci(
            group["distance_before_failure"].to_numpy(), seed=1
        )
        success_mean, success_lo, success_hi = bootstrap_mean_ci(
            group["success_rate"].to_numpy(), seed=2
        )
        rows.append(
            {
                "terrain": terrain,
                "controller": controller,
                "n": len(group),
                "speed_mean": speed_mean,
                "speed_ci_low": speed_lo,
                "speed_ci_high": speed_hi,
                "distance_mean": dist_mean,
                "distance_ci_low": dist_lo,
                "distance_ci_high": dist_hi,
                "success_mean": success_mean,
                "success_ci_low": success_lo,
                "success_ci_high": success_hi,
                "stumble_mean": float(group["stumble_count"].mean()),
                "slip_mean": float(group["slip_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def pairwise_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for terrain, terrain_group in results_df.groupby("terrain", dropna=False):
        hybrid = terrain_group[terrain_group["controller"] == "hybrid"]
        for baseline_name in ("cpg", "rule_based"):
            baseline = terrain_group[terrain_group["controller"] == baseline_name]
            for metric in ("average_speed", "distance_before_failure"):
                x = baseline[metric].to_numpy()
                y = hybrid[metric].to_numpy()
                stat, pvalue = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
                rows.append(
                    {
                        "terrain": terrain,
                        "metric": metric,
                        "baseline": baseline_name,
                        "hybrid_mean": float(np.mean(y)),
                        "baseline_mean": float(np.mean(x)),
                        "pvalue": float(pvalue),
                        "u_stat": float(stat),
                        "cliffs_delta": cliffs_delta(y, x),
                    }
                )
    return pd.DataFrame(rows)


def plot_ci_bars(summary_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    terrain_names = list(summary_df["terrain"].drop_duplicates())
    controller_names = list(summary_df["controller"].drop_duplicates())
    x = np.arange(len(controller_names))
    width = 0.22
    colors = {"flat": "#2563eb", "rough": "#ea580c", "blocks": "#7c3aed"}

    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    for idx, terrain in enumerate(terrain_names):
        subset = summary_df[summary_df["terrain"] == terrain].set_index("controller")
        means = [float(subset.loc[name, f"{metric}_mean"]) for name in controller_names]
        low = [float(subset.loc[name, f"{metric}_mean"] - subset.loc[name, f"{metric}_ci_low"]) for name in controller_names]
        high = [float(subset.loc[name, f"{metric}_ci_high"] - subset.loc[name, f"{metric}_mean"]) for name in controller_names]
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            yerr=np.array([low, high]),
            label=terrain,
            color=colors[terrain],
            capsize=4,
        )
    ax.set_xticks(x, [name.replace("_", " ").title() for name in controller_names], rotation=15)
    ax.set_title(f"{metric.replace('_', ' ').title()} With 95% Bootstrap CI")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_research_report(
    summary_df: pd.DataFrame, pairwise_df: pd.DataFrame, results_df: pd.DataFrame, output_path: Path
) -> None:
    lines = [
        "# Terrain Research Report",
        "",
        "This report upgrades the terrain benchmark into a small research-style study.",
        "",
        "Briefly: A tests how existing fly locomotion controllers behave as terrain gets harder.",
        "",
        "Neuroscience goal: study how neural-inspired gait controllers cope with increasingly difficult walking surfaces.",
        "",
        "Computational goal: compare controllers under matched seeds, terrain conditions, and metrics using bootstrap confidence intervals and nonparametric tests.",
        "",
        "## Research Question",
        "",
        "Do the NeuroMechFly locomotion controllers behave differently as terrain difficulty increases, and does the hybrid controller outperform simpler baselines under more difficult conditions?",
        "",
        "## Experimental Design",
        "",
        "- Controllers: `cpg`, `rule_based`, `hybrid`",
        "- Terrains: `flat`, `rough`, `blocks`",
        f"- Trials per condition: `{int(results_df.groupby(['terrain', 'controller']).size().min())}`",
        "- Outputs: per-run metrics, bootstrap confidence intervals, pairwise nonparametric comparisons",
        "",
        "## Summary Table",
        "",
        summary_df.to_csv(index=False).strip(),
        "",
        "## Pairwise Hybrid Comparisons",
        "",
        pairwise_df.to_csv(index=False).strip(),
        "",
        "## Interpretation",
        "",
    ]
    for terrain, terrain_group in pairwise_df.groupby("terrain", dropna=False):
        best = summary_df[summary_df["terrain"] == terrain].sort_values(
            ["speed_mean", "distance_mean"], ascending=False
        ).iloc[0]
        lines.append(
            f"- On `{terrain}`, the highest mean-speed controller was `{best['controller']}` "
            f"(speed={best['speed_mean']:.3f}, distance={best['distance_mean']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This is a research-style benchmark, not a novel controller paper.",
            "- The conclusions are only as strong as the task design, seed count, and metric definitions.",
            "- If all controllers finish every trial, interpret the result as speed/stability evidence rather than failure-robustness evidence.",
            "- Stronger claims would require larger sweeps, better calibration of stumble/slip metrics, and more terrains.",
        ]
    )
    output_path.write_text("\n".join(lines))


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/terrain_benchmark.toml"), exists=True, dir_okay=False
    ),
    output_dir: Path = typer.Option(Path("outputs/terrain_research"), "--output-dir"),
    trials: int = typer.Option(6, "--trials"),
    duration_s: float = typer.Option(0.05, "--duration-s"),
    render: bool = typer.Option(False, "--render/--no-render"),
    save_video: bool = typer.Option(False, "--save-video/--no-save-video"),
) -> None:
    config: BenchmarkConfig = load_config(config_path)
    config.output_dir = output_dir
    config.trials = trials
    config.episode_duration_s = duration_s
    config.render = render
    config.save_video = save_video
    config.terrains = ["flat", "rough", "blocks"]
    config.controllers = ["cpg", "rule_based", "hybrid"]

    output_dirs = ensure_output_dirs(config.output_dir)
    spawn_positions = build_spawn_positions(
        config.trials, config.seed, config.spawn_bbox, config.spawn_z
    )

    rows: list[dict[str, object]] = []
    for terrain_name in config.terrains:
        typer.echo(f"Terrain: {terrain_name}")
        for trial_id, spawn_pos in enumerate(spawn_positions):
            for controller_name in config.controllers:
                row = run_single_controller(
                    controller=controller_name,
                    terrain=terrain_name,
                    trial_id=trial_id,
                    spawn_pos=spawn_pos,
                    output_dirs=output_dirs,
                    config=config,
                    render=config.render,
                    save_video=config.save_video,
                )
                rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dirs["base"] / "results.csv", index=False)
    summary_df = summarize_with_ci(results_df)
    pairwise_df = pairwise_stats(results_df)
    summary_df.to_csv(output_dirs["base"] / "summary_with_ci.csv", index=False)
    pairwise_df.to_csv(output_dirs["base"] / "pairwise_stats.csv", index=False)
    plot_ci_bars(summary_df, "speed", output_dirs["plots"] / "speed_with_ci.png")
    plot_ci_bars(summary_df, "distance", output_dirs["plots"] / "distance_with_ci.png")
    write_research_report(
        summary_df, pairwise_df, results_df, output_dirs["base"] / "RESEARCH_REPORT.md"
    )
    typer.echo(f"Research results: {output_dirs['base'] / 'RESEARCH_REPORT.md'}")


if __name__ == "__main__":
    app()
