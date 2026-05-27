from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy.stats import mannwhitneyu

from neurofly_odor.odor_benchmark import (
    OdorConfig,
    build_spawn_conditions,
    ensure_output_dirs,
    load_config,
    make_simulation,
    save_trajectory,
)

app = typer.Typer(add_completion=False)

POLICIES = ("full_odor", "attractive_only", "forward_only")


def scenario_library() -> dict[str, dict[str, object]]:
    return {
        "triplet": {
            "odor_source": np.array([[8.0, 0.0, 1.5], [4.0, -3.0, 1.5], [4.0, 3.0, 1.5]]),
            "peak_odor_intensity": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        },
        "diagonal_choice": {
            "odor_source": np.array([[3.0, 3.0, 1.5], [3.0, -3.0, 1.5]]),
            "peak_odor_intensity": np.eye(2),
        },
        "full_turn": {
            "odor_source": np.array([[-5.0, 0.0, 1.5], [3.0, 3.0, 1.5]]),
            "peak_odor_intensity": np.array([[1.0, 1e-3], [0.0, 1.0]]),
        },
    }


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


def compute_policy_control(
    obs: dict, config: OdorConfig, policy: str
) -> tuple[np.ndarray, float, float]:
    attractive_intensities = np.average(
        obs["odor_intensity"][0, :].reshape(2, 2),
        axis=0,
        weights=config.attractive_palps_antennae_weights,
    )
    aversive_intensities = np.average(
        obs["odor_intensity"][1, :].reshape(2, 2),
        axis=0,
        weights=config.aversive_palps_antennae_weights,
    )
    attractive_mean = float(np.mean(attractive_intensities))
    aversive_mean = float(np.mean(aversive_intensities))

    if policy == "forward_only":
        return np.ones((2,)), attractive_mean, aversive_mean

    attractive_bias = config.attractive_gain * (
        (attractive_intensities[0] - attractive_intensities[1]) / max(attractive_mean, 1e-9)
    )
    aversive_bias = config.aversive_gain * (
        (aversive_intensities[0] - aversive_intensities[1]) / max(aversive_mean, 1e-9)
    )
    if policy == "attractive_only":
        effective_bias = attractive_bias
    elif policy == "full_odor":
        effective_bias = attractive_bias + aversive_bias
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
    control_signal = np.ones((2,))
    side_to_modulate = int(effective_bias_norm > 0)
    control_signal[side_to_modulate] -= np.abs(effective_bias_norm) * 0.8
    return control_signal, attractive_mean, aversive_mean


def run_research_trial(
    config: OdorConfig,
    scenario_name: str,
    policy: str,
    trial_id: int,
    spawn_pos: np.ndarray,
    spawn_heading: float,
    output_dirs: dict[str, Path],
) -> dict[str, object]:
    sim, _camera = make_simulation(config, spawn_pos, spawn_heading, render=False)
    target_pos = config.odor_source[0, :2]
    num_decision_steps = int(config.run_time_s / config.decision_interval)
    physics_steps_per_decision = int(config.decision_interval / sim.timestep)

    obs_hist: list[dict] = []
    attractive_hist: list[float] = []
    aversive_hist: list[float] = []
    obs, _ = sim.reset()
    success = False
    time_to_target_s = config.run_time_s

    for decision_idx in range(num_decision_steps):
        control_signal, attractive_signal, aversive_signal = compute_policy_control(
            obs, config, policy
        )
        for _ in range(physics_steps_per_decision):
            obs, _reward, _terminated, _truncated, _info = sim.step(control_signal)
            obs_hist.append(obs)
            attractive_hist.append(attractive_signal)
            aversive_hist.append(aversive_signal)
        if np.linalg.norm(obs["fly"][0, :2] - target_pos) < config.distance_threshold:
            success = True
            time_to_target_s = (decision_idx + 1) * config.decision_interval
            break

    trajectory_path = (
        output_dirs["trajectories"] / f"{scenario_name}_{policy}_trial_{trial_id:02d}.npz"
    )
    save_trajectory(trajectory_path, obs_hist, attractive_hist, aversive_hist)

    xy = np.array([obs["fly"][0, :2] for obs in obs_hist]) if obs_hist else np.zeros((0, 2))
    path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) >= 2 else 0.0
    elapsed = len(obs_hist) * config.timestep
    average_speed = path_length / elapsed if elapsed > 0 else 0.0
    final_xy = xy[-1] if len(xy) else spawn_pos[:2]

    return {
        "scenario": scenario_name,
        "policy": policy,
        "trial_id": trial_id,
        "success": success,
        "final_distance_to_target": float(np.linalg.norm(final_xy - target_pos)),
        "time_to_target_s": float(time_to_target_s),
        "path_length": path_length,
        "average_speed": average_speed,
        "attractive_signal_mean": float(np.mean(attractive_hist) if attractive_hist else 0.0),
        "aversive_signal_mean": float(np.mean(aversive_hist) if aversive_hist else 0.0),
        "spawn_x": float(spawn_pos[0]),
        "spawn_y": float(spawn_pos[1]),
        "spawn_orientation_z": float(spawn_heading),
        "target_x": float(target_pos[0]),
        "target_y": float(target_pos[1]),
        "trajectory_path": trajectory_path.as_posix(),
    }


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = results_df.groupby(["scenario", "policy"], dropna=False)
    for (scenario, policy), group in grouped:
        success_mean, success_lo, success_hi = bootstrap_mean_ci(
            group["success"].astype(float).to_numpy()
        )
        dist_mean, dist_lo, dist_hi = bootstrap_mean_ci(
            group["final_distance_to_target"].to_numpy(), seed=1
        )
        time_mean, time_lo, time_hi = bootstrap_mean_ci(
            group["time_to_target_s"].to_numpy(), seed=2
        )
        rows.append(
            {
                "scenario": scenario,
                "policy": policy,
                "n": len(group),
                "success_mean": success_mean,
                "success_ci_low": success_lo,
                "success_ci_high": success_hi,
                "distance_mean": dist_mean,
                "distance_ci_low": dist_lo,
                "distance_ci_high": dist_hi,
                "time_mean": time_mean,
                "time_ci_low": time_lo,
                "time_ci_high": time_hi,
            }
        )
    return pd.DataFrame(rows)


def pairwise_policy_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, scenario_group in results_df.groupby("scenario", dropna=False):
        full_group = scenario_group[scenario_group["policy"] == "full_odor"]
        for baseline in ("attractive_only", "forward_only"):
            baseline_group = scenario_group[scenario_group["policy"] == baseline]
            for metric in ("success", "final_distance_to_target", "time_to_target_s"):
                stat, pvalue = mannwhitneyu(
                    baseline_group[metric].to_numpy(),
                    full_group[metric].to_numpy(),
                    alternative="two-sided",
                    method="asymptotic",
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "baseline": baseline,
                        "full_odor_mean": float(full_group[metric].mean()),
                        "baseline_mean": float(baseline_group[metric].mean()),
                        "pvalue": float(pvalue),
                        "u_stat": float(stat),
                    }
                )
    return pd.DataFrame(rows)


def plot_success_by_scenario(summary_df: pd.DataFrame, output_path: Path) -> None:
    scenario_names = list(summary_df["scenario"].drop_duplicates())
    policy_names = list(summary_df["policy"].drop_duplicates())
    x = np.arange(len(scenario_names))
    width = 0.25
    colors = {"full_odor": "#2563eb", "attractive_only": "#16a34a", "forward_only": "#6b7280"}
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    for idx, policy in enumerate(policy_names):
        subset = summary_df[summary_df["policy"] == policy].set_index("scenario")
        means = [float(subset.loc[name, "success_mean"]) for name in scenario_names]
        lows = [float(subset.loc[name, "success_mean"] - subset.loc[name, "success_ci_low"]) for name in scenario_names]
        highs = [float(subset.loc[name, "success_ci_high"] - subset.loc[name, "success_mean"]) for name in scenario_names]
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            yerr=np.array([lows, highs]),
            label=policy,
            color=colors[policy],
            capsize=4,
        )
    ax.set_xticks(x, scenario_names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Success Rate By Scenario And Policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_distance_by_scenario(summary_df: pd.DataFrame, output_path: Path) -> None:
    scenario_names = list(summary_df["scenario"].drop_duplicates())
    policy_names = list(summary_df["policy"].drop_duplicates())
    x = np.arange(len(scenario_names))
    width = 0.25
    colors = {"full_odor": "#2563eb", "attractive_only": "#16a34a", "forward_only": "#6b7280"}
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    for idx, policy in enumerate(policy_names):
        subset = summary_df[summary_df["policy"] == policy].set_index("scenario")
        means = [float(subset.loc[name, "distance_mean"]) for name in scenario_names]
        lows = [float(subset.loc[name, "distance_mean"] - subset.loc[name, "distance_ci_low"]) for name in scenario_names]
        highs = [float(subset.loc[name, "distance_ci_high"] - subset.loc[name, "distance_mean"]) for name in scenario_names]
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            yerr=np.array([lows, highs]),
            label=policy,
            color=colors[policy],
            capsize=4,
        )
    ax.set_xticks(x, scenario_names)
    ax.set_title("Final Distance By Scenario And Policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_full_policy_trajectories(
    results_df: pd.DataFrame, scenarios: dict[str, dict[str, object]], output_path: Path
) -> None:
    full_df = results_df[results_df["policy"] == "full_odor"]
    scenario_names = list(scenarios.keys())
    fig, axes = plt.subplots(1, len(scenario_names), figsize=(5 * len(scenario_names), 4), tight_layout=True)
    if len(scenario_names) == 1:
        axes = [axes]
    for ax, scenario_name in zip(axes, scenario_names):
        scenario_rows = full_df[full_df["scenario"] == scenario_name]
        odor_source = scenarios[scenario_name]["odor_source"]
        ax.scatter(odor_source[0, 0], odor_source[0, 1], s=120, c="#16a34a")
        if len(odor_source) > 1:
            ax.scatter(odor_source[1:, 0], odor_source[1:, 1], s=120, c="#dc2626")
        for _, row in scenario_rows.iterrows():
            data = np.load(row["trajectory_path"])
            xy = data["fly"][:, 0, :2]
            color = "#2563eb" if row["success"] else "#94a3b8"
            ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.8)
        ax.set_title(scenario_name)
        ax.set_xlabel("x position (mm)")
        ax.set_ylabel("y position (mm)")
        ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_research_report(
    summary_df: pd.DataFrame, pairwise_df: pd.DataFrame, output_path: Path
) -> None:
    lines = [
        "# Odor Navigation Research Report",
        "",
        "This report upgrades the odor project into a small generalization and ablation study.",
        "",
        "Briefly: B tests whether odor-driven steering helps a simulated fly navigate under competing sensory cues.",
        "",
        "Neuroscience goal: study how attractive and aversive olfactory signals can shape goal-directed fly navigation.",
        "",
        "Computational goal: compare closed-loop odor policies against weaker baselines across randomized starts and multiple odor layouts.",
        "",
        "## Research Question",
        "",
        "Does odor feedback improve navigation over open-loop movement, and when does adding aversive information change behavior compared with attraction alone?",
        "",
        "## Policy Set",
        "",
        "- `full_odor`: attractive and aversive terms both active",
        "- `attractive_only`: aversive term removed",
        "- `forward_only`: open-loop forward locomotion baseline",
        "",
        "## Scenario Summary",
        "",
        summary_df.to_csv(index=False).strip(),
        "",
        "## Pairwise Full-Policy Comparisons",
        "",
        pairwise_df.to_csv(index=False).strip(),
        "",
        "## Interpretation",
        "",
    ]
    for scenario, group in summary_df.groupby("scenario", dropna=False):
        best = group.sort_values(["success_mean", "distance_mean"], ascending=[False, True]).iloc[0]
        lines.append(
            f"- In `{scenario}`, the strongest policy was `{best['policy']}` "
            f"(success={best['success_mean']:.2f}, final_distance={best['distance_mean']:.3f})."
        )
    lines.extend(
        [
            "- Treat `full_odor` versus `forward_only` as the main closed-loop control contrast.",
            "- Treat `full_odor` versus `attractive_only` as an ablation that can reveal when aversive cues help, hurt, or make little difference.",
        ]
    )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This is a stronger research-style evaluation, not a publishable navigation model by itself.",
            "- The current policy comparison should not be described as proof that aversive cues always improve navigation.",
            "- A true path-integration contribution would require a principled home-vector model and ablations around that mechanism.",
        ]
    )
    output_path.write_text("\n".join(lines))


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/odor_navigation.toml"), exists=True, dir_okay=False
    ),
    output_dir: Path = typer.Option(Path("outputs/odor_navigation_research"), "--output-dir"),
    trials: int = typer.Option(6, "--trials"),
    run_time: float = typer.Option(1.5, "--run-time"),
) -> None:
    base_config = load_config(config_path)
    base_config.output_dir = output_dir
    base_config.trials = trials
    base_config.run_time_s = run_time
    output_dirs = ensure_output_dirs(base_config.output_dir)
    scenarios = scenario_library()

    rows = []
    for scenario_name, scenario_data in scenarios.items():
        typer.echo(f"Scenario: {scenario_name}")
        scenario_config = replace(
            base_config,
            odor_source=np.array(scenario_data["odor_source"], dtype=float),
            peak_odor_intensity=np.array(scenario_data["peak_odor_intensity"], dtype=float),
        )
        conditions = build_spawn_conditions(scenario_config)
        for policy in POLICIES:
            typer.echo(f"  Policy: {policy}")
            for trial_id, (spawn_pos, spawn_heading) in enumerate(conditions):
                rows.append(
                    run_research_trial(
                        config=scenario_config,
                        scenario_name=scenario_name,
                        policy=policy,
                        trial_id=trial_id,
                        spawn_pos=spawn_pos,
                        spawn_heading=spawn_heading,
                        output_dirs=output_dirs,
                    )
                )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dirs["base"] / "results.csv", index=False)
    summary_df = summarize_results(results_df)
    pairwise_df = pairwise_policy_stats(results_df)
    summary_df.to_csv(output_dirs["base"] / "summary_with_ci.csv", index=False)
    pairwise_df.to_csv(output_dirs["base"] / "pairwise_stats.csv", index=False)
    plot_success_by_scenario(summary_df, output_dirs["plots"] / "success_by_scenario.png")
    plot_distance_by_scenario(summary_df, output_dirs["plots"] / "distance_by_scenario.png")
    plot_full_policy_trajectories(
        results_df, scenarios, output_dirs["plots"] / "full_policy_trajectories.png"
    )
    write_research_report(
        summary_df, pairwise_df, output_dirs["base"] / "RESEARCH_REPORT.md"
    )
    typer.echo(f"Research results: {output_dirs['base'] / 'RESEARCH_REPORT.md'}")


if __name__ == "__main__":
    app()
