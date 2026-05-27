from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import math
import tomllib

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from dm_control.rl.control import PhysicsError
from flygym_gymnasium import SingleFlySimulation, YawOnlyCamera
from flygym_gymnasium.arena import BlocksTerrain, FlatTerrain, MixedTerrain
from flygym_gymnasium.examples.locomotion import (
    CPGNetwork,
    ColorableFly,
    PreprogrammedSteps,
)
from flygym_gymnasium.examples.locomotion.rule_based_controller import (
    RuleBasedController,
    construct_rules_graph,
)
from flygym_gymnasium.preprogrammed import get_cpg_biases
from scipy.interpolate import interp1d

app = typer.Typer(add_completion=False)

CONTROLLERS = ("cpg", "rule_based", "hybrid")
TERRAINS = ("flat", "rough", "blocks")
HYBRID_CORRECTION_VECTORS = {
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}
HYBRID_RIGHT_LEG_INVERSION = [1, -1, -1, 1, -1, 1, 1]
HYBRID_CORRECTION_RATES = {"retraction": (800, 700), "stumbling": (2200, 1800)}
HYBRID_MAX_INCREMENT = 80
HYBRID_RETRACTION_PERSISTENCE = 20
HYBRID_PERSISTENCE_INIT_THRESHOLD = 20
RULE_BASED_WEIGHTS = {
    "rule1": -10,
    "rule2_ipsi": 2.5,
    "rule2_contra": 1,
    "rule3_ipsi": 3.0,
    "rule3_contra": 2.0,
}


@dataclass
class BenchmarkConfig:
    seed: int
    render: bool
    save_video: bool
    output_dir: Path
    episode_duration_s: float
    stumble_force_threshold: float
    slip_speed_threshold: float
    contact_force_threshold: float
    trials: int
    spawn_bbox: tuple[float, float, float, float]
    spawn_z: float
    timestep: float
    terrains: list[str]
    controllers: list[str]
    metrics: list[str]


@dataclass
class RunArtifacts:
    camera: YawOnlyCamera | None
    results_dir: Path
    video_path: Path
    trajectory_path: Path


@dataclass
class RunOutcome:
    obs_list: list[dict]
    completed_steps: int
    target_steps: int
    physics_error: bool
    terminated: bool
    truncated: bool


def load_config(path: Path) -> BenchmarkConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    return BenchmarkConfig(
        seed=int(raw["seed"]),
        render=bool(raw["render"]),
        save_video=bool(raw["save_video"]),
        output_dir=Path(raw["output_dir"]),
        episode_duration_s=float(raw["episode_duration_s"]),
        stumble_force_threshold=float(raw["stumble_force_threshold"]),
        slip_speed_threshold=float(raw["slip_speed_threshold"]),
        contact_force_threshold=float(raw["contact_force_threshold"]),
        trials=int(raw["trials"]),
        spawn_bbox=tuple(float(x) for x in raw["spawn_bbox"]),
        spawn_z=float(raw["spawn_z"]),
        timestep=float(raw["timestep"]),
        terrains=[str(x) for x in raw["terrains"]["names"]],
        controllers=[str(x) for x in raw["controllers"]["names"]],
        metrics=[str(x) for x in raw["metrics"]["names"]],
    )


def save_obs_list(save_path: Path, obs_list: list[dict]) -> None:
    if not obs_list:
        np.savez_compressed(save_path, empty=np.array([]))
        return

    array_dict = {}
    for key in obs_list[0]:
        array_dict[key] = np.array([obs[key] for obs in obs_list])
    np.savez_compressed(save_path, **array_dict)


def get_arena(terrain: str, seed: int):
    if terrain == "flat":
        return FlatTerrain()
    if terrain == "rough":
        return MixedTerrain(rand_seed=seed)
    if terrain == "blocks":
        return BlocksTerrain(rand_seed=seed)
    raise ValueError(f"Unsupported terrain: {terrain}")


def build_spawn_positions(
    trial_count: int,
    seed: int,
    spawn_bbox: tuple[float, float, float, float],
    spawn_z: float,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    positions = rng.rand(trial_count, 2) * spawn_bbox[2:] + spawn_bbox[:2]
    return np.column_stack((positions, np.full(trial_count, spawn_z)))


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    dirs = {
        "base": base_dir,
        "videos": base_dir / "videos",
        "trajectories": base_dir / "trajectories",
        "plots": base_dir / "plots",
        "montages": base_dir / "montages",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def create_artifacts(
    output_dirs: dict[str, Path],
    terrain: str,
    controller: str,
    trial_id: int,
) -> RunArtifacts:
    stem = f"{terrain}_{controller}_trial_{trial_id:02d}"
    return RunArtifacts(
        camera=None,
        results_dir=output_dirs["base"],
        video_path=output_dirs["videos"] / f"{stem}.mp4",
        trajectory_path=output_dirs["trajectories"] / f"{stem}.npz",
    )


def make_simulation(
    terrain: str,
    terrain_seed: int,
    spawn_pos: np.ndarray,
    timestep: float,
    render: bool,
    save_video: bool,
) -> tuple[SingleFlySimulation, PreprogrammedSteps, YawOnlyCamera | None]:
    preprogrammed_steps = PreprogrammedSteps()
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in preprogrammed_steps.legs
        for segment in ["Tibia"] + [f"Tarsus{i}" for i in range(1, 6)]
    ]

    fly = ColorableFly(
        enable_adhesion=True,
        draw_adhesion=render or save_video,
        init_pose="stretch",
        control="position",
        spawn_pos=tuple(float(x) for x in spawn_pos),
        contact_sensor_placements=contact_sensor_placements,
        actuator_forcerange=(-65.0, 65.0),
        detect_flip=True,
    )

    camera = None
    cameras = None
    if render or save_video:
        camera = YawOnlyCamera(
            attachment_point=fly.model.worldbody,
            camera_name="camera_right",
            targeted_fly_names=fly.name,
            play_speed=0.1,
        )
        cameras = [camera]

    sim = SingleFlySimulation(
        fly=fly,
        cameras=cameras,
        timestep=timestep,
        arena=get_arena(terrain, terrain_seed),
    )
    return sim, preprogrammed_steps, camera


def maybe_render(sim: SingleFlySimulation, render: bool) -> None:
    if render:
        sim.render()


def run_cpg_controller(
    sim: SingleFlySimulation,
    preprogrammed_steps: PreprogrammedSteps,
    duration_s: float,
    seed: int,
    render: bool,
) -> RunOutcome:
    intrinsic_freqs = np.ones(6) * 12
    intrinsic_amps = np.ones(6)
    phase_biases = get_cpg_biases("tripod")
    coupling_weights = (phase_biases > 0) * 10
    convergence_coefs = np.ones(6) * 20

    cpg_network = CPGNetwork(
        timestep=sim.timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
        seed=seed,
    )
    cpg_network.reset()

    _, _ = sim.reset()
    obs_list: list[dict] = []
    target_steps = int(duration_s / sim.timestep)
    terminated = False
    truncated = False

    for _step in range(target_steps):
        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []
        for idx, leg in enumerate(preprogrammed_steps.legs):
            joints_angles.append(
                preprogrammed_steps.get_joint_angles(
                    leg, cpg_network.curr_phases[idx], cpg_network.curr_magnitudes[idx]
                )
            )
            adhesion_onoff.append(
                preprogrammed_steps.get_adhesion_onoff(leg, cpg_network.curr_phases[idx])
            )
        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        try:
            obs, _reward, terminated, truncated, _info = sim.step(action)
            obs_list.append(obs)
            maybe_render(sim, render)
        except PhysicsError:
            return RunOutcome(obs_list, len(obs_list), target_steps, True, False, False)
        if terminated or truncated:
            break

    return RunOutcome(obs_list, len(obs_list), target_steps, False, terminated, truncated)


def run_rule_based_controller(
    sim: SingleFlySimulation,
    preprogrammed_steps: PreprogrammedSteps,
    duration_s: float,
    seed: int,
    render: bool,
) -> RunOutcome:
    preprogrammed_steps.duration = 1 / 12.0
    controller = RuleBasedController(
        timestep=sim.timestep,
        rules_graph=construct_rules_graph(),
        weights=RULE_BASED_WEIGHTS,
        preprogrammed_steps=preprogrammed_steps,
        seed=seed,
    )

    _, _ = sim.reset()
    obs_list: list[dict] = []
    target_steps = int(duration_s / sim.timestep)
    terminated = False
    truncated = False

    for _step in range(target_steps):
        controller.step()
        joint_angles = []
        adhesion_onoff = []
        for leg, phase in zip(controller.legs, controller.leg_phases):
            joint_angles.append(
                controller.preprogrammed_steps.get_joint_angles(leg, phase).flatten()
            )
            adhesion_onoff.append(
                controller.preprogrammed_steps.get_adhesion_onoff(leg, phase)
            )
        action = {
            "joints": np.concatenate(joint_angles),
            "adhesion": np.array(adhesion_onoff),
        }
        try:
            obs, _reward, terminated, truncated, _info = sim.step(action)
            obs_list.append(obs)
            maybe_render(sim, render)
        except PhysicsError:
            return RunOutcome(obs_list, len(obs_list), target_steps, True, False, False)
        if terminated or truncated:
            break

    return RunOutcome(obs_list, len(obs_list), target_steps, False, terminated, truncated)


def run_hybrid_controller(
    sim: SingleFlySimulation,
    preprogrammed_steps: PreprogrammedSteps,
    duration_s: float,
    seed: int,
    render: bool,
    stumble_force_threshold: float,
) -> RunOutcome:
    intrinsic_freqs = np.ones(6) * 12
    intrinsic_amps = np.ones(6)
    phase_biases = get_cpg_biases("tripod")
    coupling_weights = (phase_biases > 0) * 10
    convergence_coefs = np.ones(6) * 20
    cpg_network = CPGNetwork(
        timestep=sim.timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
        seed=seed,
    )
    cpg_network.reset()

    step_phase_multiplier: dict[str, interp1d] = {}
    increments = [0, 0.8, 0, -0.1, 0]
    for leg in preprogrammed_steps.legs:
        swing_start, swing_end = preprogrammed_steps.swing_period[leg]
        step_points = [
            swing_start,
            np.mean([swing_start, swing_end]),
            swing_end + np.pi / 4,
            np.mean([swing_end, 2 * np.pi]),
            2 * np.pi,
        ]
        preprogrammed_steps.swing_period[leg] = (swing_start, swing_end + np.pi / 4)
        step_phase_multiplier[leg] = interp1d(
            step_points, increments, fill_value="extrapolate"
        )

    detected_segments = {"Tibia", "Tarsus1", "Tarsus2"}
    stumbling_sensors = {leg: [] for leg in preprogrammed_steps.legs}
    for idx, sensor_name in enumerate(sim.fly.contact_sensor_placements):
        leg = sensor_name.split("/")[1][:2]
        segment = sensor_name.split("/")[1][2:]
        if segment in detected_segments:
            stumbling_sensors[leg].append(idx)
    stumbling_sensors = {key: np.array(value) for key, value in stumbling_sensors.items()}

    retraction_correction = np.zeros(6)
    stumbling_correction = np.zeros(6)
    retraction_persistence_counter = np.zeros(6)

    obs, _ = sim.reset()
    obs_list: list[dict] = []
    target_steps = int(duration_s / sim.timestep)
    terminated = False
    truncated = False

    for _step in range(target_steps):
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        sorted_idx = np.argsort(end_effector_z_pos)
        sorted_z = end_effector_z_pos[sorted_idx]
        if sorted_z[-1] > sorted_z[-3] + 0.05:
            leg_to_correct_retraction = sorted_idx[-1]
            if retraction_correction[leg_to_correct_retraction] > HYBRID_PERSISTENCE_INIT_THRESHOLD:
                retraction_persistence_counter[leg_to_correct_retraction] = 1
        else:
            leg_to_correct_retraction = None

        retraction_persistence_counter[retraction_persistence_counter > 0] += 1
        retraction_persistence_counter[
            retraction_persistence_counter > HYBRID_RETRACTION_PERSISTENCE
        ] = 0

        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []

        for idx, leg in enumerate(preprogrammed_steps.legs):
            if idx == leg_to_correct_retraction or retraction_persistence_counter[idx] > 0:
                retraction_correction[idx] += HYBRID_CORRECTION_RATES["retraction"][0] * sim.timestep
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (1.0, 0.41, 0.71))
            else:
                retraction_correction[idx] = max(
                    0.0,
                    retraction_correction[idx]
                    - HYBRID_CORRECTION_RATES["retraction"][1] * sim.timestep,
                )
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", None)

            retract = retraction_correction[idx] > 0
            contact_forces = obs["contact_forces"][stumbling_sensors[leg], :]
            force_projection = np.dot(contact_forces, obs["fly_orientation"])
            if (force_projection < -stumble_force_threshold).any():
                stumbling_correction[idx] += HYBRID_CORRECTION_RATES["stumbling"][0] * sim.timestep
                if not retract:
                    sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (0.12, 0.56, 1.0))
            else:
                stumbling_correction[idx] = max(
                    0.0,
                    stumbling_correction[idx]
                    - HYBRID_CORRECTION_RATES["stumbling"][1] * sim.timestep,
                )
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", None)

            if retraction_correction[idx] > 0:
                net_correction = retraction_correction[idx]
                stumbling_correction[idx] = 0
            else:
                net_correction = stumbling_correction[idx]

            joint_angles = preprogrammed_steps.get_joint_angles(
                leg, cpg_network.curr_phases[idx], cpg_network.curr_magnitudes[idx]
            )
            net_correction = float(np.clip(net_correction, 0, HYBRID_MAX_INCREMENT))
            if leg[0] == "R":
                net_correction *= HYBRID_RIGHT_LEG_INVERSION[idx]
            net_correction *= float(step_phase_multiplier[leg](cpg_network.curr_phases[idx] % (2 * np.pi)))
            joint_angles = joint_angles + net_correction * HYBRID_CORRECTION_VECTORS[leg[1]]
            joints_angles.append(joint_angles)
            adhesion_onoff.append(
                preprogrammed_steps.get_adhesion_onoff(leg, cpg_network.curr_phases[idx])
            )

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff),
        }
        try:
            obs, _reward, terminated, truncated, _info = sim.step(action)
            obs_list.append(obs)
            maybe_render(sim, render)
        except PhysicsError:
            return RunOutcome(obs_list, len(obs_list), target_steps, True, False, False)
        if terminated or truncated:
            break

    return RunOutcome(obs_list, len(obs_list), target_steps, False, terminated, truncated)


def get_sensor_index_groups(sim: SingleFlySimulation) -> dict[str, dict[str, np.ndarray]]:
    groups: dict[str, dict[str, list[int]]] = {}
    for idx, sensor_name in enumerate(sim.fly.contact_sensor_placements):
        leg_and_segment = sensor_name.split("/")[1]
        leg = leg_and_segment[:2]
        segment = leg_and_segment[2:]
        groups.setdefault(leg, {}).setdefault(segment, []).append(idx)

    return {
        leg: {segment: np.array(indices) for segment, indices in segments.items()}
        for leg, segments in groups.items()
    }


def count_stumble_events(
    obs_list: list[dict],
    sensor_groups: dict[str, dict[str, np.ndarray]],
    stumble_force_threshold: float,
) -> int:
    detected_segments = ("Tibia", "Tarsus1", "Tarsus2")
    previous = {leg: False for leg in sensor_groups}
    count = 0
    for obs in obs_list:
        for leg, segments in sensor_groups.items():
            indices = np.concatenate(
                [segments[name] for name in detected_segments if name in segments]
            )
            force_projection = np.dot(obs["contact_forces"][indices], obs["fly_orientation"])
            is_stumbling = bool((force_projection < -stumble_force_threshold).any())
            if is_stumbling and not previous[leg]:
                count += 1
            previous[leg] = is_stumbling
    return count


def count_slip_events(
    obs_list: list[dict],
    sensor_groups: dict[str, dict[str, np.ndarray]],
    timestep: float,
    slip_speed_threshold: float,
    contact_force_threshold: float,
) -> int:
    if len(obs_list) < 2:
        return 0

    previous = {leg: False for leg in sensor_groups}
    count = 0
    for prev_obs, curr_obs in zip(obs_list[:-1], obs_list[1:]):
        end_effector_delta = curr_obs["end_effectors"][:, :2] - prev_obs["end_effectors"][:, :2]
        horizontal_speed = np.linalg.norm(end_effector_delta, axis=1) / timestep
        for leg, segments in sensor_groups.items():
            leg_id = ("LF", "LM", "LH", "RF", "RM", "RH").index(leg)
            sensor_indices = np.concatenate(list(segments.values()))
            contact_magnitude = np.linalg.norm(curr_obs["contact_forces"][sensor_indices], axis=1).max()
            is_slipping = bool(
                contact_magnitude > contact_force_threshold
                and horizontal_speed[leg_id] > slip_speed_threshold
            )
            if is_slipping and not previous[leg]:
                count += 1
            previous[leg] = is_slipping
    return count


def summarize_run(
    outcome: RunOutcome,
    sim: SingleFlySimulation,
    terrain: str,
    controller: str,
    trial_id: int,
    seed: int,
    spawn_pos: np.ndarray,
    config: BenchmarkConfig,
) -> dict[str, object]:
    sensor_groups = get_sensor_index_groups(sim)
    success = (
        not outcome.physics_error
        and not outcome.terminated
        and not outcome.truncated
        and outcome.completed_steps == outcome.target_steps
    )

    if outcome.obs_list:
        start_xy = outcome.obs_list[0]["fly"][0][:2]
        end_xy = outcome.obs_list[-1]["fly"][0][:2]
        distance = float(np.linalg.norm(end_xy - start_xy))
    else:
        distance = 0.0

    elapsed_time_s = outcome.completed_steps * config.timestep
    average_speed = distance / elapsed_time_s if elapsed_time_s > 0 else 0.0
    stumble_count = count_stumble_events(
        outcome.obs_list, sensor_groups, config.stumble_force_threshold
    )
    slip_count = count_slip_events(
        outcome.obs_list,
        sensor_groups,
        config.timestep,
        config.slip_speed_threshold,
        config.contact_force_threshold,
    )

    failure_reason = "completed"
    if outcome.physics_error:
        failure_reason = "physics_error"
    elif outcome.terminated:
        failure_reason = "terminated"
    elif outcome.truncated:
        failure_reason = "truncated"

    return {
        "trial_id": trial_id,
        "seed": seed,
        "terrain": terrain,
        "controller": controller,
        "spawn_x": float(spawn_pos[0]),
        "spawn_y": float(spawn_pos[1]),
        "spawn_z": float(spawn_pos[2]),
        "completed_steps": outcome.completed_steps,
        "target_steps": outcome.target_steps,
        "elapsed_time_s": elapsed_time_s,
        "distance_before_failure": distance,
        "average_speed": average_speed,
        "stumble_count": stumble_count,
        "slip_count": slip_count,
        "success_rate": float(success),
        "success": success,
        "failure_reason": failure_reason,
        "flygym_gymnasium_version": version("flygym-gymnasium"),
    }


def plot_summary(results_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("distance_before_failure", "Distance"),
        ("average_speed", "Average Speed"),
        ("stumble_count", "Stumble Count"),
        ("success_rate", "Success Rate"),
    ]
    grouped = (
        results_df.groupby(["terrain", "controller"], dropna=False)
        .agg({metric: "mean" for metric, _ in metrics})
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    axes = axes.flatten()
    terrain_names = list(grouped["terrain"].drop_duplicates())
    controller_names = list(grouped["controller"].drop_duplicates())
    colors = {"flat": "#2563eb", "rough": "#c2410c"}
    x = np.arange(len(controller_names))
    width = 0.35

    for axis, (metric, label) in zip(axes, metrics):
        for idx, terrain in enumerate(terrain_names):
            subset = grouped[grouped["terrain"] == terrain].set_index("controller")
            values = [float(subset.loc[name, metric]) if name in subset.index else math.nan for name in controller_names]
            axis.bar(
                x + (idx - 0.5) * width,
                values,
                width=width,
                label=terrain.capitalize(),
                color=colors.get(terrain, "#6b7280"),
            )
        axis.set_title(label)
        axis.set_xticks(x, [name.replace("_", " ").title() for name in controller_names], rotation=15)
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle("Terrain Benchmark Summary", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_trial_trajectory_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (terrain, trial_id), group in results_df.groupby(["terrain", "trial_id"], dropna=False):
        fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)
        for _, row in group.iterrows():
            data = np.load(row["trajectory_path"])
            if "fly" not in data:
                continue
            xy = data["fly"][:, 0, :2]
            ax.plot(xy[:, 0], xy[:, 1], label=str(row["controller"]).replace("_", " ").title())
            ax.scatter(xy[0, 0], xy[0, 1], s=20)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=30, marker="x")
        ax.set_title(f"Terrain={terrain}, Trial={int(trial_id):02d}")
        ax.set_xlabel("x position (mm)")
        ax.set_ylabel("y position (mm)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.savefig(output_dir / f"{terrain}_trial_{int(trial_id):02d}_trajectories.png", dpi=200)
        plt.close(fig)


def write_summary_report(results_df: pd.DataFrame, output_path: Path) -> None:
    grouped = (
        results_df.groupby(["terrain", "controller"], dropna=False)
        .agg(
            success_rate=("success_rate", "mean"),
            average_speed=("average_speed", "mean"),
            distance_before_failure=("distance_before_failure", "mean"),
            stumble_count=("stumble_count", "mean"),
            slip_count=("slip_count", "mean"),
        )
        .reset_index()
    )
    best_rows = grouped.sort_values(["terrain", "success_rate", "average_speed"], ascending=[True, False, False])
    lines = [
        "# Terrain Benchmark Showcase Report",
        "",
        "This project turns NeuroMechFly locomotion tutorials into a small but reproducible controller benchmark.",
        "",
        "Briefly: A tests how existing fly locomotion controllers behave as terrain gets harder.",
        "",
        "Neuroscience goal: study how neural-inspired gait controllers cope with increasingly difficult walking surfaces.",
        "",
        "Computational goal: compare controllers under matched seeds, terrain conditions, and metrics using a reproducible FlyGym pipeline.",
        "",
        "The comparison asks a simple question:",
        "",
        "How do `cpg`, `rule_based`, and `hybrid` locomotion controllers behave when they are evaluated under the same terrain and spawn conditions?",
        "",
        "## Why This Is Worth Showing",
        "",
        "- It uses the real NeuroMechFly controller stack rather than a toy approximation.",
        "- It standardizes terrain, spawn conditions, and metrics so the comparison is interpretable.",
        "- It evaluates a terrain ladder: `flat`, `rough`, and `blocks`.",
        "- It produces both visual and quantitative outputs that are easy to inspect.",
        "",
        "## Mean Metrics",
        "",
        grouped.to_csv(index=False).strip(),
        "",
        "## Takeaway",
        "",
    ]
    for terrain, group in best_rows.groupby("terrain", dropna=False):
        best = group.iloc[0]
        lines.append(
            f"- On `{terrain}` terrain, `{best['controller']}` was the strongest controller in this run set "
            f"(success={best['success_rate']:.2f}, speed={best['average_speed']:.3f})."
        )
    lines.extend(
        [
            "",
            "## What Was Built",
            "",
            "- A runnable benchmark CLI",
            "- CSV export for per-run metrics",
            "- Aggregate summary plots",
            "- Trajectory overlays for each trial",
            "- Per-controller videos and side-by-side montages when rendering is enabled",
            "",
            "## Files",
            "",
            "- `results.csv` contains per-run metrics.",
            "- `plots/terrain_benchmark_summary.png` contains the aggregate figure.",
            "- `plots/*_trajectories.png` contains controller overlays for each trial.",
            "- `videos/` and `montages/` contain rendered videos when rendering is enabled.",
        ]
    )
    output_path.write_text("\n".join(lines))


def create_montage_video(
    video_paths: list[Path],
    output_path: Path,
) -> None:
    readers = [imageio.get_reader(path) for path in video_paths]
    try:
        frame_count = min(reader.count_frames() for reader in readers)
        fps = readers[0].get_meta_data().get("fps", 30)
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_idx in range(frame_count):
                frames = [reader.get_data(frame_idx) for reader in readers]
                writer.append_data(np.concatenate(frames, axis=1))
    finally:
        for reader in readers:
            reader.close()


def run_single_controller(
    controller: str,
    terrain: str,
    trial_id: int,
    spawn_pos: np.ndarray,
    output_dirs: dict[str, Path],
    config: BenchmarkConfig,
    render: bool,
    save_video: bool,
) -> dict[str, object]:
    artifacts = create_artifacts(output_dirs, terrain, controller, trial_id)
    run_seed = config.seed + trial_id
    sim, preprogrammed_steps, camera = make_simulation(
        terrain=terrain,
        terrain_seed=run_seed,
        spawn_pos=spawn_pos,
        timestep=config.timestep,
        render=render,
        save_video=save_video,
    )

    if controller == "cpg":
        outcome = run_cpg_controller(
            sim, preprogrammed_steps, config.episode_duration_s, run_seed, render
        )
    elif controller == "rule_based":
        outcome = run_rule_based_controller(
            sim, preprogrammed_steps, config.episode_duration_s, run_seed, render
        )
    elif controller == "hybrid":
        outcome = run_hybrid_controller(
            sim,
            preprogrammed_steps,
            config.episode_duration_s,
            run_seed,
            render,
            config.stumble_force_threshold,
        )
    else:
        raise ValueError(f"Unsupported controller: {controller}")

    save_obs_list(artifacts.trajectory_path, outcome.obs_list)
    if save_video and camera is not None:
        camera.save_video(artifacts.video_path.as_posix(), 0)

    result = summarize_run(
        outcome=outcome,
        sim=sim,
        terrain=terrain,
        controller=controller,
        trial_id=trial_id,
        seed=run_seed,
        spawn_pos=spawn_pos,
        config=config,
    )
    result["trajectory_path"] = artifacts.trajectory_path.as_posix()
    result["video_path"] = artifacts.video_path.as_posix() if save_video else ""
    return result


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/terrain_benchmark.toml"), exists=True, dir_okay=False
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    trial_count: int | None = typer.Option(None, "--trials"),
    duration_s: float | None = typer.Option(None, "--duration-s"),
    terrain: list[str] | None = typer.Option(None, "--terrain"),
    controller: list[str] | None = typer.Option(None, "--controller"),
    render: bool | None = typer.Option(None, "--render/--no-render"),
    save_video: bool | None = typer.Option(None, "--save-video/--no-save-video"),
) -> None:
    config = load_config(config_path)
    if output_dir is not None:
        config.output_dir = output_dir
    if trial_count is not None:
        config.trials = trial_count
    if duration_s is not None:
        config.episode_duration_s = duration_s
    if terrain:
        config.terrains = terrain
    if controller:
        config.controllers = controller
    if render is not None:
        config.render = render
    if save_video is not None:
        config.save_video = save_video

    invalid_terrains = sorted(set(config.terrains) - set(TERRAINS))
    invalid_controllers = sorted(set(config.controllers) - set(CONTROLLERS))
    if invalid_terrains:
        raise typer.BadParameter(f"Unsupported terrains: {', '.join(invalid_terrains)}")
    if invalid_controllers:
        raise typer.BadParameter(
            f"Unsupported controllers: {', '.join(invalid_controllers)}"
        )

    if config.save_video and not config.render:
        typer.echo("Enabling render because video capture requires rendered frames.")
        config.render = True

    output_dirs = ensure_output_dirs(config.output_dir)
    spawn_positions = build_spawn_positions(
        config.trials, config.seed, config.spawn_bbox, config.spawn_z
    )

    rows: list[dict[str, object]] = []
    for terrain_name in config.terrains:
        typer.echo(f"Terrain: {terrain_name}")
        for trial_id, spawn_pos in enumerate(spawn_positions):
            typer.echo(f"  Trial {trial_id:02d} spawn={spawn_pos.tolist()}")
            for controller_name in config.controllers:
                typer.echo(f"    Running {controller_name}")
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
                typer.echo(
                    "      "
                    f"distance={row['distance_before_failure']:.3f} "
                    f"speed={row['average_speed']:.3f} "
                    f"success={int(bool(row['success']))} "
                    f"failure={row['failure_reason']}"
                )

            if config.save_video and all(
                (output_dirs["videos"] / f"{terrain_name}_{name}_trial_{trial_id:02d}.mp4").exists()
                for name in config.controllers
            ):
                montage_path = output_dirs["montages"] / f"{terrain_name}_trial_{trial_id:02d}.mp4"
                create_montage_video(
                    [
                        output_dirs["videos"] / f"{terrain_name}_{name}_trial_{trial_id:02d}.mp4"
                        for name in config.controllers
                    ],
                    montage_path,
                )

    results_df = pd.DataFrame(rows)
    results_csv = output_dirs["base"] / "results.csv"
    results_df.to_csv(results_csv, index=False)
    summary_plot = output_dirs["plots"] / "terrain_benchmark_summary.png"
    plot_summary(results_df, summary_plot)
    plot_trial_trajectory_comparison(results_df, output_dirs["plots"])
    write_summary_report(results_df, output_dirs["base"] / "SHOWCASE_REPORT.md")

    typer.echo(f"Results CSV: {results_csv}")
    typer.echo(f"Summary plot: {summary_plot}")
