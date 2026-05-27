from pathlib import Path

import pandas as pd

from neurofly_odor.navigation_research import (
    pairwise_policy_stats,
    summarize_results,
    write_research_report,
)
from neurofly_odor.odor_benchmark import write_report


def sample_results() -> pd.DataFrame:
    rows = []
    for scenario in ["triplet", "diagonal_choice", "full_turn"]:
        rows.extend(
            [
                {
                    "scenario": scenario,
                    "policy": "full_odor",
                    "trial_id": 0,
                    "success": True,
                    "final_distance_to_target": 1.0,
                    "time_to_target_s": 0.5,
                    "path_length": 5.0,
                    "average_speed": 10.0,
                    "attractive_signal_mean": 0.2,
                    "aversive_signal_mean": 0.1,
                    "spawn_x": 0.0,
                    "spawn_y": 0.0,
                    "spawn_orientation_z": 0.0,
                    "target_x": 8.0,
                    "target_y": 0.0,
                    "trajectory_path": "trajectory.npz",
                },
                {
                    "scenario": scenario,
                    "policy": "attractive_only",
                    "trial_id": 0,
                    "success": True,
                    "final_distance_to_target": 0.9,
                    "time_to_target_s": 0.5,
                    "path_length": 5.1,
                    "average_speed": 10.2,
                    "attractive_signal_mean": 0.2,
                    "aversive_signal_mean": 0.1,
                    "spawn_x": 0.0,
                    "spawn_y": 0.0,
                    "spawn_orientation_z": 0.0,
                    "target_x": 8.0,
                    "target_y": 0.0,
                    "trajectory_path": "trajectory.npz",
                },
                {
                    "scenario": scenario,
                    "policy": "forward_only",
                    "trial_id": 0,
                    "success": False,
                    "final_distance_to_target": 8.0,
                    "time_to_target_s": 1.0,
                    "path_length": 8.0,
                    "average_speed": 8.0,
                    "attractive_signal_mean": 0.0,
                    "aversive_signal_mean": 0.0,
                    "spawn_x": 0.0,
                    "spawn_y": 0.0,
                    "spawn_orientation_z": 0.0,
                    "target_x": 8.0,
                    "target_y": 0.0,
                    "trajectory_path": "trajectory.npz",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_research_tables_cover_policy_ablations() -> None:
    results_df = sample_results()

    summary_df = summarize_results(results_df)
    pairwise_df = pairwise_policy_stats(results_df)

    assert set(summary_df["scenario"]) == {"triplet", "diagonal_choice", "full_turn"}
    assert set(summary_df["policy"]) == {"full_odor", "attractive_only", "forward_only"}
    assert len(pairwise_df) == 18
    assert set(pairwise_df["baseline"]) == {"attractive_only", "forward_only"}


def test_reports_include_reader_goals_and_conservative_claim(tmp_path: Path) -> None:
    results_df = sample_results()
    showcase_path = tmp_path / "SHOWCASE_REPORT.md"
    research_path = tmp_path / "RESEARCH_REPORT.md"

    write_report(results_df[results_df["policy"] == "full_odor"], showcase_path)
    summary_df = summarize_results(results_df)
    pairwise_df = pairwise_policy_stats(results_df)
    write_research_report(summary_df, pairwise_df, research_path)

    showcase = showcase_path.read_text()
    research = research_path.read_text()
    assert "Briefly: B tests whether odor-driven steering helps" in showcase
    assert "Neuroscience goal:" in showcase
    assert "Computational goal:" in research
    assert "should not be described as proof that aversive cues always improve navigation" in research
