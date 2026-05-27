from pathlib import Path

import pandas as pd

from neurofly.terrain_benchmark import load_config, write_summary_report
from neurofly.terrain_research import pairwise_stats, summarize_with_ci, write_research_report


def sample_results() -> pd.DataFrame:
    rows = []
    for terrain in ["flat", "rough", "blocks"]:
        for trial_id in range(2):
            rows.extend(
                [
                    {
                        "terrain": terrain,
                        "controller": "cpg",
                        "trial_id": trial_id,
                        "average_speed": 3.0 + trial_id,
                        "distance_before_failure": 0.3 + trial_id * 0.01,
                        "success_rate": 1.0,
                        "stumble_count": trial_id,
                        "slip_count": 2 + trial_id,
                    },
                    {
                        "terrain": terrain,
                        "controller": "rule_based",
                        "trial_id": trial_id,
                        "average_speed": 2.0 + trial_id,
                        "distance_before_failure": 0.2 + trial_id * 0.01,
                        "success_rate": 1.0,
                        "stumble_count": 2 + trial_id,
                        "slip_count": 3 + trial_id,
                    },
                    {
                        "terrain": terrain,
                        "controller": "hybrid",
                        "trial_id": trial_id,
                        "average_speed": 4.0 + trial_id,
                        "distance_before_failure": 0.4 + trial_id * 0.01,
                        "success_rate": 1.0,
                        "stumble_count": trial_id,
                        "slip_count": 1 + trial_id,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_default_showcase_terrains_include_blocks() -> None:
    config = load_config(Path("configs/terrain_benchmark.toml"))

    assert config.terrains == ["flat", "rough", "blocks"]


def test_research_tables_cover_all_hybrid_comparisons() -> None:
    results_df = sample_results()

    summary_df = summarize_with_ci(results_df)
    pairwise_df = pairwise_stats(results_df)

    assert set(summary_df["terrain"]) == {"flat", "rough", "blocks"}
    assert set(summary_df["controller"]) == {"cpg", "rule_based", "hybrid"}
    assert len(pairwise_df) == 12
    assert set(pairwise_df["baseline"]) == {"cpg", "rule_based"}


def test_reports_include_reader_goals_and_robustness_limit(tmp_path: Path) -> None:
    results_df = sample_results()
    showcase_path = tmp_path / "SHOWCASE_REPORT.md"
    research_path = tmp_path / "RESEARCH_REPORT.md"

    write_summary_report(results_df, showcase_path)
    summary_df = summarize_with_ci(results_df)
    pairwise_df = pairwise_stats(results_df)
    write_research_report(summary_df, pairwise_df, results_df, research_path)

    showcase = showcase_path.read_text()
    research = research_path.read_text()
    assert "Briefly: A tests how existing fly locomotion controllers behave as terrain gets harder." in showcase
    assert "Neuroscience goal:" in showcase
    assert "Computational goal:" in research
    assert "speed/stability evidence rather than failure-robustness evidence" in research
