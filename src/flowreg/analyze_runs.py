"""Summarize local PPO and FlowReg runs from Monitor CSV and TensorBoard logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator


FLOWREG_TAGS = [
    "Loss/FlowReg",
    "Loss/FlowReg_PaperScaled",
    "Loss/FlowReg_MSEMean",
    "Latent/Path_Length",
    "Latent/Net_Displacement",
    "Latent/Acceleration_Energy",
    "Latent/ODE_Error_Drift",
]

SCALAR_TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    *FLOWREG_TAGS,
    "FlowReg/Updates",
]


@dataclass(frozen=True)
class RunSpec:
    group: str
    seed: int
    path: Path


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _parse_run_spec(raw: str) -> RunSpec:
    parts = raw.split(":", maxsplit=2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Run specs must use GROUP:SEED:PATH format")
    group, seed_raw, path_raw = parts
    if not group:
        raise argparse.ArgumentTypeError("Run group cannot be empty")
    try:
        seed = int(seed_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed in run spec: {seed_raw}") from exc
    path = Path(path_raw)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Run path does not exist: {path}")
    return RunSpec(group=group, seed=seed, path=path)


def summarize_monitor(run_dir: Path) -> dict[str, Any]:
    rewards: list[float] = []
    lengths: list[int] = []
    for path in sorted((run_dir / "monitor").glob("*.csv.monitor.csv")):
        with path.open("r", encoding="utf-8") as handle:
            handle.readline()
            reader = csv.DictReader(handle)
            for row in reader:
                rewards.append(float(row["r"]))
                lengths.append(int(float(row["l"])))

    tail = rewards[-20:]
    return {
        "episodes": len(rewards),
        "mean_reward_all": _mean(rewards),
        "mean_reward_last20": _mean(tail),
        "max_reward": max(rewards) if rewards else None,
        "positive_episodes": sum(reward > 0 for reward in rewards),
        "mean_length_all": _mean([float(length) for length in lengths]),
    }


def summarize_scalars(run_dir: Path) -> dict[str, Any]:
    tensorboard_dirs = sorted((run_dir / "tensorboard").glob("*"))
    event_files = [
        event_file
        for tensorboard_dir in tensorboard_dirs
        for event_file in sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    ]
    if not event_files:
        return {}

    accumulator = event_accumulator.EventAccumulator(
        str(event_files[0]),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    summaries: dict[str, Any] = {}
    for tag in SCALAR_TAGS:
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        values = accumulator.Scalars(tag)
        series = [float(value.value) for value in values]
        summaries[tag] = {
            "count": len(values),
            "first": series[0],
            "last": series[-1],
            "last_step": int(values[-1].step),
            "finite": all(math.isfinite(value) for value in series),
        }
    return summaries


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def summarize_run(spec: RunSpec) -> dict[str, Any]:
    return {
        "group": spec.group,
        "seed": spec.seed,
        "run_dir": str(spec.path),
        "config": load_run_config(spec.path),
        "monitor": summarize_monitor(spec.path),
        "scalars": summarize_scalars(spec.path),
    }


def aggregate_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    monitors = [run["monitor"] for run in runs]
    result: dict[str, Any] = {
        "num_runs": len(runs),
        "episodes_total": sum(int(monitor["episodes"]) for monitor in monitors),
        "positive_episodes_total": sum(int(monitor["positive_episodes"]) for monitor in monitors),
        "max_reward_max": max(
            monitor["max_reward"] for monitor in monitors if monitor["max_reward"] is not None
        ),
    }

    for key in ("mean_reward_all", "mean_reward_last20"):
        values = [float(monitor[key]) for monitor in monitors if monitor[key] is not None]
        result[f"{key}_mean"] = _mean(values)
        result[f"{key}_std"] = _stdev(values)

    flow_metric_present = any(FLOWREG_TAGS[0] in run["scalars"] for run in runs)
    if flow_metric_present:
        result["all_flowreg_scalars_finite"] = all(
            run["scalars"].get(tag, {}).get("finite", False)
            for run in runs
            for tag in FLOWREG_TAGS
        )
        for tag in FLOWREG_TAGS:
            values = [float(run["scalars"][tag]["last"]) for run in runs if tag in run["scalars"]]
            result[f"{tag}_last_mean"] = _mean(values)
            result[f"{tag}_last_std"] = _stdev(values)

        result["flowreg_expected_updates_total"] = sum(expected_flow_updates(run) for run in runs)

    return result


def expected_flow_updates(run: dict[str, Any]) -> int:
    config = run.get("config", {})
    scalars = run.get("scalars", {})
    updates = scalars.get("FlowReg/Updates", {})
    if not config or not updates:
        return 0
    ppo_config = config.get("ppo", {})
    flow_config = config.get("flowreg", {})
    n_envs = int(config.get("n_envs", 1))
    n_steps = int(ppo_config.get("n_steps", 0))
    batch_size = int(ppo_config.get("batch_size", 1))
    n_epochs = int(ppo_config.get("n_epochs", 1))
    update_freq = int(flow_config.get("update_freq", 0))
    update_unit = str(flow_config.get("update_unit", "optimizer_step"))
    train_calls = int(updates.get("count", 0))
    if n_steps <= 0 or batch_size <= 0 or update_freq <= 0:
        return 0
    if update_unit == "train_call":
        return train_calls // update_freq
    minibatches_per_train = n_epochs * ((n_steps * n_envs) // batch_size)
    return (train_calls * minibatches_per_train) // update_freq


def total_logged_flow_updates(run: dict[str, Any]) -> int:
    run_dir = Path(run["run_dir"])
    event_files = [
        event_file
        for tensorboard_dir in sorted((run_dir / "tensorboard").glob("*"))
        for event_file in sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    ]
    if not event_files:
        return 0
    accumulator = event_accumulator.EventAccumulator(
        str(event_files[0]),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    if "FlowReg/Updates" not in accumulator.Tags().get("scalars", []):
        return 0
    return int(sum(value.value for value in accumulator.Scalars("FlowReg/Updates")))


def build_summary(specs: list[RunSpec]) -> dict[str, Any]:
    per_run: dict[str, dict[str, Any]] = {}
    for spec in specs:
        per_run.setdefault(spec.group, {})[str(spec.seed)] = summarize_run(spec)

    aggregate = {
        group: aggregate_group(list(seed_map.values())) for group, seed_map in per_run.items()
    }
    for group, seed_map in per_run.items():
        if not any("FlowReg/Updates" in run["scalars"] for run in seed_map.values()):
            continue
        for run in seed_map.values():
            run["scalars"]["FlowReg/Updates"]["total_logged"] = total_logged_flow_updates(run)
        aggregate[group]["flowreg_updates_total"] = sum(
            run["scalars"]["FlowReg/Updates"]["total_logged"]
            for run in seed_map.values()
            if "FlowReg/Updates" in run["scalars"]
        )

    return {"per_run": per_run, "aggregate": aggregate}


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = ["# Phase 4 Run Summary", ""]
    for group, aggregate in summary["aggregate"].items():
        lines.append(f"## {group}")
        lines.append("")
        lines.append(f"- runs: `{aggregate['num_runs']}`")
        lines.append(
            "- mean reward all episodes: "
            f"`{aggregate['mean_reward_all_mean']:.4f} +/- {aggregate['mean_reward_all_std']:.4f}`"
        )
        lines.append(
            "- mean reward last 20 episodes: "
            f"`{aggregate['mean_reward_last20_mean']:.4f} +/- "
            f"{aggregate['mean_reward_last20_std']:.4f}`"
        )
        lines.append(
            f"- positive episodes: `{aggregate['positive_episodes_total']} / "
            f"{aggregate['episodes_total']}`"
        )
        lines.append(f"- best observed training reward: `{aggregate['max_reward_max']:.3f}`")
        if "all_flowreg_scalars_finite" in aggregate:
            lines.append(f"- FlowReg scalars finite: `{aggregate['all_flowreg_scalars_finite']}`")
            lines.append(
                f"- FlowReg updates: `{aggregate['flowreg_updates_total']}` logged, "
                f"`{aggregate['flowreg_expected_updates_total']}` expected"
            )
            for tag in FLOWREG_TAGS:
                lines.append(
                    f"- {tag}: `{aggregate[f'{tag}_last_mean']:.4f} +/- "
                    f"{aggregate[f'{tag}_last_std']:.4f}`"
                )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local PPO/FlowReg run directories.")
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run_spec,
        required=True,
        help="Run spec in GROUP:SEED:PATH format. Can be repeated.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    summary = build_summary(args.run)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_output is not None:
        write_markdown(summary, args.markdown_output)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
