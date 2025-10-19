#!/usr/bin/env python3
"""Experiment harness for the Parallel Optimization Engine."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import poe_engine
except ImportError as exc:  # pragma: no cover - informative message
    raise SystemExit(
        "Python bindings not found. Build the project with -DENABLE_PYBIND=ON "
        "and ensure the module is in PYTHONPATH."
    ) from exc


@dataclass
class ExperimentConfig:
    agents: int
    iterations: int
    learning_rate: float
    seed: int
    device: str

    def to_engine_config(self, mode: str) -> "poe_engine.EngineConfig":
        cfg = poe_engine.EngineConfig()
        cfg.num_agents = int(self.agents)
        cfg.iterations = int(self.iterations)
        cfg.learning_rate = float(self.learning_rate)
        cfg.seed = int(self.seed)
        cfg.mode = mode
        cfg.device = self.device
        cfg.verbose = False
        return cfg


class GradientPredictor:
    """Simple linear regression predictor for gradient extrapolation."""

    def predict(self, gradients: List[float]) -> float:
        if not gradients:
            return 0.0
        if len(gradients) == 1:
            return gradients[-1]
        x = np.arange(len(gradients), dtype=np.float64)
        y = np.asarray(gradients, dtype=np.float64)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope * len(gradients) + intercept


@dataclass
class ExperimentResult:
    mode: str
    runtime: float
    final_x: float
    final_cost: float
    analytical_optimum: float
    history: List[float]
    gradient_history: List[float]


def run_engine(config: ExperimentConfig, mode: str) -> ExperimentResult:
    cfg = config.to_engine_config(mode)
    start = time.perf_counter()
    result = poe_engine.run(cfg)
    runtime = time.perf_counter() - start
    return ExperimentResult(
        mode=mode,
        runtime=runtime,
        final_x=result.final_x,
        final_cost=result.final_cost,
        analytical_optimum=result.analytical_optimum,
        history=list(result.history),
        gradient_history=list(result.gradient_history),
    )


def execute_experiments(config: ExperimentConfig, repeats: int) -> Dict[str, List[ExperimentResult]]:
    records: Dict[str, List[ExperimentResult]] = {"collaborative": [], "naive": []}
    for repeat in range(repeats):
        seed = config.seed + repeat
        local_cfg = ExperimentConfig(
            agents=config.agents,
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            seed=seed,
            device=config.device,
        )
        records["collaborative"].append(run_engine(local_cfg, "collaborative"))
        records["naive"].append(run_engine(local_cfg, "naive"))
    return records


def summarise_results(records: Dict[str, List[ExperimentResult]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for mode, runs in records.items():
        summary[mode] = {
            "runtime_mean": float(np.mean([run.runtime for run in runs])),
            "runtime_std": float(np.std([run.runtime for run in runs])),
            "final_cost_mean": float(np.mean([run.final_cost for run in runs])),
            "accuracy_mean": float(np.mean([abs(run.final_x - run.analytical_optimum) for run in runs])),
        }
    return summary


def apply_ml_predictor(result: ExperimentResult, lr: float) -> float:
    predictor = GradientPredictor()
    predicted_gradient = predictor.predict(result.gradient_history)
    predicted_x = result.history[-1] - lr * predicted_gradient if result.history else 0.0
    return predicted_x


def plot_histories(collaborative: ExperimentResult, naive: ExperimentResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "convergence.png"
    plt.figure(figsize=(10, 6))
    plt.plot(collaborative.history, label="Collaborative")
    plt.plot(naive.history, label="Naive", linestyle="--")
    plt.axhline(collaborative.analytical_optimum, color="black", linestyle=":", label="Analytical optimum")
    plt.xlabel("Iteration")
    plt.ylabel("x estimate")
    plt.title("Convergence comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=128, help="Number of agents")
    parser.add_argument("--iters", type=int, default=1000, help="Optimization iterations")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"], help="Execution device")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions")
    parser.add_argument("--output", type=Path, default=Path("python/output"), help="Directory for plots")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        agents=args.agents,
        iterations=args.iters,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
    )

    records = execute_experiments(cfg, args.repeats)
    summary = summarise_results(records)

    collab_best = min(records["collaborative"], key=lambda r: r.final_cost)
    naive_best = min(records["naive"], key=lambda r: r.final_cost)

    predicted_x = apply_ml_predictor(collab_best, cfg.learning_rate)

    plot_path = plot_histories(collab_best, naive_best, args.output)

    report = {
        "config": asdict(cfg),
        "summary": summary,
        "predicted_next_x": predicted_x,
        "convergence_plot": str(plot_path),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

