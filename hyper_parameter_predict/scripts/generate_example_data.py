#!/usr/bin/env python3
"""Generate clearly-labelled synthetic training data and candidate configurations."""

import argparse
import csv
import math
import random
from pathlib import Path

DOMAINS = ("stem", "general", "math", "code")
TEMPERATURES = (0.7, 0.9, 1.0, 1.2, 1.5, 2.0)
TOPKS = (8, 16, 32, 64, 128)
LM_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train-output", type=Path, required=True)
    result.add_argument("--candidate-output", type=Path, required=True)
    result.add_argument("--configs", type=int, default=240)
    result.add_argument("--seeds-per-config", type=int, default=3)
    result.add_argument("--candidates", type=int, default=2000)
    result.add_argument("--seed", type=int, default=2026)
    return result


def sample_config(rng: random.Random):
    row = {}
    for domain in DOMAINS:
        row[f"{domain}_temperature"] = rng.choice(TEMPERATURES)
        row[f"{domain}_topk"] = rng.choice(TOPKS)
        row[f"{domain}_lm_loss_weight"] = rng.choice(LM_WEIGHTS)
    return row


def synthetic_loss(row, rng: random.Random):
    # Artificial nonlinear response used only to make the example runnable.
    optimum = {
        "stem": (1.0, 32, 0.65),
        "general": (1.5, 64, 0.20),
        "math": (0.9, 16, 0.85),
        "code": (1.1, 32, 0.75),
    }
    loss = 1.75
    for domain, (best_t, best_k, best_w) in optimum.items():
        temperature = row[f"{domain}_temperature"]
        topk = row[f"{domain}_topk"]
        weight = row[f"{domain}_lm_loss_weight"]
        loss += 0.035 * (temperature - best_t) ** 2
        loss += 0.012 * (math.log2(topk) - math.log2(best_k)) ** 2
        loss += 0.11 * (weight - best_w) ** 2
        loss += 0.010 * (temperature - best_t) * (weight - best_w)
    loss += 0.018 * (
        row["math_lm_loss_weight"] - row["general_lm_loss_weight"] - 0.55
    ) ** 2
    return loss + rng.gauss(0.0, 0.006)


def main():
    args = parser().parse_args()
    rng = random.Random(args.seed)
    feature_names = [
        f"{domain}_{parameter}"
        for domain in DOMAINS
        for parameter in ("temperature", "topk", "lm_loss_weight")
    ]
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)

    configs = [sample_config(rng) for _ in range(args.configs)]
    with args.train_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment_id", "seed", *feature_names, "validation_loss"],
        )
        writer.writeheader()
        for config_id, config in enumerate(configs):
            for repeat in range(args.seeds_per_config):
                row = {
                    "experiment_id": f"synthetic-config-{config_id:04d}",
                    "seed": args.seed + repeat,
                    **config,
                }
                row["validation_loss"] = round(synthetic_loss(config, rng), 8)
                writer.writerow(row)

    with args.candidate_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", *feature_names])
        writer.writeheader()
        for candidate_id in range(args.candidates):
            writer.writerow(
                {
                    "candidate_id": f"candidate-{candidate_id:05d}",
                    **sample_config(rng),
                }
            )


if __name__ == "__main__":
    main()

