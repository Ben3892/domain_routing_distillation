"""Resolve a controlled distillation comparison or run it on synthetic CPU data."""

import argparse
import json
import math

DOMAINS = ("general", "math", "code")
DOMAIN_TOKEN_COUNTS = (33, 40, 27)
MATCHED_KD_WEIGHT = DOMAIN_TOKEN_COUNTS[0] / sum(DOMAIN_TOKEN_COUNTS)
PRESETS = {
    "lm": (0.0, 0.0, 0.0),
    "kd": (1.0, 1.0, 1.0),
    "domain": (1.0, 0.0, 0.0),
    "matched-mix": (MATCHED_KD_WEIGHT,) * len(DOMAINS),
    "matched-random": None,
}
TOP_K_VALUES = (1, 4, 16, 64, 256)
PAPER_ROUTING_THRESHOLDS = {
    "otmass": (0.3678, 0.9876),
    "entropy": (0.1, 0.5),
}


def make_plan(preset, top_k, temperature, seed, threshold=None):
    token_routing = preset in PAPER_ROUTING_THRESHOLDS
    weights = None if token_routing else PRESETS[preset]
    plan = {
        "schema_version": 1,
        "preset": preset,
        "seed": seed,
        "objective": "next_token_ce + forward_kl (weighted)",
        "routing": "random_token" if preset == "matched-random" else "by_domain",
        "kd_weight_by_domain": None if weights is None else dict(zip(DOMAINS, weights)),
        "lm_weight_by_domain": None if weights is None else dict(
            zip(DOMAINS, (1.0 - weight for weight in weights))
        ),
        "planned_kd_weight": MATCHED_KD_WEIGHT,
        "domain_token_counts": dict(zip(DOMAINS, DOMAIN_TOKEN_COUNTS)),
        "teacher_top_k": top_k,
        "teacher_temperature": temperature,
        "student_temperature": 1.0,
    }
    if token_routing:
        plan["routing"] = preset
        plan["routing_threshold"] = (
            PAPER_ROUTING_THRESHOLDS[preset][0] if threshold is None else threshold
        )
        plan["kd_when"] = "below_threshold" if preset == "otmass" else "above_threshold"
        plan["planned_kd_weight"] = None
    elif weights is not None:
        plan["planned_kd_weight"] = sum(
            weight * count for weight, count in zip(weights, DOMAIN_TOKEN_COUNTS)
        ) / sum(DOMAIN_TOKEN_COUNTS)
    return plan


def run_smoke(plan):
    import numpy as np

    def log_softmax(logits):
        centered = logits - logits.max(axis=-1, keepdims=True)
        return centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))

    rng = np.random.default_rng(plan["seed"])
    vocabulary = 320
    num_contexts = 8
    domains = np.repeat(np.arange(len(DOMAINS)), DOMAIN_TOKEN_COUNTS)
    rows = np.arange(len(domains))

    def fixture():
        contexts = rng.integers(num_contexts, size=len(rows))
        labels = (contexts * 11 + domains * 79 + rng.integers(3, size=len(rows))) % vocabulary
        return contexts, labels

    contexts, labels = fixture()
    valid_contexts, valid_labels = fixture()

    # Produce one dense teacher cache shared across all presets and sweeps.
    teacher_logits = rng.normal(size=(len(rows), vocabulary))
    # Give observed labels varied teacher confidence so both threshold directions are exercised.
    confidence_boost = np.resize([0.0, 6.5, 9.0, 13.0], len(rows))
    teacher_logits[rows, labels] += 1.7 + rng.permutation(confidence_boost)
    teacher_logits[rows, (labels + 1) % vocabulary] += 2.0
    support = np.argsort(-teacher_logits, axis=-1, kind="stable")[:, :plan["teacher_top_k"]]
    teacher_top_logits = teacher_logits[rows[:, None], support]
    teacher_top_logits -= teacher_top_logits[:, :1]
    with np.errstate(over="ignore"):
        scaled_teacher_logits = teacher_top_logits / plan["teacher_temperature"]
    teacher_log_probs = log_softmax(scaled_teacher_logits)
    teacher_probs = np.exp(teacher_log_probs)
    entropy_terms = np.zeros_like(teacher_probs)
    np.multiply(teacher_probs, teacher_log_probs, out=entropy_terms, where=teacher_probs > 0)
    entropy = -entropy_terms.sum(axis=-1)
    observed_mass = (teacher_probs * (support == labels[:, None])).sum(axis=-1)

    weights = plan["kd_weight_by_domain"]
    if plan["routing"] == "otmass":
        kd_weight = (observed_mass < plan["routing_threshold"]).astype(float)
    elif plan["routing"] == "entropy":
        kd_weight = (entropy > plan["routing_threshold"]).astype(float)
    elif plan["routing"] == "random_token":
        route_rng = np.random.default_rng(plan["seed"] + 1)
        num_kd_tokens = DOMAIN_TOKEN_COUNTS[0]
        kd_weight = np.zeros(len(rows))
        kd_weight[route_rng.permutation(len(rows))[:num_kd_tokens]] = 1.0
    else:
        kd_weight = np.array([weights[domain] for domain in DOMAINS])[domains]

    # The weighted CE + forward KL gradient is p_student - q_mixture.
    target = np.zeros_like(teacher_logits)
    target[rows, labels] = 1.0 - kd_weight
    target[rows[:, None], support] += kd_weight[:, None] * teacher_probs
    logits = np.zeros((num_contexts, vocabulary))
    updates = 20
    for _ in range(updates):
        gradient = np.zeros_like(logits)
        token_gradient = np.exp(log_softmax(logits[contexts])) - target
        np.add.at(gradient, contexts, token_gradient)
        logits -= gradient / len(rows)

    valid_ce = -log_softmax(logits[valid_contexts])[rows, valid_labels]
    return {
        "data": "synthetic",
        "updates": updates,
        "validation_ntp_ce": float(valid_ce.mean()),
        "validation_ntp_ce_by_domain": {
            domain: float(valid_ce[domains == i].mean()) for i, domain in enumerate(DOMAINS)
        },
        "effective_kd_weight_by_domain": {
            domain: float(kd_weight[domains == i].mean()) for i, domain in enumerate(DOMAINS)
        },
        "effective_kd_weight": float(kd_weight.mean()),
        "teacher_coverage_at_k": float((support == labels[:, None]).any(axis=-1).mean()),
        "teacher_observed_token_mass": float(observed_mass.mean()),
        "teacher_entropy": float(entropy.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--preset", choices=(*PRESETS, *PAPER_ROUTING_THRESHOLDS))
    selection.add_argument("--matrix", action="store_true", help="include matched controls and sweeps")
    selection.add_argument("--paper-matrix", action="store_true", help="include all paper treatments")
    parser.add_argument("--top-k", type=int, help="teacher support size, 1 to 256")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--routing-threshold", type=float, help="OTMass or entropy threshold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="train a synthetic CPU model with NumPy")
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.top_k is not None and not 1 <= args.top_k <= max(TOP_K_VALUES):
        parser.error("--top-k must be between 1 and 256")
    if args.temperature is not None and (
        not math.isfinite(args.temperature) or args.temperature <= 0
    ):
        parser.error("--temperature must be finite and positive")
    if (args.matrix or args.paper_matrix) and (args.top_k is not None or args.temperature is not None):
        parser.error("The matrices use fixed grids. Select --preset for custom parameters.")
    if args.routing_threshold is not None:
        if args.preset not in PAPER_ROUTING_THRESHOLDS:
            parser.error("--routing-threshold requires --preset otmass or --preset entropy")
        if not math.isfinite(args.routing_threshold) or args.routing_threshold < 0:
            parser.error("--routing-threshold must be finite and nonnegative")
        if args.preset == "otmass" and args.routing_threshold > 1:
            parser.error("OTMass cannot exceed 1")
    if args.preset == "lm" and (args.top_k is not None or args.temperature is not None):
        parser.error("LM-only ignores the teacher. Use a KD preset to vary its settings.")

    if args.matrix or args.paper_matrix:
        presets = PRESETS if args.matrix else ("lm", "kd", "domain")
        plans = [make_plan(preset, 256, 1.0, args.seed) for preset in presets]
        plans += [make_plan("kd", k, 1.0, args.seed) for k in TOP_K_VALUES if k != 256]
        plans += [make_plan("kd", 256, temperature, args.seed) for temperature in (0.5, 0.25)]
        if args.paper_matrix:
            plans += [
                make_plan(preset, 256, 1.0, args.seed, threshold)
                for preset, thresholds in PAPER_ROUTING_THRESHOLDS.items()
                for threshold in thresholds
            ]
    else:
        plans = [make_plan(
            args.preset or "domain", args.top_k or 256, args.temperature or 1.0,
            args.seed, args.routing_threshold,
        )]

    if args.smoke:
        try:
            for plan in plans:
                plan["smoke"] = run_smoke(plan)
        except ModuleNotFoundError as error:
            if error.name != "numpy":
                raise
            parser.error("--smoke needs NumPy. Install it with python -m pip install numpy.")

    print(json.dumps(plans if args.matrix or args.paper_matrix else plans[0], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
