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


def make_plan(preset, top_k, temperature, seed):
    weights = PRESETS[preset]
    return {
        "schema_version": 1,
        "preset": preset,
        "seed": seed,
        "objective": "next_token_ce + forward_kl (weighted)",
        "routing": "random_token" if weights is None else "by_domain",
        "kd_weight_by_domain": None if weights is None else dict(zip(DOMAINS, weights)),
        "lm_weight_by_domain": None if weights is None else dict(
            zip(DOMAINS, (1.0 - weight for weight in weights))
        ),
        "planned_kd_weight": MATCHED_KD_WEIGHT if weights is None else sum(
            weight * count for weight, count in zip(weights, DOMAIN_TOKEN_COUNTS)
        ) / sum(DOMAIN_TOKEN_COUNTS),
        "domain_token_counts": dict(zip(DOMAINS, DOMAIN_TOKEN_COUNTS)),
        "teacher_top_k": top_k,
        "teacher_temperature": temperature,
        "student_temperature": 1.0,
    }


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
    teacher_logits[rows, labels] += 1.7
    teacher_logits[rows, (labels + 1) % vocabulary] += 2.0
    support = np.argsort(-teacher_logits, axis=-1, kind="stable")[:, :plan["teacher_top_k"]]
    teacher_top_logits = teacher_logits[rows[:, None], support]
    teacher_top_logits -= teacher_top_logits[:, :1]
    with np.errstate(over="ignore"):
        scaled_teacher_logits = teacher_top_logits / plan["teacher_temperature"]
    teacher_probs = np.exp(log_softmax(scaled_teacher_logits))
    observed_mass = (teacher_probs * (support == labels[:, None])).sum(axis=-1)

    weights = plan["kd_weight_by_domain"]
    if weights is None:
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--preset", choices=PRESETS)
    selection.add_argument("--matrix", action="store_true", help="include controls and single-factor sweeps")
    parser.add_argument("--top-k", type=int, help="teacher support size, 1 to 256")
    parser.add_argument("--temperature", type=float)
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
    if args.matrix and (args.top_k is not None or args.temperature is not None):
        parser.error("--matrix uses a fixed grid. Select --preset for custom parameters.")
    if args.preset == "lm" and (args.top_k is not None or args.temperature is not None):
        parser.error("LM-only ignores the teacher. Use a KD preset to vary its settings.")

    if args.matrix:
        settings = [(preset, 256, 1.0) for preset in PRESETS]
        settings += [("kd", k, 1.0) for k in TOP_K_VALUES if k != 256]
        settings += [("kd", 256, temperature) for temperature in (0.5, 0.25)]
    else:
        settings = [(args.preset or "domain", args.top_k or 256, args.temperature or 1.0)]
    plans = [make_plan(*setting, args.seed) for setting in settings]

    if args.smoke:
        try:
            for plan in plans:
                plan["smoke"] = run_smoke(plan)
        except ModuleNotFoundError as error:
            if error.name != "numpy":
                raise
            parser.error("--smoke needs NumPy. Install it with python -m pip install numpy.")

    print(json.dumps(plans if args.matrix else plans[0], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
