"""
Searchable CNN architecture space for edge devices.

An architecture is a JSON-serialisable dict ("config"):

{
  "resolution": 32,                  # variable input resolution
  "stem_channels": 24,
  "stages": [                        # variable depth: 2-4 stages x 1-3 blocks
     {"block": "dwsep",              # conv | dwsep (depthwise separable) | mbconv
      "depth": 2,                    # blocks in this stage
      "channels": 48,                # variable filters/channels
      "kernel": 5,                   # 3 | 5 | 7
      "expand": 4,                   # expansion ratio (mbconv only)
      "activation": "hswish"},       # relu | relu6 | hswish | silu
     ...
  ],
  "head_channels": 256,              # 0 = no 1x1 head conv
  "dropout": 0.1
}

Each stage starts with a stride-2 block except the first, so a 4-stage network
down-samples 8x and a 2-stage network 2x.

The same space is used by both search strategies:
  * `sample_config(rng)` / `mutate(config, rng)` -> evolutionary search
  * `suggest_config(trial)`                     -> Optuna (TPE) search
"""
import copy
import hashlib
import json
import random

SPACE = {
    "resolution": [24, 28, 32],
    "stem_channels": [16, 24, 32],
    "num_stages": [2, 3, 4],
    "depth": [1, 2, 3],
    "channels": [16, 24, 32, 48, 64, 96, 128, 160],
    "kernel": [3, 5, 7],
    "block": ["conv", "dwsep", "mbconv"],
    "expand": [2, 4, 6],
    "activation": ["relu", "relu6", "hswish", "silu"],
    "head_channels": [0, 128, 256, 512],
    "dropout": [0.0, 0.1, 0.2],
}

STAGE_KEYS = ["block", "depth", "channels", "kernel", "expand", "activation"]


def _random_stage(rng, prev_channels=None):
    ch = SPACE["channels"]
    # bias towards non-decreasing width (good CNN prior) but keep it random
    if prev_channels is not None and rng.random() < 0.8:
        ch = [c for c in ch if c >= prev_channels] or ch
    return {
        "block": rng.choice(SPACE["block"]),
        "depth": rng.choice(SPACE["depth"]),
        "channels": rng.choice(ch),
        "kernel": rng.choice(SPACE["kernel"]),
        "expand": rng.choice(SPACE["expand"]),
        "activation": rng.choice(SPACE["activation"]),
    }


def sample_config(rng=None):
    """Uniformly sample a random architecture from the search space."""
    rng = rng or random.Random()
    n = rng.choice(SPACE["num_stages"])
    stages, prev = [], None
    for _ in range(n):
        s = _random_stage(rng, prev)
        stages.append(s)
        prev = s["channels"]
    return {
        "resolution": rng.choice(SPACE["resolution"]),
        "stem_channels": rng.choice(SPACE["stem_channels"]),
        "stages": stages,
        "head_channels": rng.choice(SPACE["head_channels"]),
        "dropout": rng.choice(SPACE["dropout"]),
    }


def mutate(config, rng=None, n_mutations=1):
    """Return a copy of `config` with `n_mutations` random single-gene changes."""
    rng = rng or random.Random()
    c = copy.deepcopy(config)
    for _ in range(n_mutations):
        gene = rng.choice(["resolution", "stem_channels", "head_channels", "dropout",
                           "add_stage", "remove_stage", "stage", "stage", "stage"])
        if gene in ("resolution", "stem_channels", "head_channels", "dropout"):
            options = [v for v in SPACE[gene] if v != c[gene]]
            c[gene] = rng.choice(options)
        elif gene == "add_stage" and len(c["stages"]) < max(SPACE["num_stages"]):
            c["stages"].append(_random_stage(rng, c["stages"][-1]["channels"]))
        elif gene == "remove_stage" and len(c["stages"]) > min(SPACE["num_stages"]):
            c["stages"].pop(rng.randrange(len(c["stages"])))
        else:
            s = rng.choice(c["stages"])
            key = rng.choice(STAGE_KEYS)
            options = [v for v in SPACE[key] if v != s[key]]
            s[key] = rng.choice(options)
    return c


def suggest_config(trial):
    """Define-by-run version of the space for Optuna."""
    n = trial.suggest_categorical("num_stages", SPACE["num_stages"])
    stages = []
    for i in range(n):
        stages.append({
            "block": trial.suggest_categorical(f"s{i}_block", SPACE["block"]),
            "depth": trial.suggest_categorical(f"s{i}_depth", SPACE["depth"]),
            "channels": trial.suggest_categorical(f"s{i}_channels", SPACE["channels"]),
            "kernel": trial.suggest_categorical(f"s{i}_kernel", SPACE["kernel"]),
            "expand": trial.suggest_categorical(f"s{i}_expand", SPACE["expand"]),
            "activation": trial.suggest_categorical(f"s{i}_activation", SPACE["activation"]),
        })
    return {
        "resolution": trial.suggest_categorical("resolution", SPACE["resolution"]),
        "stem_channels": trial.suggest_categorical("stem_channels", SPACE["stem_channels"]),
        "stages": stages,
        "head_channels": trial.suggest_categorical("head_channels", SPACE["head_channels"]),
        "dropout": trial.suggest_categorical("dropout", SPACE["dropout"]),
    }


def config_id(config):
    """Stable short hash used to de-duplicate / cache evaluated architectures."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]


def describe(config):
    """Compact human-readable string, e.g. 'r32 s24 | dwsep2x48k5-hswish | ... | h256'."""
    parts = [f"r{config['resolution']} s{config['stem_channels']}"]
    for s in config["stages"]:
        e = f"e{s['expand']}" if s["block"] == "mbconv" else ""
        parts.append(f"{s['block']}{s['depth']}x{s['channels']}k{s['kernel']}{e}-{s['activation']}")
    parts.append(f"h{config['head_channels']}")
    return " | ".join(parts)


def total_depth(config):
    return sum(s["depth"] for s in config["stages"])
