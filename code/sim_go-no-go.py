# sim_go-no-go.py — synthetic RT data for a Go/No-Go task with participant groups
# Simulates three groups of participants: control, frontal, non-frontal

import json
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = '/data/data'

os.makedirs(DATA_DIR, exist_ok=True)

DESIGN_PATH = os.path.join(DATA_DIR, "experimental_design.json")
DATA_PATH   = os.path.join(DATA_DIR, "simulated_rt_data_go_nogo.csv")

# ---------------------------------------------------------------------------
# SECTION 1 — Build experimental design dictionary and save as JSON
# ---------------------------------------------------------------------------

design = {
    "experiment_name": "Go/No-Go Reaction Time Task",
    "task": "simple go/no-go task",
    "timing": {
        "stimulus_duration_ms": 1000,
        "response_window_ms": 3000,
        "isi_ms": 3000,
    },
    "stimuli": {
        "go_stimulus": "A",
        "nogo_stimulus": "E",
        "note": "Equally distributed go and no-go trials",
    },
    "participant_groups": {
        "control": {
            "description": "Healthy control group",
            "mean_rt_ms": 486.08,
            "sd_rt_ms": 127.3,
            "go_rate": 0.9915,           # Rate of performed reactions (go trials with response)
            "false_positive_rate": 0.0178,  # False alarms on no-go trials
        },
        "frontal": {
            "description": "Participants with frontal lobe lesions",
            "mean_rt_ms": 612.35,
            "sd_rt_ms": 180.24,
            "go_rate": 0.963,           # Lower go rate (more omissions)
            "false_positive_rate": 0.0234,  # Higher impulsivity on no-go trials
        },
        "non-frontal": {
            "description": "Participants with non-frontal lobe lesions",
            "mean_rt_ms": 569.99,
            "sd_rt_ms": 151.25,
            "go_rate": 0.9719,
            "false_positive_rate": 0.0441,
        },
    },
    "trials_per_participant": 80,  # 40 go, 40 no-go
    "n_participants_per_group": 20,
}

with open(DESIGN_PATH, "w") as f:
    json.dump(design, f, indent=2)

print(f"Design saved to: {DESIGN_PATH}")

# ---------------------------------------------------------------------------
# SECTION 2 — Simulation helpers
# ---------------------------------------------------------------------------

# Group parameters
GROUP_PARAMS = {
    "control": {
        "mean_rt": 350,
        "sd_rt": 60,
        "go_rate": 0.95,
        "false_positive_rate": 0.05,
    },
    "frontal": {
        "mean_rt": 450,
        "sd_rt": 100,
        "go_rate": 0.88,
        "false_positive_rate": 0.25,
    },
    "non-frontal": {
        "mean_rt": 380,
        "sd_rt": 75,
        "go_rate": 0.92,
        "false_positive_rate": 0.08,
    },
}

def truncated_normal_rt(mean, sd, n, rng, min_rt=100, max_rt=3000):
    """Sample n RTs from a truncated normal distribution."""
    rt = rng.normal(mean, sd, n)
    return np.clip(rt, min_rt, max_rt)


def build_trial_list(nparticipant_id_trials, rng):
    """Return a list of trial dicts with balanced go/no-go."""
    trials = []
    half = n_trials // 2
    trial_types = ["go"] * half + ["nogo"] * half
    rng.shuffle(trial_types)
    
    for i, trial_type in enumerate(trial_types):
        trials.append(participant_id{
            "trial_index": i + 1,
            "trial_type": trial_type,
        })
    return trials


def simulate_participant(participant_id, group, n_trials, rng):
    """Simulate all trials for one participant in a specific group."""
    params = GROUP_PARAMS[group]
    rows = []
    
    for trial in build_trial_list(n_trials, rng):
        is_go = trial["trial_type"] == "go"
        
        if is_go:
            # For go trials: response based on go_rate
            if rng.random() < params["go_rate"]:
                rt = float(truncated_normal_rt(params["mean_rt"], params["sd_rt"], 1, rng)[0])
                response = "touch"
                correct = True
            else:
                # Omission (no response)
                rt = np.nan
                response = "omission"
                correct = False
        else:
            # For no-go trials: false alarm based on false_positive_rate
            if rng.random() < params["false_positive_rate"]:
                # False alarm (incorrect response on no-go)
                rt = float(truncated_normal_rt(params["mean_rt"] + 50, params["sd_rt"], 1, rng)[0])
                response = "touch"
                correct = False
            else:
                # Correct withholding of response
                rt = np.nan
                response = "withheld"
                correct = True
        
        rows.append({
            "participant_id": participant_id,
            "group": group,
            "trial_index": trial["trial_index"],
            "trial_type": trial["trial_type"],
            "response": response,
            "correct": correct,
            "rt_ms": round(rt, 2) if not np.isnan(rt) else np.nan,
        })
    
    return rows

# ---------------------------------------------------------------------------
# SECTION 3 — Simulate participants from each group and save to CSV
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n_participants_per_group = 20
n_trials = 80

all_rows = []
for group in ["control", "frontal", "non-frontal"]:
    for pid in range(1, n_participants_per_group + 1):
        participant_id = f"{group}_{pid:02d}"
        all_rows.extend(simulate_participant(participant_id, group, n_trials, rng))

df = pd.DataFrame(all_rows)
df.to_csv(DATA_PATH, index=False)

print(f"Simulated data saved to: {DATA_PATH}")
print(f"  Groups           : {list(GROUP_PARAMS.keys())}")
print(f"  Participants     : {n_participants_per_group} per group ({3 * n_participants_per_group} total)")
print(f"  Trials per person: {n_trials}")
print(f"  Total trials     : {len(df)}")
print(f"  Columns          : {list(df.columns)}")
