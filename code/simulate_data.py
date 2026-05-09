# simulate_data.py — synthetic RT data for a Go/No-Go task
# Run interactively, section by section.

import json
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

DESIGN_PATH = os.path.join(DATA_DIR, "experimental_design.json")
DATA_PATH   = os.path.join(DATA_DIR, "simulated_rt_data.csv")

# ---------------------------------------------------------------------------
# SECTION 1 — Build experimental design dictionary and save as JSON
# ---------------------------------------------------------------------------

design = {
    "experiment_name": "Go/No-Go Reaction Time Task",
    "n_conditions": 4,
    "timing": {
        "stimulus_duration_ms": 1000,
        "response_window_ms": 3000,
        "isi_ms": 3000,
    },
    "stimuli": {
        "letters": ["A", "E"],
        "locations": ["location_1", "location_2"],
        "note": "Letters and locations are equally distributed within each condition.",
    },
    "conditions": [
        {
            "condition_id": 1,
            "name": "Standard Go/No-Go",
            "rule": "letter-based",
            "go_stimulus":   {"letter": "A", "response": "touch screen"},
            "nogo_stimulus": {"letter": "E", "response": "withhold"},
            "location_relevant": False,
            "practice_trials": 4,
            "test_trials": 12,
        },
        {
            "condition_id": 2,
            "name": "Reversed Go/No-Go",
            "rule": "letter-based (reversed)",
            "go_stimulus":   {"letter": "E", "response": "touch screen"},
            "nogo_stimulus": {"letter": "A", "response": "withhold"},
            "location_relevant": False,
            "practice_trials": 4,
            "test_trials": 12,
        },
        {
            "condition_id": 3,
            "name": "Location-based Go/No-Go (go at location_1)",
            "rule": "location-based",
            "go_stimulus":   {"location": "location_1", "letters": ["A", "E"], "response": "touch screen"},
            "nogo_stimulus": {"location": "location_2", "letters": ["A", "E"], "response": "withhold"},
            "location_relevant": True,
            "practice_trials": 4,
            "test_trials": 12,
        },
        {
            "condition_id": 4,
            "name": "Location-based Go/No-Go (go at location_2)",
            "rule": "location-based (alternative location)",
            "go_stimulus":   {"location": "location_2", "letters": ["A", "E"], "response": "touch screen"},
            "nogo_stimulus": {"location": "location_1", "letters": ["A", "E"], "response": "withhold"},
            "location_relevant": True,
            "practice_trials": 4,
            "test_trials": 12,
        },
    ],
    "response_device": "touchscreen",
    "n_participants_simulated": 100,
}

with open(DESIGN_PATH, "w") as f:
    json.dump(design, f, indent=2)
print(f"Design saved to: {DESIGN_PATH}")

# ---------------------------------------------------------------------------
# SECTION 2 — Simulation helpers
# ---------------------------------------------------------------------------

# Ex-Gaussian parameters per condition; conditions 3 & 4 are slightly harder.
CONDITION_PARAMS = {
    1: {"mu": 280, "sigma": 40,  "tau": 80,  "omission_rate": 0.05, "false_alarm_rate": 0.08},
    2: {"mu": 310, "sigma": 50,  "tau": 90,  "omission_rate": 0.08, "false_alarm_rate": 0.10},
    3: {"mu": 340, "sigma": 55,  "tau": 100, "omission_rate": 0.10, "false_alarm_rate": 0.12},
    4: {"mu": 340, "sigma": 55,  "tau": 100, "omission_rate": 0.10, "false_alarm_rate": 0.12},
}

BETWEEN_SUBJECT_SD_MU  = 30  # ms
BETWEEN_SUBJECT_SD_TAU = 20  # ms


def ex_gaussian_rt(mu, sigma, tau, n, rng):
    """Sample n RTs (ms) from an ex-Gaussian, clipped to [100, 3000]."""
    rt = rng.normal(mu, sigma, n) + rng.exponential(tau, n)
    return np.clip(rt, 100, 3000)


def build_trial_list(condition):
    """Return a list of trial dicts for one condition with balanced go/no-go."""
    location_relevant = condition["location_relevant"]
    letters   = ["A", "E"]
    locations = ["location_1", "location_2"]

    def make_trials(n, phase):
        trials = []
        half = n // 2
        if not location_relevant:
            go_letter   = condition["go_stimulus"]["letter"]
            nogo_letter = condition["nogo_stimulus"]["letter"]
            letter_seq  = [go_letter] * half + [nogo_letter] * half
            np.random.shuffle(letter_seq)
            for i, letter in enumerate(letter_seq):
                trials.append({
                    "phase": phase, "trial_index": i + 1,
                    "letter": letter,
                    "location": np.random.choice(locations),
                    "trial_type": "go" if letter == go_letter else "nogo",
                })
        else:
            go_loc   = condition["go_stimulus"]["location"]
            nogo_loc = condition["nogo_stimulus"]["location"]
            quarter  = n // 4
            combos   = (
                [("A", go_loc)] * quarter + [("E", go_loc)] * quarter +
                [("A", nogo_loc)] * quarter + [("E", nogo_loc)] * quarter
            )
            while len(combos) < n:
                combos.append((np.random.choice(letters), np.random.choice(locations)))
            combos = combos[:n]
            np.random.shuffle(combos)
            for i, (letter, loc) in enumerate(combos):
                trials.append({
                    "phase": phase, "trial_index": i + 1,
                    "letter": letter, "location": loc,
                    "trial_type": "go" if loc == go_loc else "nogo",
                })
        return trials

    practice = make_trials(condition["practice_trials"], "practice")
    test     = make_trials(condition["test_trials"], "test")
    for i, t in enumerate(test):
        t["trial_index"] = i + 1
    return practice + test


def simulate_participant(participant_id, conditions, rng):
    """Simulate all trials for one participant across all conditions."""
    rows = []
    for cond in conditions:
        cid    = cond["condition_id"]
        params = CONDITION_PARAMS[cid]

        subj_mu  = params["mu"]  + rng.normal(0, BETWEEN_SUBJECT_SD_MU)
        subj_tau = max(params["tau"] + rng.normal(0, BETWEEN_SUBJECT_SD_TAU), 10)

        for trial in build_trial_list(cond):
            is_go = trial["trial_type"] == "go"
            if is_go:
                if rng.random() < params["omission_rate"]:
                    rt, response, correct = np.nan, "omission", False
                else:
                    rt, response, correct = float(ex_gaussian_rt(subj_mu, params["sigma"], subj_tau, 1, rng)[0]), "touch", True
            else:
                if rng.random() < params["false_alarm_rate"]:
                    rt, response, correct = float(ex_gaussian_rt(subj_mu + 50, params["sigma"], subj_tau, 1, rng)[0]), "touch", False
                else:
                    rt, response, correct = np.nan, "withheld", True

            rows.append({
                "participant_id": participant_id,
                "condition_id":   cid,
                "condition_name": cond["name"],
                "phase":          trial["phase"],
                "trial_index":    trial["trial_index"],
                "letter":         trial["letter"],
                "location":       trial["location"],
                "trial_type":     trial["trial_type"],
                "response":       response,
                "correct":        correct,
                "rt_ms":          round(rt, 2) if not np.isnan(rt) else np.nan,
            })
    return rows

# ---------------------------------------------------------------------------
# SECTION 3 — Simulate 100 participants and save to CSV
# ---------------------------------------------------------------------------

rng           = np.random.default_rng(42)
n_participants = 100

all_rows = []
for pid in range(1, n_participants + 1):
    all_rows.extend(simulate_participant(pid, design["conditions"], rng))

df = pd.DataFrame(all_rows)
df.to_csv(DATA_PATH, index=False)

print(f"Simulated data saved to: {DATA_PATH}")
print(f"  Participants : {n_participants}")
print(f"  Total trials : {len(df)}")
print(f"  Columns      : {list(df.columns)}")
