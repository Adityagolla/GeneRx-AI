"""
pharmacovigilance.py — Proportional Reporting Ratio (PRR) signal detection.

This is deliberately kept separate from ml_model.py's per-event classifier.
PRR is a population-level statistic (it compares one drug's serious-outcome
rate against every other drug's rate across the whole dataset) — it cannot
vary per patient. Folding it into the row-level classifier's training target
would collapse the model into memorizing a drug-name-to-constant lookup and
throw away everything the patient-level features (age, concomitant meds,
etc.) contribute. So: PRR is computed once per drug and surfaced as its own
explicitly-labeled signal, alongside — not blended into — the per-event
reporting-pattern classifier in ml_model.py.

Standard pharmacovigilance signal-detection method (Evans et al.), using a
2x2 contingency table per drug:

                    Serious outcome    Not serious      Total
    This drug             a                b            a+b
    All other drugs       c                d            c+d

    PRR = [a / (a+b)] / [c / (c+d)]
    chi-square (Yates-corrected) = N(|ad-bc| - N/2)^2 / [(a+b)(c+d)(a+c)(b+d)]
    Signal detected (Evans criteria): PRR >= 2, chi-square >= 4, a >= 3

IMPORTANT CAVEAT — surfaced in the UI, not just here: the comparator ("all
other drugs") is only the other drugs in this project's own FAERS pull (10
common medications), not the full FDA database. A real pharmacovigilance PRR
compares against the entire universe of reported drugs. This is a
within-catalog relative signal, not a validated FDA-scale one — still a more
honest, standard-methodology statistic than raw per-event severity, but it
should not be read as an official safety signal.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SIGNALS_PATH = os.path.join(DATA_DIR, "drug_signals.json")


def compute_prr_signals(faers_df) -> dict:
    """
    Compute per-drug PRR + chi-square for "serious" outcomes across the
    drugs present in faers_df. Returns {drug_name: signal_dict}.
    """
    total_serious = int(faers_df["serious"].sum())
    total_events = len(faers_df)
    results = {}

    for drug in sorted(faers_df["drug_name"].unique()):
        drug_mask = faers_df["drug_name"] == drug
        n_drug = int(drug_mask.sum())
        a = int(faers_df.loc[drug_mask, "serious"].sum())
        b = n_drug - a
        c = total_serious - a
        d = (total_events - n_drug) - c
        n_total = a + b + c + d

        if a == 0 or (a + b) == 0 or (c + d) == 0 or n_total == 0:
            results[drug] = {
                "prr": None, "chi_square": None,
                "n_reports": n_drug, "n_serious": a,
                "signal_detected": False,
            }
            continue

        prr = (a / (a + b)) / (c / (c + d))

        # Yates continuity-corrected chi-square for a 2x2 table
        numerator = n_total * (abs(a * d - b * c) - n_total / 2) ** 2
        denominator = (a + b) * (c + d) * (a + c) * (b + d)
        chi_square = numerator / denominator if denominator else 0.0

        signal_detected = bool(prr >= 2 and chi_square >= 4 and a >= 3)

        results[drug] = {
            "prr": round(prr, 2),
            "chi_square": round(chi_square, 1),
            "n_reports": n_drug,
            "n_serious": a,
            "signal_detected": signal_detected,
        }

    return results


def save_signals(signals: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SIGNALS_PATH, "w") as f:
        json.dump(signals, f, indent=2)


_signals_cache = None


def load_signals() -> dict:
    """Lazily load the precomputed per-drug PRR signals (empty dict if not
    yet computed — callers should treat a missing entry as 'no signal data',
    not as an error)."""
    global _signals_cache
    if _signals_cache is None:
        if os.path.exists(SIGNALS_PATH):
            with open(SIGNALS_PATH) as f:
                _signals_cache = json.load(f)
        else:
            _signals_cache = {}
    return _signals_cache


def get_signal(drug_name: str) -> dict:
    signals = load_signals()
    return signals.get(drug_name.lower().strip())
