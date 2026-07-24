"""
drug_features.py — Pharmacology metadata and drug-drug interaction features,
shared by the training pipeline (build_dataset.py) and live inference
(ml_model.py) so the two can never silently drift apart. If training computes
a feature one way and serving computes it another, the model is being fed
garbage at prediction time without any error — this module is the single
source of truth to prevent that.

Pharmacology values (half-life, clearance route, protein binding, CYP
enzymes) are typical/representative figures for these well-established
drugs, intended for feature engineering in this educational tool — not a
substitute for a prescribing reference.
"""

# ─────────────────────────────────────────────────────────────────────────────
# PER-DRUG PHARMACOLOGY
# ─────────────────────────────────────────────────────────────────────────────
DRUG_PHARMACOLOGY = {
    "metformin": {
        "atc_code": "A10BA02",
        "half_life_hours": 6.2,
        "hepatic_metabolism": False,
        "renal_clearance_pct": 90,
        "protein_binding_pct": 0,
        "cyp_enzymes": [],
        "black_box_warning": True,  # lactic acidosis risk in renal impairment
    },
    "atorvastatin": {
        "atc_code": "C10AA05",
        "half_life_hours": 14.0,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 2,
        "protein_binding_pct": 98,
        "cyp_enzymes": ["CYP3A4"],
        "black_box_warning": False,
    },
    "amlodipine": {
        "atc_code": "C08CA01",
        "half_life_hours": 40.0,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 10,
        "protein_binding_pct": 93,
        "cyp_enzymes": ["CYP3A4"],
        "black_box_warning": False,
    },
    "ramipril": {
        "atc_code": "C09AA05",
        "half_life_hours": 15.0,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 60,
        "protein_binding_pct": 73,
        "cyp_enzymes": [],
        "black_box_warning": False,
    },
    "metoprolol": {
        "atc_code": "C07AB02",
        "half_life_hours": 5.0,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 10,
        "protein_binding_pct": 12,
        "cyp_enzymes": ["CYP2D6"],
        "black_box_warning": False,
    },
    "warfarin": {
        "atc_code": "B01AA03",
        "half_life_hours": 40.0,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 1,
        "protein_binding_pct": 99,
        "cyp_enzymes": ["CYP2C9", "CYP3A4", "CYP1A2"],
        "black_box_warning": True,  # bleeding risk
    },
    "amoxicillin": {
        "atc_code": "J01CA04",
        "half_life_hours": 1.3,
        "hepatic_metabolism": False,
        "renal_clearance_pct": 65,
        "protein_binding_pct": 20,
        "cyp_enzymes": [],
        "black_box_warning": False,
    },
    "ibuprofen": {
        "atc_code": "M01AE01",
        "half_life_hours": 2.5,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 1,
        "protein_binding_pct": 99,
        "cyp_enzymes": ["CYP2C9"],
        "black_box_warning": True,  # NSAID class CV/GI boxed warning
    },
    "acetaminophen": {
        "atc_code": "N02BE01",
        "half_life_hours": 2.5,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 3,
        "protein_binding_pct": 20,
        "cyp_enzymes": ["CYP2E1"],
        "black_box_warning": False,
    },
    "omeprazole": {
        "atc_code": "A02BC01",
        "half_life_hours": 1.0,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 1,
        "protein_binding_pct": 95,
        "cyp_enzymes": ["CYP2C19", "CYP3A4"],
        "black_box_warning": False,
    },
    "empagliflozin": {
        "atc_code": "A10BK03",
        "half_life_hours": 12.4,
        "hepatic_metabolism": True,
        "renal_clearance_pct": 41,
        "protein_binding_pct": 86,
        "cyp_enzymes": [],
        "black_box_warning": False,
    },
    "linagliptin": {
        "atc_code": "A10BH05",
        "half_life_hours": 12.0,
        "hepatic_metabolism": False,  # ~85% excreted unchanged via bile/feces
        "renal_clearance_pct": 5,
        "protein_binding_pct": 70,
        "cyp_enzymes": ["CYP3A4"],
        "black_box_warning": False,
    },
    "insulin glargine": {
        "atc_code": "A10AE04",
        "half_life_hours": 12.0,
        "hepatic_metabolism": False,  # proteolytic degradation, not CYP
        "renal_clearance_pct": 0,
        "protein_binding_pct": 0,
        "cyp_enzymes": [],
        "black_box_warning": False,
    },
}

PHARMACOLOGY_FEATURE_NAMES = [
    "half_life_hours",
    "hepatic_metabolism",
    "renal_clearance_pct",
    "protein_binding_pct",
    "cyp_enzyme_count",
    "black_box_warning",
]

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTION-RELEVANT DRUG CLASSES
#
# Keyword lists for classifying a *concomitant* drug (which may be anything in
# FAERS, not just our 13-drug catalog) by pharmacologic class, so we can build
# clinically meaningful interaction counts like "how many NSAIDs is this
# patient also on" instead of a single opaque num_concomitant_drugs. These are
# intentionally scoped to common, well-known agents in each class — a
# heuristic keyword match, not an exhaustive drug database.
# ─────────────────────────────────────────────────────────────────────────────
NSAID_KEYWORDS = [
    "ibuprofen", "naproxen", "diclofenac", "aspirin", "celecoxib",
    "meloxicam", "indomethacin", "ketorolac", "piroxicam",
]

QT_PROLONGING_KEYWORDS = [
    "amiodarone", "sotalol", "haloperidol", "ondansetron", "azithromycin",
    "ciprofloxacin", "methadone", "citalopram", "quetiapine", "erythromycin",
]

SEDATIVE_KEYWORDS = [
    "alprazolam", "diazepam", "lorazepam", "zolpidem", "morphine",
    "oxycodone", "clonazepam", "temazepam", "hydrocodone",
]

CYP3A4_INHIBITOR_KEYWORDS = [
    "clarithromycin", "ketoconazole", "fluconazole", "ritonavir",
    "diltiazem", "verapamil", "erythromycin", "itraconazole",
]

INTERACTION_FEATURE_NAMES = [
    "concomitant_nsaid_count",
    "concomitant_qt_prolonging_count",
    "concomitant_sedative_count",
    "concomitant_cyp3a4_inhibitor_count",
]


def _count_matches(concomitant_names, keywords):
    count = 0
    for name in concomitant_names:
        name_lower = name.lower()
        if any(kw in name_lower for kw in keywords):
            count += 1
    return count


def get_pharmacology_features(drug_name: str) -> dict:
    """Look up static pharmacology features for a drug. Falls back to neutral
    zeros for anything not in DRUG_PHARMACOLOGY (e.g. an unrecognized name)."""
    info = DRUG_PHARMACOLOGY.get(drug_name.lower().strip())
    if info is None:
        return {name: 0 for name in PHARMACOLOGY_FEATURE_NAMES}
    return {
        "half_life_hours": info["half_life_hours"],
        "hepatic_metabolism": int(info["hepatic_metabolism"]),
        "renal_clearance_pct": info["renal_clearance_pct"],
        "protein_binding_pct": info["protein_binding_pct"],
        "cyp_enzyme_count": len(info["cyp_enzymes"]),
        "black_box_warning": int(info["black_box_warning"]),
    }


def get_interaction_features(concomitant_names) -> dict:
    """
    Compute interaction-relevant counts from a list of concomitant drug
    names (as free-text strings — from FAERS 'medicinalproduct' fields at
    training time, or from the live patient's current_meds at serving time).
    """
    names = [n for n in (concomitant_names or []) if n]
    return {
        "concomitant_nsaid_count": _count_matches(names, NSAID_KEYWORDS),
        "concomitant_qt_prolonging_count": _count_matches(names, QT_PROLONGING_KEYWORDS),
        "concomitant_sedative_count": _count_matches(names, SEDATIVE_KEYWORDS),
        "concomitant_cyp3a4_inhibitor_count": _count_matches(names, CYP3A4_INHIBITOR_KEYWORDS),
    }
