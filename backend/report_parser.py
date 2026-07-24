"""
report_parser.py — Extract structured lab values from an uploaded PDF lab
report and propose them as PatientProfile field values.

Uses pdfplumber's layout-preserving text extraction (`extract_text(layout=True)`),
not `extract_tables()` or plain text extraction. Both were tested against real
multi-column lab reports during development and silently misassigned values to
the wrong test name: pdfplumber's ruling-line table detector misses borderless
data rows entirely (only the header row came back), and plain text extraction
(e.g. `pdftotext -layout`) reflows multi-column pages and scrambles row order
(a value could land next to the wrong test name). `layout=True` preserves the
original character grid instead, so each row's test name/value/unit stay on
one correctly-ordered line — this was verified against a real 24-page panel
report before relying on it here.

This module never runs clinical logic (evaluate_drug, etc.) and never decides
anything by itself — it only proposes values for a human to review/edit in the
form before an assessment is run. Treat its output as untrusted user input.
"""
import io
import re

import pdfplumber

# Maps a PatientProfile field to the substrings (normalized: lowercased,
# whitespace-collapsed) that identify it in a report's test-name column.
FIELD_SYNONYMS = {
    "egfr": [
        "egfr",
        "est. glomerular filtration rate",
        "glomerular filtration rate",
    ],
    "alt": [
        "alanine aminotransferase",
        "sgpt",
    ],
    "ast": [
        "aspartate aminotransferase",
        "sgot",
    ],
    "hba1c": [
        "hba1c",
        "glycosylated haemoglobin",
        "glycosylated hemoglobin",
    ],
    "ldl": [
        "ldl cholesterol",
        "ldl - direct",
        "ldl direct",
    ],
    "inr": [
        "inr",
        "international normalized ratio",
    ],
    "systolic_bp": [
        "systolic blood pressure",
        "systolic bp",
    ],
    "diastolic_bp": [
        "diastolic blood pressure",
        "diastolic bp",
    ],
    "weight_kg": [
        "body weight",
        "weight (kg)",
    ],
}

# Cholesterol-family fields the app stores in mmol/L, but Indian/US lab reports
# almost always report them in mg/dL (e.g. "94 mg/dL"). Trusting the raw number
# unconverted would be off by ~38.7x — 94 mg/dL LDL is not 94 mmol/L LDL, and
# feeding the unconverted number into evaluate_drug()'s `ldl > 3.0` check would
# flag nearly every patient as having dangerously high cholesterol.
MGDL_TO_MMOL_CHOLESTEROL = 38.67
MGDL_TO_MMOL_FIELDS = {"ldl"}

# A results row looks like "TEST NAME   <value>   <unit>   <ref range>" once
# layout=True has preserved column alignment — name and value are separated
# by a run of 2+ spaces (single spaces appear inside multi-word names).
_VALUE_LINE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9 ./()\-,]+?)\s{2,}"
    r"(?P<value>-?\d+\.?\d*)\s+"
    r"(?P<unit>[A-Za-z%µ/.\d]+)"
)

_SKIP_NAMES = {"page", "test name", "test description", "note", "method"}

_AGE_SEX_PAREN_RE = re.compile(r"\((\d{1,3})\s*Y\s*/\s*([MF])\)", re.IGNORECASE)
_AGE_SEX_LABEL_RE = re.compile(
    r"AGE\s*/\s*GENDER\s*:?\s*(\d{1,3})\s*Y\s*/\s*(Male|Female|M|F)", re.IGNORECASE
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _match_field(test_name: str):
    normalized = _normalize(test_name)
    for field, synonyms in FIELD_SYNONYMS.items():
        for syn in synonyms:
            # Word-boundary match, not plain substring: "ldl cholesterol" must
            # not match inside "vldl cholesterol" — verified against a real
            # report where VLDL Cholesterol otherwise got misfiled as LDL.
            if re.search(r"\b" + re.escape(syn) + r"\b", normalized):
                return field
    return None


def _extract_demographics(text: str) -> dict:
    demo = {}
    m = _AGE_SEX_PAREN_RE.search(text) or _AGE_SEX_LABEL_RE.search(text)
    if m:
        demo["age"] = int(m.group(1))
        demo["sex"] = "M" if m.group(2).upper().startswith("M") else "F"
    return demo


def parse_lab_report(file_bytes: bytes) -> dict:
    """
    Parse an uploaded lab report PDF into PatientProfile field candidates.

    Returns:
        {
            "patient_fields": {field_name: value, ...},
            "matched": [{"field", "test_name", "raw_value", "unit",
                         "converted_to_mmol_l"}, ...],
            "unmatched_tests": ["TEST NAME", ...],
        }
    """
    patient_fields: dict = {}
    matched = []
    unmatched = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or ""

            if "age" not in patient_fields:
                patient_fields.update(_extract_demographics(text))

            for line in text.splitlines():
                m = _VALUE_LINE_RE.match(line)
                if not m:
                    continue

                # Collapse internal whitespace: backtracking can pull a
                # "TECHNOLOGY" column into the name when the true value is
                # one column further right (regex looks for the first 2+
                # space gap, which is sometimes name/technology, not
                # name/value) — this doesn't affect which value gets used,
                # only how the matched test name reads in the review UI.
                raw_name = _normalize(m.group("name")).upper()
                if raw_name.lower() in _SKIP_NAMES:
                    continue

                raw_value = m.group("value")
                unit = m.group("unit").strip()

                # Derived ratio rows (e.g. "SGOT / SGPT RATIO") can contain a
                # tracked field's synonym as a whole word while reporting an
                # unrelated computed ratio, not the raw lab value — verified
                # against a real report where this otherwise overwrote ALT
                # with an SGOT/SGPT ratio in the transparency log.
                if unit.lower() == "ratio":
                    unmatched.append(raw_name)
                    continue

                field = _match_field(raw_name)
                if field is None:
                    unmatched.append(raw_name)
                    continue

                try:
                    value = float(raw_value)
                except ValueError:
                    continue

                converted = False
                if field in MGDL_TO_MMOL_FIELDS and "mg" in unit.lower():
                    value = round(value / MGDL_TO_MMOL_CHOLESTEROL, 2)
                    converted = True

                # First match wins per field: these reports put an "abnormal
                # results" summary before the full per-panel detail pages, and
                # both report the same lab value, so later pages would only
                # ever re-confirm (never override) what the summary already gave.
                if field not in patient_fields:
                    patient_fields[field] = value

                matched.append({
                    "field": field,
                    "test_name": raw_name,
                    "raw_value": raw_value,
                    "unit": unit,
                    "converted_to_mmol_l": converted,
                })

    return {
        "patient_fields": patient_fields,
        "matched": matched,
        "unmatched_tests": sorted(set(unmatched)),
    }
