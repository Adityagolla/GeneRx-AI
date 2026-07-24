"""
server.py — FastAPI backend for GeneRx-AI
Combines ML predictions with clinical rules engine.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from clinical_engine import (
    DRUG_CATALOG,
    evaluate_drug,
    check_drug_interactions,
    generate_patient_summary,
    simulate_response_over_time,
    get_alternatives,
)
from ml_model import get_predictor
from report_parser import parse_lab_report
from pharmacovigilance import get_signal

app = FastAPI(
    title="GeneRx-AI API",
    description="Clinical drug safety assessment powered by ML and evidence-based rules",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────────

class PatientProfile(BaseModel):
    name: str = "Patient"
    age: int = 50
    sex: str = "F"
    weight_kg: float = 70.0
    conditions: List[str] = []
    egfr: float = 90.0
    alt: float = 25.0
    ast: float = 25.0
    hba1c: float = 5.5
    systolic_bp: int = 120
    diastolic_bp: int = 80
    ldl: float = 3.0
    inr: float = 1.0
    current_meds: List[str] = []
    allergies: List[str] = []


class AssessmentRequest(BaseModel):
    patient: PatientProfile
    drugs: List[str]


class InteractionRequest(BaseModel):
    drugs: List[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    predictor = get_predictor()
    return {
        "status": "ok",
        "ml_model_loaded": predictor.loaded,
        "ml_model_name": predictor.active_model_name,
        "ml_model_accuracy": predictor.accuracy,
        "ml_model_macro_f1": predictor.macro_f1,
        "ml_model_candidates": predictor.candidates,
        "drugs_available": len(DRUG_CATALOG),
    }


@app.get("/api/drugs")
def get_drugs():
    """Return drug catalog with basic info."""
    drugs = []
    for name, info in DRUG_CATALOG.items():
        drugs.append({
            "name": name,
            "category": info.get("category", ""),
            "description": info.get("description", ""),
            "brand_names": info.get("brand_names", []),
            "indications": info.get("indications", []),
            "faers_signal": get_signal(name),
        })
    return {"drugs": drugs}


def _run_assessment(patient: PatientProfile, drugs: List[str]):
    """
    Shared core of /api/assess and /api/report: builds the rules-engine
    patient dict, validates + evaluates each requested drug, and checks
    interactions. Raises HTTPException(400) for any drug not in the catalog —
    evaluate_drug() indexes DRUG_CATALOG[drug_name] unconditionally, so an
    unrecognized name would otherwise 500 instead of failing cleanly.
    """
    unknown = [d for d in drugs if d not in DRUG_CATALOG]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown drug(s): {', '.join(unknown)}. "
                   f"Available drugs: {', '.join(DRUG_CATALOG)}",
        )

    predictor = get_predictor()

    patient_dict = {
        "name": patient.name,
        "patient_id": "API",
        "age": patient.age,
        "sex": patient.sex,
        "weight_kg": patient.weight_kg,
        "bmi": patient.weight_kg / (1.70 ** 2),
        "conditions": patient.conditions,
        "egfr": patient.egfr,
        "alt": patient.alt,
        "ast": patient.ast,
        "hba1c": patient.hba1c,
        "systolic_bp": patient.systolic_bp,
        "diastolic_bp": patient.diastolic_bp,
        "ldl": patient.ldl,
        "inr": patient.inr,
        "current_meds": patient.current_meds,
        "allergies": patient.allergies,
    }

    results = []
    for drug_name in drugs:
        rule_result = evaluate_drug(patient_dict, drug_name)

        ml_result = predictor.predict(
            drug_name=drug_name,
            patient_age=patient.age,
            patient_sex=patient.sex,
            patient_weight=patient.weight_kg,
            num_concomitant_drugs=len(patient.current_meds),
            current_meds=patient.current_meds,
        )

        combined = {
            "drug_name": drug_name,
            "suitability": rule_result["suitability"],
            "risk_level": rule_result["risk_level"],
            "reasons": rule_result["reasons"],
            "warnings": rule_result["warnings"],
            "dose_notes": rule_result["dose_notes"],
            "monitoring": rule_result["monitoring"],
            "side_effects": rule_result["side_effects"],
            "ml_prediction": ml_result,
            "faers_signal": get_signal(drug_name),
            "response_curve": simulate_response_over_time(rule_result["suitability"]),
            "alternatives": get_alternatives(patient_dict, drug_name, rule_result["suitability"]),
        }
        results.append(combined)

    all_drugs = drugs + patient.current_meds
    interactions = check_drug_interactions(all_drugs)

    return patient_dict, results, interactions


@app.post("/api/assess")
def assess_drugs(request: AssessmentRequest):
    """Assess drugs for a patient — combines ML model + clinical rules."""
    _patient_dict, results, interactions = _run_assessment(request.patient, request.drugs)

    return {
        "patient_name": request.patient.name,
        "assessments": results,
        "interactions": interactions,
    }


@app.post("/api/report")
def get_report(request: AssessmentRequest):
    """Generate a downloadable plain-text clinical report for an assessment."""
    patient_dict, results, interactions = _run_assessment(request.patient, request.drugs)
    report_text = generate_patient_summary(patient_dict, results, interactions)
    return {"report": report_text}


@app.post("/api/interactions")
def check_interactions(request: InteractionRequest):
    """Check drug-drug interactions."""
    interactions = check_drug_interactions(request.drugs)
    return {"interactions": interactions}


@app.post("/api/parse-report")
async def parse_report(file: UploadFile = File(...)):
    """
    Parse an uploaded lab report PDF into PatientProfile field candidates.

    This only proposes values — the frontend must pre-fill them into the
    editable form for the user to review/correct, never submit them straight
    into an assessment. See report_parser.py for why (extraction can fail
    silently on unusual report layouts, and a wrong eGFR/LDL value changes a
    suitability verdict).
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()
    try:
        result = parse_lab_report(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read this PDF: {e}")

    if not result["patient_fields"]:
        result["warning"] = (
            "No recognized lab values found in this report. "
            "You can still enter values manually below."
        )

    return result


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve the static files from the 'frontend' directory for CSS/JS
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    """Serve the main frontend HTML file."""
    return FileResponse("frontend/index.html")

@app.get("/{file_name}")
async def read_file(file_name: str):
    """Serve specific root files (like style.css or app.js) directly."""
    import os
    file_path = f"frontend/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces standard port is 7860
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

