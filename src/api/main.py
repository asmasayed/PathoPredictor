"""
PathoPredictor Main API Gateway
Integrated with Module 1 (Genomics), Module 2 (Phenotypes), and Module 3 (AI-SEIR)
"""
import os
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# --- Module 1 Imports ---
from src.module1_genomic_llm.process_uploaded_fasta import clean_uploaded_fasta
from src.module1_genomic_llm.generate_mutations import process_file

# --- Module 2 & 3 Imports ---
from src.config.config import MODULE2_CLASSIFIER_CONFIG, MODULE2_REGRESSOR_CONFIG
from src.module2_data.dnabert_embedding import compute_embedding_vector
from src.module2_assessment import run_phenotype_assessment
from src.integration.seir_projection import (
    project_seir_using_module2_parameters,
)

# --- Your Module 3 Engine ---
from src.simulation.dynamic_seir import build_module3_seir_payload

app = FastAPI(
    title="PathoPredictor API",
    description="Backend API for Genomic LLM and SEIR Simulations",
    version="1.0.0"
)

# CORS configuration to allow the React frontend (localhost:3000) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path and Static File Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "module1_dnbert"
STATIC_DIR = PROJECT_ROOT / "static"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class PhenotypeRequest(BaseModel):
    embedding: List[float]
    sequence: Optional[str] = None
    include_seir_projection: bool = False
    population_N: int = 10000
    projection_days: int = 100
    seir_region: str = "us"


class Module3SeirRequest(BaseModel):
    """Numeric SEIR-LSTM dashboard request (no FASTA re-upload)."""

    beta: float
    alpha: float
    gamma: float
    risk_score_percent: Optional[float] = None
    population_n: int = 10000

# --- Internal Helper Functions (Logic preserved from original) ---

def _first_record_from_cleaned_json(cleaned_json_string: str) -> Tuple[str, str]:
    data = json.loads(cleaned_json_string)
    if isinstance(data, dict) and "error" in data:
        raise ValueError(data.get("error", "Cleaning failed"))
    if isinstance(data, list):
        if not data:
            raise ValueError("Cleaned JSON has no records.")
        record = data[0]
    else:
        record = data
    sequence = record.get("sequence")
    strain_id = record.get("strain_id", "unknown")
    if not sequence:
        raise ValueError("Cleaned record has no sequence.")
    return str(strain_id), str(sequence)

def _cleaner_error_message(cleaned_json_string: str) -> Optional[str]:
    try:
        data = json.loads(cleaned_json_string)
        if isinstance(data, dict) and "error" in data:
            return str(data["error"])
    except json.JSONDecodeError:
        return "Cleaning produced invalid JSON."
    return None

async def _optional_seir_from_phenotype(
    phenotype_bundle: dict,
    include: bool,
    population_n: int,
    projection_days: int,
    seir_region: str = "us",
):
    """
    INTEGRATED LOGIC:
    This function now uses YOUR run_dynamic_seir engine.
    It takes the biological outputs from Module 2 and feeds them into your AI simulation.
    """
    if not include:
        return None
    
    ep = phenotype_bundle.get("epidemiological_parameters") or {}
    beta = ep.get("beta")
    gamma = ep.get("gamma")
    sigma = ep.get("sigma") # sigma maps to Alpha (incubation rate) in SEIR
    
    if beta is None or gamma is None or sigma is None:
        return None

    risk = (phenotype_bundle.get("human_adaptation") or {}).get("risk_score_percent")

    def _run():
        try:
            return build_module3_seir_payload(
                N=int(population_n),
                base_beta=float(beta),
                alpha=float(sigma),
                gamma=float(gamma),
                risk_score_percent=float(risk) if risk is not None else None,
            )
        except Exception as e:
            print(f"Module 3 AI failed, using static fallback: {e}")
            return project_seir_using_module2_parameters(
                float(beta), float(gamma), float(sigma),
                N=int(population_n), days=int(projection_days)
            )

    return await run_in_threadpool(_run)

def _ensure_module2_artifacts_or_raise() -> None:
    clf_path = Path(MODULE2_CLASSIFIER_CONFIG["model_bundle_path"])
    meta_path = Path(MODULE2_REGRESSOR_CONFIG["meta_path"])
    if not clf_path.is_file() or not meta_path.is_file():
        raise HTTPException(status_code=503, detail="Module 2 artifacts missing.")

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "PathoPredictor API System Online - Module 3 Integrated"}

@app.get("/api/module3/simulate")
async def test_module3():
    """Standalone endpoint to verify Module 3 returns the full dashboard payload."""
    try:
        return await run_in_threadpool(
            build_module3_seir_payload,
            N=10000,
            base_beta=0.2993,
            alpha=0.2008,
            gamma=0.1003,
            risk_score_percent=13.5,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/module3/seir-dashboard")
async def module3_seir_dashboard(body: Module3SeirRequest):
    """Run LSTM-adjusted SEIR and return time series + milestones (JSON for the React dashboard)."""
    try:
        return await run_in_threadpool(
            build_module3_seir_payload,
            N=int(body.population_n),
            base_beta=float(body.beta),
            alpha=float(body.alpha),
            gamma=float(body.gamma),
            risk_score_percent=float(body.risk_score_percent)
            if body.risk_score_percent is not None
            else None,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/module3/variant-dashboard")
async def variant_dashboard(
    file: UploadFile = File(...),
    selected_prediction_index: int = Form(...),
    seir_population_n: int = Form(10000),
):
    """
    FASTA + variant index: Module 1 mutations, Module 2 (WT rates + variant host risk),
    and Module 3 dashboard in one response.
    """
    if not file.filename.endswith((".fasta", ".fa", ".txt")):
        raise HTTPException(status_code=400, detail="Invalid FASTA format.")

    _ensure_module2_artifacts_or_raise()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fasta_path = Path(temp_dir) / "upload.fasta"
        temp_json_path = Path(temp_dir) / "cleaned.json"

        with open(temp_fasta_path, "wb") as buffer:
            buffer.write(await file.read())

        cleaned_json_string = clean_uploaded_fasta(str(temp_fasta_path))
        cerr = _cleaner_error_message(cleaned_json_string)
        if cerr:
            raise HTTPException(status_code=422, detail=cerr)

        with open(temp_json_path, "w") as f:
            f.write(cleaned_json_string)

        mutation_results_string = process_file(str(temp_json_path), str(MODEL_DIR))
        final_json = json.loads(mutation_results_string)
        if isinstance(final_json, dict) and "error" in final_json:
            raise HTTPException(status_code=422, detail=str(final_json["error"]))

        preds = final_json.get("predictions") or []
        idx = int(selected_prediction_index)
        if idx < 0 or idx >= len(preds):
            raise HTTPException(status_code=400, detail="Invalid selected_prediction_index.")

        _, base_sequence = _first_record_from_cleaned_json(cleaned_json_string)
        t_idx = int(final_json.get("target_index"))
        nuc = str(preds[idx].get("nucleotide", "")).upper()
        mutant_sequence = base_sequence[:t_idx] + nuc + base_sequence[t_idx + 1 :]

        embedding_mutant = await run_in_threadpool(
            compute_embedding_vector, mutant_sequence, str(MODEL_DIR)
        )
        embedding_wt = await run_in_threadpool(
            compute_embedding_vector, base_sequence, str(MODEL_DIR)
        )

        phenotype = run_phenotype_assessment(
            embedding_mutant.tolist(),
            sequence=mutant_sequence,
            regressor_embedding=embedding_wt.tolist(),
            regressor_sequence=base_sequence,
        )

        ep = phenotype.get("epidemiological_parameters") or {}
        beta = ep.get("beta")
        gamma = ep.get("gamma")
        sigma = ep.get("sigma")
        host = phenotype.get("human_adaptation") or {}
        risk = host.get("risk_score_percent")

        if beta is None or gamma is None or sigma is None:
            raise HTTPException(status_code=500, detail="Module 2 did not return beta/gamma/sigma.")

        try:
            module3 = await run_in_threadpool(
                build_module3_seir_payload,
                N=int(seir_population_n),
                base_beta=float(beta),
                alpha=float(sigma),
                gamma=float(gamma),
                risk_score_percent=float(risk) if risk is not None else None,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        final_json["module2_phenotype"] = phenotype
        final_json["module3_dashboard"] = module3
        final_json["selected_prediction"] = preds[idx]

    return final_json

@app.post("/api/module1/predict-mutations")
async def predict_mutations(
    file: UploadFile = File(...),
    include_module2: bool = Form(False),
    selected_prediction_index: Optional[int] = Form(None),
    include_seir_projection: bool = Form(False),
    seir_population_n: int = Form(10000),
    projection_days: int = Form(100),
    seir_region: str = Form("us"),
):
    if not file.filename.endswith(('.fasta', '.fa', '.txt')):
        raise HTTPException(status_code=400, detail="Invalid FASTA format.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fasta_path = Path(temp_dir) / "upload.fasta"
        temp_json_path = Path(temp_dir) / "cleaned.json"

        with open(temp_fasta_path, "wb") as buffer:
            buffer.write(await file.read())

        cleaned_json_string = clean_uploaded_fasta(str(temp_fasta_path))
        
        cerr = _cleaner_error_message(cleaned_json_string)
        if cerr: raise HTTPException(status_code=422, detail=cerr)

        with open(temp_json_path, "w") as f:
            f.write(cleaned_json_string)

        mutation_results_string = process_file(str(temp_json_path), str(MODEL_DIR))
        final_json = json.loads(mutation_results_string)

        if include_module2:
            _ensure_module2_artifacts_or_raise()
            _, base_sequence = _first_record_from_cleaned_json(cleaned_json_string)

            preds = final_json.get("predictions") or []
            if selected_prediction_index is not None:
                idx = int(selected_prediction_index)
                if idx < 0 or idx >= len(preds):
                    raise HTTPException(status_code=400, detail="Invalid selected_prediction_index.")
                nuc = str(preds[idx].get("nucleotide", "")).upper()
                t_idx = int(final_json.get("target_index"))
                mutant_sequence = base_sequence[:t_idx] + nuc + base_sequence[t_idx + 1 :]
                final_json["selected_prediction"] = preds[idx]
            else:
                mutant_sequence = base_sequence

            embedding_mutant = await run_in_threadpool(
                compute_embedding_vector, mutant_sequence, str(MODEL_DIR)
            )
            embedding_wt = await run_in_threadpool(
                compute_embedding_vector, base_sequence, str(MODEL_DIR)
            )

            phenotype = run_phenotype_assessment(
                embedding_mutant.tolist(),
                sequence=mutant_sequence,
                regressor_embedding=embedding_wt.tolist(),
                regressor_sequence=base_sequence,
            )
            final_json["module2_phenotype"] = phenotype

            traj = await _optional_seir_from_phenotype(
                phenotype, include_seir_projection, seir_population_n, projection_days, seir_region
            )
            if traj:
                final_json["seir_projection"] = traj

        return final_json

@app.post("/api/module2/phenotype-from-fasta")
async def phenotype_from_fasta(
    file: UploadFile = File(...),
    include_seir_projection: bool = Form(False),
    seir_population_n: int = Form(10000),
    projection_days: int = Form(100),
    seir_region: str = Form("us"),
):
    _ensure_module2_artifacts_or_raise()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "up.fasta"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        cleaned = clean_uploaded_fasta(str(temp_path))
        strain_id, sequence = _first_record_from_cleaned_json(cleaned)
        embedding = await run_in_threadpool(compute_embedding_vector, sequence, str(MODEL_DIR))
        phenotype = run_phenotype_assessment(embedding.tolist(), sequence=sequence)

    traj = await _optional_seir_from_phenotype(
        phenotype, include_seir_projection, seir_population_n, projection_days, seir_region
    )

    return {
        "strain_id": strain_id,
        "module2_phenotype": phenotype,
        "seir_projection": traj
    }

@app.post("/api/module2/phenotype")
async def phenotype_endpoint(body: PhenotypeRequest):
    _ensure_module2_artifacts_or_raise()
    out = run_phenotype_assessment(body.embedding, sequence=body.sequence)
    traj = await _optional_seir_from_phenotype(
        out, body.include_seir_projection, body.population_N, body.projection_days, body.seir_region
    )
    if traj:
        out["seir_projection"] = traj
    return out