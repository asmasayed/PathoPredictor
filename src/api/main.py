"""
PathoPredictor Main API Gateway
Built with FastAPI to serve predictions to the React frontend.
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
    run_hybrid_seir_with_module2_rates,
)

app = FastAPI(
    title="PathoPredictor API",
    description="Backend API for Genomic LLM and SEIR Simulations",
    version="1.0.0"
)

# CRITICAL for React: This allows your localhost:3000 React app to communicate with your localhost:8000 Python API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your exact React domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the absolute path to your model
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "module1_dnbert"
STATIC_DIR = PROJECT_ROOT / "static"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PhenotypeRequest(BaseModel):
    embedding: List[float]
    sequence: Optional[str] = None
    include_seir_projection: bool = False
    population_N: int = 600000
    projection_days: int = 60
    seir_region: str = "us"


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
    Run Module 3 hybrid SEIR+LSTM (`seir_sim.run_simulation`) seeded with Module 2 β/γ/σ when
    ``models/module3_lstm/lstm_brain_{region}.pth`` exists; otherwise fall back to classical SEIR only.
    """
    if not include:
        return None
    ep = phenotype_bundle.get("epidemiological_parameters") or {}
    beta = ep.get("beta")
    gamma = ep.get("gamma")
    sigma = ep.get("sigma")
    if beta is None or gamma is None or sigma is None:
        return None

    region_key = str(seir_region).lower().strip()

    def _run():
        try:
            return run_hybrid_seir_with_module2_rates(
                float(beta),
                float(gamma),
                float(sigma),
                N=int(population_n),
                days=int(projection_days),
                region=region_key,
            )
        except FileNotFoundError:
            return project_seir_using_module2_parameters(
                float(beta),
                float(gamma),
                float(sigma),
                N=int(population_n),
                days=int(projection_days),
            )

    return await run_in_threadpool(_run)


def _ensure_module2_artifacts_or_raise() -> None:
    clf_path = Path(MODULE2_CLASSIFIER_CONFIG["model_bundle_path"])
    meta_path = Path(MODULE2_REGRESSOR_CONFIG["meta_path"])
    if not clf_path.is_file() or not meta_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Module 2 artifacts missing. Run: python -m src.module2_classifier.train_classifier && "
                "python -m src.module2_regressor.train_regressor"
            ),
        )


@app.get("/")
async def root():
    return {"message": "Welcome to the PathoPredictor API. System is online."}


@app.post("/api/module1/predict-mutations")
async def predict_mutations(
    file: UploadFile = File(...),
    include_module2: bool = Form(False),
    selected_prediction_index: Optional[int] = Form(None),
    include_seir_projection: bool = Form(False),
    seir_population_n: int = Form(600000),
    projection_days: int = Form(60),
    seir_region: str = Form("us"),
):
    """
    Endpoint for the React drag-and-drop feature.
    Accepts a .fasta file, cleans it, and returns predicted mutations.
    Set include_module2=true (multipart form) to also run DNABERT embedding → Module 2.
    Optionally set selected_prediction_index=0..3 to run Module 2 on the chosen mutation candidate.
    Set include_seir_projection=true (with include_module2) to run hybrid Module 3 when LSTM weights exist,
    seeding γ/σ and first-step β from Module 2; otherwise classical SEIR only.
    """
    # 1. Validate the file type
    if not file.filename.endswith(('.fasta', '.fa', '.txt')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a FASTA file.")

    # We use tempfile to safely handle multiple users uploading files at the same time
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fasta_path = Path(temp_dir) / "upload.fasta"
        temp_json_path = Path(temp_dir) / "cleaned.json"

        try:
            # 2. Save the uploaded file to the temporary directory
            with open(temp_fasta_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # 3. Clean the FASTA file
            cleaned_json_string = clean_uploaded_fasta(str(temp_fasta_path))

            # Catch errors generated by the cleaner
            if "error" in cleaned_json_string.lower():
                raise HTTPException(status_code=422, detail=f"Data cleaning failed: {cleaned_json_string}")
            
            cerr = _cleaner_error_message(cleaned_json_string)
            if cerr is not None:
                raise HTTPException(status_code=422, detail=cerr)

            # Save the clean JSON so the mutation generator can read it
            with open(temp_json_path, "w") as f:
                f.write(cleaned_json_string)

            # 4. Generate the Mutations using your RTX 4050
            if not MODEL_DIR.exists():
                raise HTTPException(status_code=503, detail="DNABERT model not found. Please train Module 1 first.")

            mutation_results_string = process_file(str(temp_json_path), str(MODEL_DIR))

            # 5. Parse the string back into a Python dictionary so FastAPI can send it as proper JSON
            final_json = json.loads(mutation_results_string)

            # Check if the generator threw an error
            if "error" in final_json:
                raise HTTPException(status_code=500, detail=final_json["error"])

            if include_seir_projection and not include_module2:
                raise HTTPException(
                    status_code=400,
                    detail="include_seir_projection requires include_module2=true.",
                )

            # 6. Optional: Execute Module 2 and SEIR projections 
            if include_module2:
                _ensure_module2_artifacts_or_raise()
                try:
                    _, sequence = _first_record_from_cleaned_json(cleaned_json_string)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc))

                sequence_for_module2 = sequence
                selected_pred = None
                if selected_prediction_index is not None:
                    try:
                        idx = int(selected_prediction_index)
                    except Exception:
                        raise HTTPException(
                            status_code=400,
                            detail="selected_prediction_index must be an integer in 0..3.",
                        )
                    preds = final_json.get("predictions") or []
                    if not isinstance(preds, list) or len(preds) != 4:
                        raise HTTPException(
                            status_code=500,
                            detail="Module 1 predictions are missing or invalid; expected 4 predictions.",
                        )
                    if idx < 0 or idx >= len(preds):
                        raise HTTPException(
                            status_code=400,
                            detail=f"selected_prediction_index out of range: {idx}. Expected 0..{len(preds)-1}.",
                        )
                    selected_pred = preds[idx]
                    nuc = str(selected_pred.get("nucleotide", "")).upper().strip()
                    if nuc not in {"A", "T", "C", "G"}:
                        raise HTTPException(
                            status_code=500,
                            detail="Selected prediction is missing a valid nucleotide (A/T/C/G).",
                        )
                    try:
                        t_idx = int(final_json.get("target_index"))
                    except Exception:
                        raise HTTPException(
                            status_code=500,
                            detail="Module 1 output is missing a valid target_index.",
                        )
                    if t_idx < 0 or t_idx >= len(sequence):
                        raise HTTPException(
                            status_code=500,
                            detail="Module 1 target_index is out of range for the uploaded sequence.",
                        )
                    sequence_for_module2 = sequence[:t_idx] + nuc + sequence[t_idx + 1 :]

                embedding = await run_in_threadpool(
                    compute_embedding_vector, sequence_for_module2, str(MODEL_DIR)
                )
                phenotype = run_phenotype_assessment(
                    embedding.tolist(),
                    sequence=sequence_for_module2,
                )
                final_json["embedding"] = embedding.tolist()
                final_json["module2_phenotype"] = phenotype
                if selected_pred is not None:
                    final_json["selected_prediction_index"] = int(selected_prediction_index)
                    final_json["selected_prediction"] = selected_pred
                    # Convenience aliases for "alpha/beta/gamma" naming (alpha ≡ sigma here).
                    ep = (phenotype.get("epidemiological_parameters") or {}).copy()
                    ep.setdefault("alpha", ep.get("sigma"))
                    final_json["module2_phenotype"]["epidemiological_parameters"] = ep
                
                traj = await _optional_seir_from_phenotype(
                    phenotype,
                    include_seir_projection,
                    seir_population_n,
                    projection_days,
                    seir_region=seir_region,
                )
                if traj is not None:
                    final_json["seir_projection"] = traj

            return final_json

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


@app.post("/api/module2/phenotype-from-fasta")
async def phenotype_from_fasta(
    file: UploadFile = File(...),
    include_seir_projection: bool = Form(False),
    seir_population_n: int = Form(600000),
    projection_days: int = Form(60),
    seir_region: str = Form("us"),
):
    """
    Clean FASTA → DNABERT embedding (Module 1 checkpoint) → Module 2 phenotype.
    Single upload; no manual embedding paste.

    Multipart toggles:
    include_seir_projection=true — hybrid Module 3 when lstm_brain_{region}.pth exists else classical SEIR.
    """
    if not file.filename.endswith(('.fasta', '.fa', '.txt')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a FASTA file.")
    if not MODEL_DIR.exists():
        raise HTTPException(status_code=503, detail="DNABERT model not found. Expected: models/module1_dnbert")

    _ensure_module2_artifacts_or_raise()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fasta_path = Path(temp_dir) / "upload.fasta"
        with open(temp_fasta_path, "wb") as buffer:
            buffer.write(await file.read())

        cleaned_json_string = clean_uploaded_fasta(str(temp_fasta_path))
        cerr = _cleaner_error_message(cleaned_json_string)
        if cerr is not None:
            raise HTTPException(status_code=422, detail=cerr)

        try:
            strain_id, sequence = _first_record_from_cleaned_json(cleaned_json_string)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        embedding = await run_in_threadpool(compute_embedding_vector, sequence, str(MODEL_DIR))
        phenotype = run_phenotype_assessment(embedding.tolist(), sequence=sequence)

    traj = await _optional_seir_from_phenotype(
        phenotype,
        include_seir_projection,
        seir_population_n,
        projection_days,
        seir_region=seir_region,
    )

    payload = {
        "strain_id": strain_id,
        "sequence_length": len(sequence),
        "embedding": embedding.tolist(),
        "module2_phenotype": phenotype,
    }
    if traj is not None:
        payload["seir_projection"] = traj
    return payload


@app.post("/api/module2/phenotype")
async def phenotype_endpoint(body: PhenotypeRequest):
    """
    Module 2 only: host adaptation score + predicted SEIR-related parameters from an embedding vector.
    Train models first (see SETUP_INSTRUCTIONS.txt Module 2).
    """
    _ensure_module2_artifacts_or_raise()
    try:
        out = run_phenotype_assessment(body.embedding, sequence=body.sequence)
        traj = await _optional_seir_from_phenotype(
            out,
            body.include_seir_projection,
            body.population_N,
            body.projection_days,
            seir_region=body.seir_region,
        )
        if traj is not None:
            out = {**out, "seir_projection": traj}
        return out
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))