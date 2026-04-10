"""
PathoPredictor Main API Gateway
Built with FastAPI to serve predictions to the React frontend.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import json
import os

# Import Module 1 API Functions
from src.module1_genomic_llm.process_uploaded_fasta import clean_uploaded_fasta
from src.module1_genomic_llm.generate_mutations import process_file

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


@app.get("/")
async def root():
    return {"message": "Welcome to the PathoPredictor API. System is online."}


@app.post("/api/module1/predict-mutations")
async def predict_mutations(file: UploadFile = File(...)):
    """
    Endpoint for the React drag-and-drop feature.
    Accepts a .fasta file, cleans it, and returns predicted mutations.
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
            
            # Check if the cleaner threw an error
            if "error" in cleaned_json_string.lower():
                raise HTTPException(status_code=422, detail=f"Data cleaning failed: {cleaned_json_string}")

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

            return final_json

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")