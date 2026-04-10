import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, AlertCircle, RefreshCw, Activity } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Drag and Drop Handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    validateAndSetFile(droppedFile);
  };

  const handleFileInput = (e) => {
    const selectedFile = e.target.files[0];
    validateAndSetFile(selectedFile);
  };

  const validateAndSetFile = (selectedFile) => {
    setError(null);
    setResults(null);
    if (selectedFile && (selectedFile.name.endsWith('.fasta') || selectedFile.name.endsWith('.fa') || selectedFile.name.endsWith('.txt'))) {
      setFile(selectedFile);
    } else {
      setError("Please upload a valid .fasta or .fa file.");
    }
  };

  // API Call to FastAPI Backend
  const handleUpload = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Connects to the FastAPI server running on port 8000
      const response = await fetch("http://localhost:8000/api/module1/predict-mutations", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to analyze sequence.");
      }

      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const resetTool = () => {
    setFile(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>PathoPredictor</h1>
        <p>Genomic LLM Evolutionary Forecasting Engine</p>
      </header>

      <main className="main-content">
        {!results ? (
          <div className="upload-section">
            <h2 className="section-title">Analyze H5N1 Sequence</h2>
            
            {/* The Drag and Drop Zone */}
            <div 
              className={`drop-zone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileInput} 
                accept=".fasta,.fa,.txt" 
                style={{ display: 'none' }} 
              />
              
              {!file ? (
                <>
                  <UploadCloud className="upload-icon" size={64} />
                  <h3>Select your FASTA file</h3>
                  <p>or drag and drop it here</p>
                </>
              ) : (
                <>
                  <FileText className="file-icon" size={64} />
                  <h3>{file.name}</h3>
                  <p>{(file.size / 1024).toFixed(2)} KB</p>
                </>
              )}
            </div>

            {error && (
              <div className="error-message">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            <button 
              className={`action-btn ${!file || isLoading ? 'disabled' : ''}`}
              onClick={handleUpload}
              disabled={!file || isLoading}
            >
              {isLoading ? (
                <><RefreshCw className="spinner" /> Analyzing Sequence...</>
              ) : (
                <><Activity /> Generate Mutations</>
              )}
            </button>
          </div>
        ) : (
          /* The Results Dashboard */
          <div className="results-section">
            <div className="results-header">
              <h2>Evolutionary Forecast</h2>
              <button className="reset-btn" onClick={resetTool}>Analyze Another Strain</button>
            </div>
            
            <div className="metadata-card">
              <p><strong>Strain ID:</strong> {results.strain_id}</p>
              <p><strong>Targeted Index:</strong> {results.target_index}</p>
              <p><strong>Original Nucleotide:</strong> {results.original_nucleotide}</p>
            </div>

            <h3>Biologically Plausible Variants</h3>
            <div className="predictions-grid">
              {results.predictions.map((pred, index) => (
                <div key={index} className={`prediction-card ${pred.is_original ? 'original' : 'mutation'}`}>
                  <div className="card-top">
                    <span className="rank-badge">#{index + 1}</span>
                    <h4 className="variant-name">{pred.predicted_name}</h4>
                  </div>
                  <div className="loss-score">
                    <span className="label">Perplexity (Loss):</span>
                    <span className="score">{pred.loss_score}</span>
                  </div>
                  <p className="description">
                    {pred.is_original 
                      ? "Wild-type sequence. Highly stable." 
                      : "Mathematically plausible transition. Monitor for structural viability."}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;