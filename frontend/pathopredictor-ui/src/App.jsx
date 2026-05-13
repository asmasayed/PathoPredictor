import React, { useState, useRef } from 'react';
import { AlertCircle, RefreshCw, Activity, UploadCloud, FileText } from 'lucide-react';
import SimulationDashboard from './SimulationDashboard.jsx';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  const [step, setStep] = useState('upload'); // 'upload', 'variants', 'dashboard'
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const [variants, setVariants] = useState([]);
  const [selectedPredictionIndex, setSelectedPredictionIndex] = useState(null);
  
  const [module3Dashboard, setModule3Dashboard] = useState(null);
  const [dashboardPhenotype, setDashboardPhenotype] = useState(null);
  const [dashboardVariantLabel, setDashboardVariantLabel] = useState('');
  
  const [isUploading, setIsUploading] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulatingIndex, setSimulatingIndex] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const submitFile = async () => {
    if (!file) {
      setError('Please select a FASTA file first.');
      return;
    }
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/module1/predict-mutations`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch variants.');
      }

      setVariants(data.predictions || []);
      setStep('variants');
      setSelectedPredictionIndex(null);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setIsUploading(false);
    }
  };

  const runSimulation = async (index) => {
    const targetIndex = index !== undefined ? index : selectedPredictionIndex;
    if (targetIndex === null || !file) return;

    setIsSimulating(true);
    setSimulatingIndex(targetIndex);
    setSelectedPredictionIndex(targetIndex);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('selected_prediction_index', targetIndex);

    try {
      const response = await fetch(`${API_BASE}/api/module3/variant-dashboard`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to run simulation.');
      }

      setModule3Dashboard(data.module3_dashboard);
      setDashboardPhenotype(data.module2_phenotype);
      setDashboardVariantLabel(data.selected_prediction.predicted_name);
      setStep('dashboard');
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setIsSimulating(false);
      setSimulatingIndex(null);
    }
  };

  const backToVariants = () => {
    setStep('variants');
  };

  const resetUpload = () => {
    setStep('upload');
    setFile(null);
    setVariants([]);
    setError(null);
  };

  if (step === 'dashboard' && module3Dashboard && dashboardPhenotype) {
    return (
      <div className="app-container">
        <header className="header">
          <h1>PathoPredictor</h1>
          <p>Module 3 — AI-SEIR / LSTM</p>
        </header>
        <main className="main-content main-content--wide">
          <SimulationDashboard
            module3={module3Dashboard}
            module2Phenotype={dashboardPhenotype}
            variantLabel={dashboardVariantLabel}
            onBack={backToVariants}
          />
        </main>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>PathoPredictor</h1>
        <p>Genomic AI & Epidemiological Forecasting</p>
      </header>

      <main className="main-content">
        {step === 'upload' && (
          <div className="upload-section">
            <h2 className="section-title">Upload Genomic Data</h2>
            <div
              className={`drop-zone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".fasta,.fa,.txt"
                style={{ display: 'none' }}
              />
              {file ? (
                <>
                  <FileText className="file-icon" size={48} />
                  <h3>{file.name}</h3>
                  <p>{(file.size / 1024).toFixed(2)} KB</p>
                </>
              ) : (
                <>
                  <UploadCloud className="upload-icon" size={48} />
                  <h3>Drag & Drop FASTA file here</h3>
                  <p>or click to browse</p>
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
              type="button"
              className={`action-btn ${!file || isUploading ? 'disabled' : ''}`}
              onClick={submitFile}
              disabled={!file || isUploading}
            >
              {isUploading ? (
                <>
                  <RefreshCw className="spinner" /> Analyzing Sequence...
                </>
              ) : (
                'Submit for Analysis'
              )}
            </button>
          </div>
        )}

        {step === 'variants' && (
          <div className="results-section">
            <div className="results-header">
              <h2>Module 1: Predicted Variants</h2>
              <button className="reset-btn" onClick={resetUpload}>
                Start Over
              </button>
            </div>

            <div className="module2-controls">
              <p className="hint">
                Select a predicted variant below and run the SEIR simulation (Module 2 & 3).
              </p>
              <button
                type="button"
                className={`action-btn ${selectedPredictionIndex === null || isSimulating ? 'disabled' : ''}`}
                onClick={() => runSimulation(selectedPredictionIndex)}
                disabled={selectedPredictionIndex === null || isSimulating}
              >
                {isSimulating ? (
                  <>
                    <RefreshCw className="spinner" /> Running simulation…
                  </>
                ) : (
                  <>
                    <Activity /> Run SEIR Simulation
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="error-message" style={{ marginBottom: '1.5rem' }}>
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            <div className="predictions-grid">
              {variants.map((pred, idx) => (
                <div
                  key={idx}
                  className={`prediction-card ${pred.is_original ? 'original' : 'mutation'} ${
                    selectedPredictionIndex === idx ? 'selected' : ''
                  } ${simulatingIndex === idx ? 'loading-pulse' : ''}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => !isSimulating && setSelectedPredictionIndex(idx)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      if (!isSimulating) setSelectedPredictionIndex(idx);
                    }
                  }}
                >
                  <div className="card-top">
                    <span className="rank-badge">#{idx + 1}</span>
                    <h4 className="variant-name">{pred.predicted_name}</h4>
                  </div>
                  <div className="loss-score">
                    <span className="label">Perplexity (Loss):</span>
                    <span className="score">{pred.loss_score?.toFixed(4) || 'N/A'}</span>
                  </div>
                  <p className="description">
                    {pred.is_original
                      ? 'Wild-type sequence. Highly stable.'
                      : 'Mathematically plausible transition. Monitor for structural viability.'}
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
