import React, { useState, useEffect, useCallback } from 'react';
import { AlertCircle, RefreshCw, Activity, FlaskConical } from 'lucide-react';
import SimulationDashboard from './SimulationDashboard.jsx';
import {
  DEMO_EP,
  DEMO_STRAIN_META,
  DEMO_VARIANTS,
  buildPhenotypeFromDemoVariant,
} from './module3Demo.js';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  const [showDashboard, setShowDashboard] = useState(false);
  const [module3Dashboard, setModule3Dashboard] = useState(null);
  const [dashboardPhenotype, setDashboardPhenotype] = useState(null);
  const [dashboardVariantLabel, setDashboardVariantLabel] = useState('');
  const [selectedPredictionIndex, setSelectedPredictionIndex] = useState(null);
  const [isSeirLoading, setIsSeirLoading] = useState(false);
  const [seirLoadingIndex, setSeirLoadingIndex] = useState(null);
  const [error, setError] = useState(null);

  const runSeirForDemoVariant = useCallback(async (index) => {
    const v = DEMO_VARIANTS[index];
    setIsSeirLoading(true);
    setSeirLoadingIndex(index);
    setSelectedPredictionIndex(index);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/module3/seir-dashboard`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          beta: DEMO_EP.beta,
          alpha: DEMO_EP.alpha,
          gamma: DEMO_EP.gamma,
          risk_score_percent: v.risk_score_percent,
          population_n: 10000,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        const detail =
          typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data);
        throw new Error(detail || 'SEIR dashboard request failed.');
      }

      setModule3Dashboard(data);
      setDashboardPhenotype(buildPhenotypeFromDemoVariant(v));
      setDashboardVariantLabel(v.predicted_name);
      setShowDashboard(true);
    } catch (err) {
      setError(err.message || String(err));
      setShowDashboard(false);
      setModule3Dashboard(null);
    } finally {
      setIsSeirLoading(false);
      setSeirLoadingIndex(null);
    }
  }, []);

  useEffect(() => {
    if (globalThis.__pathoM3DemoAutoOnce) return;
    globalThis.__pathoM3DemoAutoOnce = true;
    runSeirForDemoVariant(0);
  }, [runSeirForDemoVariant]);

  const runSeirSimulationForSelection = () => {
    if (selectedPredictionIndex === null) return;
    runSeirForDemoVariant(selectedPredictionIndex);
  };

  const backToVariants = () => {
    setShowDashboard(false);
  };

  const demoBannerText =
    'Demo mode: α, β, and γ are fixed for all variants (values from your Module 2 reference). ' +
    'Only host adaptation (risk %) changes the initial exposed seed; curves are produced by the Module 3 LSTM-SEIR API.';

  if (showDashboard && module3Dashboard && dashboardPhenotype) {
    return (
      <div className="app-container">
        <header className="header">
          <h1>PathoPredictor</h1>
          <p>Module 3 — AI-SEIR / LSTM (demo entry)</p>
        </header>
        <main className="main-content main-content--wide">
          <SimulationDashboard
            module3={module3Dashboard}
            module2Phenotype={dashboardPhenotype}
            variantLabel={dashboardVariantLabel}
            onBack={backToVariants}
            infoBanner={demoBannerText}
          />
        </main>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>PathoPredictor</h1>
        <p>Module 3 — AI-SEIR / LSTM (demo entry)</p>
      </header>

      <main className="main-content">
        <div className="results-section module3-demo-home">
          <div className="demo-hero">
            <FlaskConical className="demo-hero-icon" size={36} aria-hidden />
            <div>
              <h2 className="section-title" style={{ marginBottom: '0.35rem' }}>
                SEIR–LSTM policy lab
              </h2>
              <p className="hint" style={{ textAlign: 'left', maxWidth: '42rem' }}>
                This screen skips Module 1 (FASTA / mutations) and Module 2 (trained classifier + regressor
                artifacts). You still get the full Module 3 dashboard: dynamic β from the mobility-trained LSTM,
                compartment trajectories, milestones, and β dynamics. On first load, WildType runs automatically if
                the API on port 8000 is up.
              </p>
            </div>
          </div>

          {isSeirLoading && !showDashboard && (
            <div className="seir-loading-banner">
              <RefreshCw className="spinner" size={22} />
              <span>Loading Module 3 simulation…</span>
            </div>
          )}

          <div className="metadata-card">
            <p>
              <strong>Strain ID:</strong> {DEMO_STRAIN_META.strain_id}
            </p>
            <p>
              <strong>Targeted index:</strong> {DEMO_STRAIN_META.target_index}
            </p>
            <p>
              <strong>Original nucleotide:</strong> {DEMO_STRAIN_META.original_nucleotide}
            </p>
          </div>

          <div className="module2-results">
            <h3>Fixed rates (all variants)</h3>
            <div className="module2-grid">
              <div className="module2-card">
                <h4>α / β / γ</h4>
                <p>
                  <strong>alpha (σ):</strong> {DEMO_EP.alpha}
                </p>
                <p>
                  <strong>beta:</strong> {DEMO_EP.beta}
                </p>
                <p>
                  <strong>gamma:</strong> {DEMO_EP.gamma}
                </p>
              </div>
              <div className="module2-card">
                <h4>Why the full chain failed before</h4>
                <p className="description" style={{ textAlign: 'left' }}>
                  Module 1 needs the DNABERT model directory and a valid FASTA run. Module 2 needs on-disk classifier
                  and regressor checkpoints; without them the API returns 503 and no phenotype JSON. This demo calls
                  only <code className="inline-code">POST /api/module3/seir-dashboard</code> so Module 3 runs with
                  the constants above plus the host-risk preset for each variant.
                </p>
              </div>
            </div>
          </div>

          <div className="module2-controls">
            <p className="hint">
              Click a variant to open the dashboard, or select one and press <strong>SEIR simulation</strong>.
            </p>
            <button
              type="button"
              className={`action-btn ${selectedPredictionIndex === null || isSeirLoading ? 'disabled' : ''}`}
              onClick={runSeirSimulationForSelection}
              disabled={selectedPredictionIndex === null || isSeirLoading}
            >
              {isSeirLoading ? (
                <>
                  <RefreshCw className="spinner" /> Running simulation…
                </>
              ) : (
                <>
                  <Activity /> SEIR simulation
                </>
              )}
            </button>
          </div>

          <h3>Biologically plausible variants (demo presets)</h3>
          <div className="predictions-grid">
            {DEMO_VARIANTS.map((pred) => (
              <div
                key={pred.index}
                className={`prediction-card ${pred.is_original ? 'original' : 'mutation'} ${
                  selectedPredictionIndex === pred.index ? 'selected' : ''
                } ${seirLoadingIndex === pred.index ? 'loading-pulse' : ''}`}
                role="button"
                tabIndex={0}
                onClick={() => runSeirForDemoVariant(pred.index)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    runSeirForDemoVariant(pred.index);
                  }
                }}
              >
                <div className="card-top">
                  <span className="rank-badge">#{pred.index + 1}</span>
                  <h4 className="variant-name">{pred.predicted_name}</h4>
                </div>
                <div className="loss-score">
                  <span className="label">Perplexity (Loss):</span>
                  <span className="score">{pred.loss_score}</span>
                </div>
                <div className="loss-score">
                  <span className="label">risk_score_percent:</span>
                  <span className="score">{pred.risk_score_percent}%</span>
                </div>
                <p className="description">
                  {pred.is_original
                    ? 'Wild-type sequence. Highly stable.'
                    : 'Mathematically plausible transition. Monitor for structural viability.'}
                </p>
              </div>
            ))}
          </div>

          {error && (
            <div className="error-message" style={{ marginTop: '1rem' }}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          <p className="hint" style={{ marginTop: '1.5rem', fontSize: '0.85rem' }}>
            API base: <code className="inline-code">{API_BASE}</code> — set <code className="inline-code">VITE_API_BASE</code>{' '}
            in <code className="inline-code">.env</code> if your backend uses another host or port. Ensure{' '}
            <code className="inline-code">lstm_weights.pth</code> and mobility CSV exist under{' '}
            <code className="inline-code">src/module3_lstm/</code>.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
