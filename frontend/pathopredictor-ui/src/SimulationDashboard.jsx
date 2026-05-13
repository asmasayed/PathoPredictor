import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { ArrowLeft, ShieldAlert } from 'lucide-react';

function riskToneClass(percent) {
  const p = Number(percent) || 0;
  if (p < 20) return 'text-emerald-700 bg-emerald-50 border-emerald-200';
  if (p <= 50) return 'text-amber-800 bg-amber-50 border-amber-200';
  return 'text-red-800 bg-red-50 border-red-200';
}

export default function SimulationDashboard({
  module3,
  module2Phenotype,
  variantLabel,
  onBack,
  infoBanner = null,
}) {
  const ep = module2Phenotype?.epidemiological_parameters || {};
  const host = module2Phenotype?.human_adaptation || {};
  const risk = host.risk_score_percent ?? 0;

  const ts = module3?.time_series || {};
  const days = ts.days || [];
  const milestones = module3?.milestones || {};
  const r0 = milestones.r_zero ?? 0;

  const seirChartData = useMemo(() => {
    const S = ts.susceptible || [];
    const E = ts.exposed || [];
    const I = ts.infected || [];
    const R = ts.recovered || [];
    return days.map((d, i) => ({
      day: d,
      S: S[i],
      E: E[i],
      I: I[i],
      R: R[i],
    }));
  }, [days, ts]);

  const betaChartData = useMemo(() => {
    const betas = ts.dynamic_betas || [];
    return betas.map((b, i) => ({ day: i, beta: b }));
  }, [ts.dynamic_betas]);

  const betaStats = useMemo(() => {
    const arr = (ts.dynamic_betas || []).map(Number).filter((x) => !Number.isNaN(x));
    if (!arr.length) return null;
    const minB = Math.min(...arr);
    const maxB = Math.max(...arr);
    const meanB = arr.reduce((a, b) => a + b, 0) / arr.length;
    const base = ep.beta != null ? Number(ep.beta) : arr[0];
    const daysBelowBase = arr.filter((b) => b < base * 0.995).length;
    return { minB, maxB, meanB, base, daysBelowBase, n: arr.length };
  }, [ts.dynamic_betas, ep.beta]);

  const riskCard = riskToneClass(risk);

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 text-left">
      {infoBanner && (
        <div className="mb-6 rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">{infoBanner}</div>
      )}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to variants
        </button>
        <div className="flex items-center gap-2 text-slate-600">
          <ShieldAlert className="h-5 w-5 text-red-600" />
          <span className="text-sm font-medium">
            Module 3 — AI-SEIR / LSTM · {variantLabel || 'Selected variant'}
          </span>
        </div>
      </div>

      {/* Section 1 — Threat profile */}
      <section className="mb-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-lg font-semibold text-slate-900">Threat profile (executive summary)</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <div className={`rounded-xl border p-4 ${riskCard}`}>
            <p className="text-xs font-semibold uppercase tracking-wide opacity-80">Host adaptation score</p>
            <p className="mt-2 text-4xl font-bold tabular-nums">{Number(risk).toFixed(1)}%</p>
            <p className="mt-1 text-xs opacity-90">Variant-specific host adaptation (demo preset; full pipeline uses Module 2 classifier).</p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">Basic reproduction number (R₀)</p>
            <p className="mt-2 text-3xl font-bold text-slate-900 tabular-nums">{r0.toFixed(3)}</p>
            <p className="mt-2 text-xs text-slate-600">
              If R₀ is greater than 1, the virus spreads; if below 1, it decays. (Computed as base β / γ from Module
              2.)
            </p>
          </div>
          <div className="rounded-xl border border-slate-200 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">Pathogen phenotype (baseline rates)</p>
            <ul className="mt-2 space-y-1 text-sm text-slate-800">
              <li>
                <span className="font-medium text-slate-600">Transmission (β):</span>{' '}
                {ep.beta != null ? Number(ep.beta).toFixed(6) : '—'}
              </li>
              <li>
                <span className="font-medium text-slate-600">Recovery (γ):</span>{' '}
                {ep.gamma != null ? Number(ep.gamma).toFixed(6) : '—'}
              </li>
              <li>
                <span className="font-medium text-slate-600">Incubation (σ → α in SEIR):</span>{' '}
                {ep.sigma != null ? Number(ep.sigma).toFixed(6) : '—'}
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Section 2 — Main SEIR chart */}
      <section className="mb-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-1 text-lg font-semibold text-slate-900">Core AI-SEIR simulation</h2>
        <p className="mb-4 text-sm text-slate-600">
          Compartments over time with LSTM-adjusted daily β. Infected curve reflects burden on healthcare capacity.
        </p>
        <div className="h-[380px] w-full min-h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={seirChartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} label={{ value: 'Day', position: 'insideBottom', offset: -4 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${v}`)} />
              <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(2) : v)} />
              <Legend />
              <Line type="monotone" dataKey="S" name="Susceptible (S)" stroke="#64748b" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="E" name="Exposed (E)" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="I" name="Infected (I) — burden on healthcare" stroke="#dc2626" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="R" name="Recovered / removed (R)" stroke="#16a34a" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Section 3 — Milestones */}
      <section className="mb-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-lg font-semibold text-slate-900">Key milestones</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase text-slate-500">Peak infection day</p>
            <p className="mt-1 text-2xl font-bold text-slate-900">Day {milestones.peak_infection_day ?? '—'}</p>
            <p className="mt-2 text-xs text-slate-600">Hospitals hit maximum concurrent infected around this day (model index).</p>
          </div>
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase text-slate-500">Max concurrent infections</p>
            <p className="mt-1 text-2xl font-bold text-slate-900">{milestones.max_concurrent_infections ?? '—'}</p>
            <p className="mt-2 text-xs text-slate-600">Peak value of the I compartment (ceiling to integer).</p>
          </div>
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase text-slate-500">Outbreak duration (I below 1)</p>
            <p className="mt-1 text-2xl font-bold text-slate-900">Day {milestones.total_outbreak_duration ?? '—'}</p>
            <p className="mt-2 text-xs text-slate-600">First day the infected compartment drops below 1 (effective end signal).</p>
          </div>
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
            <p className="text-xs font-semibold uppercase text-slate-500">Total infected (cum. removed)</p>
            <p className="mt-1 text-2xl font-bold text-slate-900">{milestones.total_infected ?? '—'}</p>
            <p className="mt-2 text-xs text-slate-600">Final R compartment (net of initial recovered; here R₀ compartment starts at 0).</p>
          </div>
        </div>
      </section>

      {/* Section 4 — LSTM beta dynamics */}
      <section className="mb-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-1 text-lg font-semibold text-slate-900">LSTM dynamics — dynamic transmission rate</h2>
        <p className="mb-4 text-sm text-slate-600">
          AI predicts β changes based on historical mobility and lockdown constraints (pre-trained weights; no retraining on request).
        </p>
        {betaStats && (
          <div className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-xs">
              <p className="font-semibold text-slate-600">β min / max</p>
              <p className="mt-1 font-mono text-slate-900">
                {betaStats.minB.toFixed(5)} — {betaStats.maxB.toFixed(5)}
              </p>
            </div>
            <div className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-xs">
              <p className="font-semibold text-slate-600">β mean ({betaStats.n} days)</p>
              <p className="mt-1 font-mono text-slate-900">{betaStats.meanB.toFixed(5)}</p>
            </div>
            <div className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-xs">
              <p className="font-semibold text-slate-600">Days β &lt; baseline</p>
              <p className="mt-1 font-mono text-slate-900">{betaStats.daysBelowBase}</p>
            </div>
            <div className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-xs">
              <p className="font-semibold text-slate-600">Mobility prior</p>
              <p className="mt-1 text-slate-700">100-day city index → LSTM multiplier</p>
            </div>
          </div>
        )}
        <div className="h-[220px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={betaChartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
              <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(5) : v)} />
              <Legend />
              <Line type="monotone" dataKey="beta" name="Dynamic β (LSTM-adjusted)" stroke="#0ea5e9" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
