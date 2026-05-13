/**
 * Fixed epidemiological rates (same for all four variants) — values aligned with your Module 2 screenshot.
 * Host adaptation fields differ per variant and only change the initial spillover seed in Module 3 SEIR.
 */
export const DEMO_EP = {
  alpha: 0.2008000910282135,
  beta: 0.2993885576725006,
  gamma: 0.10034283250570297,
  sigma: 0.2008000910282135,
};

export const DEMO_STRAIN_META = {
  strain_id: 'Demo_H5N1_Strain',
  target_index: 65,
  original_nucleotide: 'T',
};

/** Four variants: same α/β/γ; host metrics differ (illustrative, screenshot-aligned). */
export const DEMO_VARIANTS = [
  {
    index: 0,
    predicted_name: 'WildType (T65)',
    nucleotide: 'T',
    is_original: true,
    loss_score: 30.2346,
    human_adaptation_probability: 0.082,
    risk_score_percent: 8.2,
    predicted_host_label: 0,
  },
  {
    index: 1,
    predicted_name: 'T65C',
    nucleotide: 'C',
    is_original: false,
    loss_score: 30.3314,
    human_adaptation_probability: 0.11,
    risk_score_percent: 11.0,
    predicted_host_label: 0,
  },
  {
    index: 2,
    predicted_name: 'T65A',
    nucleotide: 'A',
    is_original: false,
    loss_score: 30.3332,
    human_adaptation_probability: 0.135,
    risk_score_percent: 13.5,
    predicted_host_label: 0,
  },
  {
    index: 3,
    predicted_name: 'T65G',
    nucleotide: 'G',
    is_original: false,
    loss_score: 30.4512,
    human_adaptation_probability: 0.182,
    risk_score_percent: 18.2,
    predicted_host_label: 0,
  },
];

export function buildPhenotypeFromDemoVariant(v) {
  const beta = DEMO_EP.beta;
  const gamma = DEMO_EP.gamma;
  const sigma = DEMO_EP.sigma;
  const r0a = beta / gamma;
  const latent = 1 / sigma;
  return {
    human_adaptation: {
      human_adaptation_probability: v.human_adaptation_probability,
      risk_score_percent: v.risk_score_percent,
      predicted_host_label: v.predicted_host_label,
    },
    epidemiological_parameters: {
      alpha: DEMO_EP.alpha,
      beta,
      gamma,
      sigma,
      basic_reproduction_number_approx: r0a,
      mean_latent_period_days_approx: latent,
    },
    clinical_indicators: {
      transmission_summary: 'moderate_transmissibility_potential',
      severity_attention: 'routine_monitoring',
      latent_period_summary: 'moderate_latent_period',
    },
  };
}
