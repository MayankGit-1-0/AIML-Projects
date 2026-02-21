/**
 * Credit Card Fraud Detection — Frontend Logic
 * Fetches metrics, renders model comparison cards, handles predictions.
 */

const METRIC_LABELS = {
  accuracy:  { label: 'Accuracy',  cls: 'acc'  },
  precision: { label: 'Precision', cls: 'prec' },
  recall:    { label: 'Recall',    cls: 'rec'  },
  f1:        { label: 'F1 Score',  cls: 'f1'   },
  roc_auc:   { label: 'ROC-AUC',  cls: 'auc'  },
};

let metricsData = {};
let bestModel   = '';
let activeModel  = '';

/* ── Bootstrap ────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res  = await fetch('/api/metrics');
    const json = await res.json();
    metricsData = json.metrics;
    bestModel   = json.best_model;
    activeModel = bestModel;

    renderBestBanner();
    renderModelCards();
    updateDetailPanel();
    buildPredictForm();
  } catch (err) {
    console.error('Failed to load metrics:', err);
  } finally {
    document.getElementById('loader').classList.add('hidden');
  }
});

/* ── Best Banner ──────────────────────────────────────────────────────────── */
function renderBestBanner() {
  const m = metricsData[bestModel];
  document.getElementById('best-name').textContent  = bestModel;
  document.getElementById('best-f1-val').textContent = m.f1 + '%';
}

/* ── Model Cards ──────────────────────────────────────────────────────────── */
function renderModelCards() {
  const grid = document.getElementById('models-grid');
  grid.innerHTML = '';

  const names = Object.keys(metricsData);
  names.forEach((name, idx) => {
    const m    = metricsData[name];
    const card = document.createElement('div');
    card.className = `model-card fade-in fade-in-delay-${idx + 1}`;
    if (name === bestModel) card.classList.add('best-card');
    if (name === activeModel) card.classList.add('active');

    let metricsHTML = '';
    for (const [key, info] of Object.entries(METRIC_LABELS)) {
      const val = m[key];
      metricsHTML += `
        <div class="metric-row">
          <span class="metric-label">${info.label}</span>
          <span class="metric-value">${val}%</span>
        </div>
        <div class="metric-bar"><div class="fill ${info.cls}" style="width:${val}%"></div></div>
      `;
    }

    card.innerHTML = `<div class="card-name">${name}</div>${metricsHTML}`;
    card.addEventListener('click', () => selectModel(name));
    grid.appendChild(card);
  });
}

function selectModel(name) {
  activeModel = name;
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
  const cards = document.querySelectorAll('.model-card');
  const names = Object.keys(metricsData);
  names.forEach((n, i) => { if (n === name) cards[i].classList.add('active'); });
  updateDetailPanel();
}

/* ── Detail Panel (Confusion Matrix + Active Metrics) ─────────────────────── */
function updateDetailPanel() {
  const m = metricsData[activeModel];
  document.getElementById('detail-model-name').textContent = activeModel;

  // confusion matrix
  const cm = m.confusion_matrix;
  document.getElementById('cm-tn').textContent = cm[0][0];
  document.getElementById('cm-fp').textContent = cm[0][1];
  document.getElementById('cm-fn').textContent = cm[1][0];
  document.getElementById('cm-tp').textContent = cm[1][1];

  // active metrics summary
  const summaryEl = document.getElementById('active-metrics-summary');
  let html = '';
  for (const [key, info] of Object.entries(METRIC_LABELS)) {
    const val = m[key];
    html += `
      <div class="metric-row">
        <span class="metric-label">${info.label}</span>
        <span class="metric-value">${val}%</span>
      </div>
      <div class="metric-bar"><div class="fill ${info.cls}" style="width:${val}%"></div></div>
    `;
  }
  summaryEl.innerHTML = html;
}

/* ── Prediction Form ──────────────────────────────────────────────────────── */
const FEATURE_ORDER = [
  'Time',
  ...Array.from({length: 28}, (_, i) => `V${i + 1}`),
  'Amount',
];

function buildPredictForm() {
  const form = document.getElementById('predict-form');
  FEATURE_ORDER.forEach(name => {
    const div   = document.createElement('div');
    div.className = 'field';
    const label = document.createElement('label');
    label.textContent = name;
    label.setAttribute('for', `feat-${name}`);
    const input = document.createElement('input');
    input.type  = 'number';
    input.step  = 'any';
    input.id    = `feat-${name}`;
    input.name  = name;
    input.value = '0';
    div.appendChild(label);
    div.appendChild(input);
    form.appendChild(div);
  });
}

function fillSampleData() {
  // Typical fraud-like values for demonstration
  const sample = {
    Time: 0,
    V1: -1.36, V2: -0.07, V3: 2.54, V4: 1.38, V5: -0.34,
    V6: 0.46,  V7: 0.24,  V8: 0.10, V9: 0.36, V10: 0.09,
    V11: -0.55, V12: -0.62, V13: -0.99, V14: -0.31, V15: 1.47,
    V16: -0.47, V17: 0.21, V18: 0.03, V19: 0.40, V20: 0.25,
    V21: -0.02, V22: 0.28, V23: -0.11, V24: 0.07, V25: 0.13,
    V26: -0.19, V27: 0.13, V28: -0.02,
    Amount: 149.62,
  };

  FEATURE_ORDER.forEach(name => {
    const input = document.getElementById(`feat-${name}`);
    if (input && sample[name] !== undefined) {
      input.value = sample[name];
    }
  });
}

async function submitPrediction() {
  const features = {};
  FEATURE_ORDER.forEach(name => {
    const input = document.getElementById(`feat-${name}`);
    features[name] = parseFloat(input.value) || 0;
  });

  const resultEl = document.getElementById('predict-result');
  resultEl.className = 'predict-result';  // reset

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: activeModel, features }),
    });
    const json = await res.json();

    if (json.error) {
      alert(json.error);
      return;
    }

    const isFraud = json.prediction_code === 1;
    resultEl.classList.add('show', isFraud ? 'fraud' : 'legit');
    document.getElementById('result-icon').textContent    = isFraud ? '🚨' : '✅';
    document.getElementById('result-verdict').textContent  = json.prediction;
    document.getElementById('result-model').textContent    = `Model: ${json.model}`;

    if (json.confidence !== null) {
      document.getElementById('result-conf-val').textContent = json.confidence + '%';
    } else {
      document.getElementById('result-conf-val').textContent = 'N/A';
    }
  } catch (err) {
    console.error(err);
    alert('Prediction failed. Check console.');
  }
}
