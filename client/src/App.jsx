import { useEffect, useMemo, useRef, useState } from "react";

import {
  fetchAnalyses,
  fetchAnalysisReport,
  fetchTrainingReport,
  removeAnalysis,
  resolveImageUrl,
  uploadAnalysis,
} from "./api.js";

const initialFormState = {
  category: "",
  image: null,
};

function formatDate(value) {
  return new Intl.DateTimeFormat("fr-FR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function statusLabel(status) {
  if (status === "critical") return "Critique";
  if (status === "warning") return "Surveillance";
  return "Saine";
}

function statusDescription(status) {
  if (status === "critical") return "Action recommandee rapidement";
  if (status === "warning") return "Verification conseillee";
  return "Etat visuel rassurant";
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-box">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function HistoryChart({ title, history }) {
  const values = history || [];
  if (!values.length) {
    return null;
  }

  const width = 320;
  const height = 150;
  const padding = 24;
  const maxValue = Math.max(...values, 1);
  const minValue = Math.min(...values, 0);
  const range = Math.max(maxValue - minValue, 0.001);
  const points = values
    .map((value, index) => {
      const x = padding + (index * (width - padding * 2)) / Math.max(values.length - 1, 1);
      const y = height - padding - ((value - minValue) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="history-chart">
      <div className="chart-heading">
        <span>{title}</span>
        <strong>{values.at(-1)?.toFixed(4)}</strong>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis-line" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis-line" />
        <polyline fill="none" points={points} className="history-line" />
        {values.map((value, index) => {
          const x = padding + (index * (width - padding * 2)) / Math.max(values.length - 1, 1);
          const y = height - padding - ((value - minValue) / range) * (height - padding * 2);
          return <circle key={`${title}-${index}`} cx={x} cy={y} r="3" className="history-point" />;
        })}
      </svg>
    </div>
  );
}

function HistogramBlock({ title, channels }) {
  const width = 320;
  const height = 180;
  const padding = 28;
  const palette = {
    r: "#c65d2c",
    g: "#5a8f29",
    b: "#2d6fa3",
    h: "#b5561e",
    s: "#2f7f6d",
    v: "#5a5fcf",
  };

  return (
    <div className="panel subtle-panel">
      <h3>{title}</h3>
      <div className="histogram-groups">
        {Object.entries(channels || {}).map(([channel, values]) => (
          <div key={channel} className="histogram-group">
            <div className="chart-heading">
              <span>{channel.toUpperCase()}</span>
              <strong>{values.length} bins</strong>
            </div>
            <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} ${channel}`}>
              <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis-line" />
              <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis-line" />
              {values.map((value, index) => {
                const barWidth = (width - padding * 2) / values.length;
                const normalizedHeight = Math.max(2, value * (height - padding * 2));
                const x = padding + index * barWidth + 1;
                const y = height - padding - normalizedHeight;
                return (
                  <rect
                    key={`${channel}-${index}`}
                    x={x}
                    y={y}
                    width={Math.max(barWidth - 2, 1)}
                    height={normalizedHeight}
                    rx="1"
                    fill={palette[channel.toLowerCase()] || "#456b55"}
                    opacity="0.85"
                  />
                );
              })}
            </svg>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const fileInputRef = useRef(null);
  const [analyses, setAnalyses] = useState([]);
  const [formState, setFormState] = useState(initialFormState);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [selectedAnalysisId, setSelectedAnalysisId] = useState("");
  const [selectedReport, setSelectedReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");
  const [trainingReport, setTrainingReport] = useState(null);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [trainingError, setTrainingError] = useState("");

  useEffect(() => {
    loadAnalyses();
    loadTrainingReport();
  }, []);

  useEffect(() => {
    if (!formState.image) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(formState.image);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [formState.image]);

  const stats = useMemo(() => {
    const healthy = analyses.filter((item) => item.healthStatus === "healthy").length;
    const warning = analyses.filter((item) => item.healthStatus === "warning").length;
    const critical = analyses.filter((item) => item.healthStatus === "critical").length;

    return {
      total: analyses.length,
      healthy,
      warning,
      critical,
      mlPowered: analyses.filter((item) => item.inferenceMode === "ml").length,
    };
  }, [analyses]);

  async function loadAnalyses() {
    try {
      setLoading(true);
      setError("");
      const data = await fetchAnalyses();
      setAnalyses(data);
      if (!selectedAnalysisId && data[0]) {
        setSelectedAnalysisId(data[0]._id);
        loadAnalysisReport(data[0]._id);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadAnalysisReport(id) {
    if (!id) return;
    try {
      setReportLoading(true);
      setReportError("");
      const data = await fetchAnalysisReport(id);
      setSelectedAnalysisId(id);
      setSelectedReport(data);
    } catch (requestError) {
      setReportError(requestError.message);
    } finally {
      setReportLoading(false);
    }
  }

  async function loadTrainingReport() {
    try {
      setTrainingLoading(true);
      setTrainingError("");
      const data = await fetchTrainingReport();
      setTrainingReport(data);
    } catch (requestError) {
      setTrainingError(requestError.message);
    } finally {
      setTrainingLoading(false);
    }
  }

  function handleInputChange(event) {
    const { name, value, files } = event.target;
    setSuccessMessage("");
    setFormState((current) => ({
      ...current,
      [name]: files ? files[0] : value,
    }));
  }

  function resetForm() {
    setFormState(initialFormState);
    setError("");
    setSuccessMessage("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!formState.image) {
      setError("Ajoutez une image avant l'envoi.");
      return;
    }

    try {
      setSubmitting(true);
      setError("");
      setSuccessMessage("");

      const payload = new FormData();
      payload.append("image", formState.image);
      payload.append("category", formState.category);

      const created = await uploadAnalysis(payload);
      setAnalyses((current) => [created, ...current]);
      setSelectedAnalysisId(created._id);
      await loadAnalysisReport(created._id);
      resetForm();
      setSuccessMessage("Analyse terminee et enregistree.");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id) {
    try {
      setError("");
      setSuccessMessage("");
      await removeAnalysis(id);
      const nextAnalyses = analyses.filter((item) => item._id !== id);
      setAnalyses(nextAnalyses);

      if (selectedAnalysisId === id) {
        setSelectedReport(null);
        if (nextAnalyses[0]) {
          setSelectedAnalysisId(nextAnalyses[0]._id);
          await loadAnalysisReport(nextAnalyses[0]._id);
        } else {
          setSelectedAnalysisId("");
        }
      }
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  return (
    <div className="page-shell">
      <section className="summary-strip">
        <article className="summary-card">
          <span>{stats.total}</span>
          <p>Analyses</p>
        </article>
        <article className="summary-card">
          <span>{stats.healthy}</span>
          <p>Saines</p>
        </article>
        <article className="summary-card">
          <span>{stats.warning + stats.critical}</span>
          <p>A verifier</p>
        </article>
        <article className="summary-card">
          <span>{stats.mlPowered}</span>
          <p>Mode ML</p>
        </article>
      </section>

      <section className="content-grid">
        <form className="panel upload-panel" onSubmit={handleSubmit}>
          <div className="panel-heading">
            <h2>Nouvelle analyse</h2>
            <p>Chargez une feuille pour generer le diagnostic et le rapport detaille de traitement.</p>
          </div>

          <label className="field">
            <span>Categorie</span>
            <input
              name="category"
              type="text"
              placeholder="healthy, rust, blight..."
              value={formState.category}
              onChange={handleInputChange}
            />
          </label>

          <label className="field file-field">
            <span>Image</span>
            <input
              ref={fileInputRef}
              name="image"
              type="file"
              accept="image/*"
              onChange={handleInputChange}
            />
          </label>

          {previewUrl ? (
            <div className="preview-card">
              <img src={previewUrl} alt="Previsualisation" />
            </div>
          ) : (
            <div className="preview-placeholder">
              Deposez une image nette de feuille pour voir l'apercu ici.
            </div>
          )}

          <div className="upload-actions">
            <button className="primary-button" type="submit" disabled={submitting}>
              {submitting ? "Analyse en cours..." : "Analyser et sauvegarder"}
            </button>
            <button className="ghost-button" type="button" onClick={resetForm} disabled={submitting}>
              Reinitialiser
            </button>
          </div>

          {successMessage ? <p className="success-message">{successMessage}</p> : null}
          {error ? <p className="error-message">{error}</p> : null}
        </form>

        <section className="panel history-panel">
          <div className="panel-heading panel-heading-split">
            <div>
              <h2>Historique</h2>
              <p>Choisissez une analyse pour afficher tout le pipeline demande dans le projet.</p>
            </div>
            <button className="ghost-button" type="button" onClick={loadAnalyses}>
              Rafraichir
            </button>
          </div>

          {loading ? <p className="empty-state">Chargement des analyses...</p> : null}

          {!loading && analyses.length === 0 ? (
            <div className="empty-card">
              <h3>Aucune analyse disponible</h3>
              <p>Envoyez une premiere image pour alimenter l'historique et verifier le pipeline.</p>
            </div>
          ) : null}

          <div className="card-list">
            {analyses.map((analysis) => (
              <article
                className={`analysis-card ${selectedAnalysisId === analysis._id ? "analysis-card-active" : ""}`}
                key={analysis._id}
              >
                <img
                  className="analysis-image"
                  src={resolveImageUrl(analysis.imageUrl)}
                  alt={analysis.originalName}
                />

                <div className="analysis-body">
                  <div className="analysis-topline">
                    <div>
                      <h3>{analysis.originalName}</h3>
                      <p className="meta">{statusDescription(analysis.healthStatus)}</p>
                    </div>
                    <span className={`badge badge-${analysis.healthStatus}`}>
                      {statusLabel(analysis.healthStatus)}
                    </span>
                  </div>

                  <p className="diagnosis">{analysis.diagnosis}</p>

                  <div className="metric-strip">
                    <MetricCard label="Confiance" value={formatPercent(analysis.confidence)} />
                    <MetricCard label="Feuille detectee" value={formatPercent(analysis.stats?.leaf_coverage)} />
                    <MetricCard
                      label="Mode"
                      value={analysis.inferenceMode === "ml" ? "ML dataset" : "Heuristique"}
                    />
                  </div>

                  <p className="meta">Categorie: {analysis.category || "non renseignee"}</p>
                  <p className="meta">{formatDate(analysis.createdAt)}</p>

                  <div className="card-actions">
                    <button className="primary-button" type="button" onClick={() => loadAnalysisReport(analysis._id)}>
                      Voir le pipeline
                    </button>
                    <button className="ghost-button" type="button" onClick={() => handleDelete(analysis._id)}>
                      Supprimer
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      </section>

      <section className="report-grid">
        <section className="panel">
          <div className="panel-heading">
            <h2>Rapport image</h2>
            <p>Pretraitement, segmentation, histogrammes et vecteur de caracteristiques pour l'image choisie.</p>
          </div>

          {reportLoading ? <p className="empty-state">Generation du rapport image...</p> : null}
          {reportError ? <p className="error-message">{reportError}</p> : null}

          {selectedReport?.report ? (
            <div className="report-stack">
              <div className="report-section">
                <h3>Pretraitement des images</h3>
                <p>{selectedReport.report.preprocessing.impactExamples.join(" ")}</p>
                <div className="image-grid four-up">
                  <img src={selectedReport.report.preprocessing.images.rgb} alt="RGB" />
                  <img src={selectedReport.report.preprocessing.images.gray} alt="Gray" />
                  <img src={selectedReport.report.preprocessing.images.blurred} alt="Blurred" />
                  <img src={selectedReport.report.preprocessing.images.contourOverlay} alt="Contours" />
                </div>
                <div className="chip-list">
                  {selectedReport.report.preprocessing.steps.map((step) => (
                    <span key={step}>{step}</span>
                  ))}
                </div>
                <div className="dual-grid">
                  <HistogramBlock title="Histogrammes RGB" channels={selectedReport.report.preprocessing.histograms.rgb} />
                  <HistogramBlock title="Histogrammes HSV" channels={selectedReport.report.preprocessing.histograms.hsv} />
                </div>
              </div>

              <div className="report-section">
                <h3>Segmentation et extraction de contours</h3>
                <p>{selectedReport.report.segmentation.subjectiveComparison.join(" ")}</p>
                <div className="image-grid five-up">
                  <img src={selectedReport.report.segmentation.images.hsvMask} alt="HSV mask" />
                  <img src={selectedReport.report.segmentation.images.threshold} alt="Threshold" />
                  <img src={selectedReport.report.segmentation.images.sobel} alt="Sobel" />
                  <img src={selectedReport.report.segmentation.images.canny} alt="Canny" />
                  <img src={selectedReport.report.segmentation.images.clustered} alt="Clustered" />
                </div>
              </div>

              <div className="report-section">
                <h3>Extraction de caracteristiques</h3>
                <div className="metric-strip">
                  <MetricCard
                    label="Features totales"
                    value={selectedReport.report.features.vectorReadyForMl.totalFeatures}
                  />
                  <MetricCard
                    label="Histogrammes RGB"
                    value={selectedReport.report.features.color.rgbHistogramFeatureCount}
                  />
                  <MetricCard
                    label="Histogrammes HSV"
                    value={selectedReport.report.features.color.hsvHistogramFeatureCount}
                  />
                </div>
                <p className="meta">
                  Vecteur pret pour le ML: {selectedReport.report.features.vectorReadyForMl.ready ? "oui" : "non"}.
                  Toutes les valeurs sont numeriques: {selectedReport.report.features.vectorReadyForMl.isNumeric ? "oui" : "non"}.
                </p>
                <div className="dual-grid">
                  <div className="panel subtle-panel">
                    <h3>Texture GLCM</h3>
                    <ul className="notes compact-notes">
                      {Object.entries(selectedReport.report.features.texture).map(([key, value]) => (
                        <li key={key}>
                          {key}: {value}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="panel subtle-panel">
                    <h3>Forme</h3>
                    <ul className="notes compact-notes">
                      {Object.entries(selectedReport.report.features.shape).map(([key, value]) => (
                        <li key={key}>
                          {key}: {value}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          ) : !reportLoading ? (
            <div className="empty-card">
              <h3>Aucun rapport image charge</h3>
              <p>Selectionnez une analyse dans l'historique pour afficher le pipeline detaille.</p>
            </div>
          ) : null}
        </section>

        <section className="panel">
          <div className="panel-heading panel-heading-split">
            <div>
              <h2>Classification ML et extension Deep Learning</h2>
              <p>Entrainer, evaluer accuracy/precision/recall, puis comparer les modeles.</p>
            </div>
            <button className="ghost-button" type="button" onClick={loadTrainingReport}>
              Recalculer
            </button>
          </div>

          {trainingLoading ? <p className="empty-state">Generation du rapport dataset...</p> : null}
          {trainingError ? <p className="error-message">{trainingError}</p> : null}

          {trainingReport ? (
            <div className="report-stack">
              <div className="report-section">
                <h3>Dataset et coherence du split</h3>
                  <div className="metric-strip">
                    <MetricCard label="Images dataset" value={trainingReport.dataset?.sampleCount || 0} />
                    <MetricCard label="Classes" value={trainingReport.dataset?.classCount || 0} />
                    <MetricCard label="Features / image" value={trainingReport.dataset?.featureCount || 0} />
                    <MetricCard
                      label="Train / Test"
                      value={`${Math.round((trainingReport.dataset?.trainTestSplit?.trainRatio || 0) * 100)}/${Math.round((trainingReport.dataset?.trainTestSplit?.testRatio || 0) * 100)}`}
                    />
                </div>
                <p className="meta">{trainingReport.message}</p>
              </div>

              <div className="report-section">
                <h3>Classification classique</h3>
                {trainingReport.classicalModels?.length ? (
                  <div className="model-grid">
                    {trainingReport.classicalModels.map((model) => (
                      <div key={model.modelName} className="panel subtle-panel">
                        <h3>{model.modelName}</h3>
                        <div className="metric-strip">
                          <MetricCard label="Accuracy" value={model.metrics.accuracy} />
                          <MetricCard label="Precision" value={model.metrics.precision} />
                          <MetricCard label="Recall" value={model.metrics.recall} />
                        </div>
                        <p className="meta">F1-score: {model.metrics.f1Score}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-card">
                    <h3>ML non disponible</h3>
                    <p>Ajoutez un dataset classe par dossiers sous `data/dataset` avec au moins deux classes.</p>
                  </div>
                )}
                </div>

                <div className="report-section">
                  <h3>Extension Deep Learning</h3>
                  {trainingReport.deepLearning?.available ? (
                    <div className="model-grid">
                      {trainingReport.deepLearning.models?.map((model) => (
                        <div key={model.modelName} className="panel subtle-panel">
                          <h3>{model.modelName}</h3>
                          <div className="metric-strip">
                            <MetricCard label="Accuracy" value={model.metrics.accuracy} />
                            <MetricCard label="Precision" value={model.metrics.precision} />
                            <MetricCard label="Recall" value={model.metrics.recall} />
                          </div>
                          <p className="meta">F1-score: {model.metrics.f1Score}</p>
                          <p className="meta">
                            Classes evaluees: {model.classCount}. Cible accuracy: {trainingReport.deepLearning.targetAccuracy}.
                            Resultat: {model.meetsTarget ? "atteinte" : "non atteinte"}.
                          </p>
                          <div className="dual-grid history-grid">
                            <HistoryChart title="Train accuracy" history={model.history?.accuracy} />
                            <HistoryChart title="Validation accuracy" history={model.history?.val_accuracy} />
                            <HistoryChart title="Train loss" history={model.history?.loss} />
                            <HistoryChart title="Validation loss" history={model.history?.val_loss} />
                          </div>
                          <ul className="notes compact-notes">
                            {(model.notes || []).map((line) => (
                              <li key={line}>{line}</li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>

              <div className="report-section">
                <h3>Interpretation des resultats</h3>
                <ul className="notes compact-notes">
                  {(trainingReport.interpretation || []).map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : !trainingLoading ? (
            <div className="empty-card">
              <h3>Aucun rapport dataset charge</h3>
              <p>Utilisez le bouton de recalcul pour lancer l'evaluation ML et Deep Learning.</p>
            </div>
          ) : null}
        </section>
      </section>
    </div>
  );
}
