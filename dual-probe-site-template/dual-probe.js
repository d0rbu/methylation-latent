const splitLabels = {
  "diverse-blocks": "Diverse held-out blocks",
  "held-out-chromosome": "Held-out chromosome 7",
};
const populationLabels = {
  seen_by_held_out: "Seen × held-out",
  held_out_by_held_out: "Held-out × held-out",
};
const modeLabels = {
  sequence_inductive: "Sequence × sequence (inductive)",
  hybrid_learned_seen: "Learned seen × sequence held-out (hybrid)",
};
const distanceLabels = {
  cis_0_1kb: "0–1 kb", cis_1_4kb: "1–4 kb", cis_4_16kb: "4–16 kb",
  cis_16_64kb: "16–64 kb", cis_64_256kb: "64–256 kb",
  cis_256kb_1mb: "256 kb–1 Mb", cis_1mb_plus: ">1 Mb", trans: "trans",
};
const distanceColors = ["#9c2f2f", "#d4682d", "#d7a21b", "#72a83d", "#008d75", "#2a78b5", "#6750a4", "#5f6368"];
const contextColors = {island: "#006b57", shore: "#e18824", shelf: "#6d57a5", open_sea: "#3480b8"};
const designColors = {I: "#006b57", II: "#d37628"};
const strandColors = {F: "#2878b8", R: "#b44b55"};
const chromosomePalette = ["#006b57", "#d37628", "#6d57a5", "#3480b8", "#a94858", "#77952d", "#8c6239", "#4d7d72"];

const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "—";
const pct = value => `${(100 * Number(value)).toFixed(3)}%`;
const cellLabel = cell => `${splitLabels[cell.split_name]} · ${cell.window_size / 1024} kb`;
const metricMean = metric => Number(metric.mean);
const aggregateMetric = (cell, population, mode) => cell.results.aggregates.pair_metrics[population].uniform[mode];
const parentPair = (cell, population) => cell.results.parent_comparison.pair_metrics[population].uniform_sequence_inductive;
const distanceUniform = (cell, population) => cell.results.distance_reference.populations[population].uniform;
const latentDimension = cell => Number(cell.latent_dimension);

const assertData = data => {
  if (data.schema !== "methylation-latent.dual-probe-site-data.v2") throw new Error("Dual site-data schema differs");
  if (data.scientific_status !== "post_hoc_hypothesis_generating") throw new Error("Dual scientific-status warning differs");
  if (!Array.isArray(data.sources) || data.sources.length < 1) throw new Error("Dual source provenance is absent");
  const expectedCells = data.sources.reduce((total, source) => total + Number(source.cell_count), 0);
  if (!Array.isArray(data.cells) || data.cells.length !== expectedCells || expectedCells < 1) throw new Error("Dual experiment cell completeness differs");
  const sourceIdentities = new Set(data.sources.map(source => `${source.protocol_id}|${source.protocol_sha256}|${source.manifest_sha256}|${source.artifact_git_commit}`));
  if (sourceIdentities.size !== data.sources.length) throw new Error("Dual source provenance is duplicated");
  const identities = new Set();
  data.cells.forEach(cell => {
    identities.add(`${cell.split_name}|${cell.window_size}`);
    const sourceIdentity = `${cell.source.protocol_id}|${cell.source.protocol_sha256}|${cell.source.manifest_sha256}|${cell.source.artifact_git_commit}`;
    if (!sourceIdentities.has(sourceIdentity)) throw new Error("Dual cell source provenance is unregistered");
    if (!Number.isInteger(cell.latent_dimension) || cell.latent_dimension < 1 || !Number.isFinite(Number(cell.lambda_age))) throw new Error("Dual parent hyperparameter provenance differs");
    if (cell.results.schema !== "methylation-latent.dual-probe-results.v1") throw new Error("Dual result schema differs");
    if (cell.results.test_evaluated_strategy_count !== 1) throw new Error("Unselected alpha reached test evaluation");
    if (cell.selection.test_metrics_read !== false || cell.selection.candidates.length !== 7) throw new Error("Validation-only alpha selection differs");
    if (cell.selection.selected_alpha_strategy !== cell.results.selected_alpha_strategy) throw new Error("Selected alpha differs between artifacts");
    if (cell.results.seed_results.length !== 3) throw new Error("Three-seed result axis differs");
  });
  if (identities.size !== data.cells.length) throw new Error("Dual experiment cells are duplicated");
};

const element = (name, text = "") => {
  const node = document.createElement(name);
  node.textContent = text;
  return node;
};
const summaryCard = (title, value, detail) => {
  const card = element("div"); card.className = "summary-card";
  card.append(element("span", title), element("strong", value), element("small", detail));
  return card;
};
const finding = (title, body) => {
  const node = element("article"); node.className = "finding";
  node.append(element("strong", title), element("p", body)); return node;
};
const addCellOptions = (select, cells) => cells.forEach((cell, index) => {
  const option = element("option", cellLabel(cell)); option.value = String(index); select.append(option);
});
const replaceMetricStrip = (selector, entries) => {
  const node = document.querySelector(selector); node.replaceChildren();
  entries.forEach(([name, value]) => node.append(element("span", `${name}: ${value}`)));
};

const chartFrame = (canvas, {minimum, maximum, xLabels, yLabel}) => {
  const context = canvas.getContext("2d"); const width = canvas.width, height = canvas.height;
  const margins = {left: 78, right: 32, top: 35, bottom: 100};
  context.clearRect(0, 0, width, height); context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);
  const y = value => height - margins.bottom - (Number(value) - minimum) / (maximum - minimum) * (height - margins.top - margins.bottom);
  const x = index => margins.left + (index + .5) / xLabels.length * (width - margins.left - margins.right);
  context.strokeStyle = "#d8e0db"; context.fillStyle = "#516159"; context.font = "13px sans-serif";
  for (let index = 0; index <= 5; index += 1) {
    const value = minimum + index / 5 * (maximum - minimum); const py = y(value);
    context.beginPath(); context.moveTo(margins.left, py); context.lineTo(width - margins.right, py); context.stroke();
    context.fillText(value.toFixed(3), 18, py + 4);
  }
  context.textAlign = "center";
  xLabels.forEach((label, index) => {
    context.save(); context.translate(x(index), height - margins.bottom + 16); context.rotate(-Math.PI / 5); context.fillText(label, 0, 0); context.restore();
  });
  context.save(); context.translate(18, height / 2); context.rotate(-Math.PI / 2); context.font = "15px sans-serif"; context.fillText(yLabel, 0, 0); context.restore();
  context.textAlign = "left";
  return {context, x, y, width, height, margins};
};

const drawScatter = (canvas, xs, ys, colors, {xLabel, yLabel}) => {
  if (xs.length !== ys.length || xs.length !== colors.length || xs.length < 2) throw new Error("Scatter axes differ or are empty");
  const context = canvas.getContext("2d"); const width = canvas.width, height = canvas.height; const margin = 76;
  context.clearRect(0, 0, width, height); context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);
  const values = [...xs, ...ys].map(Number); const rawMin = Math.min(...values), rawMax = Math.max(...values);
  const pad = Math.max((rawMax - rawMin) * .08, .01); const minimum = rawMin - pad, maximum = rawMax + pad;
  const x = value => margin + (Number(value) - minimum) / (maximum - minimum) * (width - 2 * margin);
  const y = value => height - margin - (Number(value) - minimum) / (maximum - minimum) * (height - 2 * margin);
  context.strokeStyle = "#d8e0db"; context.lineWidth = 1; context.font = "13px sans-serif";
  for (let index = 0; index <= 4; index += 1) {
    const value = minimum + index / 4 * (maximum - minimum);
    context.beginPath(); context.moveTo(x(value), margin); context.lineTo(x(value), height - margin); context.stroke();
    context.beginPath(); context.moveTo(margin, y(value)); context.lineTo(width - margin, y(value)); context.stroke();
    context.fillStyle = "#516159"; context.fillText(value.toFixed(2), x(value) - 17, height - margin + 23); context.fillText(value.toFixed(2), 15, y(value) + 4);
  }
  context.strokeStyle = "#6d756f"; context.lineWidth = 2; context.beginPath(); context.moveTo(x(minimum), y(minimum)); context.lineTo(x(maximum), y(maximum)); context.stroke();
  xs.forEach((value, index) => { context.fillStyle = colors[index]; context.globalAlpha = .38; context.beginPath(); context.arc(x(value), y(ys[index]), 2.6, 0, 2 * Math.PI); context.fill(); });
  context.globalAlpha = 1; context.fillStyle = "#17221d"; context.font = "16px sans-serif"; context.textAlign = "center"; context.fillText(xLabel, width / 2, height - 16);
  context.save(); context.translate(21, height / 2); context.rotate(-Math.PI / 2); context.fillText(yLabel, 0, 0); context.restore(); context.textAlign = "left";
};

const renderFindings = data => {
  const cells = data.cells;
  const age = cells.map(cell => metricMean(cell.results.aggregates.age_metrics.pearson));
  const parentAge = cells.map(cell => Number(cell.results.parent_comparison.age_metrics.pearson));
  const held = cells.map(cell => metricMean(aggregateMetric(cell, "held_out_by_held_out", "sequence_inductive").pearson));
  const parentHeld = cells.map(cell => Number(parentPair(cell, "held_out_by_held_out").pearson));
  const hybrid = cells.map(cell => metricMean(aggregateMetric(cell, "seen_by_held_out", "hybrid_learned_seen").pearson));
  const align = cells.map(cell => metricMean(cell.results.aggregates.train_alignment.mean));
  const selectedCounts = new Map();
  cells.forEach(cell => selectedCounts.set(cell.results.selected_alpha_strategy, (selectedCounts.get(cell.results.selected_alpha_strategy) || 0) + 1));
  const selections = [...selectedCounts.entries()].map(([strategy, count]) => `${strategy.replaceAll("_", " ")} in ${count}/${cells.length} cells`).join("; ");
  const node = document.querySelector("#findings");
  node.append(
    finding("Validation-selected catch strategies.", selections),
    finding("Fully inductive pair prediction.", `Held-out × held-out Pearson spans ${fmt(Math.min(...held))}–${fmt(Math.max(...held))}, versus ${fmt(Math.min(...parentHeld))}–${fmt(Math.max(...parentHeld))} for the corresponding parent metrics.`),
    finding("Held-out age prediction.", `Probe–age Pearson spans ${fmt(Math.min(...age))}–${fmt(Math.max(...age))}; the corresponding parent full-model range is ${fmt(Math.min(...parentAge))}–${fmt(Math.max(...parentAge))}.`),
    finding("Known-site hybrid prediction and alignment.", `Hybrid seen × held-out Pearson spans ${fmt(Math.min(...hybrid))}–${fmt(Math.max(...hybrid))}; mean learned/sequence cosine after refit spans ${fmt(Math.min(...align))}–${fmt(Math.max(...align))}.`),
    finding("Read MSE together with correlation.", "Prediction shrinkage can reduce squared error while weakening rank signal. The tables therefore retain Pearson, MSE, and the training-only distance reference separately."),
  );
};

const renderHeadline = data => {
  const bestHeld = [...data.cells].sort((a, b) => metricMean(aggregateMetric(b, "held_out_by_held_out", "sequence_inductive").pearson) - metricMean(aggregateMetric(a, "held_out_by_held_out", "sequence_inductive").pearson))[0];
  const bestAge = [...data.cells].sort((a, b) => metricMean(b.results.aggregates.age_metrics.pearson) - metricMean(a.results.aggregates.age_metrics.pearson))[0];
  const strategies = [...new Set(data.cells.map(cell => cell.results.selected_alpha_strategy))];
  const node = document.querySelector("#headline");
  node.append(
    summaryCard("Selected strategy", strategies.length === 1 ? strategies[0].replaceAll("_", " ") : `${strategies.length} strategies`, `${data.cells.length} registered split/window cells`),
    summaryCard("Best held-out × held-out Pearson", fmt(metricMean(aggregateMetric(bestHeld, "held_out_by_held_out", "sequence_inductive").pearson)), cellLabel(bestHeld)),
    summaryCard("Best age Pearson", fmt(metricMean(bestAge.results.aggregates.age_metrics.pearson)), cellLabel(bestAge)),
    summaryCard("Test strategies per cell", "1", "Validation-selected only"),
  );
};

const renderAlpha = data => {
  const select = document.querySelector("#alpha-cell"); addCellOptions(select, data.cells);
  const render = () => {
    const cell = data.cells[Number(select.value)]; const candidates = cell.selection.candidates;
    const scores = candidates.map(candidate => Number(candidate.mean_seed_minimum_validation_score));
    const deviations = candidates.map(candidate => Number(candidate.standard_deviation_seed_minimum_validation_score));
    const minimum = Math.min(...scores.map((score, index) => score - deviations[index])) - .002;
    const maximum = Math.max(...scores.map((score, index) => score + deviations[index])) + .002;
    const frame = chartFrame(document.querySelector("#alpha-chart"), {minimum, maximum, xLabels: candidates.map(candidate => candidate.alpha_strategy.replaceAll("_", " ")), yLabel: "Validation pair MSE + age MSE"});
    candidates.forEach((candidate, index) => {
      const px = frame.x(index), py = frame.y(scores[index]), low = frame.y(scores[index] - deviations[index]), high = frame.y(scores[index] + deviations[index]);
      frame.context.strokeStyle = candidate.alpha_strategy === cell.selection.selected_alpha_strategy ? "#006b57" : "#78847d"; frame.context.lineWidth = 2;
      frame.context.beginPath(); frame.context.moveTo(px, low); frame.context.lineTo(px, high); frame.context.stroke();
      frame.context.fillStyle = candidate.alpha_strategy === cell.selection.selected_alpha_strategy ? "#006b57" : "#9ba69f"; frame.context.beginPath(); frame.context.arc(px, py, 7, 0, 2 * Math.PI); frame.context.fill();
    });
    const selected = candidates.find(candidate => candidate.alpha_strategy === cell.selection.selected_alpha_strategy);
    replaceMetricStrip("#alpha-details", [["selected", selected.alpha_strategy], ["mean validation score", fmt(selected.mean_seed_minimum_validation_score)], ["seed SD", fmt(selected.standard_deviation_seed_minimum_validation_score)], ["selected steps", selected.seeds.map(seed => seed.selected_step).join(", ")]]);
  };
  select.addEventListener("change", render); render();
};

const renderMetricsTable = data => {
  const table = document.querySelector("#metrics-table"); const head = element("thead"); const row = element("tr");
  ["Split/window", "d", "alpha", "Age r", "Parent age r", "HH sequence r", "Parent HH r", "HH sequence MSE", "Distance MSE", "Hybrid seen×held r", "alignment"]
    .forEach(value => row.append(element("th", value))); head.append(row); table.append(head); const body = element("tbody");
  data.cells.forEach(cell => {
    const result = cell.results; const tableRow = element("tr"); const held = aggregateMetric(cell, "held_out_by_held_out", "sequence_inductive");
    const hybrid = aggregateMetric(cell, "seen_by_held_out", "hybrid_learned_seen");
    [cellLabel(cell), latentDimension(cell), result.selected_alpha_strategy,
      fmt(metricMean(result.aggregates.age_metrics.pearson)), fmt(result.parent_comparison.age_metrics.pearson), fmt(metricMean(held.pearson)), fmt(parentPair(cell, "held_out_by_held_out").pearson), fmt(metricMean(held.mse)), fmt(distanceUniform(cell, "held_out_by_held_out").mse), fmt(metricMean(hybrid.pearson)), fmt(metricMean(result.aggregates.train_alignment.mean))]
      .forEach(value => tableRow.append(element("td", String(value)))); body.append(tableRow);
  }); table.append(body);
};

const renderPairScatter = data => {
  const cellSelect = document.querySelector("#pair-cell"), populationSelect = document.querySelector("#pair-population"), modeSelect = document.querySelector("#pair-mode");
  addCellOptions(cellSelect, data.cells);
  Object.entries(populationLabels).forEach(([value, label]) => { const option = element("option", label); option.value = value; populationSelect.append(option); });
  const rebuildModes = () => {
    const cell = data.cells[Number(cellSelect.value)], population = populationSelect.value; const predictions = cell.results.display.pairs[population].predictions;
    modeSelect.replaceChildren(); Object.keys(predictions).forEach(value => { const option = element("option", modeLabels[value]); option.value = value; modeSelect.append(option); });
  };
  const render = () => {
    const cell = data.cells[Number(cellSelect.value)], population = populationSelect.value, mode = modeSelect.value;
    const display = cell.results.display.pairs[population]; const metric = aggregateMetric(cell, population, mode);
    const colors = display.distance_class.map(value => distanceColors[Number(value)]);
    drawScatter(document.querySelector("#pair-scatter"), display.target, display.predictions[mode].mean_prediction, colors, {xLabel: "Empirical cross-person probe correlation", yLabel: "Predicted cosine"});
    replaceMetricStrip("#pair-scatter-metrics", [["MSE", fmt(metricMean(metric.mse))], ["Pearson", fmt(metricMean(metric.pearson))], ["R²", fmt(metricMean(metric.r_squared))], ["seed SD of r", fmt(metric.pearson.standard_deviation)], ["display points", display.target.length]]);
  };
  cellSelect.addEventListener("change", () => { rebuildModes(); render(); }); populationSelect.addEventListener("change", () => { rebuildModes(); render(); }); modeSelect.addEventListener("change", render);
  rebuildModes(); render();
};

const annotationColor = (annotation, mode) => {
  if (mode === "context") return contextColors[annotation.context];
  if (mode === "design") return designColors[annotation.design];
  if (mode === "manifest_strand") return strandColors[annotation.manifest_strand];
  return chromosomePalette[(Number(annotation.chromosome) - 1) % chromosomePalette.length];
};
const renderAgeScatter = data => {
  const cellSelect = document.querySelector("#age-cell"), colorSelect = document.querySelector("#age-color"); addCellOptions(cellSelect, data.cells);
  [["context", "Genomic context"], ["design", "Infinium chemistry"], ["chromosome", "Chromosome"], ["manifest_strand", "Manifest strand"]].forEach(([value, label]) => { const option = element("option", label); option.value = value; colorSelect.append(option); });
  const render = () => {
    const cell = data.cells[Number(cellSelect.value)], display = cell.results.display.age, mode = colorSelect.value;
    const colors = display.global_indices.map(index => annotationColor(data.probe_annotations[String(index)], mode));
    drawScatter(document.querySelector("#age-scatter"), display.target, display.mean_prediction, colors, {xLabel: "Empirical probe–age correlation", yLabel: "Predicted cosine to age direction"});
    const metric = cell.results.aggregates.age_metrics;
    replaceMetricStrip("#age-scatter-metrics", [["MSE", fmt(metricMean(metric.mse))], ["Pearson", fmt(metricMean(metric.pearson))], ["Parent Pearson", fmt(cell.results.parent_comparison.age_metrics.pearson)], ["R²", fmt(metricMean(metric.r_squared))], ["display probes", display.target.length]]);
  };
  cellSelect.addEventListener("change", render); colorSelect.addEventListener("change", render); render();
};

const renderDistance = data => {
  const cellSelect = document.querySelector("#distance-cell"), populationSelect = document.querySelector("#distance-population"); addCellOptions(cellSelect, data.cells);
  Object.entries(populationLabels).forEach(([value, label]) => { const option = element("option", label); option.value = value; populationSelect.append(option); });
  const render = () => {
    const cell = data.cells[Number(cellSelect.value)], population = populationSelect.value;
    const modelModes = cell.results.aggregates.pair_metrics[population].by_distance;
    const sequence = modelModes.sequence_inductive, reference = cell.results.distance_reference.populations[population].by_distance;
    const labels = sequence.map(row => distanceLabels[row.distance_class]);
    const target = sequence.map(row => Number(row.metrics.target_mean)); const seq = sequence.map(row => metricMean(row.metrics.prediction_mean));
    const baseline = reference.map(row => Number(row.metrics.prediction_mean));
    if (reference.map(row => row.distance_class).join("|") !== sequence.map(row => row.distance_class).join("|")) throw new Error("Distance strata differ between model and baseline");
    const hybrid = modelModes.hybrid_learned_seen ? modelModes.hybrid_learned_seen.map(row => metricMean(row.metrics.prediction_mean)) : null;
    const values = [...target, ...seq, ...baseline, ...(hybrid || [])]; const pad = Math.max((Math.max(...values) - Math.min(...values)) * .12, .01);
    const frame = chartFrame(document.querySelector("#distance-chart"), {minimum: Math.min(...values) - pad, maximum: Math.max(...values) + pad, xLabels: labels, yLabel: "Mean correlation / cosine"});
    const series = [["Empirical mean", target, "#17221d"], ["Distance f(|Δpos|)", baseline, "#d37628"], ["Sequence", seq, "#006b57"]]; if (hybrid) series.push(["Hybrid", hybrid, "#6d57a5"]);
    series.forEach(([name, points, color], seriesIndex) => {
      frame.context.strokeStyle = color; frame.context.fillStyle = color; frame.context.lineWidth = 2.5; frame.context.beginPath();
      points.forEach((value, index) => { const px = frame.x(index), py = frame.y(value); if (index === 0) frame.context.moveTo(px, py); else frame.context.lineTo(px, py); }); frame.context.stroke();
      points.forEach((value, index) => { frame.context.beginPath(); frame.context.arc(frame.x(index), frame.y(value), 4, 0, 2 * Math.PI); frame.context.fill(); });
      frame.context.fillText(name, frame.margins.left + seriesIndex * 180, 20);
    });
    const container = document.querySelector("#distance-table"); container.replaceChildren(); const table = element("table"), thead = element("thead"), header = element("tr");
    ["Distance", "count", "sequence MSE", "sequence r", "distance MSE", ...(hybrid ? ["hybrid MSE", "hybrid r"] : [])].forEach(value => header.append(element("th", value))); thead.append(header); table.append(thead); const body = element("tbody");
    sequence.forEach((row, index) => { const tr = element("tr"); const fields = [distanceLabels[row.distance_class], row.metrics.count, fmt(metricMean(row.metrics.mse)), fmt(metricMean(row.metrics.pearson)), fmt(reference[index].metrics.mse)]; if (hybrid) fields.push(fmt(metricMean(modelModes.hybrid_learned_seen[index].metrics.mse)), fmt(metricMean(modelModes.hybrid_learned_seen[index].metrics.pearson))); fields.forEach(value => tr.append(element("td", String(value)))); body.append(tr); }); table.append(body); container.append(table);
  };
  cellSelect.addEventListener("change", render); populationSelect.addEventListener("change", render); render();
};

const renderAudit = data => {
  const table = document.querySelector("#audit-table"), thead = element("thead"), header = element("tr");
  ["Split/window", "effective rank", "raw condition", "retained condition", "discarded spectrum", "discarded cross moment", "initial cosine", "final alignment"]
    .forEach(value => header.append(element("th", value))); thead.append(header); table.append(thead); const body = element("tbody");
  data.cells.forEach(cell => {
    const selected = cell.validation_histories.find(candidate => candidate.alpha_strategy === cell.results.selected_alpha_strategy);
    const audits = selected.seeds.map(seed => seed.initialization); const row = element("tr");
    const range = (key, formatter = fmt) => { const values = audits.map(audit => Number(audit[key])); return `${formatter(Math.min(...values))}–${formatter(Math.max(...values))}`; };
    [cellLabel(cell), range("effective_rank", value => Number(value).toFixed(0)), Number(audits[0].raw_condition_number).toExponential(2), Number(audits[0].retained_condition_number).toExponential(2), range("discarded_spectral_mass_fraction", value => Number(value).toExponential(2)), range("discarded_cross_moment_fraction", pct), range("initial_mean_cosine"), fmt(metricMean(cell.results.aggregates.train_alignment.mean))]
      .forEach(value => row.append(element("td", String(value)))); body.append(row);
  }); table.append(body);
};

fetch("dual-probe-data.json", {cache: "no-store"}).then(response => {
  if (!response.ok) throw new Error(`Dual result request failed: ${response.status}`); return response.json();
}).then(data => {
  assertData(data); document.querySelector("#status").textContent = `Complete: ${data.cells.length} split/window cells × 7 alpha strategies × 3 seeds; selected-only test evaluation.`;
  document.querySelector("#protocol").textContent = `${data.sources.length} registered dual protocol source${data.sources.length === 1 ? "" : "s"} · parent ${data.parent_protocol_id}`;
  const provenance = document.querySelector("#provenance");
  const runtimes = [...new Set(data.cells.map(cell => cell.results.identity.runtime_device))].join(", ");
  [`Parent protocol: ${data.parent_protocol_id}`, `Parent protocol SHA-256: ${data.parent_protocol_sha256}`, `Data bundle SHA-256: ${data.data_bundle_sha256}`, ...data.sources.flatMap(source => [`Dual protocol: ${source.protocol_id}`, `Protocol SHA-256: ${source.protocol_sha256}`, `${source.manifest_schema} SHA-256: ${source.manifest_sha256}`, `Artifact Git commit: ${source.artifact_git_commit}`, `Registered windows: ${source.windows.join(", ")}`]), `Site compiler Git commit: ${data.compiler_git_commit}`, `Runtime device(s): ${runtimes}; deterministic algorithms enabled and CUDA TF32 disabled`, "All validation/test learned-row counts: exactly zero"]
    .forEach(value => provenance.append(element("li", value)));
  renderFindings(data); renderHeadline(data); renderAlpha(data); renderMetricsTable(data); renderPairScatter(data); renderAgeScatter(data); renderDistance(data); renderAudit(data);
}).catch(error => {
  document.querySelector("#status").textContent = `Report failed loudly: ${error.message}`; throw error;
});
