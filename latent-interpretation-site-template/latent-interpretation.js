const splitLabels = {
  "diverse-blocks": "Diverse held-out blocks",
  "held-out-chromosome": "Held-out chromosome 7",
};
const populationLabels = {
  seen_stratified: "Seen × held-out",
  held_out_stratified: "Held-out × held-out",
};
const componentLabels = {
  total: "Total pair correlation",
  age_component: "Age rank-one component",
  age_adjusted_residual: "Age-adjusted residual",
};
const componentShortLabels = {
  total: "total",
  age_component: "age component",
  age_adjusted_residual: "residual",
};
const distanceLabels = {
  cis_0_1kb: "0–1 kb", cis_1_4kb: "1–4 kb", cis_4_16kb: "4–16 kb",
  cis_16_64kb: "16–64 kb", cis_64_256kb: "64–256 kb",
  cis_256kb_1mb: "256 kb–1 Mb", cis_1mb_plus: ">1 Mb", trans: "trans",
};
const componentColors = {total: "#17221d", age_component: "#d37628", age_adjusted_residual: "#006b57"};
const distanceColors = ["#9c2f2f", "#d4682d", "#d7a21b", "#72a83d", "#008d75", "#2a78b5", "#6750a4", "#5f6368"];
const contextColors = {island: "#006b57", shore: "#e18824", shelf: "#6d57a5", open_sea: "#3480b8"};
const designColors = {I: "#006b57", II: "#d37628"};
const booleanColors = {true: "#006b57", false: "#a3aaa6"};
const chromosomePalette = ["#006b57", "#d37628", "#6d57a5", "#3480b8", "#a94858", "#77952d", "#8c6239", "#4d7d72"];

const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "—";
const fmt3 = value => Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "—";
const pct = value => `${(100 * Number(value)).toFixed(1)}%`;
const windowLabel = window => `${window / 1024} kb`;
const cellLabel = cell => `${splitLabels[cell.split_name]} · ${windowLabel(cell.window_size)}`;
const element = (name, text = "") => { const node = document.createElement(name); node.textContent = text; return node; };
const unique = values => [...new Set(values)];
const resultSchemas = {
  v1: "methylation-latent.latent-interpretation-results.v1",
  v2: "methylation-latent.latent-interpretation-results.v2",
};
const provenanceSchemas = {
  [resultSchemas.v1]: "methylation-latent.latent-interpretation-site-provenance.v1",
  [resultSchemas.v2]: "methylation-latent.latent-interpretation-site-provenance.v2",
};

const normalizeLegacyRows = rows => rows.map(row => {
  const {prediction_1kb, prediction_4kb, ...fields} = row;
  return {...fields, predictions: {"1024": prediction_1kb, "4096": prediction_4kb}};
});
const normalizeLegacyComparison = comparison => {
  const pairwiseKeys = [
    "metric_cosine", "age_pullback_cosine", "age_prediction_pearson",
    "age_prediction_mean_absolute_difference", "age_prediction_sign_agreement", "linear_cka",
    "top_neighbour_overlap", "neighbour_sample_count", "neighbour_count",
    "pair_prediction_agreement",
  ];
  const pairwise = {first_window_size: 1024, second_window_size: 4096};
  pairwiseKeys.forEach(key => { pairwise[key] = comparison[key]; });
  const pairDisplays = Object.fromEntries(Object.entries(comparison.pair_displays).map(([population, display]) => [population, {
    source_count: display.source_count,
    display_count: display.display_count,
    distance_class: display.distance_class,
    targets: {
      total: display.target_total,
      age_component: display.target_age,
      age_adjusted_residual: display.target_residual,
    },
    predictions: {
      "1024": {total: display.prediction_1kb_total, age_component: display.prediction_1kb_age, age_adjusted_residual: display.prediction_1kb_residual},
      "4096": {total: display.prediction_4kb_total, age_component: display.prediction_4kb_age, age_adjusted_residual: display.prediction_4kb_residual},
    },
  }]));
  return {
    split_name: comparison.split_name,
    window_sizes: [1024, 4096],
    pairwise_window_comparisons: [pairwise],
    age_rank_curves: comparison.age_rank_curves,
    pair_candidate_distance_class: comparison.pair_candidate_distance_class,
    pair_rank_curves: comparison.pair_rank_curves,
    age_candidates: Object.fromEntries(Object.entries(comparison.age_candidates).map(([direction, rows]) => [direction, normalizeLegacyRows(rows)])),
    pair_candidates: Object.fromEntries(Object.entries(comparison.pair_candidates).map(([direction, rows]) => [direction, normalizeLegacyRows(rows)])),
    age_display: normalizeLegacyRows(comparison.age_display),
    pair_displays: pairDisplays,
  };
};
const normalizeData = data => {
  if (data.schema === resultSchemas.v2) return data;
  if (data.schema !== resultSchemas.v1) throw new Error("Interpretation result schema differs");
  return {...data, schema: resultSchemas.v2, window_comparisons: data.window_comparisons.map(normalizeLegacyComparison)};
};

const assertData = (data, provenance, sourceSchema) => {
  if (data.schema !== resultSchemas.v2) throw new Error("Normalized interpretation result schema differs");
  if (data.status !== "post_hoc_hypothesis_generating") throw new Error("Interpretation scientific-status warning differs");
  if (!Array.isArray(data.window_comparisons) || data.window_comparisons.length !== Object.keys(splitLabels).length) throw new Error("Expected one window comparison record per split");
  const windows = data.window_comparisons[0].window_sizes;
  if (!Array.isArray(windows) || windows.length < 2) throw new Error("Interpretation window axis differs");
  if (!Array.isArray(data.cells) || data.cells.length !== Object.keys(splitLabels).length * windows.length) throw new Error("Interpretation cell grid count differs");
  if (data.audit.gpu_used !== false || data.audit.model_parameters_changed !== false) throw new Error("Interpretation runtime contract differs");
  if (data.audit.candidate_selection_reads_empirical_targets !== false) throw new Error("Candidate ranking read empirical targets");
  if (data.audit.manifest_matched_probes !== data.audit.manifest_requested_probes) throw new Error("Manifest annotation join is incomplete");
  const identities = new Set(data.cells.map(cell => `${cell.split_name}|${cell.window_size}`));
  if (identities.size !== data.cells.length) throw new Error("Interpretation cell grid is duplicated");
  const expectedPairs = windows.flatMap((first, index) => windows.slice(index + 1).map(second => `${first}|${second}`));
  const ageDisplayCount = data.window_comparisons[0].age_display.length;
  const candidateCount = data.window_comparisons[0].age_candidates.positive.length;
  if (ageDisplayCount < 1 || candidateCount < 1) throw new Error("Interpretation display payload is empty");
  data.window_comparisons.forEach(comparison => {
    if (comparison.window_sizes.join("|") !== windows.join("|")) throw new Error("Window axes differ between splits");
    const observedPairs = comparison.pairwise_window_comparisons.map(row => `${row.first_window_size}|${row.second_window_size}`);
    if (observedPairs.join(",") !== expectedPairs.join(",")) throw new Error("Pairwise window-comparison axis differs");
    if (comparison.age_display.length !== ageDisplayCount) throw new Error("Age display sample differs");
    comparison.age_display.forEach(row => { if (Object.keys(row.predictions).length !== windows.length) throw new Error("Age display window axis differs"); });
    ["positive", "negative"].forEach(direction => {
      if (comparison.age_candidates[direction].length !== candidateCount || comparison.pair_candidates[direction].length !== candidateCount) throw new Error("Candidate table size differs");
      [...comparison.age_candidates[direction], ...comparison.pair_candidates[direction]].forEach(row => { if (Object.keys(row.predictions).length !== windows.length) throw new Error("Candidate prediction window axis differs"); });
    });
    Object.values(comparison.pair_displays).forEach(display => {
      if (display.distance_class.length !== display.display_count) throw new Error("Pair display distance axis differs");
      Object.values(display.targets).forEach(values => { if (values.length !== display.display_count) throw new Error("Pair display target axes differ"); });
      if (Object.keys(display.predictions).length !== windows.length) throw new Error("Pair display prediction window axis differs");
      Object.values(display.predictions).forEach(components => Object.values(components).forEach(values => { if (values.length !== display.display_count) throw new Error("Pair display prediction axes differ"); }));
    });
  });
  if (provenance.schema !== provenanceSchemas[sourceSchema]) throw new Error("Interpretation provenance schema differs");
  if (provenance.producer_git_commit !== data.identity.code_git_commit) throw new Error("Producer identity differs");
  if (provenance.config_sha256 !== data.identity.analysis_config_sha256) throw new Error("Analysis config identity differs");
};

const table = (headers, rows) => {
  const result = element("table"), head = element("thead"), header = element("tr");
  headers.forEach(value => header.append(element("th", value))); head.append(header); result.append(head);
  const body = element("tbody");
  rows.forEach(fields => { const row = element("tr"); fields.forEach(value => row.append(element("td", String(value)))); body.append(row); });
  result.append(body); return result;
};
const replaceTable = (selector, headers, rows) => document.querySelector(selector).replaceWith(Object.assign(table(headers, rows), {id: selector.slice(1)}));
const addOptions = (select, entries) => entries.forEach(([value, label]) => { const option = element("option", label); option.value = value; select.append(option); });
const addCellOptions = (select, cells) => addOptions(select, cells.map((cell, index) => [String(index), cellLabel(cell)]));
const replaceMetricStrip = (selector, entries) => {
  const node = document.querySelector(selector); node.replaceChildren();
  entries.forEach(([name, value]) => node.append(element("span", `${name}: ${value}`)));
};
const summaryCard = (title, value, detail) => {
  const card = element("div"); card.className = "summary-card";
  card.append(element("span", title), element("strong", value), element("small", detail)); return card;
};
const finding = (title, body, caution = false) => {
  const node = element("article"); node.className = caution ? "finding caution" : "finding";
  node.append(element("strong", title), element("p", body)); return node;
};

const chartFrame = (canvas, {minimum, maximum, xLabels, yLabel, bottom = 105}) => {
  const context = canvas.getContext("2d"), width = canvas.width, height = canvas.height;
  const margins = {left: 78, right: 34, top: 40, bottom};
  context.clearRect(0, 0, width, height); context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);
  const y = value => height - margins.bottom - (Number(value) - minimum) / (maximum - minimum) * (height - margins.top - margins.bottom);
  const x = index => margins.left + (index + .5) / xLabels.length * (width - margins.left - margins.right);
  context.strokeStyle = "#d8e0db"; context.fillStyle = "#516159"; context.font = "13px sans-serif";
  for (let index = 0; index <= 5; index += 1) {
    const value = minimum + index / 5 * (maximum - minimum), py = y(value);
    context.beginPath(); context.moveTo(margins.left, py); context.lineTo(width - margins.right, py); context.stroke();
    context.fillText(value.toFixed(2), 18, py + 4);
  }
  context.textAlign = "center";
  xLabels.forEach((label, index) => { context.save(); context.translate(x(index), height - margins.bottom + 17); context.rotate(-Math.PI / 5); context.fillText(label, 0, 0); context.restore(); });
  context.save(); context.translate(18, height / 2); context.rotate(-Math.PI / 2); context.font = "15px sans-serif"; context.fillText(yLabel, 0, 0); context.restore();
  context.textAlign = "left"; return {context, x, y, width, height, margins};
};

const drawScatter = (canvas, xs, ys, colors, {xLabel, yLabel}) => {
  if (xs.length !== ys.length || xs.length !== colors.length || xs.length < 2) throw new Error("Scatter axes differ or are empty");
  const context = canvas.getContext("2d"), width = canvas.width, height = canvas.height, margin = 78;
  context.clearRect(0, 0, width, height); context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);
  const xValues = xs.map(Number), yValues = ys.map(Number), rawMin = Math.min(...xValues, ...yValues), rawMax = Math.max(...xValues, ...yValues);
  const pad = Math.max((rawMax - rawMin) * .08, .01), minimum = rawMin - pad, maximum = rawMax + pad;
  const x = value => margin + (Number(value) - minimum) / (maximum - minimum) * (width - 2 * margin);
  const y = value => height - margin - (Number(value) - minimum) / (maximum - minimum) * (height - 2 * margin);
  context.strokeStyle = "#d8e0db"; context.lineWidth = 1; context.font = "13px sans-serif";
  for (let index = 0; index <= 4; index += 1) {
    const value = minimum + index / 4 * (maximum - minimum);
    context.beginPath(); context.moveTo(x(value), margin); context.lineTo(x(value), height - margin); context.stroke();
    context.beginPath(); context.moveTo(margin, y(value)); context.lineTo(width - margin, y(value)); context.stroke();
    context.fillStyle = "#516159"; context.fillText(value.toFixed(2), x(value) - 17, height - margin + 23); context.fillText(value.toFixed(2), 15, y(value) + 4);
  }
  context.strokeStyle = "#707873"; context.lineWidth = 2; context.beginPath(); context.moveTo(x(minimum), y(minimum)); context.lineTo(x(maximum), y(maximum)); context.stroke();
  xs.forEach((value, index) => { context.fillStyle = colors[index]; context.globalAlpha = .38; context.beginPath(); context.arc(x(value), y(ys[index]), 2.7, 0, 2 * Math.PI); context.fill(); });
  context.globalAlpha = 1; context.fillStyle = "#17221d"; context.font = "16px sans-serif"; context.textAlign = "center"; context.fillText(xLabel, width / 2, height - 17);
  context.save(); context.translate(21, height / 2); context.rotate(-Math.PI / 2); context.fillText(yLabel, 0, 0); context.restore(); context.textAlign = "left";
};

const renderSummary = data => {
  const diverse = data.reliability["diverse-blocks"], chromosome = data.reliability["held-out-chromosome"];
  const surrogateR2 = data.cells.map(cell => cell.surrogates.cpg_gc_plus_annotations.model_output.r_squared);
  const residualCorrelations = data.cells.flatMap(cell => Object.values(cell.pair_decomposition).map(population => population.overall.age_adjusted_residual.pearson));
  const pairwise = data.window_comparisons.flatMap(comparison => comparison.pairwise_window_comparisons);
  const metricCosines = pairwise.map(row => row.metric_cosine), ckaValues = pairwise.map(row => row.linear_cka);
  const bestCell = data.cells.reduce((best, cell) => cell.pair_decomposition.held_out_stratified.overall.age_adjusted_residual.pearson > best.pair_decomposition.held_out_stratified.overall.age_adjusted_residual.pearson ? cell : best);
  const node = document.querySelector("#findings");
  node.append(
    finding("The targets are not especially noisy.", `Probe–age split-half reliability is ${fmt(diverse.probe_age.mean)}–${fmt(chromosome.probe_age.mean)}; total pair reliability is ${fmt(diverse.pair_overall.total.mean)}–${fmt(chromosome.pair_overall.total.mean)}. The model's much lower correlations therefore reflect missing predictive structure or transport limits, not merely unstable labels.`),
    finding("Age helps, but is not the whole pair result.", `After subtracting the age rank-one component exactly from both empirical and predicted Gram values, residual correlations span ${fmt(Math.min(...residualCorrelations))}–${fmt(Math.max(...residualCorrelations))} across completed cells and populations.`),
    finding("Window stability is rotation-invariant.", `Across all registered window pairs, metric cosine spans ${fmt(Math.min(...metricCosines))}–${fmt(Math.max(...metricCosines))} and linear CKA spans ${fmt(Math.min(...ckaValues))}–${fmt(Math.max(...ckaValues))}. Named latent coordinates are never compared.`),
    finding("The age lobes are mostly broad locus classes.", `CpG/GC plus manifest annotations explain ${pct(Math.min(...surrogateR2))}–${pct(Math.max(...surrogateR2))} of held-out model-output variance. CpG context and Infinium chemistry are prominent; manifest strand is not.`, true),
    finding("Use rankings as directional screens, not quantitative estimates.", "Candidate ranks require a common sign across every completed window and use the smallest absolute prediction. Empirical targets are revealed only after that prediction-only order is frozen.", true),
  );
  const headline = document.querySelector("#headline");
  headline.append(
    summaryCard("Age target reliability", fmt(chromosome.probe_age.mean), "Strict chromosome split · split-half r"),
    summaryCard("Residual pair reliability", fmt(chromosome.pair_overall.age_adjusted_residual.mean), "Strict chromosome split · split-half r"),
    summaryCard("Best held-out residual r", fmt(bestCell.pair_decomposition.held_out_stratified.overall.age_adjusted_residual.pearson), cellLabel(bestCell)),
    summaryCard("Manifest probes joined", data.audit.manifest_matched_probes.toLocaleString(), "Exact GPL13534 build-37 identity"),
  );
};

const renderReliability = data => {
  const categories = ["Probe–age", "Pair total", "Pair age component", "Pair residual"];
  const keys = [["probe_age"], ["pair_overall", "total"], ["pair_overall", "age_component"], ["pair_overall", "age_adjusted_residual"]];
  const lookup = (split, path) => path.reduce((value, key) => value[key], data.reliability[split]);
  const frame = chartFrame(document.querySelector("#reliability-chart"), {minimum: .84, maximum: 1, xLabels: categories, yLabel: "Split-half Pearson r"});
  ["diverse-blocks", "held-out-chromosome"].forEach((split, splitIndex) => {
    keys.forEach((path, index) => {
      const value = lookup(split, path).mean, groupWidth = 86, width = 31, px = frame.x(index) - groupWidth / 2 + splitIndex * 45;
      frame.context.fillStyle = splitIndex === 0 ? "#006b57" : "#d37628";
      frame.context.fillRect(px, frame.y(value), width, frame.y(.84) - frame.y(value));
    });
    frame.context.fillStyle = splitIndex === 0 ? "#006b57" : "#d37628"; frame.context.fillRect(frame.margins.left + splitIndex * 260, 18, 18, 10);
    frame.context.fillStyle = "#17221d"; frame.context.fillText(splitLabels[split], frame.margins.left + 24 + splitIndex * 260, 28);
  });
  const rows = [];
  Object.keys(splitLabels).forEach(split => keys.forEach((path, index) => {
    const summary = lookup(split, path); rows.push([splitLabels[split], categories[index], fmt(summary.mean), fmt(summary.standard_deviation), fmt(summary.spearman_brown_from_mean), fmt(summary.descriptive_noise_ceiling)]);
  }));
  replaceTable("#reliability-table", ["Split", "Target", "half r", "SD", "Spearman–Brown", "√SB ceiling"], rows);
};

const renderDecomposition = data => {
  const cellSelect = document.querySelector("#decomposition-cell"), populationSelect = document.querySelector("#decomposition-population");
  addCellOptions(cellSelect, data.cells); addOptions(populationSelect, Object.entries(populationLabels));
  const render = () => {
    const cell = data.cells[Number(cellSelect.value)], population = cell.pair_decomposition[populationSelect.value], overall = population.overall;
    replaceMetricStrip("#decomposition-headline", [
      ["total Pearson", fmt(overall.total.pearson)], ["age-component Pearson", fmt(overall.age_component.pearson)], ["residual Pearson", fmt(overall.age_adjusted_residual.pearson)],
      ["empirical age vs total", fmt(overall.empirical_age_component_against_total.pearson)], ["pairs", overall.total.count.toLocaleString()],
    ]);
    const labels = Object.keys(population.by_distance), values = labels.flatMap(label => Object.keys(componentLabels).map(component => population.by_distance[label][component].pearson));
    const minimum = Math.min(-.05, ...values) - .025, maximum = Math.max(...values) + .04;
    const frame = chartFrame(document.querySelector("#decomposition-chart"), {minimum, maximum, xLabels: labels.map(label => distanceLabels[label]), yLabel: "Prediction Pearson r"});
    Object.keys(componentLabels).forEach((component, seriesIndex) => {
      const points = labels.map(label => population.by_distance[label][component].pearson), color = componentColors[component];
      frame.context.strokeStyle = color; frame.context.fillStyle = color; frame.context.lineWidth = 2.5; frame.context.beginPath();
      points.forEach((value, index) => { if (index === 0) frame.context.moveTo(frame.x(index), frame.y(value)); else frame.context.lineTo(frame.x(index), frame.y(value)); }); frame.context.stroke();
      points.forEach((value, index) => { frame.context.beginPath(); frame.context.arc(frame.x(index), frame.y(value), 4.5, 0, 2 * Math.PI); frame.context.fill(); });
      frame.context.fillRect(frame.margins.left + seriesIndex * 205, 18, 16, 9); frame.context.fillStyle = "#17221d"; frame.context.fillText(componentShortLabels[component], frame.margins.left + 22 + seriesIndex * 205, 27);
    });
    const rows = labels.map(label => { const row = population.by_distance[label]; return [distanceLabels[label], row.total.count.toLocaleString(), fmt(row.total.pearson), fmt(row.age_component.pearson), fmt(row.age_adjusted_residual.pearson), fmt(row.total.mse), fmt(row.age_adjusted_residual.mse)]; });
    replaceTable("#decomposition-table", ["Distance", "count", "total r", "age r", "residual r", "total MSE", "residual MSE"], rows);
  };
  cellSelect.addEventListener("change", render); populationSelect.addEventListener("change", render); render();
};

const renderPairScatter = data => {
  const splitSelect = document.querySelector("#pair-split"), populationSelect = document.querySelector("#pair-population"), componentSelect = document.querySelector("#pair-component"), windowSelect = document.querySelector("#pair-window");
  const windows = data.window_comparisons[0].window_sizes;
  addOptions(splitSelect, Object.entries(splitLabels)); addOptions(populationSelect, Object.entries(populationLabels)); addOptions(componentSelect, Object.entries(componentLabels)); addOptions(windowSelect, windows.map(window => [String(window), windowLabel(window)]));
  const render = () => {
    const comparison = data.window_comparisons.find(row => row.split_name === splitSelect.value), display = comparison.pair_displays[populationSelect.value];
    const component = componentSelect.value;
    const target = display.targets[component], prediction = display.predictions[windowSelect.value][component], colors = display.distance_class.map(value => distanceColors[Number(value)]);
    drawScatter(document.querySelector("#pair-scatter"), target, prediction, colors, {xLabel: `Empirical ${componentShortLabels[component]}`, yLabel: `Predicted ${componentShortLabels[component]}`});
    const cell = data.cells.find(row => row.split_name === splitSelect.value && row.window_size === Number(windowSelect.value)), metric = cell.pair_decomposition[populationSelect.value].overall[component];
    replaceMetricStrip("#pair-scatter-metrics", [["Pearson", fmt(metric.pearson)], ["MSE", fmt(metric.mse)], ["R²", fmt(metric.r_squared)], ["display pairs", display.display_count.toLocaleString()], ["source pairs", display.source_count.toLocaleString()]]);
  };
  [splitSelect, populationSelect, componentSelect, windowSelect].forEach(select => select.addEventListener("change", render)); render();
};

const annotationColor = (row, mode) => {
  if (mode === "context") return contextColors[row.context];
  if (mode === "design") return designColors[row.design];
  if (mode === "regulatory") return booleanColors[String(Boolean(row.regulatory_feature_group))];
  if (mode === "enhancer") return booleanColors[String(row.enhancer)];
  if (mode === "dhs") return booleanColors[String(row.dhs)];
  return chromosomePalette[(Number(row.chromosome) - 1) % chromosomePalette.length];
};
const renderAgeAudit = data => {
  const cellSelect = document.querySelector("#age-cell"), colorSelect = document.querySelector("#age-color"); addCellOptions(cellSelect, data.cells);
  addOptions(colorSelect, [["context", "Genomic context"], ["design", "Infinium chemistry"], ["regulatory", "Regulatory annotation"], ["enhancer", "Enhancer"], ["dhs", "DHS"], ["chromosome", "Chromosome"]]);
  const render = () => {
    const cell = data.cells[Number(cellSelect.value)], comparison = data.window_comparisons.find(row => row.split_name === cell.split_name), rows = comparison.age_display;
    drawScatter(document.querySelector("#age-scatter"), rows.map(row => row.empirical_rho), rows.map(row => row.predictions[String(cell.window_size)]), rows.map(row => annotationColor(row, colorSelect.value)), {xLabel: "Empirical probe–age correlation", yLabel: `Predicted correlation · ${windowLabel(cell.window_size)}`});
    replaceMetricStrip("#age-scatter-metrics", [["Pearson", fmt(cell.age.pearson)], ["MSE", fmt(cell.age.mse)], ["R²", fmt(cell.age.r_squared)], ["display probes", rows.length.toLocaleString()], ["held-out probes", cell.held_out_probe_count.toLocaleString()]]);
  };
  cellSelect.addEventListener("change", render); colorSelect.addEventListener("change", render); render();
  const rows = data.cells.flatMap(cell => Object.entries(cell.surrogates).flatMap(([name, record]) => [[cellLabel(cell), name === "cpg_gc" ? "CpG density + GC" : "CpG/GC + manifest annotations", record.feature_count, fmt(record.model_output.pearson), fmt(record.model_output.r_squared), fmt(record.empirical_rho.pearson), fmt(record.empirical_rho.r_squared)]]));
  replaceTable("#surrogate-table", ["Split/window", "Training-only surrogate", "features", "model-output r", "model-output R²", "empirical-ρ r", "empirical-ρ R²"], rows);
};

const renderGeometry = data => {
  replaceTable("#geometry-table", ["Split/window", "d", "metric PR", "latent PR", "pre-norm PR", "90% latent dims", "final/init M cos", "final/init age cos", "ΔW/‖W₀‖"], data.cells.map(cell => [
    cellLabel(cell), cell.parent.latent_dimension, fmt(cell.weights.metric_spectrum.participation_rank), fmt(cell.representation.normalized_latent_covariance.participation_rank), fmt(cell.representation.pre_normalization_covariance.participation_rank), cell.representation.normalized_latent_covariance.dimensions_for_90_percent_trace, fmt(cell.weights.final_initial_metric_cosine), fmt(cell.weights.final_initial_age_pullback_cosine), fmt(cell.weights.relative_weight_change),
  ]));
  const comparisons = data.window_comparisons.flatMap(comparison => comparison.pairwise_window_comparisons.map(pair => ({...pair, split_name: comparison.split_name})));
  replaceTable("#stability-table", ["Split/window pair", "M cosine", "age-direction cosine", "age prediction r", "age sign", "linear CKA", "top-25 overlap", "HH total r", "HH residual r"], comparisons.map(row => [
    `${splitLabels[row.split_name]} · ${windowLabel(row.first_window_size)}–${windowLabel(row.second_window_size)}`, fmt(row.metric_cosine), fmt(row.age_pullback_cosine), fmt(row.age_prediction_pearson), pct(row.age_prediction_sign_agreement), fmt(row.linear_cka), pct(row.top_neighbour_overlap), fmt(row.pair_prediction_agreement.held_out_stratified.total_prediction_pearson), fmt(row.pair_prediction_agreement.held_out_stratified.residual_prediction_pearson),
  ]));
};

const renderRanks = data => {
  const splitSelect = document.querySelector("#rank-split"), targetSelect = document.querySelector("#rank-target"), directionSelect = document.querySelector("#rank-direction");
  addOptions(splitSelect, Object.entries(splitLabels)); addOptions(targetSelect, [["age", "Probe–age"], ["pair", "Cross-probe pair"]]); addOptions(directionSelect, [["positive", "Positive"], ["negative", "Negative"]]);
  const render = () => {
    const comparison = data.window_comparisons.find(row => row.split_name === splitSelect.value), curve = comparison[`${targetSelect.value}_rank_curves`][directionSelect.value], labels = curve.map(row => `top ${row.rank}`), values = curve.flatMap(row => [row.empirical_sign_agreement, row.population_empirical_sign_rate]);
    const minimum = Math.max(0, Math.min(...values) - .08), maximum = Math.min(1, Math.max(...values) + .08), frame = chartFrame(document.querySelector("#rank-chart"), {minimum, maximum, xLabels: labels, yLabel: "Empirical sign fraction"});
    [["stable prediction rank", curve.map(row => row.empirical_sign_agreement), "#006b57"], ["population base rate", curve.map(row => row.population_empirical_sign_rate), "#d37628"]].forEach(([name, points, color], seriesIndex) => {
      frame.context.strokeStyle = color; frame.context.fillStyle = color; frame.context.lineWidth = 2.5; frame.context.beginPath(); points.forEach((value, index) => { if (index === 0) frame.context.moveTo(frame.x(index), frame.y(value)); else frame.context.lineTo(frame.x(index), frame.y(value)); }); frame.context.stroke(); points.forEach((value, index) => { frame.context.beginPath(); frame.context.arc(frame.x(index), frame.y(value), 4.5, 0, 2 * Math.PI); frame.context.fill(); }); frame.context.fillRect(frame.margins.left + seriesIndex * 250, 18, 16, 9); frame.context.fillStyle = "#17221d"; frame.context.fillText(name, frame.margins.left + 22 + seriesIndex * 250, 27);
    });
    replaceTable("#rank-table", ["Rank", "sign precision", "base sign rate", "empirical mean", "mean |empirical|", "prediction mean", "within-rank magnitude r"], curve.map(row => [row.rank, pct(row.empirical_sign_agreement), pct(row.population_empirical_sign_rate), fmt(row.empirical_mean), fmt(row.empirical_mean_absolute), fmt(row.prediction_mean), fmt(row.prediction_empirical_pearson)]));
  };
  [splitSelect, targetSelect, directionSelect].forEach(select => select.addEventListener("change", render)); render();
};

const genes = locus => unique(locus.refgene_names).join(", ") || "—";
const locusLabel = locus => `${locus.probe_id} · chr${locus.chromosome}:${Number(locus.position).toLocaleString()}`;
const renderCandidates = data => {
  const splitSelect = document.querySelector("#candidate-split"), targetSelect = document.querySelector("#candidate-target"), directionSelect = document.querySelector("#candidate-direction");
  const windows = data.window_comparisons[0].window_sizes, windowHeaders = windows.map(windowLabel), stableWindows = windowHeaders.join("/");
  addOptions(splitSelect, Object.entries(splitLabels)); addOptions(targetSelect, [["age", "Probe–age"], ["pair", "Cross-probe pair"]]); addOptions(directionSelect, [["positive", "Positive"], ["negative", "Negative"]]);
  const render = () => {
    const comparison = data.window_comparisons.find(row => row.split_name === splitSelect.value), target = targetSelect.value, rows = comparison[`${target}_candidates`][directionSelect.value], container = document.querySelector("#candidate-table");
    container.replaceChildren();
    document.querySelector("#candidate-note").textContent = target === "age" ? `Ranked by the smallest absolute ${stableWindows} age prediction after requiring one common sign across all windows. Empirical rho was not read until the list was frozen.` : `Ranked by the smallest absolute ${stableWindows} cosine after requiring one common sign across all windows. The frozen pair class is ${distanceLabels[comparison.pair_candidate_distance_class]}; empirical covariance was not read until ranking was frozen.`;
    if (target === "age") {
      container.append(table(["Probe/locus", "genes", "context", "design", ...windowHeaders, "empirical ρ", "sign correct"], rows.map(row => [locusLabel(row), genes(row), row.context, row.design, ...windows.map(window => fmt(row.predictions[String(window)])), fmt(row.empirical_rho), Math.sign(row.predictions[String(windows[0])]) === Math.sign(row.empirical_rho) ? "yes" : "no"])));
    } else {
      container.append(table(["Left locus", "left genes", "Right locus", "right genes", ...windowHeaders, "empirical Y", "empirical residual", "sign correct"], rows.map(row => [locusLabel(row.left), genes(row.left), locusLabel(row.right), genes(row.right), ...windows.map(window => fmt(row.predictions[String(window)])), fmt(row.empirical_correlation), fmt(row.empirical_age_adjusted_residual), Math.sign(row.predictions[String(windows[0])]) === Math.sign(row.empirical_correlation) ? "yes" : "no"])));
    }
  };
  [splitSelect, targetSelect, directionSelect].forEach(select => select.addEventListener("change", render)); render();
};

Promise.all([
  fetch("latent-interpretation-data.json", {cache: "no-store"}),
  fetch("latent-interpretation-provenance.json", {cache: "no-store"}),
]).then(async responses => {
  responses.forEach(response => { if (!response.ok) throw new Error(`Interpretation artifact request failed: ${response.status}`); });
  return Promise.all(responses.map(response => response.json()));
}).then(([rawData, provenance]) => {
  const sourceSchema = rawData.schema, data = normalizeData(rawData);
  assertData(data, provenance, sourceSchema);
  document.querySelector("#status").textContent = `Complete: ${data.reliability["diverse-blocks"].subject_split_count} subject split-halves, exact age/residual decomposition, rotation-invariant geometry, annotation surrogates, and prediction-only rankings.`;
  document.querySelector("#protocol").textContent = `${data.identity.protocol_id} · artifacts ${data.identity.code_git_commit.slice(0, 12)}`;
  const provenanceList = document.querySelector("#provenance");
  [
    `Analysis config SHA-256: ${provenance.config_sha256}`,
    `Results SHA-256: ${provenance.analysis_results_sha256}`,
    `Analysis manifest SHA-256: ${provenance.analysis_manifest_sha256}`,
    `Producer Git commit: ${provenance.producer_git_commit}`,
    `Site compiler Git commit: ${provenance.compiler_git_commit}`,
    `Primary target SHA-256: ${data.identity.target_sha256}`,
    `GPL13534 SHA-256: ${data.identity.manifest_sha256}`,
    "Runtime: CPU only, one Torch thread, no model parameters changed",
    "Candidate selection empirical-target access: false",
  ].forEach(value => provenanceList.append(element("li", value)));
  renderSummary(data); renderReliability(data); renderDecomposition(data); renderPairScatter(data); renderAgeAudit(data); renderGeometry(data); renderRanks(data); renderCandidates(data);
}).catch(error => {
  document.querySelector("#status").textContent = `Report failed loudly: ${error.message}`; throw error;
});
