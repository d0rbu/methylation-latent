const labels = {
  free: "Free vectors (transductive upper bound)",
  shared_tss: "TSS Caduceus → frozen probe W",
  tss_linear: "TSS Caduceus → learned linear map",
  amino_acid_linear: "ESM-2 amino acid → learned linear map",
  combined_linear: "TSS + amino acid → learned linear map",
  "diverse-blocks": "Diverse blocks",
  "held-out-chromosome": "Held-out chr7",
};

const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "—";
const experimentLabel = record => `${labels[record.split]} · ${record.window_size / 1024} kb · ${labels[record.representation]}`;
const strongest = record => record.evaluation.cross_populations.heldout_cpg_heldout_protein.all_overlap_safe;

const assertData = data => {
  if (data.schema !== "methylation-latent.protein-extension-results.v1") throw new Error("Protein result schema differs");
  if (!Array.isArray(data.records) || data.records.length !== 20) throw new Error("Expected exactly 20 protein experiment records");
  if (!Array.isArray(data.protein_names) || data.protein_names.length !== 52) throw new Error("Protein axis differs");
  data.records.forEach(record => {
    if (!strongest(record).defined || strongest(record).count < 2) throw new Error("Strongest evaluation metric is undefined");
  });
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

const renderHeadline = data => {
  const inductive = data.records.filter(record => record.evaluation.protein_generalization_valid);
  const best = [...inductive].sort((a, b) => strongest(a).mse - strongest(b).mse)[0];
  const bestCorrelation = [...inductive].sort((a, b) => strongest(b).pearson - strongest(a).pearson)[0];
  const free = [...data.records.filter(record => record.representation === "free")]
    .sort((a, b) => strongest(a).mse - strongest(b).mse)[0];
  const node = document.querySelector("#headline");
  node.append(
    summaryCard("Best inductive held-out MSE", fmt(strongest(best).mse), experimentLabel(best)),
    summaryCard("Best inductive held-out Pearson", fmt(strongest(bestCorrelation).pearson), experimentLabel(bestCorrelation)),
    summaryCard("Free-vector upper-bound MSE", fmt(strongest(free).mse), experimentLabel(free)),
    summaryCard("Largest |direct protein–age ρ|", fmt(best.evaluation.age_edge_summary.maximum_absolute_direct_residualized_protein_age_correlation), "Expected to be small after source-study age adjustment"),
  );
};

const renderMetricsTable = data => {
  const table = document.querySelector("#metrics-table");
  const head = element("thead"); const header = element("tr");
  ["Split/window", "Representation", "d", "step", "HH MSE", "HH Pearson", "HH R²", "held-out CpG × seen protein MSE", "protein HH Pearson"]
    .forEach(value => header.append(element("th", value)));
  head.append(header); table.append(head);
  const body = element("tbody");
  data.records.forEach(record => {
    const row = element("tr");
    if (!record.evaluation.protein_generalization_valid) row.className = "invalid-generalization";
    const strong = strongest(record);
    const heldSeen = record.evaluation.cross_populations.heldout_cpg_seen_protein.all_overlap_safe;
    const proteinHH = record.evaluation.protein_pair_metrics.heldout_heldout_off_diagonal;
    [
      `${labels[record.split]} · ${record.window_size / 1024} kb`, labels[record.representation],
      record.latent_dimension, record.selected_step, fmt(strong.mse), fmt(strong.pearson),
      fmt(strong.r_squared), fmt(heldSeen.mse), fmt(proteinHH.pearson),
    ].forEach(value => row.append(element("td", String(value))));
    body.append(row);
  });
  table.append(body);
};

const drawScatter = (canvas, xs, ys, {xLabel, yLabel, labels: pointLabels = null}) => {
  if (xs.length !== ys.length || xs.length < 2) throw new Error("Scatter axes differ or are empty");
  const context = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height, margin = 72;
  context.clearRect(0, 0, width, height); context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);
  const values = [...xs, ...ys].map(Number); const rawMin = Math.min(...values), rawMax = Math.max(...values);
  const pad = Math.max((rawMax - rawMin) * .08, .01); const minimum = rawMin - pad, maximum = rawMax + pad;
  const x = value => margin + (Number(value) - minimum) / (maximum - minimum) * (width - 2 * margin);
  const y = value => height - margin - (Number(value) - minimum) / (maximum - minimum) * (height - 2 * margin);
  context.strokeStyle = "#d8e0db"; context.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const value = minimum + index / 4 * (maximum - minimum);
    context.beginPath(); context.moveTo(x(value), margin); context.lineTo(x(value), height - margin); context.stroke();
    context.beginPath(); context.moveTo(margin, y(value)); context.lineTo(width - margin, y(value)); context.stroke();
    context.fillStyle = "#516159"; context.font = "13px sans-serif";
    context.fillText(value.toFixed(2), x(value) - 17, height - margin + 23);
    context.fillText(value.toFixed(2), 12, y(value) + 4);
  }
  context.strokeStyle = "#6d756f"; context.lineWidth = 2; context.beginPath();
  context.moveTo(x(minimum), y(minimum)); context.lineTo(x(maximum), y(maximum)); context.stroke();
  context.fillStyle = "rgba(0,107,87,.35)";
  xs.forEach((value, index) => { context.beginPath(); context.arc(x(value), y(ys[index]), pointLabels ? 5 : 2.2, 0, 2 * Math.PI); context.fill(); });
  context.fillStyle = "#17221d"; context.font = "16px sans-serif"; context.textAlign = "center";
  context.fillText(xLabel, width / 2, height - 16);
  context.save(); context.translate(20, height / 2); context.rotate(-Math.PI / 2); context.fillText(yLabel, 0, 0); context.restore();
  context.textAlign = "left";
  if (pointLabels) pointLabels.forEach((label, index) => context.fillText(label, x(xs[index]) + 6, y(ys[index]) - 5));
};

const addRecordOptions = (select, records) => records.forEach((record, index) => {
  const option = element("option", experimentLabel(record)); option.value = String(index); select.append(option);
});

const renderCorrelationScatter = data => {
  const select = document.querySelector("#scatter-select"); addRecordOptions(select, data.records);
  const render = () => {
    const record = data.records[Number(select.value)]; const scatter = record.evaluation.scatter_heldout_cpg_heldout_protein;
    drawScatter(document.querySelector("#correlation-scatter"), scatter.empirical, scatter.predicted,
      {xLabel: "Empirical CpG–protein correlation", yLabel: "Predicted cosine"});
    const metric = strongest(record); const node = document.querySelector("#scatter-metrics"); node.replaceChildren();
    [["MSE", metric.mse], ["Pearson", metric.pearson], ["R²", metric.r_squared], ["overlap-safe pairs", metric.count], ["display sample", scatter.sample_count]]
      .forEach(([name, value]) => node.append(element("span", `${name}: ${typeof value === "number" && name !== "overlap-safe pairs" && name !== "display sample" ? fmt(value) : value}`)));
  };
  select.addEventListener("change", render); select.value = "1"; render();
};

const renderStrata = (data, recordIndex) => {
  const record = data.records[recordIndex]; const populations = record.evaluation.cross_populations;
  const container = document.querySelector("#strata-table"); container.replaceChildren();
  const table = element("table"); const head = element("tr");
  ["Population", "Stratum", "count", "MSE", "Pearson", "R²"].forEach(value => head.append(element("th", value)));
  const thead = element("thead"); thead.append(head); table.append(thead); const body = element("tbody");
  Object.entries(populations).forEach(([population, strata]) => Object.entries(strata).forEach(([stratum, metric]) => {
    const row = element("tr"); [population.replaceAll("_", " "), stratum.replaceAll("_", " "), metric.count, fmt(metric.mse), fmt(metric.pearson), fmt(metric.r_squared)]
      .forEach(value => row.append(element("td", String(value)))); body.append(row);
  }));
  table.append(body); container.append(element("p", experimentLabel(record)), table);
};

const renderAge = data => {
  const select = document.querySelector("#age-select"); addRecordOptions(select, data.records);
  const render = () => {
    const record = data.records[Number(select.value)]; const points = record.evaluation.per_heldout_protein;
    const names = points.map(point => data.protein_names[point.protein_index]);
    const cosine = points.map(point => point.model_age_cosine);
    drawScatter(document.querySelector("#direct-age-scatter"), points.map(point => point.direct_residualized_protein_age_correlation), cosine,
      {xLabel: "Direct residualized protein–age ρ", yLabel: "Model cosine to age", labels: names});
    drawScatter(document.querySelector("#profile-age-scatter"), points.map(point => point.empirical_methylation_profile_age_concordance), cosine,
      {xLabel: "CpG-profile concordance with age", yLabel: "Model cosine to age", labels: names});
    const table = document.querySelector("#protein-table"); table.replaceChildren(); const head = element("tr");
    ["Held-out protein", "CpG–protein MSE", "Pearson", "age cosine", "direct residualized age ρ", "profile-age concordance"]
      .forEach(value => head.append(element("th", value))); const thead = element("thead"); thead.append(head); table.append(thead);
    const body = element("tbody"); points.forEach(point => {
      const row = element("tr"); [data.protein_names[point.protein_index], fmt(point.cross_metrics.mse), fmt(point.cross_metrics.pearson), fmt(point.model_age_cosine), fmt(point.direct_residualized_protein_age_correlation), fmt(point.empirical_methylation_profile_age_concordance)]
        .forEach(value => row.append(element("td", value))); body.append(row);
    }); table.append(body); renderStrata(data, Number(select.value));
  };
  select.addEventListener("change", render); select.value = "1"; render();
};

fetch("protein-extension-data.json", {cache: "no-store"}).then(response => {
  if (!response.ok) throw new Error(`Protein result request failed: ${response.status}`);
  return response.json();
}).then(data => {
  assertData(data);
  document.querySelector("#status").textContent = "Complete: 4 frozen probe geometries × 5 protein representations.";
  document.querySelector("#protocol").textContent = `${data.protocol_id} · ${data.git_commit.slice(0, 12)}`;
  document.querySelector("#selection-warning").textContent = data.selection_warning;
  document.querySelector("#adjustment-warning").textContent = `The released protein values are ${data.source_adjustment}. Direct protein–age correlations therefore answer a residualized, not unadjusted, question.`;
  document.querySelector("#loss-formula").textContent = data.loss_formula;
  document.querySelector("#lambda-pairs").textContent = data.lambda_protein_pairs;
  const provenance = document.querySelector("#provenance");
  [
    `Protocol SHA-256: ${data.protocol_sha256}`,
    `Protein target metadata SHA-256: ${data.protein_target_metadata_sha256}`,
    `Protein target tensor SHA-256: ${data.protein_target_tensor_sha256}`,
    `Git commit: ${data.git_commit}`,
    `Device recorded by run: ${data.device}`,
  ].forEach(value => provenance.append(element("li", value)));
  renderHeadline(data); renderMetricsTable(data); renderCorrelationScatter(data); renderAge(data);
}).catch(error => {
  document.querySelector("#status").textContent = `Report failed loudly: ${error.message}`;
  throw error;
});
