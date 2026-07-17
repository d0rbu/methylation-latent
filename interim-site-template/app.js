const colors = {
  seen_by_held_out: "#006b57",
  held_out_by_held_out: "#b24b16",
  sequence: "#4977a3",
  registered_distance_class: "#6d756f",
  psd_distance: "#b78105",
  psd_distance_plus_sequence: "#7a4da3",
  cosine_age_only: "#4977a3",
  direct_tanh: "#c43f52",
  full_latent_metric: "#7a4da3",
  island: "#006b57",
  shore: "#4977a3",
  shelf: "#b78105",
  open_sea: "#c43f52",
  negative_cluster: "#c43f52",
  positive_cluster: "#006b57",
  age_direction: "#17221d",
};

const distanceOrder = [
  "cis_0_1kb",
  "cis_1_4kb",
  "cis_4_16kb",
  "cis_16_64kb",
  "cis_64_256kb",
  "cis_256kb_1mb",
  "cis_1mb_plus",
  "trans",
];
const distanceLabels = ["0–1k", "1–4k", "4–16k", "16–64k", "64–256k", "256k–1m", "≥1m", "trans"];
const populationLabels = {
  seen_by_held_out: "seen × held-out",
  held_out_by_held_out: "held-out × held-out",
};
const stageLabels = {
  sequence_features: "CpG density + GC",
  caduceus_age_only: "Caduceus age-only cosine",
  full_latent_metric: "Full latent metric",
};
const modelLabels = {
  sequence: "Sequence cosine",
  registered_distance_class: "Registered distance-class mean",
  psd_distance: "PSD distance only",
  psd_distance_plus_sequence: "PSD distance + sequence",
};
const scatterModelLabels = {
  cosine_age_only: "Caduceus age-only cosine",
  direct_tanh: "Direct scalar tanh (post-hoc)",
  full_latent_metric: "Full latent metric",
};
const contextLabels = {
  island: "island",
  shore: "shore",
  shelf: "shelf",
  open_sea: "open sea",
};

const svgElement = (name, attributes = {}, text = null) => {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== null) element.textContent = text;
  return element;
};

const svgFrame = (node, label) => {
  const svg = svgElement("svg", { viewBox: "0 0 900 350", role: "img", "aria-label": label });
  node.append(svg);
  return svg;
};

const line = (svg, points, stroke, dashed = false) => {
  if (!points.length) return;
  const path = svgElement("path", {
    d: points.map((point, index) => `${index ? "L" : "M"}${point[0]},${point[1]}`).join(" "),
    fill: "none",
    stroke,
    "stroke-width": "2.5",
  });
  if (dashed) path.setAttribute("stroke-dasharray", "7 6");
  svg.append(path);
  points.forEach(([x, y, selected]) => {
    svg.append(svgElement("circle", {
      cx: x,
      cy: y,
      r: selected ? "6" : "3.5",
      fill: selected ? "#fffdf8" : stroke,
      stroke,
      "stroke-width": selected ? "3" : "0",
    }));
  });
};

const drawAxes = (svg, xLabels, yTicks, yCoordinate, yLabel) => {
  const left = 82;
  const right = 842;
  const top = 24;
  const bottom = 278;
  svg.append(svgElement("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "axis" }));
  svg.append(svgElement("line", { x1: left, x2: left, y1: top, y2: bottom, class: "axis" }));
  yTicks.forEach(value => {
    const y = yCoordinate(value);
    svg.append(svgElement("line", { x1: left, x2: right, y1: y, y2: y, class: "grid" }));
    svg.append(svgElement("text", { x: left - 10, y: y + 4, class: "label", "text-anchor": "end" }, value.toFixed(3)));
  });
  xLabels.forEach((label, index) => {
    const x = left + (right - left) * index / Math.max(1, xLabels.length - 1);
    svg.append(svgElement("text", { x, y: bottom + 22, class: "label", "text-anchor": "middle" }, label));
  });
  svg.append(svgElement(
    "text",
    { x: 18, y: 151, class: "axis-title", transform: "rotate(-90 18 151)", "text-anchor": "middle" },
    yLabel,
  ));
};

const legend = (node, entries) => {
  const container = document.createElement("div");
  container.className = "legend";
  entries.forEach(([label, color]) => {
    const item = document.createElement("span");
    item.style.color = color;
    item.textContent = `— ${label}`;
    container.append(item);
  });
  node.append(container);
};

const table = (headings, rows) => {
  const element = document.createElement("table");
  const headRow = document.createElement("tr");
  headings.forEach(label => {
    const cell = document.createElement("th");
    cell.textContent = label;
    headRow.append(cell);
  });
  const head = document.createElement("thead");
  head.append(headRow);
  const body = document.createElement("tbody");
  rows.forEach(row => {
    const tableRow = document.createElement("tr");
    row.forEach(value => {
      const cell = document.createElement("td");
      cell.textContent = value;
      tableRow.append(cell);
    });
    body.append(tableRow);
  });
  element.append(head, body);
  return element;
};

const metric = (value, digits = 4) => value === null ? "undefined (constant)" : Number(value).toFixed(digits);
const windowLabel = window => `${(window / 1024).toLocaleString()} kb`;
const article = title => {
  const element = document.createElement("article");
  const heading = document.createElement("h3");
  heading.textContent = title;
  element.append(heading);
  return element;
};

const canvasFrame = (label, size = 420) => {
  const canvas = document.createElement("canvas");
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = size * ratio;
  canvas.height = size * ratio;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", label);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  return [canvas, context, size];
};

const plotFigure = title => {
  const figure = document.createElement("figure");
  const heading = document.createElement("figcaption");
  heading.className = "plot-title";
  heading.textContent = title;
  figure.append(heading);
  return figure;
};

const plotNote = text => {
  const note = document.createElement("p");
  note.className = "plot-note";
  note.textContent = text;
  return note;
};

const drawCorrelationCanvas = (target, prediction, color, label, clusterLabels = null) => {
  if (target.length !== prediction.length || !target.length) throw new Error(`${label} scatter vectors differ`);
  if (clusterLabels !== null && clusterLabels.length !== target.length) throw new Error(`${label} cluster labels differ`);
  const [canvas, context, size] = canvasFrame(label);
  const left = 48;
  const right = size - 16;
  const top = 16;
  const bottom = size - 48;
  const coordinateX = value => left + (right - left) * (value + 1) / 2;
  const coordinateY = value => bottom - (bottom - top) * (value + 1) / 2;
  context.fillStyle = "#fffdf8";
  context.fillRect(0, 0, size, size);
  context.strokeStyle = "#d7d5ca";
  context.lineWidth = 1;
  [-1, -0.5, 0, 0.5, 1].forEach(value => {
    const x = coordinateX(value);
    const y = coordinateY(value);
    context.beginPath();
    context.moveTo(x, top);
    context.lineTo(x, bottom);
    context.moveTo(left, y);
    context.lineTo(right, y);
    context.stroke();
    context.fillStyle = "#5d6c65";
    context.font = "11px ui-sans-serif, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(value.toFixed(1), x, bottom + 17);
    context.textAlign = "right";
    context.fillText(value.toFixed(1), left - 7, y + 4);
  });
  context.strokeStyle = "#6d756f";
  context.setLineDash([6, 5]);
  context.beginPath();
  context.moveTo(coordinateX(-1), coordinateY(-1));
  context.lineTo(coordinateX(1), coordinateY(1));
  context.stroke();
  context.setLineDash([]);
  context.globalAlpha = 0.24;
  target.forEach((value, index) => {
    const x = coordinateX(value);
    const y = coordinateY(prediction[index]);
    context.fillStyle = clusterLabels === null
      ? color
      : colors[clusterLabels[index] === 0 ? "negative_cluster" : "positive_cluster"];
    context.fillRect(x - 1.15, y - 1.15, 2.3, 2.3);
  });
  context.globalAlpha = 1;
  context.strokeStyle = "#7e8983";
  context.strokeRect(left, top, right - left, bottom - top);
  context.fillStyle = "#5d6c65";
  context.font = "12px ui-sans-serif, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText("empirical correlation", (left + right) / 2, size - 9);
  context.save();
  context.translate(13, (top + bottom) / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("predicted correlation", 0, 0);
  context.restore();
  return canvas;
};

const drawDirectAge = rows => {
  document.querySelector("#direct-age-metrics").append(table(
    ["Model", "Window", "held-out probes", "selected step", "validation MSE", "test MSE", "test Pearson", "test R²"],
    [...rows].sort((a, b) => a.window_size - b.window_size).map(row => [
      "Direct scalar tanh (post-hoc)",
      windowLabel(row.window_size),
      row.count.toLocaleString(),
      row.selected_step,
      metric(row.validation_mse, 6),
      metric(row.mse, 6),
      metric(row.pearson),
      metric(row.r_squared),
    ]),
  ));
};

const drawAgeScatter = panels => {
  const node = document.querySelector("#age-scatter");
  panels.filter(panel => panel.population === "age")
    .sort((a, b) => a.window_size - b.window_size)
    .forEach(panel => {
      const card = article(`${windowLabel(panel.window_size)} · ${panel.target.length.toLocaleString()} of ${panel.source_count.toLocaleString()} held-out probes shown`);
      const grid = document.createElement("div");
      grid.className = "plot-grid";
      ["cosine_age_only", "direct_tanh", "full_latent_metric"].forEach(model => {
        const series = panel.predictions.find(candidate => candidate.model === model);
        const clusters = panel.clusters.find(candidate => candidate.model === model);
        if (!series || !clusters) throw new Error(`missing age scatter series or clusters ${model}`);
        const figure = plotFigure(scatterModelLabels[model]);
        figure.append(drawCorrelationCanvas(
          panel.target,
          series.values,
          colors[model],
          `${windowLabel(panel.window_size)} ${scatterModelLabels[model]} predicted versus empirical age correlation`,
          clusters.labels,
        ));
        grid.append(figure);
      });
      card.append(grid);
      legend(card, [
        ["k=2 lower empirical-age lobe", colors.negative_cluster],
        ["k=2 higher empirical-age lobe", colors.positive_cluster],
      ]);
      card.append(plotNote("Dashed diagonal: perfect prediction. Axes fixed to [−1, 1]. Cluster colors come from the full frozen-test audit, not only the displayed sample."));
      node.append(card);
    });
};

const drawPairScatter = (panels, primaryRows) => {
  const node = document.querySelector("#pair-scatter");
  panels.filter(panel => panel.population !== "age")
    .sort((a, b) => a.window_size - b.window_size || a.population.localeCompare(b.population))
    .forEach(panel => {
      const series = panel.predictions.find(candidate => candidate.model === "full_latent_metric");
      const metrics = primaryRows.find(row => row.window_size === panel.window_size && row.population === panel.population);
      if (!series || !metrics) throw new Error("pair scatter is missing its prediction series or primary metrics");
      const card = article(`${windowLabel(panel.window_size)} · ${populationLabels[panel.population]}`);
      const figure = plotFigure(`${panel.target.length.toLocaleString()} of ${panel.source_count.toLocaleString()} frozen pairs shown`);
      figure.className = "single-plot";
      figure.append(drawCorrelationCanvas(panel.target, series.values, colors[panel.population], `${windowLabel(panel.window_size)} ${populationLabels[panel.population]} predicted versus empirical pair correlation`));
      card.append(
        figure,
        plotNote(`Full-cache metrics: MSE ${metric(metrics.mse, 6)} · Pearson ${metric(metrics.pearson)} · R² ${metric(metrics.r_squared)}.`),
      );
      node.append(card);
    });
};

const drawStar = (context, x, y, outerRadius = 9, innerRadius = 4) => {
  context.beginPath();
  Array.from({ length: 10 }, (_, index) => {
    const angle = -Math.PI / 2 + index * Math.PI / 5;
    const radius = index % 2 === 0 ? outerRadius : innerRadius;
    const px = x + Math.cos(angle) * radius;
    const py = y + Math.sin(angle) * radius;
    if (index === 0) context.moveTo(px, py); else context.lineTo(px, py);
  });
  context.closePath();
  context.fillStyle = colors.age_direction;
  context.strokeStyle = "#fffdf8";
  context.lineWidth = 2;
  context.fill();
  context.stroke();
};

const drawUmapCanvas = (points, agePoint, label) => {
  if (!points.length) throw new Error(`${label} has no UMAP points`);
  const [canvas, context, size] = canvasFrame(label, 360);
  const margin = 18;
  const allPoints = [...points, agePoint];
  const xs = allPoints.map(point => point.x);
  const ys = allPoints.map(point => point.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  if (xMin === xMax || yMin === yMax) throw new Error(`${label} has a collapsed UMAP axis`);
  const x = value => margin + (size - 2 * margin) * (value - xMin) / (xMax - xMin);
  const y = value => size - margin - (size - 2 * margin) * (value - yMin) / (yMax - yMin);
  context.fillStyle = "#fffdf8";
  context.fillRect(0, 0, size, size);
  context.strokeStyle = "#d7d5ca";
  context.strokeRect(margin, margin, size - 2 * margin, size - 2 * margin);
  points.forEach(point => {
    context.beginPath();
    context.arc(x(point.x), y(point.y), 2.8, 0, 2 * Math.PI);
    context.fillStyle = colors[point.context];
    context.globalAlpha = 0.72;
    context.fill();
  });
  context.globalAlpha = 1;
  drawStar(context, x(agePoint.x), y(agePoint.y));
  return canvas;
};

const drawLatentUmap = panels => {
  const node = document.querySelector("#latent-umap");
  const windows = [...new Set(panels.map(panel => panel.window_size))].sort((a, b) => a - b);
  windows.forEach(window => {
    const card = article(`${windowLabel(window)} · λ = 0.1 · validation loci only`);
    const grid = document.createElement("div");
    grid.className = "projection-grid";
    panels.filter(panel => panel.window_size === window)
      .sort((a, b) => a.latent_dimension - b.latent_dimension)
      .forEach(panel => {
        const figure = plotFigure(`d = ${panel.latent_dimension}`);
        figure.append(
          drawUmapCanvas(
            panel.points,
            { x: panel.age_x, y: panel.age_y },
            `${windowLabel(window)} d ${panel.latent_dimension} UMAP colored by genomic context with learned age direction`,
          ),
          plotNote(`n = ${panel.points.length} probes + age · neighbors ${panel.n_neighbors} · min_dist ${panel.min_dist} · final cross-entropy ${metric(panel.final_cross_entropy, 5)}`),
        );
        grid.append(figure);
      });
    card.append(grid);
    legend(card, Object.entries(contextLabels).map(([key, value]) => [value, colors[key]]));
    const ageLegend = document.createElement("p");
    ageLegend.className = "plot-note";
    ageLegend.textContent = "★ learned age direction (included as a point in the UMAP graph)";
    card.append(ageLegend);
    node.append(card);
  });
};

const drawAgeClusterDiagnostics = (rows, agreements, data) => {
  const node = document.querySelector("#age-cluster-diagnostics");
  const modelOrder = ["cosine_age_only", "full_latent_metric", "direct_tanh"];
  [...new Set(rows.map(row => row.window_size))].sort((a, b) => a - b).forEach(window => {
    const selected = rows
      .filter(row => row.window_size === window)
      .sort((a, b) => modelOrder.indexOf(a.model) - modelOrder.indexOf(b.model));
    const full = selected.find(row => row.model === "full_latent_metric");
    const agreement = agreements.find(row => row.window_size === window);
    if (!full || !agreement) throw new Error(`missing age-cluster diagnostics for ${window}`);
    const card = article(`${windowLabel(window)} · k=2 on all ${full.source_count.toLocaleString()} frozen-test probes`);
    card.append(table(
      ["Model", "low center empirical → predicted", "high center empirical → predicted", "low / high n", "context V", "design V", "strand V", "chromosome V", "CpG-density d", "GC d", "female / male separation d"],
      selected.map(row => [
        scatterModelLabels[row.model],
        `${metric(row.negative_empirical_center)} → ${metric(row.negative_prediction_center)}`,
        `${metric(row.positive_empirical_center)} → ${metric(row.positive_prediction_center)}`,
        `${row.negative_count.toLocaleString()} / ${row.positive_count.toLocaleString()}`,
        metric(row.context_cramer_v),
        metric(row.design_cramer_v),
        metric(row.strand_cramer_v),
        row.chromosome_cramer_v === null ? "not testable (one chromosome)" : metric(row.chromosome_cramer_v),
        metric(row.cpg_density_cohen_d),
        metric(row.gc_content_cohen_d),
        `${metric(row.female_separation_cohen_d)} / ${metric(row.male_separation_cohen_d)}`,
      ]),
    ));
    const contextHeading = document.createElement("h4");
    contextHeading.textContent = "Full latent metric: genomic-context composition";
    card.append(contextHeading, table(
      ["Context", "lower lobe n", "higher lobe n", "higher-lobe fraction"],
      full.context_counts.map(row => [
        contextLabels[row.context],
        row.negative_count.toLocaleString(),
        row.positive_count.toLocaleString(),
        metric(row.positive_count / (row.negative_count + row.positive_count), 3),
      ]),
    ));
    const chromosomeFinding = full.chromosome_status === "not_testable_single_level"
      ? "Every audited probe is on chromosome 7, yet both lobes remain, so chromosome identity cannot explain the split within this holdout."
      : `Chromosome is associated only weakly (V=${metric(full.chromosome_cramer_v)}) compared with genomic context (V=${metric(full.context_cramer_v)}).`;
    card.append(plotNote(
      `The dominant measured separator is CpG context/local CpG density: context V=${metric(full.context_cramer_v)} and CpG-density d=${metric(full.cpg_density_cohen_d)}. ${chromosomeFinding} Strand V=${metric(full.strand_cramer_v)}, arguing against a strand-convention artifact. Female and male target vectors correlate at r=${metric(full.female_male_target_pearson)}, and the lobe separation remains within each sex (d=${metric(full.female_separation_cohen_d)} female; ${metric(full.male_separation_cohen_d)} male), so sex is not the sole cause. Cosine/full cluster agreement is ${(100 * agreement.permutation_invariant_fraction).toFixed(1)}% (ARI ${metric(agreement.adjusted_rand_index)}).`,
    ));
    node.append(card);
  });
  const limitation = document.createElement("p");
  limitation.className = "empty";
  limitation.textContent = `Cell composition: ${data.cell_composition_status.replaceAll("_", " ")}. The GEO phenotype has sex but no measured or precomputed leukocyte proportions, so cell-type composition remains a plausible unresolved contributor and was not inferred circularly from the methylation targets. Cohort sex counts: ${data.female_sample_count} female, ${data.male_sample_count} male.`;
  node.append(limitation);
};

const drawTuning = (rows, dimensions, lambdas) => {
  const node = document.querySelector("#tuning-sweep");
  [...new Set(rows.map(row => row.window_size))].sort((a, b) => a - b).forEach(window => {
    const selected = rows.filter(row => row.window_size === window);
    const card = article(`${windowLabel(window)} · complete ${dimensions.length} × ${lambdas.length} grid`);
    const svg = svgFrame(card, `Nested-validation selection score across dimensions for ${windowLabel(window)}`);
    const values = selected.map(row => row.selection_score);
    const lower = Math.min(...values);
    const upper = Math.max(...values);
    const padding = Math.max((upper - lower) * 0.1, 1e-4);
    const low = lower - padding;
    const high = upper + padding;
    const x = dimension => 82 + 760 * dimensions.indexOf(dimension) / Math.max(1, dimensions.length - 1);
    const y = value => 278 - 254 * (value - low) / (high - low);
    const ticks = Array.from({ length: 5 }, (_, index) => low + (high - low) * index / 4);
    drawAxes(svg, dimensions.map(String), ticks, y, "validation pair MSE + age MSE");
    const palette = ["#006b57", "#b78105", "#7a4da3"];
    lambdas.forEach((lambdaAge, index) => {
      const series = selected
        .filter(row => row.lambda_age === lambdaAge)
        .sort((a, b) => a.latent_dimension - b.latent_dimension);
      line(svg, series.map(row => [x(row.latent_dimension), y(row.selection_score), row.selected]), palette[index]);
    });
    legend(card, lambdas.map((value, index) => [`λ = ${value}`, palette[index]]));
    card.append(table(
      ["d", "λ", "best step", "pair MSE", "age MSE", "selection score", "selected"],
      selected
        .sort((a, b) => a.latent_dimension - b.latent_dimension || a.lambda_age - b.lambda_age)
        .map(row => [
          row.latent_dimension,
          row.lambda_age,
          row.selected_step,
          metric(row.pair_mse, 6),
          metric(row.age_mse, 6),
          metric(row.selection_score, 6),
          row.selected ? "★" : "",
        ]),
    ));
    node.append(card);
  });
};

const drawWindowSweep = rows => {
  const node = document.querySelector("#window-sweep");
  const windows = [...new Set(rows.map(row => row.window_size))].sort((a, b) => a - b);
  const svg = svgFrame(node, "Frozen-test pair Pearson by completed sequence window");
  const x = window => 82 + 760 * windows.indexOf(window) / Math.max(1, windows.length - 1);
  const y = pearson => 278 - 254 * (pearson + 0.05) / 0.4;
  drawAxes(svg, windows.map(windowLabel), [-0.05, 0.05, 0.15, 0.25, 0.35], y, "frozen-test pair Pearson r");
  Object.keys(populationLabels).forEach(population => {
    const series = rows.filter(row => row.population === population).sort((a, b) => a.window_size - b.window_size);
    line(svg, series.map(row => [x(row.window_size), y(row.pearson), false]), colors[population]);
  });
  legend(node, Object.entries(populationLabels).map(([key, label]) => [label, colors[key]]));
  node.append(table(
    ["Window", "Population", "selected d", "selected λ", "MSE", "Pearson", "R²"],
    [...rows].sort((a, b) => a.window_size - b.window_size || a.population.localeCompare(b.population)).map(row => [
      windowLabel(row.window_size),
      populationLabels[row.population],
      row.latent_dimension,
      row.lambda_age,
      metric(row.mse, 6),
      metric(row.pearson),
      metric(row.r_squared),
    ]),
  ));
};

const drawAge = rows => {
  document.querySelector("#age-metrics").append(table(
    ["Stage", "Window", "held-out probes", "MSE", "corr(predicted ρ, empirical ρ)"],
    [...rows].sort((a, b) => a.window_size - b.window_size || a.stage.localeCompare(b.stage)).map(row => [
      stageLabels[row.stage],
      windowLabel(row.window_size),
      row.count.toLocaleString(),
      metric(row.mse, 6),
      metric(row.pearson),
    ]),
  ));
};

const groupedDistance = rows => [...new Set(rows.map(row => `${row.window_size}:${row.population}`))]
  .sort((a, b) => {
    const [aw, ap] = a.split(":");
    const [bw, bp] = b.split(":");
    return Number(aw) - Number(bw) || ap.localeCompare(bp);
  });

const drawPrimaryDistance = rows => {
  const node = document.querySelector("#distance-metrics");
  groupedDistance(rows).forEach(group => {
    const [windowRaw, population] = group.split(":");
    const window = Number(windowRaw);
    const selected = rows
      .filter(row => row.window_size === window && row.population === population)
      .sort((a, b) => distanceOrder.indexOf(a.distance_class) - distanceOrder.indexOf(b.distance_class));
    const card = article(`${windowLabel(window)} · ${populationLabels[population]}`);
    const svg = svgFrame(card, `Distance-binned primary predictions for ${windowLabel(window)} ${populationLabels[population]}`);
    const x = distanceClass => 82 + 760 * distanceOrder.indexOf(distanceClass) / (distanceOrder.length - 1);
    const y = correlation => 278 - 254 * (correlation + 0.1) / 0.7;
    drawAxes(svg, distanceLabels, [-0.1, 0.075, 0.25, 0.425, 0.6], y, "mean pair correlation");
    line(svg, selected.map(row => [x(row.distance_class), y(row.target_mean), false]), "#17221d");
    line(svg, selected.map(row => [x(row.distance_class), y(row.prediction_mean), false]), colors[population]);
    line(svg, selected.map(row => [x(row.distance_class), y(row.distance_baseline_mean), false]), "#6d756f", true);
    legend(card, [
      ["empirical target mean", "#17221d"],
      ["sequence-model mean", colors[population]],
      ["training-only f(|Δpos|)", "#6d756f"],
    ]);
    card.append(table(
      ["Distance", "n", "target mean", "model mean", "distance f", "MSE", "Pearson", "R²"],
      selected.map(row => [
        distanceLabels[distanceOrder.indexOf(row.distance_class)],
        row.count.toLocaleString(),
        metric(row.target_mean),
        metric(row.prediction_mean),
        metric(row.distance_baseline_mean),
        metric(row.mse, 6),
        metric(row.pearson),
        metric(row.r_squared),
      ]),
    ));
    node.append(card);
  });
};

const drawIntegration = (rows, weights) => {
  const node = document.querySelector("#distance-integration");
  const orderedModels = ["sequence", "registered_distance_class", "psd_distance", "psd_distance_plus_sequence"];
  node.append(table(
    ["Window", "Population", "Model", "n", "MSE", "Pearson", "R²"],
    [...rows].sort((a, b) =>
      a.window_size - b.window_size ||
      a.population.localeCompare(b.population) ||
      orderedModels.indexOf(a.model) - orderedModels.indexOf(b.model)
    ).map(row => [
      windowLabel(row.window_size),
      populationLabels[row.population],
      modelLabels[row.model],
      row.count.toLocaleString(),
      metric(row.mse, 6),
      metric(row.pearson),
      metric(row.r_squared),
    ]),
  ));
  const heading = document.createElement("h3");
  heading.textContent = "Nested-validation-fitted combined-kernel weights";
  node.append(heading);
  node.append(table(
    ["Window", "Component", "Weight"],
    [...weights].sort((a, b) => a.window_size - b.window_size || b.weight - a.weight).map(row => [
      windowLabel(row.window_size),
      row.component,
      metric(row.weight, 6),
    ]),
  ));
};

const drawIntegratedDistance = (exploratoryRows, primaryRows) => {
  const node = document.querySelector("#integrated-distance");
  groupedDistance(primaryRows).forEach(group => {
    const [windowRaw, population] = group.split(":");
    const window = Number(windowRaw);
    const primary = primaryRows
      .filter(row => row.window_size === window && row.population === population)
      .sort((a, b) => distanceOrder.indexOf(a.distance_class) - distanceOrder.indexOf(b.distance_class));
    const rows = exploratoryRows.filter(row => row.window_size === window && row.population === population);
    const byKey = Object.fromEntries(rows.map(row => [`${row.distance_class}:${row.model}`, row]));
    const card = article(`${windowLabel(window)} · ${populationLabels[population]}`);
    const svg = svgFrame(card, `Distance-integrated predictions for ${windowLabel(window)} ${populationLabels[population]}`);
    const x = distanceClass => 82 + 760 * distanceOrder.indexOf(distanceClass) / (distanceOrder.length - 1);
    const y = correlation => 278 - 254 * (correlation + 0.1) / 0.7;
    drawAxes(svg, distanceLabels, [-0.1, 0.075, 0.25, 0.425, 0.6], y, "mean pair correlation");
    line(svg, primary.map(row => [x(row.distance_class), y(row.target_mean), false]), "#17221d");
    line(svg, primary.map(row => [x(row.distance_class), y(row.distance_baseline_mean), false]), colors.registered_distance_class, true);
    line(svg, primary.map(row => {
      const point = byKey[`${row.distance_class}:sequence`];
      return [x(row.distance_class), y(point.prediction_mean), false];
    }), colors.sequence);
    line(svg, primary.map(row => {
      const point = byKey[`${row.distance_class}:psd_distance_plus_sequence`];
      return [x(row.distance_class), y(point.prediction_mean), false];
    }), colors.psd_distance_plus_sequence);
    legend(card, [
      ["empirical target mean", "#17221d"],
      ["sequence cosine", colors.sequence],
      ["distance + sequence", colors.psd_distance_plus_sequence],
      ["training-only f(|Δpos|)", colors.registered_distance_class],
    ]);
    card.append(table(
      ["Distance", "Model", "n", "prediction mean", "MSE", "Pearson", "R²"],
      primary.flatMap(primaryRow => ["sequence", "psd_distance", "psd_distance_plus_sequence"].map(model => {
        const row = byKey[`${primaryRow.distance_class}:${model}`];
        return [
          distanceLabels[distanceOrder.indexOf(primaryRow.distance_class)],
          modelLabels[model],
          row.count.toLocaleString(),
          metric(row.prediction_mean),
          metric(row.mse, 6),
          metric(row.pearson),
          metric(row.r_squared),
        ];
      })),
    ));
    node.append(card);
  });
};

fetch("results.json", { cache: "no-store" })
  .then(response => {
    if (!response.ok) throw new Error(`results.json returned ${response.status}`);
    return response.json();
  })
  .then(data => {
    document.querySelector("#status").textContent = data.status;
    document.querySelector("#disclaimer").textContent = data.disclaimer;
    document.querySelector("#protocol").textContent = `Protocol: ${data.protocol_id}`;
    document.querySelector("#provenance").textContent =
      `Split: ${data.split_family} · ${data.retained_probe_count.toLocaleString()} probes · ${data.retained_sample_count} samples · data ${data.data_sha256} · split ${data.split_sha256}`;
    const completed = new Set(data.completed_windows);
    document.querySelector("#progress").replaceChildren(...data.planned_windows.map(window => {
      const item = document.createElement("span");
      item.className = completed.has(window) ? "complete" : "pending";
      item.textContent = `${windowLabel(window)} ${completed.has(window) ? "complete" : "pending"}`;
      return item;
    }));
    drawTuning(data.tuning_sweep, data.latent_dimensions, data.lambda_age_values);
    drawWindowSweep(data.primary_window_sweep);
    drawPairScatter(data.scatter_panels, data.primary_window_sweep);
    drawAge(data.primary_age_metrics);
    drawDirectAge(data.exploratory_age_metrics);
    drawAgeScatter(data.scatter_panels);
    drawAgeClusterDiagnostics(data.age_cluster_diagnostics, data.age_cluster_agreements, data);
    drawPrimaryDistance(data.primary_distance_metrics);
    drawIntegration(data.exploratory_uniform_metrics, data.kernel_weights);
    drawIntegratedDistance(data.exploratory_distance_metrics, data.primary_distance_metrics);
    drawLatentUmap(data.latent_umap_panels);
    const artifacts = document.querySelector("#artifacts");
    data.artifact_ids.forEach(identifier => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = identifier;
      item.append(code);
      artifacts.append(item);
    });
  })
  .catch(error => {
    document.querySelector("#status").textContent = `Site artifact load failed: ${error.message}`;
  });
