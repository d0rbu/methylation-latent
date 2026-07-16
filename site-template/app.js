const colors = {
  seen_by_held_out: "#006b57",
  held_out_by_held_out: "#b24b16",
  island: "#006b57",
  shore: "#e09200",
  shelf: "#7563aa",
  open_sea: "#5a7896",
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

const stageLabels = {
  sequence_features: "CpG density + GC",
  caduceus_age_only: "Caduceus age-only",
  full_latent_metric: "Full latent metric",
};

const svgElement = (name, attributes = {}, text = null) => {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== null) element.textContent = text;
  return element;
};

const empty = (node, message) => {
  const paragraph = document.createElement("p");
  paragraph.className = "empty";
  paragraph.textContent = message;
  node.replaceChildren(paragraph);
};

const svgFrame = (node, label) => {
  const svg = svgElement("svg", {
    viewBox: "0 0 900 340",
    role: "img",
    "aria-label": label,
  });
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
  points.forEach(([x, y]) => {
    svg.append(svgElement("circle", { cx: x, cy: y, r: "3.5", fill: stroke }));
  });
};

const drawAxes = (svg, xLabels, yTicks, yCoordinate, yLabel) => {
  const left = 82;
  const right = 842;
  const top = 24;
  const bottom = 274;
  svg.append(svgElement("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "axis" }));
  svg.append(svgElement("line", { x1: left, x2: left, y1: top, y2: bottom, class: "axis" }));
  yTicks.forEach(value => {
    const y = yCoordinate(value);
    svg.append(svgElement("line", { x1: left, x2: right, y1: y, y2: y, class: "grid" }));
    svg.append(svgElement("text", { x: left - 10, y: y + 4, class: "label", "text-anchor": "end" }, value.toFixed(1)));
  });
  xLabels.forEach((label, index) => {
    const x = left + (right - left) * index / Math.max(1, xLabels.length - 1);
    svg.append(svgElement("text", { x, y: bottom + 22, class: "label", "text-anchor": "middle" }, label));
  });
  svg.append(
    svgElement(
      "text",
      { x: 18, y: (top + bottom) / 2, class: "axis-title", transform: `rotate(-90 18 ${(top + bottom) / 2})`, "text-anchor": "middle" },
      yLabel,
    ),
  );
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

const metric = (value, digits = 4) => Number(value).toFixed(digits);

const drawWindowSweep = rows => {
  const node = document.querySelector("#window-sweep");
  if (!rows.length) return empty(node, "No primary-validated window sweep exists yet.");
  const svg = svgFrame(node, "Pearson correlation by sequence-window width and pair population");
  const windows = [...new Set(rows.map(row => row.window_size))].sort((left, right) => left - right);
  const x = window => 82 + 760 * windows.indexOf(window) / Math.max(1, windows.length - 1);
  const y = pearson => 274 - 250 * (pearson + 1) / 2;
  drawAxes(svg, windows.map(window => `${(window / 1024).toLocaleString()}k`), [-1, -0.5, 0, 0.5, 1], y, "pair Pearson r");
  ["seen_by_held_out", "held_out_by_held_out"].forEach(population => {
    const selected = rows
      .filter(row => row.population === population)
      .sort((left, right) => left.window_size - right.window_size);
    line(svg, selected.map(row => [x(row.window_size), y(row.pearson)]), colors[population]);
  });
  legend(node, [
    ["seen × held-out", colors.seen_by_held_out],
    ["held-out × held-out", colors.held_out_by_held_out],
  ]);
  node.append(
    table(
      ["Window", "Population", "d", "λ age", "MSE", "Pearson", "R²"],
      [...rows]
        .sort((left, right) => left.window_size - right.window_size || left.population.localeCompare(right.population))
        .map(row => [
          `${row.window_size.toLocaleString()} bp`,
          row.population.replaceAll("_", " "),
          row.latent_dimension,
          row.lambda_age,
          metric(row.mse, 6),
          metric(row.pearson),
          metric(row.r_squared),
        ]),
    ),
  );
};

const drawDistance = rows => {
  const node = document.querySelector("#distance-metrics");
  if (!rows.length) return empty(node, "No primary-validated distance-stratified pair metrics exist yet.");
  const groups = [...new Set(rows.map(row => `${row.window_size}:${row.population}`))].sort((left, right) => {
    const [leftWindow, leftPopulation] = left.split(":");
    const [rightWindow, rightPopulation] = right.split(":");
    return Number(leftWindow) - Number(rightWindow) || leftPopulation.localeCompare(rightPopulation);
  });
  groups.forEach(group => {
    const [windowSize, population] = group.split(":");
    const selected = rows
      .filter(row => row.window_size === Number(windowSize) && row.population === population)
      .sort((left, right) => distanceOrder.indexOf(left.distance_class) - distanceOrder.indexOf(right.distance_class));
    const article = document.createElement("article");
    const heading = document.createElement("h3");
    heading.textContent = `${Number(windowSize).toLocaleString()} bp · ${population.replaceAll("_", " ")}`;
    article.append(heading);
    const svg = svgFrame(article, `Distance-binned target, prediction, and distance baseline for ${heading.textContent}`);
    const x = distanceClass => 82 + 760 * distanceOrder.indexOf(distanceClass) / (distanceOrder.length - 1);
    const y = correlation => 274 - 250 * (correlation + 1) / 2;
    drawAxes(svg, distanceLabels, [-1, -0.5, 0, 0.5, 1], y, "mean pair correlation");
    line(svg, selected.map(row => [x(row.distance_class), y(row.target_mean)]), "#17221d");
    line(svg, selected.map(row => [x(row.distance_class), y(row.prediction_mean)]), colors[population]);
    line(svg, selected.map(row => [x(row.distance_class), y(row.distance_baseline_mean)]), "#5d6c65", true);
    legend(article, [
      ["empirical target mean", "#17221d"],
      ["model prediction mean", colors[population]],
      ["training-only distance reference", "#5d6c65"],
    ]);
    article.append(
      table(
        ["Distance", "n", "Target mean", "Pred. mean", "Distance f", "MSE", "Pearson", "R²"],
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
      ),
    );
    node.append(article);
  });
};

const drawAge = rows => {
  const node = document.querySelector("#age-metrics");
  if (!rows.length) return empty(node, "No primary-validated held-out age metrics exist yet.");
  node.append(
    table(
      ["Stage", "Window", "n", "MSE", "Pearson"],
      [...rows]
        .sort((left, right) => left.window_size - right.window_size || left.stage.localeCompare(right.stage))
        .map(row => [
          stageLabels[row.stage],
          `${row.window_size.toLocaleString()} bp`,
          row.count.toLocaleString(),
          metric(row.mse, 6),
          metric(row.pearson),
        ]),
    ),
  );
};

const drawProjection = rows => {
  const node = document.querySelector("#projection");
  if (!rows.length) return empty(node, "No primary-validated held-out latent projection exists yet.");
  const svg = svgFrame(node, "Training-fitted two-dimensional projection of held-out probe latents");
  const xs = rows.map(row => row.x);
  const ys = rows.map(row => row.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  rows.forEach(row => {
    const point = svgElement("circle", {
      cx: 62 + 790 * (row.x - minX) / xSpan,
      cy: 284 - 250 * (row.y - minY) / ySpan,
      r: "2.3",
      fill: colors[row.context],
      opacity: ".65",
    });
    const title = svgElement("title", {}, `${row.probe_id} · ${row.context}`);
    point.append(title);
    svg.append(point);
  });
  svg.append(svgElement("text", { x: 450, y: 325, class: "axis-title", "text-anchor": "middle" }, "training-fitted PC 1"));
  svg.append(
    svgElement(
      "text",
      { x: 18, y: 160, class: "axis-title", transform: "rotate(-90 18 160)", "text-anchor": "middle" },
      "training-fitted PC 2",
    ),
  );
  legend(
    node,
    Object.entries(colors).filter(([name]) => ["island", "shore", "shelf", "open_sea"].includes(name)),
  );
};

fetch("results.json", { cache: "no-store" })
  .then(response => {
    if (!response.ok) throw new Error(`results.json returned ${response.status}`);
    return response.json();
  })
  .then(data => {
    document.querySelector("#status").textContent = data.status;
    document.querySelector("#protocol").textContent = `Protocol: ${data.protocol_id}`;
    document.querySelector("#eligibility").textContent = data.eligibility;
    document.querySelector("#provenance").textContent = data.provenance
      ? `Split: ${data.provenance.split_family} · ${data.provenance.retained_probe_count.toLocaleString()} probes · ${data.provenance.retained_sample_count} samples · data ${data.provenance.data_sha256} · split ${data.provenance.split_sha256}`
      : "No primary-validated scientific provenance exists yet.";
    const artifacts = document.querySelector("#artifacts");
    if (data.artifact_ids.length) {
      data.artifact_ids.forEach(identifier => {
        const item = document.createElement("li");
        const code = document.createElement("code");
        code.textContent = identifier;
        item.append(code);
        artifacts.append(item);
      });
    } else {
      empty(artifacts, "No eligible artifacts.");
    }
    drawWindowSweep(data.window_sweep);
    drawDistance(data.distance_metrics);
    drawAge(data.age_metrics);
    drawProjection(data.projection);
  })
  .catch(error => {
    document.querySelector("#status").textContent = `Site artifact load failed: ${error.message}`;
  });
