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

const empty = (node, message) => {
  const paragraph = document.createElement("p");
  paragraph.className = "empty";
  paragraph.textContent = message;
  node.replaceChildren(paragraph);
};

const svgFrame = (node, label) => {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 900 300");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", label);
  node.append(svg);
  return svg;
};

const line = (svg, points, stroke, dashed = false) => {
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", points.map((point, index) => `${index ? "L" : "M"}${point[0]},${point[1]}`).join(" "));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", stroke);
  path.setAttribute("stroke-width", "2.5");
  if (dashed) path.setAttribute("stroke-dasharray", "7 6");
  svg.append(path);
  points.forEach(([x, y]) => {
    const point = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    point.setAttribute("cx", x);
    point.setAttribute("cy", y);
    point.setAttribute("r", "3.5");
    point.setAttribute("fill", stroke);
    svg.append(point);
  });
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

const drawWindowSweep = (rows) => {
  const node = document.querySelector("#window-sweep");
  if (!rows.length) return empty(node, "No primary-validated window sweep exists yet.");
  const svg = svgFrame(node, "Pearson correlation by sequence-window width and pair population");
  const windows = [...new Set(rows.map(row => row.window_size))].sort((left, right) => left - right);
  ["seen_by_held_out", "held_out_by_held_out"].forEach(population => {
    const selected = rows.filter(row => row.population === population).sort((left, right) => left.window_size - right.window_size);
    line(svg, selected.map(row => [
      80 + 740 * windows.indexOf(row.window_size) / Math.max(1, windows.length - 1),
      250 - 210 * (row.pearson + 1) / 2,
    ]), colors[population]);
  });
  legend(node, [
    ["seen × held-out Pearson", colors.seen_by_held_out],
    ["held-out × held-out Pearson", colors.held_out_by_held_out],
  ]);
};

const drawDistance = (rows) => {
  const node = document.querySelector("#distance-metrics");
  if (!rows.length) return empty(node, "No primary-validated distance-stratified pair metrics exist yet.");
  const groups = [...new Set(rows.map(row => `${row.window_size}:${row.population}`))].sort();
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
    const coordinates = values => values.map((value, index) => [
      70 + 760 * index / Math.max(1, values.length - 1),
      150 - value * 120,
    ]);
    line(svg, coordinates(selected.map(row => row.target_mean)), "#17221d");
    line(svg, coordinates(selected.map(row => row.prediction_mean)), colors[population]);
    line(svg, coordinates(selected.map(row => row.distance_baseline_mean)), "#5d6c65", true);
    legend(article, [
      ["empirical target mean", "#17221d"],
      ["model prediction mean", colors[population]],
      ["training-only distance reference", "#5d6c65"],
    ]);
    node.append(article);
  });
};

const drawAge = (rows) => {
  const node = document.querySelector("#age-metrics");
  if (!rows.length) return empty(node, "No primary-validated held-out age metrics exist yet.");
  const table = document.createElement("table");
  const headings = ["Stage", "Window", "n", "MSE", "corr"];
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
    const values = [row.stage, row.window_size, row.count, row.mse.toFixed(5), row.pearson.toFixed(4)];
    const tableRow = document.createElement("tr");
    values.forEach(value => {
      const cell = document.createElement("td");
      cell.textContent = value;
      tableRow.append(cell);
    });
    body.append(tableRow);
  });
  table.append(head, body);
  node.append(table);
};

const drawProjection = (rows) => {
  const node = document.querySelector("#projection");
  if (!rows.length) return empty(node, "No primary-validated held-out latent projection exists yet.");
  const svg = svgFrame(node, "Training-fitted two-dimensional projection of held-out probe latents");
  const xs = rows.map(row => row.x);
  const ys = rows.map(row => row.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  rows.forEach(row => {
    const point = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    point.setAttribute("cx", 45 + 810 * (row.x - minX) / xSpan);
    point.setAttribute("cy", 270 - 240 * (row.y - minY) / ySpan);
    point.setAttribute("r", "2.3");
    point.setAttribute("fill", colors[row.context]);
    point.setAttribute("opacity", ".65");
    svg.append(point);
  });
  legend(node, Object.entries(colors).filter(([name]) => ["island", "shore", "shelf", "open_sea"].includes(name)));
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
  });
