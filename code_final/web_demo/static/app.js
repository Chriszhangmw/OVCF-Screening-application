const ACTIONS = [
  ["frontal", "Frontal Image"],
  ["back", "Back Image"],
  ["lateral", "Lateral Image"],
  ["roll_left", "Left Roll"],
  ["roll_right", "Right Roll"],
  ["supine_to_sit", "Supine to Sit"],
  ["sit_to_supine", "Sit to Supine"],
];

const TEXT_REPLACEMENTS = new Map([
  ["\u65e0", "No"],
  ["\u6709", "Yes"],
  ["\u7537", "Male"],
  ["\u5973", "Female"],
  ["\u9ad8\u80fd\u91cf\u5916\u4f24", "High-energy trauma"],
  ["\u4f4e\u80fd\u91cf\u5916\u4f24", "Low-energy trauma"],
  ["\u65e0\u4f7f\u7528", "No use"],
  ["\u65e0\u8f85\u52a9", "No assistance"],
  ["\u65e0\u9700\u8f85\u52a9", "No assistance needed"],
  ["\u5355\u624b\u8f85\u52a9", "Single-hand assistance"],
  ["\u5355\u624b\u652f\u6491", "Single-hand support"],
  ["\u53cc\u624b\u7528\u529b\u63a8\u5e8a", "Two-hand bed push"],
  ["\u53cc\u624b\u53d1\u529b\u8f85\u52a9", "Two-hand powered assistance"],
  ["\u53cc\u624b\u8f85\u52a9", "Two-hand assistance"],
  ["\u65e0\u652f\u6491", "No support"],
  ["\u90e8\u5206\u652f\u6491", "Partial support"],
  ["\u90e8\u5206\u8f85\u52a9", "Partial assistance"],
  ["\u660e\u663e\u4f9d\u8d56\u652f\u6491", "Clear reliance on support"],
  ["\u660e\u663e\u4f9d\u8d56", "Clear reliance"],
  ["\u53cc\u624b\u652f\u6491", "Two-hand support"],
  [
    "\u7814\u7a76\u6a21\u578b\u8f93\u51fa\uff0c\u4e0d\u66ff\u4ee3\u4e34\u5e8a\u8bca\u65ad\uff1b\u89e3\u91ca\u4e3a\u6a21\u578b\u7279\u5f81\u8d21\u732e\u542f\u53d1\u5f0f\u6392\u5e8f\uff0c\u5e76\u7ed3\u5408 LLM \u7ed3\u6784\u5316\u7406\u7531\u3002",
    "Research model output; not a substitute for clinical diagnosis. Explanations are heuristic feature-contribution rankings combined with structured LLM rationale.",
  ],
  [
    "\u5f53\u524d\u4e3a\u7f13\u5b58\u6f14\u793a\u7ed3\u679c\uff1b\u771f\u5b9e\u5206\u6790\u9700\u5173\u95ed\u7f13\u5b58\u6f14\u793a\u6a21\u5f0f\u5e76\u4fdd\u8bc1\u6a21\u578b\u73af\u5883\u4e0e API \u53ef\u7528\u3002",
    "This is a cached demo result. Live analysis requires live mode plus an available model environment and API.",
  ],
  ["\u4e34\u5e8a\u57fa\u672c\u4fe1\u606f\u7279\u5f81\uff0c\u65e0 LLM explanation\u3002", "Clinical metadata feature; no LLM explanation."],
]);

const imageInput = document.querySelector("#image-input");
const videoInput = document.querySelector("#video-input");
const mediaList = document.querySelector("#media-list");
const analyzeBtn = document.querySelector("#analyze-btn");
const resetBtn = document.querySelector("#reset-btn");
const runState = document.querySelector("#run-state");
const patientForm = document.querySelector("#patient-form");
const liveMode = document.querySelector("#live-mode");
const fallbackMode = document.querySelector("#fallback-mode");
const results = document.querySelector("#results");
const logList = document.querySelector("#log-list");

let selectedFiles = [];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function containsAny(value, needles) {
  return needles.some((needle) => value.includes(needle));
}

function hasChinese(value) {
  return /[\u3400-\u9fff]/.test(String(value ?? ""));
}

function displayText(value, fallback = "") {
  const raw = String(value ?? "");
  if (!raw) return fallback;
  return TEXT_REPLACEMENTS.get(raw) || raw;
}

function displayLongText(value, fallback = "") {
  const text = displayText(value, fallback);
  if (!text) return fallback;
  return text;
}

function roleForFilename(name, type) {
  const s = name.toLowerCase();
  if (containsAny(s, ["\u6b63\u4f4d", "\u524d\u4f4d", "front", "frontal", "anterior"])) return "frontal";
  if (containsAny(s, ["\u80cc\u4f4d", "\u540e\u4f4d", "posterior", "back"])) return "back";
  if (containsAny(s, ["\u4fa7\u4f4d", "\u5074\u4f4d", "lateral", "side"])) return "lateral";
  if (containsAny(s, ["\u5de6\u7ffb", "roll_left", "left_roll"])) return "roll_left";
  if (containsAny(s, ["\u53f3\u7ffb", "roll_right", "right_roll"])) return "roll_right";
  if (containsAny(s, ["\u8eba\u5230\u5750", "\u5367\u5230\u5750", "\u5750\u8d77\u6765", "supine_to_sit", "lie_to_sit"])) return "supine_to_sit";
  if (containsAny(s, ["\u5750\u5230\u8eba", "\u5750\u5230\u5367", "sit_to_supine", "sit_to_lie"])) return "sit_to_supine";
  if (type.startsWith("image/")) return "frontal";
  if (type.startsWith("video/")) return "roll_left";
  return "unassigned";
}

function actionLabel(role) {
  const hit = ACTIONS.find(([value]) => value === role);
  return hit ? hit[1] : "Unassigned";
}

function addFiles(files) {
  selectedFiles.push(
    ...[...files].map((file) => ({
      file,
      role: roleForFilename(file.name, file.type || ""),
    })),
  );
  renderMediaList();
}

function renderMediaList() {
  if (!selectedFiles.length) {
    mediaList.innerHTML = "";
    return;
  }
  mediaList.innerHTML = selectedFiles
    .map((item, index) => {
      const url = URL.createObjectURL(item.file);
      const preview = item.file.type.startsWith("image/")
        ? `<img src="${url}" alt="">`
        : `<video src="${url}" muted></video>`;
      const options = ACTIONS.map(([value, label]) => `<option value="${value}" ${value === item.role ? "selected" : ""}>${label}</option>`).join("");
      return `
        <article class="media-item">
          <div class="media-preview">${preview}</div>
          <div class="media-meta">
            <strong>${escapeHtml(item.file.name)}</strong>
            <span>${(item.file.size / 1048576).toFixed(2)} MB</span>
          </div>
          <select data-index="${index}" class="role-select" aria-label="Media modality">
            ${options}
            <option value="unassigned" ${item.role === "unassigned" ? "selected" : ""}>Unassigned</option>
          </select>
        </article>
      `;
    })
    .join("");

  document.querySelectorAll(".role-select").forEach((select) => {
    select.addEventListener("change", (event) => {
      selectedFiles[Number(event.target.dataset.index)].role = event.target.value;
    });
  });
}

imageInput.addEventListener("change", (event) => {
  addFiles(event.target.files);
  event.target.value = "";
});

videoInput.addEventListener("change", (event) => {
  addFiles(event.target.files);
  event.target.value = "";
});

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const value = String(reader.result || "");
      resolve(value.includes(",") ? value.split(",", 2)[1] : value);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function formDataToObject(form) {
  const data = new FormData(form);
  const out = {};
  for (const [key, value] of data.entries()) {
    out[key] = value;
  }
  return out;
}

function formatPercent(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return `${(n * 100).toFixed(digits)}%`;
}

function scoreText(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return value || value === 0 ? displayText(value) : "--";
  return n.toFixed(2);
}

function setRunState(text, busy = false) {
  runState.textContent = text;
  analyzeBtn.disabled = busy;
  resetBtn.disabled = busy;
  analyzeBtn.textContent = busy ? "Analyzing..." : "Start Analysis";
}

async function runAnalysis() {
  setRunState("Preparing uploaded media", true);
  const files = [];
  for (const item of selectedFiles) {
    files.push({
      name: item.file.name,
      type: item.file.type || "application/octet-stream",
      role: item.role,
      data: await fileToBase64(item.file),
    });
  }

  setRunState(liveMode.checked ? "Running live image/video analysis. Please wait." : "Loading cached demo result", true);
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient: formDataToObject(patientForm),
      files,
      demo_mode: !liveMode.checked,
      allow_fallback: fallbackMode.checked,
    }),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    if (data.logs) {
      renderLogs(data.logs);
      results.classList.remove("hidden");
    }
    throw new Error(displayText(data.error, "Analysis failed"));
  }
  renderResults(data);
  setRunState(data.analysis_mode === "live_model" ? "Live model analysis complete" : "Cached demo result loaded", false);
}

analyzeBtn.addEventListener("click", () => {
  runAnalysis().catch((error) => {
    setRunState(error.message, false);
  });
});

resetBtn.addEventListener("click", async () => {
  selectedFiles = [];
  imageInput.value = "";
  videoInput.value = "";
  renderMediaList();
  results.classList.add("hidden");
  if (logList) logList.innerHTML = "";
  document.querySelector("#prediction-label").textContent = "Waiting for Analysis";
  document.querySelector("#probability-label").textContent = "0%";
  document.querySelector("#risk-gauge").style.setProperty("--risk", "0%");
  const caseInput = patientForm.querySelector('[name="case_id"]');
  if (caseInput) {
    const suffix = String(Math.floor(Math.random() * 900) + 100);
    caseInput.value = `OVCF-${new Date().toISOString().slice(0, 10).replaceAll("-", "")}-${suffix}`;
  }
  setRunState("Clearing temporary files", true);
  try {
    const response = await fetch("/api/reset", { method: "POST" });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(displayText(data.error, "Clear failed"));
    setRunState(displayText(data.message, "Cleared. Ready for the next patient."), false);
  } catch (error) {
    setRunState(error.message, false);
  }
});

function metricBox(label, value) {
  return `<div class="metric-box"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderResults(data) {
  const report = data.report || {};
  const probability = Number(report.ovcf_probability || 0);
  const threshold = Number(report.threshold || 0.5);
  document.querySelector("#prediction-label").textContent = displayText(report.prediction, "Waiting for Analysis");
  document.querySelector("#probability-label").textContent = formatPercent(probability, 1);
  document.querySelector("#risk-gauge").style.setProperty("--risk", `${Math.round(probability * 100)}%`);
  document.querySelector("#model-note").textContent = displayLongText(report.note || data.note || "");

  const training = report.model_training_summary || {};
  const cv = training.cv_metrics_at_youden || training.cv_metrics_at_0_50 || {};
  const missing = (report.missing_modalities || []).map(actionLabel).join(", ") || "None";
  document.querySelector("#metrics-panel").innerHTML = [
    metricBox("Decision Threshold", threshold.toFixed(3)),
    metricBox("Cross-validation AUC", Number.isFinite(Number(cv.auc)) ? Number(cv.auc).toFixed(3) : "--"),
    metricBox("Sensitivity", formatPercent(cv.sensitivity, 1)),
    metricBox("Specificity", formatPercent(cv.specificity, 1)),
    metricBox("Missing Modalities", missing),
    metricBox("Analysis Source", data.analysis_mode === "live_model" ? "Live Model" : "Cached Demo"),
  ].join("");

  const comp = data.completeness || {};
  document.querySelector("#evidence-completeness").textContent =
    `Uploaded ${comp.uploaded_count || 0}/${comp.expected_count || ACTIONS.length} modalities`;

  renderAssessment(data.llm_explanations || []);
  renderFeatureBars(report.top_explanatory_features || data.explanation_table || []);
  renderLogs(data.logs || []);
  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderAssessment(rows) {
  const grid = document.querySelector("#assessment-grid");
  if (!rows.length) {
    grid.innerHTML = `<div class="empty-state">No structured evidence yet.</div>`;
    return;
  }
  grid.innerHTML = rows
    .slice(0, 18)
    .map((row) => {
      const rawAction = row.action || row.prefix || "Assessment Item";
      const action = ACTIONS.some(([value]) => value === rawAction) ? actionLabel(rawAction) : String(rawAction).replaceAll("_", " ");
      const indicator = row.indicator || row.display || "Indicator";
      const score = scoreText(row.score || row.value_for_model || row.raw_value);
      const explanation = displayLongText(row.explanation || row.llm_explanation || "");
      return `
        <article class="assessment-card">
          <div>
            <span class="tag">${escapeHtml(action)}</span>
            <strong>${escapeHtml(indicator)}</strong>
          </div>
          <div class="score-pill">${escapeHtml(score)}</div>
          <p>${escapeHtml(explanation)}</p>
        </article>
      `;
    })
    .join("");
}

function renderFeatureBars(rows) {
  const wrap = document.querySelector("#feature-bars");
  if (!rows.length) {
    wrap.innerHTML = `<div class="empty-state">No model explanation results yet.</div>`;
    return;
  }
  const max = Math.max(...rows.map((r) => Math.abs(Number(r.risk_signal || r.model_importance || 0))), 0.001);
  wrap.innerHTML = rows
    .slice(0, 10)
    .map((row) => {
      const value = Math.abs(Number(row.risk_signal || row.model_importance || 0));
      const width = Math.max(4, Math.round((value / max) * 100));
      const display = row.display || row.feature || "Feature";
      const explanation = displayLongText(row.llm_explanation || "");
      return `
        <article class="feature-row">
          <div class="feature-main">
            <strong>${escapeHtml(display)}</strong>
            <span>${escapeHtml(row.feature_type || "Feature")} | Raw value ${escapeHtml(scoreText(row.raw_value))}</span>
          </div>
          <div class="bar-track"><div style="width:${width}%"></div></div>
          <p>${escapeHtml(explanation)}</p>
        </article>
      `;
    })
    .join("");
}

function renderLogs(rows) {
  if (!logList) return;
  if (!rows.length) {
    logList.innerHTML = `<div class="empty-state">No run logs yet.</div>`;
    return;
  }
  logList.innerHTML = rows
    .slice(-120)
    .map((row) => {
      const event = row.event || "log";
      const time = row.time || "";
      const details = { ...row };
      delete details.event;
      delete details.time;
      delete details.source;
      const detailText = JSON.stringify(details, null, 2);
      return `
        <article class="log-row">
          <div>
            <strong>${escapeHtml(event)}</strong>
            <span>${escapeHtml(time)}${row.source ? " | " + escapeHtml(row.source) : ""}</span>
          </div>
          <pre>${escapeHtml(detailText)}</pre>
        </article>
      `;
    })
    .join("");
}
