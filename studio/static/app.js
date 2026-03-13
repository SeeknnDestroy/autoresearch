const state = {
  report: null,
  runs: [],
  selectedRunId: null,
  selectedRun: null,
  selectedLesson: null,
  eventSource: null,
};

const elements = {
  sessionStatus: document.getElementById("session-status"),
  sessionMeta: document.getElementById("session-meta"),
  bestMetric: document.getElementById("best-metric"),
  bestMeta: document.getElementById("best-meta"),
  nextMove: document.getElementById("next-move"),
  reportLessons: document.getElementById("report-lessons"),
  runList: document.getElementById("run-list"),
  detailTitle: document.getElementById("detail-title"),
  detailBadges: document.getElementById("detail-badges"),
  metricsGrid: document.getElementById("metrics-grid"),
  sampleOutput: document.getElementById("sample-output"),
  diffOutput: document.getElementById("diff-output"),
  proposalNote: document.getElementById("proposal-note"),
  implementerNote: document.getElementById("implementer-note"),
  analystNote: document.getElementById("analyst-note"),
  startButton: document.getElementById("start-button"),
  stopButton: document.getElementById("stop-button"),
};

function number(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(digits);
}

function text(value, fallback = "Waiting for data.") {
  return value && String(value).trim() ? String(value).trim() : fallback;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function renderReport() {
  const session = state.report?.session;
  const bestRun = state.report?.best_run;
  elements.sessionStatus.textContent = session?.status ?? "idle";
  elements.sessionStatus.classList.toggle("status-live", session?.status === "running");
  elements.sessionMeta.textContent = session
    ? `${session.run_count} run${session.run_count === 1 ? "" : "s"} on ${session.task_id}`
    : "No active session yet.";
  elements.bestMetric.textContent = bestRun ? number(bestRun.metrics?.val_bpb, 4) : "--";
  elements.bestMeta.textContent = bestRun
    ? `${bestRun.title} kept at recipe ${bestRun.recipe_sha}`
    : "Waiting for the first baseline.";
  elements.nextMove.textContent = state.report?.next_move ?? "Start a session to generate the first baseline.";

  const lessons = state.report?.recent_lessons ?? [];
  if (!lessons.length) {
    elements.reportLessons.innerHTML = `
      <article class="lesson-chip">
        <strong>No lessons yet</strong>
        <p>Baseline first, then the Studio will start writing high-signal notes about what to keep and what to throw away.</p>
      </article>
    `;
    return;
  }

  elements.reportLessons.innerHTML = lessons
    .map(
      (lesson) => `
        <article class="lesson-chip">
          <strong>${lesson.decision.toUpperCase()}</strong>
          <p>${text(lesson.summary)}</p>
        </article>
      `,
    )
    .join("");
}

function renderRuns() {
  if (!state.runs.length) {
    elements.runList.innerHTML = `
      <article class="run-empty">
        The timeline is empty. Start the Studio to watch baseline and candidate runs land here.
      </article>
    `;
    return;
  }

  elements.runList.innerHTML = state.runs
    .map((run) => {
      const metric = run.metrics?.val_bpb !== undefined ? number(run.metrics.val_bpb, 4) : "--";
      const isActive = run.run_id === state.selectedRunId;
      return `
        <article class="run-card ${isActive ? "active" : ""}" data-run-id="${run.run_id}">
          <strong>${run.title}</strong>
          <p>${text(run.hypothesis, "No hypothesis recorded.")}</p>
          <div class="run-meta">
            <span class="badge ${run.decision}">${run.decision}</span>
            <span class="badge">${run.status}</span>
            <span class="badge">val_bpb ${metric}</span>
          </div>
        </article>
      `;
    })
    .join("");

  for (const card of elements.runList.querySelectorAll(".run-card")) {
    card.addEventListener("click", () => {
      const { runId } = card.dataset;
      state.selectedRunId = runId;
      void loadRunDetail(runId);
      renderRuns();
    });
  }
}

function renderRunDetail() {
  const run = state.selectedRun;
  const lesson = state.selectedLesson;
  if (!run) {
    elements.detailTitle.textContent = "No run selected";
    elements.detailBadges.innerHTML = `<span class="detail-badge">Idle</span>`;
    elements.metricsGrid.innerHTML = `
      <article class="metric-card">
        <span class="metric-label">val_bpb</span>
        <strong class="metric-value">--</strong>
      </article>
    `;
    elements.sampleOutput.textContent = "The selected run's sample will appear here.";
    elements.diffOutput.textContent = "No candidate diff yet.";
    elements.proposalNote.textContent = "Waiting for the first run.";
    elements.implementerNote.textContent = "The Studio will describe the recipe change and execution plan here.";
    elements.analystNote.textContent = "The keep-or-discard judgment will appear after evaluation.";
    return;
  }

  elements.detailTitle.textContent = run.title;
  elements.detailBadges.innerHTML = `
    <span class="detail-badge ${run.decision}">${run.decision}</span>
    <span class="detail-badge">${run.status}</span>
    <span class="detail-badge">${run.recipe_sha}</span>
  `;

  const metricEntries = [
    ["val_bpb", number(run.metrics?.val_bpb, 4)],
    ["train_loss", number(run.metrics?.train_loss, 4)],
    ["seconds", number(run.metrics?.elapsed_seconds, 2)],
    ["tokens/sec", number(run.metrics?.tokens_per_second, 1)],
    ["peak_mb", number(run.metrics?.peak_memory_mb, 1)],
    ["device", run.metrics?.device ?? "--"],
  ];

  elements.metricsGrid.innerHTML = metricEntries
    .map(
      ([label, value]) => `
        <article class="metric-card">
          <span class="metric-label">${label}</span>
          <strong class="metric-value">${value}</strong>
        </article>
      `,
    )
    .join("");

  elements.sampleOutput.textContent = text(run.sample_text, "No sample captured.");
  elements.diffOutput.textContent = text(run.diff_text, "No recipe diff captured.");
  elements.proposalNote.textContent = text(run.proposal_note);
  elements.implementerNote.textContent = text(run.implementer_note);
  elements.analystNote.textContent = text(lesson?.summary ?? run.analyst_summary);
}

async function refreshAll() {
  const [reportData, runsData] = await Promise.all([
    fetchJson("/api/report/latest"),
    fetchJson("/api/runs"),
  ]);
  state.report = reportData;
  state.runs = runsData.runs ?? [];
  if (!state.selectedRunId && state.runs.length) {
    state.selectedRunId = state.report?.best_run?.run_id ?? state.runs[state.runs.length - 1].run_id;
  }
  renderReport();
  renderRuns();
  if (state.selectedRunId) {
    await loadRunDetail(state.selectedRunId);
  } else {
    renderRunDetail();
  }
}

async function loadRunDetail(runId) {
  try {
    const detail = await fetchJson(`/api/runs/${runId}`);
    state.selectedRun = detail.run;
    state.selectedLesson = detail.lesson;
    renderRunDetail();
  } catch {
    state.selectedRun = null;
    state.selectedLesson = null;
    renderRunDetail();
  }
}

function connectEvents() {
  if (state.eventSource) {
    state.eventSource.close();
  }
  state.eventSource = new EventSource("/api/events");
  state.eventSource.onmessage = () => {};
  const events = ["session_started", "session_stopping", "session_finished", "run_started", "run_finished"];
  for (const eventName of events) {
    state.eventSource.addEventListener(eventName, () => {
      void refreshAll();
    });
  }
}

async function startSession() {
  elements.startButton.disabled = true;
  try {
    await fetchJson("/api/session/start", { method: "POST" });
    await refreshAll();
  } finally {
    elements.startButton.disabled = false;
  }
}

async function stopSession() {
  elements.stopButton.disabled = true;
  try {
    await fetchJson("/api/session/stop", { method: "POST" });
    await refreshAll();
  } finally {
    elements.stopButton.disabled = false;
  }
}

elements.startButton.addEventListener("click", () => {
  void startSession();
});

elements.stopButton.addEventListener("click", () => {
  void stopSession();
});

void refreshAll();
connectEvents();
