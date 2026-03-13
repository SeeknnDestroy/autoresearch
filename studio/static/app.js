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
  spotlightTitle: document.getElementById("spotlight-title"),
  spotlightMeta: document.getElementById("spotlight-meta"),
  stageHeadline: document.getElementById("stage-headline"),
  scoreStrip: document.getElementById("score-strip"),
  nextMove: document.getElementById("next-move"),
  reportLessons: document.getElementById("report-lessons"),
  mutationAtlas: document.getElementById("mutation-atlas"),
  runList: document.getElementById("run-list"),
  detailTitle: document.getElementById("detail-title"),
  detailBadges: document.getElementById("detail-badges"),
  detailHeadline: document.getElementById("detail-headline"),
  detailSubline: document.getElementById("detail-subline"),
  detailStage: document.getElementById("detail-stage"),
  metricsGrid: document.getElementById("metrics-grid"),
  sampleOutput: document.getElementById("sample-output"),
  diffOutput: document.getElementById("diff-output"),
  mutationCard: document.getElementById("mutation-card"),
  dialCard: document.getElementById("dial-card"),
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

function labelize(value, fallback = "baseline") {
  return value ? String(value).replaceAll("_", " ") : fallback;
}

function fieldIcon(changeSpec = {}) {
  return (changeSpec.icon || changeSpec.field || "baseline").replaceAll("_", " ").toUpperCase();
}

function verdictClass(run = {}) {
  return run.decision || "pending";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function stageHeadlineFor(run) {
  if (!run) return "The lab is waiting for its first baseline.";
  if (run.stage === "runner") return `${run.title} is currently being stress-tested on the local lane.`;
  if (run.stage === "analyst") return `The analyst is deciding whether ${run.title.toLowerCase()} earns a keeper badge.`;
  if (run.stage === "implementer") return `The implementer is wiring ${run.title.toLowerCase()} into the recipe snapshot.`;
  if (run.stage === "proposer") return `The proposer is teeing up ${run.title.toLowerCase()}.`;
  return "The lab is moving.";
}

function currentSpotlightRun() {
  const running = state.runs.findLast((run) => run.status === "running");
  if (running) return running;
  return state.report?.spotlight || state.runs.at(-1) || null;
}

function renderReport() {
  const session = state.report?.session;
  const bestRun = state.report?.best_run;
  const spotlight = currentSpotlightRun();
  const scoreboard = state.report?.scoreboard ?? { completed: 0, kept: 0, discarded: 0, crashes: 0 };

  elements.sessionStatus.textContent = session?.status ?? "idle";
  elements.sessionStatus.classList.toggle("status-live", session?.status === "running");
  elements.sessionMeta.textContent = session
    ? `${session.run_count} run${session.run_count === 1 ? "" : "s"} on ${session.task_id}`
    : "No active session yet.";
  elements.stageHeadline.textContent = session?.stage_headline || stageHeadlineFor(spotlight);
  elements.bestMetric.textContent = bestRun ? number(bestRun.metrics?.val_bpb, 4) : "--";
  elements.bestMeta.textContent = bestRun
    ? `${bestRun.title} held the crown at recipe ${bestRun.recipe_sha}`
    : "Waiting for the first baseline.";
  elements.spotlightTitle.textContent = spotlight?.title ?? "No run yet";
  elements.spotlightMeta.textContent = spotlight
    ? `${labelize(spotlight.change_spec?.field, "baseline")} · stage ${spotlight.stage || "queued"}`
    : "The newest run will narrate itself here.";
  elements.nextMove.textContent = state.report?.next_move ?? "Start a session to generate the first baseline.";

  const scores = [
    ["Completed", scoreboard.completed],
    ["Kept", scoreboard.kept],
    ["Discarded", scoreboard.discarded],
    ["Crashes", scoreboard.crashes],
  ];
  elements.scoreStrip.innerHTML = scores
    .map(
      ([label, value]) => `
        <article class="score-pill">
          <span class="score-label">${label}</span>
          <strong class="score-value">${value}</strong>
        </article>
      `,
    )
    .join("");

  const lessons = state.report?.recent_lessons ?? [];
  if (!lessons.length) {
    elements.reportLessons.innerHTML = `
      <article class="lesson-chip">
        <strong>No lessons yet</strong>
        <p>Baseline first, then the analyst starts writing sharp notes about what the lab should keep and what it should throw away.</p>
      </article>
    `;
  } else {
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

  if (!state.runs.length) {
    elements.mutationAtlas.innerHTML = `
      <article class="atlas-empty">
        The atlas will light up once the proposer starts mutating the recipe.
      </article>
    `;
  } else {
    elements.mutationAtlas.innerHTML = state.runs
      .slice(-4)
      .reverse()
      .map((run) => {
        const spec = run.change_spec || {};
        const mood = spec.mood || (spec.type === "baseline" ? "reference lock" : "fresh mutation");
        const why = spec.why || "Establish the current local reference point.";
        return `
          <article class="atlas-card">
            <strong>${run.title}</strong>
            <p>${mood} · ${why}</p>
            <div class="badge-row">
              <span class="badge ${verdictClass(run)}">${run.decision}</span>
              <span class="badge">${labelize(spec.field, "baseline")}</span>
              <span class="badge">${run.stage || "queued"}</span>
            </div>
          </article>
        `;
      })
      .join("");
  }
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
      const delta = state.report?.best_run && run.metrics?.val_bpb !== undefined
        ? Number(run.metrics.val_bpb) - Number(state.report.best_run.metrics.val_bpb)
        : null;
      const deltaText = delta === null ? "delta --" : `delta ${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`;
      return `
        <article class="run-card ${isActive ? "active" : ""} ${run.status === "running" ? "is-running" : ""}" data-run-id="${run.run_id}">
          <strong>${run.title}</strong>
          <p>${text(run.hypothesis, "No hypothesis recorded.")}</p>
          <div class="run-meta">
            <span class="badge ${verdictClass(run)}">${run.decision}</span>
            <span class="badge">${run.stage || run.status}</span>
            <span class="badge">val_bpb ${metric}</span>
            <span class="badge">${deltaText}</span>
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
    elements.detailHeadline.textContent = "Choose a run to inspect its pulse.";
    elements.detailSubline.textContent = "The lab will surface the mutation vibe, stage, and verdict here.";
    elements.detailStage.textContent = "idle";
    elements.metricsGrid.innerHTML = `
      <article class="metric-card">
        <span class="metric-label">val_bpb</span>
        <strong class="metric-value">--</strong>
      </article>
    `;
    elements.sampleOutput.textContent = "The selected run's sample will appear here.";
    elements.diffOutput.textContent = "No candidate diff yet.";
    elements.mutationCard.innerHTML = `
      <div class="mutation-icon">BASELINE</div>
      <div class="mutation-body">
        <strong>No mutation yet</strong>
        <p>The proposer will explain the axis, vibe, and reason for the next change here.</p>
      </div>
    `;
    elements.dialCard.innerHTML = `
      <div class="dial-values">
        <span>from <strong>--</strong></span>
        <span>to <strong>--</strong></span>
      </div>
      <p>The exact field movement will appear once a candidate run lands.</p>
    `;
    elements.proposalNote.textContent = "Waiting for the first run.";
    elements.implementerNote.textContent = "The Studio will describe the recipe change and execution plan here.";
    elements.analystNote.textContent = "The keep-or-discard judgment will appear after evaluation.";
    return;
  }

  const spec = run.change_spec || {};
  const bestMetric = state.report?.best_run?.metrics?.val_bpb;
  const delta = bestMetric !== undefined && run.metrics?.val_bpb !== undefined
    ? Number(run.metrics.val_bpb) - Number(bestMetric)
    : null;
  const deltaLabel = delta === null ? "delta --" : `delta ${delta >= 0 ? "+" : ""}${delta.toFixed(4)} vs best`;

  elements.detailTitle.textContent = run.title;
  elements.detailBadges.innerHTML = `
    <span class="detail-badge ${verdictClass(run)}">${run.decision}</span>
    <span class="detail-badge">${run.status}</span>
    <span class="detail-badge">${run.recipe_sha}</span>
    <span class="detail-badge">${deltaLabel}</span>
  `;
  elements.detailHeadline.textContent = spec.type === "baseline"
    ? "The baseline established the first local reference pulse."
    : `${spec.mood || "fresh mutation"} on ${labelize(spec.field)}.`;
  elements.detailSubline.textContent = spec.type === "baseline"
    ? "Everything after this run gets judged against this first held reference."
    : `${text(spec.why, "The proposer is exploring a new axis.")} ${text(lesson?.follow_up || run.follow_up, "")}`;
  elements.detailStage.textContent = run.stage || run.status;

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
  elements.mutationCard.innerHTML = `
    <div class="mutation-icon">${fieldIcon(spec)}</div>
    <div class="mutation-body">
      <strong>${spec.type === "baseline" ? "Reference lock" : text(spec.mood, "Fresh mutation")}</strong>
      <p>${spec.type === "baseline" ? "This run captures the untouched recipe so later moves have a real incumbent to challenge." : text(spec.rationale, "No proposer rationale recorded.")}</p>
    </div>
  `;
  elements.dialCard.innerHTML = spec.type === "baseline"
    ? `
        <div class="dial-values">
          <span>from <strong>--</strong></span>
          <span>to <strong>--</strong></span>
        </div>
        <p>The baseline does not move a dial; it anchors the rest of the session.</p>
      `
    : `
        <div class="dial-values">
          <span>from <strong>${spec.from}</strong></span>
          <span>to <strong>${spec.to}</strong></span>
        </div>
        <p>${text(spec.why, "The proposer is probing one recipe dial at a time.")}</p>
      `;
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

  const spotlight = currentSpotlightRun();
  if (!state.selectedRunId && spotlight) {
    state.selectedRunId = spotlight.run_id;
  }
  if (state.selectedRunId && !state.runs.find((run) => run.run_id === state.selectedRunId)) {
    state.selectedRunId = spotlight?.run_id ?? null;
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
  const events = ["session_started", "session_stopping", "session_finished", "run_started", "run_stage", "run_finished"];
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
