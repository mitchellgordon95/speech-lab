const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = (v) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const colors = [
  "#287454",
  "#d48b45",
  "#647cc0",
  "#a66c91",
  "#819e39",
  "#5d9aa3",
];
let status,
  clips = [],
  axes = [],
  active = localStorage.getItem("speechlab.clip"),
  recorder,
  stream,
  audioContext,
  recordingTimer,
  frame;
let currentPage = "home",
  lastResult,
  busy = false;
const icons = ["≋", "∴", "⌁", "↗", "⌇", "◎", "⇄", "◌"];

async function api(path, options = {}) {
  const response = await fetch("/api" + path, options);
  if (!response.ok) {
    let text = `Request failed (${response.status})`;
    try {
      const error = await response.json();
      text =
        typeof error.detail === "string"
          ? error.detail
          : JSON.stringify(error.detail);
    } catch {}
    throw new Error(text);
  }
  return response.json();
}
const post = (path, body) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
function toast(text) {
  $("#toast").textContent = text;
  $("#toast").style.display = "block";
  setTimeout(() => ($("#toast").style.display = "none"), 5000);
}
function error(text) {
  const el = $("#job");
  if (el) {
    el.className = "job show error";
    el.textContent = text;
  } else toast(text);
}
const selectedClip = () => clips.find((c) => c.id === active);
const values = (selector) =>
  [...$(selector).selectedOptions].map((x) => x.value);
function options(items, chosen) {
  return items
    .map(
      (c) =>
        `<option value="${esc(c.id)}" ${c.id === chosen ? "selected" : ""}>${esc(c.label || c.name)}${c.session ? " · " + esc(c.session) : ""}</option>`,
    )
    .join("");
}
function modelFields(omni = false) {
  const names = Object.keys(status.models).filter((n) =>
    omni ? n.startsWith("qwen") : !n.startsWith("qwen"),
  );
  return `<label>Encoder<select id="model">${names.map((n) => `<option value="${n}">${n} · ${esc(status.models[n].size)}</option>`).join("")}</select></label><label>Layer<input id="layer" type="number" value="${omni ? -1 : 6}" min="-1" max="32"><span class="field-help">−1 = last; layer conventions differ by model.</span></label>`;
}
function allModelFields() {
  return `<label>Encoder<select id="model">${Object.keys(status.models)
    .map((n) => `<option value="${n}">${n}</option>`)
    .join(
      "",
    )}</select></label><label>Layer<input id="layer" type="number" value="6" min="-1" max="32"></label>`;
}
function clipSelect(id, label, multi = false, filter = () => true) {
  return `<label>${label}<select id="${id}" ${multi ? "multiple" : ""}>${options(clips.filter(filter), multi ? null : active)}</select>${multi ? '<span class="field-help">⌘-click to select several recordings.</span>' : ""}</label>`;
}

function render() {
  if (recorder && recorder.state === "recording") recorder.stop();
  lastResult = null;
  currentPage = location.hash.slice(1) || "home";
  const demo = status.demos.find((d) => d.id === currentPage);
  if (!demo && currentPage !== "home") {
    location.hash = "home";
    return;
  }
  $("#navigation").innerHTML =
    `<a href="#home" class="${currentPage === "home" ? "active" : ""}"><span class="num">⌂</span>Overview</a>` +
    status.demos
      .map(
        (d) =>
          `<a href="#${d.id}" class="${d.id === currentPage ? "active" : ""}"><span class="num">${d.number}</span>${esc(d.short)}</a>`,
      )
      .join("");
  $("#breadcrumb").textContent = demo
    ? `EXPERIMENTS / ${demo.number}`
    : "EXPERIMENTS / OVERVIEW";
  $("#view").innerHTML = demo ? experiment(demo) : home();
  bind();
  if (demo) {
    renderCapture();
    loadHistory();
  }
}
function home() {
  return `<section class="hero"><div><span class="eyebrow">LISTEN DIFFERENTLY</span><h1>See what changes.</h1><p>A workbench for exploring pronunciation. Record a sound, inspect the evidence, and discover which feedback helps your next attempt.</p></div><div class="hero-art" aria-hidden="true">${Array.from({ length: 23 }, (_, i) => `<i style="height:${12 + Math.abs(Math.sin(i * 0.65)) * 60 * (1 - Math.abs(i - 11) / 20)}px"></i>`).join("")}</div></section>
  <div class="intro-strip"><span class="circle">↗</span><div><strong>Start with one sound.</strong><p>Record a vowel or a short syllable. Reuse the same clip across every experiment.</p></div><button class="primary" data-goto="acoustics">Make a recording →</button></div>
  <div class="section-head"><h2>Eight ways to explore</h2><span>Local first · ${clips.length} recordings in your library</span></div>
  <div class="grid">${status.demos.map((d, i) => `<a href="#${d.id}" class="demo-card"><div class="card-top"><div class="icon-box" aria-hidden="true">${icons[i]}</div><span class="num">EXPERIMENT ${d.number}</span></div><h3>${esc(d.title)}</h3><p>${esc(d.description)}</p><div class="card-foot"><span>${esc(d.tag)}</span><span class="arrow">↗</span></div></a>`).join("")}</div>
  <div class="section-head"><h2>A shared set of recordings</h2><button class="text-button" id="open-library">Open library →</button></div>
  <div class="panel"><div class="three-col"><div><h3>01 / Record & label</h3><p class="small">Keep the word, speaker, and session with each clip. Crop to the sound you want to study.</p></div><div><h3>02 / Compare evidence</h3><p class="small">Inspect acoustics, frozen embeddings, and estimated articulation on the same input.</p></div><div><h3>03 / Test the feedback</h3><p class="small">Fit a contrast on labeled examples. Check a new recording day before trusting the direction.</p></div></div></div>`;
}
function experiment(d) {
  return `<section class="hero"><div><span class="eyebrow">EXPERIMENT ${d.number} / ${esc(d.tag.toUpperCase())}</span><h1>${esc(d.title)}</h1><p>${esc(d.description)}</p></div><span class="badge">${d.id === "providers" ? "Optional external service" : "Runs on your Mac"}</span></section>
  <div id="capture"></div><section class="panel"><div class="panel-header"><h3>Experiment setup</h3><span class="badge">${esc(d.tag)}</span></div>${experimentForm(d.id)}<div class="actions"><button class="primary" id="run" ${busy ? "disabled" : ""}>${d.id === "providers" ? "Send selected audio →" : d.id === "directions" ? "Fit direction →" : "Run experiment →"}</button><span class="small" id="run-help">${d.id === "acoustics" ? "No model download needed." : d.id === "providers" ? "Only this action sends audio to the selected provider." : "First use downloads model weights. One model runs at a time."}</span></div></section>
  <div id="job" class="job" role="status" aria-live="polite"></div><div id="result"><div class="panel empty"><div class="symbol">${icons[Number(d.number) - 1]}</div><h3>Your results will appear here</h3><p>Choose recordings and run the experiment. Original audio is preserved.</p></div></div><section id="history"></section>`;
}
function experimentForm(id) {
  if (id === "acoustics")
    return `<div class="fields"><label>Formant ceiling (Hz)<input id="ceiling" type="number" value="5500" min="3500" max="8000" step="100"><span class="field-help">Try 5000–6500 if tracks look implausible.</span></label></div><p class="small">Choose a vowel region for formants or a consonant region for spectral moments. The selected region applies to the whole analysis.</p>`;
  if (id === "embeddings" || id === "omni")
    return `<div class="fields">${modelFields(id === "omni")}</div>${clipSelect("compare-clips", "Recordings to compare", true)}<p class="small">Use matching syllable contexts. A PCA plot shows variation, not a physical map of your mouth.</p>`;
  if (id === "directions")
    return `<div class="fields">${allModelFields()}<label>Direction name<input id="axis-name" value="My pronunciation contrast"></label><label>Validation groups<select id="group-by"><option value="session">Recording session</option><option value="speaker">Speaker</option></select></label></div><div class="two-col"><div><label>Left endpoint label<input id="negative-label" value="Less aspiration"></label><br>${clipSelect("negative-clips", "Left examples · at least 3", true)}</div><div><label>Right endpoint label<input id="positive-label" value="More aspiration"></label><br>${clipSelect("positive-clips", "Right examples · at least 3", true)}</div></div><div class="note">Use both labels in each session. Grouped validation will stay unavailable if there is not enough independent data. Labels describe your examples; the model does not verify them.</div>`;
  if (id === "articulation")
    return `<div class="note">SPARC estimates six articulator trajectories using WavLM layer 9 and the published linear projection. Coordinates are in a reference space, not measured millimeters in your mouth.</div><p class="small">Use at least 0.4 seconds. No speech synthesizer or full language model is loaded.</p>`;
  if (id === "calibration")
    return `<div class="fields">${allModelFields()}</div><div class="two-col">${clipSelect("reference-clips", "Population references · accepted examples", true)}${clipSelect("anchor-clips", "Your personal anchors · accepted examples", true)}</div><div class="note">The active recording is the held-out attempt. Choose the same target sound for references and anchors. Do not include the attempt in either set.</div>`;
  if (id === "providers")
    return `<div class="fields"><label>Provider<select id="provider">${Object.entries(
      status.providers,
    )
      .map(
        ([n, s]) =>
          `<option value="${n}">${n} · ${s.configured ? "configured" : "needs credentials"}</option>`,
      )
      .join(
        "",
      )}</select></label><label>Intended text<input id="reference" placeholder="你好" value="${esc(selectedClip()?.target || "")}"></label><label>Language<input id="provider-language" value="zh-CN"></label></div><div id="provider-note" class="note"></div><label class="checkbox"><input type="checkbox" id="cloud-consent">Send this selected audio region to the chosen provider. Their API usage and data policies apply.</label>`;
  if (id === "feedback")
    return `<div class="fields"><label>Learned direction<select id="axis">${axes.length ? options(axes) : '<option value="">Fit a direction in experiment 04 first</option>'}</select></label><label>Practice condition<select id="condition"><option value="visual">Show visual feedback</option><option value="hidden">Hide feedback · retention trial</option></select></label></div><div class="note">Record a new attempt after each adjustment. A position on this axis is not an accuracy percentage. Hidden trials save the result without showing the position.</div>`;
}

function renderCapture() {
  const clip = selectedClip();
  $("#capture").innerHTML =
    `<section class="panel"><div class="panel-header"><h3>Recording</h3><button class="text-button" id="open-library">Library · ${clips.length} clips ↗</button></div>
  <div class="fields"><label>Recording label<input id="clip-label" placeholder="e.g. shā · attempt 1" value="${esc(clip?.label || "")}"></label><label>Target / intended sound<input id="clip-target" placeholder="e.g. shā" value="${esc(clip?.target || "")}"></label><label>Speaker<input id="speaker" value="${esc(clip?.speaker || "me")}"></label><label>Session<input id="session" value="${esc(clip?.session || new Date().toISOString().slice(0, 10))}"></label></div>
  <div class="capture-controls"><button class="record-button" id="record"><span class="dot"></span>Record a new clip</button><div class="meter" aria-hidden="true"><div id="meter-level"></div></div><span class="small" id="record-time">Up to 30 seconds</span><span class="grow"></span><label class="small">Import audio<input id="upload" type="file" accept="audio/*"></label></div><p class="privacy">Stored locally. Echo cancellation, noise suppression, and automatic gain are requested off.</p>
  ${clip ? `<div class="selected-clip"><div class="panel-header" style="margin-bottom:0"><strong>${esc(clip.label)}</strong><span class="badge">${clip.role === "fixture" ? "Synthetic fixture" : esc(clip.role)} · ${clip.duration.toFixed(2)} s</span></div><audio controls src="/api/clips/${clip.id}/audio"></audio><div class="region"><label>Region start (s)<input id="region-start" type="number" min="0" step=".01" value="${clip.region_start || 0}"></label><label>Region end (s)<input id="region-end" type="number" min=".15" step=".01" value="${clip.region_end ?? clip.duration}"></label><label>Role<select id="clip-role">${["attempt", "reference", "anchor", "fixture"].map((r) => `<option ${r === clip.role ? "selected" : ""}>${r}</option>`).join("")}</select></label><button id="save-clip">Save labels & region</button></div>${clip.role === "fixture" ? '<p class="small">Generated test signal. Useful for testing the app; not a native pronunciation example.</p>' : ""}</div>` : '<p class="small" style="margin-top:18px">No clip selected. Record, import audio, or load synthetic examples from the top bar.</p>'}</section>`;
  bindCapture();
}

function bind() {
  $$("[data-goto]").forEach(
    (b) => (b.onclick = () => (location.hash = b.dataset.goto)),
  );
  if ($("#open-library")) $("#open-library").onclick = openLibrary;
  if ($("#run")) $("#run").onclick = runExperiment;
  if ($("#model"))
    $("#model").onchange = () =>
      ($("#layer").value = status.models[$("#model").value].layer);
  if ($("#provider")) {
    $("#provider").onchange = providerNote;
    providerNote();
  }
}
function providerNote() {
  const s = status.providers[$("#provider").value];
  $("#provider-note").textContent = s.configured
    ? "Credentials are configured on the server. Results will be saved locally."
    : `Add ${s.missing.join(", ")} to the local .env file and restart. See .env.example. Credentials are never entered into this page.`;
}
function bindCapture() {
  $("#record").onclick = toggleRecording;
  $("#upload").onchange = async (e) => {
    const file = e.target.files[0];
    if (file) await uploadClip(file);
  };
  $("#open-library").onclick = openLibrary;
  if ($("#save-clip"))
    $("#save-clip").onclick = async () => {
      try {
        await saveClip();
        toast("Labels and analysis region saved");
      } catch (e) {
        error(e.message);
      }
    };
}
function metadata() {
  return {
    label:
      $("#clip-label").value.trim() ||
      "Recording " + new Date().toLocaleTimeString(),
    target: $("#clip-target").value,
    speaker: $("#speaker").value || "me",
    session: $("#session").value || "session-1",
    language: "zh-CN",
    role: "attempt",
  };
}
async function saveClip() {
  const clip = selectedClip();
  if (!clip) return;
  const body = {
    ...metadata(),
    language: clip.language,
    role: $("#clip-role").value,
    notes: clip.notes || "",
    region_start: Number($("#region-start").value),
    region_end: Number($("#region-end").value),
  };
  const saved = await api("/clips/" + clip.id, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  clips = clips.map((c) => (c.id === saved.id ? saved : c));
  return saved;
}
async function uploadClip(blob, meta) {
  try {
    const recordingMeta = meta || metadata();
    if (recordingMeta.speaker === "synthetic-generator")
      recordingMeta.speaker = "me";
    if (recordingMeta.session.startsWith("fixture-"))
      recordingMeta.session = new Date().toISOString().slice(0, 10);
    const form = new FormData();
    form.append("file", blob, blob.name || "recording.webm");
    form.append("metadata", JSON.stringify(recordingMeta));
    const clip = await api("/clips", { method: "POST", body: form });
    clips.unshift(clip);
    active = clip.id;
    localStorage.setItem("speechlab.clip", active);
    render();
    toast("Recording saved locally");
  } catch (e) {
    error(e.message);
  }
}
async function toggleRecording() {
  if (recorder?.state === "recording") {
    recorder.stop();
    return;
  }
  try {
    const meta = metadata();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    const type = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm"].find(
      (t) => MediaRecorder.isTypeSupported(t),
    );
    recorder = new MediaRecorder(stream, type ? { mimeType: type } : {});
    const chunks = [];
    const started = performance.now();
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.onstop = async () => {
      clearInterval(recordingTimer);
      cancelAnimationFrame(frame);
      stream.getTracks().forEach((t) => t.stop());
      await audioContext?.close();
      recorder = null;
      await uploadClip(new Blob(chunks, { type: type || "audio/webm" }), meta);
    };
    audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream),
      analyser = audioContext.createAnalyser();
    source.connect(analyser);
    analyser.fftSize = 256;
    const samples = new Uint8Array(analyser.fftSize);
    const tick = () => {
      analyser.getByteTimeDomainData(samples);
      const rms =
        Math.sqrt(
          samples.reduce((s, v) => s + (v - 128) ** 2, 0) / samples.length,
        ) / 128;
      const meter = $("#meter-level");
      if (meter) meter.style.width = Math.min(100, rms * 500) + "%";
      frame = requestAnimationFrame(tick);
    };
    tick();
    recorder.start();
    $("#record").innerHTML = '<span class="dot"></span>Stop recording';
    $("#record").classList.add("recording");
    recordingTimer = setInterval(() => {
      const seconds = (performance.now() - started) / 1000;
      if ($("#record-time"))
        $("#record-time").textContent = seconds.toFixed(1) + " s";
      if (seconds >= 29.5 && recorder?.state === "recording") recorder.stop();
    }, 100);
  } catch (e) {
    stream?.getTracks().forEach((t) => t.stop());
    error(
      "Microphone unavailable: " +
        e.message +
        ". You can import an audio file instead.",
    );
  }
}

async function openLibrary() {
  clips = await api("/clips");
  $("#library-content").innerHTML = clips.length
    ? clips
        .map(
          (c) =>
            `<div class="library-row ${c.id === active ? "active" : ""}"><div><strong>${esc(c.label)}</strong><p>${esc(c.speaker)} · ${esc(c.session)} · ${c.duration.toFixed(2)} s · ${esc(c.role)}</p></div><button data-choose="${c.id}">Use recording</button></div>`,
        )
        .join("")
    : '<div class="empty">No recordings yet. Record or import a clip from any experiment.</div>';
  $$("[data-choose]").forEach(
    (b) =>
      (b.onclick = () => {
        active = b.dataset.choose;
        localStorage.setItem("speechlab.clip", active);
        $("#library-dialog").close();
        if (currentPage === "home") location.hash = "acoustics";
        else render();
      }),
  );
  $("#library-dialog").showModal();
}

async function runExperiment() {
  if (busy) return;
  try {
    if (selectedClip() && $("#save-clip")) await saveClip();
    const body = { demo: currentPage, clip_id: active };
    if ($("#model")) {
      body.model = $("#model").value;
      body.layer = Number($("#layer").value);
    }
    if (
      [
        "acoustics",
        "articulation",
        "calibration",
        "providers",
        "feedback",
      ].includes(currentPage) &&
      !selectedClip()
    )
      throw new Error("Record or select a clip first.");
    if (currentPage === "acoustics") body.ceiling = Number($("#ceiling").value);
    if (["embeddings", "omni"].includes(currentPage))
      body.clip_ids = values("#compare-clips");
    if (currentPage === "directions")
      Object.assign(body, {
        negative_ids: values("#negative-clips"),
        positive_ids: values("#positive-clips"),
        name: $("#axis-name").value,
        negative_label: $("#negative-label").value,
        positive_label: $("#positive-label").value,
        group_by: $("#group-by").value,
      });
    if (currentPage === "calibration")
      Object.assign(body, {
        reference_ids: values("#reference-clips"),
        anchor_ids: values("#anchor-clips"),
      });
    if (currentPage === "providers")
      Object.assign(body, {
        provider: $("#provider").value,
        reference: $("#reference").value,
        language: $("#provider-language").value,
        cloud_consent: $("#cloud-consent").checked,
      });
    if (currentPage === "feedback")
      Object.assign(body, {
        axis_id: $("#axis").value,
        condition: $("#condition").value,
      });
    busy = true;
    $("#run").disabled = true;
    let job = await post("/analyze", body);
    const page = currentPage;
    while (["queued", "running"].includes(job.status)) {
      if ($("#job")) {
        $("#job").className = "job show running";
        $("#job").textContent = job.message;
      }
      await new Promise((r) => setTimeout(r, 800));
      job = await api("/jobs/" + job.id);
    }
    if (job.status === "error") throw new Error(job.message);
    if (currentPage === page) {
      $("#job").className = "job show";
      $("#job").textContent = "Analysis saved locally";
      renderResult(job.result);
      loadHistory();
    } else toast("Analysis complete. Open its experiment history to view it.");
    axes = await api("/axes");
  } catch (e) {
    error(e.message);
  } finally {
    busy = false;
    if ($("#run")) $("#run").disabled = false;
  }
}
async function loadHistory() {
  const history = await api("/results");
  const kinds = {
    acoustics: "acoustics",
    embeddings: "comparison",
    omni: "comparison",
    directions: "trained-axis",
    articulation: "articulation",
    calibration: "calibration",
    providers: "provider",
    feedback: "feedback",
  };
  const rows = history.filter((r) => r.kind === kinds[currentPage]).slice(0, 8);
  if (!$("#history")) return;
  $("#history").innerHTML = rows.length
    ? `<div class="section-head"><h2>Recent runs</h2><span>Saved on this machine</span></div><div class="panel">${rows.map((r) => `<div class="history-item"><span>${new Date(r.created).toLocaleString()}</span><strong>${esc(r.name || r.model || r.provider || r.kind)}</strong>${r.condition ? `<span>${esc(r.condition)}</span>` : ""}<button data-result="${r.id}">View result</button></div>`).join("")}</div>`
    : "";
  $$("[data-result]").forEach(
    (b) =>
      (b.onclick = async () =>
        renderResult(await api("/results/" + b.dataset.result))),
  );
}

function metric(value, label) {
  return `<div class="metric"><strong>${value == null ? "—" : typeof value === "number" ? value.toFixed(0) : esc(value)}</strong><span>${esc(label)}</span></div>`;
}
function plotPanel(title, id, cls = "", extra = "") {
  return `<div class="panel"><h3>${title}</h3><canvas id="${id}" class="plot ${cls}" role="img" aria-label="${title}"></canvas>${extra}</div>`;
}
function renderResult(r) {
  lastResult = r;
  const container = $("#result");
  if (!container) return;
  const note = `<p class="result-note">${esc(r.note || "")}</p>`;
  if (r.kind === "acoustics") {
    container.innerHTML = `<div class="panel"><div class="panel-header"><h3>Acoustic measurements</h3><span class="badge">Selected region ${(r.end - r.start).toFixed(2)} s</span></div><div class="metric-row">${r.formant_medians.map((v, i) => metric(v, `Median F${i + 1} · Hz`)).join("")}${metric(r.features[3], "Spectral center · Hz")}${metric((100 * r.features[6]).toFixed(0) + "%", "Voiced frames")}</div>${r.warnings.map((w) => `<div class="note warn">${esc(w)}</div>`).join("")}${note}</div>
    ${plotPanel("Waveform · click twice to mark release and voicing onset", "wave", "short", '<div class="region"><label>Release (s)<input type="number" step=".001" id="release"></label><label>Voicing onset (s)<input type="number" step=".001" id="voicing"></label><button id="vot">Measure interval</button><span class="small" id="vot-value">Manual landmarks; inspect the waveform and spectrogram.</span></div>')}
    ${plotPanel("Spectrogram · energy over time", "spectrogram", "large")}
    <div class="two-col">${plotPanel("Vowel space · F1 / F2", "vowel", "", '<p class="small">F2 decreases left to right; F1 increases downward. Each dot is a voiced frame.</p>')}${plotPanel("Formant trajectories", "formants", "", legend(["F1", "F2", "F3"]))}</div><div class="two-col">${plotPanel("Pitch · voiced frames", "pitch")}${plotPanel("Power spectrum", "spectrum")}</div>`;
    lines("wave", r.wave_times, [r.waveform], {
      xLabel: "Time (s)",
      yLabel: "Amplitude",
    });
    spectrogram(r.spectrogram);
    scatter(
      "vowel",
      r.formants.filter((v) => v[0] && v[1]).map((v) => [v[1], v[0]]),
      {
        xLabel: "F2 (Hz)",
        yLabel: "F1 (Hz)",
        xRange: [3000, 400],
        yRange: [150, 1100],
        down: true,
      },
    );
    lines(
      "formants",
      r.times,
      [0, 1, 2].map((i) => r.formants.map((v) => v[i])),
      { xLabel: "Time (s)", yLabel: "Hz" },
    );
    lines("pitch", r.times, [r.pitch], { xLabel: "Time (s)", yLabel: "Hz" });
    lines("spectrum", r.spectrum.hz, [r.spectrum.db], {
      xLabel: "Frequency (Hz)",
      yLabel: "dB",
      xRange: [0, 8000],
    });
    let click = 0;
    $("#wave").onclick = (e) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const frac = Math.max(
        0,
        Math.min(1, (e.clientX - rect.left - 52) / (rect.width - 70)),
      );
      $(click++ % 2 === 0 ? "#release" : "#voicing").value = (
        r.start +
        frac * (r.end - r.start)
      ).toFixed(3);
    };
    $("#vot").onclick = () => {
      const a = Number($("#release").value),
        b = Number($("#voicing").value);
      $("#vot-value").textContent =
        $("#release").value &&
        $("#voicing").value &&
        a >= r.start &&
        b <= r.end &&
        b >= a
          ? `${((b - a) * 1000).toFixed(1)} ms between marked events`
          : "Enter two ordered landmarks within the region.";
    };
  } else if (r.kind === "comparison") {
    container.innerHTML = `<div class="panel"><div class="panel-header"><h3>${esc(r.model)} · layer ${r.layer}</h3><span class="badge">${r.clip_ids.length} clips</span></div>${note}<canvas class="plot large" id="embedding-map" role="img" aria-label="PCA projection of recordings"></canvas><div class="legend">${r.labels.map((l, i) => `<span><i style="background:${colors[i % colors.length]}"></i>${i + 1}. ${esc(l)}</span>`).join("")}</div></div><div class="panel"><h3>Pairwise cosine distance</h3><p class="small">0 = identical pooled representations. There is no universal passing threshold.</p><div class="table-wrap"><table class="distance-table"><thead><tr><th>Recording</th>${r.labels.map((_, i) => `<th>${i + 1}</th>`).join("")}</tr></thead><tbody>${r.cosine_distances.map((row, i) => `<tr><td>${i + 1}. ${esc(r.labels[i])}</td>${row.map((v) => `<td style="background:rgba(36,116,84,${Math.min(0.2, v * 0.5)})">${v.toFixed(3)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></div><div class="panel"><h3>Extraction details</h3><div class="table-wrap"><table><tr><th>Clip</th><th>Frames × dimensions</th><th>Device</th><th>Initial extraction time</th></tr>${r.features.map((f, i) => `<tr><td>${esc(r.labels[i])}</td><td>${f.frames} × ${f.dimensions}</td><td>${esc(f.device)}</td><td>${f.seconds.toFixed(2)} s ${f.cached ? "· cached now" : ""}</td></tr>`).join("")}</table></div></div>`;
    scatter("embedding-map", r.points, {
      xLabel: "Principal component 1",
      yLabel: "Principal component 2",
      numbered: true,
    });
  } else if (r.kind === "trained-axis") {
    const v = r.validation;
    container.innerHTML = `<div class="panel"><span class="eyebrow">DIRECTION SAVED</span><h2>${esc(r.name)}</h2><div class="metric-row">${metric(r.training_ids.length, "Training examples")}${metric(r.groups, "Independent " + r.group_by + " groups")}${metric(v.balanced_accuracy == null ? "Unavailable" : (v.balanced_accuracy * 100).toFixed(1) + "%", "Held-out balanced accuracy")}</div><div class="axis-labels"><span>${esc(r.negative_label)}</span><span>${esc(r.positive_label)}</span></div><div class="axis-track"></div>${note}${v.balanced_accuracy == null ? '<div class="note warn">Collect both labels across multiple sessions or speakers before relying on this direction.</div>' : ""}<button class="primary" id="practice">Try a new recording →</button></div>`;
    $("#practice").onclick = () => (location.hash = "feedback");
  } else if (r.kind === "articulation") {
    container.innerHTML = `<div class="panel"><h3>Estimated articulator trajectories</h3>${note}<div class="fields"><label>Articulator<select id="articulator">${["Tongue dorsum", "Tongue blade", "Tongue tip", "Lower incisor", "Upper lip", "Lower lip"].map((v, i) => `<option value="${i}">${v}</option>`).join("")}</select></label></div><div class="two-col"><div><canvas id="art-time" class="plot large" role="img" aria-label="Estimated articulator coordinates over time"></canvas>${legend(["X", "Y"])}</div><div><canvas id="art-xy" class="plot large" role="img" aria-label="Estimated articulator path"></canvas><p class="small">Trajectory in reference coordinates; not an anatomical mouth drawing.</p></div></div></div>`;
    const draw = () => {
      const i = Number($("#articulator").value) * 2;
      lines(
        "art-time",
        r.times,
        [r.trajectories.map((v) => v[i]), r.trajectories.map((v) => v[i + 1])],
        { xLabel: "Time (s)", yLabel: "Reference units" },
      );
      scatter(
        "art-xy",
        r.trajectories.map((v) => [v[i], v[i + 1]]),
        {
          xLabel: "X · reference units",
          yLabel: "Y · reference units",
          connected: true,
        },
      );
    };
    $("#articulator").onchange = draw;
    draw();
  } else if (r.kind === "calibration") {
    container.innerHTML = `<div class="panel"><h3>Two reference comparisons</h3><div class="metric-row">${metric(r.reference_distance.toFixed(4), "Distance to population centroid")}${metric(r.personal_distance.toFixed(4), "Distance to your accepted examples")}</div>${note}<p class="small">${r.reference_ids.length} population references · ${r.anchor_ids.length} personal anchors. Test whether these distances agree with independent human judgments.</p></div>`;
  } else if (r.kind === "provider") {
    container.innerHTML = `<div class="panel"><h3>${esc(r.provider)} response</h3>${note}<pre>${esc(r.response.text || JSON.stringify(r.response, null, 2))}</pre></div>`;
  } else if (r.kind === "feedback" || r.kind === "direction") {
    if (r.condition === "hidden")
      container.innerHTML = `<div class="panel"><h3>Retention attempt saved</h3><div class="hidden-feedback"><h2>Feedback is hidden</h2><p>Make your next attempt using what you learned. The result is saved for later review.</p><button id="reveal">Reveal after practice</button></div></div>`;
    else drawFeedback(r);
    if ($("#reveal")) $("#reveal").onclick = () => drawFeedback(r);
  }
  if (r.id) {
    const link = document.createElement("a");
    link.className = "reading-link";
    link.href = "/api/results/" + r.id;
    link.target = "_blank";
    link.textContent = "Open saved result JSON ↗";
    container.append(link);
  }
}
function drawFeedback(r) {
  $("#result").innerHTML =
    `<div class="panel"><div class="panel-header"><h3>${esc(r.name)}</h3><span class="badge">${esc(r.condition || "visual")} trial</span></div><div class="big-position">${r.position < 0.5 ? esc(r.negative_label) : esc(r.positive_label)}</div><div class="axis-track"><i class="needle" style="left:${r.position * 100}%"></i></div><div class="axis-labels"><span>${esc(r.negative_label)}</span><span>${esc(r.positive_label)}</span></div><p class="result-note">${esc(r.note)}</p>${r.in_training ? '<div class="note warn">This recording was used to fit the direction. Record a new attempt for a meaningful check.</div>' : ""}<p class="small">Make one small adjustment and record again. The display does not prescribe a tongue movement.</p></div>`;
}
function legend(labels) {
  return `<div class="legend">${labels.map((l, i) => `<span><i style="background:${colors[i]}"></i>${l}</span>`).join("")}</div>`;
}

function setupCanvas(id) {
  const c = $("#" + id),
    rect = c.getBoundingClientRect(),
    dpr = devicePixelRatio || 1;
  c.width = rect.width * dpr;
  c.height = rect.height * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  return {
    ctx,
    w: rect.width,
    h: rect.height,
    p: { l: 52, r: 18, t: 25, b: 38 },
  };
}
function axesPlot(plot, xmin, xmax, ymin, ymax, opt = {}) {
  const { ctx, w, h, p } = plot;
  const X = (x) => p.l + ((x - xmin) / (xmax - xmin || 1)) * (w - p.l - p.r),
    Y = (y) =>
      opt.down
        ? p.t + ((y - ymin) / (ymax - ymin || 1)) * (h - p.t - p.b)
        : h - p.b - ((y - ymin) / (ymax - ymin || 1)) * (h - p.t - p.b);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "10px system-ui";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    let y = p.t + ((h - p.t - p.b) * i) / 4;
    ctx.strokeStyle = "#e7ece3";
    ctx.beginPath();
    ctx.moveTo(p.l, y);
    ctx.lineTo(w - p.r, y);
    ctx.stroke();
    ctx.fillStyle = "#7e8b79";
    ctx.textAlign = "right";
    const v = opt.down
      ? ymin + ((ymax - ymin) * i) / 4
      : ymax - ((ymax - ymin) * i) / 4;
    ctx.fillText(
      Math.abs(v) > 100 ? v.toFixed(0) : v.toFixed(1),
      p.l - 8,
      y + 3,
    );
  }
  for (let i = 0; i <= 4; i++) {
    const x = p.l + ((w - p.l - p.r) * i) / 4;
    const v = xmin + ((xmax - xmin) * i) / 4;
    ctx.textAlign = "center";
    ctx.fillStyle = "#7e8b79";
    ctx.fillText(
      Math.abs(v) > 100 ? v.toFixed(0) : v.toFixed(2),
      x,
      h - p.b + 17,
    );
  }
  ctx.fillStyle = "#6d7d65";
  ctx.textAlign = "center";
  ctx.fillText(opt.xLabel || "", (p.l + w - p.r) / 2, h - 3);
  ctx.textAlign = "left";
  ctx.fillText(opt.yLabel || "", p.l, 12);
  return { X, Y };
}
function lines(id, x, series, opt = {}) {
  const plot = setupCanvas(id),
    { ctx, w, h, p } = plot;
  const vals = series.flat().filter((v) => v != null && Number.isFinite(v));
  if (!vals.length) return;
  let xmin = opt.xRange?.[0] ?? Math.min(...x),
    xmax = opt.xRange?.[1] ?? Math.max(...x);
  let ymin = Math.min(...vals),
    ymax = Math.max(...vals);
  const pad = (ymax - ymin) * 0.08 || 1;
  ymin -= pad;
  ymax += pad;
  const { X, Y } = axesPlot(plot, xmin, xmax, ymin, ymax, opt);
  ctx.save();
  ctx.beginPath();
  ctx.rect(p.l, p.t, w - p.l - p.r, h - p.t - p.b);
  ctx.clip();
  series.forEach((s, i) => {
    ctx.strokeStyle = colors[i % colors.length];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let pen = false;
    s.forEach((v, j) => {
      if (v == null) {
        pen = false;
        return;
      }
      if (pen) ctx.lineTo(X(x[j]), Y(v));
      else {
        ctx.moveTo(X(x[j]), Y(v));
        pen = true;
      }
    });
    ctx.stroke();
  });
  ctx.restore();
}
function scatter(id, points, opt = {}) {
  const plot = setupCanvas(id),
    { ctx } = plot;
  if (!points.length) {
    ctx.fillText("No valid voiced frames", 55, 65);
    return;
  }
  const xs = points.map((p) => p[0]),
    ys = points.map((p) => p[1]);
  let xmin = Math.min(...xs),
    xmax = Math.max(...xs),
    ymin = Math.min(...ys),
    ymax = Math.max(...ys);
  const xp = (xmax - xmin) * 0.18 || 1,
    yp = (ymax - ymin) * 0.18 || 1;
  const { X, Y } = axesPlot(
    plot,
    opt.xRange?.[0] ?? xmin - xp,
    opt.xRange?.[1] ?? xmax + xp,
    opt.yRange?.[0] ?? ymin - yp,
    opt.yRange?.[1] ?? ymax + yp,
    opt,
  );
  if (opt.connected) {
    ctx.strokeStyle = "#99bfa8";
    ctx.beginPath();
    points.forEach((p, i) =>
      i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])),
    );
    ctx.stroke();
  }
  points.forEach((p, i) => {
    ctx.fillStyle = opt.numbered ? colors[i % colors.length] : "#28745488";
    ctx.beginPath();
    ctx.arc(X(p[0]), Y(p[1]), opt.numbered ? 7 : 2.5, 0, Math.PI * 2);
    ctx.fill();
    if (opt.numbered) {
      ctx.fillStyle = "#29432f";
      ctx.font = "11px system-ui";
      ctx.fillText(i + 1, X(p[0]) + 11, Y(p[1]) + 4);
    }
  });
}
function spectrogram(s) {
  const plot = setupCanvas("spectrogram"),
    { ctx, w, h, p } = plot;
  if (!s.times.length) return;
  axesPlot(plot, s.times[0], s.times.at(-1), 0, s.hz.at(-1), {
    xLabel: "Time (s)",
    yLabel: "Frequency (Hz)",
  });
  const off = document.createElement("canvas");
  off.width = s.times.length;
  off.height = s.hz.length;
  const oc = off.getContext("2d"),
    image = oc.createImageData(off.width, off.height);
  for (let y = 0; y < off.height; y++)
    for (let x = 0; x < off.width; x++) {
      const v = Math.max(
        0,
        Math.min(1, (s.db[off.height - 1 - y][x] + 75) / 75),
      );
      const i = (y * off.width + x) * 4;
      image.data[i] = Math.round(244 - v * 205);
      image.data[i + 1] = Math.round(248 - v * 129);
      image.data[i + 2] = Math.round(235 - v * 155);
      image.data[i + 3] = 255;
    }
  oc.putImageData(image, 0, 0);
  ctx.drawImage(off, p.l, p.t, w - p.l - p.r, h - p.t - p.b);
}

$("#close-library").onclick = () => $("#library-dialog").close();
$("#seed").onclick = async () => {
  try {
    await post("/fixtures", {});
    clips = await api("/clips");
    if (!active) {
      active = clips[0]?.id;
      localStorage.setItem("speechlab.clip", active);
    }
    render();
    toast(
      "12 synthetic test clips are ready. These are not native references.",
    );
  } catch (e) {
    error(e.message);
  }
};
window.addEventListener("hashchange", render);
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (lastResult && $("#result")) renderResult(lastResult);
  }, 200);
});
try {
  [status, clips, axes] = await Promise.all([
    api("/status"),
    api("/clips"),
    api("/axes"),
  ]);
  if (!clips.some((c) => c.id === active)) active = clips[0]?.id;
  $("#device").textContent =
    status.device.toUpperCase() + " · Local worker ready";
  render();
} catch (e) {
  $("#view").innerHTML =
    `<div class="panel" style="margin-top:30px"><h2>Could not connect</h2><p>${esc(e.message)}</p><p>Start the server with ./start.sh and refresh.</p></div>`;
}
