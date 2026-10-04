const $ = (s) => document.querySelector(s);
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
let catalog,
  current,
  recorder,
  stream,
  timer,
  busy = false,
  player,
  playbackURL;
let exampleIndices = [0, 0],
  history = [];

async function api(path, options) {
  const r = await fetch(path, options);
  const body = await r.json();
  if (!r.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "Could not complete that request.",
    );
  return body;
}
function notice(text) {
  $("#notice").textContent = text;
  $("#notice").style.display = "block";
  setTimeout(() => ($("#notice").style.display = "none"), 4500);
}
function status(text, error = false) {
  $("#status").textContent = text;
  $("#status").classList.toggle("error", error);
}
function lock(value) {
  busy = value;
  document.body.classList.toggle("busy", value);
  document
    .querySelectorAll(".pair-card,.voice button,.another,#upload-button")
    .forEach((b) => (b.disabled = value));
  $("#record").disabled = value;
}
function play(url) {
  if (player) player.pause();
  player = new Audio(url);
  player
    .play()
    .catch(() =>
      notice("Audio playback was blocked. Try pressing play again."),
    );
}
function exampleURL(side, kind = "word") {
  return `/api/guided/${current.id}/example/${side}/${exampleIndices[side]}/${kind}`;
}
function voices() {
  $("#voices").innerHTML = current.labels
    .map((label, side) => {
      const example = current.examples[side][exampleIndices[side]];
      return `<div><div class="voice"><div class="voice-main"><button class="play" data-word="${side}" aria-label="Play ${esc(example.word)}, the ${esc(label)} example"><span class="sound-label">${esc(label)}</span><span class="word">in “${esc(example.word)}”</span><strong>Hear the word <span aria-hidden="true">▶</span></strong></button></div><div class="voice-actions"><button class="text-button" data-phone="${side}">Just the sound</button><button class="text-button" data-analyze="${side}">See its result</button></div></div><button class="text-button another" data-next="${side}">Another voice ↻</button></div>`;
    })
    .join("");
  document
    .querySelectorAll("[data-word]")
    .forEach((b) => (b.onclick = () => play(exampleURL(+b.dataset.word))));
  document
    .querySelectorAll("[data-phone]")
    .forEach(
      (b) => (b.onclick = () => play(exampleURL(+b.dataset.phone, "phone"))),
    );
  document.querySelectorAll("[data-analyze]").forEach(
    (b) =>
      (b.onclick = () =>
        analyze({
          example_side: +b.dataset.analyze,
          example_index: exampleIndices[+b.dataset.analyze],
        })),
  );
  document.querySelectorAll("[data-next]").forEach(
    (b) =>
      (b.onclick = () => {
        const side = +b.dataset.next;
        exampleIndices[side] =
          (exampleIndices[side] + 1) % current.examples[side].length;
        voices();
      }),
  );
}
function choose(id) {
  if (busy) return;
  current = catalog.contrasts.find((c) => c.id === id);
  exampleIndices = [0, 0];
  history = [];
  if (player) player.pause();
  if (playbackURL) {
    URL.revokeObjectURL(playbackURL);
    playbackURL = null;
  }
  document
    .querySelectorAll(".pair-card")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.id === id)),
    );
  $("#practice").hidden = false;
  $("#practice").className = "practice";
  $("#practice").innerHTML =
    `<div class="practice-head"><h2 id="practice-heading">${esc(current.name)}</h2><span class="tag">ENGLISH / ${current.category}</span></div><div class="practice-grid"><div><p class="step-label"><b>1</b> HEAR THE DIFFERENCE</p><div class="voice-pair" id="voices"></div><p class="source-note">Real audiobook speech, with the target sound cropped out for comparison. “Another voice” switches speakers.</p></div><div><p class="step-label"><b>2</b> TRY A SOUND</p><p class="instruction">${esc(current.instruction)}</p><div class="record-row"><button class="record" id="record"><span class="dot" aria-hidden="true"></span><span>Record a sound</span></button><button class="text-button" id="upload-button">Use an audio file</button><input type="file" id="upload" accept="audio/*" hidden></div><p class="status" id="status" role="status" aria-live="polite">Recording stops after 2.5 seconds. Audio stays on this Mac.</p><div id="result" class="result empty" aria-live="polite"><span class="empty-icon" aria-hidden="true">∿∿</span><p>Your result will appear here.<br>Or try “See its result” on an example.</p></div><details id="history" class="history" hidden><summary>Earlier attempts this visit</summary><ol id="attempts"></ol></details></div></div>`;
  voices();
  $("#record").onclick = record;
  $("#upload-button").onclick = () => $("#upload").click();
  $("#upload").onchange = async (e) => {
    const file = e.target.files[0];
    if (file) await upload(file);
    e.target.value = "";
  };
}
async function record() {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  if (busy) return;
  lock(true);
  if (player) player.pause();
  try {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error(
        "Microphone access requires localhost. Open this app at http://127.0.0.1:8767.",
      );
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    const type = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm"].find(
      (t) => MediaRecorder.isTypeSupported(t),
    );
    recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
    const parts = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size) parts.push(e.data);
    };
    recorder.onerror = () => {
      clearTimeout(timer);
      stream.getTracks().forEach((t) => t.stop());
      lock(false);
      status("The microphone stopped unexpectedly. Try again.", true);
    };
    recorder.onstop = async () => {
      clearTimeout(timer);
      stream.getTracks().forEach((t) => t.stop());
      $("#record").classList.remove("is-recording");
      $("#record").innerHTML =
        '<span class="dot" aria-hidden="true"></span><span>Record again</span>';
      await upload(new Blob(parts, { type: recorder.mimeType }));
    };
    recorder.start();
    $("#record").disabled = false;
    $("#record").classList.add("is-recording");
    $("#record").innerHTML =
      '<span class="dot" aria-hidden="true"></span><span>Stop recording</span>';
    status("Recording… hold just the sound now.");
    timer = setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, 2500);
  } catch (e) {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    lock(false);
    status(
      e.name === "NotAllowedError"
        ? "Allow microphone access in your browser, then try again."
        : e.message,
      true,
    );
  }
}
async function upload(blob) {
  lock(true);
  status("Preparing your recording…");
  try {
    const data = new FormData();
    data.append(
      "file",
      blob,
      blob.name || (blob.type.includes("mp4") ? "sound.m4a" : "sound.webm"),
    );
    data.append(
      "metadata",
      JSON.stringify({
        label: `${current.name} · guided practice`,
        language: "en-US",
        speaker: "me",
        session: "guided-" + new Date().toISOString().slice(0, 10),
        role: "attempt",
      }),
    );
    const clip = await api("/api/clips", { method: "POST", body: data });
    if (playbackURL) URL.revokeObjectURL(playbackURL);
    playbackURL = URL.createObjectURL(blob);
    await analyze({ clip_id: clip.id });
  } catch (e) {
    status(e.message, true);
    lock(false);
  }
}
async function analyze(input) {
  lock(true);
  status(
    "Listening… the first result may take a few seconds while the model loads.",
  );
  try {
    let job = await api("/api/guided/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contrast: current.id, ...input }),
    });
    const deadline = Date.now() + 300000;
    while (job.status === "queued" || job.status === "running") {
      if (Date.now() > deadline)
        throw new Error(
          "The model is taking longer than expected. Please try again in a moment.",
        );
      await new Promise((resolve) => setTimeout(resolve, 450));
      job = await api("/api/jobs/" + job.id);
    }
    if (job.status !== "done")
      throw new Error(job.message.replace(/^\w+Error: /, ""));
    show(job.result);
    status(
      input.clip_id
        ? "Saved locally. Try the other sound and watch the marker change."
        : "Real speech example · analyzed with the same fitted contrast.",
    );
  } catch (e) {
    status(e.message, true);
  } finally {
    lock(false);
  }
}
function show(r) {
  const example = r.example !== null;
  const title = r.title;
  const detail = example
    ? r.detail.replace("Your sound", "This example")
    : r.detail;
  $("#result").className = "result";
  const meter =
    r.position === null
      ? ""
      : `<div class="meter" role="img" aria-label="${esc(title)}; position on the ${esc(current.labels[0])} to ${esc(current.labels[1])} contrast">${r.windows.map((w) => `<span class="window-dot" style="left:${w.position * 100}%"></span>`).join("")}<span class="needle" style="left:${r.position * 100}%"></span></div><div class="meter-labels"><span>${esc(current.labels[0])}-like</span><span>${esc(current.labels[1])}-like</span></div><p class="meter-caption">${example ? "One timestamped sound" : "Dots = short slices of your recording"} · position, not percent correct</p>`;
  const obs = r.observation;
  $("#result").innerHTML =
    `<p class="eyebrow">${example ? "REAL SPEECH EXAMPLE" : "YOUR SOUND"}</p><h3>${esc(title)}</h3><p class="result-detail">${esc(detail)}</p>${meter}${obs ? `<div class="observation"><strong>${esc(obs.label)}: ${obs.value} ${esc(obs.unit)}</strong><p>Reference medians: S ${obs.reference_medians[0]} kHz · SH ${obs.reference_medians[1]} kHz.</p><p>${esc(obs.description)}</p></div>` : ""}${!example && playbackURL ? '<div class="playback"><button id="replay" class="text-button">▶ Hear my recording</button><a class="text-button" href="/evidence.html">How to read this</a></div>' : ""}`;
  if ($("#replay")) $("#replay").onclick = () => play(playbackURL);
  if (!example) {
    history.unshift({
      title: r.title,
      time: new Date().toLocaleTimeString([], {
        hour: "numeric",
        minute: "2-digit",
      }),
    });
    $("#history").hidden = history.length < 2;
    $("#attempts").innerHTML = history
      .slice(1, 6)
      .map(
        (h) =>
          `<li>${esc(h.title)} <span class="word">${esc(h.time)}</span></li>`,
      )
      .join("");
  }
}
try {
  catalog = await api("/api/guided");
  if (!catalog.contrasts.length)
    throw new Error(
      "No contrasts passed the benchmark gates. See the research results.",
    );
  $("#pairs").innerHTML = catalog.contrasts
    .map(
      (c) =>
        `<button class="pair-card" data-id="${esc(c.id)}" aria-pressed="false"><span class="eyebrow">${esc(c.category)}</span><span class="pair-title">${esc(c.name)}</span><span class="pair-description">${esc(c.description)}</span><span class="pair-arrow" aria-hidden="true">↗</span></button>`,
    )
    .join("");
  document
    .querySelectorAll(".pair-card")
    .forEach((b) => (b.onclick = () => choose(b.dataset.id)));
  $("#study-summary").textContent =
    `${catalog.tokens.toLocaleString()} sound segments, 80 speakers, and six audio encoders compared. These contrasts passed checks on ${catalog.test_speakers} held-out speakers, added noise, and small timestamp shifts. The test measures English sound discrimination; it does not establish Mandarin coaching or learner improvement.`;
  choose(catalog.contrasts[0].id);
} catch (e) {
  $("#pairs").textContent = e.message;
}
window.addEventListener("pagehide", () => {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  clearTimeout(timer);
});
