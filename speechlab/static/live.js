const $ = (id) => document.getElementById(id);
const canvas = $("plot"),
  ctx = canvas.getContext("2d");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let catalog,
  map,
  running = false,
  starting = false,
  generation = 0;
let stream,
  context,
  node,
  source,
  mute,
  request,
  busy = false,
  player;
let target = null,
  displayed = null,
  trail = [],
  lastPoint = 0,
  lastStatus = "",
  reference = false;
let width = 500,
  height = 500,
  frameCount = 0,
  intervalStart = 0,
  lastRate = "";

function status(message, detail) {
  if (message !== lastStatus) {
    $("status").textContent = message;
    lastStatus = message;
  }
  if (detail !== undefined) $("detail").textContent = detail;
}
function clearPoint() {
  target = null;
  displayed = null;
  trail = [];
  lastPoint = 0;
}
function transform(point) {
  const size = Math.min(width, height) - 55,
    scale = size / (2 * map.extent);
  return [width / 2 + point[0] * scale, height / 2 - point[1] * scale];
}
function draw(now) {
  requestAnimationFrame(draw);
  ctx.clearRect(0, 0, width, height);
  if (!map) return;
  const size = Math.min(width, height) - 55,
    left = (width - size) / 2,
    top = (height - size) / 2;
  ctx.strokeStyle = "#eaece4";
  ctx.lineWidth = 1;
  for (let n = 0; n <= 8; n++) {
    const v = (n / 8) * size;
    ctx.beginPath();
    ctx.moveTo(left + v, top);
    ctx.lineTo(left + v, top + size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(left, top + v);
    ctx.lineTo(left + size, top + v);
    ctx.stroke();
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, size, size);
  ctx.clip();
  for (const c of map.categories) {
    const [x, y] = transform(c.center),
      cov = c.covariance;
    const delta = Math.sqrt((cov[0][0] - cov[1][1]) ** 2 + 4 * cov[0][1] ** 2);
    const angle = -0.5 * Math.atan2(2 * cov[0][1], cov[0][0] - cov[1][1]);
    const scale = size / (2 * map.extent);
    const major =
      Math.sqrt(Math.max(0, (cov[0][0] + cov[1][1] + delta) / 2) * 3.219) *
      scale;
    const minor =
      Math.sqrt(Math.max(0, (cov[0][0] + cov[1][1] - delta) / 2) * 3.219) *
      scale;
    ctx.beginPath();
    ctx.ellipse(x, y, major, minor, angle, 0, Math.PI * 2);
    ctx.fillStyle = c.color + "17";
    ctx.fill();
    ctx.strokeStyle = c.color + "65";
    ctx.stroke();
    ctx.fillStyle = c.color + "45";
    for (const p of c.points) {
      const [px, py] = transform(p);
      ctx.beginPath();
      ctx.arc(px, py, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  trail = trail.filter((p) => now - p.time < 1600);
  for (let n = 1; n < trail.length; n++) {
    const a = transform(trail[n - 1].point),
      b = transform(trail[n].point);
    ctx.globalAlpha = Math.max(0, 1 - (now - trail[n].time) / 1600) * 0.5;
    ctx.strokeStyle = "#b97922";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.restore();
  for (const c of map.categories) {
    const [x, y] = transform(c.center);
    ctx.font = "600 19px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 5;
    ctx.strokeStyle = "#fffefa";
    ctx.strokeText(c.label, x, y);
    ctx.fillStyle = c.color;
    ctx.fillText(c.label, x, y);
  }
  if (target && (reference || now - lastPoint < 700)) {
    if (!displayed || reducedMotion) displayed = [...target];
    else displayed = displayed.map((v, i) => v + (target[i] - v) * 0.28);
    const [rawX, rawY] = transform(displayed);
    const x = Math.max(left, Math.min(left + size, rawX)),
      y = Math.max(top, Math.min(top + size, rawY));
    ctx.globalAlpha = reference
      ? 1
      : Math.min(1, (700 - (now - lastPoint)) / 200);
    ctx.beginPath();
    ctx.arc(x, y, 14, 0, Math.PI * 2);
    ctx.fillStyle = "#bc853120";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = reference ? "#fffefa" : "#b97922";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#b97922";
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}
new ResizeObserver(() => {
  const rect = canvas.getBoundingClientRect(),
    ratio = devicePixelRatio || 1;
  width = rect.width;
  height = rect.height;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}).observe(canvas);
requestAnimationFrame(draw);

function renderMap(ident) {
  stop();
  map = catalog.maps.find((m) => m.id === ident);
  clearPoint();
  $("map-title").textContent = "Mandarin voices";
  $("map-note").textContent =
    "Each word contains the labeled sound. Play a voice to see its reference dot; ↻ changes the speaker.";
  const container = $("references");
  container.replaceChildren();
  for (const c of map.categories) {
    let index = 0;
    const row = document.createElement("div");
    row.className = "reference";
    const phone = document.createElement("span");
    phone.className = "phone";
    phone.textContent = c.label;
    phone.style.color = c.color;
    phone.style.background = c.color + "18";
    const play = document.createElement("button");
    const next = document.createElement("button");
    next.className = "next";
    next.textContent = "↻";
    next.title = `Another ${c.label} reference`;
    next.setAttribute("aria-label", next.title);
    function label() {
      const e = c.examples[index];
      play.replaceChildren(document.createTextNode(`▷ ${e.word}`));
      const speaker = document.createElement("span");
      speaker.className = "speaker";
      speaker.textContent = `${index + 1} / ${c.examples.length}`;
      play.append(speaker);
      play.setAttribute(
        "aria-label",
        `Play ${c.label} in ${e.word}, voice ${index + 1}`,
      );
    }
    label();
    next.onclick = () => {
      index = (index + 1) % c.examples.length;
      label();
    };
    play.onclick = async () => {
      stop();
      const playbackGeneration = generation;
      const e = c.examples[index];
      reference = true;
      target = [...e.point];
      lastPoint = performance.now();
      $("dot-label").textContent = `Reference · ${c.label}`;
      status(
        `${e.word} · ${c.label} · Mandarin reference`,
        "Hollow dot marks the measured sound within this word.",
      );
      player = new Audio(
        `/api/live/${map.id}/example/${encodeURIComponent(c.phone)}/${index}`,
      );
      try {
        await player.play();
      } catch {
        if (playbackGeneration === generation)
          status("Could not play this reference. Try again.");
      }
    };
    row.append(phone, play, next);
    container.append(row);
  }
  const t = map.test;
  $("evidence").textContent =
    `This ${map.name.toLowerCase()} map uses ${map.model}. On ${t.tokens} labeled sounds from ${t.speakers} speakers held out from fitting and model selection, its nearest-center balanced accuracy was ${(t.balanced_accuracy * 100).toFixed(1)}%. This checks separation of native speech; it does not establish how well the map diagnoses learner errors. Regions are covariance ellipses fitted to the development voices (an 80% contour under a Gaussian approximation).`;
  $("listen").disabled = false;
  status("Ready when you are.", "Audio stays on this Mac and is not saved.");
}

function stop() {
  generation++;
  running = false;
  starting = false;
  request?.abort();
  request = null;
  busy = false;
  node?.disconnect();
  source?.disconnect();
  mute?.disconnect();
  node = source = mute = null;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  if (context) {
    context.close().catch(() => {});
    context = null;
  }
  player?.pause();
  player = null;
  reference = false;
  clearPoint();
  $("listen").textContent = "● Start microphone";
  $("listen").dataset.live = "false";
  $("listen").disabled = !map;
  $("level").style.width = "0%";
  $("dot-label").textContent = "Your sound";
}
async function send(buffer, token) {
  if (!running || token !== generation) return;
  const samples = new Float32Array(buffer);
  let sum = 0;
  for (const s of samples) sum += s * s;
  $("level").style.width =
    `${Math.min(100, Math.sqrt(sum / samples.length) * 800)}%`;
  if (busy) return; // Discard this frame; never queue old microphone windows.
  busy = true;
  const sentAt = performance.now();
  const controller = new AbortController();
  request = controller;
  const deadline = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(`/api/live/${map.id}/frame`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: buffer,
      signal: controller.signal,
    });
    const result = await response.json();
    if (token !== generation) return;
    if (!response.ok)
      throw new Error(result.detail || "Could not measure this sound.");
    if (performance.now() - sentAt > 600) {
      clearPoint();
      status(
        "Model is warming up · keep holding the sound.",
        "Waiting for a fresh measurement.",
      );
      return;
    }
    if (result.active) {
      const now = performance.now();
      reference = false;
      target = [result.x, result.y];
      lastPoint = now;
      trail.push({ point: [...target], time: now });
      frameCount++;
      if (now - intervalStart > 1500) {
        lastRate = `${((frameCount * 1000) / (now - intervalStart)).toFixed(1)} updates/s`;
        frameCount = 0;
        intervalStart = now;
      }
      const outside =
        Math.max(Math.abs(result.x), Math.abs(result.y)) > map.extent;
      status(
        outside
          ? "Sound is outside the displayed reference area."
          : "Listening · follow the amber dot",
        `${lastRate || "Warming up"} · 160 ms window · audio is not saved`,
      );
    } else {
      clearPoint();
      status(
        {
          quiet: "Listening · waiting for a sound",
          clipping: "Input is clipping · reduce the microphone level",
        }[result.reason] || "Listening…",
      );
    }
  } catch (error) {
    if (token !== generation) return;
    stop();
    status(
      error.name === "AbortError"
        ? "The local model took too long. Start again to retry."
        : error.message,
      "The microphone has stopped.",
    );
  } finally {
    clearTimeout(deadline);
    if (token === generation) {
      busy = false;
      request = null;
    }
  }
}
async function start() {
  stop();
  starting = true;
  const token = generation;
  $("listen").disabled = true;
  status("Opening microphone…", "Allow microphone access in your browser.");
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    if (token !== generation) {
      acquired.getTracks().forEach((t) => t.stop());
      return;
    }
    stream = acquired;
    context = new AudioContext({ sampleRate: 16000 });
    if (context.sampleRate !== 16000)
      throw new Error(
        "This browser cannot capture at 16 kHz. Please use Chrome.",
      );
    const audioContext = context;
    await audioContext.audioWorklet.addModule("/capture-worklet.js");
    if (token !== generation) return;
    await audioContext.resume();
    if (token !== generation) return;
    source = audioContext.createMediaStreamSource(stream);
    node = new AudioWorkletNode(audioContext, "sound-window");
    mute = audioContext.createGain();
    mute.gain.value = 0;
    node.port.onmessage = (e) => send(e.data, token);
    source.connect(node).connect(mute).connect(audioContext.destination);
    running = true;
    starting = false;
    frameCount = 0;
    intervalStart = performance.now();
    lastRate = "";
    $("listen").disabled = false;
    $("listen").textContent = "■ Stop microphone";
    $("listen").dataset.live = "true";
    status(
      "Listening · loading the sound model…",
      "The first sound may take a moment. Keep holding it.",
    );
    stream.getTracks()[0].onended = () => {
      if (running && token === generation) {
        stop();
        status("Microphone disconnected.");
      }
    };
  } catch (error) {
    if (token !== generation) return;
    stop();
    status(
      error.name === "NotAllowedError"
        ? "Microphone access was denied. Allow it in your browser to continue."
        : error.message,
      "Reference recordings are still available.",
    );
  }
}
$("listen").onclick = () => {
  if (running || starting) {
    stop();
    status("Microphone stopped.", "Audio stays on this Mac and is not saved.");
  } else start();
};
window.addEventListener("pagehide", stop);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && (running || starting)) {
    stop();
    status("Microphone paused while this page was hidden.");
  }
});
try {
  const response = await fetch("/api/live");
  if (!response.ok)
    throw new Error(
      "Could not load the Mandarin maps. Check the local server.",
    );
  catalog = await response.json();
  renderMap("mandarin");
} catch (error) {
  status(error.message);
}
