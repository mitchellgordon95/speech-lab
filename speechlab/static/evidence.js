const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const pct = (n) => (100 * n).toFixed(1) + "%";
const labels = {
  s_sh: "S / SH",
  r_l: "R / L",
  f_th: "F / TH",
  b_p: "B / P",
  iy_ih: "EE / IH",
  ae_eh: "A / EH",
  uw_uh: "OO / UH",
};
try {
  const response = await fetch("/api/guided");
  if (!response.ok) throw new Error("Could not load the completed study.");
  const data = await response.json();
  document.querySelector("#results").innerHTML =
    `<h2>The selected models, on the final test</h2><p>Balanced accuracy averages the success rate for each of the two sounds, so an uneven number of examples cannot inflate it. Chance is 50%. Intervals resample whole speakers. A contrast passes only if it meets the preset accuracy and robustness gates.</p><div class="table-wrap"><table><thead><tr><th>Sounds</th><th>Selected using validation</th><th>Test accuracy</th><th>95% interval</th><th>Clips</th><th>All gates</th></tr></thead><tbody>${data.results.map((r) => `<tr><td>${labels[r.contrast]}</td><td>${esc(r.model)}</td><td>${pct(r.test.balanced_accuracy)}</td><td>${r.test.speaker_bootstrap_95_ci.map(pct).join("–")}</td><td>${r.test.n}</td><td class="${r.eligible ? "pass" : ""}">${r.eligible ? "Passed" : "Withheld"}</td></tr>`).join("")}</tbody></table></div><p>Gates: ≥85% validation and final-test accuracy, final interval lower bound ≥80%, and ≥80% on a fixed subset with 20 dB noise, ±20 ms boundary shifts, and a central crop of at most 160 ms. Passing is evidence for corpus sound discrimination, not validated pronunciation coaching.</p>
  <details><summary>Compare all representations</summary><p>Each row uses the same linear-classifier recipe. Test results below were not used to select winners or replacements.</p><div class="table-wrap"><table><thead><tr><th>Representation</th>${data.results.map((r) => `<th>${labels[r.contrast]}</th>`).join("")}</tr></thead><tbody>${Object.keys(
    data.results[0].representations,
  )
    .map(
      (name) =>
        `<tr><td>${esc(name)}</td>${data.results.map((r) => `<td>${pct(r.representations[name].balanced_accuracy)}</td>`).join("")}</tr>`,
    )
    .join("")}</tbody></table></div></details>
  <details><summary>Did raw distance or personal references work?</summary><p>These use the selected representation for each contrast. Personal references require two known-correct examples per category from that speaker, and queries from different utterances. They cannot solve “I cannot make this sound yet.” The reference comparison is exploratory and uses eligible test speakers, a different evaluation from the main classifier test.</p><div class="table-wrap"><table><thead><tr><th>Sounds</th><th>Learned direction</th><th>Population centroid</th><th>Personal centroid*</th><th>Other speakers*</th></tr></thead><tbody>${data.results.map((r) => `<tr><td>${labels[r.contrast]}</td><td>${pct(r.test.balanced_accuracy)}</td><td>${pct(r.test.centroid.balanced_accuracy)}</td><td>${pct(r.test.personalization.personal_mean_balanced_accuracy)}</td><td>${pct(r.test.personalization.other_speakers_mean_balanced_accuracy)}</td></tr>`).join("")}</tbody></table></div><p>*Mean accuracy across eligible speakers on the separate adaptation queries. Raw neural features use cosine distance; acoustic features are first standardized using development data.</p></details>
  <details><summary>Does speaker matching help when reference counts are equal?</summary><p>An exploratory follow-up controls the number of reference clips: two accepted examples per sound for both methods, 30 random draws per speaker, and the same query clips for each paired comparison. Matching the speaker helped several contrasts, but both sparse-centroid methods still performed poorly. This supports further work on adaptation; it does not solve the need for correct personal examples.</p><div class="table-wrap"><table><thead><tr><th>Sounds</th><th>Personal</th><th>Other voices, same count</th><th>Paired advantage</th></tr></thead><tbody>${Object.entries(
    data.adaptation_control.results,
  )
    .map(
      ([id, r]) =>
        `<tr><td>${labels[id]}</td><td>${pct(r.personal_mean_balanced_accuracy)}</td><td>${pct(r.equal_size_other_speakers_mean_balanced_accuracy)}</td><td>+${(r.paired_personal_minus_other * 100).toFixed(1)} percentage points</td></tr>`,
    )
    .join(
      "",
    )}</tbody></table></div><p>This control was added after the main test and did not change the chosen models or demo gates. It is not a learner improvement experiment.</p></details>
  <details><summary>Noise, volume, and timestamp checks</summary><div class="table-wrap"><table><thead><tr><th>Sounds</th><th>Subset</th><th>¼ volume</th><th>20 dB noise</th><th>−20 ms</th><th>+20 ms</th><th>Central 160 ms</th></tr></thead><tbody>${data.results.map((r) => `<tr><td>${labels[r.contrast]}</td>${["baseline_subset", "gain_quarter", "noise_20db", "boundary_minus_20ms", "boundary_plus_20ms", "central_160ms"].map((k) => `<td>${pct(r.conditions[k].balanced_accuracy)}</td>`).join("")}</tr>`).join("")}</tbody></table></div><p>160 deterministic test clips per contrast, 80 of each sound. Noise is additive Gaussian noise; real rooms and microphones can behave differently.</p></details>`;
  document.querySelector("#audit").innerHTML = data.contrasts
    .map(
      (c) =>
        `<details><summary>${esc(c.name)} · ${c.examples.flat().length} real speech examples</summary>${c.examples.map((side, s) => side.map((e, i) => `<div class="caption-audit"><audio controls preload="none" src="/api/guided/${c.id}/example/${s}/${i}/word"></audio><span>“${esc(e.word)}” · ${esc(c.labels[s])}<br><code>${esc(e.utterance)} · ${e.phone_start.toFixed(2)}–${e.phone_end.toFixed(2)} s</code></span></div>`).join("")).join("")}</details>`,
    )
    .join("");
} catch (e) {
  document.querySelector("#results").textContent = e.message;
}
