import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

const SYSTEM_ANCHOR =
  "You are a Wolof language assistant. Prefer informal / non-standard Wolof Latin orthography " +
  "(phone and social spelling), not school/CLAD formal orthography. <|lang_wo|> <|ortho_ns|>";

const MODELS = {
  "0.8B · Instruct (default)": {
    repo_id: "Tonic/Qwen3.5-08B-Wolof-Instruct-v1-ONNX",
    metrics: { BPB: 1.2333, BLEU: 4.65, chrF: 24.13 },
    summary: "Smallest family Instruct checkpoint — best WebGPU fit.",
  },
  "2B · Instruct": {
    repo_id: "Tonic/Qwen3.5-2B-Wolof-Instruct-v1-ONNX",
    metrics: { BPB: 1.1649, BLEU: 4.88, chrF: 23.96 },
    summary: "Mid-size Instruct; stronger fluency, heavier download.",
  },
  "4B · Instruct": {
    repo_id: "Tonic/Qwen3.5-4B-Wolof-Instruct-v1-ONNX",
    metrics: { BPB: 1.0987, BLEU: 6.28, chrF: 26.26 },
    summary: "Largest browser option in this Space (9B stays on ZeroGPU).",
  },
};

const DEFAULT_MODEL = "0.8B · Instruct (default)";
const DEFAULT_SYSTEM =
  "Yawu jëmmal Wolof. Tontu ci Wolof bu gaaw te leer, ak orthographie informal / téléphone (pas CLAD formel).";

const EXAMPLES = [
  "Naka nga def? Waxal ma lu am solo ci sa bés.",
  "Lan mooy Dakar, te lan mooy yu ko defar?",
  "Nan lañu koy togg ceebu jën? Waxal ma ay etap yu yomb.",
  "Translate to Wolof: Good morning, how are you today?",
  "Bindal ma ab xew-xew bu gatt ci Marché Sandaga.",
];

const $ = (id) => document.getElementById(id);
const modelSelect = $("modelSelect");
const systemEl = $("system");
const userEl = $("user");
const statusEl = $("status");
const assistantEl = $("assistant");
const modelCard = $("modelCard");
const btnLoad = $("btnLoad");
const btnGen = $("btnGen");

let generator = null;
let loadedRepo = null;

function stripAnchors(text) {
  return (text || "")
    .replaceAll("<|lang_wo|>", "")
    .replaceAll("<|ortho_ns|>", "")
    .replace(/\s+/g, " ")
    .trim();
}

function systemWithAnchors(extra) {
  const cleaned = stripAnchors(extra);
  return cleaned ? `${SYSTEM_ANCHOR}\n${cleaned}` : SYSTEM_ANCHOR;
}

function formatCard(label) {
  const entry = MODELS[label];
  const m = entry.metrics;
  modelCard.innerHTML = `
    <p><code>${entry.repo_id}</code></p>
    <p>${entry.summary}</p>
    <p><strong>BPB</strong> ${m.BPB.toFixed(4)} · <strong>BLEU</strong> ${m.BLEU.toFixed(2)} · <strong>chrF</strong> ${m.chrF.toFixed(2)}</p>
  `;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function populate() {
  for (const label of Object.keys(MODELS)) {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = DEFAULT_MODEL;
  systemEl.value = DEFAULT_SYSTEM;
  userEl.value = EXAMPLES[0];
  formatCard(DEFAULT_MODEL);

  const ex = $("examples");
  for (const prompt of EXAMPLES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = prompt.slice(0, 42) + (prompt.length > 42 ? "…" : "");
    btn.title = prompt;
    btn.addEventListener("click", () => {
      userEl.value = prompt;
    });
    ex.appendChild(btn);
  }
}

modelSelect.addEventListener("change", () => {
  formatCard(modelSelect.value);
  assistantEl.innerHTML = "<em>Assistant reply will appear here…</em>";
  if (loadedRepo && loadedRepo !== MODELS[modelSelect.value].repo_id) {
    btnGen.disabled = true;
    setStatus("Model changed — click Load model.");
  }
});

btnLoad.addEventListener("click", async () => {
  const label = modelSelect.value;
  const repo = MODELS[label].repo_id;
  btnLoad.disabled = true;
  btnGen.disabled = true;
  setStatus(`Loading ${repo} (WebGPU q4f16)…`);
  try {
    if (generator) {
      await generator.dispose?.();
      generator = null;
    }
    generator = await pipeline("text-generation", repo, {
      device: "webgpu",
      dtype: "q4f16",
    });
    loadedRepo = repo;
    btnGen.disabled = false;
    setStatus(`Ready: ${repo}`);
  } catch (err) {
    console.error(err);
    setStatus(`Load failed: ${err.message || err}`);
  } finally {
    btnLoad.disabled = false;
  }
});

btnGen.addEventListener("click", async () => {
  if (!generator) return;
  const user = userEl.value.trim();
  if (!user) {
    setStatus("User message is required.");
    return;
  }
  btnGen.disabled = true;
  assistantEl.textContent = "";
  setStatus("Generating…");
  const messages = [
    { role: "system", content: systemWithAnchors(systemEl.value) },
    { role: "user", content: user },
  ];
  const max_new_tokens = Number($("maxNew").value) || 160;
  const temperature = Number($("temp").value);
  const top_p = Number($("topP").value);
  const top_k = Number($("topK").value);
  const repetition_penalty = Number($("rep").value);
  try {
    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token) => {
        assistantEl.textContent += token;
      },
    });
    const out = await generator(messages, {
      max_new_tokens,
      do_sample: temperature > 1e-5,
      temperature: Math.max(temperature, 1e-5),
      top_p,
      top_k: top_k > 0 ? top_k : undefined,
      repetition_penalty,
      streamer,
    });
    const final =
      out?.[0]?.generated_text?.at?.(-1)?.content ||
      assistantEl.textContent ||
      "(empty)";
    if (!assistantEl.textContent.trim()) {
      assistantEl.textContent = final;
    }
    setStatus("Done.");
  } catch (err) {
    console.error(err);
    assistantEl.textContent = `Generation error: ${err.message || err}`;
    setStatus("Error.");
  } finally {
    btnGen.disabled = false;
  }
});

populate();
