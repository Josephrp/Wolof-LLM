"""ZeroGPU Gradio demo for Tonic/Qwen3.5-9B-Wolof-{Instruct,CPT}-v1."""

from __future__ import annotations

import os
import threading
from typing import Any

# ZeroGPU: spaces must patch torch before CUDA-touching imports.
import spaces
import torch
import gradio as gr
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

import catalog
import examples_data

_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _token:
    try:
        login(token=_token, add_to_git_credential=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[auth] HF login warning: {exc}", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

TEMP_MIN, TEMP_MAX, TEMP_DEFAULT = 0.0, 1.5, 0.7
TOP_P_MIN, TOP_P_MAX, TOP_P_DEFAULT = 0.1, 1.0, 0.99
TOP_K_MIN, TOP_K_MAX, TOP_K_DEFAULT = 0, 200, 99
REP_MIN, REP_MAX, REP_DEFAULT = 1.0, 2.0, 1.5
MAX_NEW_MIN, MAX_NEW_MAX, MAX_NEW_DEFAULT = 16, 512, 160

# Training anchors (Experiments/.../common/anchoring.py) — injected, never shown in UI defaults.
ANCHOR_SPECIALS = ("<|lang_wo|>", "<|ortho_ns|>")
SYSTEM_ANCHOR = (
    "You are a Wolof language assistant. "
    "Prefer informal / non-standard Wolof Latin orthography "
    "(phone and social spelling), not school/CLAD formal orthography. "
    f"{ANCHOR_SPECIALS[0]} {ANCHOR_SPECIALS[1]}"
)
ANCHOR_PREFIX = f"{ANCHOR_SPECIALS[0]} {ANCHOR_SPECIALS[1]}"
OUTPUT_PLACEHOLDER = "_Assistant reply will appear here…_"

_registry: dict[str, dict[str, Any]] = {}
_registry_lock = threading.Lock()
_active_label: str | None = None


def _load_bundle(repo_id: str) -> dict[str, Any]:
    print(f"[load] downloading/loading {repo_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(repo_id, token=_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        token=_token,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    # Eager .to(cuda) registers weights for ZeroGPU packing (monkey-patched off-GPU).
    model = model.to(DEVICE)
    model.eval()
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    print(f"[load] ready {repo_id}", flush=True)
    return {"model": model, "tok": tok, "repo_id": repo_id}


def _unload_all() -> None:
    global _active_label
    for key in list(_registry.keys()):
        bundle = _registry.pop(key)
        try:
            del bundle["model"]
        except Exception:  # noqa: BLE001
            pass
    _active_label = None
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def ensure_model(label: str) -> dict[str, Any]:
    """Load one 9B checkpoint; keep a single resident model to save VRAM."""
    global _active_label
    with _registry_lock:
        if label in _registry:
            _active_label = label
            return _registry[label]
        if _registry:
            _unload_all()
        repo = catalog.MODELS[label]["repo_id"]
        bundle = _load_bundle(repo)
        _registry[label] = bundle
        _active_label = label
        return bundle


# Eager-pack default on ZeroGPU (weights registered at startup).
try:
    ensure_model(catalog.DEFAULT_MODEL)
except Exception as exc:  # noqa: BLE001
    print(f"[warmup] default model not loaded at startup: {exc}", flush=True)


def _strip_anchors(text: str) -> str:
    out = text or ""
    for special in ANCHOR_SPECIALS:
        out = out.replace(special, "")
    return " ".join(out.split()).strip()


def _has_anchors(text: str) -> bool:
    return all(a in (text or "") for a in ANCHOR_SPECIALS)


def _system_with_anchors(system: str) -> str:
    """Always inject training SYSTEM_ANCHOR; merge optional user system without duplicates."""
    extra = _strip_anchors(system or "")
    if not extra:
        return SYSTEM_ANCHOR
    if _has_anchors(system or ""):
        # User pasted a full anchored system — keep their text but ensure both specials exist.
        return (system or "").strip()
    return f"{SYSTEM_ANCHOR}\n{extra}"


def _cpt_prompt_with_anchors(system: str, user: str) -> str:
    """Prepend anchors for CPT continuation; keep optional context above the seed."""
    user = (user or "").strip()
    if not user:
        raise ValueError("Prompt is required.")
    seed = user if _has_anchors(user) else f"{ANCHOR_PREFIX} {user}"
    system = (system or "").strip()
    if system:
        ctx = system if _has_anchors(system) else f"{ANCHOR_PREFIX} {system}"
        return f"{ctx}\n\n{seed}"
    return seed


def build_messages(system: str, user: str) -> list[dict[str, str]]:
    user = (user or "").strip()
    if not user:
        raise ValueError("User message is required.")
    return [
        {"role": "system", "content": _system_with_anchors(system)},
        {"role": "user", "content": user},
    ]


def _prompt_for_mode(mode: str, tok: Any, system: str, user: str) -> str:
    user = (user or "").strip()
    if not user:
        raise ValueError("Prompt is required.")
    if mode == "completion":
        return _cpt_prompt_with_anchors(system, user)
    messages = build_messages(system, user)
    # Match SFT: empty <think> block via template (enable_thinking=False).
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generate_text(
    label: str,
    system: str,
    user: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    entry = catalog.MODELS[label]
    bundle = ensure_model(label)
    model = bundle["model"]
    tok = bundle["tok"]
    prompt = _prompt_for_mode(entry.get("mode", "chat"), tok, system, user)

    encoded = tok(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_len = int(encoded["input_ids"].shape[-1])

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": float(temperature) > 1e-5,
        "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
        "repetition_penalty": float(repetition_penalty),
    }
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = float(top_p)
        if int(top_k) > 0:
            gen_kwargs["top_k"] = int(top_k)

    with torch.no_grad():
        out = model.generate(**encoded, **gen_kwargs)
    new_tokens = out[0, input_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return text or "(empty — try a clearer Wolof prompt or raise temperature slightly)"


def gpu_duration(
    model_label: str,
    system: str,
    user: str,
    max_new_tokens: float,
    temperature: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
) -> int:
    del system, user, temperature, top_p, top_k, repetition_penalty
    n = int(max_new_tokens or MAX_NEW_DEFAULT)
    # 9B on ZeroGPU large: ~0.35–0.55 s/tok cold; budget generously.
    per_tok = 0.45
    load_pad = 90 if model_label != _active_label else 20
    return int(min(240, max(45, load_pad + n * per_tok)))


def _clamp_params(
    max_new_tokens: float,
    temperature: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
) -> tuple[int, float, float, int, float]:
    return (
        max(MAX_NEW_MIN, min(MAX_NEW_MAX, int(max_new_tokens))),
        max(TEMP_MIN, min(TEMP_MAX, float(temperature))),
        max(TOP_P_MIN, min(TOP_P_MAX, float(top_p))),
        max(TOP_K_MIN, min(TOP_K_MAX, int(top_k))),
        max(REP_MIN, min(REP_MAX, float(repetition_penalty))),
    )


def _safe_generate(
    model_label: str,
    system: str,
    user: str,
    max_new_tokens: float,
    temperature: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
) -> str:
    if model_label not in catalog.MODELS:
        return f"Unknown model: {model_label}"
    try:
        mn, temp, tp, tk, rep = _clamp_params(
            max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )
        return _generate_text(model_label, system, user, mn, temp, tp, tk, rep)
    except Exception as exc:  # noqa: BLE001
        return (
            f"**Generation error:** `{type(exc).__name__}: {exc}`\n\n"
            "Qwen3.5 requires `transformers>=5.19`. ZeroGPU size: **large** (~18 GB bf16). "
            "Anchors `<|lang_wo|>` `<|ortho_ns|>` are injected automatically."
        )


@spaces.GPU(duration=gpu_duration, size="large")
def generate(
    model_label: str,
    system: str,
    user: str,
    max_new_tokens: float,
    temperature: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
) -> str:
    return _safe_generate(
        model_label,
        system,
        user,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    )


def on_model_change(label: str):
    if label not in catalog.MODELS:
        return (
            "Select a model.",
            gr.update(),
            "",
            gr.update(),
            gr.update(),
        )
    entry = catalog.MODELS[label]
    card = catalog.format_model_card(label)
    mode = entry.get("mode", "chat")
    status = (
        f"_Selected `{entry['repo_id']}` · mode `{mode}` · ZeroGPU `large` · "
        "weights load on Generate (or at Space boot for the default)._"
    )
    if mode == "completion":
        sys_val = ""
        user_val = examples_data.CPT_PREFIXES[0]
        prompt_label = "Continuation prompt"
    else:
        sys_val = examples_data.SYSTEM_DEFAULT
        user_val = examples_data.CHAT_PROMPTS[0]
        prompt_label = "User message"
    return (
        card,
        OUTPUT_PLACEHOLDER,
        status,
        gr.update(value=sys_val),
        gr.update(value=user_val, label=prompt_label),
    )


def build_examples() -> list[list[Any]]:
    instruct = catalog.DEFAULT_MODEL
    cpt = "9B · CPT-v1 (best BPB)"
    # [model, system, user, max, temp, top_p, top_k, rep]
    p, k, r = TOP_P_DEFAULT, TOP_K_DEFAULT, REP_DEFAULT
    return [
        [instruct, examples_data.SYSTEM_DEFAULT, examples_data.CHAT_PROMPTS[0], 160, 0.7, p, k, r],
        [instruct, examples_data.SYSTEM_DEFAULT, examples_data.CHAT_PROMPTS[1], 160, 0.7, p, k, r],
        [instruct, examples_data.SYSTEM_TEACHER, examples_data.CHAT_PROMPTS[4], 192, 0.7, p, k, r],
        [instruct, examples_data.SYSTEM_TRANSLATE, examples_data.CHAT_PROMPTS[5], 128, 0.6, p, k, r],
        [instruct, examples_data.SYSTEM_TRANSLATE, examples_data.CHAT_PROMPTS[6], 128, 0.6, p, k, r],
        [instruct, examples_data.SYSTEM_STORY, examples_data.CHAT_PROMPTS[7], 192, 0.8, p, k, r],
        [instruct, examples_data.SYSTEM_DEFAULT, examples_data.CHAT_PROMPTS[8], 160, 0.7, p, k, r],
        [instruct, examples_data.SYSTEM_DEFAULT, examples_data.CHAT_PROMPTS[9], 160, 0.7, p, k, r],
        [cpt, "", examples_data.CPT_PREFIXES[0], 128, 0.8, p, k, r],
        [cpt, "", examples_data.CPT_PREFIXES[1], 128, 0.8, p, k, r],
        [cpt, "", examples_data.CPT_PREFIXES[2], 128, 0.8, p, k, r],
        [cpt, "", examples_data.CPT_PREFIXES[4], 128, 0.8, p, k, r],
    ]


THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="stone",
    neutral_hue="stone",
    font=gr.themes.GoogleFont("Source Sans 3"),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
).set(
    body_background_fill="linear-gradient(165deg, #eef6f4 0%, #e7eef2 45%, #f5f1e8 100%)",
    block_background_fill="rgba(255,255,255,0.84)",
    border_color_primary="#b9cfc8",
)

CSS = """
.hero-title { font-size: 1.85rem !important; font-weight: 700; letter-spacing: -0.02em; }
.hero-sub { opacity: 0.85; margin-top: 0.25rem; }
.assistant-md {
  min-height: 22rem;
  padding: 0.85rem 1rem;
  border: 1px solid #b9cfc8;
  border-radius: 10px;
  background: rgba(255,255,255,0.92);
  overflow-y: auto;
}
footer { display: none !important; }
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CSS, title="Qwen3.5-9B Wolof") as demo:
        gr.Markdown(
            '<p class="hero-title">Qwen3.5-9B Wolof</p>'
            '<p class="hero-sub">GalsenAI · largest family checkpoint · Instruct + CPT · ZeroGPU</p>'
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Accordion("About this model", open=True):
                    gr.Markdown(catalog.EXPERIMENT_OVERVIEW)
            with gr.Column(scale=1):
                with gr.Accordion("Selected checkpoint", open=True):
                    model_info = gr.Markdown(catalog.format_model_card(catalog.DEFAULT_MODEL))
                gr.Markdown(
                    "**Prompting:** Instruct uses Qwen `apply_chat_template` with the "
                    "training system anchor injected automatically "
                    "(`<|lang_wo|>` `<|ortho_ns|>` — invisible in the UI). "
                    "CPT seeds get the same anchors prepended."
                )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=catalog.model_choices(),
                    value=catalog.DEFAULT_MODEL,
                    label="Model",
                    info=(
                        "Instruct-v1 = chat (recommended). CPT-v1 = raw continuation "
                        "(best BPB). Wolof anchors are injected automatically."
                    ),
                )
                system_tb = gr.Textbox(
                    label="System (optional extra)",
                    lines=3,
                    value=examples_data.SYSTEM_DEFAULT,
                    placeholder="Optional instructions — training anchors are added for you…",
                )
                user_tb = gr.Textbox(
                    label="User message",
                    lines=6,
                    value=examples_data.CHAT_PROMPTS[0],
                    placeholder="Wolof chat turn, or CPT continuation seed (no anchors needed)…",
                )
                with gr.Accordion("Generation parameters", open=False):
                    gr.Markdown(
                        "Defaults tuned for Wolof: **temp=0.7**, **top-p=0.99**, "
                        "**top-k=99**, **rep=1.5**. Set top-k=0 to disable top-k."
                    )
                    max_new = gr.Slider(
                        MAX_NEW_MIN,
                        MAX_NEW_MAX,
                        value=MAX_NEW_DEFAULT,
                        step=8,
                        label="Max new tokens",
                    )
                    temperature = gr.Slider(
                        TEMP_MIN, TEMP_MAX, value=TEMP_DEFAULT, step=0.05, label="Temperature"
                    )
                    top_p = gr.Slider(
                        TOP_P_MIN, TOP_P_MAX, value=TOP_P_DEFAULT, step=0.01, label="Top-p"
                    )
                    top_k = gr.Slider(
                        TOP_K_MIN, TOP_K_MAX, value=TOP_K_DEFAULT, step=1, label="Top-k (0 = off)"
                    )
                    rep_pen = gr.Slider(
                        REP_MIN, REP_MAX, value=REP_DEFAULT, step=0.01, label="Repetition penalty"
                    )
                run_btn = gr.Button("Generate", variant="primary")
                status_md = gr.Markdown("_Select a model and generate._")

            with gr.Column(scale=1):
                gr.Markdown("### Assistant")
                output_md = gr.Markdown(
                    OUTPUT_PLACEHOLDER,
                    elem_classes=["assistant-md"],
                )

        model_dd.change(
            fn=on_model_change,
            inputs=model_dd,
            outputs=[model_info, output_md, status_md, system_tb, user_tb],
        )

        gen_inputs = [
            model_dd,
            system_tb,
            user_tb,
            max_new,
            temperature,
            top_p,
            top_k,
            rep_pen,
        ]
        run_btn.click(fn=generate, inputs=gen_inputs, outputs=output_md)
        user_tb.submit(fn=generate, inputs=gen_inputs, outputs=output_md)

        gr.Examples(
            examples=build_examples(),
            inputs=gen_inputs,
            label="Examples (Instruct chat + CPT continuation)",
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(max_size=32).launch()
