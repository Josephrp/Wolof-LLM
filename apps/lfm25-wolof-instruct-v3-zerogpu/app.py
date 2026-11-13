"""ZeroGPU Gradio demo for Tonic/LFM2.5-*-Wolof-Instruct-v3 (×5)."""

from __future__ import annotations

import os
import threading
from typing import Any

import gradio as gr
import spaces
import torch
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
TOP_P_MIN, TOP_P_MAX, TOP_P_DEFAULT = 0.1, 1.0, 0.9
TOP_K_MIN, TOP_K_MAX, TOP_K_DEFAULT = 0, 200, 50
REP_MIN, REP_MAX, REP_DEFAULT = 1.0, 1.5, 1.05
MAX_NEW_MIN, MAX_NEW_MAX, MAX_NEW_DEFAULT = 16, 512, 160

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
    )
    # Eager .to(cuda) registers weights for ZeroGPU packing (monkey-patched off-GPU).
    model = model.to(DEVICE)
    model.eval()
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
    """Load one Instruct-v3 checkpoint; keep a single resident model to save VRAM."""
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


def build_messages(system: str, user: str) -> list[dict[str, str]]:
    user = (user or "").strip()
    if not user:
        raise ValueError("User message is required.")
    messages: list[dict[str, str]] = []
    system = (system or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


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
    bundle = ensure_model(label)
    model = bundle["model"]
    tok = bundle["tok"]
    messages = build_messages(system, user)

    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
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
    n = int(max_new_tokens or MAX_NEW_DEFAULT)
    size = catalog.MODELS.get(model_label, {}).get("size_key", "1.2b")
    per_tok = {
        "230m": 0.08,
        "350m": 0.10,
        "1.2b": 0.18,
        "2.6b": 0.28,
        "8b": 0.45,
    }.get(size, 0.2)
    # Extra budget when cold-loading a new checkpoint inside the worker.
    load_pad = 45 if model_label != _active_label else 10
    return int(min(240, max(30, load_pad + n * per_tok)))


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
            "Private `Tonic/*` repos need Space secret `HF_TOKEN` (read). "
            "8B-A1B requires ZeroGPU **xlarge**."
        )


@spaces.GPU(duration=gpu_duration, size="large")
def generate_large(
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


@spaces.GPU(duration=gpu_duration, size="xlarge")
def generate_xlarge(
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
    entry = catalog.MODELS.get(model_label, {})
    if entry.get("zerogpu_size") == "xlarge":
        return generate_xlarge(
            model_label,
            system,
            user,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        )
    return generate_large(
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
        return "Select a model.", gr.update(), ""
    entry = catalog.MODELS[label]
    card = catalog.format_model_card(label)
    zsize = entry.get("zerogpu_size", "large")
    status = (
        f"_Selected `{entry['repo_id']}` · ZeroGPU `{zsize}` · "
        "weights load on Generate._"
    )
    return card, gr.update(value=""), status


def build_examples() -> list[list[Any]]:
    hero = catalog.DEFAULT_MODEL
    best = "2.6B · Instruct-v3 (best BPB)"
    small = "350M · Instruct-v3"
    tiny = "230M · Instruct-v3"
    moe = "8B-A1B · Instruct-v3 (MoE)"
    # [model, system, user, max, temp, top_p, top_k, rep]
    return [
        [hero, examples_data.SYSTEM_WOLOF_SHORT, examples_data.CHAT_PROMPTS[1], 128, 0.7, 0.9, 50, 1.05],
        [hero, examples_data.SYSTEM_WOLOF, examples_data.CHAT_PROMPTS[2], 160, 0.7, 0.9, 50, 1.05],
        [hero, examples_data.SYSTEM_TRANSLATE, examples_data.CHAT_PROMPTS[5], 96, 0.5, 0.9, 40, 1.05],
        [best, examples_data.SYSTEM_WOLOF_SHORT, examples_data.CHAT_PROMPTS[4], 160, 0.7, 0.9, 50, 1.05],
        [best, examples_data.SYSTEM_BILINGUAL, examples_data.CHAT_PROMPTS[6], 128, 0.6, 0.9, 50, 1.05],
        [small, examples_data.SYSTEM_WOLOF_SHORT, examples_data.CHAT_PROMPTS[0], 96, 0.7, 0.9, 40, 1.05],
        [tiny, examples_data.SYSTEM_WOLOF_SHORT, examples_data.CHAT_PROMPTS[3], 80, 0.7, 0.9, 40, 1.05],
        [moe, examples_data.SYSTEM_WOLOF, examples_data.CHAT_PROMPTS[7], 192, 0.7, 0.9, 50, 1.05],
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
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CSS, title="LFM2.5 Wolof Instruct-v3") as demo:
        gr.Markdown(
            '<p class="hero-title">LFM2.5 Wolof Instruct-v3</p>'
            '<p class="hero-sub">GalsenAI · LiquidAI LFM2.5 · five sizes · ZeroGPU</p>'
        )
        with gr.Accordion("About this family", open=True):
            gr.Markdown(catalog.EXPERIMENT_OVERVIEW)

        with gr.Row():
            with gr.Column(scale=5):
                model_dd = gr.Dropdown(
                    choices=catalog.model_choices(),
                    value=catalog.DEFAULT_MODEL,
                    label="Model",
                    info="All five are Instruct-v3 chat checkpoints. 8B-A1B uses ZeroGPU xlarge.",
                )
                system_tb = gr.Textbox(
                    label="System",
                    lines=3,
                    value=examples_data.SYSTEM_WOLOF_SHORT,
                    placeholder="Optional system prompt…",
                )
                user_tb = gr.Textbox(
                    label="User message",
                    lines=6,
                    value=examples_data.CHAT_PROMPTS[1],
                    placeholder="Wolof or bilingual user turn…",
                )
                with gr.Accordion("Generation parameters", open=False):
                    gr.Markdown(
                        "Defaults: **temp=0.7**, **top-p=0.9**, top-k **50**, "
                        "mild **rep=1.05**. Set top-k=0 to disable top-k."
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
                output_tb = gr.Textbox(label="Assistant", lines=12)
                status_md = gr.Markdown("_Select a model and generate._")

            with gr.Column(scale=4):
                model_info = gr.Markdown(catalog.format_model_card(catalog.DEFAULT_MODEL))
                gr.Markdown(
                    "**Prompting:** `apply_chat_template` with each repo’s Jinja template "
                    "(`<|im_start|>` roles). System is optional."
                )

        model_dd.change(
            fn=on_model_change,
            inputs=model_dd,
            outputs=[model_info, output_tb, status_md],
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
        run_btn.click(fn=generate, inputs=gen_inputs, outputs=output_tb)
        user_tb.submit(fn=generate, inputs=gen_inputs, outputs=output_tb)

        gr.Examples(
            examples=build_examples(),
            inputs=gen_inputs,
            label="Examples (all five Instruct-v3 sizes)",
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(max_size=32).launch()
