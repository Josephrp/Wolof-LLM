"""ZeroGPU Gradio demo for the GalsenAI MORENA Wolof continuum family.

Chat markers `<reserved_0>` / `<reserved_1>` are a *checkpoint* property:
  - Instruct / Stock-Instruct → wrap with markers (upstream load_example_instruct).
  - CPT / Distill (base) → plain completion (upstream load_example); optional system
    is prepended as plain context, never as fake chat turns.

Sampler restores the first-demo knobs (top-k + mild repetition penalty) with
MORENA-native defaults (temp=0.7, top_p=0.9) and wider UI ranges.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import spaces
import torch
from huggingface_hub import login, snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

import catalog
import examples_data
import modeling_morena as M

_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _token:
    try:
        login(token=_token, add_to_git_credential=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[auth] HF login warning: {exc}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
CACHE_ROOT = Path(os.environ.get("MORENA_CACHE", Path.home() / ".cache" / "morena-wolof-demo"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

USER_MARK = "<reserved_0>"
ASSIST_MARK = "<reserved_1>"

# UI ranges (first demo) + MORENA-native defaults from load_example*.py.
TEMP_MIN, TEMP_MAX, TEMP_DEFAULT = 0.0, 1.5, 0.7
TOP_P_MIN, TOP_P_MAX, TOP_P_DEFAULT = 0.1, 1.0, 0.9
TOP_K_MIN, TOP_K_MAX, TOP_K_DEFAULT = 0, 200, 50  # 0 = off (matches official loader)
REP_MIN, REP_MAX, REP_DEFAULT = 1.0, 1.5, 1.05
MAX_NEW_MIN, MAX_NEW_MAX, MAX_NEW_DEFAULT = 16, 256, 128

_registry: dict[str, dict[str, Any]] = {}
_registry_lock = threading.Lock()


def _download_repo(repo_id: str) -> Path:
    dest = CACHE_ROOT / repo_id.replace("/", "__")
    snapshot_download(
        repo_id,
        local_dir=str(dest),
        token=_token,
        allow_patterns=[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "modeling_morena.py",
            "README.md",
        ],
    )
    return dest


def _load_bundle(repo_id: str) -> dict[str, Any]:
    model_dir = _download_repo(repo_id)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_cfg = cfg["model"] if "model" in cfg else cfg
    model = M.Transformer(M.ModelConfig(**model_cfg), "sdpa")
    state = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model = model.to(device=DEVICE, dtype=DTYPE).eval()
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    eos = tok.token_to_id("<eos>")
    if eos is None:
        eos = 2
    return {
        "model": model,
        "tok": tok,
        "eos": int(eos),
        "repo_id": repo_id,
        "config": cfg,
    }


def ensure_model(label: str) -> dict[str, Any]:
    with _registry_lock:
        if label not in _registry:
            repo = catalog.MODELS[label]["repo_id"]
            print(f"[load] {label} <- {repo}", flush=True)
            _registry[label] = _load_bundle(repo)
            print(f"[load] ready {label}", flush=True)
        return _registry[label]


try:
    ensure_model(catalog.DEFAULT_MODEL)
except Exception as exc:  # noqa: BLE001
    print(f"[warmup] default model not loaded at startup: {exc}", flush=True)


def catalog_prompt_mode(label: str) -> str:
    entry = catalog.MODELS.get(label, {})
    mode = entry.get("prompt_mode")
    if mode in {"chat", "completion"}:
        return str(mode)
    # Back-compat if catalog only has stage / chat_template.
    stage = str(entry.get("stage", "")).lower()
    if "instruct" in stage:
        return "chat"
    return "completion"


def resolve_prompt_mode(label: str, format_choice: str) -> str:
    choice = (format_choice or "Auto").strip()
    if choice.startswith("Chat"):
        return "chat"
    if choice.startswith("Completion"):
        return "completion"
    return catalog_prompt_mode(label)


def build_prompt(label: str, system: str, user: str, format_choice: str) -> str:
    """Architecture-aware prompt assembly."""
    user = (user or "").strip()
    if not user:
        raise ValueError("User / document text is required.")
    system = (system or "").strip()
    mode = resolve_prompt_mode(label, format_choice)

    if mode == "chat":
        # Instruct checkpoint property: single-token markers ids 3 / 4.
        body = f"{system}\n\n{user}".strip() if system else user
        return f"{USER_MARK}\n{body}\n{ASSIST_MARK}\n"

    # Base / CPT / Distill: plain completion. Soft system = leading context only.
    if system:
        return f"{system}\n\n{user}"
    return user


@torch.no_grad()
def _sample(
    bundle: dict[str, Any],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    """Nucleus sampler with optional top-k + repetition penalty (first-demo path)."""
    model = bundle["model"]
    tok = bundle["tok"]
    eos = bundle["eos"]
    ids = [eos] + tok.encode(prompt, add_special_tokens=False).ids
    if len(ids) < 2:
        raise ValueError("Prompt produced no tokens.")
    x = torch.tensor([ids], device=DEVICE)
    out: list[int] = []
    for _ in range(int(max_new_tokens)):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        logits = model(x, pos, None, x.shape[1], None)[:, -1, :].float()
        if repetition_penalty and abs(float(repetition_penalty) - 1.0) > 1e-6 and out:
            for tid in set(out[-128:]):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty
        if temperature <= 1e-5:
            nxt = int(torch.argmax(logits, dim=-1).item())
        else:
            logits = logits / max(float(temperature), 1e-5)
            if top_k and int(top_k) > 0:
                v, _ = torch.topk(logits, k=min(int(top_k), logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True, dim=-1)
                # Official MORENA nucleus mask (same as load_example.py).
                sp = sp.clone()
                sp[sp.cumsum(-1) - sp > float(top_p)] = 0.0
                sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
                choice = torch.multinomial(sp, 1)
                nxt = int(si.gather(-1, choice)[0, 0].item())
            else:
                nxt = int(torch.multinomial(probs, 1)[0, 0].item())
        if nxt == eos:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return tok.decode(out, skip_special_tokens=True)


def gpu_duration(
    model_label: str,
    format_choice: str,
    system: str,
    user: str,
    max_new_tokens: float,
    temperature: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
) -> int:
    n = int(max_new_tokens or MAX_NEW_DEFAULT)
    size = catalog.MODELS.get(model_label, {}).get("size_key", "0.5b")
    per_tok = {"1.5b": 0.35, "0.5b": 0.18, "0.2b": 0.12}.get(size, 0.25)
    return int(min(180, max(30, 20 + n * per_tok + 15)))


@spaces.GPU(duration=gpu_duration)
def generate(
    model_label: str,
    format_choice: str,
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
        max_new_tokens = max(MAX_NEW_MIN, min(MAX_NEW_MAX, int(max_new_tokens)))
        temperature = max(TEMP_MIN, min(TEMP_MAX, float(temperature)))
        top_p = max(TOP_P_MIN, min(TOP_P_MAX, float(top_p)))
        top_k = max(TOP_K_MIN, min(TOP_K_MAX, int(top_k)))
        repetition_penalty = max(REP_MIN, min(REP_MAX, float(repetition_penalty)))
        prompt = build_prompt(model_label, system, user, format_choice)
        mode = resolve_prompt_mode(model_label, format_choice)
        bundle = ensure_model(model_label)
        text = _sample(
            bundle,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        header = f"_[format={mode}]_\n\n"
        body = text.strip() or (
            "(empty — try top-k=50, rep=1.05, or a longer Wolof prefix / question)"
        )
        return header + body
    except Exception as exc:  # noqa: BLE001
        return (
            f"**Generation error:** `{type(exc).__name__}: {exc}`\n\n"
            "Private `Tonic/*` repos need Space secret `HF_TOKEN`."
        )


def on_model_change(label: str):
    entry = catalog.MODELS.get(label)
    if not entry:
        return (
            "Select a model.",
            gr.update(value="Auto (per checkpoint)"),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )
    try:
        ensure_model(label)
        card = catalog.format_model_card(label)
    except Exception as exc:  # noqa: BLE001
        card = (
            catalog.format_model_card(label)
            + f"\n\n⚠️ **Load warning:** `{type(exc).__name__}: {exc}`"
        )

    mode = catalog_prompt_mode(label)
    if mode == "chat":
        help_md = (
            f"**Auto → chat markers** `{USER_MARK}` / `{ASSIST_MARK}` "
            "(token ids 3/4). System is folded into the user turn."
        )
        sys_val = examples_data.SYSTEM_WOLOF_SHORT
        user_val = examples_data.SFT_PROMPTS[1]
        user_label = "User message"
        out_label = "Assistant"
    else:
        help_md = (
            "**Auto → completion** (no chat markers). System is optional plain "
            "context prepended before the document prefix — do not invent `<|user|>` tags."
        )
        sys_val = examples_data.SYSTEM_COMPLETION_HINT
        user_val = examples_data.CLM_PREFIXES[0]
        user_label = "Document prefix"
        out_label = "Completion"

    return (
        card,
        gr.update(value="Auto (per checkpoint)"),
        gr.update(value=sys_val),
        gr.update(value=user_val, label=user_label),
        gr.update(label=out_label, value=""),
        help_md,
    )


def build_examples() -> list[list[Any]]:
    distill = catalog.DEFAULT_MODEL
    nano = "0.2B Nano · Distill (recommended)"
    cpt = "0.5B Mini · Wolof CPT"
    stock = "Stock 1.5B Instruct (public baseline)"
    wolof_sft = "1.5B · Wolof Instruct (SFT)"
    # [model, format, system, user, max, temp, top_p, top_k, rep]
    return [
        # Distill / CPT: completion + soft Wolof cue + CLM / natural prefixes
        [
            distill,
            "Auto (per checkpoint)",
            examples_data.SYSTEM_COMPLETION_HINT,
            examples_data.CLM_PREFIXES[0],
            128,
            0.7,
            0.9,
            50,
            1.05,
        ],
        [
            distill,
            "Auto (per checkpoint)",
            examples_data.SYSTEM_COMPLETION_HINT,
            examples_data.WOLOF_NATURAL[1],
            128,
            0.7,
            0.9,
            50,
            1.05,
        ],
        [
            nano,
            "Completion (base / CPT / Distill)",
            "",
            examples_data.CLM_PREFIXES[1],
            96,
            0.7,
            0.9,
            40,
            1.05,
        ],
        [
            cpt,
            "Auto (per checkpoint)",
            examples_data.SYSTEM_COMPLETION_HINT,
            examples_data.CLM_PREFIXES[2],
            128,
            0.6,
            0.9,
            50,
            1.1,
        ],
        # Explicit chat override on distill (compare vs Auto) — often weaker on base
        [
            distill,
            "Chat markers (Instruct)",
            examples_data.SYSTEM_WOLOF,
            examples_data.WOLOF_NATURAL[0],
            128,
            0.7,
            0.9,
            50,
            1.05,
        ],
        # Stock instruct: real chat + system
        [
            stock,
            "Auto (per checkpoint)",
            examples_data.SYSTEM_BILINGUAL,
            "Explain briefly what MORENA is, then greet in Wolof.",
            128,
            0.6,
            0.9,
            50,
            1.05,
        ],
        [
            stock,
            "Auto (per checkpoint)",
            examples_data.SYSTEM_WOLOF_SHORT,
            examples_data.SFT_PROMPTS[1],
            96,
            0.7,
            0.9,
            50,
            1.05,
        ],
        [
            stock,
            "Auto (per checkpoint)",
            examples_data.SYSTEM_WOLOF,
            examples_data.WOLOF_NATURAL[0],
            128,
            0.7,
            0.9,
            50,
            1.05,
        ],
        # Collapsed Wolof SFT (transparency)
        [
            wolof_sft,
            "Auto (per checkpoint)",
            examples_data.SYSTEM_WOLOF_SHORT,
            examples_data.SFT_PROMPTS[0],
            64,
            0.7,
            0.9,
            50,
            1.05,
        ],
    ]


THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="stone",
    neutral_hue="stone",
    font=gr.themes.GoogleFont("Source Sans 3"),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
).set(
    body_background_fill="linear-gradient(165deg, #f4f7f2 0%, #e8efe6 45%, #f7f3ea 100%)",
    block_background_fill="rgba(255,255,255,0.82)",
    border_color_primary="#c5d4c0",
)

CSS = """
.hero-title { font-size: 1.85rem !important; font-weight: 700; letter-spacing: -0.02em; }
.hero-sub { opacity: 0.85; margin-top: 0.25rem; }
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CSS, title="MORENA Wolof · ZeroGPU") as demo:
        gr.Markdown(
            '<p class="hero-title">MORENA Wolof continuum</p>'
            '<p class="hero-sub">GalsenAI · native Muon trainer · ZeroGPU Gradio demo</p>'
        )
        with gr.Accordion("About this experiment", open=True):
            gr.Markdown(catalog.EXPERIMENT_OVERVIEW)

        with gr.Row():
            with gr.Column(scale=5):
                model_dd = gr.Dropdown(
                    choices=catalog.model_choices(),
                    value=catalog.DEFAULT_MODEL,
                    label="Model",
                    info="Distill = completion demos; Stock Instruct = chat markers; Wolof SFT collapsed.",
                )
                format_dd = gr.Radio(
                    choices=[
                        "Auto (per checkpoint)",
                        "Chat markers (Instruct)",
                        "Completion (base / CPT / Distill)",
                    ],
                    value="Auto (per checkpoint)",
                    label="Prompt format",
                    info="Auto follows the checkpoint. Override only to A/B-test templates.",
                )
                system_tb = gr.Textbox(
                    label="System",
                    lines=3,
                    value=examples_data.SYSTEM_COMPLETION_HINT,
                    placeholder="Chat: folded into user turn. Completion: plain leading context.",
                )
                user_tb = gr.Textbox(
                    label="Document prefix",
                    lines=6,
                    value=examples_data.CLM_PREFIXES[0],
                    placeholder="User message or Wolof document start…",
                )
                with gr.Accordion("Generation parameters", open=False):
                    gr.Markdown(
                        "Defaults: **temp=0.7**, **top-p=0.9** (official MORENA). "
                        "Top-k **50** + mild **rep=1.05** restored from the first demo "
                        "(set top-k=0 and rep=1.0 for the exact upstream sampler)."
                    )
                    max_new = gr.Slider(
                        MAX_NEW_MIN, MAX_NEW_MAX, value=MAX_NEW_DEFAULT, step=8, label="Max new tokens"
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
                output_tb = gr.Textbox(label="Completion", lines=10)

            with gr.Column(scale=4):
                model_info = gr.Markdown(catalog.format_model_card(catalog.DEFAULT_MODEL))
                format_help = gr.Markdown(
                    "**Auto → completion** for Distill/CPT (no chat markers). "
                    "System is optional plain context before the document prefix."
                )

        model_dd.change(
            fn=on_model_change,
            inputs=model_dd,
            outputs=[model_info, format_dd, system_tb, user_tb, output_tb, format_help],
        )

        gen_inputs = [
            model_dd,
            format_dd,
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
            label="Examples (system + format + Wolof CLM / SFT)",
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue(max_size=32).launch()
