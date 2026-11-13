"""Wolof orthography crowdsourcing annotator (Hugging Face Spaces / Gradio)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import gradio as gr

from config import Settings, load_settings
from orthography import heuristic_hint
from storage import AnnotationStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
LABEL_HELP = {
    "standard": "School / CLAD Latin Wolof with proper special letters (à é ë ñ ŋ …).",
    "non_standard": "Informal phone/social spelling (no accents, gn/kh swaps, fused words).",
    "mixed": "Both systems in one text, or unclear which dominates.",
    "not_wolof": "French/English only, garbage, or not Wolof.",
}

GUIDE_MD = """
### How to label

Wolof is primarily oral — written text often mixes **two Latin spellings**:

| Label | Meaning |
|---|---|
| **standard** | Official / school orthography with diacritics (`jàpp`, `ñu`, `ŋànt`) |
| **non_standard** | Informal spelling (`japp`, `gnu`, `kh` for `x`) |
| **mixed** | Both in one passage, or truly ambiguous |
| **not_wolof** | Not Wolof / unusable |

You only **tag** the existing string — do not rewrite it.
When `CONSENSUS_THRESHOLD` volunteers choose the same label, the row is sealed
into the consensus dataset.
"""


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _fmt_stats(store: AnnotationStore) -> str:
    s = store.stats()
    return (
        f"**Source rows:** {s['source_rows']:,}  ·  "
        f"**Open:** {s['open_rows']:,}  ·  "
        f"**Votes:** {s['votes']:,}  ·  "
        f"**Consensus:** {s['consensus']:,}  ·  "
        f"**Threshold:** {s['threshold']}"
    )


def _fmt_item(item: dict[str, Any] | None, settings: Settings) -> tuple[str, str, str, str]:
    if item is None:
        return (
            "",
            "",
            "_No open items left for you (or corpus fully sealed)._",
            "Heuristic: —",
        )
    text = _clip(item["text"], settings.max_text_chars)
    tally = item.get("labels_so_far") or {}
    tally_s = ", ".join(f"{k}:{v}" for k, v in sorted(tally.items())) or "none"
    meta = (
        f"`row_id={item['row_id']}` · votes so far: **{item['n_votes_so_far']}** "
        f"({tally_s})"
    )
    hint = f"Heuristic (not a label): {heuristic_hint(item['text'])}"
    return item["row_id"], text, meta, hint


def build_app(settings: Settings, store: AnnotationStore) -> gr.Blocks:
    label_choices = list(settings.labels)

    with gr.Blocks(title=settings.app_title, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# {settings.app_title}\n\n"
            f"Source: `{settings.source_dataset}` (`{settings.source_split}`) → "
            f"votes `{settings.votes_dataset}` → consensus "
            f"`{settings.consensus_dataset}`."
        )
        gr.Markdown(GUIDE_MD)

        stats_md = gr.Markdown(_fmt_stats(store))
        status_md = gr.Markdown("_Enter a volunteer id, then load an item._")

        with gr.Row():
            volunteer = gr.Textbox(
                label="Volunteer id",
                placeholder="your HF username or nickname",
                scale=3,
            )
            load_btn = gr.Button("Load next item", variant="primary", scale=1)

        row_id_state = gr.State("")
        text_state = gr.State("")

        text_box = gr.Textbox(
            label="Text to annotate",
            lines=10,
            max_lines=20,
            interactive=False,
        )
        meta_md = gr.Markdown("")
        hint_md = gr.Markdown("")

        with gr.Row():
            label_radio = gr.Radio(
                choices=label_choices,
                label="Orthography label",
                info=" · ".join(f"**{k}**: {v}" for k, v in LABEL_HELP.items()),
            )
        with gr.Row():
            submit_btn = gr.Button("Submit annotation", variant="primary")
            skip_btn = gr.Button("Skip / next")

        with gr.Accordion("Environment (read-only)", open=False):
            gr.Markdown(
                f"- `CONSENSUS_THRESHOLD` = **{settings.consensus_threshold}**\n"
                f"- `SOURCE_DATASET` = `{settings.source_dataset}`\n"
                f"- `SOURCE_TEXT_COLUMN` = `{settings.source_text_column}`\n"
                f"- `SOURCE_MAX_ROWS` = **{settings.source_max_rows}**\n"
                f"- `VOTES_DATASET` = `{settings.votes_dataset}`\n"
                f"- `CONSENSUS_DATASET` = `{settings.consensus_dataset}`\n"
                f"- Write token set: **{'yes' if settings.hf_token else 'no'}**\n"
            )

        def _require_volunteer(vid: str) -> str:
            vid = (vid or "").strip()
            if settings.require_volunteer_id and not vid:
                raise gr.Error("Please enter a volunteer id first.")
            if not vid:
                return "anonymous"
            return vid

        def do_load(vid: str):
            vid = _require_volunteer(vid)
            item = store.next_item(vid)
            rid, text, meta, hint = _fmt_item(item, settings)
            if item is None:
                return (
                    "",
                    "",
                    "",
                    meta,
                    hint,
                    _fmt_stats(store),
                    "_Queue empty for this volunteer._",
                    None,
                )
            return (
                rid,
                text,
                text,
                meta,
                hint,
                _fmt_stats(store),
                f"_Loaded item for **{vid}**._",
                None,
            )

        def do_submit(vid: str, rid: str, text: str, label: str | None):
            vid = _require_volunteer(vid)
            if not rid or not text:
                raise gr.Error("Load an item before submitting.")
            if not label:
                raise gr.Error("Pick a label.")
            result = store.submit_vote(vid, rid, text, label)
            if not result["ok"]:
                # Still advance if sealed / duplicate.
                item = store.next_item(vid)
                rid2, text2, meta, hint = _fmt_item(item, settings)
                return (
                    rid2,
                    text2,
                    text2,
                    meta,
                    hint,
                    _fmt_stats(store),
                    f"⚠ {result['message']}",
                    None,
                )
            item = store.next_item(vid)
            rid2, text2, meta, hint = _fmt_item(item, settings)
            return (
                rid2,
                text2,
                text2,
                meta,
                hint,
                _fmt_stats(store),
                f"✓ {result['message']}",
                None,
            )

        def do_skip(vid: str):
            return do_load(vid)

        outs = [
            row_id_state,
            text_state,
            text_box,
            meta_md,
            hint_md,
            stats_md,
            status_md,
            label_radio,
        ]
        load_btn.click(do_load, inputs=[volunteer], outputs=outs)
        skip_btn.click(do_skip, inputs=[volunteer], outputs=outs)
        submit_btn.click(
            do_submit,
            inputs=[volunteer, row_id_state, text_state, label_radio],
            outputs=outs,
        )

    return demo


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    settings = load_settings()
    store = AnnotationStore(settings)
    boot = store.bootstrap()
    logger.info(boot)
    demo = build_app(settings, store)
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
