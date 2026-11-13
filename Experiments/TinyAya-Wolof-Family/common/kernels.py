"""H200 / torch-2.11+cu128 kernels for Tiny Aya (Cohere2): FA3 + Cut Cross-Entropy."""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator


def enable_cuda_speedups() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def probe_kernels() -> dict[str, bool]:
    status = {
        "kernels": _can_import("kernels"),
        "flash_attn": _can_import("flash_attn"),
        "flash_attn_3": _can_import("flash_attn_3") or _can_import("flash_attn_interface"),
        "cut_cross_entropy": _can_import("cut_cross_entropy"),
        "liger_kernel": _can_import("liger_kernel"),
    }
    for key, ok in status.items():
        print(f"  kernel {key}: {'ok' if ok else 'missing'}", flush=True)
    return status


def resolve_attn_implementation(requested: str | None = None) -> str:
    """Cohere2 sliding-window + GQA: FA3 preferred on Hopper."""
    demandee = (
        (requested or "").strip()
        or os.environ.get("TINYAYA_ATTN_IMPLEMENTATION", "").strip()
        or os.environ.get("ATTN_IMPLEMENTATION", "").strip()
        or "auto"
    )
    if demandee not in {"", "auto"}:
        print(f"  attn_implementation={demandee} (forced)", flush=True)
        return demandee

    # Prefer Hub FA3 when only `kernels` is installed (no local flash-attn-3 wheel).
    if _can_import("flash_attn_3") or _can_import("flash_attn_interface"):
        print("  attn_implementation=flash_attention_3", flush=True)
        return "flash_attention_3"
    if _can_import("kernels"):
        print("  attn_implementation=kernels-community/vllm-flash-attn3", flush=True)
        return "kernels-community/vllm-flash-attn3"
    if _can_import("flash_attn"):
        print("  attn_implementation=flash_attention_2", flush=True)
        return "flash_attention_2"
    print("  attn_implementation=sdpa (fallback)", flush=True)
    return "sdpa"


@contextlib.contextmanager
def attention_context(attn_implementation: str) -> Iterator[None]:
    """Prefer cuDNN/Flash SDPA ordering when falling back to stock SDPA."""
    enable_cuda_speedups()
    if attn_implementation != "sdpa":
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
            set_priority=True,
        ):
            yield
    except Exception:  # noqa: BLE001
        yield


def apply_fused_ce_cohere2(*, apply_cce: bool = True) -> dict[str, Any]:
    """No Liger Cohere2 auto-patch — Cut Cross-Entropy is the 262k-vocab win."""
    meta: dict[str, Any] = {
        "cut_cross_entropy": False,
        "liger_manual": False,
        "apply_cce_requested": apply_cce,
    }
    if not apply_cce or os.environ.get("TINYAYA_DISABLE_FUSED_CE") == "1":
        print("  fused_ce: disabled", flush=True)
        return meta
    if _can_import("cut_cross_entropy"):
        meta["cut_cross_entropy"] = True
        print("  fused_ce: cut-cross-entropy available (262k vocab)", flush=True)
        # Best-effort transformers patch (API names vary by CCE version / Axolotl forks).
        for attr in ("cce_apply_to_model", "cce_patch"):
            try:
                mod = __import__("cut_cross_entropy.transformers", fromlist=[attr])
                fn = getattr(mod, attr, None)
                if callable(fn):
                    meta["cce_apply"] = fn
                    meta["cce_api"] = attr
                    break
            except Exception:  # noqa: BLE001
                continue
        else:
            meta["cce_apply"] = None
        return meta
    try:
        from liger_kernel.transformers.functional import (  # noqa: F401
            liger_fused_linear_cross_entropy,
        )

        meta["liger_manual"] = True
        print(
            "  fused_ce: liger FLCE functional available "
            "(no apply_liger_kernel_to_cohere2)",
            flush=True,
        )
    except ImportError:
        print("  fused_ce: neither cut-cross-entropy nor liger FLCE installed", flush=True)
    return meta


def prefer_sft_packing(*, configured: bool, smoke: bool) -> bool:
    """Enable packing only when explicitly requested.

    Auto-enabling under FA3 breaks Tiny Aya SFT: we pass a custom
    ``DataCollatorForSeq2Seq`` for pretokenized assistant-only labels, and TRL
    rejects that combination with padding-free packing.
    """
    del smoke  # kept for call-site compatibility
    if configured:
        return True
    return os.environ.get("TINYAYA_SFT_PACKING") == "1"


def load_causal_lm(
    model_id: str,
    *,
    token: str | None = None,
    torch_dtype: Any = None,
    device_map: Any = None,
    quantization_config: Any = None,
    quant: Any = None,
    attn_implementation: str | None = None,
    apply_cce: bool = True,
    trust_remote_code: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Load CausalLM with FA3 (or fallback) and optional CCE hook metadata."""
    import torch
    from transformers import AutoModelForCausalLM

    enable_cuda_speedups()
    probe = probe_kernels()
    fused = apply_fused_ce_cohere2(apply_cce=apply_cce)
    attn = resolve_attn_implementation(attn_implementation)
    quant_cfg = quantization_config if quantization_config is not None else quant
    kwargs: dict[str, Any] = {
        "token": token,
        "quantization_config": quant_cfg,
        "torch_dtype": torch_dtype if torch_dtype is not None else torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if attn == "sdpa":
            raise
        print(f"  attn {attn} failed ({exc}); falling back to sdpa", flush=True)
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        attn = "sdpa"

    cce_apply = fused.get("cce_apply")
    if callable(cce_apply):
        try:
            # cce_patch(model) vs cce_patch("llama") — try model object first.
            patched = cce_apply(model)
            if patched is not None:
                model = patched
            fused["cce_applied"] = True
            print(f"  fused_ce: {fused.get('cce_api', 'cce')} applied", flush=True)
        except Exception as exc:  # noqa: BLE001
            fused["cce_applied"] = False
            print(
                f"  fused_ce: patch unavailable under this transformers "
                f"({type(exc).__name__}: {exc}); continuing without CCE",
                flush=True,
            )

    meta = {
        "attn_implementation": attn,
        "fused_ce": fused,
        "probe": probe,
    }
    return model, meta
