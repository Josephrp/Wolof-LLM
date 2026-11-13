"""H200 / torch-2.11+cu128 attention + Liger helpers for Qwen3.5.

Call signature matches existing CPT/SFT scripts:

    model = load_causal_lm(..., quant=..., attn_implementation=...)
    # model._qwen_kernel_meta = {"attn_implementation": ..., "liger": ...}
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Any


def preferred_attn_chain() -> list[str]:
    override = os.environ.get("QWEN_ATTN_IMPL", "").strip()
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    return [
        "flash_attention_3",
        "kernels-community/vllm-flash-attn3",
        "flash_attention_2",
        "sdpa",
    ]


def resolve_attn_implementation(preferred: str | None = None) -> str:
    chain = [preferred] if preferred else preferred_attn_chain()
    chain = [c for c in chain if c]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in chain:
        if item not in seen:
            seen.add(item)
            ordered.append(item)

    for name in ordered:
        if name == "sdpa":
            return "sdpa"
        if name.startswith("kernels-community/") or name.startswith("kernels-"):
            try:
                import kernels  # noqa: F401

                return name
            except ImportError:
                continue
        if name in {"flash_attention_3", "flash_attn_3"}:
            try:
                import flash_attn_3  # noqa: F401

                return "flash_attention_3"
            except ImportError:
                try:
                    import kernels  # noqa: F401

                    return "kernels-community/vllm-flash-attn3"
                except ImportError:
                    continue
        if name in {"flash_attention_2", "flash_attn_2", "flash_attention"}:
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except ImportError:
                continue
        return name
    return "sdpa"


@contextmanager
def sdpa_cudnn_priority():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION],
            set_priority=True,
        ):
            yield
    except Exception:  # noqa: BLE001
        yield


def attention_context(attn_implementation: str):
    if attn_implementation == "sdpa":
        return sdpa_cudnn_priority()
    return nullcontext()


def prefer_sft_packing(*, smoke: bool, config_packing: bool | None) -> bool:
    """Enable packing when FA3/FLA path is likely available (skip on smoke)."""
    if smoke:
        return False
    if config_packing is not None:
        return bool(config_packing)
    # Default on for production once kernels are present.
    try:
        import kernels  # noqa: F401

        return True
    except ImportError:
        return False


def apply_liger_qwen35() -> str:
    if os.environ.get("QWEN_DISABLE_LIGER") == "1":
        return "liger_disabled"
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
    except ImportError:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3

            apply_liger_kernel_to_qwen3(
                rms_norm=True,
                swiglu=True,
                fused_linear_cross_entropy=True,
                rope=False,
            )
            return "liger:qwen3"
        except Exception:  # noqa: BLE001
            return "liger_unavailable"
    try:
        apply_liger_kernel_to_qwen3_5(
            rms_norm=True,
            swiglu=True,
            fused_linear_cross_entropy=True,
            rope=False,
        )
        return "liger:qwen3_5"
    except TypeError:
        apply_liger_kernel_to_qwen3_5()
        return "liger:qwen3_5:default"
    except Exception as exc:  # noqa: BLE001
        return f"liger_failed:{type(exc).__name__}"


def maybe_use_liger_training_arg(train_args: dict[str, Any], params) -> dict[str, Any]:
    if os.environ.get("QWEN_DISABLE_LIGER") == "1":
        return train_args
    if "use_liger_kernel" not in set(params):
        return train_args
    try:
        import liger_kernel  # noqa: F401
    except ImportError:
        return train_args
    out = dict(train_args)
    out.setdefault("use_liger_kernel", True)
    return out


def probe_gdn_kernels() -> dict[str, bool]:
    """Qwen3.5 hybrid GDN companions (FA3 only covers full-attn layers)."""

    def ok(name: str) -> bool:
        try:
            __import__(name)
            return True
        except Exception:  # noqa: BLE001
            return False

    status = {
        "flash_linear_attention": ok("fla") or ok("flash_linear_attention"),
        "causal_conv1d": ok("causal_conv1d"),
        "flash_qla": ok("flash_qla") or ok("flashqla"),
        "kernels": ok("kernels"),
        "liger_kernel": ok("liger_kernel"),
    }
    for key, present in status.items():
        print(f"  gdn_kernel {key}: {'ok' if present else 'missing'}", flush=True)
    return status


def enable_cuda_speedups() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def load_causal_lm(
    model_id: str,
    *,
    token: str | None = None,
    quant: Any = None,
    quantization_config: Any = None,
    torch_dtype: Any = None,
    device_map: Any = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
    apply_liger: bool = True,
):
    """Load Qwen3.5 with FA3→FA2→SDPA fallback; attach ``_qwen_kernel_meta``."""
    from transformers import AutoModelForCausalLM

    enable_cuda_speedups()
    gdn = probe_gdn_kernels()
    quant_cfg = quantization_config if quantization_config is not None else quant
    chain = preferred_attn_chain()
    if attn_implementation:
        chain = [attn_implementation, *[c for c in chain if c != attn_implementation]]

    last_error: Exception | None = None
    model = None
    used_attn = "sdpa"
    for attn in chain + (["sdpa"] if "sdpa" not in chain else []):
        try:
            kwargs: dict[str, Any] = {
                "token": token,
                "torch_dtype": torch_dtype,
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                "attn_implementation": attn,
            }
            if quant_cfg is not None:
                kwargs["quantization_config"] = quant_cfg
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            used_attn = attn
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if model is None:
        raise RuntimeError(f"failed to load {model_id} with any attn backend") from last_error

    meta = {
        "attn_implementation": used_attn,
        "liger": apply_liger_qwen35() if apply_liger else "skipped",
        "gdn": gdn,
    }
    model._qwen_kernel_meta = meta
    print(f"  kernels: attn={meta['attn_implementation']} liger={meta['liger']}", flush=True)
    return model
