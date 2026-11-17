"""One-off generator for eval configs (kept so configs can be regenerated)."""
import pathlib

base = pathlib.Path(__file__).resolve().parents[2] / "configs"
sizes = [
    ("06b", "0.6B", "Orange/Wolof-Qwen3-0.6B-v2610", "Qwen/Qwen3-0.6B"),
    ("17b", "1.7B", "Orange/Wolof-Qwen3-1.7B-v2610", "Qwen/Qwen3-1.7B"),
    ("4b", "4B", "Orange/Wolof-Qwen3-4B-v2610", "Qwen/Qwen3-4B"),
    ("8b", "8B", "Orange/Wolof-Qwen3-8B-v2610", "Qwen/Qwen3-8B"),
    ("14b", "14B", "Orange/Wolof-Qwen3-14B-v2610", "Qwen/Qwen3-14B"),
]
tmpl = """# {name} - Tier-1 HF CLM eval (BPB/PPL/BLEU/chrF), protocol: Evaluation/scripts/evaluate_clm.py
corpus_csv: "../data/corpus_test_clm.csv"
text_column: "text"
n_texts: null
apply_verbalization: true
bpb_max_len: 1024
bleu_n_texts: 100
bleu_seed: 0
bleu_prompt_ratio: 0.5
bleu_length_ratio: 1.3
bleu_min_new_tokens: 16
bleu_max_new_tokens: 256
base_model_id: "{hub}"
subfolder: null
tokenizer_subfolder: null
output_json: "../metrics/clm/{out}.json"
"""
for key, label, hub, baseid in sizes:
    (base / "eval/clm" / f"orange_wolof_qwen3_{key}.yaml").write_text(
        tmpl.format(name=f"Orange Wolof-Qwen3-{label}", hub=hub, out=f"orange_wolof_qwen3_{key}"), encoding="utf-8")
    (base / "eval/clm" / f"qwen3_{key}.yaml").write_text(
        tmpl.format(name=f"Stock Qwen3-{label} baseline", hub=baseid, out=f"qwen3_{key}"), encoding="utf-8")
    if key != "17b":
        gguf = f"""# GGUF Q4_K_M eval for Wolof-Qwen3-{label} - protocol: Evaluation/scripts/evaluate_gguf.py (gguf-llama-cpp)
run: Wolof-Qwen3-{label}-v2610-Q4_K_M
gguf_path: "/tmp/gguf/Wolof-Qwen3-{label}-v2610-Q4_K_M.gguf"
quant: Q4_K_M
hf_repo: Tonic/Wolof-Qwen3-{label}-v2610-GGUF
hf_clm_ref: "../metrics/clm/orange_wolof_qwen3_{key}.json"
corpus_csv: "../data/corpus_test_clm.csv"
text_column: "text"
n_texts: null
apply_verbalization: true
ctx_size: 1024
n_gpu_layers: 99
bleu_n_texts: 100
bleu_seed: 0
bleu_prompt_ratio: 0.5
bleu_length_ratio: 1.3
bleu_min_new_tokens: 16
bleu_max_new_tokens: 256
output_json: "../metrics/gguf/orange_wolof_qwen3_{key}_q4_k_m.json"
"""
        (base / "eval/gguf" / f"orange_wolof_qwen3_{key}_q4_k_m.yaml").write_text(gguf, encoding="utf-8")
print("configs written")
