"""Gate tests for publish_quant.py (stdlib only, no Hub, no GPU).

Run: python -m pytest test_publish_gate.py  OR  python test_publish_gate.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("publish_quant", HERE / "publish_quant.py")
pq = importlib.util.module_from_spec(spec)
sys.modules["publish_quant"] = pq
spec.loader.exec_module(pq)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class GateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.staging = self.tmp / "staging"
        self.eval_dir = self.tmp / "eval"
        self.staging.mkdir()
        self.eval_dir.mkdir()
        self.tier1 = {"BPB": 1.0, "perplexity": 10.0, "BLEU": 5.0, "chrF": 30.0,
                      "src_sha": "abc1234567"}
        (self.tmp / "tier1.json").write_text(json.dumps(self.tier1))

    def write_quant(self, quant: str, chrf: float, ppl: float) -> None:
        blob = b"GGUF" + f"fake-{quant} tokenizer.chat_template".encode()
        (self.staging / f"Wolof-Qwen3-4B-v2610-{quant}.gguf").write_bytes(blob)
        ev = {"protocol": "gguf-llama-cpp", "quant": quant, "chrF": chrf,
              "perplexity": ppl, "file_sha256": sha256(blob), "n_texts_evaluated": 4282}
        (self.eval_dir / f"{quant}_gguf_eval.json").write_text(json.dumps(ev))

    def test_missing_eval_blocks(self) -> None:
        (self.staging / "Wolof-Qwen3-4B-v2610-Q4_K_M.gguf").write_bytes(b"fake")
        with self.assertRaises(pq.GateError):
            pq.validate_gguf(self.staging, self.eval_dir, self.tier1)

    def test_stale_tier1_blocks(self) -> None:
        with self.assertRaises(pq.GateError):
            pq.check_tier1(self.tmp / "tier1.json", "ffffffff99")

    def test_passing_quant_renders_card_with_eval(self) -> None:
        self.write_quant("F16", 30.0, 10.0)
        self.write_quant("Q4_K_M", 29.0, 10.4)  # dchrF 1.0 <= 1.5, dPPL 4% <= 5%
        records = pq.validate_gguf(self.staging, self.eval_dir, self.tier1)
        card = pq.render_card(kind="gguf", dst_repo="Tonic/Wolof-Qwen3-4B-v2610-GGUF",
                              src="Orange/Wolof-Qwen3-4B-v2610", src_sha="abc1234567",
                              stem="Wolof-Qwen3-4B-v2610", tier1=self.tier1,
                              records=records, meta={})
        self.assertIn("BPB 1.0", card)
        self.assertIn("Q4_K_M", card)
        self.assertIn("model-index", card)
        uploadable = [r for r in records if r["verdict"] == "pass"]
        self.assertEqual([r["quant"] for r in uploadable], ["Q4_K_M"])  # F16 never uploaded

    def test_failing_quant_not_uploadable(self) -> None:
        self.write_quant("F16", 30.0, 10.0)
        self.write_quant("Q4_K_M", 20.0, 12.0)  # dchrF 10 >> 1.5 -> fail
        # A mixed matrix still publishes the passing quants only...
        self.write_quant("Q8_0", 29.8, 10.05)
        records = pq.validate_gguf(self.staging, self.eval_dir, self.tier1)
        verdicts = {r["quant"]: r["verdict"] for r in records}
        self.assertEqual(verdicts["Q4_K_M"], "fail")
        self.assertEqual(verdicts["Q8_0"], "pass")
        # ...but if every quant fails, the gate fails closed entirely.
        (self.eval_dir / "Q8_0_gguf_eval.json").unlink()
        (self.staging / "Wolof-Qwen3-4B-v2610-Q8_0.gguf").unlink()
        with self.assertRaises(pq.GateError):
            pq.validate_gguf(self.staging, self.eval_dir, self.tier1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
