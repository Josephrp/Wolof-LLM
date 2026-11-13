"""Load the Morena base weights (morena1p5b-mid, step 75083) and sample from them.

Morena is not a transformers architecture; it is `Transformer` from modeling_morena.py, and that
class has NO .generate() and a training-shaped forward:

    forward(idx, positions, cu_seqlens, max_seqlen, mask)

For single-sequence eval pass cu_seqlens=None and mask=None, which is plain causal SDPA. The
sampler below is the one the eval harness uses (no KV cache; fine for a couple hundred tokens).

This is a BASE (completion) model with no chat markers and no instruction-following training. Prime
it with the start of a document, not a user/assistant turn -- there is no such thing here.

    pip install torch safetensors tokenizers
    python load_example.py
"""
import json, torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
import modeling_morena as M

cfg = json.load(open("config.json"))
model = M.Transformer(M.ModelConfig(**cfg["model"]), "sdpa")     # "sdpa" = plain causal attention
model.load_state_dict(load_file("model.safetensors"), strict=True)
model = model.to(torch.bfloat16).cuda().eval()
tok = Tokenizer.from_file("tokenizer.json")

EOS = tok.token_to_id("<eos>")


@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    ids = [EOS] + tok.encode(prompt, add_special_tokens=False).ids   # docs start with <eos>
    x = torch.tensor([ids], device="cuda")
    out = []
    for _ in range(max_new_tokens):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        logits = model(x, pos, None, x.shape[1], None)[:, -1, :].float()
        probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
        sp, si = torch.sort(probs, descending=True, dim=-1)
        sp[sp.cumsum(-1) - sp > top_p] = 0.0
        nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
        t = int(nxt[0, 0])
        if t == EOS:
            break
        out.append(t)
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(out, skip_special_tokens=True)


# Plain completion prompt -- this is a base model, not a chat model. No markers.
print(generate("Harare ndiro guta guru re"))
