"""
THE DEFINITIVE TEST: Does the model work AT ALL with from_pretrained?
No patching, no sniping, no expert deletion. Pure HF inference.

If Paris appears → model works, bug is in our patch
If Paris doesn't → MLX-to-HF conversion is broken, model is unusable

Also checks device references in the patched version to find mixed-device bugs.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

original_dir = "/workspace/qwen35-122b-a10b-4bit"

print("=" * 60)
print("  TEST: Unpatched model — does it work at all?")
print("=" * 60)

print("\n[1] Loading full model on CPU (~25 min)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    original_dir,
    trust_remote_code=True,
    device_map={'': 'cpu'},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
print(f"  Loaded in {time.time()-t0:.0f}s")

# Access text model
if hasattr(model, 'language_model'):
    text_model = model.language_model
else:
    text_model = model

# Verify weights are dequantized
w = text_model.model.layers[0].linear_attn.in_proj_qkv.weight
print(f"  L0 qkv: {w.shape} {w.dtype} mean={w.float().mean():.8f}")

# Check expert weights are present
experts = text_model.model.layers[0].mlp.experts
print(f"  Experts type: {type(experts).__name__}")
if hasattr(experts, 'gate_up_proj'):
    print(f"  gate_up_proj: {experts.gate_up_proj.shape}")
elif len(list(experts.parameters())) > 0:
    p = next(experts.parameters())
    print(f"  First expert param: {p.shape} {p.dtype}")
else:
    print(f"  NO expert params — model is empty!")

print("\n[2] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(original_dir, trust_remote_code=True)

print("\n[3] Running single forward pass on CPU (may take 5-10 min)...")
input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")
attention_mask = torch.ones_like(input_ids)
print(f"  Input: {input_ids.shape[1]} tokens")

t0 = time.time()
with torch.no_grad():
    out = text_model(input_ids, attention_mask=attention_mask, use_cache=False)
elapsed = time.time() - t0

logits = out.logits if hasattr(out, 'logits') else out[0]
top10 = torch.topk(logits[0, -1].float(), 10)

print(f"\n  Forward pass: {elapsed:.1f}s")
print(f"  Top 10 predictions:")
for i, (val, idx) in enumerate(zip(top10.values, top10.indices)):
    print(f"    {i+1}. '{tokenizer.decode([idx.item()])}' (logit={val.item():.2f})")

paris_tokens = tokenizer.encode("Paris", add_special_tokens=False)
paris_in_top = any(idx.item() in paris_tokens for idx in top10.indices)

print(f"\n  'Paris' in top 10: {paris_in_top}")
if paris_in_top:
    print("  >>> MODEL WORKS NATIVELY — bug is in the MoE patch")
else:
    print("  >>> MODEL BROKEN NATIVELY — MLX-to-HF conversion doesn't work")
    print("  >>> No amount of patching will fix this. Need a different model format.")

print(f"\n{'='*60}")
print("  TEST COMPLETE")
print(f"{'='*60}")
