"""
DEFINITIVE pinned weight extraction from Q4_K_M GGUF.

Extracts ALL non-expert tensors with:
1. Correct GGUF→mlx-lm name mapping (including SSM remap for linear layers)
2. Non-zero verification for every tensor
3. Proper quantization: large linear → uint32, small SSM → float16/32
4. Conv1d 2D→3D reshape
5. Single-pass — no intermediate save/load that could lose data

This replaces the broken pinned.safetensors with a consistent Q4_K_M extraction.
"""

import sys, os, gc, json
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gguf import GGUFReader
from dequant_gguf import dequantize as dequant_gguf_tensor

GGUF_PATH = "/Users/bigneek/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
OUTPUT_PATH = "/Users/bigneek/models/qwen35-35b-moe-stream/pinned.safetensors"
BITS = 4; GROUP_SIZE = 64; NUM_LAYERS = 40

# Full GGUF→MLX name mapping
GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

LAYER_MAP = {
    # Attention (full attention layers)
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # SSM / linear attention
    "attn_qkv.weight": "self_attn.qkv_proj.weight",
    "attn_gate.weight": "self_attn.attn_gate.weight",
    "ssm_a": "self_attn.a_param",
    "ssm_alpha.weight": "self_attn.alpha_proj.weight",
    "ssm_beta.weight": "self_attn.beta_proj.weight",
    "ssm_conv1d.weight": "self_attn.conv1d.weight",
    "ssm_dt.bias": "self_attn.dt_bias",
    "ssm_norm.weight": "self_attn.ssm_norm.weight",
    "ssm_out.weight": "self_attn.out_proj.weight",
    # Norms
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    # MoE routing + shared expert
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
}

# SSM remap: self_attn.* → linear_attn.* for linear layers
SSM_REMAP = {
    "self_attn.a_param": "linear_attn.A_log",
    "self_attn.alpha_proj": "linear_attn.in_proj_a",
    "self_attn.attn_gate": "linear_attn.in_proj_z",
    "self_attn.beta_proj": "linear_attn.in_proj_b",
    "self_attn.conv1d": "linear_attn.conv1d",
    "self_attn.dt_bias": "linear_attn.dt_bias",
    "self_attn.out_proj": "linear_attn.out_proj",
    "self_attn.qkv_proj": "linear_attn.in_proj_qkv",
    "self_attn.ssm_norm": "linear_attn.norm",
}

LINEAR_LAYERS = set(i for i in range(NUM_LAYERS) if (i + 1) % 4 != 0)

# Tensors that must NOT be quantized (small or 1D SSM params)
NO_QUANTIZE = {"A_log", "dt_bias", "in_proj_a", "in_proj_b", "shared_expert_gate",
               "conv1d", "norm.weight", "q_norm", "k_norm", "layernorm"}


def should_quantize(name, arr):
    """Determine if tensor should be 4-bit quantized or kept as float."""
    if arr.ndim < 2:
        return False
    if min(arr.shape) < GROUP_SIZE:
        return False
    if any(k in name for k in NO_QUANTIZE):
        return False
    return True


def main():
    print("=" * 60)
    print("  REBUILD PINNED — Q4_K_M Consistent Extraction")
    print("=" * 60)

    reader = GGUFReader(GGUF_PATH)
    print(f"\n  {len(reader.tensors)} tensors in GGUF")

    pinned = {}
    zeros = []
    extracted = 0

    for t in reader.tensors:
        name = t.name
        shape = tuple(t.shape)
        n_elements = int(np.prod(shape))
        ggml_type = int(t.tensor_type)

        # Skip expert tensors
        if "exps" in name:
            continue

        # Map GGUF name → MLX name
        if name in GLOBAL_MAP:
            mlx_name = GLOBAL_MAP[name]
        else:
            parts = name.split(".")
            if len(parts) < 3 or parts[0] != "blk":
                continue
            layer_idx = int(parts[1])
            tensor_key = ".".join(parts[2:])
            if tensor_key not in LAYER_MAP:
                print(f"  UNMAPPED: {name}")
                continue
            mlx_name = f"model.layers.{layer_idx}.{LAYER_MAP[tensor_key]}"

            # Remap for linear attention layers
            if layer_idx in LINEAR_LAYERS:
                for old, new in SSM_REMAP.items():
                    if old in mlx_name:
                        mlx_name = mlx_name.replace(old, new)
                        break

        # Dequantize from GGUF
        raw = t.data
        if ggml_type == 0:  # F32
            flat = np.frombuffer(raw.reshape(-1).tobytes(), dtype=np.float32, count=n_elements)
        elif ggml_type == 1:  # F16
            flat = np.frombuffer(raw.reshape(-1).tobytes(), dtype=np.float16, count=n_elements)
        else:
            flat = dequant_gguf_tensor(raw, ggml_type, n_elements)

        # Reshape based on tensor type
        is_1d_ssm = any(k in mlx_name for k in ["A_log", "dt_bias"])
        if is_1d_ssm and len(shape) == 1:
            arr = mx.array(flat.astype(np.float32).reshape(shape))
        elif "conv1d" in mlx_name and len(shape) == 2:
            arr = mx.array(flat.astype(np.float16).reshape(shape[1], shape[0])[:, :, None])
        elif "shared_expert_gate" in mlx_name and len(shape) == 2:
            # [ne0, ne1] = [2048, 1] → reshape to [1, 2048]
            arr = mx.array(flat.astype(np.float16).reshape(shape[1], shape[0]))
        elif len(shape) == 2 and n_elements > 100:
            # Standard 2D linear: GGUF col-major → reshape(ne1, ne0) = [out, in]
            arr = mx.array(flat.astype(np.float16).reshape(shape[1], shape[0]))
        elif len(shape) == 1:
            # Norms, small 1D tensors
            arr = mx.array(flat.astype(np.float32).reshape(shape))
        else:
            arr = mx.array(flat.astype(np.float32).reshape(shape))

        mx.eval(arr)

        # Non-zero check
        is_nonzero = mx.abs(arr.astype(mx.float32)).sum().item() > 0
        if not is_nonzero:
            zeros.append(mlx_name)

        # Quantize or store as float
        base = mlx_name.replace(".weight", "").replace(".bias", "")
        if should_quantize(mlx_name, arr):
            qw, sc, bi = mx.quantize(arr, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(qw, sc, bi)
            pinned[f"{base}.weight"] = qw
            pinned[f"{base}.scales"] = sc
            pinned[f"{base}.biases"] = bi
        else:
            pinned[mlx_name] = arr

        extracted += 1
        gc.collect()

    del reader
    gc.collect()

    print(f"\n  Extracted: {extracted} tensors → {len(pinned)} arrays")

    if zeros:
        print(f"\n  WARNING: {len(zeros)} zero tensors:")
        for z in zeros[:10]:
            print(f"    {z}")
    else:
        print(f"\n  ALL TENSORS NON-ZERO ✓")

    # Verify critical SSM tensors
    print(f"\n  === VERIFICATION ===")
    checks = [
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.dt_bias",
        "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.0.linear_attn.norm.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
    ]
    all_good = True
    for k in checks:
        if k in pinned:
            v = pinned[k]
            mx.eval(v)
            nz = mx.abs(v.astype(mx.float32)).sum().item() > 0
            print(f"  {'✓' if nz else '✗'} {k}: shape={v.shape} dtype={v.dtype} nonzero={nz}")
            if not nz:
                all_good = False
        else:
            # Check quantized version
            base = k.replace(".weight", "")
            if f"{base}.weight" in pinned:
                print(f"  ✓ {k}: QUANTIZED")
            else:
                print(f"  ✗ {k}: MISSING")
                all_good = False

    if not all_good:
        print("\n  ⚠ SOME TENSORS FAILED VERIFICATION")
    else:
        print("\n  ALL CHECKS PASSED ✓")

    # Save
    print(f"\n  Saving {len(pinned)} arrays to {OUTPUT_PATH}...")
    mx.save_safetensors(OUTPUT_PATH, pinned)

    # Post-save verification
    p2 = mx.load(OUTPUT_PATH)
    a = p2.get("model.layers.0.linear_attn.A_log")
    if a is not None:
        mx.eval(a)
        print(f"  POST-SAVE A_log: {a[:3].tolist()} (non-zero: {mx.abs(a).sum().item() > 0})")

    total = sum(v.nbytes for v in pinned.values())
    print(f"\n  Total: {total/1e9:.2f} GB")
    print("  DONE!")


if __name__ == "__main__":
    main()
