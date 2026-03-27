"""Quick dequant verification — compare our dequant against HF's quantization config."""
import torch, json
from safetensors import safe_open

def our_dequant(w, s, b):
    ww = w.to(torch.int32)
    shifts = torch.arange(0, 32, 4)
    u = (ww.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0xF
    inf = u.shape[1] * 8
    u = u.reshape(w.shape[0], inf).float()
    ng = inf // 64
    u = u.reshape(w.shape[0], ng, 64)
    dq = u * s.float().unsqueeze(-1) + b.float().unsqueeze(-1)
    return dq.reshape(w.shape[0], inf).to(torch.bfloat16)

# Load and dequant one attention weight
f = safe_open('/workspace/qwen35-122b-stream/pinned.safetensors', framework='pt', device='cpu')
w = f.get_tensor('language_model.model.layers.0.linear_attn.in_proj_qkv.weight')
s = f.get_tensor('language_model.model.layers.0.linear_attn.in_proj_qkv.scales')
b = f.get_tensor('language_model.model.layers.0.linear_attn.in_proj_qkv.biases')

dq = our_dequant(w, s, b)
print(f'Dequant: {dq.shape} mean={dq.float().mean():.8f} std={dq.float().std():.6f}')
print(f'Range: [{dq.min():.6f}, {dq.max():.6f}]')
print(f'First 10: {dq[0,:10].tolist()}')

# Check the quantization config
with open('/workspace/qwen35-122b-a10b-4bit/config.json') as cf:
    config = json.load(cf)
qc = config.get('quantization_config', config.get('quantization', {}))
print(f'\nQuantization config: {qc}')

# CRITICAL: Check if MLX uses a different dequant formula
# MLX affine: dequant = scale * packed + bias  (our formula)
# Some schemes: dequant = scale * (packed - zero_point)
# If mode is "affine" with bias, our formula should be correct
print(f'Mode: {qc.get("mode", "unknown")}')

# Now check: does HF transformers understand this quantization?
# If there's a quantization_config, HF might auto-dequant on load
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/workspace/qwen35-122b-a10b-4bit', trust_remote_code=True)
tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
if hasattr(tc, 'quantization_config'):
    print(f'HF text_config quantization_config: {tc.quantization_config}')
else:
    print('HF text_config has NO quantization_config')

# Check if the model weights in HF are supposed to be pre-dequantized
# by looking at whether the model class has special quantization handling
print(f'\nModel type: {tc.model_type}')
print(f'Architectures: {getattr(tc, "architectures", "N/A")}')
