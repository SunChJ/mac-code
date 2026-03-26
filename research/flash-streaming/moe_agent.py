#!/usr/bin/env python3
"""
MoE Sniper Agent — Qwen3.5-35B-A3B (22 GB) on 16 GB Mac.

256 experts, 8 active per token. 1.42 GB RAM. 1.75 tok/s.
Router picks which experts to snipe from SSD. Full 4-bit quality.
"""

import json, sys, os, time, gc, random
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.padding import Padding

from expert_io import MoEExpertReader

console = Console()

MODEL_DIR = "/Users/bigneek/models/qwen35-35b-moe-stream"
BITS = 4
GROUP_SIZE = 64
TEMPERATURE = 0.7
TOP_P = 0.9
REP_PENALTY = 1.15
MAX_TOKENS = 1024

CREATURES = [
    ["   ⚡( ᐛ )⚡  ", "  ⚡( ᐛ )⚡   ", " ⚡( ᐛ )⚡    ", "  ⚡( ᐛ )⚡   "],
    ["  ⠋  ", "  ⠙  ", "  ⠹  ", "  ⠸  ", "  ⠼  ", "  ⠴  ", "  ⠦  ", "  ⠧  "],
]
CREATURE = CREATURES[random.randint(0, len(CREATURES) - 1)]


def run_expert_ffn(x, expert_data, top_k_indices, top_k_weights):
    B, L, D = x.shape
    K = top_k_indices.shape[-1]
    output = mx.zeros_like(x)
    for k in range(K):
        eid = top_k_indices[0, 0, k].item()
        weights = top_k_weights[..., k:k+1]
        if eid not in expert_data:
            continue
        ed = expert_data[eid]
        gate = mx.quantized_matmul(x, ed["mlp.switch_mlp.gate_proj.weight"],
            scales=ed["mlp.switch_mlp.gate_proj.scales"],
            biases=ed["mlp.switch_mlp.gate_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS)
        up = mx.quantized_matmul(x, ed["mlp.switch_mlp.up_proj.weight"],
            scales=ed["mlp.switch_mlp.up_proj.scales"],
            biases=ed["mlp.switch_mlp.up_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS)
        hidden = nn.silu(gate) * up
        expert_out = mx.quantized_matmul(hidden, ed["mlp.switch_mlp.down_proj.weight"],
            scales=ed["mlp.switch_mlp.down_proj.scales"],
            biases=ed["mlp.switch_mlp.down_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS)
        output = output + weights * expert_out
    return output


class MoESniperEngine:
    def __init__(self):
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self.num_layers = 40

    def load(self):
        with open(f"{MODEL_DIR}/config.json") as f:
            config = json.load(f)
        self.num_layers = config["num_hidden_layers"]
        streaming = config["streaming"]

        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
        args = TextModelArgs(
            model_type=config.get("model_type"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=self.num_layers,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            max_position_embeddings=config["max_position_embeddings"],
            head_dim=config.get("head_dim"),
            tie_word_embeddings=config["tie_word_embeddings"],
            num_experts=config["num_experts"],
            num_experts_per_tok=config["num_experts_per_tok"],
            shared_expert_intermediate_size=config["shared_expert_intermediate_size"],
            moe_intermediate_size=config["moe_intermediate_size"],
            linear_num_value_heads=config.get("linear_num_value_heads"),
            linear_num_key_heads=config.get("linear_num_key_heads"),
            linear_key_head_dim=config.get("linear_key_head_dim"),
            linear_value_head_dim=config.get("linear_value_head_dim"),
            linear_conv_kernel_dim=config.get("linear_conv_kernel_dim"),
            full_attention_interval=config.get("full_attention_interval"),
            rope_parameters=config.get("rope_parameters"),
        )

        self.model = TextModel(args)

        SSM_PROTECT = {"conv1d"}
        def should_quantize(path, module):
            if isinstance(module, nn.Embedding):
                return True
            if not isinstance(module, nn.Linear):
                return False
            if any(k in path for k in SSM_PROTECT):
                return False
            if module.weight.shape[-1] < GROUP_SIZE:
                return False
            return True

        nn.quantize(self.model, group_size=GROUP_SIZE, bits=BITS, class_predicate=should_quantize)

        mx.set_memory_limit(10 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)

        pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
        self.model.load_weights(list(pinned.items()), strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters()) if "switch_mlp" not in name]
        mx.eval(*params)
        del pinned
        gc.collect()
        mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9
        self.reader = MoEExpertReader(f"{MODEL_DIR}/{streaming['expert_dir']}", self.num_layers, num_workers=8)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)

        return pinned_gb

    def reset_cache(self):
        self.cache = self.model.make_cache()

    def forward(self, input_ids):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        h = self.model.model.embed_tokens(input_ids)
        ssm_idx = self.model.model.ssm_idx
        fa_idx = self.model.model.fa_idx
        fa_mask = create_attention_mask(h, self.cache[fa_idx])
        ssm_mask = create_ssm_mask(h, self.cache[ssm_idx])

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=self.cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=self.cache[i])
            h = h + attn_out
            mx.eval(h)

            normed = layer.post_attention_layernorm(h)
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)

            active_ids = list(set(int(e) for e in np.array(inds).flatten()))
            if i + 1 < self.num_layers:
                self.reader.prefetch_experts(i + 1, active_ids)

            expert_data = self.reader.get_experts(i, active_ids)
            expert_out = run_expert_ffn(normed, expert_data, inds, scores)

            shared_out = layer.mlp.shared_expert(normed)
            shared_gate = mx.sigmoid(layer.mlp.shared_expert_gate(normed))
            if shared_gate.ndim < shared_out.ndim:
                shared_gate = shared_gate[..., None]
            expert_out = expert_out + shared_gate * shared_out

            h = h + expert_out
            mx.eval(h)
            del expert_data, expert_out, normed, attn_out
            mx.clear_cache()

        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    def sample(self, logits, generated):
        next_logits = logits[:, -1, :]
        if generated:
            seen = mx.array(list(set(generated[-100:])))
            pl = next_logits[:, seen]
            pl = mx.where(pl > 0, pl / REP_PENALTY, pl * REP_PENALTY)
            next_logits[:, seen] = pl
        probs = mx.softmax(next_logits / TEMPERATURE, axis=-1)
        sorted_idx = mx.argsort(-probs, axis=-1)
        sorted_p = mx.take_along_axis(probs, sorted_idx, axis=-1)
        cumsum = mx.cumsum(sorted_p, axis=-1)
        mask = (cumsum - sorted_p) <= TOP_P
        sorted_p = sorted_p * mask
        sorted_p = sorted_p / (sorted_p.sum(axis=-1, keepdims=True) + 1e-10)
        token = mx.random.categorical(mx.log(sorted_p + 1e-10))
        token = mx.take_along_axis(sorted_idx, token[:, None], axis=-1).squeeze(-1)
        mx.eval(token)
        return token.item()

    def generate_stream(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])
        logits = self.forward(input_ids)
        mx.eval(logits)
        generated = []
        for _ in range(MAX_TOKENS):
            token_id = self.sample(logits, generated)
            if token_id in (248044, 248045):
                break
            generated.append(token_id)
            yield self.tokenizer.decode([token_id])
            logits = self.forward(mx.array([[token_id]]))
            mx.eval(logits)


class ThinkingDisplay:
    def __init__(self):
        self.frame = 0
        self.start = time.time()
    def render(self):
        self.frame += 1
        cf = CREATURE[self.frame % len(CREATURE)]
        t = Text()
        t.append(f"  {cf}", style="bright_cyan")
        t.append("  sniping experts from SSD", style="bold bright_cyan")
        t.append(f"  {time.time()-self.start:.0f}s", style="dim")
        return t


def print_banner(pinned_gb):
    console.print()
    logo = Text()
    logo.append("  moe", style="bold bright_cyan")
    logo.append("-", style="dim")
    logo.append("sniper", style="bold bright_yellow")
    console.print(logo)
    sub = Text()
    sub.append("  22 GB 35B model · 1.4 GB RAM · experts sniped from SSD", style="dim italic")
    console.print(sub)
    console.print()
    rows = [
        ("model", "Qwen3.5-35B-A3B", "Q4_K_M · 256 experts · 8 active"),
        ("pinned", f"{pinned_gb:.1f} GB", "attention + SSM + router + shared expert"),
        ("sniped", "18.1 GB", "8/256 experts from SSD per token"),
        ("speed", "~1.75 tok/s", "8-thread F_NOCACHE pread"),
        ("cost", "$0.00/hr", "Apple Silicon Metal · local"),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append(f"  {label:8s} ", style="bold dim")
        line.append(value, style="bold white")
        if extra:
            line.append(f"  {extra}", style="dim")
        console.print(line)
    console.print()
    console.print(Rule(style="dim"))
    console.print()


def main():
    console.clear()
    with console.status("[bold bright_cyan]  Loading MoE Sniper engine...", spinner="dots"):
        engine = MoESniperEngine()
        pinned_gb = engine.load()
    print_banner(pinned_gb)

    messages = [{"role": "system", "content":
        "You are a helpful AI assistant powered by the MoE Expert Sniper engine — "
        "a 22 GB model streaming from SSD on a 16 GB Mac. Be concise and helpful."}]
    session_tokens = 0
    session_time = 0.0

    while True:
        try:
            console.print("  [bold bright_yellow]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]goodbye.[/]\n")
            break

        if not user_input.strip():
            continue
        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        elif cmd == "/clear":
            messages = [messages[0]]
            engine.reset_cache()
            console.clear()
            print_banner(pinned_gb)
            continue
        elif cmd == "/stats":
            avg = session_tokens / session_time if session_time > 0 else 0
            mem = mx.get_active_memory() / 1e9
            t = Table(show_header=False, box=None, padding=(0, 1))
            t.add_column(style="bold bright_cyan", width=12)
            t.add_column()
            t.add_row("tokens", f"{session_tokens:,}")
            t.add_row("time", f"{session_time:.1f}s")
            t.add_row("speed", f"{avg:.2f} tok/s")
            t.add_row("memory", f"{mem:.2f} GB")
            console.print(t)
            console.print()
            continue
        elif cmd in ("/help", "/?"):
            for c, d in [("/clear", "Reset"), ("/stats", "Stats"), ("/quit", "Exit")]:
                console.print(f"  [bold bright_cyan]{c:10s}[/] [dim]{d}[/]")
            console.print()
            continue

        messages.append({"role": "user", "content": user_input})
        engine.reset_cache()
        console.print()

        display = ThinkingDisplay()
        full = ""
        tokens = 0
        start = time.time()
        first_token = True
        in_think = False

        with Live(display.render(), console=console, refresh_per_second=6, transient=True) as live:
            for chunk in engine.generate_stream(messages):
                if first_token:
                    first_token = False
                    live.stop()
                    console.print("  ", end="")

                # Handle <think> tags — show thinking in dim
                if "<think>" in chunk:
                    in_think = True
                    chunk = chunk.replace("<think>", "")
                if "</think>" in chunk:
                    in_think = False
                    chunk = chunk.replace("</think>", "\n  ")

                if in_think:
                    console.print(chunk, end="", style="dim italic", highlight=False)
                else:
                    console.print(chunk, end="", highlight=False)

                full += chunk
                tokens += 1

        elapsed = time.time() - start
        if elapsed > 0 and tokens > 0:
            speed = tokens / elapsed
            clr = "bright_green" if speed > 1 else "yellow"
            s = Text()
            s.append(f"\n  {speed:.2f} tok/s", style=f"bold {clr}")
            s.append(f"  ·  {tokens} tokens  ·  {elapsed:.1f}s", style="dim")
            console.print(s)
        console.print()

        messages.append({"role": "assistant", "content": full})
        session_tokens += tokens
        session_time += elapsed
        if len(messages) > 20:
            messages = [messages[0]] + messages[-10:]


if __name__ == "__main__":
    main()
