#!/usr/bin/env python3
"""
 ██╗  ██╗██╗██╗   ██╗███████╗
 ██║  ██║██║██║   ██║██╔════╝
 ███████║██║██║   ██║█████╗
 ██╔══██║██║╚██╗ ██╔╝██╔══╝
 ██║  ██║██║ ╚████╔╝ ███████╗
 ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝  chat
"""

import json, sys, os, time, threading, urllib.request, urllib.error, codecs
from collections import deque
from datetime import datetime

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.columns import Columns
from rich.align import Align

SERVER = os.environ.get("LLAMA_URL", "http://localhost:8000")
console = Console()

# ── state ──────────────────────────────────────────
messages = []
session_tokens = 0
session_time = 0.0
session_turns = 0
model_name = ""
model_detail = ""

# ── detect model ───────────────────────────────────
def detect():
    global model_name, model_detail
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            model_name = "Qwen3.5-35B-A3B"
            model_detail = "MoE 34.7B · 3B active · IQ2_M · Metal"
        elif "27B" in alias:
            model_name = "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
            if "Q4_K_M" in alias:
                model_detail = "27B dense · Q4_K_M · Metal"
            elif "Q4_K_S" in alias:
                model_detail = "27B dense · Q4_K_S · Metal"
            elif "Q3_K_M" in alias:
                model_detail = "27B dense · Q3_K_M · Metal"
            elif "Q3_K_S" in alias:
                model_detail = "27B dense · Q3_K_S · Metal"
            elif "Q2_K" in alias:
                model_detail = "27B dense · Q2_K · Metal"
            else:
                model_detail = "27B dense · GGUF · Metal"
        elif "9B" in alias:
            model_name = "Qwen3.5-9B"
            model_detail = "8.95B dense · Q4_K_M · Metal"
        else:
            model_name = alias.replace(".gguf","").split("/")[-1]
            model_detail = "local model"
    except Exception:
        model_name = "connecting..."
        model_detail = ""

# ── streaming request ──────────────────────────────
def stream(msgs):
    payload = json.dumps({
        "model": "local",
        "messages": msgs,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full = ""
    start = time.time()
    tokens = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = ""
        decoder = codecs.getincrementaldecoder("utf-8")()
        while True:
            chunk = resp.read(1024)
            if not chunk:
                buf += decoder.decode(b"", final=True)
                break
            buf += decoder.decode(chunk)

            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    elapsed = time.time() - start
                    speed = tokens / elapsed if elapsed > 0 else 0
                    return full, tokens, elapsed, speed
                try:
                    obj = json.loads(raw)
                    delta = obj["choices"][0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        full += c
                        tokens += 1
                        yield c, None
                except Exception:
                    pass

    elapsed = time.time() - start
    speed = tokens / elapsed if elapsed > 0 else 0
    return full, tokens, elapsed, speed

# ── non-streaming fallback ─────────────────────────
def ask(msgs):
    payload = json.dumps({
        "model": "local",
        "messages": msgs,
        "max_tokens": 4096,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        d = json.loads(resp.read())
    content = d["choices"][0]["message"]["content"]
    t = d.get("timings", {})
    u = d.get("usage", {})
    tokens = u.get("completion_tokens", 0)
    speed = t.get("predicted_per_second", 0)
    elapsed = t.get("predicted_ms", 0) / 1000
    return content, tokens, elapsed, speed

# ── render helpers ─────────────────────────────────
def header():
    t = Text()
    t.append(" HIVE ", style="bold black on bright_yellow")
    t.append("  ", style="default")
    t.append(model_name, style="bold cyan")
    if model_detail:
        t.append(f"  {model_detail}", style="dim")
    return Panel(t, style="bright_yellow", height=3)

def stat_bar(tokens, elapsed, speed):
    s = Text()
    clr = "bright_green" if speed > 20 else "yellow" if speed > 10 else "red"
    s.append(f"  {speed:.1f} tok/s", style=f"bold {clr}")
    s.append(f"  ·  {tokens} tokens", style="dim")
    s.append(f"  ·  {elapsed:.1f}s", style="dim")
    return s

def session_stats():
    avg = session_tokens / session_time if session_time > 0 else 0
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan", width=14)
    t.add_column()
    t.add_row("Turns", str(session_turns))
    t.add_row("Tokens", f"{session_tokens:,}")
    t.add_row("Time", f"{session_time:.1f}s")
    t.add_row("Avg speed", f"{avg:.1f} tok/s")
    return Panel(t, title="[bold cyan]Session", border_style="cyan")

def help_panel():
    h = Text()
    h.append("/clear", style="bold cyan")
    h.append(" reset · ", style="dim")
    h.append("/stats", style="bold cyan")
    h.append(" session info · ", style="dim")
    h.append("/system", style="bold cyan")
    h.append(" <msg> set persona · ", style="dim")
    h.append("/model", style="bold cyan")
    h.append(" show model · ", style="dim")
    h.append("/quit", style="bold cyan")
    h.append(" exit", style="dim")
    return h

# ── main loop ──────────────────────────────────────
def main():
    global session_tokens, session_time, session_turns

    detect()
    console.clear()
    console.print(header())
    console.print(help_panel())
    console.print()

    while True:
        # prompt
        try:
            console.print("[bold bright_yellow]▶[/] ", end="")
            user = input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user.strip():
            continue

        cmd = user.strip().lower()

        if cmd in ("/quit", "/exit", "/q"):
            break
        elif cmd == "/clear":
            messages.clear()
            console.clear()
            console.print(header())
            console.print("[dim]  History cleared.[/]\n")
            continue
        elif cmd == "/stats":
            console.print(session_stats())
            console.print()
            continue
        elif cmd == "/model":
            detect()
            console.print(f"  [bold cyan]{model_name}[/]  [dim]{model_detail}[/]\n")
            continue
        elif cmd == "/help":
            console.print(help_panel())
            console.print()
            continue
        elif cmd.startswith("/system "):
            sys_msg = user[8:].strip()
            # Insert or replace system message
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = sys_msg
            else:
                messages.insert(0, {"role": "system", "content": sys_msg})
            console.print(f"  [dim italic]System: {sys_msg[:80]}{'...' if len(sys_msg)>80 else ''}[/]\n")
            continue

        messages.append({"role": "user", "content": user})

        # response
        console.print()
        full = ""
        tokens = 0
        elapsed = 0.0
        speed = 0.0

        try:
            # Try streaming
            gen = stream(messages)
            first = True
            result = None

            for chunk in gen:
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    text_chunk, meta = chunk
                    if text_chunk is not None:
                        if first:
                            console.print("  ", end="")
                            first = False
                        console.print(text_chunk, end="", highlight=False)
                        full += text_chunk
                elif isinstance(chunk, tuple) and len(chunk) == 4:
                    # Final return value
                    full, tokens, elapsed, speed = chunk
                    break

            if not full:
                # Streaming returned nothing, try non-streaming
                raise Exception("empty stream")

            # If stream worked but didn't give stats, estimate
            if tokens == 0:
                tokens = len(full.split())
                elapsed = 0
                speed = 0

            console.print()

        except Exception:
            # Fallback non-streaming
            try:
                console.print("  [dim]thinking...[/]", end="\r")
                full, tokens, elapsed, speed = ask(messages)
                console.print(f"  {full}")
            except Exception as e:
                console.print(f"  [bold red]Error:[/] {e}\n")
                messages.pop()  # remove failed user message
                continue

        # Stats line
        if speed > 0:
            console.print(stat_bar(tokens, elapsed, speed))
        elif tokens > 0:
            console.print(f"  [dim]{tokens} tokens[/]")
        console.print()

        messages.append({"role": "assistant", "content": full})
        session_tokens += tokens
        session_time += elapsed
        session_turns += 1

    # goodbye
    console.print()
    if session_turns > 0:
        avg = session_tokens / session_time if session_time > 0 else 0
        console.print(
            f"  [bold bright_yellow]HIVE[/] [dim]"
            f"{session_turns} turns · {session_tokens:,} tokens · {avg:.1f} avg tok/s[/]"
        )
    console.print()

if __name__ == "__main__":
    main()
