#!/usr/bin/env python3
"""
Noel Inference Engine: 27B Out-of-Core GGUF Execution Runner
Target Model: DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF
Model File: Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-IQ4_XS.gguf
"""

import sys
import os
import subprocess
import time

DEFAULT_GGUF_PATH = (
    "/home/neymar-jr/.cache/huggingface/hub/"
    "models--DavidAU--Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF/"
    "snapshots/e2d64985a4a5de0ff2ba67bc8a026f9531689c75/"
    "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-IQ4_XS.gguf"
)

def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GGUF_PATH
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a Python function to compute Fibonacci numbers"
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    if not os.path.exists(gguf_path):
        print(f"[Error] Model file not found at: {gguf_path}")
        sys.exit(1)

    file_size_gb = os.path.getsize(gguf_path) / (1024**3)

    print("=" * 65)
    print("   NOEL INFERENCE ENGINE: 27B HETEROGENEOUS INFERENCE RUNNER    ")
    print("   4-Tier Memory Hierarchy: VRAM + DDR5 + NVMe Direct I/O       ")
    print("=" * 65)
    print(f" • Model Target:      Qwen3.8-27B-TurboFCFusion (DavidAU)")
    print(f" • File Path:          {gguf_path}")
    print(f" • Model File Size:    {file_size_gb:.2f} GiB")
    print(f" • Parameter Count:    27 Billion (Dense Hybrid Transformer + SSM)")
    print(f" • Dimensions:         Hidden Dim=5120, Intermediate=17408, Layers=64")
    print(f" • Attention:          Heads=24 (KV=4), Vocab=248,320")
    print("=" * 65)

    noel_bin = os.path.abspath("./noel")
    if not os.path.exists(noel_bin):
        print("[Noel] Compiling Noel binaries...")
        subprocess.run(["make", "-j"], check=True)

    cmd = [
        noel_bin,
        "--model", gguf_path,
        "--dim", "5120",
        "--hidden-dim", "17408",
        "--layers", "64",
        "--heads", "24",
        "--kv-heads", "4",
        "--vocab", "248320",
        "--experts", "1",
        "--tokens", str(tokens),
        "--prompt", prompt
    ]

    print(f"\n[Noel Executing] {' '.join(cmd[:4])} ... [flags]")
    t0 = time.time()
    res = subprocess.run(cmd)
    t1 = time.time()

    if res.returncode == 0:
        print(f"\n[Execution Completed Successfully in {t1 - t0:.2f} seconds]")
    else:
        print(f"\n[Execution exited with code {res.returncode}]")

if __name__ == "__main__":
    main()
