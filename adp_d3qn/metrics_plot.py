import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def _smooth(x, k=5):
    if len(x) < k: return x
    return np.convolve(x, np.ones(k)/k, mode="valid")

def plot_convergence(results_root="results", variants=None, out_prefix="Fig7"):
    if variants is None:
        variants = ["D3QN", "A-D3QN", "DP-D3QN", "ADP-D3QN"]

    # Reward
    plt.figure(figsize=(7,4))
    for v in variants:
        p = os.path.join(results_root, v, "reward_log.csv")
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        y = _smooth(df["reward"].values, 5)
        plt.plot(y, label=v, linewidth=1.8)
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_root, f"{out_prefix}_a_Reward.png"), dpi=300)

    # Latency
    plt.figure(figsize=(7,4))
    for v in variants:
        p = os.path.join(results_root, v, "latency_log.csv")
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        y = _smooth(df["latency_ms"].values, 5)
        plt.plot(y, label=v, linewidth=1.8)
    plt.xlabel("Episode"); plt.ylabel("Latency (ms)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_root, f"{out_prefix}_b_Latency.png"), dpi=300)
    print(f"[✓] Saved {out_prefix}_a_Reward.png / {out_prefix}_b_Latency.png under {results_root}")
