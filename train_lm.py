"""Train a character-level GPT-style DecoderTransformer on a compact sequence corpus.

Demonstrates end-to-end learning on the custom NumPy autograd engine:
- Token + learned positional embeddings (`Embedding` with `np.add.at` backward)
- 2-layer pre-norm causal `DecoderTransformer` (`d_model=32`, `num_heads=4`)
- Next-token cross-entropy loss (`sequence_cross_entropy_loss`) + SGD with momentum
- Saves training curve & before/after generated completions to `results/lm_training_results.md`
"""

from __future__ import annotations

import os
import time
import numpy as np

from nn.attention import DecoderTransformer
from nn.layers import sequence_cross_entropy_loss
from nn.optim import SGD

CORPUS = (
    "attention mixes tokens across positions; "
    "mlp transforms each token independently; "
    "layernorm stabilizes residual activations; "
    "autograd composes backward gradients via the chain rule.\n"
) * 6


def build_dataset(text: str, seq_len: int):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    xs, ys = [], []
    for start in range(0, len(data) - seq_len - 1, 3):
        xs.append(data[start : start + seq_len])
        ys.append(data[start + 1 : start + seq_len + 1])
    return np.stack(xs), np.stack(ys), stoi, itos


def decode(ids, itos) -> str:
    return "".join(itos[i] for i in ids)


def main():
    np.random.seed(7)
    seq_len = 24
    x_all, y_all, stoi, itos = build_dataset(CORPUS, seq_len=seq_len)
    vocab_size = len(stoi)

    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=seq_len,
        mlp_ratio=2,
    )
    optimizer = SGD(model.parameters(), lr=0.08, momentum=0.9)

    prompts = ["attention ", "mlp ", "autograd "]
    untrained_samples = {
        p: decode(
            model.generate([stoi[c] for c in p], max_new_tokens=32, temperature=0.0),
            itos,
        )
        for p in prompts
    }

    batch_size = 16
    epochs = 80
    history: list[tuple[int, float, float]] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(x_all))
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for i in range(0, len(x_all), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_all[idx], y_all[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = sequence_cross_entropy_loss(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.data))
            preds = logits.data.argmax(axis=-1)
            epoch_correct += int((preds == yb).sum())
            epoch_total += yb.size

        mean_loss = float(np.mean(epoch_losses))
        acc = 100.0 * epoch_correct / epoch_total
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append((epoch, mean_loss, acc))
            print(f"epoch {epoch:3d}/{epochs} | loss {mean_loss:.4f} | next-char acc {acc:5.1f}%")

    elapsed = time.perf_counter() - t0

    trained_samples = {
        p: decode(
            model.generate([stoi[c] for c in p], max_new_tokens=36, temperature=0.0),
            itos,
        )
        for p in prompts
    }

    os.makedirs("results", exist_ok=True)
    lines = [
        "# End-to-End Decoder Transformer Training Results",
        "",
        f"Trained a 2-layer causal `DecoderTransformer` (`vocab_size={vocab_size}`, `d_model=32`, "
        f"`num_heads=4`, `seq_len={seq_len}`) for {epochs} epochs in {elapsed:.1f}s purely on the "
        "custom NumPy autograd engine.",
        "",
        "## Loss & Next-Token Accuracy Curve",
        "",
        "| Epoch | Cross-Entropy Loss | Next-Token Accuracy |",
        "|---:|---:|---:|",
    ]
    for ep, l_val, a_val in history:
        lines.append(f"| {ep} | {l_val:.4f} | {a_val:.1f}% |")

    lines += [
        "",
        "## Autoregressive Generation (Greedy Decoding, Before vs. After Training)",
        "",
        "| Prompt | Untrained Completion | Trained Completion |",
        "|---|---|---|",
    ]
    for p in prompts:
        u = untrained_samples[p].replace("\n", "\\n")
        tr = trained_samples[p].replace("\n", "\\n")
        lines.append(f"| `{p}` | `{u}` | `{tr}` |")

    out_path = os.path.join("results", "lm_training_results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved training report -> {out_path}")


if __name__ == "__main__":
    main()
