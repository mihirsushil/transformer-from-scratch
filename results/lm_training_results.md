# End-to-End Decoder Transformer Training Results

Trained a 2-layer causal `DecoderTransformer` (`vocab_size=28`, `d_model=32`, `num_heads=4`, `seq_len=24`) for 80 epochs in 9.4s purely on the custom NumPy autograd engine.

## Loss & Next-Token Accuracy Curve

| Epoch | Cross-Entropy Loss | Next-Token Accuracy |
|---:|---:|---:|
| 1 | 2.9081 | 16.2% |
| 10 | 0.2988 | 90.8% |
| 20 | 0.1605 | 94.8% |
| 30 | 0.1340 | 95.1% |
| 40 | 0.1319 | 95.0% |
| 50 | 0.1193 | 95.4% |
| 60 | 0.1183 | 95.4% |
| 70 | 0.1134 | 95.6% |
| 80 | 0.1097 | 95.5% |

## Autoregressive Generation (Greedy Decoding, Before vs. After Training)

| Prompt | Untrained Completion | Trained Completion |
|---|---|---|
| `attention ` | `attention zbybbebbb;fe;;bbbb;b;b;yammbhg;h` | `attention mixes tokens across positions; mlp t` |
| `mlp ` | `mlp \nwu\n\n\n\n\n\n\nzfzzzzgfff\n\n\n\n.ebab;f\n` | `mlp transforms each token independently;` |
| `autograd ` | `autograd zzzz..z.zeziu.abbbybuebar;byyaae` | `autograd composes backward gradients via the ` |
