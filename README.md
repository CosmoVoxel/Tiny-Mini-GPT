# Tiny-Mini-GPT

A GPT-2 implementation from scratch using PyTorch Lightning.

---

## Architecture

| Parameter | Value |
|---|---|
| Decoder layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| FFN dimension | 3072 |
| Max sequence length | 1024 |
| Vocabulary size | 50,304 |
| Tokenizer | `r50k_base` (tiktoken / GPT-2) |
| Attention dropout | 15 % |
| FFN dropout | 15 % |
| Attention output dropout | 13 % |
| FFN output dropout | 13 % |
| Output dropout | 10 % |

---

## Technical Highlights

- **Decoder-only Transformer** — standard GPT-2 architecture with pre-norm residual blocks.
- **RMSNorm** — Root Mean Square Layer Normalisation used in place of LayerNorm (pre-norm, applied before each sub-layer).
- **Multi-head causal self-attention** via PyTorch `scaled_dot_product_attention` — automatically uses Flash Attention when available.
- **Sinusoidal positional encoding** — generated dynamically at runtime; no learned position embeddings.
- **BFloat16 mixed precision** — token generation uses `torch.autocast('cuda', dtype=torch.bfloat16)`.
- **`torch.compile`** with `fullgraph=True, mode='max-autotune'` for maximum kernel fusion (enabled by default, disable with `--no-compile`).
- **GrokFast** gradient filtering (`gradfilter_ema`) applied during the backward pass.

---

## Tech Stack

- [PyTorch](https://pytorch.org/) — neural network framework
- [PyTorch Lightning](https://lightning.ai/) — training loop, checkpointing, and multi-GPU support
- [tiktoken](https://github.com/openai/tiktoken) — fast BPE tokenisation (`r50k_base` / `cl100k_base` / `p50k_base`)
- [TensorBoard](https://www.tensorflow.org/tensorboard) — loss, accuracy, gradient norm, and learning-rate logging
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — streaming FineWeb-Edu dataset
- [NumPy](https://numpy.org/) — pre-tokenised dataset storage (`.npy` files)

---

## Training

### Quick start

```bash
# War and Peace (~570 K tokens) — default
python training_GPT2.py --dataset war_and_peace --log --saving

# Russian books corpus
python training_GPT2.py --dataset russian --log --saving

# FineWeb-Edu (1 T tokens, streaming)
python training_GPT2.py --dataset finewebedu --log --saving

# Disable torch.compile (useful for debugging)
python training_GPT2.py --dataset war_and_peace --no-compile

# Resume from a checkpoint
python training_GPT2.py --dataset war_and_peace --load_checkpoint ./checkpoints/last.ckpt
```

### Key CLI options

| Flag | Default | Description |
|---|---|---|
| `-d` / `--dataset` | `war_and_peace` | Dataset to train on (`war_and_peace`, `russian`, `finewebedu`) |
| `-l` / `--log` | `False` | Enable TensorBoard logging |
| `-s` / `--saving` | `False` | Save checkpoints after each epoch |
| `-c` / `--config` | — | Path to a custom JSON config file |
| `--compile` / `--no-compile` | `True` | Toggle `torch.compile` |
| `--validation` / `--no-validation` | `True` | Run validation at the end of each epoch |
| `-i` / `--inference` | `False` | Run interactive inference after training |
| `-lc` / `--load_checkpoint` | — | Resume training from a `.ckpt` file |
| `-ts` / `--train_size` | `1.0` | Fraction of the dataset to use (0–1) |

### Default hyperparameters (`configs/config.json`)

| Parameter | Value |
|---|---|
| Learning rate | 6e-4 |
| Batch size | 6 |
| Epochs | 20 |
| Gradient accumulation steps | 1 |
| Gradient clip value | 1.0 |
| Precision | bf16-true |
| Data loader workers | 11 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=5) |
| LR schedule | OneCycleLR (warm-up 10 %) |

### Custom config

Pass a JSON file with any subset of the config keys to override defaults:

```bash
python training_GPT2.py --config configs/config.json --dataset war_and_peace
```

---

## Dataset Support

| Dataset module | Source | Split |
|---|---|---|
| `WarAndPeaceDataModule` | Pre-tokenised `.npy` file | 90 % train / 10 % val |
| `RussianBooksDataModule` | Pre-tokenised `.npy` file (selectable token count) | 90 % train / 10 % val |
| `FineWebEduDataModule` | HuggingFace streaming `IterableDataset` | 90 % train / 10 % val |

To prepare the War and Peace or Russian books datasets, use the tokenisation scripts in `working-with-text-dataset/`.

---

## Inference

```bash
python training_GPT2.py --no-train --inference --load_checkpoint ./checkpoints/last.ckpt
```

An interactive prompt loop will start. Type any text and the model will continue it. Enter `exit` to quit.

---

## Project Structure

```
Tiny-Mini-GPT/
├── model.py                   # Decoder, MultiHeadAttention, MLP, RMSNorm, positional encoding
├── training_GPT2.py           # PyTorch Lightning training script & data modules
├── config.py                  # DecoderConfig dataclass, load/save helpers
├── grokfast.py                # GrokFast gradient filter
├── configs/
│   └── config.json            # Default GPT-2 scale configuration
├── working-with-text-dataset/ # Tokenisation utilities and dataset preparation scripts
└── logo/
    └── logo.txt               # ASCII art logo
```
