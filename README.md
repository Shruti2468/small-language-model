# small-language-model

# Mini GPT (PyTorch) — README

A compact, educational implementation of a GPT-style causal transformer in PyTorch.
This repository contains a minimal GPT implementation including:
- token & positional embeddings
- multi-head causal self-attention (with optional Flash Attention)
- feed-forward (MLP) blocks
- layer normalization, residual connections
- weight tying (embedding <-> lm_head)

## Features

- Simple, readable PyTorch implementation of transformer decoder blocks.
- Causal self-attention (supports PyTorch's `scaled_dot_product_attention` when available).
- GELU feed-forward (4× expansion).
- Weight tying between token embedding and lm head.
- Dropout support and configurable sizes via a `GPTConfig` dataclass.

---

## File / Class Overview

- `LayerNorm(nn.Module)` — custom layer norm wrapper around `F.layer_norm`.
- `CausalSelfAttention(nn.Module)` — multi-head causal attention. Uses Flash Attention if available, otherwise uses classic masked scaled-dot-product attention.
- `MLP(nn.Module)` — FFN with `c_fc -> GELU -> c_proj`.
- `Block(nn.Module)` — single transformer block: LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual.
- `GPTConfig` (dataclass) — hyperparameters (block_size, vocab_size, n_layer, n_head, n_embd, dropout, bias).
- `GPT(nn.Module)` — top-level model (embeddings, blocks, final layernorm, lm head, generate method).

