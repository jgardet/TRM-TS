# TRM-TS: True Recursive Model for Time Series

A lightweight PyTorch architecture for financial time-series forecasting with **full recurrent cycling** — enabling multi-pass reasoning over temporal patterns.

## Introduction

Financial time-series forecasting presents unique analytical challenges: markets generate data that is inherently noisy, non-stationary, and multivariate. Success requires models that can discern faint signals from random fluctuations while capturing complex, multi-scale temporal dependencies—from fleeting intraday momentum to persistent multi-day trends.

While large-scale Transformer architectures have shown promise in sequence modeling, their brute-force application to financial forecasting is challenged by high computational complexity and over-parameterization. Research suggests that simpler models can sometimes be equally effective, questioning the necessity of massive scale.

**TRM-TS** addresses these issues through three core principles:

1. **Deep Reasoning via Recursive Refinement** — Iterative refinement rather than massive parameterization
2. **Exceptional Computational Efficiency** — ~60-80K parameters (vs. millions in standard Transformers)
3. **Domain-Specific Adaptation** — Tailored for financial markets from the ground up

## Key Features

| Feature | Description |
|---------|-------------|
| **Full Recurrent Cycling** | Latent and token sequences iteratively refine each other |
| **Distributional Output** | Predicts mean + uncertainty (logvar) via Gaussian NLL |
| **Multi-Scale Temporal Encoding** | Captures short/mid/long-term patterns (8, 32, 128 timesteps) |
| **Learned Positional Embeddings** | Non-periodic, suitable for financial data |
| **RMSNorm + ResidualGate** | Modern normalization and adaptive residual connections |
| **Lightweight** | ~60-80K parameters (two orders of magnitude smaller than standard TRM) |

## Theoretical Foundation

TRM-TS builds on concepts from:

- **Hierarchical Reasoning Model (HRM)** — Two-level processing for multi-level reasoning
- **Tiny Recursive Model (TRM)** — Parameter-efficient iterative refinement (7M params in original research)

The central innovation is applying TRM's iterative refinement to time-series forecasting. Instead of a single deep forward pass, TRM-TS iteratively refines temporal predictions with each recursive step, building progressively more sophisticated understanding of market dynamics while maintaining an exceptionally small footprint.

## Architecture Overview

```
Input: (B, T, F) — Batch × Timesteps × Features
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  1. FEATURE EMBEDDING (FeatureEmbed)                                  │
│     ────────────────────────────────                                  │
│     F features → D latent dimensions                                  │
│     MLP: Linear(F→2D) → GELU → Linear(2D→D) → RMSNorm                 │
│                                                                       │
│     Purpose: Scale-invariant representation (RSI: 0-100, MACD: ±∞)   │
└───────────────────────────────────────────────────────────────────────┘
                    │ (B, T, D)
                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  2. CROSS-FEATURE ATTENTION (CrossFeatureAttention)                   │
│     ───────────────────────────────────────────────                   │
│     True Multi-Head Attention across indicator groups per timestep    │
│     Models: How volume confirms price, RSI divergence, etc.           │
└───────────────────────────────────────────────────────────────────────┘
                    │ (B, T, D)
                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  3. LEARNED POSITIONAL EMBEDDING                                      │
│     ────────────────────────────────                                  │
│     Non-sinusoidal (financial data is non-periodic)                   │
│     Learns: recency bias, session patterns, arbitrary temporal dynamics│
└───────────────────────────────────────────────────────────────────────┘
                    │ (B, T, D)
          ┌─────────┴─────────────────────┐
          ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────────────────────┐
│  4. MULTI-SCALE ENCODER │   │  5. TEMPORAL TRANSFORMER                │
│  ───────────────────────│   │  ────────────────────────               │
│  Scales: 8, 32, 128     │   │  TinyTransformerBlock                   │
│                         │   │  Full sequence attention (Pre-LN)       │
│  • 8 steps: ~2h @15m    │   │  RMSNorm + ResidualGate                 │
│    (short-term momentum)│   │                                         │
│  • 32 steps: ~8h @15m   │   │  Captures: sequential dependencies,     │
│    (intraday patterns)  │   │  how earlier events influence outcomes  │
│  • 128 steps: ~32h @15m │   │                                         │
│    (multi-day trends)   │   │                                         │
│                         │   │                                         │
│  Each: Transformer →    │   │                                         │
│        MeanPool → (B,D) │   │                                         │
│  Concat → Linear → (B,D)│   │                                         │
└─────────────────────────┘   └─────────────────────────────────────────┘
          │ (B, D)                        │ (B, T, D) → mean pool → (B, D)
          └─────────────┬─────────────────┘
                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  6. LATENT FUSION                                                     │
│     ─────────────                                                     │
│     z0 = pooled + scale_latent + dropout                              │
│                                                                       │
│     Optional additions:                                               │
│     + RegimeEmbedding(regime_idx)  — Market state (0-7)               │
│     + CalendarTokens(hour, dow)    — Session/weekly patterns          │
│     + CrossAssetContext(ctx)       — BTC/ETH correlations             │
└───────────────────────────────────────────────────────────────────────┘
                        │ (B, D)
                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  7. FULL RECURRENT CYCLING (TRMCore) — THE KEY INNOVATION             │
│     ─────────────────────────────────────────────────────             │
│                                                                       │
│     Initialize:                                                       │
│       m = tanh(memory_proj(z0))     — Memory state                    │
│       z = z0                         — Latent state                   │
│       tokens = x_time                — Token sequence (B, T, D)       │
│                                                                       │
│     ┌─────────────────────────────────────────────────────────────┐   │
│     │  FOR step IN range(recursive_steps):  # Default: 4          │   │
│     │                                                             │   │
│     │  ┌─────────────────────────────────────────────────────┐    │   │
│     │  │ STEP 1: Cross-Attention Read (Latent → Tokens)      │    │   │
│     │  │         q = z.unsqueeze(1)           # (B, 1, D)    │    │   │
│     │  │         attn_out = MHA(q, tokens, tokens)           │    │   │
│     │  │         attn_read = attn_out.squeeze(1)  # (B, D)   │    │   │
│     │  │                                                     │    │   │
│     │  │         → Extracts relevant temporal evidence       │    │   │
│     │  └─────────────────────────────────────────────────────┘    │   │
│     │                          │                                  │   │
│     │                          ▼                                  │   │
│     │  ┌─────────────────────────────────────────────────────┐    │   │
│     │  │ STEP 2: Latent Update (TRMCore)                     │    │   │
│     │  │         z_in = z + attn_read                        │    │   │
│     │  │         z = core(z_in, memory=m) + dropout          │    │   │
│     │  │                                                     │    │   │
│     │  │         → Primary reasoning operation               │    │   │
│     │  └─────────────────────────────────────────────────────┘    │   │
│     │                          │                                  │   │
│     │                          ▼                                  │   │
│     │  ┌─────────────────────────────────────────────────────┐    │   │
│     │  │ STEP 3: Token Reconditioning (FiLM-style)           │    │   │
│     │  │         token_cond = adapter(z)         # (B, D)    │    │   │
│     │  │         tokens = tokens + token_cond * scale        │    │   │
│     │  │                                                     │    │   │
│     │  │         → Embeds global context back into sequence  │    │   │
│     │  └─────────────────────────────────────────────────────┘    │   │
│     │                          │                                  │   │
│     │                          ▼                                  │   │
│     │  ┌─────────────────────────────────────────────────────┐    │   │
│     │  │ STEP 4: Memory Update (GRU-style gating)            │    │   │
│     │  │         m = core.update_memory(z, m)                │    │   │
│     │  │                                                     │    │   │
│     │  │         → Selective information retention           │    │   │
│     │  └─────────────────────────────────────────────────────┘    │   │
│     │                                                             │   │
│     └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│     Multi-pass reasoning enables:                                     │
│       • Pass 1: Detect momentum patterns                              │
│       • Pass 2: Refine with volatility context                        │
│       • Pass 3: Incorporate regime information                        │
│       • Pass 4: Final prediction refinement                           │
└───────────────────────────────────────────────────────────────────────┘
                        │ (B, D)
                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  8. DISTRIBUTIONAL HEAD                                               │
│     ───────────────────                                               │
│     z_final = RMSNorm(z)                                              │
│     mean   = Linear(D → H)      — Expected return                     │
│     logvar = Linear(D → H)      — Log-variance (uncertainty)          │
│                                                                       │
│     Enables:                                                          │
│       • Monte Carlo simulations for outcome modeling                  │
│       • Explicit confidence intervals                                 │
│       • Risk-aware position sizing                                    │
└───────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
Output: {'mean': (B, H), 'logvar': (B, H), 'latent': (B, D)}
```

## Technical Innovations

### Advanced Normalization and Residual Connections

| Component | Description |
|-----------|-------------|
| **RMSNorm** | Replaces LayerNorm. Faster (no mean-centering), proven in LLaMA/Gemma. Compounding benefits in recursive architecture. |
| **ResidualGate** | Replaces fixed residuals (`x + h`) with learned gating: `x + σ(Linear(h)) ⊙ h`. Critical for preventing gradient issues in multi-pass recursion. |

### Domain-Specific Adaptations

| Input | Shape | Purpose |
|-------|-------|---------|
| `regime` | `(B,)` | Market regime index (0-7) for volatility/trend states |
| `hour` | `(B,)` | Hour of day (0-23) for session patterns |
| `dow` | `(B,)` | Day of week (0-6) for weekly seasonality |
| `ctx` | `(B, C)` | Cross-asset context (BTC/ETH correlations) |

## Installation

```bash
cd user_data/research
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

## Quick Start

### Train on Synthetic Data

```bash
python train_infer_trm_ts.py --epochs 20 --device cpu
```

### Train on CSV Data

```bash
python train_infer_trm_ts.py --csv path/to/candles.csv --epochs 20 --device cuda
```

CSV must contain at least a `close` column. Other columns (`open`, `high`, `low`, `volume`) are used if present.

### Load and Infer

```bash
python train_infer_trm_ts.py --load trm_ts_example.pt --device cpu
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | None | Path to CSV with OHLCV data |
| `--device` | cpu | Device: `cpu` or `cuda` |
| `--epochs` | 20 | Max training epochs |
| `--batch` | 64 | Batch size |
| `--window` | 128 | Sequence window length |
| `--lr` | 5e-5 | Learning rate |
| `--weight_decay` | 0.05 | L2 regularization |
| `--latent_dim` | 32 | Latent dimension |
| `--steps` | 4 | Recursive refinement steps |
| `--heads` | 2 | Attention heads |
| `--dropout` | 0.2 | Dropout rate |
| `--save` / `--no-save` | save | Save model after training |
| `--load` | None | Load saved model (skip training) |

## Model Parameters

```python
from trm_ts import trm_ts

model = trm_ts(
    n_features=20,          # Number of input features
    latent_dim=32,          # Latent space dimension
    recursive_steps=4,      # Recurrent cycling iterations
    forecast_horizon=1,     # Prediction horizon
    multi_scales=(16, 64),  # Temporal window scales
    n_regimes=8,            # Regime embedding count
    ctx_dim=0,              # Cross-asset context dim (0=disabled)
    attn_heads=2,           # Attention heads
    dropout=0.2,            # Dropout rate
)
```

## Training Output

```
Epoch 001 - Train: -0.2008, Test: -0.8922 ★ (best)
Epoch 002 - Train: -0.7381, Test: -1.2861 ★ (best)
Epoch 003 - Train: -0.9511, Test: -1.4594 ★ (best)
...
Early stopping at epoch 15 - best test loss: -1.8234
Saved model to trm_ts_example.pt
Inference metrics on test split - MAE: 0.068836, RMSE: 0.076009
```

- **Negative loss is good** (Gaussian NLL - more negative = higher likelihood)
- Early stopping restores best model automatically
- MAE/RMSE reported on held-out test set

## FreqAI Integration

For use with Freqtrade/FreqAI, see:
- `../freqaimodels/TRMTSRegressor.py` - FreqAI prediction model with Gaussian NLL loss
- `../strategies/TRMTSStrategy.py` - Trading strategy with comprehensive feature engineering
- `../config_trm_ts.json` - FreqAI configuration

```powershell
# Quick start with FreqAI
.\user_data\run_trm_ts.ps1 -Action download -Timerange "20240101-20241201"
.\user_data\run_trm_ts.ps1 -Action backtest -Timerange "20240601-20241201"
```

## Conclusion

TRM-TS represents a synthesis of recursive efficiency and the specific analytical demands of multivariate financial time-series. By replacing brute-force parameterization with elegant iterative refinement, it offers a compelling solution for practitioners seeking both performance and resource efficiency.

### Core Advantages

| Advantage | Description |
|-----------|-------------|
| **Computational Efficiency** | ~60-80K parameters. Suitable for resource-constrained or latency-sensitive environments. |
| **Deep Reasoning via Recursion** | Full Recurrent Cycling enables multi-pass reasoning, achieving depth that far exceeds physical size. |
| **Robust Architecture** | RMSNorm + ResidualGate ensure training stability in deep recursive systems. |
| **Financial Specialization** | Learned positional embeddings, multi-scale analysis, regime/calendar awareness, cross-asset context. |

TRM-TS challenges the prevailing notion that performance must scale with parameter count.

## Files

| File | Description |
|------|-------------|
| `trm_ts.py` | Core model implementation (~700 lines) |
| `train_infer_trm_ts.py` | Standalone training/inference script |
| `requirements.txt` | Python dependencies |
| `LICENSE` | Apache License 2.0 |
| `README.md` | This documentation |

## Attribution / Acknowledgements

This project, **TRM-TS**, is inspired by the concepts introduced in:

- **Hierarchical Reasoning Model (HRM)** – for multi-level recursive reasoning
- **Tiny Recursive Model (TRM)** – for parameter-efficient iterative refinement in reasoning tasks

All code in this repository is originally implemented by the author, and any resemblance to prior implementations is limited to the conceptual level.

If you use or build upon this work, please cite or acknowledge the project and its author as follows:

> "TRM-TS implementation by Joël Gardet, inspired by TRM/HRM concepts."

## Disclaimer

This project is provided **as is**, without warranty of any kind, express or implied, and **will not be actively maintained**. Use it at your own risk and evaluate its suitability for your specific use case.

## License

Licensed under the [Apache License 2.0](LICENSE) – see LICENSE for details.
