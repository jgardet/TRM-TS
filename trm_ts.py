"""
TRM-TS: True Recursive Model for Time Series
-------------------------------------------------
File: TRM_TS.py

This file implements a complete PyTorch model implementing the
Tiny Recursive Model (TRM) with FULL RECURRENT CYCLING for financial
time-series, adapted for FreqAI / Freqtrade pipeline.

## Full Recurrent Cycling (Key Innovation)

Unlike simple pooled-latent refinement, true TRM iteratively lets the
latent and token-level sequence influence each other:

  At each recursion step:
  1. Latent attends to sequence tokens (cross-attention readout)
  2. Tokens are reconditioned by the latent (FiLM-style adapter)
  3. Core updates the latent using attended information + memory
  4. Memory is gated and updated (GRU-style)

This enables multi-pass reasoning:
  - Pass 1: Detect momentum patterns
  - Pass 2: Refine with volatility context
  - Pass 3: Incorporate regime information
  - etc.

For trading, this helps capture complex pattern interactions and
regime changes that single-pass models miss.

The architecture includes:
 - Multi-scale temporal windows (8, 32, 128 timesteps)
 - Learnable per-feature embeddings
 - Cross-feature attention (true MHA across feature groups)
 - Cross-time attention (temporal transformer block)
 - Learned positional embeddings (non-periodic, for financial data)
 - **Latent-to-token cross-attention** (latent queries tokens each step)
 - **Token reconditioning** (FiLM-style adapter from latent)
 - RMSNorm throughout (faster than LayerNorm)
 - ResidualGate (learned residual scaling)
 - Regime embedding + calendar tokens
 - Distributional output head (Gaussian mean + logvar)
 - Cross-asset context branch
 - GRU-style gated memory updates
 - Dropout regularization throughout (10%)

The file contains:
 - Lightweight building blocks (RMSNorm, ResidualGate, TinyTransformer, etc.)
 - The main model class: trm_ts
 - Adapter helpers to make integration with Freqtrade/FreqAI easier

Notes:
 - This module intentionally avoids heavy 3rd-party dependencies.
 - If you integrate into FreqAI, wrap the model into your BasePyTorchModel
   or adapt the save/load hooks required by your training loop.

Usage (high level):
  model = trm_ts(n_features=20, latent_dim=64, recursive_steps=12)
  preds = model(batch)  # batch is a dict with 'x', 'mask', 'regime', 'hour', 'dow', 'ctx'

"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default dropout rate used throughout the model
DEFAULT_DROPOUT = 0.1

# -------------------------
# Basic utilities
# -------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
    def __init__(self, fn: nn.Module, scale: float = 1.0):
        super().__init__()
        self.fn = fn
        self.scale = scale
    def forward(self, x, *args, **kwargs):
        return x + self.scale * self.fn(x, *args, **kwargs)


# Simple MLP with LayerNorm
class LayerNormMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, out_dim=None, activation=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or (dim * 2)
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.ln = nn.LayerNorm(out_dim)
    def forward(self, x):
        return self.ln(self.net(x))


# -------------------------
# RMSNorm (Root Mean Square Layer Normalization)
# -------------------------
class RMSNorm(nn.Module):
    """RMSNorm: simpler and often faster than LayerNorm.
    
    Normalizes by RMS without mean centering. Often works better for
    residual networks and is used in LLaMA, Gemma, etc.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# -------------------------
# Residual Gate (learned residual scaling)
# -------------------------
class ResidualGate(nn.Module):
    """Learned residual gating for adaptive skip connections.
    
    Instead of fixed `x + 0.5 * h`, learns:
        gate = sigmoid(linear(h))
        output = x + gate * h
    
    This allows the model to learn how much of the residual to add,
    which is especially useful for deep/recursive networks.
    """
    def __init__(self, dim: int, init_bias: float = -2.0):
        super().__init__()
        # Project to scalar gate per feature
        self.gate_proj = nn.Linear(dim, dim)
        # Initialize bias negative so gate starts near 0 (conservative updates)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)
        
    def forward(self, x, h):
        """Apply gated residual: x + gate * h"""
        gate = torch.sigmoid(self.gate_proj(h))
        return x + gate * h


# -------------------------
# Feature embedding (per-feature MLP)
# -------------------------
class FeatureEmbed(nn.Module):
    """Embed each input feature vector (per-timestep features) into latent dim.

    Input: (B, T, F) -> Output: (B, T, D)

    The embedding is learnable per-feature and allows scale-invariant, nonlinear
    projection of engineered indicators.
    """
    def __init__(self, n_features: int, latent_dim: int, hidden: int = None, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        hidden = hidden or latent_dim * 2
        self.embed = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    def forward(self, x):
        # x: (B, T, F)
        return self.embed(x)  # (B, T, D)


# -------------------------
# Cross-feature attention (attend across feature groups for each timestep)
# -------------------------
class CrossFeatureAttention(nn.Module):
    """Apply multi-head attention across feature groups within each timestep.

    We reshape (B, T, D) -> (B*T, n_groups, group_dim) and apply MHA across groups.
    This allows features to attend to each other, learning indicator interactions.

    Input: (B, T, D) -> Output: (B, T, D)
    """
    def __init__(self, dim, n_heads=4, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        # Number of feature groups = n_heads, group_dim = dim // n_heads
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.group_dim = dim // n_heads
        
        # True multi-head attention across feature groups
        self.attn = nn.MultiheadAttention(self.group_dim, num_heads=1, batch_first=True, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_gate = ResidualGate(dim)
        
    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        
        # Reshape to (B*T, n_heads, group_dim) - treat feature groups as sequence
        x_groups = x.reshape(B * T, self.n_heads, self.group_dim)  # (B*T, n_groups, group_dim)
        
        # Apply attention across feature groups
        h, _ = self.attn(x_groups, x_groups, x_groups)  # (B*T, n_groups, group_dim)
        
        # Reshape back to (B, T, D) - use contiguous().view for safety
        h = h.contiguous().view(B, T, D)
        h = self.out_proj(h)
        h = self.dropout(h)
        
        # RMSNorm + learned residual gate
        return self.norm(self.residual_gate(x, h))


# -------------------------
# Learned Positional Embedding (for non-periodic financial data)
# -------------------------
class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings for financial time series.
    
    Unlike sinusoidal encoding which assumes periodicity, learned embeddings
    can capture arbitrary temporal patterns in financial data:
    - Recent candles matter more than old ones
    - Specific positions may have special meaning (e.g., session open/close)
    - Non-periodic market dynamics
    """
    def __init__(self, dim: int, max_len: int = 512, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        # Initialize with small values for stability
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)  # (T,)
        pos_emb = self.embedding(positions)  # (T, D)
        x = x + pos_emb.unsqueeze(0)  # Broadcast over batch
        return self.dropout(x)


# -------------------------
# Tiny Transformer block with RMSNorm + Residual Gating
# -------------------------
class TinyTransformerBlock(nn.Module):
    """Transformer block with RMSNorm and learned residual gates.
    
    Uses RMSNorm (faster, no mean centering) and ResidualGate (learned
    scaling) instead of LayerNorm and fixed residual connections.
    """
    def __init__(self, dim, n_heads=4, mlp_ratio=2.0, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.gate1 = ResidualGate(dim)
        
        self.norm2 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
        self.gate2 = ResidualGate(dim)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (B, T, D)
        # Pre-norm with RMSNorm, gated residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=mask)
        x = self.gate1(x, h)  # Learned gating instead of x + h
        
        h = self.norm2(x)
        h = self.mlp(h)
        x = self.gate2(x, h)  # Learned gating instead of x + h
        return x


# -------------------------
# Multi-scale encoder
# -------------------------
class MultiScaleEncoder(nn.Module):
    """Encode multiple temporal scales and merge them into a single latent.

    We'll compute embeddings for three windows: short, mid, long. Each is a
    sliding view over the input window. To keep things simple we assume the
    input is already sampled at the desired frequency and that `seq_len >= long`.
    """
    def __init__(self, latent_dim, scales=(8, 32, 128), attn_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.scales = tuple(scales)
        # For each scale we use a small transformer block and a pooling
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(latent_dim, n_heads=attn_heads, mlp_ratio=mlp_ratio)
            for _ in self.scales
        ])
        # projection to merge scales
        self.merge = nn.Linear(len(self.scales) * latent_dim, latent_dim)
    def forward(self, x):
        # x: (B, T, D) where T >= max(scales)
        B, T, D = x.size()
        scale_outs = []
        for i, s in enumerate(self.scales):
            if T < s:
                # fallback: use full length
                view = x
            else:
                # use the last s timesteps (focus on recent history for each scale)
                view = x[:, -s:, :]
            # transform and pool (mean pooling)
            h = self.blocks[i](view)  # (B, s, D)
            p = h.mean(dim=1)         # (B, D)
            scale_outs.append(p)
        cat = torch.cat(scale_outs, dim=-1)  # (B, len(scales)*D)
        merged = self.merge(cat)
        return merged


# -------------------------
# TRM core — recursive shared block with RMSNorm + gated residuals
# -------------------------
class TRMCore(nn.Module):
    """Recursive refinement block with RMSNorm, residual gating, and GRU-style memory.
    
    Uses RMSNorm instead of LayerNorm and learned residual gates instead of
    fixed 0.5 scaling for adaptive updates during recursive refinement.
    """
    def __init__(self, dim, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learned residual gate (replaces fixed 0.5 scale)
        self.residual_gate = ResidualGate(dim, init_bias=-2.0)
        
        # Gated memory update (GRU-style)
        self.gate_proj = nn.Linear(dim * 2, dim)  # Combines z and m to produce gate
        self.update_proj = nn.Linear(dim * 2, dim)  # Produces candidate memory update
        
    def forward(self, z, memory=None):
        # z: (B, D), memory: (B, D) optional
        if memory is not None:
            x = z + memory
        else:
            x = z
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        h = self.dropout(h)
        # Learned residual gating instead of fixed z + 0.5 * h
        return self.residual_gate(z, h)
    
    def update_memory(self, z, memory):
        """GRU-style gated memory update.
        
        gate = sigmoid(W_g * [z, m])  # What to forget
        candidate = tanh(W_u * [z, m])  # New information
        m_new = gate * m + (1 - gate) * candidate
        """
        combined = torch.cat([z, memory], dim=-1)  # (B, 2D)
        gate = torch.sigmoid(self.gate_proj(combined))  # (B, D)
        candidate = torch.tanh(self.update_proj(combined))  # (B, D)
        return gate * memory + (1 - gate) * candidate


# -------------------------
# Regime embedding helper
# -------------------------
class RegimeEmbedding(nn.Module):
    def __init__(self, n_regimes: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_regimes, dim)
    def forward(self, regime_idx):
        # regime_idx: (B,) or None
        if regime_idx is None:
            return None
        return self.emb(regime_idx)


# -------------------------
# Calendar/event tokens
# -------------------------
class CalendarTokens(nn.Module):
    """Embeddings for temporal patterns: hour-of-day and day-of-week."""
    def __init__(self, dim: int, dow=7, hour=24):
        super().__init__()
        self.hour_emb = nn.Embedding(hour, dim)
        self.dow_emb = nn.Embedding(dow, dim)
        
    def forward(self, hour_idx: Optional[torch.LongTensor], dow_idx: Optional[torch.LongTensor]):
        # Inputs: (B,) indexes or None
        if hour_idx is None and dow_idx is None:
            return None
        
        token = None
        if hour_idx is not None:
            token = self.hour_emb(hour_idx)
        if dow_idx is not None:
            dow_token = self.dow_emb(dow_idx)
            token = dow_token if token is None else token + dow_token
        return token


# -------------------------
# Distributional head (Gaussian)
# -------------------------
class DistributionalHead(nn.Module):
    """Predict mean and log-variance for distributional forecasting.

    Output: (B, H) mean and (B, H) logvar
    """
    def __init__(self, latent_dim, horizon: int):
        super().__init__()
        self.mean = nn.Linear(latent_dim, horizon)
        self.logvar = nn.Linear(latent_dim, horizon)
    def forward(self, z):
        return self.mean(z), self.logvar(z)


# -------------------------
# Cross-asset context branch
# -------------------------
class CrossAssetContext(nn.Module):
    def __init__(self, ctx_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ctx_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self, ctx):
        if ctx is None:
            return None
        return self.proj(ctx)


# -------------------------
# Full TRM model with True Recurrent Cycling
# -------------------------
class trm_ts(nn.Module):
    """TRM-TS: True Recursive Model with full recurrent cycling.

    Key difference from simple TRM: at each recursion step, the latent
    and tokens mutually influence each other:
    
    1. Latent queries tokens via cross-attention (evidence extraction)
    2. Tokens are reconditioned by latent (FiLM adapter)
    3. Core updates latent with attended info + memory
    4. Memory is gated and updated
    
    This enables iterative inference - e.g., first pass detects momentum,
    second pass refines with volatility context, etc.

    Expected batch input (dictionary):
      batch['x']         -> torch.FloatTensor (B, T, F) feature matrix per candle
      batch['mask']      -> optional (B, T) key_padding_mask for transformer
      batch['regime']    -> optional (B,) long tensor of regime indices
      batch['hour']      -> optional (B,) long tensor for hour-of-day
      batch['dow']       -> optional (B,) long tensor for day-of-week
      batch['ctx']       -> optional (B, C) cross-asset context vector

    The forward() returns a dict with:
      'mean': (B, H) prediction means
      'logvar': (B, H) prediction log-variances
      'latent': (B, D) final latent for inspection

    """
    def __init__(
        self,
        n_features: int,
        latent_dim: int = 64,
        recursive_steps: int = 12,
        forecast_horizon: int = 1,
        multi_scales=(8, 32, 128),
        n_regimes: int = 8,
        ctx_dim: int = 0,
        attn_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recursive_steps = recursive_steps
        self.forecast_horizon = forecast_horizon
        self.dropout_rate = dropout

        # 1) Feature embedding (learned nonlinear projection)
        self.feature_embed = FeatureEmbed(n_features, latent_dim, dropout=dropout)

        # 2) Cross-feature attention — gives the model ability to combine indicators
        self.cross_feature_attn = CrossFeatureAttention(latent_dim, n_heads=attn_heads, dropout=dropout)
        
        # 3) Learned positional embedding (non-periodic, suitable for financial data)
        self.pos_encoding = LearnedPositionalEmbedding(latent_dim, max_len=max(multi_scales) * 2, dropout=dropout)

        # 4) Multi-scale temporal encoder — short, mid, long term
        self.multi_scale = MultiScaleEncoder(latent_dim, scales=multi_scales, attn_heads=attn_heads, mlp_ratio=mlp_ratio)

        # 5) Small transformer over the full window (temporal attention)
        self.temporal_transformer = TinyTransformerBlock(latent_dim, n_heads=attn_heads, mlp_ratio=mlp_ratio, dropout=dropout)

        # 6) TRM core & memory
        self.core = TRMCore(latent_dim, dropout=dropout)
        self.memory_proj = nn.Linear(latent_dim, latent_dim)
        
        # 7) Full recurrent cycling components
        # Latent-to-token cross-attention: latent (query) attends to tokens (key/value)
        self.latent_cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads=attn_heads, batch_first=True, dropout=dropout
        )
        
        # Additional dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Token reconditioning adapter (FiLM-style): conditions tokens on current latent
        self.token_adapter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Learnable token conditioning scale (starts small for gentle updates)
        self.token_cond_scale = nn.Parameter(torch.tensor(0.1))

        # 8) Regime embedding
        self.regime_emb = RegimeEmbedding(n_regimes, latent_dim)

        # 9) Calendar tokens
        self.calendar = CalendarTokens(latent_dim)

        # 10) Cross-asset context branch
        self.ctx_branch = CrossAssetContext(ctx_dim, latent_dim) if ctx_dim and ctx_dim > 0 else None

        # 11) Readout (distributional)
        self.dist_head = DistributionalHead(latent_dim, forecast_horizon)

        # Final normalization (RMSNorm for consistency)
        self.final_ln = RMSNorm(latent_dim)

    def forward(self, batch: dict):
        # Extract batch fields
        x = batch.get('x')          # required: (B, T, F)
        mask = batch.get('mask', None)
        regime = batch.get('regime', None)
        hour = batch.get('hour', None)
        dow = batch.get('dow', None)
        ctx = batch.get('ctx', None)  # cross-asset context vector

        if x is None:
            raise ValueError("batch must contain 'x' with shape (B, T, F)")

        B, T, n_feat = x.shape

        # 1) Feature embedding (nonlinear projection) -> (B, T, D)
        x_proj = self.feature_embed(x)
        # comment: this reduces noise, rescales indicators and compresses features

        # 2) Cross-feature attention (per-timestep) -> residual update on (B, T, D)
        x_feat = self.cross_feature_attn(x_proj)
        # comment: learns interactions between features at each timestep
        
        # 3) Add positional encoding for temporal order awareness
        x_pos = self.pos_encoding(x_feat)

        # 4) Temporal transformer (global attention across time) -> (B, T, D)
        x_time = self.temporal_transformer(x_pos, mask)
        # comment: learns which past timesteps to attend to

        # 5) Multi-scale summary -> (B, D) - uses temporally-aware features
        scale_latent = self.multi_scale(x_time)
        # comment: captures short/mid/long horizons as a single vector

        # 6) Pooled global latent (mean pooling of time-aware features)
        pooled = x_time.mean(dim=1)

        # 7) Combine pooled + multi-scale (concatenate-like fusion)
        z0 = pooled + scale_latent  # simple fusion; could be concat+proj
        z0 = self.dropout(z0)  # Regularization dropout

        # 8) Add regime embedding and calendar tokens
        reg_vec = self.regime_emb(regime)  # (B, D) or None
        cal_vec = self.calendar(hour, dow) # (B, D) or None
        if reg_vec is not None:
            z0 = z0 + reg_vec
        if cal_vec is not None:
            z0 = z0 + cal_vec

        # 9) Add cross-asset context if provided
        if self.ctx_branch is not None and ctx is not None:
            ctx_vec = self.ctx_branch(ctx)
            z0 = z0 + ctx_vec

        # 10) Initialize memory and tokens for full recurrent cycling
        m = torch.tanh(self.memory_proj(z0))  # Initial memory state
        z = z0                                  # Initial latent
        tokens = x_time                         # Token sequence (B, T, D) - temporally aware
        
        # 11) Full recurrent cycling: latent <-> tokens mutual refinement
        for step in range(self.recursive_steps):
            # Step 1: Latent queries tokens via cross-attention
            # z (B, D) -> (B, 1, D) as query; tokens (B, T, D) as key/value
            q = z.unsqueeze(1)  # (B, 1, D)
            attn_out, _ = self.latent_cross_attn(q, tokens, tokens)  # (B, 1, D)
            attn_read = attn_out.squeeze(1)  # (B, D) - evidence extracted from tokens
            
            # Step 2: Update latent via core using attended info + memory
            # Combine current latent with attention readout
            z_in = z + attn_read
            z = self.core(z_in, memory=m)
            z = self.dropout(z)  # Dropout in recursive loop
            
            # Step 3: Token reconditioning (FiLM-style)
            # Let tokens see the updated latent for next iteration
            token_cond = self.token_adapter(z)  # (B, D)
            # Add conditioning with learnable scale (broadcast over T)
            tokens = tokens + token_cond.unsqueeze(1) * self.token_cond_scale
            
            # Step 4: Gated memory update (GRU-style)
            m = self.core.update_memory(z, m)

        z = self.final_ln(z)

        # 12) Distributional readout
        mean, logvar = self.dist_head(z)

        return {'mean': mean, 'logvar': logvar, 'latent': z}

    # -------------------------
    # Utility methods for inference/training integration
    # -------------------------
    def predict_mean(self, batch: dict):
        self.eval()
        with torch.no_grad():
            out = self.forward(batch)
            return out['mean']

    def sample(self, batch: dict, n_samples: int = 16) -> torch.Tensor:
        """Draw samples from the predicted Gaussian for each batch element.

        Returns: (n_samples, B, H)
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(batch)
            mu = out['mean']
            lv = out['logvar']
            std = torch.exp(0.5 * lv)
            B, H = mu.shape
            eps = torch.randn(n_samples, B, H, device=mu.device)
            samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
            return samples

    def save_checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, map_location=None):
        sd = torch.load(path, map_location=map_location or get_device())
        self.load_state_dict(sd)


# -------------------------
# Small example / sanity test
# -------------------------
if __name__ == '__main__':
    # toy config
    cfg = {
        'n_features': 20,
        'latent_dim': 64,
        'recursive_steps': 12,
        'forecast_horizon': 5,
        'multi_scales': (8, 32, 128),
        'n_regimes': 8,
        'ctx_dim': 6,
        'attn_heads': 4,
    }

    model = trm_ts(
        n_features=cfg['n_features'],
        latent_dim=cfg['latent_dim'],
        recursive_steps=cfg['recursive_steps'],
        forecast_horizon=cfg['forecast_horizon'],
        multi_scales=cfg['multi_scales'],
        n_regimes=cfg['n_regimes'],
        ctx_dim=cfg['ctx_dim'],
        attn_heads=cfg['attn_heads'],
    )

    B, T, n_feat = 16, 128, cfg['n_features']
    x = torch.randn(B, T, n_feat)
    mask = None
    regime = torch.randint(0, cfg['n_regimes'], (B,))
    hour = torch.randint(0, 24, (B,))
    dow = torch.randint(0, 7, (B,))
    ctx = torch.randn(B, cfg['ctx_dim'])

    batch = {'x': x, 'mask': mask, 'regime': regime, 'hour': hour, 'dow': dow, 'ctx': ctx}

    out = model(batch)
    print('mean', out['mean'].shape)
    print('logvar', out['logvar'].shape)
    print('latent', out['latent'].shape)

