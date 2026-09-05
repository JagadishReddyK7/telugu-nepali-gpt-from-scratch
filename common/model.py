"""
Decoder-only (GPT-style) Transformer language model, implemented from scratch.

Built from ``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm`` and ``nn.Dropout``
only. No ``nn.Transformer*``, no HuggingFace model classes, and no pre-built
attention primitive -- in particular the scaled dot-product attention here is
written out as explicit matmul / mask / softmax / matmul rather than delegating
to ``F.scaled_dot_product_attention``, because the point of this phase is to own
every tensor operation in the forward pass.

THE FORWARD PASS, END TO END
----------------------------
For a batch of token ids ``idx`` of shape (B, T):

    (B, T) ids
      -> token embedding        (B, T, d)      lookup into (vocab, d)
      +  positional embedding   (1, T, d)      lookup into (block_size, d)
      -> embedding dropout      (B, T, d)
      -> N x TransformerBlock   (B, T, d)
      -> final LayerNorm        (B, T, d)
      -> output projection      (B, T, vocab)  weights tied to token embedding

Each block is pre-norm:

    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

WHY PRE-NORM
------------
In post-norm (the original 2017 arrangement) the residual stream passes through
a LayerNorm on every sublayer, so the gradient reaching early layers is
repeatedly rescaled and deep stacks need a learning-rate warmup to train at all.
In pre-norm the residual path from the output back to the token embedding is a
clean sum of identity connections -- normalization sits only on the *branch*.
Gradients reach layer 0 undistorted, which is why pre-norm is the standard
choice for deep stacks and why it trains stably without heroic warmup schedules.
The cost is that the residual stream is unnormalized at the top, which is why a
final LayerNorm is applied before the output projection.

WHAT WOULD BREAK IF EACH PIECE WERE REMOVED
-------------------------------------------
* **Positional embeddings** -- self-attention is a permutation-equivariant
  operation: it computes a weighted sum over positions where the weights depend
  only on content, not on order. Remove them and "A > B" and "B > A" produce
  identical representations. This is the optional ablation (``positional="none"``).
* **Causal mask** -- position t would attend to t+1, so predicting the next
  token becomes copying an input the model can already see. Training loss would
  collapse toward zero and the model would generate nothing useful.
* **1/sqrt(d_k) scaling** -- see :class:`CausalSelfAttention`.
* **Residual connections** -- gradients would have to pass through every
  sublayer's Jacobian in series; a 10-layer stack would train very poorly.
* **Final LayerNorm** -- the unnormalized pre-norm residual stream grows in
  scale with depth, so the output logits would be badly conditioned at init.
* **Weight tying** -- nothing breaks, but the parameter count rises by
  vocab x d_model (6.14M here, ~25% of the whole model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Architecture hyperparameters for one model.

    Both languages use the same *shape*, but they are separate models with
    separate weights and their own vocabulary from Phase 1. Nothing is shared.

    Attributes:
        vocab_size: Size of this language's tokenizer vocabulary.
        n_layer: Number of transformer blocks stacked.
        n_head: Number of attention heads. Must divide ``d_model``.
        d_model: Residual stream width.
        d_ff: Inner width of the position-wise feed-forward network.
        block_size: Maximum sequence length. With learned absolute positional
            embeddings this is a HARD cap -- there is no embedding row for
            position ``block_size``, so longer inputs cannot be encoded at all.
        dropout: Dropout probability, applied to embeddings, attention
            probabilities, and both sublayer outputs.
        tie_embeddings: Share the token embedding matrix with the output
            projection.
        positional: ``"learned"`` for learned absolute embeddings, or ``"none"``
            to remove position information entirely (the bonus ablation).
        bias: Whether linear layers carry bias terms.
    """

    vocab_size: int
    n_layer: int = 10
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    block_size: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    positional: str = "learned"
    bias: bool = True

    def __post_init__(self):
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_head={self.n_head}; "
                f"each head works in a d_model/n_head dimensional subspace."
            )
        if self.positional not in ("learned", "none"):
            raise ValueError(f"positional must be 'learned' or 'none', got {self.positional!r}")

    @property
    def d_head(self) -> int:
        """Dimension of each attention head's subspace, ``d_model // n_head``."""
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy, for writing into checkpoints."""
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written out from first principles.

    Shapes, for input ``x`` of shape (B, T, d):

        x                     (B, T, d)
        -> qkv projection     (B, T, 3d)          one Linear, split three ways
        -> split              3 x (B, T, d)
        -> view               3 x (B, T, h, d_k)  d_k = d // h
        -> transpose(1, 2)    3 x (B, h, T, d_k)  heads become a batch dimension
        scores = q @ k^T      (B, h, T, T)
        scores /= sqrt(d_k)
        scores += mask        (1, 1, T, T) broadcast, 0 allowed / -inf future
        probs = softmax       (B, h, T, T)        rows sum to 1
        out = probs @ v       (B, h, T, d_k)
        -> transpose(1, 2)    (B, T, h, d_k)
        -> reshape            (B, T, d)           concatenate the heads
        -> output projection  (B, T, d)

    The transpose to (B, h, T, d_k) is what makes the heads independent: the
    matmuls then treat ``(B, h)`` as batch dimensions, so head i never mixes
    with head j until the final reshape and ``W_O``.

    WHY THE 1/sqrt(d_k) SCALING
    ---------------------------
    q and k entries are roughly zero-mean, unit-variance at initialization, so
    their dot product over ``d_k`` terms has variance ~``d_k`` and typical
    magnitude ~``sqrt(d_k)``. With d_k = 64 that puts raw scores in a range wide
    enough to push softmax into saturation, where one weight is ~1 and the rest
    are ~0. Softmax gradients vanish there, so attention would stop learning
    which positions to look at. Dividing by ``sqrt(d_k)`` returns the scores to
    unit variance and keeps the softmax in its responsive region.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_head
        self.dropout_p = config.dropout

        # One fused projection producing Q, K and V together. Mathematically
        # identical to three separate Linear layers -- it is one matmul instead
        # of three, which is faster and is the only reason to fuse them.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask, registered as a buffer so it moves with
        # .to(device) and is saved/loaded with the module but is NOT a parameter
        # and receives no gradient.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size,
                                                      config.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        """Apply causal self-attention.

        Args:
            x: Input of shape (B, T, d_model).
            need_weights: If True, also return the (B, n_head, T, T) attention
                probability tensor. Used by the attention analysis in Phase 2.3;
                off by default because materializing it costs memory.

        Returns:
            ``(y, attn)`` where ``y`` has shape (B, T, d_model) and ``attn`` is
            the attention probabilities or None.
        """
        B, T, C = x.shape

        qkv = self.qkv(x)                                   # (B, T, 3d)
        q, k, v = qkv.split(self.d_model, dim=2)            # 3 x (B, T, d)

        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Additive mask: 0 where attention is allowed, -inf where it is not.
        # Applied BEFORE softmax so the forbidden positions receive exactly zero
        # probability rather than a small non-zero one.
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0,
                                    float("-inf"))

        attn = F.softmax(scores, dim=-1)                    # (B, h, T, T)
        attn_out = self.attn_dropout(attn)

        y = attn_out @ v                                    # (B, h, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, d)
        y = self.resid_dropout(self.proj(y))

        return y, (attn if need_weights else None)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear -> dropout.

    "Position-wise" means the same two matrices are applied independently at
    every position -- there is no mixing across the time dimension here at all.
    Attention is the only operation in the block that moves information between
    positions; this sublayer is where each position does its own computation.

    The inner dimension is larger than ``d_model`` (4x here) because the
    expansion is where most of the block's capacity lives: at d=384, d_ff=1536,
    the two matrices hold 8d^2 = 1.18M parameters against attention's 4d^2 =
    0.59M, so roughly two thirds of every block is this sublayer.

    GELU rather than ReLU: it is smooth near zero, so it does not produce the
    dead-unit behaviour ReLU can, and it is the activation used by GPT-2.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)."""
        return self.dropout(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    """One pre-norm transformer block.

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Both sublayers are wrapped in residual connections, and normalization is
    applied to the *input* of each sublayer rather than to the sum. See the
    module docstring for why pre-norm.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        """Run one block, optionally returning its attention probabilities."""
        attn_out, attn = self.attn(self.ln_1(x), need_weights=need_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn


class GPT(nn.Module):
    """Decoder-only Transformer language model.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if config.positional == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        else:
            # Ablation: no position information at all. Registered as None so
            # every downstream shape check still works and the difference is a
            # config flag rather than a separate model class.
            self.position_embedding = None

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config)
                                     for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # Weight tying: the output projection reuses the token embedding
            # matrix. Saves vocab x d_model = 6.14M parameters here, ~25% of the
            # model. It also ties the two roles together -- a token's input
            # representation and the direction that predicts it become the same
            # vector, which is a real inductive bias, not only a saving.
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Scaled init for the residual projections. Each layer adds its output
        # into the residual stream, so after N layers the stream's variance has
        # grown ~N-fold. Scaling these two projections by 1/sqrt(2N) keeps the
        # stream's scale roughly constant with depth at initialization.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights: normal(0, 0.02) for linears and embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------- parameters

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count trainable parameters.

        Args:
            non_embedding: If True, exclude the token and positional embedding
                tables. Useful because embeddings dominate the count for a small
                model but contribute comparatively little compute per token.

        Returns:
            Number of trainable parameters.
        """
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.token_embedding.weight.numel()
            if self.position_embedding is not None:
                n -= self.position_embedding.weight.numel()
        return n

    def parameter_breakdown(self) -> dict:
        """Return a per-component parameter count, for the report's table."""
        cfg = self.config
        emb = self.token_embedding.weight.numel()
        pos = (self.position_embedding.weight.numel()
               if self.position_embedding is not None else 0)
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        total = self.num_parameters()
        return {
            "token_embedding": emb,
            "positional_embedding": pos,
            "per_block": per_block,
            "all_blocks": per_block * cfg.n_layer,
            "final_layernorm": sum(p.numel() for p in self.ln_f.parameters()),
            "lm_head": 0 if cfg.tie_embeddings else self.lm_head.weight.numel(),
            "total": total,
            "total_non_embedding": self.num_parameters(non_embedding=True),
            "embedding_share": round(emb / total, 4),
            "tied": cfg.tie_embeddings,
        }

    # ---------------------------------------------------------------- forward

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                need_weights: bool = False):
        """Run the model.

        Args:
            idx: Token ids of shape (B, T), with T <= ``block_size``.
            targets: Optional next-token targets of shape (B, T). When given,
                the causal-LM cross-entropy loss is returned.
            need_weights: Collect attention probabilities from every layer.

        Returns:
            ``(logits, loss, attentions)``. ``logits`` has shape
            (B, T, vocab_size); ``loss`` is a scalar or None; ``attentions`` is
            a list of (B, n_head, T, T) tensors or None.
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}. "
                f"With learned absolute positional embeddings there is no "
                f"embedding row for position {T - 1}, so longer sequences cannot "
                f"be encoded at all."
            )

        tok = self.token_embedding(idx)                      # (B, T, d)
        if self.position_embedding is not None:
            pos_ids = torch.arange(T, device=idx.device)     # (T,)
            tok = tok + self.position_embedding(pos_ids)     # broadcast (1,T,d)
        x = self.drop(tok)

        attentions = [] if need_weights else None
        for block in self.blocks:
            x, attn = block(x, need_weights=need_weights)
            if need_weights:
                attentions.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)                             # (B, T, vocab)

        loss = None
        if targets is not None:
            # Cross-entropy between the logits at position t and the token at
            # t+1. The shift is done by the caller when building batches, so
            # here logits[:, t] is already aligned with targets[:, t].
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss, attentions

    # --------------------------------------------------------------- sampling

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 greedy: bool = False, eos_id: int | None = None) -> torch.Tensor:
        """Autoregressively extend a prompt.

        Args:
            idx: Context token ids, shape (B, T).
            max_new_tokens: How many tokens to append.
            temperature: Logit temperature. Lower is more deterministic.
            top_k: If set, sample only from the k highest-probability tokens.
            greedy: Take the argmax instead of sampling. Overrides temperature.
            eos_id: Stop early once every sequence in the batch has emitted this.

        Returns:
            The context extended with generated tokens, shape (B, T + n).
        """
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens -- the model has no positional
            # embedding beyond that, so a longer context cannot be encoded.
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]                        # (B, vocab)

            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)

            if eos_id is not None:
                nxt = torch.where(finished.unsqueeze(1),
                                  torch.full_like(nxt, eos_id), nxt)
                finished = finished | (nxt.squeeze(1) == eos_id)

            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool(finished.all()):
                break

        return idx

    # ---------------------------------------------------------------- optimizer

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: tuple[float, float], device_type: str):
        """Build AdamW with decay applied only to matrix parameters.

        Weight decay on LayerNorm gains, biases and embeddings is at best
        useless and at worst harmful -- those are not the parameters where
        L2 shrinkage acts as a useful regularizer. Splitting them into two
        groups is standard practice for transformer training.

        Args:
            weight_decay: Decay coefficient for the decayed group.
            learning_rate: Peak learning rate.
            betas: AdamW beta coefficients.
            device_type: ``"cuda"`` or ``"cpu"``; enables the fused kernel.

        Returns:
            A configured ``torch.optim.AdamW``.
        """
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in params.items() if p.dim() >= 2]
        no_decay = [p for n, p in params.items() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        try:
            return torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                     fused=use_fused)
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    def estimate_flops_per_token(self) -> float:
        """Approximate forward+backward FLOPs per training token (6N + attention).

        The usual ``6N`` rule counts the matmuls whose cost scales with
        parameters. Attention's score matrix costs an extra term that scales
        with sequence length instead, which matters at long context.
        """
        cfg = self.config
        n = self.num_parameters(non_embedding=True)
        attn = 12 * cfg.n_layer * cfg.d_model * cfg.block_size
        return 6 * n + attn
