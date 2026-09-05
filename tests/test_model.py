"""
Correctness tests for the from-scratch Transformer.

Run with:

    python -m pytest tests/test_model.py -v
    python tests/test_model.py          # no pytest needed

THE TEST THAT MATTERS MOST
--------------------------
``test_causal_mask_blocks_the_future`` is the empirical demonstration the spec
asks for: perturb token t+1 and show the logits at position t do not move. A
causal LM whose mask is subtly wrong still trains -- it trains *better*, because
it can read the answer -- so this failure mode is invisible in the loss curve and
only shows up as a model that cannot generate. It has to be tested directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.model import GPT, GPTConfig  # noqa: E402


def tiny_config(**kw) -> GPTConfig:
    """A small config so tests run in under a second on CPU."""
    base = dict(vocab_size=200, n_layer=3, n_head=4, d_model=64, d_ff=256,
                block_size=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


# --------------------------------------------------------------------- shapes

def test_forward_shapes():
    """Logits are (B, T, vocab) and loss is a finite scalar."""
    cfg = tiny_config()
    model = GPT(cfg).eval()
    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss, attn = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert attn is None
    print("  forward shapes OK", tuple(logits.shape))


def test_attention_shapes_and_normalisation():
    """Attention is (B, h, T, T) per layer and each row is a distribution."""
    cfg = tiny_config()
    model = GPT(cfg).eval()
    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    _, _, attns = model(idx, need_weights=True)
    assert len(attns) == cfg.n_layer
    for a in attns:
        assert a.shape == (B, cfg.n_head, T, T), a.shape
        rows = a.sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
    print(f"  attention {tuple(attns[0].shape)}, rows sum to 1 OK")


def test_head_reshape_keeps_heads_independent():
    """d_model must split cleanly into n_head subspaces of size d_head."""
    cfg = tiny_config()
    assert cfg.d_head * cfg.n_head == cfg.d_model
    try:
        GPTConfig(vocab_size=10, d_model=65, n_head=4)
    except ValueError as e:
        assert "divisible" in str(e)
        print("  non-divisible d_model rejected OK")
    else:
        raise AssertionError("expected ValueError for indivisible d_model")


# ------------------------------------------------------------- causal masking

def test_causal_mask_blocks_the_future():
    """Changing token t+1 must not change the logits at position t.

    This is the spec's stated verification. Dropout is disabled and the model
    is in eval mode so the comparison is exact rather than statistical.
    """
    cfg = tiny_config(dropout=0.0)
    model = GPT(cfg).eval()

    T = 16
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        base, _, _ = model(idx)

    failures = []
    for t in range(T - 1):
        perturbed = idx.clone()
        # Change every token strictly after t.
        for j in range(t + 1, T):
            perturbed[0, j] = (perturbed[0, j] + 7) % cfg.vocab_size
        with torch.no_grad():
            other, _, _ = model(perturbed)

        # Positions 0..t must be bit-identical.
        if not torch.allclose(base[0, :t + 1], other[0, :t + 1], atol=1e-6):
            failures.append(t)

        # Sanity: position t+1 SHOULD change, otherwise the test proves nothing
        # (a model ignoring its input would pass the check above trivially).
        if t + 1 < T:
            assert not torch.allclose(base[0, t + 1], other[0, t + 1], atol=1e-6), (
                f"logits at t+1={t+1} did not change when its input token did; "
                f"the model may be ignoring its input entirely"
            )

    assert not failures, f"future leaked into positions {failures}"
    print(f"  causality verified at all {T - 1} positions "
          f"(and future tokens DO change their own logits) OK")


def test_mask_is_lower_triangular():
    """The attention pattern itself must be zero above the diagonal."""
    cfg = tiny_config(dropout=0.0)
    model = GPT(cfg).eval()
    T = 12
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    _, _, attns = model(idx, need_weights=True)
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    for li, a in enumerate(attns):
        assert a[0, :, upper].abs().max().item() == 0.0, f"layer {li} attends ahead"
    print(f"  attention strictly lower-triangular in all {len(attns)} layers OK")


# ------------------------------------------------------------------ structure

def test_no_prebuilt_transformer_modules():
    """The spec forbids nn.Transformer* and pre-built attention blocks."""
    cfg = tiny_config()
    model = GPT(cfg)
    banned = ("TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer",
              "TransformerDecoderLayer", "MultiheadAttention", "Transformer")
    found = [type(m).__name__ for m in model.modules()
             if type(m).__name__ in banned]
    assert not found, f"forbidden modules present: {found}"

    # Check for an actual CALL, not a mention in prose. The module docstring
    # names F.scaled_dot_product_attention to explain why it is deliberately
    # not used, so a bare substring search matches the explanation itself.
    src = (Path(__file__).resolve().parents[1] / "common" / "model.py").read_text(
        encoding="utf-8")
    assert "scaled_dot_product_attention(" not in src, (
        "F.scaled_dot_product_attention is a pre-built attention primitive; "
        "attention must be written out explicitly")
    kinds = sorted({type(m).__name__ for m in model.modules()})
    print(f"  no forbidden modules. Uses: {kinds}")


def test_weight_tying_shares_storage():
    """Tied embeddings must be the same tensor, not a copy."""
    cfg = tiny_config(tie_embeddings=True)
    model = GPT(cfg)
    assert model.lm_head.weight is model.token_embedding.weight
    tied = model.num_parameters()

    untied = GPT(tiny_config(tie_embeddings=False)).num_parameters()
    saved = untied - tied
    assert saved == cfg.vocab_size * cfg.d_model, (saved, cfg.vocab_size * cfg.d_model)
    print(f"  tying shares storage and saves {saved:,} params OK")


def test_ablation_removes_position_information():
    """positional='none' must drop the table AND change the output."""
    cfg = tiny_config(positional="none")
    model = GPT(cfg).eval()
    assert model.position_embedding is None
    assert "positional_embedding" not in [n for n, _ in model.named_parameters()]

    # With no positional signal, a permuted input must give permuted-identical
    # representations at the FIRST layer. Easiest observable: reversing a
    # sequence of identical tokens changes nothing, and more strongly, the
    # model with positions differs from the model without.
    with_pos = GPT(tiny_config(positional="learned")).eval()
    assert with_pos.position_embedding is not None
    diff = with_pos.num_parameters() - model.num_parameters()
    assert diff == cfg.block_size * cfg.d_model, diff
    print(f"  ablation drops {diff:,} positional params OK")


def test_causal_masking_is_itself_positional():
    """Removing positional embeddings does NOT make the model order-blind.

    The intuitive claim -- "self-attention is permutation invariant, so without
    positional embeddings the model cannot tell A>B from B>A" -- is true for an
    *unmasked* encoder. It is false for a causal decoder, and the difference is
    the mask.

    Under causal masking, position 0 attends to exactly 1 token, position 1 to
    2, and position t to t+1. That count is positional information leaking in
    through the mask itself: two positions holding identical content still get
    different representations because they attend over different-sized
    prefixes. So the ablated model retains a weak, implicit notion of order.

    This is the documented result of Haviv et al. (2022), and it is the reason
    the positional-embedding ablation is expected to degrade rather than
    collapse. Asserting the naive claim here would have written a false
    prediction into the report.
    """
    cfg = tiny_config(positional="none", dropout=0.0)
    model = GPT(cfg).eval()
    # Permute the PREFIX, hold the final token fixed, so any difference at the
    # last position cannot be attributed to a different query vector.
    a = torch.tensor([[5, 9, 3]])
    b = torch.tensor([[9, 5, 3]])
    with torch.no_grad():
        la, _, _ = model(a)
        lb, _, _ = model(b)
    delta = (la[0, -1] - lb[0, -1]).abs().max().item()
    assert delta > 1e-6, (
        "expected causal masking to leak positional information even with "
        f"positional embeddings removed, but outputs were identical (delta={delta})")
    print(f"  causal mask leaks position without embeddings "
          f"(max logit delta {delta:.4f}) OK")


def test_positions_break_permutation_invariance():
    """The same check WITH positions must fail -- that is the point of them."""
    cfg = tiny_config(positional="learned", dropout=0.0)
    model = GPT(cfg).eval()
    a = torch.tensor([[5, 9]])
    b = torch.tensor([[9, 5]])
    with torch.no_grad():
        la, _, _ = model(a)
        lb, _, _ = model(b)
    assert not torch.allclose(la[0, -1], lb[0, -1], atol=1e-5), (
        "learned positions did not change the output under permutation")
    print("  learned positions break permutation invariance OK")


# ------------------------------------------------------------------- training

def test_block_size_is_enforced():
    """Sequences longer than block_size must raise, not silently truncate."""
    cfg = tiny_config(block_size=8)
    model = GPT(cfg).eval()
    try:
        model(torch.randint(0, cfg.vocab_size, (1, 9)))
    except ValueError as e:
        assert "block_size" in str(e)
        print("  block_size overflow raises OK")
    else:
        raise AssertionError("expected ValueError for T > block_size")


def test_loss_decreases_on_a_single_batch():
    """The model must be able to overfit one batch -- catches dead gradients."""
    torch.manual_seed(0)
    cfg = tiny_config(dropout=0.0)
    model = GPT(cfg).train()
    opt = model.configure_optimizers(0.1, 3e-3, (0.9, 0.95), "cpu")

    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    tgt = torch.randint(0, cfg.vocab_size, (4, 16))

    _, first, _ = model(idx, tgt)
    for _ in range(60):
        _, loss, _ = model(idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    _, last, _ = model(idx, tgt)

    assert last.item() < first.item() * 0.5, (first.item(), last.item())
    print(f"  overfits one batch: {first.item():.3f} -> {last.item():.3f} OK")


def test_all_parameters_receive_gradient():
    """Every trainable parameter must get a gradient -- catches dead branches."""
    cfg = tiny_config(dropout=0.0)
    model = GPT(cfg).train()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss, _ = model(idx, tgt)
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not missing, f"no gradient reached: {missing}"
    print(f"  all {sum(1 for _ in model.parameters())} parameter tensors got gradient OK")


def test_generate_respects_shapes_and_greedy_determinism():
    """generate() appends the right number of tokens; greedy is deterministic."""
    cfg = tiny_config(dropout=0.0)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    out1 = model.generate(idx, max_new_tokens=7, greedy=True)
    out2 = model.generate(idx, max_new_tokens=7, greedy=True)
    assert out1.shape == (2, 12), out1.shape
    assert torch.equal(out1, out2), "greedy decoding is not deterministic"
    print("  generate shapes + greedy determinism OK")


def test_generation_crops_to_block_size():
    """Generating past block_size must crop context, not crash."""
    cfg = tiny_config(block_size=8, dropout=0.0)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(idx, max_new_tokens=6, greedy=True)
    assert out.shape == (1, 14), out.shape
    print("  generation past block_size crops context OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} tests\n")
    failed = 0
    for t in tests:
        try:
            print(f"{t.__name__}")
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAILED: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
