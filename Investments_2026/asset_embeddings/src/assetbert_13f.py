"""Ranked-portfolio AssetBERT core for point-in-time 13F representations.

This module deliberately accepts only integer asset IDs from a caller-owned,
training-fitted vocabulary. It does not select a universe, join identifiers,
read files, or decide whether a holding was available at a decision time.
Those safeguards belong to the sealed benchmark that supplies the portfolios.

Paper-aligned choices are 64-stock contexts, four encoder layers, two attention
heads, learned asset and position embeddings, and a 15% masked-token objective
with 80/10/10 [MASK]/unchanged/random replacement. The paper says portfolios
longer than 64 stocks are divided into equal chunks. Here chunks are contiguous
rank blocks whose sizes differ by at most one; any surplus stock goes in an
earlier chunk. The source paper does not fully specify the optimizer,
feed-forward width, dropout, or output-weight tying. Here feed-forward width
is 4x the embedding dimension, dropout defaults to 0.1, and the asset output
projection is untied. No optimizer is selected here.
CLS/SEP/UNK tokens are unnecessary for this token-prediction-only core, so the
only reserved input tokens are PAD and MASK. These implementation choices are
adaptations, not grounds to claim exact reproduction of published results.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import random
from typing import Sequence

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    nn = None
    F = None


PAD_ID = 0
MASK_ID = 1
ASSET_TOKEN_OFFSET = 2
IGNORE_INDEX = -100
PAPER_CONTEXT_STOCKS = 64
PAPER_ENCODER_LAYERS = 4
PAPER_ATTENTION_HEADS = 2
PAPER_MASK_PROBABILITY = 0.15


@dataclass(frozen=True)
class PortfolioChunk:
    """Fixed-width tokens and true positions; no identifier strings are retained."""

    token_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]


@dataclass(frozen=True)
class MaskedChunk:
    """One dynamically masked training example with asset-index labels."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class AssetBERTConfig:
    """The paper's encoder depth/head count with declared implementation choices."""

    vocabulary_size: int
    embedding_dim: int = 4
    max_stocks: int = PAPER_CONTEXT_STOCKS
    feedforward_multiplier: int = 4
    dropout: float = 0.1
    seed: int = 17

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        if self.embedding_dim < 2 or self.embedding_dim % PAPER_ATTENTION_HEADS:
            raise ValueError("embedding_dim must be even and at least two")
        if not 2 <= self.max_stocks <= PAPER_CONTEXT_STOCKS:
            raise ValueError("max_stocks must be between two and 64")
        if self.feedforward_multiplier < 1:
            raise ValueError("feedforward_multiplier must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def _validate_asset_ids(asset_ids: Sequence[int], vocabulary_size: int) -> None:
    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be at least two")
    if len(asset_ids) == 0:
        raise ValueError("A portfolio must contain at least one visible asset")
    if any(
        isinstance(asset_id, bool)
        or not isinstance(asset_id, Integral)
        or not 0 <= asset_id < vocabulary_size
        for asset_id in asset_ids
    ):
        raise ValueError("Asset IDs must be integer indices in the fitted vocabulary")
    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError("A ranked portfolio cannot contain duplicate assets")


def _make_chunks(tokens: Sequence[int], *, max_stocks: int) -> tuple[PortfolioChunk, ...]:
    if not 2 <= max_stocks <= PAPER_CONTEXT_STOCKS:
        raise ValueError("max_stocks must be between two and 64")
    chunk_count = (len(tokens) + max_stocks - 1) // max_stocks
    smaller_size, larger_count = divmod(len(tokens), chunk_count)
    chunks = []
    start = 0
    for chunk_number in range(chunk_count):
        size = smaller_size + (chunk_number < larger_count)
        block = tuple(tokens[start : start + size])
        start += size
        pad_count = max_stocks - len(block)
        chunks.append(
            PortfolioChunk(
                token_ids=block + (PAD_ID,) * pad_count,
                attention_mask=(True,) * len(block) + (False,) * pad_count,
            )
        )
    return tuple(chunks)


def encode_ranked_portfolio(
    asset_ids: Sequence[int],
    vocabulary_size: int,
    *,
    max_stocks: int = PAPER_CONTEXT_STOCKS,
) -> tuple[PortfolioChunk, ...]:
    """Encode descending-value asset IDs; caller must break value ties stably."""

    _validate_asset_ids(asset_ids, vocabulary_size)
    return _make_chunks(
        [int(asset_id) + ASSET_TOKEN_OFFSET for asset_id in asset_ids],
        max_stocks=max_stocks,
    )


def encode_masked_rank2(
    ranked_asset_ids: Sequence[int],
    vocabulary_size: int,
    *,
    max_stocks: int = PAPER_CONTEXT_STOCKS,
) -> tuple[PortfolioChunk, ...]:
    """Map the evaluator's rank-two ``-1`` sentinel to [MASK].

    ``ranked_asset_ids`` preserves original descending-value ranks and must
    contain exactly one ``-1`` at index one. All other entries are visible
    train-vocabulary indices. The hidden asset label is never passed here.
    The first balanced chunk (at most 64 stocks) is the context available to
    the rank-two query; later chunks cannot influence that query.
    """

    if len(ranked_asset_ids) < 2 or ranked_asset_ids[1] != -1:
        raise ValueError("The original rank-two position must contain -1")
    visible = [asset_id for position, asset_id in enumerate(ranked_asset_ids) if position != 1]
    _validate_asset_ids(visible, vocabulary_size)
    tokens = [
        MASK_ID if position == 1 else int(asset_id) + ASSET_TOKEN_OFFSET
        for position, asset_id in enumerate(ranked_asset_ids)
    ]
    return _make_chunks(tokens, max_stocks=max_stocks)


def mask_chunk_for_training(
    chunk: PortfolioChunk,
    vocabulary_size: int,
    rng: random.Random,
    *,
    mask_probability: float = PAPER_MASK_PROBABILITY,
) -> MaskedChunk:
    """Select about 15% of visible stocks and apply BERT's 80/10/10 policy.

    A fixed rounded count (at least one when assets are present) is sampled per
    chunk, rather than independent Bernoulli draws. Pre-existing [MASK] slots
    and padding are never selected. The same seeded ``rng`` should be advanced
    across chunks to vary masks reproducibly.
    """

    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be at least two")
    if not 0 < mask_probability <= 1:
        raise ValueError("mask_probability must be in (0, 1]")
    if len(chunk.token_ids) != len(chunk.attention_mask):
        raise ValueError("Tokens and attention mask must have equal lengths")
    visible_positions = [
        position
        for position, (token, attends) in enumerate(
            zip(chunk.token_ids, chunk.attention_mask)
        )
        if attends and ASSET_TOKEN_OFFSET <= token < vocabulary_size + ASSET_TOKEN_OFFSET
    ]
    if not visible_positions:
        raise ValueError("A training chunk must contain a visible asset")
    selected_count = max(1, round(mask_probability * len(visible_positions)))
    selected = rng.sample(visible_positions, selected_count)
    input_ids = list(chunk.token_ids)
    labels = [IGNORE_INDEX] * len(input_ids)
    for position in selected:
        original = input_ids[position]
        labels[position] = original - ASSET_TOKEN_OFFSET
        draw = rng.random()
        if draw < 0.8:
            input_ids[position] = MASK_ID
        elif draw < 0.9:
            pass  # The unchanged-token branch is part of BERT training only.
        else:
            input_ids[position] = rng.randrange(vocabulary_size) + ASSET_TOKEN_OFFSET
    return MaskedChunk(
        input_ids=tuple(input_ids),
        attention_mask=chunk.attention_mask,
        labels=tuple(labels),
    )


if nn is not None:

    class AssetBERT(nn.Module):
        """Four-layer, two-head encoder with a full-asset-vocabulary output."""

        def __init__(self, config: AssetBERTConfig) -> None:
            super().__init__()
            self.config = config
            # Initialization is reproducible without changing the caller's CPU RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed)
                self.asset_embedding = nn.Embedding(
                    config.vocabulary_size + ASSET_TOKEN_OFFSET,
                    config.embedding_dim,
                    padding_idx=PAD_ID,
                )
                self.position_embedding = nn.Embedding(
                    config.max_stocks, config.embedding_dim
                )
                self.embedding_norm = nn.LayerNorm(config.embedding_dim)
                # Construct each layer separately. TransformerEncoder deep-copies
                # one layer, leaving all four with identical initial weights.
                self.encoder_layers = nn.ModuleList(
                    nn.TransformerEncoderLayer(
                        d_model=config.embedding_dim,
                        nhead=PAPER_ATTENTION_HEADS,
                        dim_feedforward=(
                            config.feedforward_multiplier * config.embedding_dim
                        ),
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=False,
                    )
                    for _ in range(PAPER_ENCODER_LAYERS)
                )
                self.asset_output = nn.Linear(
                    config.embedding_dim, config.vocabulary_size
                )

        def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """Return ``[batch, positions, vocabulary_size]`` asset logits."""

            if token_ids.ndim != 2 or attention_mask.shape != token_ids.shape:
                raise ValueError("token_ids and attention_mask must be matching 2D tensors")
            if token_ids.shape[1] != self.config.max_stocks:
                raise ValueError("Input width must match config.max_stocks")
            if token_ids.dtype != torch.long or attention_mask.dtype != torch.bool:
                raise TypeError("token_ids must be long and attention_mask must be bool")
            if token_ids.min() < PAD_ID or token_ids.max() >= self.asset_embedding.num_embeddings:
                raise ValueError("Input token is outside the fitted vocabulary")
            if not torch.equal(attention_mask, token_ids.ne(PAD_ID)):
                raise ValueError("Padding positions must be the only unattended tokens")
            if not attention_mask.any(dim=1).all():
                raise ValueError("Each sequence must contain at least one attended position")
            positions = torch.arange(
                token_ids.shape[1], device=token_ids.device, dtype=torch.long
            )
            states = self.embedding_norm(
                self.asset_embedding(token_ids) + self.position_embedding(positions)
            )
            encoded = states
            for layer in self.encoder_layers:
                encoded = layer(encoded, src_key_padding_mask=~attention_mask)
            return self.asset_output(encoded)

else:

    class AssetBERT:  # pragma: no cover - exercised only without optional torch
        """Importable placeholder explaining the optional PyTorch requirement."""

        def __init__(self, config: AssetBERTConfig) -> None:
            raise ImportError("AssetBERT requires the optional PyTorch dependency")


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy over selected real-stock targets only."""

    if torch is None:
        raise ImportError("Masked-token training requires the optional PyTorch dependency")
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, positions, assets] and matching labels")
    selected = labels.ne(IGNORE_INDEX)
    if not selected.any():
        raise ValueError("At least one selected stock target is required")
    return F.cross_entropy(logits[selected], labels[selected])


def rank2_logits(model: AssetBERT, ranked_asset_ids: Sequence[int]) -> torch.Tensor:
    """Predict the hidden rank-two asset using an explicit [MASK] input.

    The hidden label is not an argument. Returns one logit per asset in the
    fitted vocabulary. Candidate exclusion and ASMP normalization are separate
    benchmark responsibilities; this model must not see those labels or splits.
    """

    if torch is None:
        raise ImportError("AssetBERT inference requires the optional PyTorch dependency")
    chunk = encode_masked_rank2(
        ranked_asset_ids,
        model.config.vocabulary_size,
        max_stocks=model.config.max_stocks,
    )[0]
    device = next(model.parameters()).device
    token_ids = torch.tensor([chunk.token_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [chunk.attention_mask], dtype=torch.bool, device=device
    )
    prior_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            return model(token_ids, attention_mask)[0, 1]
    finally:
        model.train(prior_training)
