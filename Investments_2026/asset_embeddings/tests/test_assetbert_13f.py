"""No-source-data contract checks for ranked-portfolio AssetBERT."""

import random
import unittest

from src.assetbert_13f import (
    ASSET_TOKEN_OFFSET,
    IGNORE_INDEX,
    MASK_ID,
    PAD_ID,
    AssetBERT,
    AssetBERTConfig,
    encode_masked_rank2,
    encode_ranked_portfolio,
    mask_chunk_for_training,
    masked_cross_entropy,
    rank2_logits,
    torch,
)


class _BranchRng:
    """Force one draw from each BERT replacement branch."""

    def __init__(self) -> None:
        self.draws = iter((0.0, 0.85, 0.95))

    def sample(self, population: list[int], count: int) -> list[int]:
        return population[:count]

    def random(self) -> float:
        return next(self.draws)

    def randrange(self, stop: int) -> int:
        return stop - 1


class PortfolioEncodingTests(unittest.TestCase):
    def test_chunking_preserves_stock_positions_and_padding(self) -> None:
        chunks = encode_ranked_portfolio(list(range(130)), vocabulary_size=131)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk.token_ids) == 64 for chunk in chunks))
        self.assertEqual([sum(chunk.attention_mask) for chunk in chunks], [44, 43, 43])
        self.assertEqual(chunks[0].token_ids[0], ASSET_TOKEN_OFFSET)
        self.assertEqual(chunks[0].token_ids[43], 43 + ASSET_TOKEN_OFFSET)
        self.assertEqual(chunks[1].token_ids[0], 44 + ASSET_TOKEN_OFFSET)
        self.assertEqual(chunks[2].token_ids[0], 87 + ASSET_TOKEN_OFFSET)
        self.assertEqual(chunks[2].token_ids[42], 129 + ASSET_TOKEN_OFFSET)
        self.assertEqual(chunks[2].token_ids[43:], (PAD_ID,) * 21)
        self.assertEqual(chunks[2].attention_mask, (True,) * 43 + (False,) * 21)

    def test_inference_inserts_mask_at_original_rank_two_without_label(self) -> None:
        # Asset 7 is the hidden rank-two label; it is never passed to this API.
        ranked = [0, -1, 2, 3, 4]
        chunk = encode_masked_rank2(ranked, vocabulary_size=8)[0]
        self.assertEqual(
            chunk.token_ids[:5],
            (
                0 + ASSET_TOKEN_OFFSET,
                MASK_ID,
                2 + ASSET_TOKEN_OFFSET,
                3 + ASSET_TOKEN_OFFSET,
                4 + ASSET_TOKEN_OFFSET,
            ),
        )
        self.assertTrue(chunk.attention_mask[1])
        self.assertNotIn(7 + ASSET_TOKEN_OFFSET, chunk.token_ids)

    def test_rank_two_mask_does_not_shift_later_chunk_boundary(self) -> None:
        ranked = [0, -1] + list(range(2, 65))
        chunks = encode_masked_rank2(ranked, vocabulary_size=65)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([sum(chunk.attention_mask) for chunk in chunks], [33, 32])
        self.assertEqual(chunks[0].token_ids[1], MASK_ID)
        self.assertEqual(chunks[0].token_ids[32], 32 + ASSET_TOKEN_OFFSET)
        self.assertEqual(chunks[1].token_ids[0], 33 + ASSET_TOKEN_OFFSET)
        self.assertEqual(chunks[1].token_ids[31], 64 + ASSET_TOKEN_OFFSET)

    def test_65_stock_training_portfolio_has_no_singleton_chunk(self) -> None:
        chunks = encode_ranked_portfolio(list(range(65)), vocabulary_size=65)
        self.assertEqual([sum(chunk.attention_mask) for chunk in chunks], [33, 32])
        self.assertTrue(all(sum(chunk.attention_mask) > 1 for chunk in chunks))
        self.assertEqual(chunks[1].token_ids[0], 33 + ASSET_TOKEN_OFFSET)

    def test_masking_excludes_existing_hidden_slot_and_padding(self) -> None:
        chunk = encode_masked_rank2([0, -1] + list(range(1, 20)), vocabulary_size=30)[0]
        masked = mask_chunk_for_training(chunk, 30, random.Random(17))
        self.assertEqual(sum(label != IGNORE_INDEX for label in masked.labels), 3)
        self.assertEqual(masked.labels[1], IGNORE_INDEX)
        self.assertEqual(masked.input_ids[1], MASK_ID)
        self.assertTrue(all(label == IGNORE_INDEX for label in masked.labels[21:]))
        self.assertEqual(masked.attention_mask, chunk.attention_mask)

    def test_masking_replacement_branches_and_seed_are_reproducible(self) -> None:
        chunk = encode_ranked_portfolio(list(range(20)), vocabulary_size=21)[0]
        masked = mask_chunk_for_training(chunk, 21, _BranchRng())
        self.assertEqual(masked.labels[:3], (0, 1, 2))
        self.assertEqual(masked.input_ids[0], MASK_ID)
        self.assertEqual(masked.input_ids[1], chunk.token_ids[1])
        self.assertEqual(masked.input_ids[2], 20 + ASSET_TOKEN_OFFSET)
        self.assertEqual(
            mask_chunk_for_training(chunk, 21, random.Random(4)),
            mask_chunk_for_training(chunk, 21, random.Random(4)),
        )

    def test_bad_or_duplicate_vocabulary_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            encode_ranked_portfolio([0, 0], vocabulary_size=3)
        with self.assertRaisesRegex(ValueError, "fitted vocabulary"):
            encode_ranked_portfolio([0, 3], vocabulary_size=3)
        with self.assertRaisesRegex(ValueError, "fitted vocabulary"):
            encode_masked_rank2([0, -1, True], vocabulary_size=3)
        with self.assertRaisesRegex(ValueError, "rank-two"):
            encode_masked_rank2([0, 1, -1], vocabulary_size=3)
        with self.assertRaisesRegex(ValueError, "fitted vocabulary"):
            encode_masked_rank2([0, -1, -1], vocabulary_size=3)


@unittest.skipUnless(torch is not None, "Optional PyTorch is not installed")
class AssetBERTTorchTests(unittest.TestCase):
    def test_forward_shape_and_selected_token_gradient(self) -> None:
        config = AssetBERTConfig(
            vocabulary_size=12, embedding_dim=4, max_stocks=8, dropout=0.0, seed=5
        )
        model = AssetBERT(config)
        self.assertEqual(len(model.encoder_layers), 4)
        self.assertEqual(model.encoder_layers[0].self_attn.num_heads, 2)
        self.assertFalse(
            torch.equal(
                model.encoder_layers[0].linear1.weight,
                model.encoder_layers[1].linear1.weight,
            )
        )
        chunks = [
            encode_ranked_portfolio(list(range(8)), 12, max_stocks=8)[0],
            encode_ranked_portfolio(list(range(8, 12)), 12, max_stocks=8)[0],
        ]
        masked = [
            mask_chunk_for_training(chunk, 12, random.Random(seed))
            for seed, chunk in enumerate(chunks)
        ]
        tokens = torch.tensor([item.input_ids for item in masked], dtype=torch.long)
        attention = torch.tensor([item.attention_mask for item in masked], dtype=torch.bool)
        labels = torch.tensor([item.labels for item in masked], dtype=torch.long)
        logits = model(tokens, attention)
        self.assertEqual(tuple(logits.shape), (2, 8, 12))
        loss = masked_cross_entropy(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.position_embedding.weight.grad.abs().sum().item(), 0)

    def test_inference_uses_mask_and_restores_training_mode(self) -> None:
        model = AssetBERT(
            AssetBERTConfig(vocabulary_size=10, embedding_dim=4, max_stocks=8, seed=7)
        )
        self.assertTrue(model.training)
        logits_a = rank2_logits(model, [0, -1, 2, 3, 4])
        logits_b = rank2_logits(model, [0, -1, 2, 3, 4])
        self.assertEqual(tuple(logits_a.shape), (10,))
        self.assertTrue(torch.allclose(logits_a, logits_b))
        self.assertTrue(model.training)

    def test_seeded_initialization_and_empty_target_guard(self) -> None:
        config = AssetBERTConfig(vocabulary_size=8, max_stocks=8, seed=7)
        first, second = AssetBERT(config), AssetBERT(config)
        for left, right in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(left, right))
        with self.assertRaisesRegex(ValueError, "selected stock target"):
            masked_cross_entropy(
                torch.zeros(1, 8, 8), torch.full((1, 8), IGNORE_INDEX)
            )

    def test_forward_rejects_attended_padding(self) -> None:
        model = AssetBERT(AssetBERTConfig(vocabulary_size=8, max_stocks=8))
        chunk = encode_ranked_portfolio([0, 1], 8, max_stocks=8)[0]
        tokens = torch.tensor([chunk.token_ids], dtype=torch.long)
        wrong_mask = torch.ones_like(tokens, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "Padding positions"):
            model(tokens, wrong_mask)


if __name__ == "__main__":
    unittest.main()
