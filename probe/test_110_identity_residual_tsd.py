#!/usr/bin/env python3
"""Unit-level tests for identity residual TSD helpers; no model load required."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).with_name("110_identity_residual_tsd.py")
SPEC = importlib.util.spec_from_file_location("identity_residual_tsd", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class IdentityResidualTsdTest(unittest.TestCase):
    def test_sigma_resolution_does_not_require_step_index(self):
        scheduler = SimpleNamespace(
            timesteps=torch.tensor([900.0, 600.0, 300.0]),
            sigmas=torch.tensor([0.9, 0.6, 0.3, 0.0]),
            step_index=None,
        )
        index, sigma = M.resolve_scheduler_sigma(scheduler, torch.tensor(590.0), torch.zeros(1))
        self.assertEqual(index, 1)
        self.assertAlmostEqual(float(sigma), 0.6, places=6)

    def test_mean_consensus_preserves_shared_direction(self):
        residual = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
        direction, similarities, _ = M.aggregate_residuals([residual, 3.0 * residual], "mean")
        self.assertTrue(torch.allclose(direction, M.rms_normalize(residual), atol=1e-5))
        self.assertTrue(all(value > 0.999 for value in similarities))

    def test_svd_consensus_is_finite(self):
        residual = torch.randn(2, 1, 4, 4, generator=torch.Generator().manual_seed(7))
        direction, similarities, extra = M.aggregate_residuals([residual, residual + 0.01], "svd")
        self.assertTrue(torch.isfinite(direction).all())
        self.assertEqual(len(similarities), 2)
        self.assertEqual(len(extra["gram_eigenvalues"]), 2)

    def test_zero_residual_produces_zero_update(self):
        direction, similarities, _ = M.aggregate_residuals([torch.zeros(2, 1, 8, 8)], "mean")
        update = M.latent_pseudo_update(direction, 0.8, step_size=0.1, scale=1.0)
        self.assertEqual(tuple(update.shape), (1, 2, 8, 8))
        self.assertEqual(float(update.abs().max()), 0.0)
        self.assertEqual(similarities, [0.0])

    def test_low_frequency_ratio_distinguishes_constant_from_checkerboard(self):
        constant = torch.ones(2, 1, 16, 16)
        grid = torch.arange(16)
        checkerboard = ((grid[:, None] + grid[None, :]) % 2).float().mul(2).sub(1)
        checkerboard = checkerboard.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
        self.assertGreater(M.lowfreq_ratio(constant), 0.99)
        self.assertLess(M.lowfreq_ratio(checkerboard), 0.01)

    def test_nuisance_filters_remove_global_and_low_frequency_fields(self):
        constant = torch.ones(2, 1, 16, 16)
        self.assertEqual(float(M.filter_residual(constant, "channel_center").abs().max()), 0.0)
        self.assertEqual(float(M.filter_residual(constant, "highpass").abs().max()), 0.0)

        ramp = torch.linspace(-1, 1, 16).reshape(1, 1, 1, 16).repeat(2, 1, 16, 1)
        filtered = M.filter_residual(ramp, "highpass_channel")
        self.assertLess(M.lowfreq_ratio(filtered), M.lowfreq_ratio(ramp))

    def test_local_reference_agreement_suppresses_disputed_patch(self):
        shared = torch.ones(2, 1, 2, 2)
        left = shared.clone()
        right = shared.clone()
        right[:, :, 0, 0] = -1
        gate, stats = M.local_reference_agreement([left, right])
        self.assertLess(float(gate[0, 0]), float(gate[1, 1]))
        self.assertLess(stats["local_agreement_min"], stats["local_agreement_mean"])

    def test_nuisance_projection_removes_known_direction(self):
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 2, 2)
        nuisance = torch.tensor([0.0, 1.0, 0.0, 0.0]).reshape(1, 1, 2, 2)
        mixed = identity + nuisance
        cleaned, stats = M.project_nuisance_subspace(
            mixed,
            [nuisance],
            ridge=0.0,
            max_removal=1.0,
        )
        self.assertGreater(M.cosine(cleaned, identity), M.cosine(mixed, identity))
        self.assertAlmostEqual(float(cleaned.reshape(-1)[1]), 0.0, places=5)
        self.assertEqual(stats["nuisance_count"], 1)

    def test_semantic_aggregate_keeps_shared_identity(self):
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 2, 2)
        nuisance = torch.tensor([0.0, 1.0, 0.0, 0.0]).reshape(1, 1, 2, 2)
        anchors = [identity + 0.2 * nuisance, identity + 0.2 * nuisance]
        direction, similarities, stats = M.semantic_aggregate_residuals(
            anchors,
            [nuisance],
            nuisance_ridge=0.0,
            max_nuisance_removal=1.0,
        )
        self.assertGreater(M.cosine(direction, identity), 0.99)
        self.assertTrue(all(value > 0.99 for value in similarities))
        self.assertEqual(stats["nuisance_count"], 1)

    def test_hidden_layer_spec_parses_ranges(self):
        self.assertEqual(M.parse_layer_spec("21-23,25", 30), [21, 22, 23, 25])
        with self.assertRaises(ValueError):
            M.parse_layer_spec("29-31", 30)

    def test_hidden_support_rejects_nuisance_unstable_token(self):
        anchor = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
        )
        nuisance = anchor.clone()
        nuisance[:, 3] = torch.tensor([0.0, 1.0])
        hidden = {
            "r0:identity": {21: anchor},
            "r1:identity": {21: anchor.clone()},
            "r0:crop": {21: nuisance},
        }
        gate, stats = M.hidden_semantic_support_gate(
            hidden,
            anchor_names=["r0:identity", "r1:identity"],
            nuisance_pairs=[("r0:crop", "r0:identity")],
            output_size=(2, 2),
            floor=0.1,
        )
        self.assertLess(float(gate[1, 1]), float(gate[0, 0]))
        self.assertEqual(stats["hidden_support_layers"], [21])

    def test_identity_interaction_cancels_generic_reference_effect(self):
        base_without = torch.tensor([1.0, 2.0])
        generic_reference = torch.tensor([0.5, -0.25])
        identity_effect = torch.tensor([0.0, 0.75])
        teacher_without = base_without + generic_reference
        base_with = base_without + torch.tensor([2.0, 3.0])
        teacher_with = base_with + generic_reference + identity_effect
        residual = M.identity_interaction_residual(
            teacher_with,
            base_with,
            teacher_without,
            base_without,
        )
        self.assertTrue(torch.allclose(residual, identity_effect))


if __name__ == "__main__":
    unittest.main()
