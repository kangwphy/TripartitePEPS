#!/usr/bin/env python3
"""Small administrative/provenance tests; run through Slurm/preempt."""
import importlib.util
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "production/rk_ising/rk_ising_campaign.py"
SPEC = importlib.util.spec_from_file_location("rk_ising_campaign", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def test_four_independent_modes(self):
        self.assertEqual(campaign.series_label(False, False), "no_constraint")
        self.assertEqual(campaign.series_label(True, False), "Z2")
        self.assertEqual(campaign.series_label(True, True), "Z2_fix")
        self.assertEqual(campaign.series_label(False, True), "no_constraint_fix")

    def test_grid_keeps_original_precision(self):
        fields = ["task_id", "point_id", "beta", "chi", "seed"]
        rows = [dict(zip(fields, values)) for values in
                (("0", "p000", "0.44068679350977147", "4", "1"),
                 ("1", "p001", "0.45000000000000001", "36", "2"))]
        original = campaign.csv_bytes(rows, fields)
        result = campaign.validate_grid(fields, rows, True, 2)
        self.assertEqual(result["point_count"], 2)
        self.assertEqual(result["resource_groups"]["32G"]["array_spec"], "0")
        self.assertEqual(result["resource_groups"]["192G"]["array_spec"], "1")
        self.assertEqual(original, campaign.csv_bytes(rows, fields))
        with self.assertRaises(ValueError):
            campaign.validate_grid(fields, rows, True, 306)
        rows[1]["beta"] = rows[0]["beta"]
        rows[1]["chi"] = rows[0]["chi"]
        with self.assertRaises(ValueError):
            campaign.validate_grid(fields, rows, True)

    def test_z2_odd_chi_rejected_only_when_enabled(self):
        rows = [dict(task_id="0", point_id="p000", beta="0.45", chi="3", seed="1")]
        campaign.validate_grid(list(rows[0]), rows, False)
        with self.assertRaises(ValueError):
            campaign.validate_grid(list(rows[0]), rows, True)

    def test_flat_toml_roundtrip(self):
        config = {"use_z2": True, "use_cap_fix": False, "series_label": "Z2",
                  "threshold": 1e-10, "chis": [4, 6, 8], "iterations": 240000}
        self.assertEqual(config, campaign.tomllib.loads(campaign.toml_bytes(config).decode()))
        with self.assertRaises(ValueError):
            campaign.toml_bytes({"unsupported": {"nested": 1}})

    def test_source_switch_is_authoritative(self):
        self.assertTrue(campaign.source_is_z2({"boundary_symmetry": "bosonic_Z2"}))
        self.assertFalse(campaign.source_is_z2({"use_z2": False, "boundary_symmetry": "bosonic_Z2"}))
        with self.assertRaises(ValueError):
            campaign.source_is_z2({"use_z2": "false"})

    def test_safe_paths(self):
        self.assertEqual(campaign.safe_relative("checkpoints/p000_input_boundary.jls"),
                         Path("checkpoints/p000_input_boundary.jls"))
        for value in ("/tmp/elsewhere", "../outside", "code/../../outside"):
            with self.assertRaises(ValueError):
                campaign.safe_relative(value)

    def test_byte_preserving_copy_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="rk-campaign-test-") as folder:
            source, output = Path(folder) / "source", Path(folder) / "copy"
            source.write_bytes(b"\x00saved MPS\xff\n")
            digest = campaign.copy_stable(source, output)
            self.assertEqual(source.read_bytes(), output.read_bytes())
            self.assertEqual(digest, campaign.sha(output))
            with self.assertRaises(FileExistsError):
                campaign.copy_stable(source, output)

    def test_nonfinite_not_counted_as_entropy(self):
        self.assertTrue(campaign.finite("-1.234"))
        for value in ("NaN", "Inf", "", None):
            self.assertFalse(campaign.finite(value))

    def test_full_snapshot_collect_and_series_isolation(self):
        with tempfile.TemporaryDirectory(prefix="rk-campaign-integration-") as folder:
            base = Path(folder)
            clean, source, root = base / "clean", base / "Z2", base / "Z2_fix"
            required = ("Project.toml", "Manifest.toml", "src/LMPSVUMPS.jl",
                        "src/models/rk_ising_z2.jl", "src/algorithms/sixr_cap_selected.jl",
                        "production/rk_ising/rk_ising_point.jl", "production/rk_ising/bare_strip_checks.jl",
                        "production/rk_ising/rk_ising_campaign.py", "jobs/rk_ising_campaign.sh")
            for name in required:
                path = clean / campaign.PACKAGE / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture for provenance-only test\n")
            (source / "checkpoints").mkdir(parents=True)
            source_config = source / "config.toml"
            source_config.write_text('boundary_symmetry = "bosonic_Z2"\nsixr_tolerance = 1e-10\n')
            grid = source / "requested_grid.csv"
            grid.write_bytes(b"task_id,point_id,beta,chi,seed\r\n0,p000,0.45000000000000001,4,7\r\n")
            old_boundary = source / "checkpoints/p000_boundary.jls"
            old_boundary.write_bytes(b"fixture checkpoint: immutable old boundary")
            args = argparse.Namespace(clean=clean, root=root, source_grid=grid, source_config=source_config,
                                      reuse_from=source, use_z2=True, use_cap_fix=True, expected_count=1)
            campaign.init_campaign(args)
            provenance, config, _, specs = campaign.verify_campaign(root)
            self.assertEqual((root / "requested_grid.csv").read_bytes(), grid.read_bytes())
            self.assertEqual(config["series_label"], "Z2_fix")
            self.assertEqual(config["cap_fix_tolerance"], 1e-12)
            _, inputs, _ = campaign.read_csv(root / "inputs.csv")
            self.assertEqual([row["status"] for row in inputs], ["copied", "missing", "missing"])
            old_boundary.write_bytes(b"old job has moved to a new boundary")
            campaign.verify_campaign(root)  # No live-source dependency after initialization.
            campaign.init_campaign(args)    # Identical init only verifies; never imports new bytes.
            result = dict(specs[0], use_z2="true", use_cap_fix="true", series_label="Z2_fix",
                          config_sha256=provenance["config_sha256"], code_sha256=provenance["source_manifest_sha256"],
                          stage="complete", status="diagnostic_failed", numerical_pass="false", tildeS="-0.123")
            (root / "points/p000.csv").write_bytes(campaign.csv_bytes([result]))
            campaign.collect_campaign(root)
            report = json.loads((root / "collection.json").read_text())
            self.assertEqual(report["finite_entropy_count"], 1)
            self.assertEqual(report["by_quality"], {"complete_diagnostic": 1})
            result["series_label"] = "Z2"
            (root / "points/p000.csv").write_bytes(campaign.csv_bytes([result]))
            with self.assertRaises(ValueError):
                campaign.collect_campaign(root)
            args.use_cap_fix = False
            with self.assertRaises(ValueError):
                campaign.init_campaign(args)
            (root / "checkpoints/p000_input_boundary.jls").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                campaign.verify_campaign(root)


if __name__ == "__main__":
    campaign.require_preempt()
    unittest.main()
