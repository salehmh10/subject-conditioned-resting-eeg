"""Standard-library tests of portability safeguards, never scientific execution.

All HDF5/NPZ-named files created here contain tiny arbitrary text sentinels, not
waveforms. No prepare/run operation is exercised with real research inputs.
"""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/analyze_validation_portable.py"
ORIGINAL = ROOT / "evidence/validation_reference/original_source/analyze_round3.py"


def load_module():
    spec = importlib.util.spec_from_file_location("portable_validation_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def original_plan_constant(name):
    """Read literal method definitions from archived source without importing it."""
    tree = ast.parse(ORIGINAL.read_text(encoding="utf-8-sig"))
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    assignment = next(node for node in ast.walk(prepare) if isinstance(node, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == "plan" for t in node.targets))
    value = next(v for k, v in zip(assignment.value.keys, assignment.value.values) if isinstance(k, ast.Constant) and k.value == name)

    class BandsLiteral(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "BANDS":
                return ast.parse("((1.,4.),(4.,8.),(8.,13.),(13.,30.))", mode="eval").body
            return node

    return json.loads(json.dumps(ast.literal_eval(BandsLiteral().visit(copy.deepcopy(value)))))


class PortableValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="validation metadata test spaces ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.module = load_module()

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", "-S", str(SCRIPT), *map(str, args)],
                              cwd=self.base, text=True, capture_output=True, check=False)

    def fixture(self, with_plan=True):
        """Create synthetic metadata and unread waveform-file sentinels."""
        inputs = self.base / "external inputs with spaces"
        inputs.mkdir()
        processed = inputs / "hdf5"
        processed.mkdir()
        tables = inputs / "tables"
        tables.mkdir()
        pool_dirs = {g: inputs / (g + " pool") for g in ("B0", "B1", "B2")}
        sentinel = b"fixture only; not waveform data\n"
        digest = hashlib.sha256(sentinel).hexdigest()
        for root in pool_dirs.values():
            (root / "files").mkdir(parents=True)
            (root / "files/pool.npz").write_bytes(sentinel)
        subjects = [f"fixture-{i:03d}" for i in range(198)]
        split_rows = []
        selections = {}
        manifests = {"B0": {"committed": {}}, "B1": {"committed": {}}}
        inventories = []
        for si, sid in enumerate(subjects):
            (processed / (sid + "_all_blocks.h5")).write_bytes(sentinel)
            inventories.append({"subject_id": sid, "path": "UNAVAILABLE_OLD_ROOT/" + sid + ".h5",
                                "bytes": len(sentinel), "expected_sha256": digest})
            for ti, state in enumerate(("EO", "EC")):
                stratum = 2 * si + ti
                counts = {"Train": 120 + (stratum < 315), "Validation": 24 + (stratum < 131)}
                for split, count in counts.items():
                    split_rows.append({"subject_id": sid, "split": split, "block_eligible": "True",
                                       "eye_state": state, "usable_windows": count,
                                       "source_output_sha256": digest, "block_index": si,
                                       "block_uid": sid + "_" + state + "_" + split})
                n = min(counts.values())
                selection = {"counts": counts, "matched_n": n,
                             "Train_indices": list(range(n)), "Validation_indices": list(range(n)),
                             "synthetic_indices": list(range(n)), "synthetic": {}}
                for g in ("B0", "B1", "B2"):
                    manifests["B0" if g == "B0" else "B1"]["committed"][g + "/" + sid + "/" + state] = {
                        "file": "files\\pool.npz", "sha256": digest, "expected_count": counts["Train"]}
                    selection["synthetic"][g] = {"path": "UNAVAILABLE_OLD_ROOT/" + g + ".npz",
                                                  "bytes": len(sentinel), "sha256": digest}
                selections[sid + "/" + state] = selection
        with (inputs / "split.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
            writer.writeheader()
            writer.writerows(split_rows)
        norm = {"channel_order": ["fixture-channel-" + str(i) for i in range(19)],
                "means_uv": [0.0] * 19, "stds_uv": [1.0] * 19}
        write_json(inputs / "norm.json", norm)
        write_json(inputs / "aggregate.json", {"fixture": True})
        for g in ("B0", "B1"):
            write_json(inputs / (g + "_manifest.json"), manifests[g])
        (tables / "subject_state_metrics.csv").write_text(
            "subject_id,state,generator,log_psd_mae_db,sd_md_pooled_uv,nearest_real_rmse_z_median\nfixture,EO,B0,1,1,1\n", encoding="utf-8")
        (tables / "subject_alpha_effect.csv").write_text(
            "subject_id,generator,channel_mae_db\nfixture,B0,1\n", encoding="utf-8")
        for name in ("B0", "B1_B2"):
            (inputs / (name + ".py")).write_text("# Synthetic reference-source fixture: " + name + "\n", encoding="utf-8")
        config = {"schema": self.module.CONFIG_SCHEMA,
                  "processed_hdf5_dir": str(processed), "split_manifest": str(inputs / "split.csv"),
                  "normalization": str(inputs / "norm.json"), "pool_dirs": {g: str(p) for g, p in pool_dirs.items()},
                  "pool_manifests": {g: str(inputs / (g + "_manifest.json")) for g in ("B0", "B1")},
                  "original_fidelity_tables": str(tables), "original_aggregate_fidelity": str(inputs / "aggregate.json"),
                  "original_fidelity_scripts": {g: str(inputs / (g + ".py")) for g in ("B0", "B1_B2")},
                  "original_script": str(ORIGINAL), "output_dir": str(self.base / "fresh external output")}
        config_path = self.base / "configuration with spaces.json"
        write_json(config_path, config)
        self.module._configure(config_path)
        split, normalization, loaded_manifests = self.module._metadata_check()
        plan = {
            "schema": "round3_exploratory_validation_v1", "status": "SETTINGS_FROZEN_BEFORE_WAVEFORM_READS",
            "allowed_real_splits": ["Train", "Validation"], "new_training": False,
            "new_generation": False, "model_selection": False, "threads": 2,
            "window": {"fs": 128, "samples": 512, "channels": 19}, "channels": norm["channel_order"],
            "subjects": subjects, "processing_order": subjects,
            "bootstrap": original_plan_constant("bootstrap"),
            "features": original_plan_constant("features"), "comparisons": original_plan_constant("comparisons"),
            "real_file_inventory": inventories, "selections": selections,
            "illustration": {"seed": 20261017, "subject_id": subjects[0], "state": "EO",
                             "channel_index": 0, "channel": norm["channel_order"][0],
                             "window_indices": {g: 0 for g in ("Validation", "B0", "B1", "B2")}},
            "source_hashes": [{"path": "UNAVAILABLE_OLD_METADATA/" + role,
                               "sha256": self.module.sha(path)} for role, path in self.module._metadata_paths().items()
                              if role != "portable_script"],
        }
        if with_plan:
            plan_path = inputs / "private fixture plan.json"
            write_json(plan_path, plan)
            config.update(frozen_plan=str(plan_path), frozen_plan_sha256=self.module.sha(plan_path))
            write_json(config_path, config)
            self.module._configure(config_path)
        return config_path, config, plan, split, normalization, loaded_manifests

    def test_help_is_stdlib_only_from_unrelated_path_with_spaces(self):
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check-config", result.stdout)
        self.assertNotIn("numpy", self.module.__dict__)
        self.assertNotIn("np", self.module.__dict__)
        self.assertNotIn("h5py", self.module.__dict__)

    def test_missing_and_corrupt_configuration_fail(self):
        missing = self.cli("--config", self.base / "missing.json", "--check-config")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("Missing configuration", missing.stderr)
        corrupt = self.base / "corrupt.json"
        corrupt.write_text("{broken", encoding="utf-8")
        result = self.cli("--config", corrupt, "--check-config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR:", result.stderr)

    def test_frozen_mapping_validates_without_reading_waveform_bytes(self):
        config_path, _, plan, split, norm, manifests = self.fixture()
        old_bytes = self.module._CONFIG["frozen_plan"].read_bytes()
        actual_sha = self.module.sha

        def metadata_hash_only(path):
            self.assertNotIn(Path(path).suffix.lower(), (".h5", ".npz"))
            return actual_sha(path)

        with patch.object(self.module, "sha", metadata_hash_only):
            mapping = self.module._resolve_plan(plan, split, norm, manifests)
        self.assertGreater(len(mapping), 198)
        self.assertEqual(old_bytes, self.module._CONFIG["frozen_plan"].read_bytes())
        result = self.cli("--config", config_path, "--check-config")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no waveform bytes read", result.stdout)
        self.assertFalse(self.module.OUT.exists())

    def test_prepare_of_synthetic_fixture_copies_frozen_plan_without_science(self):
        _, _, _, split, norm, manifests = self.fixture()
        original_bytes = self.module._CONFIG["frozen_plan"].read_bytes()
        with patch.object(self.module, "_science_imports", side_effect=AssertionError("Unexpected scientific import")):
            self.module._prepare_portable(split, norm, manifests)
        self.assertEqual((self.module.OUT / "plan.json").read_bytes(), original_bytes)
        self.assertEqual(self.module._prepared_state()["state"], "PREPARED")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.module._prepare_portable(split, norm, manifests)
        (self.module.OUT / "participants").mkdir()
        with self.assertRaisesRegex(RuntimeError, "caches"):
            self.module._prepared_state()

    def test_modified_config_or_script_identity_cannot_reuse_prepared_output(self):
        _, _, _, split, norm, manifests = self.fixture()
        self.module._prepare_portable(split, norm, manifests)
        state = self.module.load(self.module._state_path())
        state["portable_script_sha256"] = "0" * 64
        write_json(self.module._state_path(), state)
        with self.assertRaisesRegex(RuntimeError, "script changed"):
            self.module._prepared_state()
        state["portable_script_sha256"] = self.module.sha(SCRIPT)
        state["configuration_sha256"] = "0" * 64
        write_json(self.module._state_path(), state)
        with self.assertRaisesRegex(RuntimeError, "Configuration changed"):
            self.module._prepared_state()

    def test_missing_metadata_and_modified_metadata_hash_fail(self):
        config_path, config, plan, split, norm, manifests = self.fixture()
        aggregate = Path(config["original_aggregate_fidelity"])
        aggregate.write_text('{"modified_fixture": true}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "no configured match"):
            self.module._resolve_plan(plan, split, norm, manifests)
        Path(config["normalization"]).unlink()
        result = self.cli("--config", config_path, "--check-config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing metadata", result.stderr)

    def test_modified_frozen_plan_digest_fails(self):
        config_path, config, plan, _, _, _ = self.fixture()
        plan["threads"] = 99
        write_json(Path(config["frozen_plan"]), plan)
        result = self.cli("--config", config_path, "--check-config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Frozen plan SHA-256 mismatch", result.stderr)

    def test_output_overlap_and_unknown_config_keys_fail(self):
        config_path, config, _, _, _, _ = self.fixture(with_plan=False)
        for output in (ROOT / "must-never-create", Path(config["processed_hdf5_dir"]) / "nested-output", self.base):
            with self.subTest(output=output):
                config["output_dir"] = str(output)
                write_json(config_path, config)
                with self.assertRaises(RuntimeError):
                    self.module._configure(config_path)
        config["output_dir"] = str(self.base / "new-output")
        config["typo_input"] = "wrong"
        write_json(config_path, config)
        with self.assertRaisesRegex(RuntimeError, "Unknown configuration keys"):
            self.module._configure(config_path)

    def test_manifest_paths_are_portable_and_cannot_escape(self):
        _, _, _, _, _, _ = self.fixture(with_plan=False)
        self.assertEqual(self.module._pool_path("B0", {"file": "files\\pool.npz"}),
                         self.module.POOL_DIRS["B0"] / "files/pool.npz")
        for value in ("../private.npz", "files/../../private.npz", "/absolute.npz", "D:\\absolute.npz"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.module._pool_path("B0", {"file": value})

    def test_manifest_hash_and_missing_or_changed_waveform_stat_fail(self):
        _, _, plan, _, _, manifests = self.fixture()
        key = next(iter(plan["selections"]))
        changed = copy.deepcopy(manifests)
        changed["B0"]["committed"]["B0/" + key]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            self.module._bind_wave_paths(plan, changed, {})
        pool_file = self.module.POOL_DIRS["B0"] / "files/pool.npz"
        pool_file.write_bytes(b"changed synthetic non-waveform fixture")
        with self.assertRaisesRegex(RuntimeError, "byte count changed"):
            self.module._bind_wave_paths(plan, manifests, {})
        pool_file.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing external waveform file"):
            self.module._bind_wave_paths(plan, manifests, {})

    def test_plan_cannot_change_splits_bootstraps_definitions_or_indices(self):
        _, _, original, split, norm, _ = self.fixture()
        mutations = (
            lambda p: p["allowed_real_splits"].append("Test"),
            lambda p: p["bootstrap"].update(repetitions=10),
            lambda p: p["features"].update(precision="float16"),
            lambda p: next(iter(p["selections"].values()))["Train_indices"].append(9999),
        )
        for change in mutations:
            plan = copy.deepcopy(original)
            change(plan)
            with self.assertRaises(RuntimeError):
                self.module._validate_plan(plan, split, norm)

    def test_test_split_refused_before_any_hdf5_lookup(self):
        with self.assertRaisesRegex(RuntimeError, "Forbidden split"):
            self.module.read_real("fixture", "Test", "EO", [], {}, None)

    def test_scientific_core_functions_match_archived_source_ast(self):
        old = ast.parse(ORIGINAL.read_text(encoding="utf-8-sig"))
        new = ast.parse(SCRIPT.read_text(encoding="utf-8-sig"))
        originals = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
        derivatives = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
        for name in ("corr", "signal_features", "small_features", "distance_metrics", "nearest", "read_real", "aggregate"):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(originals[name], include_attributes=False),
                                 ast.dump(derivatives[name], include_attributes=False))


if __name__ == "__main__":
    unittest.main()
