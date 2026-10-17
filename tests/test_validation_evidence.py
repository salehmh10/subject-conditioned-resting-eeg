"""Offline regression checks against copies of packaged aggregates, never EEG."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / 'tools/verify_validation_evidence.py'
spec = importlib.util.spec_from_file_location('validation_evidence_verifier', VERIFIER)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
BASE = Path('evidence/validation_reference')
INVENTORY = BASE / 'publication_inventory.json'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def record(root, relative):
    path = root / relative
    return {'path': str(relative).replace('\\', '/'), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


class SavedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='validation evidence with spaces ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'fresh checkout'
        self.root.mkdir()
        inventory = load(ROOT / INVENTORY)
        paths = {entry['path'] for entry in inventory['files']} | {INVENTORY.as_posix(), 'tools/verify_validation_evidence.py'}
        paths.add(verifier.PORTABILITY)
        paths.update(entry['path'] for entry in load(ROOT / verifier.PORTABILITY)['files'])
        # Include only metadata references to packaged files, never external paths.
        metadata = load(ROOT / BASE / 'public_metadata.json')
        paths.update(r['publication_path'] for r in metadata['source_references'] if r['publication_path'])
        for relative in paths:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)

    def rehash(self, relative):
        inventory = load(self.root / INVENTORY)
        for entry in inventory['files']:
            if entry['path'] == str(relative).replace('\\', '/'):
                entry.update(record(self.root, relative))
                break
        else:
            self.fail(f'No inventory entry to rehash: {relative}')
        write(self.root / INVENTORY, inventory)

    def change_summary(self, mutate):
        path = self.root / BASE / 'summary.json'
        summary = load(path)
        mutate(summary)
        write(path, summary)
        self.rehash(BASE / 'summary.json')
        completion_path = self.root / BASE / 'completion.json'
        completion = load(completion_path)
        completion['summary_sha256'] = record(self.root, BASE / 'summary.json')['sha256']
        write(completion_path, completion)
        self.rehash(BASE / 'completion.json')

    def test_packaged_evidence_passes(self):
        result = verifier.verify(self.root)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['reported_correspondence_checks'], 105)

    def test_cli_unrelated_working_directory_path_with_spaces_without_site_packages(self):
        run = subprocess.run([sys.executable, '-B', '-S', str(self.root / 'tools/verify_validation_evidence.py')],
                             cwd=Path(self.temporary.name), capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['status'], 'PASS')

    def test_missing_input_fails(self):
        (self.root / BASE / 'summary.json').unlink()
        with self.assertRaisesRegex(verifier.VerificationError, 'Missing packaged input'):
            verifier.verify(self.root)

    def test_corrupt_json_fails_after_hash_update(self):
        (self.root / BASE / 'summary.json').write_text('{invalid JSON', encoding='utf-8')
        self.rehash(BASE / 'summary.json')
        with self.assertRaisesRegex(verifier.VerificationError, 'Invalid JSON'):
            verifier.verify(self.root)

    def test_changed_summary_fails_semantics_even_with_updated_hashes(self):
        def mutate(summary):
            summary['fidelity']['full']['versus_Validation']['log_psd_mae_db']['B2']['mean'] += 0.1
        self.change_summary(mutate)
        with self.assertRaisesRegex(verifier.VerificationError, 'Arithmetic/correspondence mismatch'):
            verifier.verify(self.root)

    def test_bad_percentage_fails_even_with_updated_hashes(self):
        self.change_summary(lambda s: s['fingerprint']['full']['Validation_EO_B1'].__setitem__('top1_percent', 60.0))
        with self.assertRaisesRegex(verifier.VerificationError, 'fingerprint/full/Validation_EO_B1/top1'):
            verifier.verify(self.root)

    def test_interval_order_fails_even_with_updated_hashes(self):
        self.change_summary(lambda s: s['nearest']['summaries']['Validation']['ci95_mean'].reverse())
        with self.assertRaisesRegex(verifier.VerificationError, 'Interval ordering'):
            verifier.verify(self.root)

    def test_nonfinite_value_fails(self):
        self.change_summary(lambda s: s['nearest']['summaries']['Validation'].__setitem__('mean', float('nan')))
        with self.assertRaisesRegex(verifier.VerificationError, 'Nonfinite JSON'):
            verifier.verify(self.root)

    def test_missing_schema_branch_fails(self):
        self.change_summary(lambda s: s['fidelity'].pop('matched'))
        with self.assertRaisesRegex(verifier.VerificationError, 'Invalid schema: fidelity modes'):
            verifier.verify(self.root)

    def test_original_utility_drift_fails_even_with_updated_hash(self):
        path = self.root / 'results/utility_data.json'
        utility = load(path)
        utility['summary'][0]['test_mean_percent'] += 0.1
        write(path, utility)
        self.rehash('results/utility_data.json')
        with self.assertRaisesRegex(verifier.VerificationError, 'Original utility changed'):
            verifier.verify(self.root)

    def test_path_escape_fails(self):
        inventory = load(self.root / INVENTORY)
        inventory['files'][0]['path'] = '../outside.json'
        write(self.root / INVENTORY, inventory)
        with self.assertRaisesRegex(verifier.VerificationError, 'Unsafe publication path'):
            verifier.verify(self.root)

    def test_unmodified_hash_detects_tampering(self):
        path = self.root / BASE / 'summary.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(verifier.VerificationError, 'Size mismatch'):
            verifier.verify(self.root)


if __name__ == '__main__':
    unittest.main()
