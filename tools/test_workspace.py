"""Run with: python -m unittest discover -s tools -p 'test_workspace.py'."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import workspace


class WorkspaceTests(unittest.TestCase):
    def test_seal_success_missing_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix='crow brain ') as directory:
            root = Path(directory)
            file = root / 'result.txt'
            file.write_bytes(b'original')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'files': [{'path': 'result.txt', 'bytes': 8,
                'sha256': hashlib.sha256(b'original').hexdigest()}],
                'file_count': 1, 'total_bytes': 8}))
            current = {'seals': [{'manifest': 'manifest.json', 'label': 'fixture',
                                 'sha256': workspace.digest(manifest)}]}
            self.assertEqual(workspace.verify(root, current)['verified_files'], 1)
            file.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'Changed'):
                workspace.verify(root, current)
            file.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                workspace.verify(root, current)
            manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Manifest hash mismatch'):
                workspace.verify(root, current)

    def test_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                workspace.local(Path(directory), '../outside')
            with self.assertRaises(ValueError):
                workspace.local(Path(directory), str(Path(directory).resolve()))

    def test_live_state_references(self):
        current = workspace.state(workspace.ROOT)
        self.assertEqual(current['schema_version'], 1)
        self.assertTrue(current['seals'])


if __name__ == '__main__':
    unittest.main()
