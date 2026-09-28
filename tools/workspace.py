"""Read-only, standard-library workspace entry point. Never launches training."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def local(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside workspace: {name}')
    return path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(root):
    result = json.loads((root / 'workspace/state.json').read_text(encoding='utf-8'))
    if result['schema_version'] != 1:
        raise ValueError('Unsupported state schema')
    for name in result['references'].values():
        if not local(root, name).is_file():
            raise ValueError(f'Missing reference: {name}')
    if result.get('handoff') and not local(root, result['handoff']).is_file():
        raise ValueError('Missing handoff')
    return result


def verify(root, current):
    count = size = 0
    errors = []
    for seal in current['seals']:
        path = local(root, seal['manifest'])
        if digest(path) != seal['sha256']:
            raise ValueError(f'Manifest hash mismatch: {seal["manifest"]}')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if len(manifest['files']) != manifest['file_count']:
            raise ValueError('Manifest count mismatch')
        subtotal = 0
        seen = set()
        for item in manifest['files']:
            name = item['path']
            if name in seen:
                raise ValueError(f'Duplicate manifest path: {name}')
            seen.add(name)
            file = local(root, name)
            if not file.is_file():
                errors.append(f'Missing: {name}')
            elif file.stat().st_size != item['bytes'] or digest(file) != item['sha256']:
                errors.append(f'Changed: {name}')
            subtotal += item['bytes']
        if subtotal != manifest['total_bytes']:
            raise ValueError('Manifest byte count mismatch')
        count += len(seen)
        size += subtotal
    if errors:
        raise ValueError('\n'.join(errors))
    return {'verified_files': count, 'verified_bytes': size,
            'scope': [s['label'] for s in current['seals']],
            'note': 'Only indexed seals checked; this is not a new scientific audit.'}


def doctor(root):
    packages = {}
    missing = []
    for name in ('numpy', 'scipy', 'torch', 'scikit-learn', 'matplotlib', 'pyyaml'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            missing.append(name)
    ready = sys.version_info >= (3, 12) and not missing
    return {'python': sys.version, 'executable': sys.executable,
            'root': str(root), 'free_disk_gib': round(shutil.disk_usage(root).free / 2**30, 2),
            'packages': packages, 'missing': missing, 'dependencies_present': ready,
            'note': 'Metadata check only; validate imports, device, disk budget and training before a new run.'}, ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'doctor', 'verify'))
    args = parser.parse_args()
    try:
        current = state(ROOT)
        code = 0
        if args.command == 'status':
            result = current
        elif args.command == 'verify':
            result = verify(ROOT, current)
        else:
            result, ready = doctor(ROOT)
            code = 0 if ready else 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return code
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Workspace check failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
