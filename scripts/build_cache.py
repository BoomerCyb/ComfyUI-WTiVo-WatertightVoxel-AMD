# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep native build caches separate when the toolchain or inputs change."""
import hashlib
import json
from pathlib import Path


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def tree_identity(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError('Build input directory is missing: ' + str(path))
    digest = hashlib.sha256()
    for item in sorted(path.rglob('*')):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode('utf-8'))
            digest.update(file_identity(item)['sha256'].encode('ascii'))
    return {'path': str(path), 'sha256': digest.hexdigest()}


def cache_directory(parent, identity):
    # Never reuse the legacy cache: it has no reliable toolchain provenance.
    encoded = json.dumps(identity, sort_keys=True).encode('utf-8')
    key = hashlib.sha256(encoded).hexdigest()[:24]
    directory = Path(parent) / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory
