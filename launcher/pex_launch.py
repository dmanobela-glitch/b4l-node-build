#!/usr/bin/env python3
"""pex_launch — the PERMANENT, DUMB fetch-launcher (Master 2026-08-05: "build the launcher idea … let's be over this
auto-update already"). This is the reference implementation that the Windows `pex.exe` wraps (and the Linux launcher runs
directly). It is the ONE piece of code that must never go stale, so it does the LEAST possible:

  1. GET  {box}/node_version         → the box's live consensus fingerprint (the ONE source of truth).
  2. GET  {box}/download/pex-node.tar.gz → the current code bundle (consensus + join/update logic + UI).
  3. VERIFY bundle_fingerprint(bytes) == box fp  → fork-safe: never run code the box isn't running.
  4. EXTRACT to a clean run dir (atomic: fresh temp → swap), keeping the last good tree as a fallback.
  5. EXEC  python -m node.pex_join_out from that dir → the in-bundle program joins the live chain, shows the UI, AND
     auto-updates itself live when the box flips (the join loop already does this).

WHY this ends the 5-day strand: the launcher carries NO consensus code and NO update logic of its own — it FETCHES them
every launch. So a client can never be stuck on old code across a flip: reopen → it pulls the current bundle → runs it.
The only durable bytes are this tiny fetch-and-run, which almost never changes. If the box is unreachable, it falls back
to the last-good extracted tree so the app still opens offline (read-only) instead of dying.

Out-of-fp (not consensus) → shipping/improving it never forks the chain. __B4L_PEX_LAUNCHER_v1__.
"""
from __future__ import annotations
import hashlib
import io
import os
import sys
import tarfile
import tempfile
import urllib.request

BOX = os.environ.get('PEX_BOX_URL', 'https://compute.bull4life.com').rstrip('/')
HOME = os.environ.get('PEX_LAUNCH_HOME', os.path.join(os.path.expanduser('~'), '.pex'))
RUN_DIR = os.path.join(HOME, 'current')          # the extracted, running code tree
LAST_GOOD = os.path.join(HOME, 'last_good')      # fallback if the box is unreachable
UA = {'User-Agent': 'pex-launch/1'}


def _log(m):
    print(f'[pex-launch] {m}', flush=True)


def _get(url, timeout, binary=False):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read() if binary else r.read().decode('utf-8')


def _box_fp():
    """The box's live consensus fingerprint via /node_version (JSON)."""
    import json
    return json.loads(_get(f'{BOX}/node_version', 15)).get('consensus_fingerprint', '')


def _bundle_fp(data: bytes) -> str:
    """Recompute the bundle's fingerprint the SAME way the mesh does (sha over the extracted consensus tree). Mirrors
    node.pex_self_update.bundle_fingerprint so the launcher is self-contained (no import from an un-extracted tree)."""
    # extract to a scratch dir + call the tree's own consensus_fingerprint — this is the authoritative check (identical
    # to what a running node computes), so a tampered/partial bundle is caught.
    with tempfile.TemporaryDirectory() as td:
        _safe_extract(data, td)
        sys.path.insert(0, td)
        try:
            # import fresh from the extracted tree
            import importlib
            for m in [k for k in list(sys.modules) if k.startswith('node.') or k == 'node']:
                sys.modules.pop(m, None)
            cf = importlib.import_module('node.consensus_fingerprint')
            return cf.consensus_fingerprint()
        finally:
            sys.path.remove(td)


def _safe_extract(data: bytes, dest: str):
    """Extract a tar.gz, refusing any path that escapes dest (no absolute paths / .. traversal)."""
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
        base = os.path.abspath(dest)
        for m in t.getmembers():
            target = os.path.abspath(os.path.join(dest, m.name))
            if not (target == base or target.startswith(base + os.sep)):
                raise ValueError(f'unsafe path in bundle: {m.name}')
        t.extractall(dest)


def _atomic_swap(new_tree: str, dest: str):
    """Replace dest with new_tree atomically-ish: move dest→dest.old, new_tree→dest, rm dest.old."""
    old = dest + '.old'
    if os.path.exists(old):
        import shutil; shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(dest):
        os.rename(dest, old)
    os.rename(new_tree, dest)
    if os.path.exists(old):
        import shutil; shutil.rmtree(old, ignore_errors=True)


def fetch_current() -> str:
    """Fetch+verify+extract the current bundle into RUN_DIR. Returns the run dir on success. On ANY box failure, falls
    back to LAST_GOOD (so the app still opens). Raises only if there's no bundle AND no last-good tree."""
    os.makedirs(HOME, exist_ok=True)
    try:
        box_fp = _box_fp()
        if not box_fp:
            raise RuntimeError('box returned empty fingerprint')
        _log(f'box is on {box_fp[:16]} — fetching matching bundle')
        data = _get(f'{BOX}/download/pex-node.tar.gz', 60, binary=True)
        got = _bundle_fp(data)
        if got != box_fp:
            raise RuntimeError(f'bundle fp {got[:16]} != box {box_fp[:16]} (box mid-publish?) — not adopting')
        # extract to a fresh temp tree, then atomic-swap into RUN_DIR (a crash mid-write can't leave a half tree)
        staging = tempfile.mkdtemp(dir=HOME, prefix='.stage_')
        _safe_extract(data, staging)
        _atomic_swap(staging, RUN_DIR)
        # keep a known-good copy for offline fallback
        import shutil
        shutil.rmtree(LAST_GOOD, ignore_errors=True)
        shutil.copytree(RUN_DIR, LAST_GOOD)
        _log(f'✅ running code updated to {box_fp[:16]} ({len(data)} bytes) → {RUN_DIR}')
        return RUN_DIR
    except Exception as e:
        _log(f'⚠ could not fetch current bundle ({type(e).__name__}: {e})')
        if os.path.isdir(RUN_DIR):
            _log('→ using the already-extracted current tree'); return RUN_DIR
        if os.path.isdir(LAST_GOOD):
            _log('→ box unreachable; opening the last-good tree (offline/read-only)')
            import shutil; shutil.rmtree(RUN_DIR, ignore_errors=True); shutil.copytree(LAST_GOOD, RUN_DIR)
            return RUN_DIR
        raise RuntimeError('no bundle from box and no local tree — cannot start') from e


def _real_python() -> str:
    """The REAL interpreter to run the fetched source with. CRITICAL for the frozen case (engine bridge 2024): if this
    launcher is itself a frozen exe, sys.executable == pex.exe, and re-execing THAT just relaunches the unchanged
    launcher (the exact "falls out of the world" trap). So when frozen, use the bundled embeddable python that ships
    ALONGSIDE the exe (pythonw.exe next to it), NOT sys.executable. When running as plain Python, sys.executable is
    correct."""
    frozen = getattr(sys, 'frozen', False) or '__compiled__' in globals()
    if frozen:
        here = os.path.dirname(os.path.abspath(sys.executable))
        for cand in ('pythonw.exe', 'python.exe', 'pythonw', 'python3', 'python'):
            p = os.path.join(here, cand)
            if os.path.isfile(p):
                return p
        # bundled interpreter expected next to the exe; fall through to PEX_PYTHON override or sys.executable as last resort
        return os.environ.get('PEX_PYTHON') or sys.executable
    return sys.executable


def run(run_dir: str):
    """Exec the in-bundle program (node.pex_join_out.main) from the fetched tree with a REAL interpreter (never re-exec
    the frozen launcher itself). os.execve so the launcher process BECOMES the program (clean, single process)."""
    py = _real_python()
    env = dict(os.environ)
    env['PYTHONPATH'] = run_dir + os.pathsep + env.get('PYTHONPATH', '')
    _log(f'launching {py} -m node.pex_join_out from {run_dir}')
    os.execve(py, [py, '-m', 'node.pex_join_out'] + sys.argv[1:], env)


def main():
    run(fetch_current())


if __name__ == '__main__':
    main()
