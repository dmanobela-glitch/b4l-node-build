#!/usr/bin/env python3
"""pex_launch — the PERMANENT, DUMB fetch-launcher (Master 2026-08-05: "build the launcher idea … let's be over this
auto-update already"). Reference implementation the Windows `pex.exe` / Linux `pex` wraps. The ONE piece that must never
go stale, so it does the LEAST possible and delegates ALL real work to a fetched, current bundle:

  1. GET  {box}/node_version         → the box's live consensus fingerprint (the ONE source of truth).
  2. GET  {box}/download/pex-node.tar.gz → the current code bundle (consensus + join/update logic + UI).
  3. EXTRACT to a staging dir, then VERIFY with the SIBLING real python: the extracted tree's own
     consensus_fingerprint() must == box fp (fork-safe). The verify runs in the sibling interpreter (full stdlib),
     NOT this launcher's frozen interpreter (which Nuitka strips down — no `secrets` etc.; engine bridge 2033).
  4. ATOMIC swap staging → run dir (a crash mid-write can't leave a half tree); keep last_good for offline open.
  5. EXEC  {sibling python} -m node.pex_join_out → the in-bundle program joins the live chain, shows the UI, and
     auto-updates itself live when the box flips.

WHY this ends the strand: the launcher carries NO consensus code and NO update logic of its own — it FETCHES them every
launch. Reopen → current. Out-of-fp → shipping/improving it never forks the chain. __B4L_PEX_LAUNCHER_v2__.
"""
from __future__ import annotations
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

BOX = os.environ.get('PEX_BOX_URL', 'https://compute.bull4life.com').rstrip('/')
HOME = os.environ.get('PEX_LAUNCH_HOME', os.path.join(os.path.expanduser('~'), '.pex'))
RUN_DIR = os.path.join(HOME, 'current')
LAST_GOOD = os.path.join(HOME, 'last_good')
UA = {'User-Agent': 'pex-launch/2'}
_IS_WIN = os.name == 'nt'


def _log(m):
    """CRASH-PROOF logging (engine bridge 2033): a windowed/frozen build can have sys.stdout==None, and Windows cp1252
    stdout raises UnicodeEncodeError on non-ASCII → a bare print() would kill the launcher SILENTLY. Never let logging
    take the app down."""
    msg = f'[pex-launch] {m}'
    for sink in (sys.stdout, sys.stderr):
        try:
            if sink is None:
                continue
            try:
                sink.write(msg + '\n')
            except UnicodeEncodeError:
                sink.write(msg.encode('ascii', 'replace').decode('ascii') + '\n')
            sink.flush()
            return
        except Exception:
            continue


def _get(url, timeout, binary=False):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read() if binary else r.read().decode('utf-8')


def _box_fp():
    import json
    return json.loads(_get(f'{BOX}/node_version', 15)).get('consensus_fingerprint', '')


def _find_python() -> str:
    """A WORKING real interpreter to verify + run the fetched node source. CRITICAL (engine 2033): if this launcher is a
    frozen exe, sys.executable == the launcher itself; re-execing THAT re-runs the launcher (the "falls out of the world"
    trap). So when frozen, find the embeddable python shipped in the package — and because packaging can misplace it
    (e.g. a top-level python3 whose rpath doesn't resolve while bin/python3 does), TEST each candidate and pick the first
    that actually runs. When running as plain Python, sys.executable is correct."""
    frozen = getattr(sys, 'frozen', False) or '__compiled__' in globals()
    if not frozen:
        return sys.executable
    here = os.path.dirname(os.path.abspath(sys.executable))
    names = (['pythonw.exe', 'python.exe'] if _IS_WIN else ['python3', 'python'])
    cands = []
    for d in (here, os.path.join(here, 'bin')):
        for n in names:
            cands.append(os.path.join(d, n))
    env_py = os.environ.get('PEX_PYTHON')
    if env_py:
        cands.insert(0, env_py)
    for p in cands:
        if os.path.isfile(p):
            try:
                r = subprocess.run([p, '-c', 'import secrets,hashlib,ssl'], capture_output=True, timeout=30)
                if r.returncode == 0:
                    return p
            except Exception:
                continue
    # last resort: sys.executable (may be the frozen launcher — verify/run will fail loudly rather than silently strand)
    _log('WARNING: no working sibling python found in package; falling back to sys.executable')
    return sys.executable


def _tree_fp(tree_dir: str, py: str) -> str:
    """Compute the extracted tree's consensus_fingerprint using the SIBLING python (full stdlib). Subprocess so we never
    import node code into the (possibly stripped) launcher interpreter."""
    code = 'from node.consensus_fingerprint import consensus_fingerprint as f; print(f())'
    env = dict(os.environ)
    env['PYTHONPATH'] = tree_dir + os.pathsep + env.get('PYTHONPATH', '')
    r = subprocess.run([py, '-c', code], cwd=tree_dir, env=env, capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        raise RuntimeError(f'fp compute failed in sibling python: {(r.stderr or "").strip()[:300]}')
    return (r.stdout or '').strip()


def _safe_extract(data: bytes, dest: str):
    """Extract a tar.gz, refusing any path that escapes dest (no absolute paths / .. traversal)."""
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
        base = os.path.abspath(dest)
        for m in t.getmembers():
            target = os.path.abspath(os.path.join(dest, m.name))
            if not (target == base or target.startswith(base + os.sep)):
                raise ValueError(f'unsafe path in bundle: {m.name}')
        try:
            t.extractall(dest, filter='data')            # 3.12+ safe-extract filter (also silences the 3.14 warning)
        except TypeError:
            t.extractall(dest)


def _atomic_swap(new_tree: str, dest: str):
    import shutil
    old = dest + '.old'
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(dest):
        os.rename(dest, old)
    os.rename(new_tree, dest)
    shutil.rmtree(old, ignore_errors=True)


def fetch_current(py: str) -> str:
    """Fetch → verify (via sibling python) → atomic-extract into RUN_DIR. On ANY box failure, fall back to a local tree so
    the app still opens. Raises only if there's no bundle AND no local tree."""
    import shutil
    os.makedirs(HOME, exist_ok=True)
    try:
        box_fp = _box_fp()
        if not box_fp:
            raise RuntimeError('box returned empty fingerprint')
        _log(f'box is on {box_fp[:16]} - fetching matching bundle')
        data = _get(f'{BOX}/download/pex-node.tar.gz', 60, binary=True)
        staging = tempfile.mkdtemp(dir=HOME, prefix='.stage_')
        _safe_extract(data, staging)
        got = _tree_fp(staging, py)                       # verify with the REAL interpreter (full stdlib)
        if got != box_fp:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f'bundle fp {got[:16]} != box {box_fp[:16]} (box mid-publish?) - not adopting')
        _atomic_swap(staging, RUN_DIR)
        shutil.rmtree(LAST_GOOD, ignore_errors=True)
        shutil.copytree(RUN_DIR, LAST_GOOD)
        _log(f'OK running code updated to {box_fp[:16]} ({len(data)} bytes) -> {RUN_DIR}')
        return RUN_DIR
    except Exception as e:
        _log(f'could not fetch current bundle ({type(e).__name__}: {e})')
        if os.path.isdir(RUN_DIR):
            _log('-> using the already-extracted current tree'); return RUN_DIR
        if os.path.isdir(LAST_GOOD):
            _log('-> box unreachable; opening the last-good tree (offline/read-only)')
            shutil.rmtree(RUN_DIR, ignore_errors=True); shutil.copytree(LAST_GOOD, RUN_DIR)
            return RUN_DIR
        raise RuntimeError('no bundle from box and no local tree - cannot start') from e


def run(run_dir: str, py: str):
    """Exec the in-bundle program with the SIBLING interpreter (never re-exec the frozen launcher itself)."""
    env = dict(os.environ)
    env['PYTHONPATH'] = run_dir + os.pathsep + env.get('PYTHONPATH', '')
    _log(f'launching {py} -m node.pex_join_out from {run_dir}')
    if _IS_WIN:
        # os.execv on Windows is flaky for GUI processes; spawn + exit so the parent doesn't linger.
        os.chdir(run_dir)
        p = subprocess.Popen([py, '-m', 'node.pex_join_out'] + sys.argv[1:], env=env)
        sys.exit(p.wait())
    os.execve(py, [py, '-m', 'node.pex_join_out'] + sys.argv[1:], env)


def main():
    py = _find_python()
    run(fetch_current(py), py)


if __name__ == '__main__':
    main()
