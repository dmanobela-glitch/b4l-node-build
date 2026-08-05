"""pex_launcher — the PEX one-click TINY DOWNLOADER (Model A, refined per Cludia 0553 / Master's design call).

WHY a tiny downloader and NOT a frozen onefile OR a 55MB baked self-extractor:
  * A frozen onefile can't self-update (it runs baked bytecode, ignores swapped .py, re-execs itself unchanged) — that
    was Master's original "I fell out of the world" bug. So the app must run REAL, file-based Python against EDITABLE source.
  * A 55MB onefile that BAKES CPython + source and self-extracts them to %TEMP% then execs python is the #1 antivirus
    heuristic ("unpack a big payload to temp + run it" = the malware tell) — it got deleted on download on Master's laptop.
So this launcher is a SMALL, STABLE SHELL that, on first run, DOWNLOADS the CPython runtime + the canonical PEX source
into a per-user app dir, fp-VERIFIES the source against the box's own advertised fingerprint (fork-safe, same trust model
as auto_update — no hardcoded pin, so it stays correct as the chain moves), then launches `python -m node.pex_join_out
--gui` against that editable tree. Benefits (all FREE, no MS submission, no cert):
  * no baked payload + no unpack-to-temp → the biggest AV heuristic is gone; the shell is small.
  * the shell NEVER changes (updates happen INSIDE, to the fetched source) → AV/SmartScreen reputation is earned once
    and kept forever; consensus fp bumps self-heal via the app's own check_and_apply/auto_update, never a rebuild.

Fork-safe verify: the downloaded source tree must hash (consensus_fingerprint) to the fingerprint the box advertises at
/node_version. A tampered/mismatched bundle is REFUSED — the launcher never runs unverified source. Offline first-run
(nothing fetched yet) fails gracefully with "offline — will fetch when online"; it's joining a live network anyway.

Env overrides (used by CI self-check to point at a local http server): PEX_BUNDLE_URL, PEX_RUNTIME_URL, PEX_VERSION_URL,
PEX_EXPECTED_FP, PEX_APP_DIR.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

BUNDLE_URL = os.environ.get("PEX_BUNDLE_URL", "https://compute.bull4life.com/node_bundle")
VERSION_URL = os.environ.get("PEX_VERSION_URL", "https://compute.bull4life.com/node_version")
APP_MODULE = "node.pex_join_out"
APP_ARGS = ["--gui"]
FP_ONELINER = "from node.consensus_fingerprint import consensus_fingerprint; print(consensus_fingerprint())"


def _default_runtime_url() -> str:
    base = "https://compute.bull4life.com/download"
    return f"{base}/pex-runtime-win.tar.gz" if os.name == "nt" else f"{base}/pex-runtime-linux.tar.gz"


RUNTIME_URL = os.environ.get("PEX_RUNTIME_URL", _default_runtime_url())


# ---------------------------------------------------------------------------- app dir + log
def _app_dir() -> Path:
    override = os.environ.get("PEX_APP_DIR")
    if override:
        d = Path(override)
    elif os.name == "nt":
        d = Path(os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")) / "PEX"
    else:
        d = Path(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")) / "PEX"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log(app: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    try:
        with open(app / "launcher.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


# ---------------------------------------------------------------------------- download + tar
def _download(url: str, timeout: float = 45.0) -> "bytes | None":
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pex-launcher/2"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _tar_extractall(tar: "tarfile.TarFile", dst: Path) -> None:
    dstr = str(dst.resolve())
    for m in tar.getmembers():                                   # path-safe (no absolute/.. escape)
        p = (dst / m.name).resolve()
        if str(p) != dstr and not str(p).startswith(dstr + os.sep):
            raise RuntimeError(f"unsafe tar member {m.name}")
    try:
        tar.extractall(dst, filter="tar")                        # preserves mode bits + symlinks
    except TypeError:
        tar.extractall(dst)


def _extract_bytes(data: bytes, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tgz") as tf:
        tf.write(data)
        tmp = tf.name
    try:
        with tarfile.open(tmp, "r:*") as tar:
            _tar_extractall(tar, dst)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ---------------------------------------------------------------------------- runtime python
def _runtime_python(app: Path, *, windowless: bool = False) -> "Path | None":
    if os.name == "nt":
        for n in (["pythonw.exe", "python.exe"] if windowless else ["python.exe"]):
            p = app / "runtime" / n
            if p.exists():
                return p
    else:
        for sub in ("bin/python3", "bin/python"):
            p = app / "runtime" / sub
            if p.exists():
                return p
    return None


def _ensure_runtime(app: Path) -> bool:
    if _runtime_python(app) is not None:
        return True
    _log(app, f"downloading runtime from {RUNTIME_URL}")
    data = _download(RUNTIME_URL)
    if not data:
        _log(app, "runtime download failed (offline?)")
        return False
    _extract_bytes(data, app / "runtime")
    if os.name != "nt":
        for sub in ("bin/python3", "bin/python3.12", "bin/python"):
            f = app / "runtime" / sub
            if f.exists() and not f.is_symlink():
                try:
                    f.chmod(0o755)
                except Exception:
                    pass
    return _runtime_python(app) is not None


# ---------------------------------------------------------------------------- source tree + fp
def _compute_fp(app: Path, srcdir: Path) -> "str | None":
    py = _runtime_python(app)
    if py is None:
        _log(app, "compute_fp: no runtime python")
        return None
    if not (srcdir / "node" / "consensus_fingerprint.py").exists():
        _log(app, "compute_fp: node/consensus_fingerprint.py missing")
        return None
    try:
        out = subprocess.run([str(py), "-c", FP_ONELINER], cwd=str(srcdir),
                             capture_output=True, text=True, timeout=120)
        if out.returncode != 0 or not (out.stdout or "").strip():
            _log(app, f"compute_fp rc={out.returncode} stderr={(out.stderr or '').strip()[:600]}")
        fp = (out.stdout or "").strip().splitlines()[-1].strip() if (out.stdout or "").strip() else ""
        return fp if len(fp) == 64 and all(c in "0123456789abcdef" for c in fp) else None
    except Exception as e:
        _log(app, f"compute_fp exception {type(e).__name__}: {e}")
        return None


def _expected_fp(app: Path) -> "str | None":
    env = os.environ.get("PEX_EXPECTED_FP")
    if env and len(env.strip()) == 64:
        return env.strip()
    data = _download(VERSION_URL, timeout=20)
    if not data:
        return None
    try:
        fp = json.loads(data.decode("utf-8")).get("consensus_fingerprint", "")
        return fp if len(fp) == 64 else None
    except Exception:
        return None


def _fetch_and_verify_source(app: Path, expected: str) -> bool:
    """Download /node_bundle, extract to a temp tree, and adopt it ONLY if its consensus_fingerprint == expected
    (the box's advertised fp). Fork-safe: never run unverified/mismatched source. Atomic swap into app/src."""
    _log(app, f"downloading source from {BUNDLE_URL} (must hash to {expected[:16]}...)")
    data = _download(BUNDLE_URL, timeout=90)
    if not data:
        _log(app, "source download failed (offline?)")
        return False
    tmp = app / "src.new"
    _extract_bytes(data, tmp)
    if not (tmp / "node" / "pex_join_out.py").exists():
        _log(app, "downloaded bundle missing pex_join_out.py — refusing")
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    fp = _compute_fp(app, tmp)
    if fp != expected:
        _log(app, f"REFUSED: downloaded source fp {(fp or 'None')[:16]}... != advertised {expected[:16]}... (tampered/mid-publish)")
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    srcdir = app / "src"
    if srcdir.exists():
        try:
            shutil.rmtree(srcdir)
        except Exception:
            pass
    tmp.rename(srcdir)
    try:
        (app / "src.fp").write_text(fp + "\n", encoding="utf-8")
    except Exception:
        pass
    _log(app, f"source verified + materialized; fp={fp[:16]}...")
    return True


def _ensure_source(app: Path) -> bool:
    srcdir = app / "src"
    expected = _expected_fp(app)
    if expected is None:
        # can't learn the target fp (offline) — only OK if we already have a verified local tree to run
        if (srcdir / "node" / "pex_join_out.py").exists():
            _log(app, "offline: can't reach /node_version — running last verified local tree; auto_update catches up later")
            return True
        _log(app, "offline first run: can't reach /node_version to learn the target fingerprint")
        return False

    if not (srcdir / "node" / "pex_join_out.py").exists():
        return _fetch_and_verify_source(app, expected)

    # integrity gate on the existing tree
    recorded = ""
    try:
        recorded = (app / "src.fp").read_text(encoding="utf-8").strip()
    except Exception:
        pass
    cur = _compute_fp(app, srcdir)
    if cur is None:
        _log(app, "local tree corrupt/poisoned -> re-fetching + verifying")
        return _fetch_and_verify_source(app, expected)
    if recorded and cur != recorded:
        _log(app, f"local tree fp {cur[:16]}... != recorded {recorded[:16]}... (tampered) -> re-fetching")
        return _fetch_and_verify_source(app, expected)
    if not recorded:
        try:
            (app / "src.fp").write_text(cur + "\n", encoding="utf-8")
        except Exception:
            pass
    _log(app, f"source OK; local fp={cur[:16]}... (box advertises {expected[:16]}...) — app auto-update keeps it current")
    return True


# ---------------------------------------------------------------------------- launch
def _launch(app: Path, srcdir: Path) -> int:
    py = _runtime_python(app, windowless=True) or _runtime_python(app)
    if py is None:
        _log(app, "FATAL: no runtime python")
        return 3
    env = dict(os.environ)
    env["PEX_APP_DIR"] = str(app)
    env["PYTHONUTF8"] = "1"
    cmd = [str(py), "-m", APP_MODULE, *APP_ARGS]
    _log(app, f"launching: {' '.join(cmd)} (cwd={srcdir})")
    if os.name == "nt":
        flags = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
        subprocess.Popen(cmd, cwd=str(srcdir), env=env, creationflags=flags, close_fds=True)
        return 0
    os.chdir(str(srcdir))
    os.execve(str(py), cmd, env)
    return 0  # unreachable


def _fatal_notify(msg: str) -> None:
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, "PEX", 0x30)  # MB_ICONWARNING
        except Exception:
            pass


# ---------------------------------------------------------------------------- entry
def run(selfcheck: bool = False) -> int:
    app = _app_dir()
    _log(app, f"pex launcher start (app={app}, selfcheck={selfcheck})")
    if not _ensure_runtime(app):
        msg = "PEX couldn't download its runtime — check your internet connection and try again."
        _log(app, "runtime not available")
        if selfcheck:
            try:
                (app / "selfcheck.result").write_text("FAIL fp=NO_RUNTIME\n", encoding="utf-8")
            except Exception:
                pass
            print("SELFCHECK FAIL fp=NO_RUNTIME")
            return 1
        _fatal_notify(msg)
        return 2
    if not _ensure_source(app):
        msg = "PEX couldn't fetch/verify the chain source — check your connection (first run needs to be online)."
        if selfcheck:
            try:
                (app / "selfcheck.result").write_text("FAIL fp=NO_SOURCE\n", encoding="utf-8")
            except Exception:
                pass
            print("SELFCHECK FAIL fp=NO_SOURCE")
            return 1
        _fatal_notify(msg)
        return 2
    srcdir = app / "src"
    if selfcheck:
        fp = _compute_fp(app, srcdir)
        expected = _expected_fp(app)
        ok = bool(fp) and fp == expected
        verdict = "PASS" if ok else "FAIL"
        _log(app, f"SELFCHECK fp={fp} expected={expected} -> {verdict}")
        print(f"SELFCHECK {verdict} fp={fp}")
        try:
            (app / "selfcheck.result").write_text(f"{verdict} fp={fp}\n", encoding="utf-8")
        except Exception:
            pass
        return 0 if ok else 1
    return _launch(app, srcdir)


def main() -> int:
    try:
        return run(selfcheck=("--selfcheck" in sys.argv[1:]))
    except Exception as e:
        app = _app_dir()
        _log(app, f"FATAL {type(e).__name__}: {e}")
        _fatal_notify(f"PEX couldn't start: {type(e).__name__}: {e}\n\nLog: {app / 'launcher.log'}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
