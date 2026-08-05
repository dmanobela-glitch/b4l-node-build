"""pex_entry — thin frozen-exe entry that makes Cludia's launcher/pex_launch.py bullet-proof on Windows, WITHOUT
changing her launcher (she leads PEX; this is the build wrapper's job).

Two Windows-frozen hazards this guards, both of which would kill the app SILENTLY (the exact "app won't open" class we're
fixing):
  1. `pex_launch._log` prints an em-dash + ✅/⚠. Under Windows' default cp1252 stdout that raises UnicodeEncodeError.
     → force UTF-8 (PYTHONUTF8 + reconfigure the streams).
  2. A `--windows-console-mode=disable` (GUI-subsystem) exe has NO console, so `sys.stdout`/`sys.stderr` can be None →
     `print()` raises AttributeError. → give them a real UTF-8 sink so every `print` is safe.
Then hand off to her launcher unchanged.
"""
from __future__ import annotations
import io
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Ensure stdout/stderr exist and are UTF-8 (frozen windowless → None; console → cp1252).
for _name in ("stdout", "stderr"):
    _s = getattr(sys, _name, None)
    if _s is None:
        try:
            setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))
        except Exception:
            pass
    else:
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")   # py3.7+; no-op-safe
        except Exception:
            try:
                setattr(sys, _name, io.TextIOWrapper(_s.buffer, encoding="utf-8", errors="replace"))
            except Exception:
                pass

# Bug 3 (Cludia 0564): a Nuitka --onefile sets sys.executable to its /tmp/onefile_* EXTRACTION dir, NOT the install dir,
# so _find_python() would search the temp dir and miss pex/bin/python3. But sys.argv[0] is the launched exe path
# (pex/pex[.exe]) = the REAL install dir. We know it authoritatively here at bootstrap → point the launcher straight at
# the sibling interpreter via PEX_PYTHON (her _find_python prefers it). setdefault so a user override still wins.
try:
    _install = os.path.dirname(os.path.realpath(sys.argv[0]))
    os.environ.setdefault("PEX_INSTALL_DIR", _install)
    for _cand in ("pythonw.exe", "python.exe", os.path.join("bin", "python3"), "python3"):
        _p = os.path.join(_install, _cand)
        if os.path.isfile(_p):
            os.environ.setdefault("PEX_PYTHON", _p)
            break
except Exception:
    pass

import pex_launch  # noqa: E402  (her launcher, unchanged)

if __name__ == "__main__":
    pex_launch.main()
