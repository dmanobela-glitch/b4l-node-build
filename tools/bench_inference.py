#!/usr/bin/env python3
"""bench_inference.py — 7B-army Slice 6 benchmark harness (pure stdlib).

A tiny, dependency-free harness to batch-run a llama.cpp-style CLI model
(any binary that accepts llama-cli flags: -m, -p, -n, --temp, -st, ...)
against a fixed prompt set, score the outputs, and summarize per-model
stats. No numpy, no torch, no requests — only the Python standard library.

Public API:
    load_prompts(path=None)      -> list[dict]  (embedded 8-prompt set by default)
    run_model(prompt, model_bin, model_path, max_tokens=128, temp=0.0,
              timeout_s=120)     -> dict
    score(expect, actual, kind)  -> dict
    summarize(results)           -> dict

Running ``python3 bench_inference.py`` executes a self-test that fakes a
model binary with tiny shell scripts (no real model required).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

# --------------------------------------------------------------------------
# Prompt set
# --------------------------------------------------------------------------

#: Kind values: 'code' (write code), 'nl' (natural-language answer),
#: 'classify' (label a line), 'complete' (finish a code snippet).
DEFAULT_PROMPTS: list[dict] = [
    {
        "id": "fib",
        "kind": "code",
        "task": "write the nth fibonacci number",
        "prompt": "write a function that returns the nth fibonacci number",
        "expect": "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)",
    },
    {
        "id": "traceback",
        "kind": "nl",
        "task": "explain traceback in one sentence",
        "prompt": (
            "explain this traceback in one sentence: "
            'Traceback (most recent call last): File "x.py", line 3, in <module> '
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        ),
        "expect": "the code tried to add a number and a string",
    },
    {
        "id": "logclass",
        "kind": "classify",
        "task": "classify log line",
        "prompt": (
            "classify this log line as error/warn/info: "
            '"2024-05-01 12:00:01 ERROR connection refused to 10.0.0.5:8080"'
        ),
        "expect": "error",
    },
    {
        "id": "clamp",
        "kind": "complete",
        "task": "complete the function",
        "prompt": "complete the function: def clamp(x, lo, hi):",
        "expect": "return max(lo, min(hi, x))",
    },
    {
        "id": "bubble",
        "kind": "code",
        "task": "write bubble sort",
        "prompt": "write a function that sorts a list using bubble sort",
        "expect": "def bubble_sort(xs):\n    for i in range(len(xs)):\n        for j in range(len(xs) - 1 - i):\n            if xs[j] > xs[j + 1]:\n                xs[j], xs[j + 1] = xs[j + 1], xs[j]\n    return xs",
    },
    {
        "id": "regex",
        "kind": "code",
        "task": "write an email regex",
        "prompt": "write a python regex that matches an email address",
        "expect": "import re\nEMAIL = re.compile(r\"[^@\\s]+@[^@\\s]+\\.[^@\\s]+\")",
    },
    {
        "id": "retry",
        "kind": "nl",
        "task": "suggest retry strategy",
        "prompt": "the API call failed with a 503. suggest a retry strategy in one sentence.",
        "expect": "retry with exponential backoff and jitter",
    },
    {
        "id": "sum_list",
        "kind": "complete",
        "task": "complete sum_list",
        "prompt": "complete the function: def sum_list(xs): return",
        "expect": "return sum(xs)",
    },
]


def load_prompts(path: str | None = None) -> list[dict]:
    """Load prompt dicts.

    - path=None      -> return a fresh copy of the embedded DEFAULT_PROMPTS.
    - path=<file>    -> read a JSON array or JSON-lines file; each record
                        must carry at least 'id', 'prompt', 'kind' (task
                        optional), kind in {'code','nl','classify','complete'}.

    Raises FileNotFoundError / ValueError on bad input — never silently
    returns a partial set.
    """
    if path is None:
        return [dict(p) for p in DEFAULT_PROMPTS]

    if not os.path.isfile(path):
        raise FileNotFoundError(f"prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()

    import json

    if raw.startswith("["):
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError(f"{path}: expected a JSON array")
    else:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]

    out: list[dict] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"{path}: entry {i} is not an object")
        missing = [k for k in ("id", "prompt", "kind") if k not in it]
        if missing:
            raise ValueError(f"{path}: entry {i} missing keys {missing}")
        if it["kind"] not in ("code", "nl", "classify", "complete"):
            raise ValueError(
                f"{path}: entry {i} has invalid kind {it['kind']!r} "
                "(expected code|nl|classify|complete)"
            )
        out.append(
            {
                "id": str(it["id"]),
                "kind": it["kind"],
                "prompt": str(it["prompt"]),
                "task": str(it.get("task", "")),
                "expect": str(it.get("expect", "")),
            }
        )
    return out


# --------------------------------------------------------------------------
# Model runner
# --------------------------------------------------------------------------

#: llama-cli flag template. Module-level constant so a worker can tune a
#: single knob (e.g. thread count) without forking the function.
#: -st = --single-turn (one-shot non-interactive; the Slice-1/2b anti-hang fix,
#: documented as "will not be interactive if first turn is predefined with
#: --prompt"). Kept so the harness invocation mirrors the production worker
#: byte-for-byte. --no-display-prompt = stdout is generated text only.
BASE_ARGS: list[str] = ["-t", "2", "-c", "2048", "-st", "--no-display-prompt"]


def run_model(
    prompt: str,
    model_bin: str,
    model_path: str,
    max_tokens: int = 128,
    temp: float = 0.0,
    timeout_s: int = 120,
) -> dict:
    """Run the model binary once and return a result dict.

    The command is built as a LIST and passed to subprocess.run — never a
    shell string, so prompts cannot inject shell metacharacters.

    Returns:
        {'rc', 'stdout', 'stderr', 'elapsed_s', 'timed_out'}

      - rc        : process return code, or 124 if the timeout fired.
      - timed_out : True iff subprocess.TimeoutExpired was raised.
      - elapsed_s : wall time of the run.

    NOTE on timeout escalation: subprocess.run cannot send SIGKILL after its
    timeout fires (Python 3.13 added kill_after, but a SIGTERM-ignoring
    llama-cli can still linger). The REAL worker therefore wraps this call
    in shell ``timeout -k 30 <secs>`` which escalates to SIGKILL; this
    harness deliberately reports the HONEST rc (124) and timed_out=True so
    bench numbers are not silently corrupted by killed-but-unreported runs.
    """
    cmd: list[str] = [
        model_bin,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--temp",
        str(temp),
        *BASE_ARGS,
    ]

    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            start_new_session=True,  # don't leak our process group
        )
        rc = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    elapsed = time.monotonic() - t0

    return {
        "rc": rc,
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_s": round(elapsed, 4),
        "timed_out": timed_out,
    }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Strip an outer ```python ... ``` (or bare ``` ... ```) markdown fence."""
    t = text.strip()
    lines = t.splitlines()
    if not lines or not lines[0].startswith("```"):
        return t
    for i in range(1, len(lines)):
        if lines[i].strip().startswith("```"):
            return "\n".join(lines[1:i]).strip()
    # no closing fence — strip the opener only
    return "\n".join(lines[1:]).strip()


def _norm(text: str) -> str:
    """Whitespace normalization for norm_match comparisons."""
    return " ".join(text.split())


def score(expect: str, actual: str, kind: str) -> dict:
    """Score one model output against an expectation. NEVER raises.

    - kind 'code'      : strip fences, try compile(); report compiles,
                         exact_match, norm_match (whitespace-normalized).
                         compile() failures are captured, not raised.
    - kinds 'nl'/'classify'/'complete': exact_match + norm_match + length
                         sanity (actual non-empty and < 2000 chars).

    Returns dict with keys: kind, exact_match, norm_match, length_ok,
    actual_len, plus compiles/compile_err for kind == 'code'.
    """
    if not isinstance(actual, str):
        actual = ""
    if not isinstance(expect, str):
        expect = ""

    actual_clean = _strip_fences(actual) if kind == "code" else actual
    expect_clean = _strip_fences(expect) if kind == "code" else expect

    exact_match = expect_clean == actual_clean
    norm_match = _norm(expect_clean) == _norm(actual_clean)
    length_ok = bool(actual.strip()) and len(actual) < 2000

    result: dict = {
        "kind": kind,
        "exact_match": exact_match,
        "norm_match": norm_match,
        "length_ok": length_ok,
        "actual_len": len(actual),
    }

    if kind == "code":
        try:
            compile(actual_clean, "<bench>", "exec")
            compiles, compile_err = True, ""
        except SyntaxError as exc:  # the only exception compile() raises
            compiles, compile_err = False, f"{exc.__class__.__name__}: {exc}"
        result["compiles"] = compiles
        result["compile_err"] = compile_err

    return result


# --------------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    """Aggregate run+score dicts into per-model stats.

    Input: list of dicts produced by the bench loop — merge of run_model()
    output and score() output, plus 'model' and 'prompt_id' keys.

    Output (per model):
        prompts        total prompts run
        compiled       count of code prompts whose output compiled
        exact          count of exact matches
        norm           count of whitespace-normalized matches
        avg_elapsed_s  mean elapsed seconds across prompts
        tokens_per_s   rough estimate: tokens ≈ len(stdout chars)/4,
                       divided by total elapsed (0.0 if none).
    """
    per_model: dict[str, list[dict]] = {}
    for r in results:
        m = r.get("model", "<unknown>")
        per_model.setdefault(m, []).append(r)

    out: dict[str, dict] = {}
    for model, runs in per_model.items():
        n = len(runs)
        compiled = sum(1 for r in runs if r.get("compiles"))
        exact = sum(1 for r in runs if r.get("exact_match"))
        norm = sum(1 for r in runs if r.get("norm_match"))
        total_elapsed = sum(float(r.get("elapsed_s", 0.0)) for r in runs)
        total_chars = sum(len(r.get("stdout", "") or "") for r in runs)
        est_tokens = total_chars / 4.0
        tokens_per_s = round(est_tokens / total_elapsed, 2) if total_elapsed > 0 else 0.0
        out[model] = {
            "prompts": n,
            "compiled": compiled,
            "exact": exact,
            "norm": norm,
            "avg_elapsed_s": round(total_elapsed / n, 4) if n else 0.0,
            "tokens_per_s": tokens_per_s,
        }
    return out


def run_bench(
    model_bin: str,
    model_path: str,
    prompt_path: str | None = None,
    max_tokens: int = 128,
    temp: float = 0.0,
    timeout_s: int = 120,
) -> list[dict]:
    """Run the full prompt set against one model; return merged run+score dicts.

    One entry per prompt: {model, prompt_id, kind, rc, timed_out, elapsed_s,
    stdout, stderr, exact_match, norm_match, length_ok, compiles?}.
    Timeouts are HONEST (rc=124 / timed_out=True); the workflow wraps this
    in shell ``timeout -k`` for SIGKILL escalation exactly like the worker.
    """
    results: list[dict] = []
    for p in load_prompts(prompt_path):
        run = run_model(
            p["prompt"], model_bin, model_path,
            max_tokens=max_tokens, temp=temp, timeout_s=timeout_s,
        )
        scored = score(p.get("expect", ""), run["stdout"], p["kind"])
        results.append({
            "model": model_path,
            "prompt_id": p["id"],
            "kind": p["kind"],
            **run,
            **scored,
        })
    return results


def _write_bench_outputs(results: list[dict], out_dir: str) -> dict:
    """Write bench-details.jsonl + bench-summary.json into out_dir; return summary."""
    import json
    os.makedirs(out_dir, exist_ok=True)
    details_path = os.path.join(out_dir, "bench-details.jsonl")
    with open(details_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=True) + "\n")
    summary = summarize(results)
    summary_path = os.path.join(out_dir, "bench-summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "results": len(results)}, fh, indent=2)
    print(f"wrote {details_path} ({len(results)} rows) + {summary_path}")
    return summary


def _selftest() -> int:
    """Self-test with FAKE model binaries (no real model needed). Exits 0 iff all pass."""
    import shutil
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    # 1) prompts
    ps = load_prompts()
    check("load_prompts returns 8", len(ps) == 8, f"len={len(ps)}")
    check("prompts carry expect", all("expect" in p for p in ps))

    # 2) run_model with a good fake binary
    tmp = tempfile.mkdtemp(prefix="bench_selftest_")
    good = os.path.join(tmp, "good_model.sh")
    with open(good, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\nprintf 'def fib(n):\\n    return n\\n'\n")
    os.chmod(good, 0o755)
    r = run_model("write fib", good, "/nonexistent.gguf", max_tokens=16, timeout_s=10)
    check("run_model good rc==0", r["rc"] == 0, f"rc={r['rc']}")
    check("run_model stdout captured", "def fib" in r["stdout"], f"stdout={r['stdout'][:30]!r}")
    check("run_model not timed out", r["timed_out"] is False)

    # 3) scoring: compile + exact + norm
    s = score("def fib(n):\n    return n", "```python\ndef fib(n):\n    return n\n```", "code")
    check("score code compiles", s.get("compiles") is True)
    check("score code exact", s.get("exact_match") is True)
    s2 = score("def fib(n):\n    return n", "def   fib(n):\n    return n", "code")
    check("score code norm-only", s2.get("exact_match") is False and s2.get("norm_match") is True)
    s3 = score("return sum(xs)", "def broken(:\n", "code")
    check("score broken code never raises", s3.get("compiles") is False and s3.get("compile_err"))

    # 4) timeout path with a sleeping fake binary
    slow = os.path.join(tmp, "slow_model.sh")
    with open(slow, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\nsleep 5\n")
    os.chmod(slow, 0o755)
    r2 = run_model("slow", slow, "/nonexistent.gguf", max_tokens=8, timeout_s=1)
    check("run_model timeout timed_out", r2["timed_out"] is True and r2["rc"] == 124,
          f"timed_out={r2['timed_out']} rc={r2['rc']}")

    # 5) summarize totals on mixed results
    res = [
        {"model": "A", "prompt_id": "fib", "stdout": "x" * 40, "elapsed_s": 1.0,
         "compiles": True, "exact_match": True, "norm_match": True},
        {"model": "A", "prompt_id": "clamp", "stdout": "y" * 80, "elapsed_s": 3.0,
         "compiles": False, "exact_match": False, "norm_match": False},
        {"model": "B", "prompt_id": "sum_list", "stdout": "z" * 20, "elapsed_s": 2.0,
         "compiles": True, "exact_match": False, "norm_match": True},
    ]
    sm = summarize(res)
    check("summarize per-model", set(sm) == {"A", "B"})
    check("summarize totals", sm["A"]["prompts"] == 2 and sm["A"]["exact"] == 1
          and sm["A"]["compiled"] == 1 and sm["A"]["norm"] == 1)
    check("summarize tokens_per_s", sm["A"]["tokens_per_s"] > 0, f"tps={sm['A']['tokens_per_s']}")

    # 6) run_bench end-to-end with a fake good binary + output writers
    r = run_bench(good, "/fake.gguf", max_tokens=16, timeout_s=10)
    check("run_bench runs all prompts", len(r) == len(DEFAULT_PROMPTS), f"n={len(r)}")
    check("run_bench rows carry rc+score", all("rc" in x and "exact_match" in x for x in r))
    out_dir = os.path.join(tmp, "benchout")
    sm2 = _write_bench_outputs(r, out_dir)
    check("bench output files exist",
          os.path.isfile(os.path.join(out_dir, "bench-details.jsonl"))
          and os.path.isfile(os.path.join(out_dir, "bench-summary.json")))
    check("bench summary per-model", "/fake.gguf" in sm2 and sm2["/fake.gguf"]["prompts"] == len(DEFAULT_PROMPTS))

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {'ALL SELFTESTS PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


def _main(argv: list[str] | None = None) -> int:
    """CLI: ``python3 bench_inference.py --selftest`` | real-bench mode.

    Real-bench mode (used by the Slice-6 workflow, off PROD on GitHub runners):
        python3 bench_inference.py --bench --bin <llama-cli> --model <gguf> \
            [--prompts <file>] [--max-tokens 128] [--temp 0.0] [--timeout 120] \
            [--out <dir>]
    Exits 0 iff every prompt ran and every output row scored without raising.
    """
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="7B-army Slice 6 benchmark harness")
    parser.add_argument("--selftest", action="store_true", help="run fake-model self-tests only")
    parser.add_argument("--bench", action="store_true", help="run a real benchmark")
    parser.add_argument("--bin", help="llama-cli (or compatible) binary path")
    parser.add_argument("--model", help="GGUF model path")
    parser.add_argument("--prompts", default=None, help="prompt JSON/JSONL file (default embedded 8)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", default="benchout", help="output dir for details.jsonl + summary.json")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.bench:
        parser.print_help()
        return 2
    if not args.bin or not args.model:
        print("FAIL: --bench requires --bin <llama-cli> and --model <gguf>", file=sys.stderr)
        return 2

    results = run_bench(args.bin, args.model, args.prompts,
                        max_tokens=args.max_tokens, temp=args.temp, timeout_s=args.timeout)
    summary = _write_bench_outputs(results, args.out)
    print(_json.dumps({"summary": summary, "results": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
