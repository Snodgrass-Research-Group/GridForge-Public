"""MATLAB call-graph analyzer for the GridForge conversion project.

Parses every ``.m`` file in the two dissertation code folders, builds a static
call graph rooted at the two main scripts (``NationalGarverExpansion.m`` and
``NationalConductorSizing.m``), and reports:

* the call tree (who calls whom, to arbitrary depth),
* which functions are ACTIVE (reachable from a root) vs ORPHANED,
* which active functions exist in BOTH the "KCL-Compliant" and
  "KCL-KVL Compliant" folders (the KCL-KVL copy is authoritative per the
  design decisions in docs/ARCHITECTURE.md),
* which external dependencies each file touches (CPLEX, Optimization
  Toolbox, MEX files, Excel I/O, .mat I/O, plotting).

Outputs (written to docs/ unless --out is given):
    call_tree.txt   - indented text tree + flat adjacency list (machine-parseable)
    call_graph.json - full structured dump
    CALL_TREE.md    - Mermaid diagram (renders on GitHub / VS Code)

Usage:  python tools/analyze_matlab_calls.py --matlab-parent <folder> [--out docs]
where <folder> holds "KCL-Compliant Grid Generator" and "KCL-KVL Compliant
Grid Generator". Reads the MATLAB sources; writes only the three outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# Set from the command line in main().
REPO = Path(".")
KCL = REPO / "KCL-Compliant Grid Generator"
KVL = REPO / "KCL-KVL Compliant Grid Generator"
DOCS = Path(__file__).resolve().parents[1] / "docs"

ROOTS = ["NationalGarverExpansion", "NationalConductorSizing"]

# External symbols worth tracking (not part of the dissertation codebase).
EXTERNALS = {
    "cplexmilp": "CPLEX MILP solver  -> gurobipy",
    "cplexlp": "CPLEX LP solver    -> gurobipy",
    "cplexqp": "CPLEX QP solver    -> gurobipy",
    "linprog": "Opt. Toolbox LP    -> gurobipy or scipy.optimize.linprog",
    "quadprog": "Opt. Toolbox QP    -> gurobipy",
    "intlinprog": "Opt. Toolbox MILP  -> gurobipy",
    "fmincon": "Opt. Toolbox NLP   -> scipy.optimize / gurobipy",
    "VChooseK": "C MEX combinations -> itertools.combinations / numpy",
    "xlswrite": "Excel write        -> pandas/openpyxl",
    "xlsread": "Excel read         -> pandas/openpyxl",
    "eigs": "sparse eigensolver -> scipy.sparse.linalg.eigsh",
    "graphconncomp": "Bioinfo Toolbox    -> scipy.sparse.csgraph / networkx",
    "distance": "Mapping Toolbox    -> haversine (numpy)",
    "deg2km": "Mapping Toolbox    -> numpy constant",
}


def matlab_files(folder: Path) -> list[Path]:
    """All .m files under folder, skipping OLD/backup subdirectories."""
    out = []
    for p in folder.rglob("*.m"):
        rel = str(p.relative_to(folder))
        if "OLD" in rel.split("\\")[0:1] or "\\OLD" in rel or "/OLD" in rel:
            continue
        out.append(p)
    return out


def strip_comments(src: str) -> str:
    """Remove %{...%} blocks and % line comments (string-aware, best effort)."""
    src = re.sub(r"^\s*%\{.*?^\s*%\}", "", src, flags=re.M | re.S)
    lines = []
    for line in src.splitlines():
        out, in_str = [], False
        i = 0
        while i < len(line):
            c = line[i]
            if c == "'":
                # naive string toggle; '' escape handled by double toggle (harmless)
                in_str = not in_str
                out.append(c)
            elif c == "%" and not in_str:
                break
            else:
                out.append(c)
            i += 1
        lines.append("".join(out))
    return "\n".join(lines)


def find_calls(code: str, names: set[str], self_name: str) -> set[str]:
    """Names from `names` referenced in code as calls (paren or bare-script)."""
    found = set()
    for name in names:
        if name == self_name:
            continue
        # standard call:  Name(  ... )    OR bare script/command invocation on
        # its own statement:  Name;  /  Name<EOL>  (needed for GMDData-style scripts)
        if re.search(rf"\b{re.escape(name)}\s*\(", code):
            found.add(name)
        elif re.search(rf"(?m)^\s*{re.escape(name)}\s*;?\s*$", code):
            found.add(name)
    return found


def main(argv: list[str] | None = None) -> None:
    global REPO, KCL, KVL, DOCS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matlab-parent", required=True, type=Path,
                    help="folder holding the two MATLAB code folders")
    ap.add_argument("--out", type=Path, default=DOCS,
                    help="output folder (default: this repository's docs/)")
    args = ap.parse_args(argv)
    REPO = args.matlab_parent
    KCL = REPO / "KCL-Compliant Grid Generator"
    KVL = REPO / "KCL-KVL Compliant Grid Generator"
    DOCS = args.out
    if not KCL.is_dir() or not KVL.is_dir():
        sys.exit(f"{REPO} must hold both MATLAB folders; nothing written")

    # ---- collect the function universe -------------------------------------
    kcl_files = {p.stem: p for p in matlab_files(KCL)}
    kvl_files = {p.stem: p for p in matlab_files(KVL / "Matlab Functions")}

    # Universe = KCL folder functions + KCL-KVL functions (for cross-folder
    # resolution, e.g. CreateOLMatrices only exists in the KCL-KVL folder).
    universe: dict[str, dict] = {}
    for name, p in kvl_files.items():
        universe[name] = {"kvl": str(p.relative_to(REPO)), "kcl": None}
    for name, p in kcl_files.items():
        universe.setdefault(name, {"kvl": None, "kcl": None})
        universe[name]["kcl"] = str(p.relative_to(REPO))

    names = set(universe) | set(EXTERNALS)

    # ---- parse every file, record edges ------------------------------------
    edges: dict[str, list[str]] = {}
    ext_use: dict[str, list[str]] = {}
    loc: dict[str, int] = {}
    for name, info in universe.items():
        # KCL copy is what the mains actually run today; fall back to KVL copy.
        path = REPO / (info["kcl"] or info["kvl"])
        try:
            src = path.read_text(errors="replace")
        except OSError:
            continue
        loc[name] = src.count("\n") + 1
        code = strip_comments(src)
        called = find_calls(code, names, name)
        edges[name] = sorted(c for c in called if c in universe)
        ext_use[name] = sorted(c for c in called if c in EXTERNALS)

    # ---- BFS from roots -> active set --------------------------------------
    active: dict[str, int] = {}  # name -> min depth
    queue = [(r, 0) for r in ROOTS]
    while queue:
        cur, d = queue.pop(0)
        if cur in active and active[cur] <= d:
            continue
        active[cur] = d
        for callee in edges.get(cur, []):
            queue.append((callee, d + 1))

    orphans = sorted(
        n for n, info in universe.items()
        if n not in active and info["kcl"] is not None  # orphan = unused KCL file
    )

    # ---- text tree ----------------------------------------------------------
    def tree_lines(node: str, depth: int, seen: frozenset) -> list[str]:
        pad = "    " * depth
        tags = []
        info = universe.get(node, {})
        if info.get("kcl") and info.get("kvl"):
            tags.append("BOTH-FOLDERS")
        elif info.get("kvl") and not info.get("kcl"):
            tags.append("KVL-ONLY")
        if ext_use.get(node):
            tags.append("ext:" + ",".join(ext_use[node]))
        tag = ("  [" + "; ".join(tags) + "]") if tags else ""
        n_loc = loc.get(node, 0)
        if node in seen:
            return [f"{pad}{node} (^ repeat){tag}"]
        lines = [f"{pad}{node} ({n_loc} loc){tag}"]
        for callee in edges.get(node, []):
            lines += tree_lines(callee, depth + 1, seen | {node})
        return lines

    txt = [
        "# GridForge MATLAB call tree  (generated " + date.today().isoformat() + ")",
        "# Format: indentation = call depth; (N loc) = lines of code;",
        "#   [BOTH-FOLDERS] = exists in KCL and KCL-KVL folders (KVL authoritative);",
        "#   [KVL-ONLY] = only in KCL-KVL folder; (^ repeat) = subtree shown above;",
        "#   ext: = external/toolbox dependencies.",
        "",
    ]
    for root in ROOTS:
        txt += tree_lines(root, 0, frozenset()) + [""]

    txt += ["# --- FLAT ADJACENCY (caller -> callees) ---"]
    for name in sorted(active, key=lambda n: (active[n], n)):
        txt.append(f"{name} [depth {active[name]}] -> " + (", ".join(edges.get(name, [])) or "(leaf)"))
    txt += ["", "# --- ORPHANED (in KCL folder, unreachable from mains) ---"]
    txt += [f"{n} ({loc.get(n, 0)} loc)" for n in orphans]

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "call_tree.txt").write_text("\n".join(txt), encoding="utf-8")

    # ---- JSON dump ----------------------------------------------------------
    dump = {
        "generated": date.today().isoformat(),
        "roots": ROOTS,
        "active": {n: {"depth": d, "loc": loc.get(n, 0),
                       "calls": edges.get(n, []),
                       "externals": ext_use.get(n, []),
                       "kcl_path": universe.get(n, {}).get("kcl"),
                       "kvl_path": universe.get(n, {}).get("kvl")}
                   for n, d in sorted(active.items())},
        "orphans": {n: {"loc": loc.get(n, 0), "kcl_path": universe[n]["kcl"]}
                    for n in orphans},
    }
    (DOCS / "call_graph.json").write_text(json.dumps(dump, indent=1), encoding="utf-8")

    # ---- Mermaid ------------------------------------------------------------
    mm = ["# GridForge MATLAB Call Tree", "",
          f"Generated {date.today().isoformat()} by `tools/analyze_matlab_calls.py`.",
          "Solid boxes = active functions; annotations list external deps.",
          "", "```mermaid", "flowchart TD"]
    shown = set()
    for caller in sorted(active, key=lambda n: active[n]):
        for callee in edges.get(caller, []):
            edge = f"    {caller} --> {callee}"
            if edge not in shown:
                shown.add(edge)
                mm.append(edge)
    for n in active:
        if ext_use.get(n):
            mm.append(f'    {n}["{n}<br/><i>{", ".join(ext_use[n])}</i>"]')
    mm += ["```", ""]
    (DOCS / "CALL_TREE.md").write_text("\n".join(mm), encoding="utf-8")

    print(f"Active functions: {len(active)}   Orphans (KCL): {len(orphans)}")
    print(f"Wrote call_tree.txt, call_graph.json, CALL_TREE.md to {DOCS}")


if __name__ == "__main__":
    main()
