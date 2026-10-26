"""Generate a Jupyter notebook next to every non-empty .py file in app/, src/ and tests/.

The .py files stay the source of truth (the app, tests and `python -m src...` commands import them);
re-run this after editing any of them.  Run from the project root:  python tools/py_to_notebooks.py
Each notebook gets a title cell, a setup cell that moves to the project root, then one cell for the
imports and one per top-level function/class.
"""
import ast
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


SETUP_CELL = '''\
# Setup: work from the project root so `src` / `app` imports and relative data paths resolve,
# wherever Jupyter was started.
import os
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.yaml").exists())
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))'''

# Notebook-safe replacements for `if __name__ == "__main__":` blocks that misbehave in a kernel.
MAIN_OVERRIDES = {
    "src/evaluation/run_evaluation.py": '''\
# Choose what this cell does:
#   None       -> nothing (default, safe for a demo)
#   "analyse"  -> recompute tables/figures from the saved out-of-fold predictions (fast)
#   "full"     -> re-run the whole nested cross-validation (about 10 min) and overwrite results/
MODE = None
if MODE is not None:
    main(MODE == "analyse")'''
}


def convert(path: Path) -> Path:
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    body = list(tree.body)
    cells = []

    # Module docstring -> markdown title cell
    title = f"# `{path.as_posix()}`"
    start_line = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        title += "\n\n" + ast.get_docstring(tree, clean=True)
        start_line = body[0].end_lineno
        body = body[1:]
    cells.append(new_markdown_cell(title))
    cells.append(new_code_cell(SETUP_CELL))

    # Group statements: each def/class (with decorators + leading comments) is its own cell;
    # consecutive other statements are grouped together.
    chunks = []  # (start_idx, end_idx, is_def)
    for node in body:
        s = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
        e = node.end_lineno
        is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        is_main = isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
        sep = is_def or is_main
        if chunks and not sep and not chunks[-1][2]:
            chunks[-1] = (chunks[-1][0], e, False)
        else:
            chunks.append((s, e, sep))

    # Extend each chunk's start back to cover comments/blank lines after the previous chunk
    prev_end = start_line
    for i, (s, e, d) in enumerate(chunks):
        chunks[i] = (prev_end, e, d)
        prev_end = e
    if chunks:  # trailing comments
        s, e, d = chunks[-1]
        chunks[-1] = (s, len(lines), d)

    for s, e, _ in chunks:
        code = "\n".join(lines[s:e]).strip("\n")
        if not code.strip():
            continue
        override = MAIN_OVERRIDES.get(path.as_posix())
        if override and "__main__" in code:
            code = code[:code.index("if __name__")] + override
        # Notebooks have no __file__; the setup cell already made the project root the cwd.
        code = re.sub(r"Path\(__file__\)\.resolve\(\)\.parents\[\d\]", "Path.cwd()", code)
        cells.append(new_code_cell(code))

    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    out = path.with_suffix(".ipynb")
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    paths = [Path(a) for a in sys.argv[1:]] or sorted(
        p for d in ("app", "src", "tests") for p in Path(d).rglob("*.py"))
    for p in paths:
        if not p.read_text().strip():
            print("skip (empty):", p)
            continue
        print("wrote", convert(p))
