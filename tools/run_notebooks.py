"""Execute every notebook under a folder, each from its own directory (as Jupyter does), and report failures.

Usage:  python tools/run_notebooks.py [ROOT]   (default: the project root)
Running the notebooks rewrites files in results/ and reports/ just as the scripts do; point ROOT at a copy
of the project to leave the real outputs untouched.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
fails = []
for p in sorted(q for q in root.glob("**/*.ipynb") if ".venv" not in q.parts and ".ipynb_checkpoints" not in q.parts):
    t = time.time()
    nb = nbformat.read(p, 4)
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(p.parent)}}).execute()
        print(f"OK   {time.time()-t:6.1f}s {p.relative_to(root)}", flush=True)
    except Exception as e:
        fails.append(p); print(f"FAIL {time.time()-t:6.1f}s {p.relative_to(root)}\n     {str(e)[-800:]}", flush=True)
print("failures:", len(fails))
sys.exit(1 if fails else 0)
