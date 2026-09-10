"""Rebuild and execute examples/quickstart.ipynb from examples/quickstart.py.

One markdown + one code cell per numbered section of the script.  Needs nbformat, nbclient
and ipykernel (``pip install -e ".[dev]"``).
"""
import os
import pathlib
import re

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "quickstart.py").read_text()
body = src.split('"""', 2)[2].lstrip("\n")                       # drop the module docstring
body = body.replace('matplotlib.use("Agg")\n', "").replace("import matplotlib\n", "")
body = body.replace("HERE = pathlib.Path(__file__).resolve().parent", "HERE = pathlib.Path('.').resolve()")
parts = re.split(r"\n(?=# \d\. |# Figure )", body)

cells = [new_markdown_cell(
    "# kac_zwanzig quick start\n\nSpectral density → kernel → exact C(t), dC/dt → one trajectory with its random "
    "force → sampled C(t) with error bars → recovering the memory kernel.\n\n"
    "Equation numbers refer to the README. Run from the `examples/` directory.")]
for p in parts:
    p = p.strip("\n")
    if not p:
        continue
    m = re.match(r"# (\d\. [^\n]*?|Figure)[ -]*\n", p)
    if m:
        cells.append(new_markdown_cell("## " + m.group(1).rstrip(" -")))
        p = p[m.end():]
    cells.append(new_code_cell(p))

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)",
                                                        "language": "python"}})
os.chdir(HERE)
# KZ_KERNEL selects the Jupyter kernel used for execution (default: the "python3" kernelspec);
# the notebook itself is saved with the standard "python3" kernelspec.
NotebookClient(nb, timeout=900, kernel_name=os.environ.get("KZ_KERNEL", "python3")).execute()
nbformat.write(nb, HERE / "quickstart.ipynb")
print(f"executed and wrote {HERE / 'quickstart.ipynb'} ({len(cells)} cells)")
