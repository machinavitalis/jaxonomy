# Installation

## Requirements

- **Python** 3.10 or newer  
- A **64-bit** environment is assumed for JAX wheels on most platforms  

## Recommended: virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate         # Windows cmd
pip install --upgrade pip
pip install jaxonomy
```

## Platform notes

- **Windows:** for double precision in JAX you may need  
  `set JAX_ENABLE_X64=True` (cmd) or `$env:JAX_ENABLE_X64="True"` (PowerShell) before importing JAX.  
- **macOS:** some optional builds (e.g. pieces of the NMPC stack, and other compiled dependencies on Apple Silicon) need **cmake** (`brew install cmake`).  

## Optional dependency groups

Install extras in brackets:

| Command | Typical use |
|---------|-------------|
| `pip install jaxonomy[safe]` | Scientific / ML extras, **no** NMPC/IPOPT. Note this is a **large, multi-GB** install: it pulls in PyTorch, TensorFlow, pandas, SymPy, python-control, PySINDy, pyTwin, Matplotlib, OpenCV, evosax, and nlopt. |
| `pip install jaxonomy[nmpc]` | Nonlinear MPC blocks (cyipopt + OSQP) — requires **IPOPT** on the system. |
| `pip install jaxonomy[all]` | Everything: the `[safe]` and `[nmpc]` sets **plus** MuJoCo / MJX / Brax. Largest install. |

### Nonlinear MPC (IPOPT)

NMPC blocks expect **IPOPT** to be available on the machine:

- **Ubuntu:** `sudo apt install coinor-libipopt-dev`  
- **macOS:** `brew install ipopt` (and `brew install cmake` if builds fail)  

Then:

```bash
pip install jaxonomy[nmpc]
# or
pip install jaxonomy[all]
```

If `pip install jaxonomy` or `pip install jaxonomy[all]` fails in the resolver (for example conflicting NumPy pins between transitive dependencies), fix or relax constraints in your environment, or install the project **from a clone** with `pip install -e . --no-deps` after you have compatible JAX/NumPy/etc. already installed.

## Development install (git clone)

From the repository root:

```bash
pip install -e .
```

That makes `import jaxonomy` work from any working directory for that interpreter.

**Jupyter / VS Code notebooks:** pick a kernel that uses the **same** Python where you ran `pip install -e .`, or add the repo root to `PYTHONPATH` / use:

```python
import sys
sys.path.insert(0, "/absolute/path/to/repo")
```

**Building the documentation site** (MkDocs):

```bash
pip install -r requirements.docs.txt
mkdocs serve    # preview at http://127.0.0.1:8000 by default
```

---

<details>
<summary>License</summary>

The `jaxonomy` package is released under the <a href="https://mit-license.org/">MIT</a> license.

</details>
