#!/usr/bin/env bash
# Build a PythonFMU FMI wrapper for this host and install it into the
# active pythonfmu package.
#
# ``build_fmu`` compiles nothing: it bundles the pre-built wrapper that
# ships inside the pythonfmu wheel. That wrapper is fine when the FMI
# master is itself a Python process (FMPy, jaxonomy) on x86-64, and
# unusable otherwise. Three separate reasons, all invisible to the FMI
# validators — they read modelDescription.xml and never load the binary:
#
#   1. ISA. FMI 2.0's platform folders (linux64 / darwin64 / win64) name
#      a word size, not an instruction set, and the wheels carry x86-64
#      builds. On aarch64 the FMU fails to load with a bare "cannot open
#      shared object file".
#   2. Python linkage. Upstream links against CMake's Python3::Module,
#      which deliberately leaves libpython unresolved. A C/C++ master
#      (OpenModelica, fmusim, most commercial tools) then cannot resolve
#      Py_Initialize and dlopen fails outright.
#   3. Symbol visibility. Even linked, a master that dlopens the wrapper
#      RTLD_LOCAL keeps libpython out of the global namespace, and numpy's
#      C extensions fail with "undefined symbol: PyObject_SelfIter".
#
# And one crash: PyState calls Py_Finalize() on teardown, which is not
# safe once numpy/jax are loaded — a conforming master segfaults on
# fmi2Terminate / fmi2FreeInstance.
#
# This script builds from source for the host ISA and applies all three
# fixes. Requires cmake, a C++17 compiler, git, and the Python
# development headers (python3-dev / python3.x-dev).
#
# Usage:
#   scripts/build_pythonfmu_wrapper.sh [--ref <git-ref>] [--keep-finalize]
#
# Verify afterwards with:
#   python -c "from jaxonomy.library.fmu_export import wrapper_diagnostics as w; print(w())"

set -euo pipefail

REF="master"
KEEP_FINALIZE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        --keep-finalize) KEEP_FINALIZE=1; shift ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The wrapper must be built against the interpreter that will host the FMU,
# which is the one holding pythonfmu — not necessarily whatever `python3`
# resolves to. Under pyenv or conda those differ, and building against the
# wrong one links a libpython the host never loads. Probe candidates unless
# PYTHON was set explicitly.
_find_python() {
    local candidate
    local pyenv_python=""
    if command -v pyenv >/dev/null 2>&1; then
        pyenv_python="$(pyenv which python 2>/dev/null || true)"
    fi
    for candidate in "${PYTHON:-}" python3 python "$pyenv_python"; do
        [[ -n "$candidate" ]] || continue
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c "import pythonfmu" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}
PYTHON="$(_find_python)" || {
    echo "no interpreter found that can import pythonfmu (tried PYTHON, python3, python, pyenv)" >&2
    echo "install it, or set PYTHON to the interpreter that has it" >&2
    exit 1
}
echo "building against $PYTHON"
command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }

DEST_DIR="$("$PYTHON" - <<'PY'
import os
import pythonfmu
print(os.path.join(os.path.dirname(pythonfmu.__file__), "resources", "binaries"))
PY
)"
[[ -d "$DEST_DIR" ]] || { echo "pythonfmu resources not found at $DEST_DIR" >&2; exit 1; }

case "$(uname -s)" in
    Linux)  SLOT=linux64;  LIBNAME=libpythonfmu-export.so ;;
    Darwin) SLOT=darwin64; LIBNAME=libpythonfmu-export.dylib ;;
    *)      echo "unsupported host: $(uname -s)" >&2; exit 1 ;;
esac

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git clone --quiet --depth 1 --branch "$REF" \
    https://github.com/NTNU-IHB/PythonFMU.git "$WORK/PythonFMU"
cd "$WORK/PythonFMU"

"$PYTHON" - "$KEEP_FINALIZE" <<'PY'
import sys

keep_finalize = sys.argv[1] == "1"

# 1. Ask CMake for the embedding component and link it, so the wrapper
#    carries libpython instead of expecting a Python host to supply it.
cml = "CMakeLists.txt"
text = open(cml).read()
for component in ("Development.SABIModule", "Development.Module"):
    text = text.replace(
        f"find_package(Python3 REQUIRED COMPONENTS {component})",
        f"find_package(Python3 REQUIRED COMPONENTS {component} Development.Embed)",
    )
open(cml, "w").write(text)

src_cml = "src/CMakeLists.txt"
text = open(src_cml).read()
anchor = "target_link_libraries (pythonfmu-export PRIVATE Python3::Module)"
if anchor not in text:
    raise SystemExit("PythonFMU layout changed: link target not found")
replacement = (
    "target_link_libraries (pythonfmu-export PRIVATE Python3::Python)\n\n"
    "if (UNIX AND NOT APPLE)\n"
    "  target_compile_definitions(pythonfmu-export PRIVATE\n"
    '      PYTHONFMU_LIBPYTHON_SONAME="libpython'
    "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
    '.so.1.0")\n'
    "endif ()"
)
open(src_cml, "w").write(text.replace(anchor, replacement))

# 0. PythonFMU guards a destructor attribute for Windows and Linux only and
#    hard-fails anywhere else ("#error port the code"), so the source build
#    is impossible on macOS until __APPLE__ is accepted too.
slave = "src/pythonfmu/PySlaveInstance.cpp"
text = open(slave).read()
if "defined(__linux__) || defined(__APPLE__)" not in text:
    if "#elif defined(__linux__)" not in text:
        raise SystemExit("PythonFMU layout changed: platform guard not found")
    open(slave, "w").write(
        text.replace("#elif defined(__linux__)",
                     "#elif defined(__linux__) || defined(__APPLE__)")
    )
    print("patched PySlaveInstance.cpp for macOS")

state = "src/pythonfmu/PyState.hpp"
text = open(state).read()

# 2. Promote libpython to global scope before Py_Initialize, so numpy's
#    C extensions resolve even when the master dlopened us RTLD_LOCAL.
text = text.replace(
    "#include <Python.h>",
    "#include <Python.h>\n"
    "#include <cstdlib>\n"
    "#if defined(PYTHONFMU_LIBPYTHON_SONAME)\n"
    "#include <dlfcn.h>\n"
    "#endif",
    1,
)
init_anchor = (
    "                auto const justInitialized = !Py_IsInitialized();\n"
    "                if (justInitialized) {\n"
    "                    Py_Initialize();"
)
if init_anchor not in text:
    raise SystemExit("PythonFMU layout changed: Py_Initialize block not found")
text = text.replace(
    init_anchor,
    "                auto const justInitialized = !Py_IsInitialized();\n"
    "                if (justInitialized) {\n"
    "#if defined(PYTHONFMU_LIBPYTHON_SONAME)\n"
    "                    // A master that dlopened this wrapper RTLD_LOCAL keeps\n"
    "                    // libpython out of the global namespace, and numpy's C\n"
    "                    // extensions then fail to resolve the CPython API.\n"
    "                    dlopen(PYTHONFMU_LIBPYTHON_SONAME,\n"
    "                           RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);\n"
    "#endif\n"
    "                    Py_Initialize();",
)

# 3. Skip Py_Finalize: tearing down an interpreter that loaded numpy/jax
#    segfaults, and a conforming master always calls fmi2FreeInstance.
if not keep_finalize:
    finalize_anchor = (
        "                    PyEval_RestoreThread(mainPyThread);\n"
        "                    Py_Finalize();"
    )
    if finalize_anchor not in text:
        raise SystemExit("PythonFMU layout changed: Py_Finalize block not found")
    text = text.replace(
        finalize_anchor,
        "                    PyEval_RestoreThread(mainPyThread);\n"
        "                    // Py_Finalize() tears down an interpreter that may\n"
        "                    // have loaded C extension modules whose teardown is\n"
        "                    // not re-entrant; under a non-Python master that\n"
        "                    // segfaults on fmi2Terminate / fmi2FreeInstance.\n"
        "                    // Leaking the interpreter is the safe trade.\n"
        '                    if (std::getenv("PYTHONFMU_FINALIZE") != nullptr) {\n'
        "                        Py_Finalize();\n"
        "                    }",
    )

open(state, "w").write(text)
print("patched PythonFMU sources")
PY

cmake -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v "$PYTHON")"
cmake --build build --config Release

BUILT="$WORK/PythonFMU/pythonfmu/resources/binaries/$SLOT/$LIBNAME"
[[ -f "$BUILT" ]] || { echo "build produced no $BUILT" >&2; exit 1; }

mkdir -p "$DEST_DIR/$SLOT"
if [[ -f "$DEST_DIR/$SLOT/$LIBNAME" && ! -f "$DEST_DIR/$SLOT/$LIBNAME.stock" ]]; then
    cp "$DEST_DIR/$SLOT/$LIBNAME" "$DEST_DIR/$SLOT/$LIBNAME.stock"
    echo "kept the shipped wrapper as $LIBNAME.stock"
fi
cp "$BUILT" "$DEST_DIR/$SLOT/$LIBNAME"
echo "installed $SLOT wrapper into $DEST_DIR/$SLOT/"

"$PYTHON" - <<'PY'
from jaxonomy.library.fmu_export import wrapper_diagnostics

info = wrapper_diagnostics()
print(f"  platform:        {info['platform']}")
print(f"  machine:         {info['machine']} (host {info['host_machine']})")
print(f"  arch matches:    {info['arch_matches_host']}")
print(f"  embeds python:   {info['embeds_python']}")
if not (info["arch_matches_host"] and info["embeds_python"]):
    raise SystemExit("wrapper still does not satisfy both checks")
PY
