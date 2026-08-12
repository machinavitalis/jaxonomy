PIController
============

Discrete-time PI controller. Sampled dynamics with two real inputs and
one output.

Exported from Jaxonomy (https://py.jaxonomy.com/) as an FMI 2.0
co-simulation FMU via jaxonomy.library.fmu_export.build_fmu.

IMPORTANT — this FMU is tool-coupled. The slave runs as Python,
so the importing side needs a Python environment with jaxonomy
installed:

    pip install "jaxonomy[fmu]"

It is not a self-contained binary. Platform binaries come from
the PythonFMU wrapper and are x86-64.

Files
-----
PIController.fmu          the FMU
PIController_ref.csv      reference solution computed by Jaxonomy
PIController_ref.opt      options used to compute it
PIController_in.csv       input signals
picontroller_slave.py  the model source, for reference

Checked with fmpy.validate_fmu, INTO-CPS VDMCheck 1.1.3 and
fmusim validate. See ../README.md for the full compatibility
matrix.
