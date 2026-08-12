Parameterized
=============

FMI parameters applied at initialization: an EXPOSE_INITIAL_STATES
initial state and a Constant-backed rate, both set before
exitInitializationMode.

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
Parameterized.fmu          the FMU
Parameterized_ref.csv      reference solution computed by Jaxonomy
Parameterized_ref.opt      options used to compute it
parameterized_slave.py  the model source, for reference

Parameters
----------
The reference was computed with x0 = 2.5, decay_rate = 0.8, set during
initialization mode. _ref.opt records the remaining options.

Checked with fmpy.validate_fmu, INTO-CPS VDMCheck 1.1.3 and
fmusim validate. See ../README.md for the full compatibility
matrix.
