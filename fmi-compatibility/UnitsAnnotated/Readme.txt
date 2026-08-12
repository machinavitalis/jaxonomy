UnitsAnnotated
==============

Ports carrying jaxonomy Unit annotations. The model is unit-consistent
internally; the FMU does not carry the units (see Readme.txt).

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
UnitsAnnotated.fmu          the FMU
UnitsAnnotated_ref.csv      reference solution computed by Jaxonomy
UnitsAnnotated_ref.opt      options used to compute it
unitsannotated_slave.py  the model source, for reference

Units do not cross the boundary
-------------------------------
The source annotates its ports with jaxonomy Units (m for the level,
m3/s for the inflow) and jaxonomy checks them at connect time. The FMU
does not carry them: modelDescription.xml has no unit attribute on
`level` and no UnitDefinitions block, because the PythonFMU wrapper
this export is built on emits neither.

So this model is the negative result in the set. Its trajectory is
correct -- it matches the analytic h(t) = q_in*R*(1 - exp(-t/(A*R)))
exactly -- but an importer learns nothing about what the number means.

Checked with fmpy.validate_fmu, INTO-CPS VDMCheck 1.1.3 and
fmusim validate. See ../README.md for the full compatibility
matrix.
