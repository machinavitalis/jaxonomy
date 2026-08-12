VectorIO
========

Array-valued variables. FMI 2.0 has no array type, so a vector port
flattens to one scalar variable per element, named name[i].

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
VectorIO.fmu          the FMU
VectorIO_ref.csv      reference solution computed by Jaxonomy
VectorIO_ref.opt      options used to compute it
vectorio_slave.py  the model source, for reference

Array variables
---------------
FMI 2.0 has no array type, so each element is its own ScalarVariable:
u_vec[0..2] as inputs and y_vec[0..2] as outputs, plus the scalar
y_sum. Vector Constants and vector output ports both flatten this way.

An *exported* vector input port (bld.export_input) does not: the port
carries no shape until something feeds it, so the slave registers a
single scalar. Drive array inputs from a vector Constant, as here.

Checked with fmpy.validate_fmu, INTO-CPS VDMCheck 1.1.3 and
fmusim validate. See ../README.md for the full compatibility
matrix.
