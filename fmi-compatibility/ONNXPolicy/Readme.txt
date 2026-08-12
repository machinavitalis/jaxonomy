ONNXPolicy
==========

A neural policy exported to ONNX, evaluated inside the FMU by jaxonomy's
ONNXJax block and held at the sample rate it was trained for.

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
ONNXPolicy.fmu          the FMU
ONNXPolicy_ref.csv      reference solution computed by Jaxonomy
ONNXPolicy_ref.opt      options used to compute it
onnxpolicy_slave.py  the model source, for reference
policy.onnx       bundled into the FMU resources

Checked with fmpy.validate_fmu, INTO-CPS VDMCheck 1.1.3 and
fmusim validate. See ../README.md for the full compatibility
matrix.
