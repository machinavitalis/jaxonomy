#!/bin/env pytest
# SPDX-License-Identifier: MIT

import pytest
import jaxonomy.testing as test
import os
import shutil


def copy_to_workdir(srcdir, file_name, workdir):
    src = os.path.join(srcdir, file_name)
    dst = os.path.join(workdir, file_name)
    shutil.copyfile(src, dst)


@pytest.mark.skip(
    reason=(
        "manual-only harness: needs a downloaded model project directory "
        "passed as projdir (None under pytest → TypeError). Run this file "
        "directly with JAXONOMY_PALLASCAT_PROJDIR set; see __main__ block."
    )
)
def test_pallascat_model(projdir: str = None):
    # get list of all files in the run/*model_name* directory of pallascat download
    model_files = [
        f for f in os.listdir(projdir) if os.path.isfile(os.path.join(projdir, f))
    ]
    print(f"model_files={model_files}")

    # create test paths so that we can copy the files to a workdir for jaxonomy
    # NOTE: untested in CI — this is a manual-only harness (see the skip mark).
    test_paths = test.get_paths(
        None, testdir_=__file__, test_name_=os.path.basename(projdir)
    )
    print(f"test_paths={test_paths}")

    # copy the files
    for model_file in model_files:
        print(f"model_file={model_file}")
        copy_to_workdir(projdir, model_file, test_paths["workdir"])

    # make jaxonomy look in workdir for json files
    test_paths["testdir"] = test_paths["workdir"]

    # run jaxonomy
    test.run(test_paths=test_paths)


if __name__ == "__main__":
    # Manual harness: point this at a downloaded model project directory
    # (a "run/<model name>" folder containing the exported model JSON) and run
    # this file directly. Pass the path via the JAXONOMY_PALLASCAT_PROJDIR
    # environment variable so nothing machine-specific is committed.
    projdir = os.environ.get("JAXONOMY_PALLASCAT_PROJDIR")
    if not projdir:
        raise SystemExit(
            "Set JAXONOMY_PALLASCAT_PROJDIR to a model project directory "
            "(a 'run/<model>' folder) to use this manual harness."
        )
    test_pallascat_model(projdir)
