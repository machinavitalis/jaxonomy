# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import dataclasses
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum


# TODO: these dataclasses should be generated from openapi.yaml spec
class ModelKind(StrEnum):
    MODEL = "Model"
    SUBMODEL = "Submodel"


@dataclasses.dataclass
class ModelSummary:
    uuid: str
    kind: ModelKind
    name: str


@dataclasses.dataclass
class FileSummary:
    uuid: str
    name: str  # url
    status: str


@dataclasses.dataclass
class ProjectSummary:
    uuid: str
    title: str
    models: list[ModelSummary]
    reference_submodels: list[ModelSummary]
    files: list[FileSummary]
