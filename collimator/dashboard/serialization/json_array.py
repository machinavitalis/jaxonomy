# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""A very basic wrapper to serialize np/jnp arrays to JSON"""

import json


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


def dump(*args, **kwargs):
    json.dump(*args, cls=JsonEncoder, **kwargs)


def dumps(*args, **kwargs):
    return json.dumps(*args, cls=JsonEncoder, **kwargs)
