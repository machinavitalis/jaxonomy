# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import concurrent.futures
from functools import lru_cache
import io

import numpy as np

from collimator.dashboard import api

from ..lazy_loader import LazyLoader

requests = LazyLoader("requests", globals(), "requests")


@lru_cache()
def _download(s3_url):
    return requests.get(s3_url).content


def get_signals(
    model_uuid, simulation_uuid, signals: list[str] = None
) -> dict[str, np.ndarray]:
    response = api.get(f"/models/{model_uuid}/simulations/{simulation_uuid}/signals")
    results = {}
    futures = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for s3_url in response["s3_urls"]:
            signal_name = s3_url.get("name").replace(".npy", "").replace(".npz", "")
            if signals is not None and signal_name not in signals:
                continue
            futures[signal_name] = executor.submit(_download, s3_url.get("url"))
        for name, future in futures.items():
            result = future.result()
            results[name] = np.load(io.BytesIO(result))

    return results
