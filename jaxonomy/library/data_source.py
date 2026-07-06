# SPDX-License-Identifier: MIT

"""DataSource block for loading time-series data from files.

Supported file formats:

- **.csv** — Comma-separated values. Parsed with pandas when available, otherwise
  NumPy (:func:`numpy.loadtxt` / :func:`numpy.genfromtxt`). Use ``header_as_first_row``
  for a header row; set ``time_samples_as_column`` when the first column (or
  ``time_column``) stores sample times.
- **.npy** — NumPy array. A 1-D array is treated as the signal with synthetic time
  ``arange(n) * sampling_interval``. A 2-D array uses column 0 as time and remaining
  columns as data when ``time_samples_as_column`` is True; otherwise synthetic time is
  used and all columns are data.
- **.npz** — NumPy archive. Loads ``time`` and ``y`` arrays if present; otherwise uses
  the first two arrays in sorted key order as time and a single data column.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Optional

import numpy as np

from .generic import SourceBlock
from ..framework import DependencyTicket, LeafSystem, parameters
from jaxonomy.backend import numpy_api as npa

if TYPE_CHECKING:
    from jaxonomy.simulation.types import SimulationResults

__all__ = [
    "DataSource",
    "SimulationResultsSource",
]


def is_literal_eval_compatible(s):
    try:
        v = ast.literal_eval(s)
        return v
    except (ValueError, SyntaxError):
        return None


def make_time_col(start, step, n):
    return np.array([i * step + start for i in range(n)])


def str2colindices(s):
    slice_thing = s.split(":")
    list_thing = s.split(",")
    if s.isdigit():
        index = int(s)
        return [index]
    elif len(slice_thing) == 2:
        start = slice_thing[0]
        end = slice_thing[1]
        if start.isdigit() and end.isdigit():
            return range(int(start), int(end))
        else:
            raise ValueError(f"DataSource data_columns={s} is not a proper slice")
    elif len(list_thing) > 1:
        is_list_of_digits = [i.isdigit() for i in list_thing]
        if all(is_list_of_digits):
            return [int(i) for i in list_thing]
        elif any(is_list_of_digits):
            raise ValueError(
                f"DataSource data_columns={s} is not a list of either ints or col names. mixting not allowed."
            )
        else:
            retval = is_literal_eval_compatible(s)
            if isinstance(retval, list):
                is_list_of_strings = [isinstance(i, str) for i in retval]
                if all(is_list_of_strings):
                    return retval

            raise ValueError(f"DataSource data_columns={s} is not comprehensible")

    elif len(slice_thing) == 1 and len(list_thing) == 1:
        return [s]
    else:
        raise ValueError(f"DataSource data_columns={s} is not comprehensible")


def _usecols_to_indices(usecols, names: list[str]) -> list[int]:
    if isinstance(usecols, range):
        seq = list(usecols)
    else:
        seq = list(usecols)
    idxs = []
    for u in seq:
        if isinstance(u, int):
            idxs.append(u)
        else:
            us = str(u)
            if us in names:
                idxs.append(names.index(us))
            elif us.isdigit():
                idxs.append(int(us))
            else:
                raise ValueError(
                    f"DataSource: unknown column specifier {u!r} among columns {names}"
                )
    return idxs


def _load_csv_numpy(
    file_name: str,
    data_columns: str,
    header_as_first_row: bool,
    sampling_interval: float,
    time_column: str,
    time_samples_as_column: bool,
):
    if header_as_first_row:
        tab = np.genfromtxt(
            file_name,
            delimiter=",",
            names=True,
            dtype=np.float64,
            encoding=None,
        )
        if tab.dtype.names is None:
            raise ValueError(f"DataSource: could not read CSV header from {file_name!r}")
        names = list(tab.dtype.names)
        raw = np.stack([tab[n] for n in names], axis=1)
    else:
        raw = np.loadtxt(file_name, delimiter=",", dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        names = [str(i) for i in range(raw.shape[1])]

    usecols_spec = str2colindices(data_columns)
    data_indices = _usecols_to_indices(usecols_spec, names)

    if time_samples_as_column:
        rv = is_literal_eval_compatible(time_column)
        if isinstance(rv, int):
            time_col_idx = rv
        else:
            tc = str(time_column)
            if tc in names:
                time_col_idx = names.index(tc)
            elif header_as_first_row:
                time_col_idx = 0
            else:
                if tc.isdigit():
                    time_col_idx = int(tc)
                else:
                    raise ValueError(
                        f"DataSource: could not find time column {time_column!r} in {names}."
                    )
        times = np.asarray(raw[:, time_col_idx], dtype=np.float64)
        data_indices = [i for i in data_indices if i != time_col_idx]
    else:
        times = make_time_col(0.0, sampling_interval, raw.shape[0])

    if not data_indices:
        raise ValueError("DataSource: no data columns selected after removing time column.")
    data = raw[:, data_indices]
    return times, data


def load_csv(
    file_name: str,
    data_columns: str = "1",
    header_as_first_row: bool = False,
    sampling_interval: float = 1.0,
    time_column: str = "0",
    time_samples_as_column: bool = False,
):
    try:
        import pandas as pd
    except ImportError:
        return _load_csv_numpy(
            file_name,
            data_columns,
            header_as_first_row,
            sampling_interval,
            time_column,
            time_samples_as_column,
        )

    header = 0 if header_as_first_row else None
    usecols = str2colindices(data_columns)
    df = pd.read_csv(file_name, header=header, skipinitialspace=True, dtype=np.float64)

    if time_samples_as_column:
        rv = is_literal_eval_compatible(time_column)
        if isinstance(rv, int):
            index = df.columns[rv]
            time_col_idx = rv
        elif rv is None:
            time_column = str(time_column)
            if time_column not in df.columns:
                if header_as_first_row:
                    time_col_idx = 0
                    index = df.columns[0]
                else:
                    raise ValueError(
                        f"DataSource: could not find time column {time_column} in "
                        f"set of column names: {df.columns}."
                    )
            else:
                index = time_column
                time_col_idx = df.columns.get_loc(time_column)
    else:
        index = make_time_col(0.0, sampling_interval, df.shape[0])
        time_col_idx = None

    col_filter = df.columns.isin(usecols)
    if not any(col_filter):
        if len(usecols) == 1:
            if usecols[0] > len(df.columns) - 1:
                usecols = [len(df.columns) - 1]
        else:
            lwr = min(usecols)
            upr = max(usecols) + 1
            if upr > len(df.columns) + 1:
                upr = len(df.columns) + 1
            usecols = range(lwr, upr)

        col_filter = df.columns.isin(df.columns[usecols])

    df.set_index(index, inplace=True)
    if time_samples_as_column:
        col_filter = np.delete(col_filter, time_col_idx)

    df = df.loc[:, col_filter]

    times = np.array(df.index.to_numpy())
    data = np.array(df.to_numpy())

    return times, data


def load_npy(
    file_name: str,
    sampling_interval: float,
    time_samples_as_column: bool,
):
    arr = np.load(file_name, allow_pickle=False)
    if arr.ndim == 1:
        n = arr.shape[0]
        times = make_time_col(0.0, sampling_interval, n)
        data = arr.reshape(-1, 1)
        return times, data
    if arr.ndim != 2:
        raise ValueError(f"DataSource: .npy array must be 1-D or 2-D, got shape {arr.shape}")
    if time_samples_as_column:
        times = np.asarray(arr[:, 0], dtype=np.float64)
        data = np.asarray(arr[:, 1:], dtype=np.float64)
        if data.shape[1] == 0:
            raise ValueError("DataSource: .npy with time column needs at least two columns.")
        return times, data
    times = make_time_col(0.0, sampling_interval, arr.shape[0])
    data = np.asarray(arr, dtype=np.float64)
    return times, data


def load_npz(file_name: str, sampling_interval: float):
    z = np.load(file_name, allow_pickle=False)
    keys = sorted(z.files)
    if "time" in z.files and "y" in z.files:
        t = np.asarray(z["time"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return t, y
    if len(keys) < 2:
        raise ValueError(
            f"DataSource: .npz must contain 'time' and 'y' or at least two arrays; got {keys}"
        )
    t = np.asarray(z[keys[0]], dtype=np.float64).reshape(-1)
    y = np.asarray(z[keys[1]], dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if t.shape[0] != y.shape[0]:
        raise ValueError(
            f"DataSource: .npz time length {t.shape[0]} != data length {y.shape[0]}"
        )
    return t, y


def load_data_source_file(
    file_name: str,
    data_columns: str,
    header_as_first_row: bool,
    sampling_interval: float,
    time_column: str,
    time_samples_as_column: bool,
):
    path = str(file_name)
    lower = path.lower()
    if lower.endswith(".npy"):
        return load_npy(path, sampling_interval, time_samples_as_column)
    if lower.endswith(".npz"):
        return load_npz(path, sampling_interval)
    if lower.endswith(".csv"):
        return load_csv(
            path,
            data_columns,
            header_as_first_row,
            sampling_interval,
            time_column,
            time_samples_as_column,
        )
    # Backward compatible: paths without a recognized suffix are treated as CSV.
    return load_csv(
        path,
        data_columns,
        header_as_first_row,
        sampling_interval,
        time_column,
        time_samples_as_column,
    )


class DataSource(SourceBlock):
    """Produces outputs from an imported data file (.csv, .npy, .npz).

    CSV files are read with pandas when installed; otherwise NumPy is used.

    Parameters:
        file_name: Path to ``.csv``, ``.npy``, or ``.npz``.
        column: Optional. When set, selects the signal column(s) by name or index
            string and overrides ``data_columns`` for CSV loading. When ``None``,
            ``data_columns`` is used (default index ``"1"`` is the second column,
            i.e. first column is often time at index ``0``).
        time_column: For CSV with ``time_samples_as_column=True``, column name (e.g.
            ``"t"``) or index string (e.g. ``"0"``). If the name is missing but the
            file has a header row, the first column is used as time.
        data_columns: Column index, name, slice (e.g. ``3:8``), or list string for CSV.
        See module docstring for ``.npy`` / ``.npz`` layout.
    """

    @parameters(
        static=[
            "file_name",
            "data_columns",
            "column",
            "extrapolation",
            "header_as_first_row",
            "interpolation",
            "sampling_interval",
            "time_column",
            "time_samples_as_column",
        ]
    )
    def __init__(
        self,
        file_name: str,
        data_columns: str = "1",
        column: Optional[str] = None,
        extrapolation: str = "hold",
        header_as_first_row: bool = False,
        interpolation: str = "zero_order_hold",
        sampling_interval: float = 1.0,
        time_column: str = "0",
        time_samples_as_column: bool = False,
        **kwargs,
    ):
        kwargs.pop("data_integration_id", None)

        super().__init__(self._callback, **kwargs)

        effective_columns = str(column) if column is not None else str(data_columns)

        times, data = load_data_source_file(
            str(file_name),
            effective_columns,
            bool(header_as_first_row),
            float(sampling_interval),
            str(time_column),
            bool(time_samples_as_column),
        )

        times = npa.array(times)
        data = npa.array(data)

        if data.size == 0:
            raise ValueError(
                f"DataSource {self.name_path_strme} could not get the requested data columns."
            )

        max_i_zoh = len(times) - 1
        max_i_interp = max(len(times) - 2, 0)
        output_dim = data.shape[1]
        self._scalar_output = output_dim == 1

        def get_below_row_idx(time, max_i):
            time_clipped = npa.clip(time, times[0], times[-1])
            index = npa.searchsorted(times[: max_i + 1], time_clipped, side="right")
            return index - 1, time_clipped

        def _func_zoh(time):
            i, _ = get_below_row_idx(time, max_i_zoh)
            if extrapolation != "zero":
                return data[i, :]
            return npa.where(time > times[-1], npa.zeros(output_dim), data[i, :])

        def _func_interp(time):
            if len(times) < 2:
                return data[0, :]
            i, time_clipped = get_below_row_idx(time, max_i_interp)
            ap1 = data[i, :]
            ap2 = data[i + 1, :]
            if extrapolation != "zero":
                return (ap2 - ap1) / (times[i + 1] - times[i]) * (
                    time_clipped - times[i]
                ) + ap1

            return npa.where(
                time > times[-1],
                npa.zeros(output_dim),
                (ap2 - ap1) / (times[i + 1] - times[i]) * (time_clipped - times[i])
                + ap1,
            )

        def _wrap_func(_func):
            def _ds_wrapped_func(time):
                output = _func(time)
                return output[0]

            return _ds_wrapped_func

        if interpolation == "zero_order_hold":
            _func = _func_zoh
        else:
            _func = _func_interp

        if self._scalar_output:
            _func = _wrap_func(_func)

        self._func = npa.jit(_func)

    def _callback(self, time):
        return self._func(time)


class SimulationResultsSource(LeafSystem):
    """Replays one recorded trajectory from :class:`~jaxonomy.simulation.types.SimulationResults`.

    Output port ``y`` is the signal value at the current simulation time, using linear
    interpolation or zero-order hold. Values clamp to the first/last sample outside
    the recorded time range (``jnp.interp`` semantics for linear mode).
    """

    def __init__(
        self,
        results: "SimulationResults",
        signal_name: str,
        interpolation: str = "linear",
        **kwargs,
    ):
        from jaxonomy.simulation.types import SimulationResults as SR

        if not isinstance(results, SR):
            raise TypeError(f"results must be SimulationResults, got {type(results)}")
        if results.outputs is None:
            raise ValueError("SimulationResults.outputs is None; cannot replay.")
        if signal_name not in results.outputs:
            raise KeyError(f"signal_name {signal_name!r} not in results.outputs")

        super().__init__(**kwargs)

        if interpolation not in ("linear", "zero_order_hold"):
            raise ValueError(
                f"interpolation must be 'linear' or 'zero_order_hold', got {interpolation!r}"
            )

        t = np.asarray(results.time, dtype=np.float64).reshape(-1)
        y = np.asarray(results.outputs[signal_name], dtype=np.float64)
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError(
                f"Replayed signal {signal_name!r} must be 1-D per time step; got shape {y.shape}"
            )
        if t.shape[0] != y.shape[0]:
            raise ValueError(
                f"time length {t.shape[0]} != signal length {y.shape[0]} for {signal_name!r}"
            )

        self._t = npa.array(t, dtype=npa.float64)
        self._y = npa.array(y, dtype=npa.float64)

        def linear(time):
            return npa.interp(time, self._t, self._y)

        def zoh(time):
            if self._t.shape[0] == 0:
                return npa.array(0.0, dtype=self._y.dtype)
            if self._t.shape[0] == 1:
                return self._y[0]
            tc = npa.clip(time, self._t[0], self._t[-1])
            idx = npa.searchsorted(self._t, tc, side="right") - 1
            idx = npa.maximum(idx, 0)
            return self._y[idx]

        self._interp = linear if interpolation == "linear" else zoh
        self._jit_interp = npa.jit(self._interp)

        def _out_cb(time, state, *inputs, **parameters):
            return self._jit_interp(time)

        self.declare_output_port(
            _out_cb,
            name="y",
            prerequisites_of_calc=[DependencyTicket.time],
            requires_inputs=False,
        )
