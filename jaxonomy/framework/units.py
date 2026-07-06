# SPDX-License-Identifier: MIT

"""Lightweight SI dimensional algebra for port unit checking (T-104 phase 1).

This module provides a minimal, immutable :class:`Unit` value type that tracks
exponents over the seven SI base dimensions (mass, length, time, current,
temperature, amount, luminosity). Phase 1 of T-104 only requires:

* an algebra over units (multiply / divide / power / equality), so that
  composite units like ``meter * second / second == meter`` reduce correctly;
* a sentinel :data:`dimensionless` constant used as the default for any port
  that does not declare an explicit unit;
* an :func:`assert_unit_compatible` helper raising :class:`UnitMismatchError`
  when two ports declare incompatible units.

The module deliberately does *not* depend on ``pint``. The architecture
section of T-104 calls for ``pint`` as a possible Phase-2/3 backend behind
an optional extra; the Phase-1 work here is the in-tree algebra used by the
build-time consistency check inside
:class:`jaxonomy.framework.diagram_builder.DiagramBuilder.connect`.

Design notes:

* Units are immutable (``frozen=True``); no global mutable state, vmap-safe
  by virtue of being plain Python data, and trivially PyTree-irrelevant
  (units are metadata, never tensor values).
* :data:`dimensionless` connects to any other unit. This preserves the
  invariant that pre-existing ports (which never declared a unit) keep
  working unchanged.
* ``Unit`` carries an optional ``scale`` factor so that scalar prefixes
  (e.g. kilometres vs metres) do not silently equate to the base SI unit
  — Phase-2 will use this for warn-on-conversion behaviour. For now,
  identical dimensions with differing scales raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

from .error import StaticError

__all__ = [
    "Unit",
    "BusUnit",
    "UnitMismatchError",
    "assert_unit_compatible",
    "are_units_compatible",
    "assert_units_compatible_with_scale",
    "conversion_factor",
    "convert_offset_aware",
    "parse_unit",
    "resolve_unit",
    "dimensionless",
    "dimensionless_unit",
    # SI base units
    "kilogram",
    "meter",
    "second",
    "ampere",
    "kelvin",
    "mole",
    "candela",
    # Common derived units (only what's exercised by phase-1 tests / docs)
    "newton",
    "joule",
    "watt",
    "hertz",
    "radian",
    # Common scaled units (T-104 followup conversion)
    "kilometer",
    "millimeter",
    "millisecond",
    "minute",
    "hour",
    "gram",
    # Offset-aware temperature units (T-104 followup temperature conversion)
    "celsius",
    "fahrenheit",
    # Additional derived SI units (T-104 followup derived-units)
    "coulomb",
    "volt",
    "ohm",
    "farad",
    "weber",
    "henry",
    "pascal",
    "tesla",
    "derived_unit",
    # T-104-followup-currency-units: monetary units + FX conversion.
    "usd",
    "eur",
    "gbp",
    "jpy",
    "cad",
    "set_fx_rate",
    "get_fx_rate",
    "clear_fx_rates",
    "convert_currency",
    "CURRENCY_CODES",
]


# Indices into the SI base-dimension exponent tuple.
_DIM_NAMES: Tuple[str, ...] = (
    "kg",   # mass
    "m",    # length
    "s",    # time
    "A",    # current
    "K",    # temperature
    "mol",  # amount of substance
    "cd",   # luminous intensity
)

# ---------------------------------------------------------------------
# T-104-followup-currency-units: monetary "dimension" tuple.
# ---------------------------------------------------------------------
#
# Currency is modelled as an extra, independent set of exponents on
# :class:`Unit` (one axis per recognised currency code). We deliberately
# keep this OUT of the SI ``dims`` tuple — the SI 7-tuple shape is
# load-bearing across the existing test suite (every dims literal is a
# 7-element tuple) and changing it would be invasive.
#
# Instead, ``Unit.currency`` is a separate 5-tuple of integer exponents,
# one per code in :data:`CURRENCY_CODES`. The dataclass equality / hash
# / multiplicative algebra all participate in the currency axis the
# same way they participate in the SI axis. Two consequences:
#
#   * ``usd * meter`` is a perfectly valid composite unit ($/m, etc.),
#     mathematically distinct from ``meter``.
#   * ``usd == second`` is False (different currency / SI exponents),
#     so :func:`convert_currency` can refuse a USD → second conversion
#     loudly, the way the task spec demands.
#
# The default currency tuple is all-zero, which makes pre-existing
# non-currency units (``meter``, ``newton``, ``volt``, ...) byte-
# equivalent under the new equality definition: same SI dims, same
# scale, same offset, same all-zero currency.
CURRENCY_CODES: Tuple[str, ...] = (
    "USD",  # US dollar
    "EUR",  # euro
    "GBP",  # pound sterling
    "JPY",  # Japanese yen
    "CAD",  # Canadian dollar
)
_ZERO_CURRENCY: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)


class UnitMismatchError(StaticError):
    """Raised at diagram build time when two connected ports have
    incompatible units.

    Attributes are populated through :class:`StaticError` so the regular
    ``ErrorCollector`` / system-locator machinery still works.
    """


@dataclass(frozen=True, eq=False)
class Unit:
    """Immutable SI dimensional value.

    Units are compared by their dimension exponents and scale factor.
    The optional ``name`` is informational (used in error messages) and is
    not part of equality.
    """

    # Tuple of seven integers: exponents on (kg, m, s, A, K, mol, cd).
    dims: Tuple[int, int, int, int, int, int, int] = field(
        default=(0, 0, 0, 0, 0, 0, 0)
    )
    # Multiplicative scale on the base SI unit (1.0 == base).  Phase 1 only
    # supports unit / unit comparisons that are dimension-equal AND
    # scale-equal; conversions are deferred to Phase 2.
    scale: float = 1.0
    # Additive offset applied after the multiplicative ``scale`` when
    # mapping a raw value into base SI:
    #   physical_value_in_base_SI = scale * raw_value + offset
    # The vast majority of physical units (length, mass, time, etc.) have
    # offset == 0; the canonical exceptions are temperature scales
    # (Celsius, Fahrenheit). Unit multiplication / division for units
    # with a non-zero ``offset`` is not well-defined and raises.
    # (T-104 followup temperature conversion.)
    offset: float = 0.0
    # T-104-followup-currency-units: per-currency exponents.
    # Tuple of five integers (USD, EUR, GBP, JPY, CAD). Default all-zero,
    # which keeps every pre-existing :class:`Unit` instance byte-equal
    # to its previous form under the dataclass equality below. See the
    # module-level :data:`CURRENCY_CODES` for the canonical axis order.
    currency: Tuple[int, int, int, int, int] = field(
        default=_ZERO_CURRENCY
    )
    # Optional human-readable label, e.g. "m/s".  Not part of equality.
    name: str | None = None
    # T-104 phase 3: physical-quantity disambiguation.
    #
    # Two units with identical SI dimensions can describe different
    # physical quantities (the textbook example: ``N·m`` is both torque
    # AND energy; ``Pa·m^3`` is both work and pressure*volume). When set,
    # ``physical_quantity`` tags the *interpretation* so the
    # connect-time check refuses a torque-out → energy-in connection
    # even though their dims agree.
    #
    # Compatibility rule (in :func:`are_units_compatible`):
    #   * either side ``None`` → match (default-off byte-equivalence)
    #   * both set → must be string-equal
    #
    # Multiplicative algebra drops the tag (``torque * angular_velocity``
    # has no canonical interpretation; user must re-tag the result).
    # Default ``None`` keeps every pre-existing :class:`Unit` instance
    # byte-equal to its previous form.
    physical_quantity: str | None = None

    # ---- algebra -----------------------------------------------------

    def __post_init__(self):
        # Coerce dims into a 7-tuple of ints to make the dataclass robust
        # against being constructed from lists / numpy ints.
        if not isinstance(self.dims, tuple) or len(self.dims) != 7:
            object.__setattr__(self, "dims", tuple(int(d) for d in self.dims))
        # Same defensive coercion for the currency 5-tuple. We accept
        # any iterable of length 5 (lists, generators) so that callers
        # can spell ``currency=[1, 0, 0, 0, 0]`` without surprises.
        if (
            not isinstance(self.currency, tuple)
            or len(self.currency) != len(CURRENCY_CODES)
        ):
            coerced = tuple(int(c) for c in self.currency)
            if len(coerced) != len(CURRENCY_CODES):
                raise ValueError(
                    f"Unit.currency must have {len(CURRENCY_CODES)} entries "
                    f"(one per code in CURRENCY_CODES = {CURRENCY_CODES}); "
                    f"got {coerced!r}."
                )
            object.__setattr__(self, "currency", coerced)

    def _require_zero_offset_for_algebra(self, other: "Unit", op: str) -> None:
        """Raise a clear error when multiplying / dividing offsetted units.

        Offsetted (affine) units like Celsius / Fahrenheit do not form
        a multiplicative algebra: ``celsius * celsius`` is undefined,
        and ``celsius / second`` cannot be expressed as a scalar
        ``Unit`` either. We refuse the operation rather than silently
        returning a numerically-bogus result.
        """
        if self.offset != 0.0 or other.offset != 0.0:
            raise UnitMismatchError(
                f"Cannot {op} offsetted units: {self!r} {op} {other!r}. "
                "Affine units (e.g. Celsius, Fahrenheit) do not form a "
                "multiplicative algebra. Convert to a base unit "
                "(e.g. kelvin) via convert_offset_aware() first."
            )

    def __mul__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        self._require_zero_offset_for_algebra(other, "multiply")
        new_dims = tuple(a + b for a, b in zip(self.dims, other.dims))
        new_currency = tuple(
            a + b for a, b in zip(self.currency, other.currency)
        )
        return Unit(
            dims=new_dims,
            scale=self.scale * other.scale,
            currency=new_currency,
        )

    def __truediv__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        self._require_zero_offset_for_algebra(other, "divide")
        new_dims = tuple(a - b for a, b in zip(self.dims, other.dims))
        new_currency = tuple(
            a - b for a, b in zip(self.currency, other.currency)
        )
        return Unit(
            dims=new_dims,
            scale=self.scale / other.scale,
            currency=new_currency,
        )

    def __pow__(self, exponent: int) -> "Unit":
        if not isinstance(exponent, int):
            raise TypeError(
                f"Unit exponent must be int, got {type(exponent).__name__}"
            )
        if self.offset != 0.0 and exponent != 1:
            raise UnitMismatchError(
                f"Cannot raise offsetted unit {self!r} to a non-unity "
                "power. Affine units (e.g. Celsius, Fahrenheit) do not "
                "form a multiplicative algebra."
            )
        new_dims = tuple(d * exponent for d in self.dims)
        new_currency = tuple(c * exponent for c in self.currency)
        return Unit(
            dims=new_dims,
            scale=self.scale ** exponent,
            currency=new_currency,
        )

    # ---- equality / hashing -----------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        # T-104 phase 3: physical_quantity participates in equality so two
        # Units with the same dims but different physical interpretations
        # (e.g. N·m as torque vs. energy) compare distinct. ``name`` stays
        # informational and out of the equality contract.
        return (
            self.dims == other.dims
            and self.scale == other.scale
            and self.offset == other.offset
            and self.currency == other.currency
            and self.physical_quantity == other.physical_quantity
        )

    def __hash__(self) -> int:
        return hash((
            self.dims, self.scale, self.offset, self.currency,
            self.physical_quantity,
        ))

    # ---- dimensional helpers ----------------------------------------

    @property
    def is_dimensionless(self) -> bool:
        return (
            all(d == 0 for d in self.dims)
            and all(c == 0 for c in self.currency)
            and self.scale == 1.0
            and self.offset == 0.0
        )

    def same_dimension_as(self, other: "Unit") -> bool:
        """True if exponents match (ignoring scale).  Phase 1 doesn't use
        this for the connect check (which is strict-equal), but it's part
        of the public surface so Phase 2 can layer scalar-conversion
        warnings on top."""
        return (
            isinstance(other, Unit)
            and self.dims == other.dims
            and self.currency == other.currency
        )

    # ---- repr -------------------------------------------------------

    def __repr__(self) -> str:
        if self.name:
            return f"Unit({self.name!r})"
        if self.is_dimensionless:
            return "Unit(dimensionless)"
        parts = []
        for exp, label in zip(self.dims, _DIM_NAMES):
            if exp == 0:
                continue
            parts.append(label if exp == 1 else f"{label}^{exp}")
        # T-104-followup-currency-units: render non-zero currency
        # exponents alongside the SI dimensions so a $/m composite
        # prints as ``Unit(USD*m^-1)`` rather than collapsing.
        for exp, code in zip(self.currency, CURRENCY_CODES):
            if exp == 0:
                continue
            parts.append(code if exp == 1 else f"{code}^{exp}")
        body = "*".join(parts) if parts else "1"
        if self.scale != 1.0:
            body = f"{self.scale}*{body}"
        if self.offset != 0.0:
            sign = "+" if self.offset >= 0 else "-"
            body = f"{body}{sign}{abs(self.offset)}"
        if self.physical_quantity is not None:
            body = f"{body}@{self.physical_quantity}"
        return f"Unit({body})"

    # ---- T-104 phase 3: serialization + human-readable summary ----

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict representation of this Unit.

        Round-trips losslessly via :meth:`from_dict`. Keys are stable
        across versions; new optional fields are always added with
        defaults so older serialised forms continue to load.

        The default value for any field is omitted from the output for
        compactness — every legacy ``Unit()`` instance serialises to
        ``{}``.
        """
        out: dict = {}
        if self.dims != (0, 0, 0, 0, 0, 0, 0):
            out["dims"] = list(self.dims)
        if self.scale != 1.0:
            out["scale"] = self.scale
        if self.offset != 0.0:
            out["offset"] = self.offset
        if self.currency != _ZERO_CURRENCY:
            out["currency"] = list(self.currency)
        if self.name is not None:
            out["name"] = self.name
        if self.physical_quantity is not None:
            out["physical_quantity"] = self.physical_quantity
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Unit":
        """Construct a :class:`Unit` from a dict produced by
        :meth:`to_dict`. Missing keys take their dataclass defaults so
        the empty dict ``{}`` round-trips to ``Unit()``.
        """
        return cls(
            dims=tuple(data.get("dims", (0, 0, 0, 0, 0, 0, 0))),
            scale=float(data.get("scale", 1.0)),
            offset=float(data.get("offset", 0.0)),
            currency=tuple(data.get("currency", _ZERO_CURRENCY)),
            name=data.get("name"),
            physical_quantity=data.get("physical_quantity"),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise :meth:`to_dict` via :func:`json.dumps`.

        Args:
            indent: Optional JSON indent (default ``None`` for compact
                form; pass an int for pretty-printed output).
        """
        import json

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "Unit":
        """Inverse of :meth:`to_json`.

        Raises:
            ValueError: If ``json_str`` is not a JSON object.
        """
        import json

        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError(
                f"Unit.from_json: expected a JSON object at the top "
                f"level; got {type(data).__name__}."
            )
        return cls.from_dict(data)

    def summary(self) -> str:
        """Return a human-readable one-line summary of this Unit.

        Designed for ``print()`` / display contexts where ``repr(unit)``
        is too terse. Includes the dimension exponents (with SI labels),
        scale, offset, currency exponents, and the
        ``physical_quantity`` tag when set.
        """
        if self.is_dimensionless:
            base = "dimensionless"
        else:
            parts = []
            for exp, label in zip(self.dims, _DIM_NAMES):
                if exp != 0:
                    parts.append(label if exp == 1 else f"{label}^{exp}")
            for exp, code in zip(self.currency, CURRENCY_CODES):
                if exp != 0:
                    parts.append(code if exp == 1 else f"{code}^{exp}")
            base = " · ".join(parts) if parts else "1"
        bits = [base]
        if self.scale != 1.0:
            bits.append(f"scale={self.scale}")
        if self.offset != 0.0:
            bits.append(f"offset={self.offset}")
        if self.name is not None:
            bits.append(f'name="{self.name}"')
        if self.physical_quantity is not None:
            bits.append(f'physical_quantity="{self.physical_quantity}"')
        return "Unit(" + ", ".join(bits) + ")"


# ---------------------------------------------------------------------
# Canonical instances
# ---------------------------------------------------------------------


def _base(idx: int, label: str) -> Unit:
    dims = [0] * 7
    dims[idx] = 1
    return Unit(dims=tuple(dims), name=label)


# The default for any port that does not declare a unit.  Two names so
# callers can write either ``units=dimensionless`` or
# ``units=dimensionless_unit`` per the task spec.
dimensionless = Unit(name="dimensionless")
dimensionless_unit = dimensionless

kilogram = _base(0, "kg")
meter    = _base(1, "m")
second   = _base(2, "s")
ampere   = _base(3, "A")
kelvin   = _base(4, "K")
mole     = _base(5, "mol")
candela  = _base(6, "cd")

# Selected derived units (named for friendlier error messages).
newton = Unit(dims=(1, 1, -2, 0, 0, 0, 0), name="N")
joule  = Unit(dims=(1, 2, -2, 0, 0, 0, 0), name="J")
watt   = Unit(dims=(1, 2, -3, 0, 0, 0, 0), name="W")
hertz  = Unit(dims=(0, 0, -1, 0, 0, 0, 0), name="Hz")
# Radians are dimensionally dimensionless but tagged so error messages
# can carry the intended meaning.  Equality with `dimensionless` is True
# because they share the all-zero exponent tuple and unit scale.
radian = Unit(name="rad")

# ---------------------------------------------------------------------
# Additional derived SI units (T-104 followup derived-units)
# ---------------------------------------------------------------------
#
# Each derived unit below is defined by composition of the seven base
# SI dimensions; the corresponding ``name`` is the canonical SI symbol
# used in error messages and string parsing. All have ``scale=1.0``
# and ``offset=0.0`` (no offset-aware units in this batch — affine
# temperatures stay in their own block above).
#
# Dimensional decomposition (kg, m, s, A, K, mol, cd):
#
#   coulomb (C)  = A * s             → (0, 0,  1,  1, 0, 0, 0)
#   volt    (V)  = J / C = W / A     → (1, 2, -3, -1, 0, 0, 0)
#   ohm     (Ω)  = V / A             → (1, 2, -3, -2, 0, 0, 0)
#   farad   (F)  = C / V             → (-1, -2, 4, 2, 0, 0, 0)
#   weber   (Wb) = V * s             → (1, 2, -2, -1, 0, 0, 0)
#   henry   (H)  = Wb / A = V*s / A  → (1, 2, -2, -2, 0, 0, 0)
#   pascal  (Pa) = N / m^2           → (1, -1, -2, 0, 0, 0, 0)
#   tesla   (T)  = Wb / m^2          → (1, 0, -2, -1, 0, 0, 0)
coulomb = Unit(dims=(0, 0, 1, 1, 0, 0, 0),   name="C")
volt    = Unit(dims=(1, 2, -3, -1, 0, 0, 0), name="V")
ohm     = Unit(dims=(1, 2, -3, -2, 0, 0, 0), name="Ω")
farad   = Unit(dims=(-1, -2, 4, 2, 0, 0, 0), name="F")
weber   = Unit(dims=(1, 2, -2, -1, 0, 0, 0), name="Wb")
henry   = Unit(dims=(1, 2, -2, -2, 0, 0, 0), name="H")
pascal  = Unit(dims=(1, -1, -2, 0, 0, 0, 0), name="Pa")
tesla   = Unit(dims=(1, 0, -2, -1, 0, 0, 0), name="T")


# Selected scaled units used by T-104 followup conversion path.  Each
# shares its base-dimension exponents with an SI base unit but carries
# a non-unity ``scale`` factor (so ``kilometer.same_dimension_as(meter)``
# is True while ``kilometer == meter`` is False).
kilometer   = Unit(dims=meter.dims,  scale=1_000.0,  name="km")
millimeter  = Unit(dims=meter.dims,  scale=0.001,    name="mm")
millisecond = Unit(dims=second.dims, scale=0.001,    name="ms")
minute      = Unit(dims=second.dims, scale=60.0,     name="min")
hour        = Unit(dims=second.dims, scale=3_600.0,  name="hr")
gram        = Unit(dims=kilogram.dims, scale=0.001,  name="g")


# ---------------------------------------------------------------------
# Offset-aware temperature units (T-104 followup temperature conversion)
# ---------------------------------------------------------------------
#
# Convention used by this module:
#
#     physical_value_in_base_SI = scale * raw_value + offset
#
# i.e. the ``(scale, offset)`` pair describes the affine map *from* a
# raw reading in this unit *to* base SI (kelvin in the temperature
# case). With that convention:
#
#   * ``kelvin``    : scale = 1,         offset = 0           (base)
#   * ``celsius``   : scale = 1,         offset = +273.15
#       (raw_C →  raw_C + 273.15  K)
#   * ``fahrenheit``: scale = 5/9,       offset = (5/9)*459.67
#       (raw_F →  (5/9)*raw_F + (5/9)*459.67  K
#              =  (5/9)*(raw_F - 32) + 273.15  K)
#
# Quick checks (also covered by the test file):
#   * 0   °C   →  273.15  K              (offset added)
#   * 100 °C   →  373.15  K
#   * 32  °F   →  (5/9)*32 + (5/9)*459.67 = (5/9)*491.67 = 273.15 K ✓
#   * -459.67 °F → 0 K (absolute zero) ✓
#
# Round-tripping K → C → K is therefore exact under
# :func:`convert_offset_aware` (the offsets cancel).
celsius = Unit(
    dims=kelvin.dims,
    scale=1.0,
    offset=273.15,
    name="degC",
)
fahrenheit = Unit(
    dims=kelvin.dims,
    scale=5.0 / 9.0,
    offset=(5.0 / 9.0) * 459.67,
    name="degF",
)


# ---------------------------------------------------------------------
# T-104-followup-currency-units: monetary unit constants + FX helpers.
# ---------------------------------------------------------------------
#
# Each currency is its own axis under :attr:`Unit.currency`. The SI
# ``dims`` tuple stays all-zero, so a currency unit's *SI* dimension
# is dimensionless — but the currency axis exponent makes it
# arithmetically distinguishable from the SI dimensionless wildcard
# (``usd != dimensionless``), and from every other currency
# (``usd != eur``), so the build-time consistency check still refuses
# nonsense like wiring a USD port to a second-typed port.
#
# Conventions:
#   * ``Unit.scale = 1.0`` for every canonical currency constant.
#     Multi-currency conversion is done via the explicit FX table
#     (:func:`set_fx_rate` / :func:`convert_currency`), NOT via the
#     ``scale`` field — because FX rates are time-varying and
#     model-config-dependent, not properties of the unit itself.
#   * ``Unit.offset = 0.0`` (no affine map; currency is purely
#     multiplicative).
#   * The currency exponent for the named code is ``+1``; all other
#     currency axes are ``0``.
def _currency(code: str) -> Unit:
    """Build the canonical Unit for ``code`` (e.g. ``"USD"``)."""
    idx = CURRENCY_CODES.index(code)
    exps = [0] * len(CURRENCY_CODES)
    exps[idx] = 1
    return Unit(currency=tuple(exps), name=code)


usd = _currency("USD")
eur = _currency("EUR")
gbp = _currency("GBP")
jpy = _currency("JPY")
cad = _currency("CAD")


# Module-private FX rate table. Keys are ``(from_code, to_code)``
# string pairs, values are the multiplicative rate such that
# ``value_to = value_from * rate``. The table is intentionally
# mutable global state — FX rates are an environment fact, not a
# pure property of the unit system, and applications routinely refresh
# them at runtime (e.g. from a daily snapshot). Default-off
# byte-equivalence is preserved: no FX rate is consulted until a
# user-facing ``convert_currency`` call is made.
_FX_RATES: dict[tuple[str, str], float] = {}
# Reverse-direction keys that were auto-populated as ``1 / forward`` (as
# opposed to explicitly set by the user). Tracking these lets a re-set of
# an existing pair refresh the derived reverse instead of leaving it stale,
# while still preserving a user-provided asymmetric reverse (bid/ask).
_FX_AUTO_REVERSE: set[tuple[str, str]] = set()


def _canonical_currency_code(unit_or_code: "Unit | str") -> str:
    """Resolve a currency code from either a :class:`Unit` (which must
    have exactly one non-zero currency exponent, equal to ``+1``) or a
    bare string code (``"USD"`` or ``"usd"``).
    """
    if isinstance(unit_or_code, str):
        code = unit_or_code.strip().upper()
        if code not in CURRENCY_CODES:
            raise ValueError(
                f"Unknown currency code {unit_or_code!r}; "
                f"recognised codes are {CURRENCY_CODES}."
            )
        return code
    if not isinstance(unit_or_code, Unit):
        raise TypeError(
            "Expected a Unit instance or a currency-code string, got "
            f"{type(unit_or_code).__name__}: {unit_or_code!r}"
        )
    # Must look like a pure currency: all SI dims zero, scale 1, offset 0,
    # exactly one currency exponent equal to +1.
    if any(d != 0 for d in unit_or_code.dims):
        raise UnitMismatchError(
            f"Unit {unit_or_code!r} is not a pure currency unit "
            "(non-zero SI dimensions); cannot resolve a currency code."
        )
    if unit_or_code.offset != 0.0:
        raise UnitMismatchError(
            f"Unit {unit_or_code!r} has a non-zero offset and is not a "
            "currency unit."
        )
    nonzero = [
        (code, exp)
        for code, exp in zip(CURRENCY_CODES, unit_or_code.currency)
        if exp != 0
    ]
    if len(nonzero) != 1 or nonzero[0][1] != 1:
        raise UnitMismatchError(
            f"Unit {unit_or_code!r} is not a pure single-currency unit "
            "(must have exactly one currency axis with exponent +1); "
            f"got currency exponents {dict(zip(CURRENCY_CODES, unit_or_code.currency))!r}."
        )
    return nonzero[0][0]


def set_fx_rate(
    from_currency: "Unit | str",
    to_currency: "Unit | str",
    rate: float,
) -> None:
    """Record an FX rate so that one unit of ``from_currency`` equals
    ``rate`` units of ``to_currency``.

    Both directions are written: setting USD→EUR at 0.92 simultaneously
    sets EUR→USD at ``1.0 / 0.92`` so round-trips are exact under the
    floating-point reciprocal. A zero or non-finite ``rate`` is rejected
    (FX rates must be positive finite numbers).

    Args:
        from_currency: Source currency, either a :class:`Unit` (such as
            :data:`usd`) or a string code (``"USD"``).
        to_currency: Destination currency, ditto.
        rate: Strictly positive multiplicative conversion factor.

    Raises:
        ValueError: if ``rate`` is non-positive or non-finite.
        UnitMismatchError: if either argument is not a pure currency.

    Example:

        >>> set_fx_rate("USD", "EUR", 0.92)
        >>> get_fx_rate(usd, eur)
        0.92
        >>> # Self-rate is always 1.0 and need not be set explicitly.
        >>> get_fx_rate(usd, usd)
        1.0
    """
    src = _canonical_currency_code(from_currency)
    dst = _canonical_currency_code(to_currency)
    rate_f = float(rate)
    if not (rate_f > 0.0):
        raise ValueError(
            f"set_fx_rate({from_currency!r}, {to_currency!r}, {rate!r}): "
            "FX rate must be a strictly positive finite number."
        )
    # math.isinf check without importing math: rate_f != rate_f for NaN,
    # rate_f - rate_f == 0 is False for ±inf.
    if rate_f != rate_f or (rate_f - rate_f) != 0.0:
        raise ValueError(
            f"set_fx_rate({from_currency!r}, {to_currency!r}, {rate!r}): "
            "FX rate must be a finite number (got NaN or inf)."
        )
    _FX_RATES[(src, dst)] = rate_f
    # This direction is now explicitly user-set, so it is no longer a
    # candidate for auto-refresh from its own reverse.
    _FX_AUTO_REVERSE.discard((src, dst))
    # Auto-populate (or refresh) the reverse direction unless the user
    # already set an explicit (potentially asymmetric) reverse — common in
    # real markets when bid/ask spreads matter. Re-setting an existing pair
    # (e.g. a daily-snapshot refresh) must update the derived reverse too,
    # otherwise it goes stale and round-trips stop being exact.
    reverse = (dst, src)
    if reverse not in _FX_RATES or reverse in _FX_AUTO_REVERSE:
        _FX_RATES[reverse] = 1.0 / rate_f
        _FX_AUTO_REVERSE.add(reverse)


def get_fx_rate(
    from_currency: "Unit | str",
    to_currency: "Unit | str",
) -> float:
    """Return the previously-set FX rate from ``from_currency`` to
    ``to_currency``. Self-rates are always ``1.0`` even when unset.

    Raises:
        KeyError: if no rate has been set for the requested pair AND
            the two codes differ.
        UnitMismatchError: if either argument is not a pure currency.
    """
    src = _canonical_currency_code(from_currency)
    dst = _canonical_currency_code(to_currency)
    if src == dst:
        return 1.0
    try:
        return _FX_RATES[(src, dst)]
    except KeyError as e:
        raise KeyError(
            f"No FX rate registered for {src}->{dst}. "
            f"Call set_fx_rate({src!r}, {dst!r}, ...) first."
        ) from e


def clear_fx_rates() -> None:
    """Empty the FX rate table. Tests use this to keep their state
    isolated; production code should rarely need to call it.
    """
    _FX_RATES.clear()
    _FX_AUTO_REVERSE.clear()


def convert_currency(
    value,
    from_unit: "Unit | str",
    to_unit: "Unit | str",
):
    """Convert a numeric ``value`` carried in ``from_unit`` to the
    equivalent value in ``to_unit`` using the current FX rate table.

    Self-conversion (same currency on both sides) is a no-op and
    returns the value unchanged. Cross-currency conversion looks up
    the rate via :func:`get_fx_rate` and multiplies; a missing rate
    raises :class:`KeyError`.

    Args:
        value: Numeric value (Python scalar, NumPy array, JAX array).
            The helper only uses ``*``, so it composes transparently
            through ``jit`` / ``vmap`` / ``grad``.
        from_unit: Source currency, either a :class:`Unit` (such as
            :data:`usd`) or a string code (``"USD"``).
        to_unit: Destination currency, ditto.

    Returns:
        ``value * rate`` where ``rate = get_fx_rate(from_unit, to_unit)``.

    Raises:
        UnitMismatchError: if either argument carries non-currency
            dimensions (e.g. seconds), so the conversion is undefined.
        KeyError: if the relevant FX rate has not been registered.

    Example:

        >>> set_fx_rate("USD", "EUR", 0.92)
        >>> convert_currency(100.0, usd, eur)
        92.0
    """
    # Both sides must be pure currency units — otherwise it's a
    # mistake (a USD value cannot be converted to seconds). Reuse
    # the canonical-currency resolver so the error message names the
    # actual offending unit.
    src = _canonical_currency_code(from_unit)
    dst = _canonical_currency_code(to_unit)
    if src == dst:
        return value
    rate = _FX_RATES.get((src, dst))
    if rate is None:
        raise KeyError(
            f"No FX rate registered for {src}->{dst}. "
            f"Call set_fx_rate({src!r}, {dst!r}, ...) first."
        )
    return value * rate


# ---------------------------------------------------------------------
# T-104-followup-derived-units: define-once helper for composite units
# ---------------------------------------------------------------------


def derived_unit(
    name: str,
    symbol: str | None = None,
    components: "Unit | None" = None,
) -> "Unit":
    """Define a new derived :class:`Unit` from existing components.

    This is a convenience constructor for users who want to spell a
    composite unit once (with a friendly name) rather than recomposing
    its base components at every port declaration site. The returned
    :class:`Unit` has the same ``(dims, scale, offset)`` as
    ``components`` — it is therefore equal (under ``Unit.__eq__``) to
    any other unit with matching dimensions and scale — but carries
    a custom ``name`` for friendlier error messages and pprint output.

    Args:
        name: Long-form descriptive name (e.g. ``"my_torque"``).
            Used only when ``symbol`` is omitted.
        symbol: Short-form printable symbol (e.g. ``"τ"``). When
            provided, it overrides ``name`` as the Unit's display label.
        components: A :class:`Unit` expression describing the
            dimensions of the new unit (e.g. ``meter * newton``).
            Must not be ``None`` and must have ``offset == 0`` — affine
            units cannot be re-aliased this way.

    Returns:
        A fresh :class:`Unit` with ``components.dims`` /
        ``components.scale`` / ``components.offset`` and a ``name``
        set to ``symbol`` (when provided) or ``name``.

    Raises:
        TypeError: if ``components`` is not a :class:`Unit`.
        UnitMismatchError: if ``components`` has a non-zero offset
            (affine units cannot be aliased).

    Example:

        >>> from jaxonomy.framework.units import (
        ...     derived_unit, meter, newton,
        ... )
        >>> torque = derived_unit("torque", "N·m", meter * newton)
        >>> torque.dims == (1, 2, -2, 0, 0, 0, 0)
        True
        >>> torque == meter * newton
        True
    """
    if components is None or not isinstance(components, Unit):
        raise TypeError(
            "derived_unit(...) requires a `components` Unit expression, "
            f"got {type(components).__name__}: {components!r}"
        )
    if components.offset != 0.0:
        raise UnitMismatchError(
            f"derived_unit({name!r}, ...) cannot alias an offsetted "
            f"(affine) unit {components!r}; affine units do not form a "
            "multiplicative algebra and cannot be re-aliased."
        )
    label = symbol if symbol is not None else name
    return Unit(
        dims=components.dims,
        scale=components.scale,
        offset=components.offset,
        name=label,
    )


# ---------------------------------------------------------------------
# T-117-followup-bus-units: per-field compound unit for bus signals.
# ---------------------------------------------------------------------
#
# ``BusUnit`` is the unit-side counterpart of the NamedTuple-typed bus
# signal produced by :class:`jaxonomy.library.BusCreator`. It carries
# one :class:`Unit` per named field of the bus, so that
# :class:`BusSelector` can declare the *right* unit on its output port
# (the unit of the selected field, not of the whole bus).
#
# Design notes:
#
#   * ``BusUnit`` is intentionally NOT a subclass of :class:`Unit` —
#     buses do not participate in the multiplicative SI algebra
#     (``meter * bus`` is meaningless). Instead it is a distinct value
#     type, recognised explicitly by the connect-time compatibility
#     check below.
#   * Two ``BusUnit``s are compatible when the union of their field
#     names matches and each shared field's :class:`Unit` is
#     pair-wise compatible under the existing Phase-1 / followup
#     rules (``are_units_compatible``).
#   * A ``BusUnit`` connected to a port with no unit declared
#     (``None``) is always compatible — preserves the default-off
#     byte-equivalence with T-117-fu-bus-namedtuple.
#   * The ``fields`` mapping is normalised to an ordinary ``dict``
#     internally so that ordering is deterministic for error messages.
#     The class is frozen for hashability / immutability parity with
#     :class:`Unit`.


@dataclass(frozen=True, eq=False)
class BusUnit:
    """Compound unit carrying one :class:`Unit` per named bus field.

    Attached to the output port of a :class:`BusCreator` (and the
    matching input port of a :class:`BusSelector`) so that the
    connect-time consistency check can verify each field's unit
    individually.

    Attributes:
        fields: Mapping from bus field name to its :class:`Unit`.
            Stored as a plain ``dict`` (insertion order preserved).
    """

    fields: Mapping[str, Unit] = field(default_factory=dict)

    def __post_init__(self):
        # Normalise to a plain dict so the class is hashable and the
        # ordering is deterministic. Validate each entry is a Unit.
        normalised: dict[str, Unit] = {}
        for k, v in dict(self.fields).items():
            if not isinstance(k, str):
                raise TypeError(
                    f"BusUnit field name must be a str, got {type(k).__name__}: {k!r}"
                )
            if not isinstance(v, Unit):
                raise TypeError(
                    f"BusUnit field {k!r} must map to a Unit instance, "
                    f"got {type(v).__name__}: {v!r}"
                )
            normalised[k] = v
        object.__setattr__(self, "fields", normalised)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BusUnit):
            return NotImplemented
        if set(self.fields.keys()) != set(other.fields.keys()):
            return False
        return all(self.fields[k] == other.fields[k] for k in self.fields)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.fields.items(), key=lambda kv: kv[0])))

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.fields.items())
        return f"BusUnit({{{body}}})"

    def field_unit(self, name: str) -> Unit | None:
        """Return the :class:`Unit` for ``name``, or ``None`` if absent.

        Used by :class:`BusSelector` to look up its output-port unit
        when wired downstream of a unit-tagged bus.
        """
        return self.fields.get(name)


# ---------------------------------------------------------------------
# Compatibility check used by DiagramBuilder.connect
# ---------------------------------------------------------------------


def resolve_unit(unit: Unit | None) -> Unit:
    """Return ``unit`` if not None, otherwise the dimensionless sentinel.

    Phase 1 treats unit-less ports as inheriting / dimensionless so
    legacy diagrams that never call ``units=`` continue to work
    unchanged.
    """
    return unit if unit is not None else dimensionless


def are_units_compatible(
    src: Unit | BusUnit | None,
    dst: Unit | BusUnit | None,
) -> bool:
    """Return True if a connection from ``src`` to ``dst`` should be
    allowed under the Phase-1 rules:

    * Either side being ``None`` (unset) is always OK.
    * If both sides are :class:`BusUnit`, compatible iff every shared
      field's :class:`Unit` is pair-wise compatible AND the field sets
      match. A ``BusUnit`` on one side and ``None`` on the other is
      always OK (default-off byte-equivalence with the no-units bus).
    * Otherwise, both sides must be plain :class:`Unit`; either being
      :data:`dimensionless` is OK, else units must be equal.

    Scalar conversion (Phase 2) is layered on top by
    :func:`assert_units_compatible_with_scale` — see there.
    """
    # T-117-followup-bus-units: handle compound bus-unit case first.
    if isinstance(src, BusUnit) or isinstance(dst, BusUnit):
        # ``None`` wildcards match anything (preserves default-off
        # byte-equivalence with the unit-less BusCreator).
        if src is None or dst is None:
            return True
        if not (isinstance(src, BusUnit) and isinstance(dst, BusUnit)):
            # BusUnit cannot connect to a scalar Unit and vice-versa.
            return False
        if set(src.fields.keys()) != set(dst.fields.keys()):
            return False
        return all(
            are_units_compatible(src.fields[k], dst.fields[k])
            for k in src.fields
        )

    src_u = resolve_unit(src)
    dst_u = resolve_unit(dst)
    if src_u.is_dimensionless or dst_u.is_dimensionless:
        return True
    # T-104 phase 3: dims/scale/offset/currency must match, AND the
    # physical_quantity disambiguation tag must agree when both sides
    # carry one. One side ``None`` is treated as a wildcard so
    # legacy callers that never set the tag stay byte-equivalent.
    if not (
        src_u.dims == dst_u.dims
        and src_u.scale == dst_u.scale
        and src_u.offset == dst_u.offset
        and src_u.currency == dst_u.currency
    ):
        return False
    if src_u.physical_quantity is None or dst_u.physical_quantity is None:
        return True
    return src_u.physical_quantity == dst_u.physical_quantity


def assert_unit_compatible(
    src: Unit | BusUnit | None,
    dst: Unit | BusUnit | None,
    *,
    src_label: str = "source port",
    dst_label: str = "destination port",
) -> None:
    """Raise :class:`UnitMismatchError` if the two units are not
    Phase-1-compatible.  See :func:`are_units_compatible`.

    The labels are interpolated into the message so the caller (typically
    :meth:`DiagramBuilder.connect`) can name both ports.
    """
    if are_units_compatible(src, dst):
        return
    # T-117-followup-bus-units: render BusUnit and Unit cases distinctly
    # so the error message points at the actual mismatched value rather
    # than passing a BusUnit through ``resolve_unit`` (which would
    # collapse it to the dimensionless sentinel).
    src_repr = src if isinstance(src, BusUnit) else resolve_unit(src)
    dst_repr = dst if isinstance(dst, BusUnit) else resolve_unit(dst)
    raise UnitMismatchError(
        f"Unit mismatch: {src_label} has units {src_repr!r} but "
        f"{dst_label} has units {dst_repr!r}."
    )


# ---------------------------------------------------------------------
# T-104 followup: scalar-conversion helpers
# ---------------------------------------------------------------------


def conversion_factor(src: Unit | None, dst: Unit | None) -> float:
    """Return the multiplicative factor that converts a value carried in
    ``src`` units into the equivalent value in ``dst`` units.

    Both inputs are first resolved through :func:`resolve_unit`. The two
    units must share their base-dimension exponents — otherwise a
    :class:`UnitMismatchError` is raised. When either side is the
    dimensionless wildcard, the factor is ``1.0``.

    The factor is ``src.scale / dst.scale`` so that:

        value_in_dst = value_in_src * conversion_factor(src, dst)

    For instance, ``conversion_factor(meter, kilometer) == 1e-3`` and
    ``conversion_factor(kilometer, meter) == 1e3``.
    """
    src_u = resolve_unit(src)
    dst_u = resolve_unit(dst)
    # Wildcard: dimensionless on either side connects without conversion.
    if src_u.is_dimensionless or dst_u.is_dimensionless:
        return 1.0
    if src_u.dims != dst_u.dims:
        raise UnitMismatchError(
            f"Unit dimension mismatch: cannot convert {src_u!r} to "
            f"{dst_u!r} (different base-dim exponents)."
        )
    if src_u.currency != dst_u.currency:
        raise UnitMismatchError(
            f"Cannot express conversion {src_u!r} -> {dst_u!r} as a scalar "
            "factor: the units carry different currencies. A fixed unit "
            "factor cannot represent a cross-currency conversion (it needs "
            "a live FX rate) — use convert_currency(value, src, dst) "
            "together with set_fx_rate(...)."
        )
    # Scalar conversion factors are only well-defined for affine
    # units with a zero offset. Affine conversions (e.g. celsius
    # <-> kelvin) cannot be expressed as a single multiplicative
    # factor; callers must use :func:`convert_offset_aware` instead.
    if src_u.offset != 0.0 or dst_u.offset != 0.0:
        raise UnitMismatchError(
            f"Cannot express conversion {src_u!r} -> {dst_u!r} as a "
            "scalar factor: at least one unit has a non-zero offset "
            "(affine unit, e.g. Celsius / Fahrenheit). "
            "Use convert_offset_aware(src, dst, value) instead."
        )
    return src_u.scale / dst_u.scale


def convert_offset_aware(
    src: Unit | None,
    dst: Unit | None,
    value,
):
    """Convert a numeric ``value`` carried in ``src`` units into the
    equivalent value in ``dst`` units, supporting affine units with
    non-zero ``offset`` (Celsius, Fahrenheit, ...).

    The convention used throughout this module is

        physical_value_in_base_SI = scale * raw_value + offset

    so the value-space conversion is

        result = (value * src.scale + src.offset - dst.offset) / dst.scale

    The two units must share their base-dimension exponents — otherwise
    a :class:`UnitMismatchError` is raised. Wildcard (``None`` or
    :data:`dimensionless`) sides pass through unchanged.

    Examples:

        >>> convert_offset_aware(kelvin, celsius, 273.15)
        0.0
        >>> convert_offset_aware(celsius, fahrenheit, 100.0)
        212.0
        >>> # K -> C -> K round-trips exactly:
        >>> v = 300.0
        >>> convert_offset_aware(celsius, kelvin,
        ...     convert_offset_aware(kelvin, celsius, v)) == v
        True

    The helper accepts plain Python scalars, NumPy arrays, and JAX
    arrays equally — it only relies on ``+`` / ``-`` / ``*`` / ``/``
    on ``value``, so it composes through ``jit`` / ``vmap`` / ``grad``
    without further work.
    """
    src_u = resolve_unit(src)
    dst_u = resolve_unit(dst)
    # Wildcard: dimensionless on either side passes the value through.
    if src_u.is_dimensionless or dst_u.is_dimensionless:
        return value
    if src_u.dims != dst_u.dims:
        raise UnitMismatchError(
            f"Unit dimension mismatch: cannot convert {src_u!r} to "
            f"{dst_u!r} (different base-dim exponents)."
        )
    if src_u.currency != dst_u.currency:
        raise UnitMismatchError(
            f"Cannot convert {src_u!r} -> {dst_u!r}: the units carry "
            "different currencies. Cross-currency conversion needs a live "
            "FX rate — use convert_currency(value, src, dst) together with "
            "set_fx_rate(...)."
        )
    return (value * src_u.scale + src_u.offset - dst_u.offset) / dst_u.scale


def assert_units_compatible_with_scale(
    src: Unit | BusUnit | None,
    dst: Unit | BusUnit | None,
    *,
    src_label: str = "source port",
    dst_label: str = "destination port",
) -> float:
    """Like :func:`assert_unit_compatible` but tolerant of differing
    ``scale`` so long as the base-dim exponents match.  Returns the
    multiplicative conversion factor (``src.scale / dst.scale``) that
    callers can apply to the signal value.

    Raises :class:`UnitMismatchError` for genuine dimensional
    incompatibility (e.g. ``meter`` vs ``second``).

    This is the building block used by
    :meth:`DiagramBuilder.connect` under the new ``unit_conversion``
    flag (modes ``"auto"`` / ``"warn"``); diagrams that prefer the
    Phase-1 strict-equal behaviour can pass ``unit_conversion="error"``
    or call :func:`assert_unit_compatible` directly.
    """
    # T-117-followup-bus-units: bus signals route through the strict
    # equality path. We deliberately do not synthesise a per-field
    # scalar conversion factor here — bus signals carry NamedTuples
    # whose leaves are heterogeneous (different fields may even hold
    # different shapes), and applying a single multiplicative factor
    # would be ill-defined. Callers that need per-field rescaling
    # should pre-convert inside the BusCreator's upstream ports.
    if isinstance(src, BusUnit) or isinstance(dst, BusUnit):
        assert_unit_compatible(
            src, dst, src_label=src_label, dst_label=dst_label
        )
        return 1.0

    src_u = resolve_unit(src)
    dst_u = resolve_unit(dst)
    # Wildcards still pass with factor 1.
    if src_u.is_dimensionless or dst_u.is_dimensionless:
        return 1.0
    if src_u.dims != dst_u.dims:
        raise UnitMismatchError(
            f"Unit mismatch: {src_label} has units {src_u!r} but "
            f"{dst_label} has units {dst_u!r}."
        )
    if src_u.currency != dst_u.currency:
        raise UnitMismatchError(
            f"Currency mismatch: {src_label} has units {src_u!r} but "
            f"{dst_label} has units {dst_u!r}. A connection cannot apply a "
            "fixed scalar factor across currencies — convert explicitly "
            "with convert_currency()/set_fx_rate() upstream of the port."
        )
    # Scalar factor is only well-defined for non-offsetted units;
    # affine units (Celsius / Fahrenheit) need convert_offset_aware().
    if src_u.offset != 0.0 or dst_u.offset != 0.0:
        raise UnitMismatchError(
            f"Cannot convert {src_label} ({src_u!r}) to {dst_label} "
            f"({dst_u!r}) via a scalar factor: at least one side has a "
            "non-zero offset (affine unit). Use convert_offset_aware()."
        )
    return src_u.scale / dst_u.scale


# ---------------------------------------------------------------------
# T-104-followup-pint-bridge: optional pint-backed string parser
# ---------------------------------------------------------------------
#
# Goal: let users spell a unit as a string ("m/s²", "N·m", "kg·m/s²")
# and get back a Unit instance compatible with jaxonomy's hand-rolled
# algebra. Two backends:
#
#   * Preferred: ``pint`` (declared as an optional extra under
#     ``[units]`` in pyproject.toml). We read the dimensionality of
#     the pint Quantity (kg^a · m^b · s^c · A^d · K^e · mol^f · cd^g)
#     and translate it directly into a 7-tuple of integer exponents
#     for :class:`Unit`. Pint's scale factor for the parsed string is
#     also captured so that ``parse_unit("km")`` returns a unit with
#     ``scale=1000``, matching the hand-rolled :data:`kilometer`.
#   * Fallback: a small built-in lexer that recognises the curated
#     set of SI symbols already defined as module-level constants
#     (m, kg, s, A, K, mol, cd, N, J, W, Hz, rad, km, mm, ms, min,
#     hr, g, degC, degF). The fallback is intentionally limited;
#     anything beyond simple products / quotients / integer powers
#     is delegated to pint (or, if pint is absent, raises ValueError).
#
# Both backends agree on the byte-for-byte identity of the returned
# Unit's ``dims``, ``scale`` and ``offset`` for every string in the
# curated set, which is the part the connect-time consistency check
# actually inspects. The ``name`` attribute is informational and may
# differ between backends (we keep the canonical name from the
# hand-rolled table when we can identify the result, otherwise we
# echo the input string).


# Curated table of recognised symbols. Maps the bare symbol (no
# exponent, no scale prefix beyond what we already special-case) to
# the corresponding module-level Unit instance.  The fallback parser
# consults this table; the pint backend also uses it to recover a
# friendly ``name`` for known dimensional patterns.
_SYMBOL_TABLE: dict[str, "Unit"] = {
    # SI base units
    "m": meter,
    "kg": kilogram,
    "s": second,
    "A": ampere,
    "K": kelvin,
    "mol": mole,
    "cd": candela,
    # Selected derived units
    "N": newton,
    "J": joule,
    "W": watt,
    "Hz": hertz,
    "rad": radian,
    # T-104-followup-derived-units extras
    "C":  coulomb,
    "V":  volt,
    "Ω":  ohm,
    "ohm": ohm,
    "F":  farad,
    "Wb": weber,
    "H":  henry,
    "Pa": pascal,
    "T":  tesla,
    # Selected scaled units
    "km": kilometer,
    "mm": millimeter,
    "ms": millisecond,
    "min": minute,
    "hr": hour,
    "h": hour,
    "g": gram,
    # Offset-aware temperature units
    "degC": celsius,
    "degF": fahrenheit,
    "°C": celsius,
    "°F": fahrenheit,
    # Dimensionless / pure-number sentinels
    "1": dimensionless,
    "": dimensionless,
}


# Superscript unicode digits → ASCII int. We don't normalise the
# *whole* string; we only handle exponents (one or more contiguous
# superscript digits, optionally with a leading superscript minus).
_SUPERSCRIPT_DIGITS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3",
    "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
    "⁸": "8", "⁹": "9",
    "⁻": "-",  # superscript minus
    "⁺": "+",  # superscript plus
}


def _normalise_exponents(spec: str) -> str:
    """Replace inline superscript digits with ``**N`` exponent tokens.

    ``"m/s²"`` becomes ``"m/s**2"``; ``"m²·s⁻¹"`` becomes
    ``"m**2·s**-1"``. The function is stable on inputs that contain no
    superscripts.
    """
    out: list[str] = []
    i = 0
    n = len(spec)
    while i < n:
        ch = spec[i]
        if ch in _SUPERSCRIPT_DIGITS:
            # Greedy: consume a contiguous run of superscript chars.
            j = i
            digits: list[str] = []
            while j < n and spec[j] in _SUPERSCRIPT_DIGITS:
                digits.append(_SUPERSCRIPT_DIGITS[spec[j]])
                j += 1
            out.append("**")
            out.append("".join(digits))
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _tokenise_factor(token: str) -> tuple[str, int]:
    """Split ``"m**2"`` into ``("m", 2)``; ``"kg"`` into ``("kg", 1)``.

    Caret notation (``"m^2"``) is also accepted for friendliness.
    Whitespace is stripped. Raises ``ValueError`` on malformed input.
    """
    token = token.strip()
    if not token:
        return "", 1
    # Accept both ** and ^ as exponent separators.
    if "**" in token:
        head, _, tail = token.partition("**")
    elif "^" in token:
        head, _, tail = token.partition("^")
    else:
        return token, 1
    head = head.strip()
    tail = tail.strip()
    try:
        exp = int(tail)
    except ValueError as e:
        raise ValueError(
            f"Unit exponent must be an integer literal, got {tail!r}"
        ) from e
    return head, exp


def _parse_unit_builtin(spec: str) -> "Unit":
    """Built-in fallback parser. Handles the curated subset described
    in the module docstring; raises ``ValueError`` for anything else
    so callers can decide whether to surface the error or fall back
    further.
    """
    # Canonicalise exponents first so the rest of the parser only has
    # to deal with ASCII tokens of the form ``symbol[**int]``.
    s = _normalise_exponents(spec).strip()
    if not s:
        return dimensionless

    # Split on '/' into numerator / denominator halves. We accept
    # at most ONE '/' (the conventional "m/s", "kg/m**3" form);
    # nested quotients ("m/s/s") are rejected, but ``m/s**2`` works.
    if s.count("/") > 1:
        raise ValueError(
            f"Built-in parser cannot handle nested quotients in {spec!r}; "
            "install the 'pint' optional extra for richer unit strings."
        )
    if "/" in s:
        num_str, _, den_str = s.partition("/")
    else:
        num_str, den_str = s, ""

    # Fast path: a single curated symbol with no algebra at all.
    # Important for affine units (degC, degF) which don't participate
    # in the multiplicative algebra and would otherwise be rejected
    # by ``dimensionless * celsius`` below.
    bare = num_str.strip()
    if den_str == "" and bare in _SYMBOL_TABLE:
        return _SYMBOL_TABLE[bare]

    def _parse_product(part: str, sign: int) -> Unit:
        # Treat '·' (middle dot, U+00B7) as multiplication separator,
        # but ensure we don't break the ``**`` exponent token. We
        # temporarily replace ``**`` with a sentinel before splitting
        # on multiplication characters, then restore it.
        normalised = (
            part.replace("**", "\x00")
                .replace("·", "*")
                .replace("*", " ")
                .replace("\x00", "**")
        )
        if not normalised.strip():
            return dimensionless
        result = dimensionless
        for raw in normalised.split():
            tok = raw.strip()
            if not tok:
                continue
            symbol, exp = _tokenise_factor(tok)
            if symbol not in _SYMBOL_TABLE:
                raise ValueError(
                    f"Unknown unit symbol {symbol!r} in {spec!r}. "
                    "Either spell it using the curated set "
                    "(m, kg, s, A, K, mol, cd, N, J, W, Hz, rad, "
                    "km, mm, ms, min, hr, g, degC, degF) or install "
                    "the 'pint' optional extra for richer parsing."
                )
            base = _SYMBOL_TABLE[symbol]
            # Affine units (degC, degF) participate in algebra only at
            # exponent 1 and only on the numerator side. Anything else
            # is meaningless (and ``Unit.__mul__`` would refuse it).
            if base.offset != 0.0 and (sign != 1 or exp != 1):
                raise ValueError(
                    f"Affine unit {symbol!r} cannot be combined or raised to "
                    f"a non-unity power in {spec!r}; pass it on its own."
                )
            result = result * (base ** (exp * sign))
        return result

    num = _parse_product(num_str, +1)
    den = _parse_product(den_str, -1) if den_str else dimensionless
    return num * den


def _unit_from_dims_and_scale(
    dims: Tuple[int, int, int, int, int, int, int],
    scale: float,
    fallback_name: str,
) -> Unit:
    """Look up the curated module-level constant matching
    ``(dims, scale, offset=0)`` and return it (with its canonical
    name) if found; otherwise return a freshly constructed Unit
    tagged with ``fallback_name``.

    The lookup matters because the diagram-time consistency check
    compares Unit objects via dataclass equality on
    ``(dims, scale, offset)`` — the result is correct either way,
    but reusing the canonical constant means error messages render
    nicely (``Unit('N')`` rather than ``Unit('N*m/m')``).
    """
    for candidate in (
        meter, kilogram, second, ampere, kelvin, mole, candela,
        newton, joule, watt, hertz, radian,
        coulomb, volt, ohm, farad, weber, henry, pascal, tesla,
        kilometer, millimeter, millisecond, minute, hour, gram,
        dimensionless,
    ):
        if (
            candidate.dims == dims
            and candidate.scale == scale
            and candidate.offset == 0.0
        ):
            return candidate
    return Unit(dims=dims, scale=scale, name=fallback_name)


def _parse_unit_pint(spec: str) -> "Unit":
    """Pint-backed parser. Caller is responsible for verifying that
    pint is importable. Raises ``ValueError`` on parse failure or on
    a quantity whose base-SI dimensions don't reduce to integer
    exponents (which would mean pint's view of the unit doesn't
    match jaxonomy's integer-exponent algebra).
    """
    import pint  # local import: only reached when pint is installed

    # One UnitRegistry per call is wasteful; cache the registry the
    # first time we use it.
    registry = _get_pint_registry()
    try:
        # ``Quantity(1, spec)`` is the documented way to obtain a
        # 1-magnitude quantity with the parsed unit. We then convert
        # to base SI to read off the integer exponents.
        qty = registry.Quantity(1, spec)
        base = qty.to_base_units()
    except Exception as e:  # pint raises a zoo of exceptions
        raise ValueError(f"pint failed to parse unit {spec!r}: {e}") from e

    # Map pint's base-SI dimensionality to our 7-tuple.
    # The dimensions exposed by ``base.dimensionality`` use pint's
    # canonical names ([mass], [length], ...).
    dim_map = {
        "[mass]":              0,
        "[length]":            1,
        "[time]":              2,
        "[current]":           3,
        "[temperature]":       4,
        "[substance]":         5,
        "[luminosity]":        6,
    }
    exps = [0] * 7
    for dim_name, exp in base.dimensionality.items():
        if dim_name not in dim_map:
            # Unknown / extra dimension (e.g. [printing_unit] under
            # exotic pint contexts). Bail to the caller; the built-in
            # fallback will then be tried.
            raise ValueError(
                f"pint reports an unsupported dimension {dim_name!r} "
                f"while parsing {spec!r}."
            )
        if int(exp) != exp:
            raise ValueError(
                f"pint reports non-integer exponent {exp} for dimension "
                f"{dim_name!r} in {spec!r}; jaxonomy's Unit algebra "
                "only supports integer exponents."
            )
        exps[dim_map[dim_name]] = int(exp)
    dims = (exps[0], exps[1], exps[2], exps[3], exps[4], exps[5], exps[6])
    # ``magnitude`` after to_base_units() is the scalar prefix
    # relative to base SI (so 1 km becomes magnitude=1000 m).
    try:
        scale = float(base.magnitude)
    except Exception:
        scale = 1.0
    return _unit_from_dims_and_scale(dims, scale, fallback_name=spec)


# Module-level lazy cache for the pint registry. Constructing a
# ``UnitRegistry`` is moderately expensive (~50ms) and there's no
# reason to repeat it per ``parse_unit`` call.
_pint_registry = None


def _get_pint_registry():
    global _pint_registry
    if _pint_registry is None:
        import pint
        _pint_registry = pint.UnitRegistry()
    return _pint_registry


def parse_unit(spec: str) -> "Unit":
    """Parse a unit-string spec into a :class:`Unit`.

    The function prefers ``pint`` when it is installed (declared as
    an optional extra via ``pip install jaxonomy[units]``), and falls
    back to a small built-in parser covering the curated SI subset
    (``m``, ``kg``, ``s``, ``A``, ``K``, ``mol``, ``cd``, ``N``,
    ``J``, ``W``, ``Hz``, ``rad``, ``km``, ``mm``, ``ms``, ``min``,
    ``hr``, ``g``, ``degC``, ``degF``) when pint is absent.

    The returned :class:`Unit` is compatible with the rest of
    jaxonomy's hand-rolled unit algebra: its ``dims`` and ``scale``
    are equal (by ``Unit.__eq__``) to the corresponding constants
    when the spec matches one of them, e.g. ``parse_unit("N") ==
    newton`` and ``parse_unit("kg·m/s²") == newton``.

    Examples:

        >>> from jaxonomy.framework.units import parse_unit, newton, meter
        >>> parse_unit("m") == meter
        True
        >>> parse_unit("N·m").dims == (1, 2, -2, 0, 0, 0, 0)
        True
        >>> parse_unit("kg·m/s²") == newton
        True

    Args:
        spec: The unit-string to parse. Accepts ``*`` or ``·`` as
            multiplication separators, ``/`` as division, and either
            unicode superscript digits (``"m/s²"``) or the ASCII
            ``"m/s**2"`` / ``"m/s^2"`` notation for exponents.

    Returns:
        A :class:`Unit` instance with matching dimensions and scale.

    Raises:
        TypeError: if ``spec`` is not a string.
        ValueError: if neither backend can parse the spec.
    """
    if not isinstance(spec, str):
        raise TypeError(
            f"parse_unit() expects a string, got {type(spec).__name__}"
        )

    # Honest-fallback short-circuit for affine units: pint's
    # ``to_base_units()`` evaluates an offsetted Quantity at magnitude
    # 1 (so ``1 degC`` becomes ``274.15 kelvin``), which would map to
    # ``Unit(dims=K, scale=274.15)`` — not byte-equal to jaxonomy's
    # ``celsius`` (which carries ``scale=1.0, offset=273.15``). For
    # the curated affine names we therefore prefer the built-in
    # table directly; everything else goes through pint (if installed)
    # and falls back to the built-in parser on failure.
    stripped = spec.strip()
    if stripped in _SYMBOL_TABLE and _SYMBOL_TABLE[stripped].offset != 0.0:
        return _SYMBOL_TABLE[stripped]

    try:
        import pint  # noqa: F401
        have_pint = True
    except ImportError:
        have_pint = False

    if have_pint:
        try:
            return _parse_unit_pint(spec)
        except ValueError:
            # Fall through to the built-in parser; raise its error
            # (if any) so the user sees the friendlier message that
            # actually lists the supported symbols.
            pass

    return _parse_unit_builtin(spec)
