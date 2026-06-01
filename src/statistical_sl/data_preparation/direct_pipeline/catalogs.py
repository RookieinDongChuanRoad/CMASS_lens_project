"""Catalog readers for the direct source-to-canonical pipeline.

Catalog readers intentionally stop at :class:`BaseLensRecord`.  They may record
catalog sigma columns in provenance, but they must not produce trusted
``SigmaObservation`` objects.  That separation is the core defense against the
CMASS/slit case where the summary table contains numeric sigma columns that the
science workflow explicitly does not trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from statistical_sl.data_preparation.direct_pipeline.records import BaseLensRecord


CMASS_REQUIRED_COLUMNS = (
    "name",
    "zd",
    "zs",
    "rein_arcsec",
    "re_arcsec",
    "nser",
    "logmchab",
    "logmchab_err",
    "sigma",
    "sigma_err",
    "imag_Ser",
    "imag_deV",
    "reff_deV",
    "logmchab_deV",
)

SLACS_REQUIRED_COLUMNS = (
    "name",
    "RA",
    "dec",
    "zd",
    "zs",
    "Reff_arcsec",
    "Reff_kpc",
    "theta_Ein",
    "Rein_kpc",
    "lMstar_Chab",
    "lMstar_err",
    "veldisp(km/s)",
    "veldisp_err(km/s)",
)


@dataclass(frozen=True)
class CatalogProvenance:
    """Audit information produced while reading one source catalog.

    Parameters
    ----------
    source_path:
        Resolved path of the catalog file that was read.
    catalog_type:
        Logical reader type, for example ``cmass_summary_table`` or
        ``slacs_table``.
    row_count:
        Number of data rows converted into ``BaseLensRecord`` objects.
    ignored_columns:
        Columns intentionally ignored as trusted data.  CMASS summary-table
        sigma columns live here because their mere presence is not evidence of
        likelihood availability.
    available_measurement_columns:
        Columns that a later resolver may choose as a trusted measurement
        source, such as SLACS ``veldisp(km/s)``.
    extra:
        Small structured audit payload.  It is kept immutable to make reader
        outputs easier to reason about in tests and downstream stages.
    """

    source_path: Path
    catalog_type: str
    row_count: int
    ignored_columns: Mapping[str, str] = field(default_factory=dict)
    available_measurement_columns: Mapping[str, str] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize paths and freeze shallow mappings for audit stability."""

        object.__setattr__(self, "source_path", Path(self.source_path).expanduser().resolve())
        object.__setattr__(self, "ignored_columns", MappingProxyType(dict(self.ignored_columns)))
        object.__setattr__(
            self,
            "available_measurement_columns",
            MappingProxyType(dict(self.available_measurement_columns)),
        )
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


@dataclass(frozen=True)
class CatalogReadResult:
    """Result returned by a direct-pipeline catalog reader."""

    records: tuple[BaseLensRecord, ...]
    provenance: CatalogProvenance


def _read_commented_table(path: Path, required_columns: tuple[str, ...]) -> tuple[list[str], list[dict[str, str]]]:
    """Read a whitespace table with one commented header line.

    The project raw tables are simple enough that using the standard library is
    clearer than adding a pandas dependency.  This helper keeps validation
    strict: the header must name every required column and each data row must
    match the header width exactly.
    """

    resolved_path = Path(path).expanduser().resolve()
    header: list[str] | None = None
    rows: list[dict[str, str]] = []

    for line_number, raw_line in enumerate(resolved_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            candidate_header = stripped[1:].strip().split()
            if candidate_header:
                header = candidate_header
            continue
        if header is None:
            raise ValueError(f"{resolved_path}:{line_number} appears before a commented header row.")

        values = stripped.split()
        if len(values) != len(header):
            raise ValueError(
                f"{resolved_path}:{line_number} has {len(values)} columns but the header defines {len(header)}."
            )
        rows.append(dict(zip(header, values, strict=True)))

    if header is None:
        raise ValueError(f"{resolved_path} has no commented header row.")
    missing_columns = sorted(set(required_columns).difference(header))
    if missing_columns:
        raise ValueError(f"{resolved_path} is missing required columns: {', '.join(missing_columns)}")
    if not rows:
        raise ValueError(f"{resolved_path} contains no catalog rows.")
    return header, rows


def _float(row: Mapping[str, str], column: str, path: Path, lens_id: str) -> float:
    """Parse one numeric table value with a useful catalog-specific error."""

    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has non-numeric {column!r} for lens {lens_id}.") from exc


def _validate_unique_lens_ids(records: list[BaseLensRecord], path: Path) -> None:
    """Reject duplicate catalog identities before measurement joins."""

    seen: set[str] = set()
    for record in records:
        if record.lens_id in seen:
            raise ValueError(f"{path} contains duplicate lens_id {record.lens_id!r}.")
        seen.add(record.lens_id)


class CmassSummaryCatalogReader:
    """Read ``summary_table_deV.txt`` style catalogs as catalog-only records."""

    def __init__(self, catalog_path: Path | str, *, profile_name: str) -> None:
        """Store reader configuration without reading the file eagerly."""

        self.catalog_path = Path(catalog_path).expanduser().resolve()
        self.profile_name = str(profile_name).strip().lower()
        if not self.profile_name:
            raise ValueError("profile_name must be non-empty.")

    def read(self) -> CatalogReadResult:
        """Read CMASS summary rows and record catalog sigma as untrusted."""

        _, rows = _read_commented_table(self.catalog_path, CMASS_REQUIRED_COLUMNS)

        records: list[BaseLensRecord] = []
        untrusted_sigma_values: dict[str, dict[str, float]] = {}
        for row in rows:
            lens_id = row["name"].strip()
            if not lens_id:
                raise ValueError(f"{self.catalog_path} contains an empty lens name.")

            # For devauc builds, the summary table's deV radius and deV stellar
            # mass are the catalog facts used by downstream mass-grid builders.
            # The free-Sersic branch keeps the generic radius/mass columns.
            if self.profile_name == "devauc":
                effective_radius_arcsec = _float(row, "reff_deV", self.catalog_path, lens_id)
                log_stellar_mass = _float(row, "logmchab_deV", self.catalog_path, lens_id)
                sersic_index = 4.0
            else:
                effective_radius_arcsec = _float(row, "re_arcsec", self.catalog_path, lens_id)
                log_stellar_mass = _float(row, "logmchab", self.catalog_path, lens_id)
                sersic_index = _float(row, "nser", self.catalog_path, lens_id)

            records.append(
                BaseLensRecord(
                    lens_id=lens_id,
                    z_lens=_float(row, "zd", self.catalog_path, lens_id),
                    z_source=_float(row, "zs", self.catalog_path, lens_id),
                    theta_ein_arcsec=_float(row, "rein_arcsec", self.catalog_path, lens_id),
                    effective_radius_arcsec=effective_radius_arcsec,
                    log_stellar_mass=log_stellar_mass,
                    log_stellar_mass_err=_float(row, "logmchab_err", self.catalog_path, lens_id),
                    profile_name=self.profile_name,
                    sersic_index=sersic_index,
                    source_metadata={
                        "catalog_type": "cmass_summary_table",
                        "imag_Ser": _float(row, "imag_Ser", self.catalog_path, lens_id),
                        "imag_deV": _float(row, "imag_deV", self.catalog_path, lens_id),
                    },
                )
            )
            untrusted_sigma_values[lens_id] = {
                "sigma": _float(row, "sigma", self.catalog_path, lens_id),
                "sigma_err": _float(row, "sigma_err", self.catalog_path, lens_id),
            }

        _validate_unique_lens_ids(records, self.catalog_path)
        return CatalogReadResult(
            records=tuple(records),
            provenance=CatalogProvenance(
                source_path=self.catalog_path,
                catalog_type="cmass_summary_table",
                row_count=len(records),
                ignored_columns={
                    "sigma": "untrusted_catalog_value",
                    "sigma_err": "untrusted_catalog_value",
                },
                extra={"untrusted_sigma_values": untrusted_sigma_values},
            ),
        )


class SlacsTableCatalogReader:
    """Read ``SLACS_table.cat`` rows as lens records plus catalog sigma audit."""

    def __init__(self, catalog_path: Path | str, *, profile_name: str) -> None:
        """Store reader configuration without reading the file eagerly."""

        self.catalog_path = Path(catalog_path).expanduser().resolve()
        self.profile_name = str(profile_name).strip().lower()
        if not self.profile_name:
            raise ValueError("profile_name must be non-empty.")

    def read(self) -> CatalogReadResult:
        """Read SLACS catalog rows without directly creating observations."""

        _, rows = _read_commented_table(self.catalog_path, SLACS_REQUIRED_COLUMNS)

        records: list[BaseLensRecord] = []
        catalog_sigma_values: dict[str, dict[str, float]] = {}
        for row in rows:
            lens_id = row["name"].strip()
            if not lens_id:
                raise ValueError(f"{self.catalog_path} contains an empty lens name.")

            records.append(
                BaseLensRecord(
                    lens_id=lens_id,
                    z_lens=_float(row, "zd", self.catalog_path, lens_id),
                    z_source=_float(row, "zs", self.catalog_path, lens_id),
                    theta_ein_arcsec=_float(row, "theta_Ein", self.catalog_path, lens_id),
                    theta_ein_kpc=_float(row, "Rein_kpc", self.catalog_path, lens_id),
                    effective_radius_arcsec=_float(row, "Reff_arcsec", self.catalog_path, lens_id),
                    effective_radius_kpc=_float(row, "Reff_kpc", self.catalog_path, lens_id),
                    log_stellar_mass=_float(row, "lMstar_Chab", self.catalog_path, lens_id),
                    log_stellar_mass_err=_float(row, "lMstar_err", self.catalog_path, lens_id),
                    profile_name=self.profile_name,
                    sersic_index=4.0 if self.profile_name == "devauc" else None,
                    ra_deg=_float(row, "RA", self.catalog_path, lens_id),
                    dec_deg=_float(row, "dec", self.catalog_path, lens_id),
                    source_metadata={"catalog_type": "slacs_table"},
                )
            )
            catalog_sigma_values[lens_id] = {
                "sigma_kms": _float(row, "veldisp(km/s)", self.catalog_path, lens_id),
                "sigma_err_kms": _float(row, "veldisp_err(km/s)", self.catalog_path, lens_id),
            }

        _validate_unique_lens_ids(records, self.catalog_path)
        return CatalogReadResult(
            records=tuple(records),
            provenance=CatalogProvenance(
                source_path=self.catalog_path,
                catalog_type="slacs_table",
                row_count=len(records),
                available_measurement_columns={
                    "sigma_kms": "veldisp(km/s)",
                    "sigma_err_kms": "veldisp_err(km/s)",
                },
                extra={"catalog_sigma_values": catalog_sigma_values},
            ),
        )


__all__ = [
    "CatalogProvenance",
    "CatalogReadResult",
    "CmassSummaryCatalogReader",
    "SlacsTableCatalogReader",
]
