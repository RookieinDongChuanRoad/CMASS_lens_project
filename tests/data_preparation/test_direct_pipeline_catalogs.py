"""Tests for direct-pipeline catalog readers.

Catalog readers own only source-catalog facts.  They must not create trusted
velocity-dispersion observations unless a later measurement source explicitly
chooses catalog columns as the trusted source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from statistical_sl.data_preparation.direct_pipeline.catalogs import (
    CmassSummaryCatalogReader,
    SlacsTableCatalogReader,
)


CMASS_HEADER = (
    "# name zd zs rein_arcsec re_arcsec nser logmchab logmchab_err "
    "sigma sigma_err imag_Ser imag_deV reff_deV logmchab_deV\n"
)

SLACS_HEADER = (
    "# name RA dec zd zs Reff_arcsec Reff_kpc theta_Ein Rein_kpc "
    "lMstar_Chab lMstar_err veldisp(km/s) veldisp_err(km/s)\n"
)


def _write_text(path: Path, content: str) -> Path:
    """Write a small ASCII-table fixture and return its path."""

    path.write_text(content, encoding="utf-8")
    return path


def test_cmass_summary_reader_ignores_untrusted_sigma_columns(tmp_path: Path) -> None:
    """CMASS summary-table sigma columns are provenance, not observations."""

    catalog_path = _write_text(
        tmp_path / "summary_table_deV.txt",
        CMASS_HEADER
        + "015618-010747 0.542 1.167 0.841 3.358 8.443 11.722 0.069 "
        + "299 38 18.76 19.38 1.00 11.47\n",
    )

    result = CmassSummaryCatalogReader(catalog_path, profile_name="devauc").read()

    assert len(result.records) == 1
    record = result.records[0]
    assert record.lens_id == "015618-010747"
    assert record.z_lens == pytest.approx(0.542)
    assert record.z_source == pytest.approx(1.167)
    assert record.theta_ein_arcsec == pytest.approx(0.841)
    assert record.effective_radius_arcsec == pytest.approx(1.00)
    assert record.log_stellar_mass == pytest.approx(11.47)
    assert record.profile_name == "devauc"

    assert result.provenance.source_path == catalog_path.resolve()
    assert result.provenance.ignored_columns["sigma"] == "untrusted_catalog_value"
    assert result.provenance.ignored_columns["sigma_err"] == "untrusted_catalog_value"
    assert result.provenance.row_count == 1
    assert result.provenance.extra["untrusted_sigma_values"]["015618-010747"]["sigma"] == pytest.approx(299)
    assert result.provenance.extra["untrusted_sigma_values"]["015618-010747"]["sigma_err"] == pytest.approx(38)


def test_cmass_summary_reader_rejects_duplicate_lens_ids(tmp_path: Path) -> None:
    """Catalog identity must be unique before joining external measurements."""

    catalog_path = _write_text(
        tmp_path / "summary_table_deV.txt",
        CMASS_HEADER
        + "lens-a 0.5 1.1 0.8 3.0 4.0 11.7 0.1 250 20 18.0 18.5 1.0 11.5\n"
        + "lens-a 0.6 1.2 0.9 3.1 4.1 11.8 0.1 251 21 18.1 18.6 1.1 11.6\n",
    )

    with pytest.raises(ValueError, match="duplicate"):
        CmassSummaryCatalogReader(catalog_path, profile_name="devauc").read()


def test_slacs_table_reader_records_catalog_sigma_columns_for_possible_later_use(tmp_path: Path) -> None:
    """SLACS catalog sigma stays in provenance until the resolver trusts it."""

    catalog_path = _write_text(
        tmp_path / "SLACS_table.cat",
        SLACS_HEADER
        + "SDSSJ0029-0055 7.282417 -0.930694 0.227 0.931 2.30 8.36 "
        + "0.96 3.48 11.33 0.13 229 18\n",
    )

    result = SlacsTableCatalogReader(catalog_path, profile_name="devauc").read()

    assert len(result.records) == 1
    record = result.records[0]
    assert record.lens_id == "SDSSJ0029-0055"
    assert record.ra_deg == pytest.approx(7.282417)
    assert record.dec_deg == pytest.approx(-0.930694)
    assert record.z_lens == pytest.approx(0.227)
    assert record.z_source == pytest.approx(0.931)
    assert record.effective_radius_arcsec == pytest.approx(2.30)
    assert record.effective_radius_kpc == pytest.approx(8.36)
    assert record.theta_ein_arcsec == pytest.approx(0.96)
    assert record.theta_ein_kpc == pytest.approx(3.48)
    assert record.log_stellar_mass == pytest.approx(11.33)
    assert record.log_stellar_mass_err == pytest.approx(0.13)

    assert result.provenance.available_measurement_columns == {
        "sigma_kms": "veldisp(km/s)",
        "sigma_err_kms": "veldisp_err(km/s)",
    }
    assert result.provenance.extra["catalog_sigma_values"]["SDSSJ0029-0055"]["sigma_kms"] == pytest.approx(229)
    assert result.provenance.extra["catalog_sigma_values"]["SDSSJ0029-0055"]["sigma_err_kms"] == pytest.approx(18)
