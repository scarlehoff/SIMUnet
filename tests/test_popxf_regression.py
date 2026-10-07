"""Full-runcard POPxf regressions against reviewed reference outputs."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from validphys.utils import yaml_safe

from helper import run_simunet

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests/regression/popxf"
RUNCARD = ROOT / "runcards/simufac_to_popxf.yaml"
DATASET = "ATLAS_Z0_7TEV_49FB_HIMASS"
# Allow numerical backend/platform rounding while detecting physics changes.
RTOL, ATOL = 1e-6, 1e-9


def read_outputs(directory):
    """Load the POPxf prediction and measurement documents."""
    return (
        json.loads((directory / f"{DATASET}.json").read_text()),
        json.loads((directory / f"{DATASET}_measurement.json").read_text()),
    )


def measurement(document):
    return document[DATASET][0]


def assert_close(actual, expected):
    assert np.shape(actual) == np.shape(expected)
    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.fixture(scope="module")
def reference():
    return read_outputs(REFERENCE)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """Run the example runcard once for all output checks."""
    work = tmp_path_factory.mktemp("popxf") / "baseline"
    run_simunet(RUNCARD, cwd=work)

    card = yaml_safe.load(RUNCARD.read_text())
    pdf = card["pdf"]
    pdf_id = pdf["id"] if isinstance(pdf, dict) else pdf
    outputs = work / "likelihood_files" / pdf_id
    assert {path.name for path in outputs.glob("*.json")} == {
        f"{DATASET}.json",
        f"{DATASET}_measurement.json",
    }
    return read_outputs(outputs)


@pytest.mark.parametrize("field", ["central_value", "standard_deviation", "correlation"])
def test_measurement_regression(baseline, reference, field):
    assert_close(measurement(baseline[1])[field], measurement(reference[1])[field])


def test_export_metadata(baseline, reference):
    pop, pdf = baseline
    ref_pop, ref_pdf = reference
    assert pop["$schema"] == ref_pop["$schema"]
    assert pop["metadata"] == ref_pop["metadata"]
    assert pdf["$schema"] == ref_pdf["$schema"]
    assert set(pdf) == set(ref_pdf)
    assert len(pdf[DATASET]) == 1
    actual, expected = measurement(pdf), measurement(ref_pdf)
    assert actual["observables"] == expected["observables"]
    assert actual["distribution_type"] == expected["distribution_type"]
    assert pop["metadata"]["observable_names"] == actual["observables"]


def test_popxf_coefficients(baseline, reference):
    actual = baseline[0]["data"]["observable_central"]
    expected = reference[0]["data"]["observable_central"]
    assert set(actual) == set(expected)
    for name, coefficients in expected.items():
        assert_close(actual[name], coefficients)


def test_example_covariance(baseline):
    card = yaml_safe.load(RUNCARD.read_text())
    assert card["covmat_paths"]
    n = len(measurement(baseline[1])["central_value"])
    expected_index = pd.MultiIndex.from_tuples(
        [("ALL", DATASET, i) for i in range(n)], names=["group", "dataset", "id"]
    )
    extra = np.zeros((n, n))
    for path in card["covmat_paths"]:
        frame = pd.read_csv(ROOT / path, sep="\t", index_col=[0, 1, 2], header=[0, 1, 2])
        assert frame.shape == (n, n)
        pd.testing.assert_index_equal(frame.index, expected_index)
        # CSV headers are strings, while row bin IDs are parsed as integers.
        assert list(frame.columns) == [(g, d, str(i)) for g, d, i in expected_index]
        values = frame.to_numpy(dtype=float)
        assert_close(values, values.T)
        assert np.linalg.eigvalsh(values).min() >= -ATOL
        extra += values
    assert np.any(extra - np.diag(np.diag(extra)))


def test_valid_covariance_and_finite_outputs(baseline):
    pop, pdf = baseline
    entry = measurement(pdf)
    std = np.asarray(entry["standard_deviation"])
    corr = np.asarray(entry["correlation"])
    assert np.all(std > 0)
    assert_close(corr, corr.T)
    assert_close(np.diag(corr), np.ones(len(std)))
    assert np.all(np.abs(corr) <= 1 + ATOL)
    assert np.linalg.eigvalsh(corr).min() >= -ATOL
    for values in [
        entry["central_value"],
        std,
        corr,
        *pop["data"]["observable_central"].values(),
    ]:
        assert np.isfinite(values).all()
