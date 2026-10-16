"""Full-runcard POPxf regressions against reference outputs."""

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from validphys.utils import yaml_safe

from helper import run_simunet

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests/regression/popxf"
RUNCARD = ROOT / "runcards/simufac_to_popxf.yaml"
DATASET = "ATLAS_Z0_7TEV_49FB_HIMASS"

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
    """
    Check the shape and values of the outputted measurement against the reference.
    """
    assert_close(measurement(baseline[1])[field], measurement(reference[1])[field])


def test_export_metadata(baseline, reference):
    """
    Check the outputted metadata and observable names against the reference.
    """
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
    """
    Check the exported observable-central coefficients against the reference.
    """
    actual = baseline[0]["data"]["observable_central"]
    expected = reference[0]["data"]["observable_central"]
    assert set(actual) == set(expected)
    for name, coefficients in expected.items():
        assert_close(actual[name], coefficients)
