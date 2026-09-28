"""Full-runcard POPxf regressions against reviewed reference outputs."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from validphys.utils import yaml_safe

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests/regression/popxf"
RUNCARD = ROOT / "runcards/simufac_to_popxf.yaml"
DATASET = "ATLAS_Z0_7TEV_49FB_HIMASS"
# Allow numerical backend/platform rounding while detecting physics changes.
RTOL, ATOL = 1e-6, 1e-9


def read_outputs(directory):
    return (
        json.loads((directory / f"{DATASET}.json").read_text()),
        json.loads((directory / f"{DATASET}_measurement.json").read_text()),
    )


def measurement(document):
    return document[DATASET][0]


def covariance(document):
    entry = measurement(document)
    std = np.asarray(entry["standard_deviation"])
    return np.asarray(entry["correlation"]) * np.outer(std, std)


def assert_close(actual, expected):
    assert np.shape(actual) == np.shape(expected)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def run_card(work, card):
    work.mkdir()
    env = dict(os.environ, MPLBACKEND="Agg", BROWSER="true")
    result = subprocess.run(
        [sys.executable, "-m", "simunet.app", str(card)], cwd=work,
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=1200,
    )
    (work / "run.log").write_text(result.stdout)
    assert result.returncode == 0, f"Runcard {card} failed:\n{result.stdout}"
    outputs = work / "likelihood_files/CT18NNLO"
    assert {p.name for p in outputs.glob("*.json")} == {
        f"{DATASET}.json", f"{DATASET}_measurement.json"
    }
    return read_outputs(outputs)


@pytest.fixture(scope="module")
def reference():
    return read_outputs(REFERENCE)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return run_card(tmp_path_factory.mktemp("popxf") / "baseline", RUNCARD)


@pytest.fixture(scope="module")
def with_covariances(tmp_path_factory, reference):
    work = tmp_path_factory.mktemp("popxf-covariances")
    n = len(measurement(reference[1])["central_value"])
    v = np.linspace(-0.3, 0.7, n)
    # PSD additions with off-diagonal entries test correlations as well as errors.
    additions = [np.diag(np.linspace(0.2, 1.0, n)), np.outer(v, v)]
    index = pd.MultiIndex.from_tuples(
        [("ALL", DATASET, i) for i in range(n)], names=["group", "dataset", "id"]
    )
    paths = []
    for i, extra in enumerate(additions):
        path = work / f"extra_covariance_{i}.csv"
        pd.DataFrame(extra, index=index, columns=index).to_csv(path, sep="\t")
        paths.append(str(path))
    card = yaml_safe.load(RUNCARD.read_text())
    card["covmat_paths"] = paths
    augmented_card = work / "with_covariances.yaml"
    with augmented_card.open("w") as stream:
        yaml_safe.dump(card, stream)
    return run_card(work / "output", augmented_card), sum(additions)


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


def test_covariance_additions(baseline, with_covariances):
    augmented, extra = with_covariances
    # Additional uncertainty must not change the model, central values or ordering.
    assert augmented[0] == baseline[0]
    actual, base = measurement(augmented[1]), measurement(baseline[1])
    assert actual["observables"] == base["observables"]
    assert_close(actual["central_value"], base["central_value"])
    expected_cov = covariance(baseline[1]) + extra
    expected_std = np.sqrt(np.diag(expected_cov))
    assert_close(actual["standard_deviation"], expected_std)
    assert_close(actual["correlation"], expected_cov / np.outer(expected_std, expected_std))
    assert_close(covariance(augmented[1]), expected_cov)


@pytest.mark.parametrize("variant", ["baseline", "with_covariances"])
def test_valid_covariance_and_finite_outputs(request, variant):
    outputs = request.getfixturevalue(variant)
    pop, pdf = outputs if variant == "baseline" else outputs[0]
    entry = measurement(pdf)
    std = np.asarray(entry["standard_deviation"])
    corr = np.asarray(entry["correlation"])
    assert np.all(std > 0)
    assert_close(corr, corr.T)
    assert_close(np.diag(corr), np.ones(len(std)))
    assert np.all(np.abs(corr) <= 1 + ATOL)
    assert np.linalg.eigvalsh(corr).min() >= -ATOL
    for values in [entry["central_value"], std, corr,
                   *pop["data"]["observable_central"].values()]:
        assert np.isfinite(values).all()
