"""Setupfit coverage checks with runcards and on-disk SIMU data."""

from io import StringIO
from types import SimpleNamespace

import pytest
from reportengine.configparser import ConfigError
from validphys.utils import yaml_safe

from simunet.loader import SIMUnetLoader
from simunet.simufit.simusetupfit_exec import SimuSetupConfig


def setup_config(tmp_path, datasets, parameters, simu_data):
    folder = tmp_path / "simu_factors"
    folder.mkdir()
    for name, factors in simu_data.items():
        with (folder / f"SIMU_{name}.yaml").open("w") as stream:
            yaml_safe.dump(factors, stream)
    loader = object.__new__(SIMUnetLoader)
    loader.simudata_path = tmp_path
    card = {"dataset_inputs": datasets, "simu_parameters": parameters,
            "allow_legacy_names": False}
    stream = StringIO()
    yaml_safe.dump(card, stream)
    stream.seek(0)
    return SimuSetupConfig.from_yaml(stream, environment=SimpleNamespace(loader=loader))


def test_missing_parameters_reported_together(tmp_path):
    datasets = [{"dataset": "included", "simu_fac": "EFT_NLO"}]
    parameters = [{"name": "present"}, {"name": "missing_a"}, {"name": "missing_b"}]
    simu_data = {"included": {"EFT_NLO": {"SM": [1], "present": [2]}}}

    with pytest.raises(ConfigError) as error:
        setup_config(tmp_path, datasets, parameters, simu_data)

    assert "missing_a, missing_b" in str(error.value)
    assert "selected simu_fac block" in str(error.value)


def test_only_included_datasets_and_selected_blocks_count(tmp_path):
    datasets = [
        {"dataset": "included", "simu_fac": "EFT_NLO"},
        {"dataset": "ignored"},
    ]
    parameters = [{"name": "missing"}]
    simu_data = {
        "included": {"EFT_NLO": {"SM": [1]}, "EFT_LO": {"missing": [1]}},
        "ignored": {"EFT_NLO": {"missing": [1]}},
        "excluded": {"EFT_NLO": {"missing": [1]}},
    }

    with pytest.raises(ConfigError, match="missing"):
        setup_config(tmp_path, datasets, parameters, simu_data)


def test_coverage_across_datasets_accepts_zero_entries_and_linear_combinations(tmp_path):
    datasets = [
        {"dataset": "first", "simu_fac": "EFT_NLO"},
        {"dataset": "second", "simu_fac": "EFT_LO"},
    ]
    parameters = [
        {"name": "a"},
        {"name": "b"},
        {"name": "combination", "linear_combination": {"a": -2, "absent": 1}},
    ]
    simu_data = {"first": {"EFT_NLO": {"a": [0]}}, "second": {"EFT_LO": {"b": [0]}}}

    setup_config(tmp_path, datasets, parameters, simu_data)


def test_zero_coefficient_does_not_cover_linear_combination(tmp_path):
    datasets = [{"dataset": "included", "simu_fac": "EFT_NLO"}]
    parameters = [{"name": "combination", "linear_combination": {"present": 0, "absent": 1}}]
    simu_data = {"included": {"EFT_NLO": {"present": [0]}}}

    with pytest.raises(ConfigError, match="combination"):
        setup_config(tmp_path, datasets, parameters, simu_data)


def test_no_parameters_does_not_load_simu_data(tmp_path):
    datasets = [{"dataset": "no_file", "simu_fac": "EFT_NLO"}]

    setup_config(tmp_path, datasets, parameters=None, simu_data={})
