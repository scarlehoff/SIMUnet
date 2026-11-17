"""
Simunet version of vp-setupfit.
"""

from n3fit.scripts.vp_setupfit import (
    SETUPFIT_PROVIDERS,
    SetupFitApp,
    SetupFitConfig,
    SetupFitEnvironment,
)
from nnpdf_data import legacy_to_new_map
from reportengine.configparser import ConfigError
from validphys.loader import LoaderError

from simunet.config import SIMUConfig, SIMUEnvironment

SIMUNET_PROVIDERS = SETUPFIT_PROVIDERS + ["simunet.n3fit_providers"]


class SimuSetupEnvironment(SIMUEnvironment, SetupFitEnvironment):
    pass


class SimuSetupConfig(SIMUConfig, SetupFitConfig):
    def validate_simu_parameter_coverage(self):
        """Require global operator presence in the included, selected SIMU blocks."""
        parameters = self.input_params.get("simu_parameters")
        if not parameters:
            return

        operators = set()
        for dataset in self.input_params.get("dataset_inputs", []):
            block = dataset.get("simu_fac")
            if block is None:
                continue
            name = dataset["dataset"]
            if self.input_params.get("allow_legacy_names", True):
                name, _ = legacy_to_new_map(name, dataset.get("sys"))
            try:
                path, factors = self.loader._load_simu_factor(name)
            except LoaderError as exc:
                raise ConfigError(str(exc)) from exc
            if block not in factors:
                raise ConfigError(
                    f"Selected simu_fac block '{block}' for dataset '{name}' "
                    f"is absent from {path}."
                )
            operators.update(factors[block])

        missing = []
        for parameter in parameters:
            components = parameter.get("linear_combination", {parameter["name"]: 1})
            if not any(
                coefficient != 0 and name in operators
                for name, coefficient in components.items()
            ):
                missing.append(parameter["name"])
        if missing:
            raise ConfigError(
                "simu_parameters absent from every included dataset's selected simu_fac "
                "block: " + ", ".join(missing)
                + ". Each parameter requires at least one present operator with a "
                "nonzero linear_combination coefficient."
            )

    def produce_fktable_hasher(self, data, output_path):
        """Skip the fktable hasher as not all data will have an fktable."""
        pass

    @classmethod
    def from_yaml(cls, o, *args, **kwargs):
        """For closure test, the filtering for the level 0 closure test is a chain
        of internal imports, leading to ``dataset_t0_predictions``.
        In order to perform closuretests with our own version of ``dataset_t0_predictions``
        (with fixed predictions)
        it is necessary to substitute the validphys internal function.
        """
        tmp = super().from_yaml(o, *args, **kwargs)
        tmp.validate_simu_parameter_coverage()
        if tmp.input_params.get("closuretest") is not None:
            from simunet.n3fit_providers import dataset_t0_predictions
            from validphys import covmats

            covmats.dataset_t0_predictions = dataset_t0_predictions
        return tmp


class SimunetSetupfitApp(SetupFitApp):
    environment_class = SimuSetupEnvironment
    config_class = SimuSetupConfig

    def __init__(self):
        super(SetupFitApp, self).__init__(name="SimunetSetupfitApp", providers=SIMUNET_PROVIDERS)


def main():
    a = SimunetSetupfitApp()
    a.main()
