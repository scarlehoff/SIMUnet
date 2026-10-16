"""
Simunet version of vp-setupfit.
"""

from n3fit.scripts.vp_setupfit import (
    SETUPFIT_PROVIDERS,
    SetupFitApp,
    SetupFitConfig,
    SetupFitEnvironment,
)
from simunet.config import SIMUConfig, SIMUEnvironment

SIMUNET_PROVIDERS = SETUPFIT_PROVIDERS + ["simunet.n3fit_providers"]


class SimuSetupEnvironment(SIMUEnvironment, SetupFitEnvironment):
    pass


class SimuSetupConfig(SIMUConfig, SetupFitConfig):
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
