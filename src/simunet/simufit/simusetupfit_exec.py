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


class SimunetSetupfitApp(SetupFitApp):
    environment_class = SimuSetupEnvironment
    config_class = SimuSetupConfig

    def __init__(self):
        super(SetupFitApp, self).__init__(name="SimunetSetupfitApp", providers=SIMUNET_PROVIDERS)


def main():
    a = SimunetSetupfitApp()
    a.main()
