"""
n3fit version of simunet
"""

import pandas as pd

from n3fit.scripts.n3fit_exec import N3FIT_PROVIDERS, N3FitApp, N3FitConfig, N3FitEnvironment
from reportengine.configparser import ConfigError

# Import simufit to override the wrapper
from simunet import simufit
from simunet.config import SIMUConfig, SIMUEnvironment

SIMUNET_PROVIDERS = N3FIT_PROVIDERS + ["simunet.n3fit_providers"]


class SimufitEnvironment(SIMUEnvironment, N3FitEnvironment):
    pass


class SimufitConfig(SIMUConfig, N3FitConfig):

    def produce_simu_layer(
        self, simu_parameters=None, freeze_pdf=False, load=None, load_weights_from_fit=None
    ):
        """
        Parses the simu_parameters dictionary and
        generates a function that produces
        the simunet layer that will be applied to all obserables.
        """
        if simu_parameters is None:
            return None

        if freeze_pdf and load is None and load_weights_from_fit is None:
            raise ConfigError(
                "freeze_pdf: true requires PDF weights via `load` or `load_weights_from_fit`"
            )

        from simunet.simufit.combine_cfac import CombineCfacLayer

        lay = lambda simpar: CombineCfacLayer(simpar)

        # Update the register
        simufit._REGISTRY["freeze"] = freeze_pdf

        return lay


class SimunfitApp(N3FitApp):
    environment_class = SimufitEnvironment
    config_class = SimufitConfig

    def __init__(self):
        super(N3FitApp, self).__init__(name="SimunfitApp", providers=SIMUNET_PROVIDERS)

    def run(self):
        """Save the simunet weights after the fit has run completely."""
        super().run()
        weights = simufit._REGISTRY["best_weights"]
        layer = simufit._REGISTRY["layer"]
        # TODO: for multireplica, need to loop over replicas
        # instead of just taking the first one
        ret = {i.name: (w / s).tolist() for i, w, s in zip(layer.weights, weights, layer.scales)}
        bsm_fac_df = pd.DataFrame(ret)

        simu_path = (
            self.environment.replica_path
            / f"replica_{self.environment.replicas[0]}"
            / "bsm_fac.csv"
        )
        with simu_path.open("w") as f:
            bsm_fac_df.to_csv(f)


def main():
    SimunfitApp().main()


if __name__ == "__main__":
    main()
