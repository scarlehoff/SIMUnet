"""
n3fit version of simunet
"""

import pandas as pd
from reportengine.configparser import ConfigError

from n3fit.scripts.n3fit_exec import N3FIT_PROVIDERS, N3FitApp, N3FitConfig, N3FitEnvironment

# Import simufit to override the wrapper
from simunet import simufit
from simunet.config import SIMUConfig, SIMUEnvironment

SIMUNET_PROVIDERS = N3FIT_PROVIDERS + ["simunet.n3fit_providers"]


class SimufitEnvironment(SIMUEnvironment, N3FitEnvironment):
    pass


class SimufitConfig(SIMUConfig, N3FitConfig):

    def parse_frozen_pdf(
        self,
        name,
        replicas,
        freeze_pdf=False,
        load=None,
        load_weights_from_fit=None,
        simu_parameters=None,
        positivity=None,
        integrability=None,
    ):
        """Validate the frozen PDF configuration and select the matching replica."""
        from simunet.simufit.frozen_pdf import FrozenPDFMember

        if not freeze_pdf:
            raise ConfigError("frozen_pdf requires freeze_pdf: true")
        if load is not None or load_weights_from_fit is not None:
            raise ConfigError("frozen_pdf cannot be combined with PDF weights")
        if simu_parameters is None:
            raise ConfigError("frozen_pdf requires simu_parameters")
        if len(replicas) != 1:
            raise ConfigError("frozen_pdf requires fitting one replica per invocation")
        if (positivity or {}).get("posdatasets") or (integrability or {}).get("integdatasets"):
            raise ConfigError("frozen_pdf does not support positivity or integrability penalties")
        try:
            frozen_pdf = FrozenPDFMember(self.parse_pdf(name), replicas[0])
        except ValueError as error:
            raise ConfigError(str(error)) from error

        # The output writer needs the same member that supplies the fixed predictions.
        simufit._REGISTRY["frozen_pdf"] = frozen_pdf
        return frozen_pdf

    def produce_simu_layer(
        self,
        simu_parameters=None,
        freeze_pdf=False,
        load=None,
        load_weights_from_fit=None,
        frozen_pdf=None,
    ):
        """Return the factory for the BSM correction layer."""
        if simu_parameters is None:
            return None

        if freeze_pdf and frozen_pdf is None and load is None and load_weights_from_fit is None:
            raise ConfigError(
                "freeze_pdf: true requires frozen_pdf or PDF weights via `load` or `load_weights_from_fit`"
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
        if simufit._REGISTRY["simunet_active"]:
            weights = simufit._REGISTRY["best_weights"]
            layer = simufit._REGISTRY["layer"]
            # TODO: for multireplica, need to loop over replicas
            # instead of just taking the first one
            ret = {
                i.name: (w / s).tolist() for i, w, s in zip(layer.weights, weights, layer.scales)
            }
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
