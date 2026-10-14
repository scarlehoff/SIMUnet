import functools

from validphys.core import DataSetInput, DataSetSpec, FKTableSpec


class SIMUnetDataSetSpec(DataSetSpec):
    """
    Dataset specification for SIMUnet-based PDF fits.

    Extends DataSetSpec with SIMUnet-specific parameters.
    Below the parameters that are added with respect to the standard validphys DataSetSpec.
    They are all optional as "normal" datasets would also go through this path.

    Parameters
    ----------
    ...
    simu_parameters_names_CF : dict, optional
        Mapping of CF file keys to paths of CFactor files
    simu_parameters_names : list of str, optional
        BSM coefficients to include in the fit.
    simu_parameters_linear_combinations : dict, optional
        Linear combinations of simulation parameters for the fit
    use_fixed_predictions : bool, optional
        Whether to use pre-computed fixed predictions instead of generating them with the fitted PDF
    fixed_predictions : array-like, optional
        Pre-computed fixed prediction values (used when use_fixed_predictions is True).
    contamination : str, optional
        Contamination parameter for closure tests.
    contamination_data : dict, optional
        Contamination data dictionary for closure tests.
    """

    def __init__(
        self,
        *,
        name,
        commondata,
        fkspecs,
        thspec,
        cuts,
        frac=1,
        op=None,
        weight=1,
        simu_parameters_names_CF=None,
        simu_parameters_names=None,
        simu_parameters_linear_combinations=None,
        use_fixed_predictions=False,
        fixed_predictions=None,
        contamination=None,
        contamination_data=None,
    ):
        super().__init__(
            name=name,
            commondata=commondata,
            fkspecs=fkspecs,
            thspec=thspec,
            cuts=cuts,
            frac=frac,
            op=op,
            weight=weight,
            rules=(),
        )
        self.simu_parameters_names_CF = simu_parameters_names_CF

        self.simu_parameters_names = simu_parameters_names
        self.simu_parameters_linear_combinations = simu_parameters_linear_combinations
        self.use_fixed_predictions = use_fixed_predictions
        self.fixed_predictions = fixed_predictions
        self.contamination = contamination
        self.contamination_data = contamination_data

    @functools.lru_cache
    def load_commondata(self):
        """Attaches contamination to the loaded commondata"""

        cd = super().load_commondata()

        cd.contamination = self.contamination
        cd.contamination_data = self.contamination_data

        return cd


class SIMUnetFKTableSpec(FKTableSpec):
    def __init__(
        self,
        fkpath,
        cfactors,
        metadata=None,
        use_fixed_predictions=False,
        fixed_predictions_path=None,
        contamination=None,
    ):
        super().__init__(fkpath=fkpath, cfactors=cfactors, metadata=metadata)
        self.use_fixed_predictions = use_fixed_predictions
        self.fixed_predictions_path = fixed_predictions_path
        self.contamination = contamination


class SIMUnetDataSetInput(DataSetInput):
    def __init__(
        self,
        *,
        name,
        cfac,
        frac,
        weight,
        custom_group,
        variant,
        simu_parameters_names,
        simu_parameters_linear_combinations,
        use_fixed_predictions,
        contamination,
        simu_fac,
    ):
        super().__init__(
            name=name,
            cfac=cfac,
            frac=frac,
            weight=weight,
            custom_group=custom_group,
            variant=variant,
        )

        self.simu_parameters_names = simu_parameters_names
        self.simu_parameters_linear_combinations = simu_parameters_linear_combinations
        self.use_fixed_predictions = use_fixed_predictions
        self.contamination = contamination
        self.simu_fac = simu_fac
