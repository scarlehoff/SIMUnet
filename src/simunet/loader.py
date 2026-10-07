import importlib.resources
import logging
from pathlib import Path

from nnpdf_data.validphys_compatibility import new_to_legacy_map
from validphys.core import CutsPolicy, TheoryIDSpec
from validphys.loader import CfactorNotFound, FallbackLoader, Loader, LoaderError
from validphys.utils import yaml_safe

from .core import SIMUnetDataSetSpec

log = logging.getLogger(__name__)


class SIMUnetLoader(Loader):
    def __init__(self, profile=None):
        super().__init__(profile)
        package_root: Path = importlib.resources.files("simunet")

        project_root = package_root.parent.parent

        simudata_path = project_root / "simudata"
        local_commondata_path = simudata_path / "commondata"
        if local_commondata_path.exists():
            self.commondata_folders = (local_commondata_path, *self.commondata_folders)

        self.simudata_path = simudata_path

    def _simu_factor_names(self, setname, use_commondata_name=False):
        """Try to figure out the name of the simufactor, with some heuristics.
        In order:
            1) The dataset as given
            2) The possible ``legacy_name`` if any
            3) The commondata folder name
        """
        # TODO: eventually remove 3) and possibly 2)
        names = [setname]
        legacy_names = new_to_legacy_map(setname, "legacy")
        if legacy_names is not None:
            names.extend(legacy_names)
        if use_commondata_name and "_" in setname:
            names.append(setname.rsplit("_", 1)[0])
        return names

    def _load_simu_factor(self, setname, use_commondata_name=False):
        """Go over the paths where to find the simu factor (as given by ``_simu_factor_names``)
        and return the first one that is found.
        """
        for simu_factor_name in self._simu_factor_names(setname, use_commondata_name):
            simufactorpath = self.simudata_path / "simu_factors" / f"SIMU_{simu_factor_name}.yaml"
            if simufactorpath.exists():
                return simufactorpath, yaml_safe.load(simufactorpath.read_text())

        raise CfactorNotFound(f"Could not find a SIMU factor for {setname}")

    def get_simu_parameters_name_dict(self, setname, simu_parameters_names):
        """
        Parameters
        ----------
        setname: str
                name of the dataset

        simu_parameters_names: list
                list containing the joined `simu_fac` and operator

        Returns
        -------
        dict

        """
        simu_fac_names_paths = {}
        simufactorpath, cfac_file = self._load_simu_factor(setname)

        # Check whether we have the keys we need
        for key in ["metadata", "SM_fixed"]:
            if key not in cfac_file:
                raise KeyError(
                    f"The '{key}' key is not present in the SIMU file at {simufactorpath}."
                )

        # TODO: to ask, why can't we read here the file directly instead of doing it in the provider
        # assign to each operator name the same simufactorpath
        for simu_parameters_name in simu_parameters_names:
            simu_fac_names_paths[simu_parameters_name] = simufactorpath

        return simu_fac_names_paths

    def check_dataset(
        self,
        name,
        *,
        rules=None,
        sysnum=None,
        theoryid,
        cfac=(),
        frac=1,
        cuts=CutsPolicy.INTERNAL,
        use_fitcommondata=False,
        fit=None,
        weight=1,
        simu_parameters_names=None,
        simu_parameters_linear_combinations=None,
        use_fixed_predictions=False,
        contamination=None,
        contamination_data=None,
        variant=None,
    ):
        if not isinstance(theoryid, TheoryIDSpec):
            theoryid = self.check_theoryID(theoryid)

        commondata = self.check_commondata(
            name, sysnum, use_fitcommondata=use_fitcommondata, fit=fit, variant=variant
        )

        # Note this is simply for convenience when scripting. The config will
        # construct the actual Cuts object by itself
        if isinstance(cuts, str):
            cuts = CutsPolicy(cuts)
        if isinstance(cuts, CutsPolicy):
            if cuts is CutsPolicy.NOCUTS:
                cuts = None
            elif cuts is CutsPolicy.FROMFIT:
                cuts = self.check_fit_cuts(commondata, fit)
            elif cuts is CutsPolicy.INTERNAL:
                if rules is None:
                    rules = self.check_default_filter_rules(theoryid)
                cuts = self.check_internal_cuts(commondata, rules)
            elif cuts is CutsPolicy.FROM_CUT_INTERSECTION_NAMESPACE:
                raise LoaderError(f"Intersection cuts not supported in loader calls.")

        if simu_parameters_names is not None:
            simu_parameters_names_CF = self.get_simu_parameters_name_dict(
                name, simu_parameters_names
            )
        else:
            simu_parameters_names_CF = None

        if use_fixed_predictions:
            fkspec, op = (), "NULL"
            _, cfac_file = self._load_simu_factor(name, use_commondata_name=True)
            fixed_predictions = cfac_file["SM_fixed"]
        else:
            fkspec, op = self._check_theory_old_or_new(theoryid, commondata, cfac)
            fixed_predictions = None

        return SIMUnetDataSetSpec(
            name=name,
            commondata=commondata,
            fkspecs=fkspec,
            thspec=theoryid,
            cuts=cuts,
            frac=frac,
            op=op,
            weight=weight,
            simu_parameters_names_CF=simu_parameters_names_CF,
            simu_parameters_names=simu_parameters_names,
            simu_parameters_linear_combinations=simu_parameters_linear_combinations,
            use_fixed_predictions=use_fixed_predictions,
            fixed_predictions=fixed_predictions,
            contamination=contamination,
            contamination_data=contamination_data,
        )


class SIMUFallbackLoader(SIMUnetLoader, FallbackLoader):
    """
    A loader that first tries to find resources locally
    (using SIMUnetLoader.check_*) and if it fails,
    it tries to download them (using RemoteLoader.download_*).
    """

    pass
