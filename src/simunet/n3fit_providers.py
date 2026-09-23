"""
Contains the providers that n3fit will use and that we want to override
"""

import numpy as np

from validphys.n3fit_data import fittable_datasets_masked as vanilla_fittable_datasets_masked
from simunet import simufit
from simunet.results import SIMUnetThPredictionsResult
import scipy as sp

from simunet.loader import SIMUnetLoader
import logging

l = SIMUnetLoader()
log = logging.getLogger(__name__)


def _analytic_solution(data, theorySM, theorylin, covmat):
    """
    Returns the minimum of the chi2 function:

      chi2 = (data - theorySM - theorylin c)^T invcovmat (data - theorySM - theorylin c),

    """

    diff = data - theorySM

    part1 = np.linalg.solve(covmat, theorylin)
    part2 = np.linalg.solve(covmat, diff)

    sol = np.linalg.solve(theorylin.T @ part1, theorylin.T @ part2)

    minval = (diff - theorylin @ sol).T @ np.linalg.solve(covmat, diff - theorylin @ sol)
    minval = minval / len(diff)

    return (sol, minval)


def _construct_analytic_initialisation(
    data,
    analytic_initialisation_pdf,
    make_replica,
    groups_covmat,
    simu_parameters,
    use_th_covmat=False,
):
    """
    Constructs the analytic initialisation for the simu_parameters.
    """
    sm_predictions = []
    linear_bsm = []
    th_covmat = []
    all_pred_replicas = []
    exp_data = make_replica
    # TODO: Check that this changes with contamination
    nop = len(simu_parameters)
    # Reuse the configured datasets so predictions retain the fit's cuts and ordering.
    for ds in data.datasets:
        cuts = ds.cuts.load()
        ndat = len(cuts)
        pred_values = SIMUnetThPredictionsResult.from_convolution(
            analytic_initialisation_pdf, ds, load_dataset_contamination=None
        )
        central_value = pred_values.central_value
        sm_predictions.append(central_value)
        pred_replicas = pred_values.error_members
        all_pred_replicas.append(pred_replicas)

        if ds.simu_parameters_names is not None:
            simu_path = next(iter(ds.simu_parameters_names_CF.values()))
            simu_info = l.load_simu_factors(simu_path)
            columns = []
            for param in ds.simu_parameters_linear_combinations:
                model = "_".join(param.split("_")[:-1])
                column = np.zeros((ndat,))
                for key in ds.simu_parameters_linear_combinations[param]:
                    if key in simu_info[model].keys():
                        model_values = np.asarray(simu_info[model][key])[cuts]
                        column += np.array(
                            model_values * ds.simu_parameters_linear_combinations[param][key]
                        )
                column = (
                    column / np.array([simu_info[model]["SM"][i] for i in cuts]) * central_value
                )
                columns += [column]
            linear_bsm.append(np.array(columns).T)

            dataset_th_covmat = np.zeros((ndat, ndat))
            if use_th_covmat:
                stored_covmat = np.asarray(simu_info.get("theory_cov", []), dtype=float)
                # YAML rows describe the covariance in the original data-point order.
                # Empty placeholders ([[]], [[[[]]]], etc.) carry no uncertainty.
                if stored_covmat.size:
                    expected_shape = (ds.commondata.ndata, ds.commondata.ndata)
                    if stored_covmat.shape != expected_shape:
                        raise ValueError(
                            f"{ds.name}: theory_cov in {simu_path} must have uncut shape "
                            f"{expected_shape}, got {stored_covmat.shape}."
                        )
                    dataset_th_covmat = stored_covmat[np.ix_(cuts, cuts)]
            th_covmat.append(dataset_th_covmat)
        else:

            linear_bsm.append(np.zeros((ndat, nop)))
            th_covmat += [np.zeros((ndat, ndat))]

    sm_predictions = np.concatenate(sm_predictions)
    linear_bsm = np.concatenate(linear_bsm)

    th_covmat = sp.linalg.block_diag(*th_covmat)
    th_covmat = th_covmat.T
    pred_replicas_all_datasets = np.concatenate(all_pred_replicas, axis=0)
    pdf_covmat = np.cov(pred_replicas_all_datasets)
    total_covmat = groups_covmat + th_covmat + pdf_covmat

    sol, minval = _analytic_solution(exp_data, sm_predictions, linear_bsm, total_covmat)
    simu_parameters_scales = [1 / abs(ini) if ini != 0 else 1.0 for ini in sol]
    log.info("The analytic solution is " + str(sol))
    log.info("The minimum is achieved at chi2=" + str(minval))
    for param, scale, init in zip(simu_parameters, simu_parameters_scales, sol):
        param["scale"] = float(scale)
        param["initialisation"] = {"type": "constant", "value": float(init)}
    return simu_parameters


def simu_parameters_analytic(
    data,
    make_replica,
    groups_covmat,
    simu_parameters,
    analytic_initialisation_pdf=None,
    analytic_initialisation=False,
    use_th_covmat=False,
):
    """
    Constructs the analytic initialisation for the simu_parameters if requested.
    """
    if analytic_initialisation:
        return _construct_analytic_initialisation(
            data=data,
            analytic_initialisation_pdf=analytic_initialisation_pdf,
            make_replica=make_replica,
            groups_covmat=groups_covmat,
            simu_parameters=simu_parameters,
            use_th_covmat=use_th_covmat,
        )
    return simu_parameters


# I'm assuming the information necessary is in the data and needs to be propagated to the fittable dataset
# minimal changes are necessary if instead we need to propagate this to the fktable instead


def fittable_datasets_masked(data, simu_layer=None, simu_parameters_analytic=None):
    # TODO: Looks at use_th_covmat
    """Note: for anayltic solution the data must be grouped together (default in simunet: ALL)."""

    ret = vanilla_fittable_datasets_masked(data)
    if simu_layer is None:
        return ret

    if simufit._REGISTRY.get("layer") is None:
        simu_layer_generated = simu_layer(simu_parameters_analytic)
        simufit._REGISTRY["layer"] = simu_layer_generated
    else:
        simu_layer_generated = simufit._REGISTRY["layer"]

    # At this point we have the information on the simunet parameters twice
    # once in the `simu_layer` and once in
    # data[X].simu_parameters_linear_combinations
    # but `simu_layer` is the right one (they could be made to be the same)

    # The cfactors themselves must be part of the fittable dataset (because they are cfactors applied to the whole dataset)
    # so this function must
    # 1) Read all cfactors that are asked by `simu_fac` of each dataset
    # 2) Pass a dictionary of [key] : [value] to the fittable dataset

    # Loop over the SIMUnetDataSetSpec
    for data_input, dataset, fittable_dataset in zip(data, data.datasets, ret):
        if dataset.simu_parameters_names_CF is None:
            # Nothing to do here
            continue

        # Load the SIMU file used to construct the BSM correction factors
        # TODO (will there be ever more than one? if so... how to deal with it?)
        cuts = dataset.cuts.load().tolist()
        for simu_path in dataset.simu_parameters_names_CF.values():
            simu_info = l.load_simu_factors(simu_path)

            cfactors_raw = simu_layer_generated.apply_linear_comb(simu_info[data_input.simu_fac])
            cfactors = [np.take(i, indices=cuts, mode="clip") for i in cfactors_raw]
            break

        # TODO this is ugly, but needs to be beautified in n3fit not here
        fittable_dataset.fktables_data[0].simunet_cfactors = cfactors
        fittable_dataset.fktables_data[0].simunet_layer = simu_layer_generated

    return ret
