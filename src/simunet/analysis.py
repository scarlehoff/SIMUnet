import matplotlib.pyplot as plt
import pandas as pd
from reportengine.figure import figuregen
from simunet.loader import SIMUnetLoader
from validphys.dataplots import check_normalize_to, plot_fancy
import logging
import numpy as np
import os
from validphys.fitdata import replica_paths
from validphys.commondata import loaded_commondata_with_cuts
from validphys.covmats import sqrt_covmat
import json
from pathlib import Path
from simunet.fitdata import read_bsm_facs

from reportengine.table import table
from reportengine.figure import figure, figuregen
from validphys.convolution import central_predictions

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import seaborn as sns

log = logging.getLogger(__name__)
l = SIMUnetLoader()


def load_datasets_contamination(data):
    """
    Parameters
    ----------

    contamination_parameters: dict with

    theoryid: TheoryIDSpec

    dataset_inputs: NSList of DataSetInput objects

    Returns
    -------

    dict
        dictionary of BSM k-factors to apply on certain datasets

    """

    bsm_dict = {}

    for dataset in data.datasets:
        if dataset.cuts is not None:
            cuts = dataset.cuts.load()
        cont_params = dataset.contamination_data

        simu_dict = l.get_simu_parameters_name_dict(
            dataset.name, simu_parameters_names=[dataset.contamination]
        )
        cont_path = list(simu_dict.values())[0]
        cont_params = dataset.contamination_data
        cont_order = dataset.contamination

        if cont_order == None:
            log.warning(f"{dataset.name} is not contaminated. Is it right?")

            bsm_dict[dataset.name] = np.zeros(len(cuts))

        elif not os.path.exists(cont_path):
            log.error(
                f"Could not find a BSM-factor for {dataset.name}. Are you sure they exist in the given theory?"
            )
            bsm_dict[dataset.name] = np.zeros(len(cuts))
        else:
            log.info(f"Loading {dataset.name}.")
            simu_card = l.load_simu_factors(cont_path)

            k_factors = np.zeros(len(cuts))
            for param in cont_params:
                value = param["value"]
                lin_comb = param["linear_combination"]
                bsm_xs = np.zeros(len(cuts))
                for op in lin_comb:
                    if op in simu_card[dataset.contamination]:
                        bsm_xs += (
                            lin_comb.get(op, 0)
                            * np.array(simu_card[dataset.contamination][op])[cuts]
                        )
                    else:
                        # Log a warning and keep SMEFT K-factor to zero
                        log.warning(
                            f"Operator '{op}' not found for {dataset.name}. Setting K-factor to zero."
                        )
                k_factors += value * bsm_xs / np.array(simu_card[dataset.contamination]["SM"])[cuts]
            bsm_dict[dataset.name] = k_factors

    return bsm_dict


@check_normalize_to
@figuregen
def plot_data_theory_contaminated(
    simunet_one_or_more_results, commondata, cuts, normalize_to: (int, str, type(None)) = None
):
    """Plot contaminated data and theory, optionally normalized as in plot_fancy."""
    return plot_fancy(simunet_one_or_more_results, commondata, cuts, normalize_to=normalize_to)


@check_normalize_to
@figuregen
def plot_data_theory_contaminated(
    simunet_one_or_more_results, commondata, cuts, normalize_to: (int, str, type(None)) = None
):
    """Plot contaminated data and theory, optionally normalized as in plot_fancy."""
    return plot_fancy(simunet_one_or_more_results, commondata, cuts, normalize_to=normalize_to)


@figuregen
def plot_nd_bsm_facs_fits(
    fits,
    read_bsm_facs_fits,
    bsm_names_to_latex,
    posterior_plots_settings,
    contamination_data=None,
    bsm_names_to_plot_scales=None,
):
    """
    Compare histograms of BSM factors between different fits in SIMUnet.

    Parameters
    ----------
    fits : NSList
        List of FitSpec to be compared.
    read_bsm_facs_fits : list of pandas.DataFrame
        BSM factors for each fit, in the same order as fits.
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to their LaTeX representations.
    posterior_plots_settings : dict, optional
        Dictionary containing settings for posterior plots such as 'n_bins', 'rangex', 'rangey', and 'same_bins'.

    bsm_names_to_plot_scales : dict, optional
        Display multipliers for coefficients. Missing entries default to one.
        Histogram bins, bounds and expected values use the scaled coordinates;
        rangex and rangey are interpreted in the displayed coordinates.

    Yields
    ------
    fig : matplotlib.figure.Figure
        A matplotlib figure object for each BSM coefficient comparison.
    """
    plot_scales = bsm_names_to_plot_scales or {}
    labels = bsm_names_to_latex or {}
    all_ops = {op for bsm_facs_df in read_bsm_facs_fits for op in bsm_facs_df.columns}
    unknown_ops = (set(labels) | set(plot_scales)) - all_ops
    if unknown_ops:
        raise ValueError(
            "bsm_plot_settings contains coefficients absent from the fit results: "
            + ", ".join(sorted(unknown_ops))
        )

    # extract settings

    same_bins = posterior_plots_settings.get("same_bins", False)
    n_bins = posterior_plots_settings.get("n_bins", 10)
    rangex = posterior_plots_settings.get("rangex", None)
    rangey = posterior_plots_settings.get("rangey", None)
    add_bounds = posterior_plots_settings.get("add_bounds", False)
    exp_val_lines = posterior_plots_settings.get("exp_val_lines", None)
    # Produce the vertical line values
    if exp_val_lines == 'SM':
        exp_val_lines_dict = {op: 0 for op in all_ops}

    if exp_val_lines == 'CONT':
        exp_val_lines_dict = {op: 0 for op in all_ops}

        for item in contamination_data:
            val = item["value"]

            for op, coeff in item["linear_combination"].items():
                exp_val_lines_dict[op] += val * coeff

    # If same_bins=True, create binnings
    if same_bins:
        min_bins = pd.Series(dict(zip(list(all_ops), np.full(len(all_ops), np.inf))))
        max_bins = pd.Series(dict(zip(list(all_ops), np.full(len(all_ops), -np.inf))))
        for bsm_facs_df in read_bsm_facs_fits:
            bsm_facs_df = bsm_facs_df.mul(
                [plot_scales.get(op, 1) for op in bsm_facs_df.columns], axis="columns"
            )
            min_df = bsm_facs_df.min()
            max_df = bsm_facs_df.max()
            min_bins = pd.concat([min_bins, min_df], axis=1).min(axis=1)
            max_bins = pd.concat([max_bins, max_df], axis=1).max(axis=1)

    # plot all operators
    for op in all_ops:
        plot_scale = plot_scales.get(op, 1)
        fig, ax = plt.subplots()
        for fit, bsm_facs_df in zip(fits, read_bsm_facs_fits):
            if same_bins:
                bins = np.linspace(min_bins.loc[op], max_bins.loc[op], n_bins)
            else:
                bins = n_bins

            if bsm_facs_df.get([op]) is not None:
                values = plot_scale * bsm_facs_df.get([op]).values
                if add_bounds:
                    hist_label = f"{fit.label} (mean & std dev)"
                else:
                    hist_label = fit.label
                ax.hist(values, bins=bins, density=True, alpha=0.5, label=hist_label)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                ax.set_ylabel("Prob. density", fontsize=14)
                label = labels.get(op, op)
                if plot_scale != 1:
                    label = str(plot_scale) + r"$\cdot$" + label
                if bsm_names_to_latex is None:
                    ax.set_xlabel(label, fontsize=14)
                else:
                    ax.set_xlabel(label + r"$/\Lambda^2$ [TeV$^{-2}$]", fontsize=16)

                ax.grid(False)
                if rangex is not None:
                    ax.set_xlim(rangex)
                if rangey is not None:
                    ax.set_ylim(rangey)
                if add_bounds:
                    mean = values.mean()
                    std = values.std()
                    # Add axis ticks at mean and one standard deviation
                    ax.axvline(mean, linestyle='--', linewidth=2.5)
                    ax.axvline(mean - std, linestyle='dotted', linewidth=2.5)
                    ax.axvline(mean + std, linestyle='dotted', linewidth=2.5)
                ax.legend(fontsize=14)
                if exp_val_lines is not None and exp_val_lines_dict.get(op) is not None:
                    ax.axvline(
                        plot_scale * exp_val_lines_dict[op],
                        color='r',
                        linestyle='-.',
                        linewidth=2.5,
                        label='Expected value',
                    )
                    ax.legend(fontsize=14)
        yield fig


def simu_fac_to_popxf(
    data, pdf, groups_covmat_no_table, simunet_one_or_more_results, external_covmats
):
    "This function produces a popxf file with the BSM factors for each dataset"
    written = []
    cov = groups_covmat_no_table

    for data_input, dataset in zip(data, data.datasets):
        log.info(f"Processing dataset {dataset.name} for popxf generation.")
        dataset_name = dataset.name
        simu_fac = data_input.simu_fac
        if simu_fac is None:
            raise ValueError(
                f"{dataset_name}: Popxf export requires simu_fac in dataset_inputs "
                "(for example, simu_fac: EFT_NLO)."
            )
        cuts = dataset.cuts.load()
        simu_dict = l.get_simu_parameters_name_dict(
            dataset.name, simu_parameters_names=[dataset.contamination]
        )
        simu_path = list(simu_dict.values())[0]
        simu = l.load_simu_factors(simu_path)

        if dataset.use_fixed_predictions:
            SM_predictions = np.array(simu.get("SM_fixed", []))[cuts]
            pdf_name = "fixed_predictions"
        else:
            SM_predictions = simunet_one_or_more_results[1].central_value
            pdf_name = str(pdf.name)

        eft_factors = simu.get(simu_fac)

        SMEFT_K_factors = {k: np.array(v)[cuts] for k, v in eft_factors.items()}
        parameters = [k for k in SMEFT_K_factors.keys() if k != "SM"]

        obs_names = [f"({dataset_name}, bin_{i})" for i in range(len(cuts))]

        observable_central = {"('', '', 'RR')": np.zeros(len(cuts)).tolist()}
        for k in parameters:
            b = SM_predictions * SMEFT_K_factors[k] / SMEFT_K_factors["SM"] * 1000**2  # in GeV^-2
            observable_central[f"('', '{k}', 'RR')"] = b.tolist()

        popxf_dict = {
            "$schema": "https://json.schemastore.org/popxf-1.0.json",
            "metadata": {
                "basis": {"custom": {"name": "u2"}},
                "scale": 1000.0,
                "parameters": parameters,
                "observable_names": obs_names,
                "reproducibility": {"tools": {"name": "validphys"}},
                "pdf": pdf_name,
                "QCD": "",
                "EWK": "",
                "SMEFT": simu_fac,
            },
            "data": {"observable_central": observable_central},
        }

        cd = dataset.commondata.load().get_cv()[cuts]
        central_value = cd - SM_predictions
        dataset_cov = cov.loc[(slice(None), dataset_name), (slice(None), dataset_name)].to_numpy(
            copy=True
        )
        for extra_covmat in external_covmats:
            if dataset_name not in extra_covmat.index.get_level_values("dataset"):
                continue
            extra_dataset_cov = extra_covmat.xs(dataset_name, level="dataset", axis=0).xs(
                dataset_name, level="dataset", axis=1
            )
            dataset_cov += extra_dataset_cov.loc[cuts, cuts].to_numpy()

        standard_deviation = np.sqrt(np.diag(dataset_cov))
        correlation = dataset_cov / np.outer(standard_deviation, standard_deviation)

        pdfxf_dict = {
            "$schema": "https://json.schemastore.org/pdfxf-1.0.json",
            dataset_name: [
                {
                    "observables": obs_names,
                    "distribution_type": "MultivariateNormalDistribution",
                    "central_value": central_value.tolist(),
                    "standard_deviation": standard_deviation.tolist(),
                    "correlation": correlation.tolist(),
                }
            ],
        }

        likelihood_files = Path("likelihood_files")

        pdf_dir = likelihood_files / str(pdf_name)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        popxf_path = pdf_dir / f"{dataset_name}.json"
        pdfxf_path = pdf_dir / f"{dataset_name}_measurement.json"

        popxf_path.write_text(json.dumps(popxf_dict, indent=4))
        pdfxf_path.write_text(json.dumps(pdfxf_dict, indent=4))

        written.append(dataset_name)

    log.info(f"Written popxf and pdfxf files for datasets: {', '.join(written)} to {pdf_dir}.")

    return

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a colormap to a specific range.

    This function creates a new colormap based on a given colormap but truncated to the range specified
    by minval and maxval. This is useful for adjusting the range of colors used in a plot.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The original colormap to be truncated.
    minval : float, optional
        The minimum value of the new colormap, by default 0.0.
    maxval : float, optional
        The maximum value of the new colormap, by default 1.0.
    n : int, optional
        The number of discrete colors in the new colormap, by default 100.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The truncated colormap.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


@table
def fisher_information_by_sector(data, groups_covmat, simu_parameters_names, pdf):
    """
    Obtains the Fisher information matrices for each of the BSM sectors.

    This function computes the Fisher information matrix for each sector in a dataset, providing a
    measure of the amount of information each sector contributes to the parameters. The function
    accumulates datasets by sectors, calculates the reduced covariance matrices for each sector,
    and computes the Fisher information matrices. The diagonal elements of these matrices are
    extracted and normalized to provide a comparative view across sectors.

    Parameters
    ----------
    data : DataGroupSpec
        Configured datasets used for computing the Fisher information matrices.
    groups_covmat : pd.DataFrame
        Covariance matrices for the groups in the datasets.
    simu_parameters_names : list of str
        List of names of the simulation parameters.
    pdf : PDF object
        The parton distribution function object.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the normalized diagonal elements of the Fisher information matrices,
        indexed by simulation parameters and with sectors as columns.
    """
    # First, get the names of the BSM sectors.

    bsm_dataset_inputs_sectors = {}

    for dataset in data.datasets:
        # import IPython; IPython.embed()
        if dataset.bsm_sector in bsm_dataset_inputs_sectors.keys():
            bsm_dataset_inputs_sectors[dataset.bsm_sector] += [dataset]
        else:
            bsm_dataset_inputs_sectors[dataset.bsm_sector] = [dataset]

    all_sectors_duplicates = list(bsm_dataset_inputs_sectors.keys())
    all_sectors = []
    [all_sectors.append(x) for x in all_sectors_duplicates if x not in all_sectors]
    fisher_by_sector = []
    for sec in all_sectors:
        if sec in bsm_dataset_inputs_sectors.keys():
            datasets = bsm_dataset_inputs_sectors[sec]
            dataset_names = [ds.name for ds in datasets]
        else:
            datasets = None
            dataset_names = []

        ds_and_fo_names = dataset_names

        # Take correct submatrix of groups_covmat
        reduced_covmats = []
        for name in ds_and_fo_names:
            reduced_covmats += [groups_covmat.xs(name, axis=1, level=1, drop_level=False)]

        reduced_covmat = pd.concat(reduced_covmats, axis=1)

        reduced_covmats = []
        for name in ds_and_fo_names:
            reduced_covmats += [reduced_covmat.T.xs(name, axis=1, level=1, drop_level=False)]

        reduced_covmat = pd.concat(reduced_covmats, axis=1)

        # Hence construct the Fisher matrices
        fisher_by_sector += [
            _compute_fisher_information_matrix(datasets, reduced_covmat, simu_parameters_names, pdf)
        ]

    # Now go through the matrices one-by-one, and take the diagonal
    fisher_diags_by_sector = []

    for matrix in fisher_by_sector:
        diagonal = np.diagonal(matrix.to_numpy())
        fisher_diags_by_sector += [diagonal.tolist()]

    # Rescale array
    array = np.array(fisher_diags_by_sector).T
    sums = np.sum(array, axis=1)
    rows, columns = array.shape
    for i in range(rows):
        array[i, :] = array[i, :] / sums[i] * 100

    df = pd.DataFrame(array, columns=all_sectors, index=simu_parameters_names)

    return df


def _compute_fisher_information_matrix(datasets, groups_covmat, simu_parameters_names, pdf):
    """
    Computes the Fisher information matrix for a given set of datasets and simulation parameters.

    This function calculates the Fisher information matrix, which quantifies the amount of information
    that an observable random variable carries about an unknown parameter upon which the probability
    of the random variable depends. It takes into account the datasets, groups covariance
    matrix, simulation parameters, and the parton distribution function (PDF).

    Parameters
    ----------
    datasets : sequence of SIMUnetDataSetSpec
        Configured datasets used for computing the Fisher information matrix.
    groups_covmat : pd.DataFrame
        Covariance matrices for the groups in the datasets.
    simu_parameters_names : list of str
        List of names of the simulation parameters.
    pdf : PDF object
        The parton distribution function object.

    Returns
    -------
    pd.DataFrame
        The computed Fisher information matrix as a pandas DataFrame.
    """
    bsm_factors = []
    if datasets is not None:
        for ds in datasets:
            cuts = ds.cuts.load()
            ndat = len(cuts)
            simu_dict = l.get_simu_parameters_name_dict(
                ds.name, simu_parameters_names=ds.simu_parameters_names
            )
            simu_path = list(simu_dict.values())[0]
            simu_info = l.load_simu_factors(simu_path)
            central_sm = central_predictions(ds, pdf)
            # bsm_factors = []
            operator_central_values = []
            for param, lincomb in ds.simu_parameters_linear_combinations.items():

                model = "_".join(param.split("_")[:-1])

                numerator = np.zeros(ndat)
                for op, coeff in lincomb.items():
                    if op in simu_info[model]:
                        numerator += coeff * np.array(simu_info[model][op])[cuts]
                sm_prediction = np.array(simu_info[model]["SM"])[cuts]
                central = numerator / sm_prediction
                operator_central_values.append(central)

            coefficients = central_sm.to_numpy().T * np.array(operator_central_values)
            bsm_factors += [coefficients]
            # bsm_fac = parse_simu_parameters_names_CF(ds.simu_parameters_names_CF, ds.simu_parameters_linear_combinations, cuts=ds.cuts)
            # central_sm = central_predictions(ds, pdf)
            # coefficients = central_sm.to_numpy().T * np.array([i.central_value for i in bsm_fac.values()])
            # bsm_factors += [coefficients]

    # Make bsm_factors into a nice numpy array.
    # The rows are the data, the columns are the operator
    bsm_factors = np.concatenate(bsm_factors, axis=1).T
    cov = groups_covmat.to_numpy()
    inv_cov = np.linalg.inv(cov)
    fisher = bsm_factors.T @ inv_cov @ bsm_factors

    fisher = pd.DataFrame(fisher, index=simu_parameters_names)
    fisher = fisher.T
    fisher.index = simu_parameters_names

    return fisher


@table
def fisher_information_matrix(data, groups_covmat, simu_parameters_names, pdf):
    """
    Obtains the full Fisher information matrix for the BSM parameters.

    This function computes the Fisher information matrix for Beyond the Standard Model (BSM) parameters
    given a dataset. It utilizes an internal function `_compute_fisher_information_matrix` to perform the computation.

    Parameters
    ----------
    data : DataGroupSpec
        Configured datasets used for computing the Fisher information matrix.
    groups_covmat : array-like
        Covariance matrices for the groups in the dataset.
    simu_parameters_names : list
        List of names of the simulation parameters.
    pdf : PDF object
        The parton distribution function object.

    Returns
    -------
    array-like
        The computed Fisher information matrix.
    """
    return _compute_fisher_information_matrix(
        data.datasets, groups_covmat, simu_parameters_names, pdf
    )


@figure
def plot_fisher_information_by_sector(
    fisher_information_by_sector, bsm_names_to_latex, bsm_sectors_to_latex
):
    """
    Produces a heatmap plot from the Fisher information by sector table.

    This function creates a heatmap visualizing the Fisher information matrix, where rows correspond to BSM coefficients
    and columns correspond to different sectors. This visualization helps in understanding the impact of various
    sectors on the coefficients.

    Parameters
    ----------
    fisher_information_by_sector : pandas.DataFrame
        The Fisher information matrix with sectors as columns and BSM coefficients as rows.
    bsm_names_to_latex : dict
        Dictionary mapping BSM factor names to their LaTeX representations.
    bsm_sectors_to_latex : dict
        Dictionary mapping sector names to their LaTeX representations.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the heatmap plot of the Fisher information.
    """
    f = fisher_information_by_sector

    coeff_names = [bsm_names_to_latex[x] for x in f.index]
    sector_names = [bsm_sectors_to_latex[x] for x in f.columns]

    ncoeff, ndata = f.shape

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim([-1.5, ncoeff - 1.5])
    ax.set_ylim([-1.5, ndata - 1.5])
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))

    old_cmap = plt.get_cmap('YlGnBu')
    new_cmap = truncate_colormap(old_cmap, minval=0.0, maxval=0.65)

    ax = sns.heatmap(f.T, vmin=0.0, vmax=100.0, cmap=new_cmap, cbar=False)

    ax.set_xticklabels(coeff_names, rotation=0.0, va='top', ha='center', fontsize=14)
    ax.set_yticklabels(sector_names, rotation=0.0, va='center', ha='right', fontsize=14)

    for y, val in enumerate(f.index):
        ax.plot([y, y], [-1.5, ndata + 1], ls='solid', c='lightgray', lw=0.8)

    fisher_rounded = np.round(f.to_numpy(), 0)

    # Plot numbers
    nrow, ncol = np.shape(f.T)
    for i in range(nrow):
        for j in range(ncol):
            if fisher_rounded.T[i, j] != 0:
                plt.text(x=j + 0.15, y=i + 0.6, s=str(fisher_rounded.T[i, j]), fontsize=10)

    plt.tight_layout()

    return fig
