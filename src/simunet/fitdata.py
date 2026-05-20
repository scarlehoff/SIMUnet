import pandas as pd

BSM_FAC_FILE = 'bsm_fac.csv'

#@_check_has_bsm_facs
def read_bsm_facs(replica_paths):
    """
    Read the csv saved BSM factors, accounting for the
    postfit reshuffling, and return a concatenated dataframe
    for replicas as indices and the list BSM factors as columns
    Parameters
    ----------
        replica_paths: list 
    Output
    ------
        bsm_fac_results: pd.DataFrame
    """
    # Need to account for postfit reshuffling of replicas
    paths = [p / BSM_FAC_FILE for p in replica_paths]
    bsm_fac_results = pd.concat([pd.read_csv(i, index_col=0) for i in paths])

    rows, _columns = bsm_fac_results.shape
    bsm_fac_results.index = range(1, rows + 1)
    return bsm_fac_results 