"""Backend-independent preparation of the linear SIMUnet responses."""

import numpy as np

def parameter_responses(simu_parameters, cfactors=None):
    """Take all cfactors and returns a list of pre-computed values to call this function with."""
    if cfactors is None:
        return [0.0] * len(simu_parameters)
    sm = np.asarray(cfactors["SM"], dtype=float)

    lin_comb = []
    for parameter in simu_parameters:
        scale = float(parameter.get("scale", 1.0))
        response = np.zeros_like(sm)
        linear_combination = parameter.get("linear_combination", {parameter["name"]: 1.0})
        for name, coefficient in linear_combination.items():
            factor = np.asarray(cfactors.get(name, 0.0), dtype=float)
            if factor.ndim and factor.shape != sm.shape:
                raise ValueError(f"SIMU factor {name} does not match SM point count")
            response += factor / sm * coefficient
        response /= scale

        lin_comb.append(response)
    return lin_comb

def apply_smeft_correction(sm_predictions, weights, responses):
    """
    Apply the SIMUnet SMEFT correction to SM predictions.
    Arguments:
        sm_predictions: array of SM predictions
        weights: array of weights (scale * physical_coefficient)
        responses: array of SMEFT responses corresponding to each parameter
    """
    if len(weights) != len(responses):
        raise ValueError("Expected one SIMU weight per response")
    correction = 1.0
    for weight, response in zip(weights, responses):
        correction = correction + weight * response
    return sm_predictions * correction
