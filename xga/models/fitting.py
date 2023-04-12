#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

from typing import List

import numpy as np


# This set of functions is to support the MCMC fitting found in the BaseProfile class(es) - its here because
#  local functions can't be pickled

def log_likelihood(theta: np.ndarray, r: np.ndarray, y: np.ndarray, y_err: np.ndarray, m_func) -> np.ndarray:
    """
    Uses a simple Gaussian likelihood function, returns the logged value.

    :param np.ndarray theta: The knowledge we have (think theta in Bayesian parlance) - gets fed
        into the model we've chosen.
    :param np.ndarray r: The radii at which we have measured profile values.
    :param np.ndarray y: The values we have measured for the profile.
    :param np.ndarray y_err: The uncertainties on the measured profile values.
    :param m_func: The model function that is being fit to.
    :return: The log-likelihood value.
    :rtype: np.ndarray
    """
    # Just in case something goes wrong in the model function
    try:
        lik = -np.sum(np.log(y_err*np.sqrt(2*np.pi)) + (((y - m_func(r, *theta))**2) / (2*y_err**2)))
    except ZeroDivisionError:
        lik = np.NaN
    return lik


def log_uniform_prior(theta: np.ndarray, pr: List) -> float:
    """
    This function acts as a uniform prior. Using the limits for the parameters in the chosen
    model (either user defined or default), the function checks whether the passed theta values
    sit within those limits. If they do then of course probability is 1, so we return the natural
    log (as this is a log prior), otherwise the probability is 0, so return -infinity.

    :param np.ndarray theta: The knowledge we have (think theta in Bayesian parlance) - gets fed
        into the model we've chosen.
    :param List pr: A list of upper and lower limits for the parameters in theta, the limits of the
        uniform, uninformative priors.
    :return: The log prior value.
    :rtype: float
    """
    # Check whether theta values are within limits
    theta_check = [pr[t_ind][0] <= t <= pr[t_ind][1] for t_ind, t in enumerate(theta)]
    # If all parameters are within limits, probability is 1, thus log(p) is 0.
    if all(theta_check):
        ret_val = 0.0
    # Otherwise probability is 0, so log(p) is -inf.
    else:
        ret_val = -np.inf

    return ret_val


def log_prob(theta: np.ndarray, r: np.ndarray, y: np.ndarray, y_err: np.ndarray,
             m_func, pr) -> np.ndarray:
    """
    The combination of the log prior and log likelihood.

    :param np.ndarray theta: The knowledge we have (think theta in Bayesian parlance) - gets fed
        into the model we've chosen.
    :param np.ndarray r: The radii at which we have measured profile values.
    :param np.ndarray y: The values we have measured for the profile.
    :param np.ndarray y_err: The uncertainties on the measured profile values.
    :param m_func: The model function that is being fit to.
    :param List pr: A list of upper and lower limits for the parameters in theta, the limits of the
        uniform, uninformative priors.
    :return: The log probability value.
    :rtype: np.ndarray
    """
    lp = log_uniform_prior(theta, pr)
    if not np.isfinite(lp):
        ret_val = -np.inf
    else:
        ret_val = lp + log_likelihood(theta, r, y, y_err, m_func)

    if np.isnan(ret_val):
        ret_val = -np.inf

    return ret_val




