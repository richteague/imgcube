"""
Random functions to help with the analysis.
"""

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def _keplerian(rvals, inc, mstar, dist):
    """Keplerian rotation with stellar mass as free parameter."""
    vkep = np.sqrt(sc.G * mstar * 1.989e30 / rvals / sc.au / dist)
    return vkep * np.sin(np.radians(inc))


def plotbeam(bmaj, bmin=None, bpa=0.0, ax=None, **kwargs):
    """Plot a beam. Input must be same units as axes. PA in degrees E of N."""
    if ax is None:
        fig, ax = plt.subplots()
    if bmin is None:
        bmin = bmaj
    if bmin > bmaj:
        temp = bmin
        bmin = bmaj
        bmaj = temp
    offset = kwargs.get('offset', 0.125)
    ax.add_patch(Ellipse(ax.transLimits.inverted().transform((offset, offset)),
                         width=bmin, height=bmaj, angle=-bpa,
                         fill=False, hatch=kwargs.get('hatch', '////////'),
                         lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                         color=kwargs.get('color', kwargs.get('c', 'k'))))
    return


def plotscale(scale, dx=0.9, dy=0.9, ax=None, text=None, text_above=True,
              **kwargs):
    """Plot a linear scale on the provided axes."""

    # Generate axes if not provided.
    if ax is None:
        fig, ax = plt.subplots()

    # Draw the scale bar.
    x, y = ax.transLimits.inverted().transform((dx, dy))
    ax.errorbar(x, y, xerr=0.5*scale, capsize=1.5, capthick=1.0,
                lw=kwargs.get('linewidth', kwargs.get('lw', 1.0)),
                color=kwargs.get('color', kwargs.get('c', 'k')))

    # Include the labelling.
    if text:
        if text_above:
            x, y = ax.transLimits.inverted().transform((dx, 1.2 * dy))
        else:
            x, y = ax.transLimits.inverted().transform((dx, 0.8 * dy))
        text = text if type(text) is not bool else '%.2f' % scale
        ax.text(x, y, text, ha='center', va='bottom' if text_above else 'top',
                fontsize=kwargs.get('fontsize', kwargs.get('fs', 7.0)),
                color=kwargs.get('color', kwargs.get('c', 'k')))
    return


def plot_walkers(samples, nburnin=None, labels=None):
    """Plot the walkers to check if they are burning in."""

    # Import matplotlib.
    import matplotlib.pyplot as plt

    # Check the length of the label list.
    if labels is None:
        if samples.shape[0] != len(labels):
            raise ValueError("Not correct number of labels.")

    # Cycle through the plots.
    for s, sample in enumerate(samples):
        fig, ax = plt.subplots()
        for walker in sample.T:
            ax.plot(walker, alpha=0.1)
        ax.set_xlabel('Steps')
        if labels is not None:
            ax.set_ylabel(labels[s])
        if nburnin is not None:
            ax.axvline(nburnin, ls=':', color='r')


def plot_corner(samples, labels=None, quantiles=[0.16, 0.5, 0.84]):
    """Plot the corner plot to check for covariances."""
    import corner
    corner.corner(samples, labels=labels,
                  quantiles=quantiles, show_titles=True)


def percentiles_to_errors(pcnts):
    """Covert [16, 50, 84]th percentiles to [y, -dy, +dy]."""
    pcnts = np.squeeze([pcnts])
    if pcnts.ndim > 1:
        if pcnts.shape[1] != 3 and pcnts.shape[0] == 3:
            pcnts = pcnts.T
        if pcnts.shape[1] != 3:
            raise TypeError("Must provide a Nx3 or 3xN array.")
        return np.array([[p[1], p[1]-p[0], p[2]-p[1]] for p in pcnts]).T
    return np.squeeze([pcnts[1], pcnts[1]-pcnts[0], pcnts[2]-pcnts[1]])


def sort_arrays(x, y, dy=None):
    """Sort the data for monotonically increasing x."""
    idx = np.argsort(x)
    if dy is None:
        return x[idx], y[idx]
    return x[idx], y[idx], dy[idx]


def running_variance(y, window=5, x=None):
    """Calculate the running variance using a simple window."""

    # Define the window size.
    window = int(window)
    if window >= len(y):
        raise ValueError("Window too big.")

    # Sort the data if x values provided.
    if x is not None:
        x, y = sort_arrays(x, y)

    # Include dummy values.
    pad_low = y[0] * np.ones(window)
    pad_high = y[-1] * np.ones(window)
    y_pad = np.concatenate([pad_low, y, pad_high])

    # Define the window indices.
    a = int(np.ceil(window / 2))
    b = window - a

    # Loop through and calculate.
    var = [np.nanvar(y_pad[i-a:i+b]) for i in range(window, len(y) + window)]
    return np.squeeze(var)


def running_stdev(y, window=5, x=None):
    """Calculate the running standard deviation within a window."""
    return running_variance(y, window, x)**0.5


def running_mean(y, window=5, x=None):
    """Calculate the running mean using a simple window."""

    # Define the window size.
    window = int(window)
    if window >= len(y):
        raise ValueError("Window too big.")

    # Sort the data if x values provided.
    if x is not None:
        x, y = sort_arrays(x, y)

    # Include dummy values.
    pad_low = y[0] * np.ones(window)
    pad_high = y[-1] * np.ones(window)
    y_pad = np.concatenate([pad_low, y, pad_high])

    # Define the window indices.
    a = int(np.ceil(window / 2))
    b = window - a

    # Loop through and calculate.
    mu = [np.nanmean(y_pad[i-a:i+b]) for i in range(window, len(y) + window)]
    return np.squeeze(mu)


def Matern32_model(x, y, dy, fit_mean=True, jitter=True, oversample=True,
                   return_var=True, verbose=False):
    """
    Return a model using a Matern 3/2 kernel. Most simple model possible.

    - Input -

    x:          Sampling locations.
    y:          Observations.
    dy:         Uncertainties.
    fit_mean:   Fit the mean of the observations.
    jitter:     Include a Jitter term in the GP model.
    oversample: If true, sample the GP at a higher resolution. If true will use
                a default of 5, otherwise a value can be specified.
    return_var: Return the variance of the fit.
    verbose:    Print out the best-fit values.

    - Returns -

    xx:         Sampling points of the GP
    yy:         Samples of the GP.
    zz:         Variance of the GP fit if return_var is True.
    """

    # Import necessary packages.
    try:
        import celerite
        from scipy.optimize import minimize
    except:
        raise ValueError("celerite or scipy.optimize.minimize not found.")

    # Make sure arrays are ordered.
    x, y, dy = sort_arrays(x, y, dy)

    # Define the Kernel.
    lnsigma, lnrho = np.log(np.nanstd(y)), np.log(np.nanmean(abs(np.diff(x))))
    kernel = celerite.terms.Matern32Term(log_sigma=lnsigma, log_rho=lnrho)

    if jitter:
        if np.nanmean(dy) != 0.0:
            noise = celerite.terms.JitterTerm(log_sigma=np.log(np.nanmean(dy)))
        else:
            noise = celerite.terms.JitterTerm(log_sigma=np.log(np.nanstd(y)))
        kernel = kernel + noise
    gp = celerite.GP(kernel, mean=np.nanmean(y), fit_mean=fit_mean)
    gp.compute(x, dy)

    # Minimize the results.
    params = gp.get_parameter_vector()
    params += 1e-2 * np.random.randn(len(params))
    soln = minimize(neg_log_like, params, jac=grad_neg_log_like,
                    args=(y, gp), method='L-BFGS-B')
    gp.set_parameter_vector(soln.x)

    # Define the new sampling rate for the GP.
    if oversample:
        if type(oversample) is float or type(oversample) is int:
            xx = np.linspace(x[0], x[-1], oversample * x.size)
        else:
            xx = np.linspace(x[0], x[-1], 5. * x.size)
    else:
        xx = x

    if soln.success:
        if verbose:
            print 'Solution:', soln.x
        yy = gp.predict(y, xx, return_cov=False, return_var=return_var)
        if return_var:
            return xx, yy[0], yy[1]**0.5
        return xx, yy
    else:
        if verbose:
            print 'No Solution.'
        if return_var:
            return x, y, np.zeros(x.size)
        return x, y


def neg_log_like(params, y, gp):
    """Negative log-likelihood fucntion."""
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)


def grad_neg_log_like(params, y, gp):
    """Gradient of the negative log-likelihood function."""
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]


def gaussian(x, x0, dx, A):
    """Gaussian function with Doppler width."""
    return A * np.exp(-np.power((x-x0) / dx, 2))


def offsetSHO(theta, A, y0):
    """Simple harmonic oscillator with an offset."""
    return A * np.cos(theta) + y0


def random_p0(p0, scatter, nwalkers):
    """Get the starting positions."""
    p0 = np.squeeze(p0)
    dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
    dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
    return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)


def solve_quadratic(x, y, inc_in, psi_in):
    """Solve Eqn.(5) from Rosenfeld et al. (2013)."""
    inc, psi = np.radians(inc_in), np.radians(psi_in)
    a = np.cos(2.*inc) + np.cos(2.*psi)
    b = -4. * np.sin(psi)**2 * y * np.tan(inc)
    c = -2. * np.sin(psi)**2 * (x**2 + np.power(y / np.cos(inc), 2))
    t_p = -b + np.sqrt(b**2 - 4 * a * c) / 2. / a
    t_n = -b - np.sqrt(b**2 - 4 * a * c) / 2. / a
    return t_p, t_n
