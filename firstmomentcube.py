"""
Class for fitting the first moment maps.
"""

import plotting
import numpy as np
from cube import imagecube
from functions import random_p0


class firstmomentcube(imagecube):

    def __init__(self, path, mstar=None, inc=None, dist=None, clip=None,
                 suppress_warnings=True):
        """Read in the first moment map."""
        imagecube.__init__(self, path, absolute=False, kelvin=False)
        if mstar is None:
            raise ValueError("WARNING: Must specify mstar [Msun].")
        self.mstar = mstar
        if inc is None:
            raise ValueError("WARNING: Must specify inclination [deg].")
        self.inc = inc
        if dist is None:
            raise ValueError("WARNING: Must specify distance [pc].")
        self.dist = dist
        if self.data.ndim != 2:
            raise ValueError("WARNING: Not a 2D image.")
        if clip is not None:
            self._clip_cube(clip)
        self.mask = np.isfinite(self.data)

        # Suppres warnings.
        if suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore")

    def fit_keplerian(self, p0=None, fit_Mstar=True, beam=True, r_min=None,
                      r_max=None, nwalkers=128, nburnin=200, nsteps=50,
                      scatter=1e-2, error=None, plot_walkers=True,
                      plot_corner=True, return_samples=False):
        """Fit a Keplerian rotation profile to the first moment map."""

        # Load up emcee.
        try:
            import emcee
        except:
            raise ValueError("Cannot find emcee.")

        # Find the starting positions.
        if p0 is None:
            print("WARNING: No starting values provided. May not converge.")
            p0 = [0.0, 0.0, 1.0 if fit_Mstar else 30., 45., 3e3]
        p0 = random_p0(np.squeeze(p0), scatter, nwalkers)

        # Run the sampler.
        error = 0.1 if error is None else error
        if np.atleast_1d(error).size != self.data.size:
            error = np.ones(self.data.shape) * error
        theta_fixed = self.inc if fit_Mstar else self.mstar
        args = (theta_fixed, fit_Mstar, error, beam, r_min, r_max)
        sampler = emcee.EnsembleSampler(nwalkers, 5, self._ln_probability,
                                        args=args)
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -nsteps:]
        samples = samples.reshape(-1, samples.shape[-1])

        # Diagnosis plots.
        labels = [r'$x_0$', r'$y_0$', r'$M_{\star}$' if fit_Mstar else r'$i$',
                  r'${\rm PA}$', r'$v_{\rm LSR}$']
        if plot_walkers:
            plotting.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            plotting.plot_corner(samples, labels)

        # Return the fits.
        if return_samples:
            return samples
        return np.percentile(samples, [16, 50, 84], axis=0)

    def _ln_probability(self, theta, theta_fixed, fit_Mstar, error,
                        beam=True, r_min=None, r_max=None):
        """Log-probability function for the MCMC fit."""
        if not np.isfinite(self._ln_prior(theta, theta_fixed, fit_Mstar)):
            return -np.inf
        return self._ln_likelihood(theta, theta_fixed, fit_Mstar, error,
                                   beam=True, r_min=None, r_max=None)

    def _ln_likelihood(self, theta, theta_fixed, fit_Mstar, error,
                       beam=True, r_min=None, r_max=None):
        """Log-likelihood function for the MCMC fit."""

        # Build the model.
        params = self._unpack_theta(theta, theta_fixed, fit_Mstar)
        model = self._get_model(params, beam).flatten()

        # Mask the data based on NaNs and inner and outer radii.
        x0, y0, inc, _, PA, _ = params
        rvals = self.disk_coordinates(x0=x0, y0=y0, inc=inc, PA=PA)
        mask = self.mask
        if r_max is not None:
            mask *= rvals <= r_max
        if r_max is not None:
            mask *= rvals >= r_min
        mask = mask.flatten()
        error = np.atleast_1d(error).flatten()

        # Chi-squared log-likelihood.
        lnx2 = (self.data.flatten()[mask] - model[mask]) / error[mask]
        lnx2 = np.power(lnx2, 2) - np.log(error[mask]**2 * np.sqrt(2 * np.pi))
        lnx2 = -0.5 * np.nansum(lnx2)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _get_model(self, params, beam):
        """Return the model rotation profile."""
        x0, y0, inc, Mstar, PA, vlsr = params
        vkep = self._keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA,
                                       mstar=Mstar, dist=self.dist,
                                       vlsr=vlsr) / 1e3
        if beam:
            return self._convolve_image(vkep, self._beamkernel())
        return vkep

    def _ln_prior(self, theta, theta_fixed, fit_Mstar):
        """Log-priors for the MCMC fit."""
        params = self._unpack_theta(theta, theta_fixed, fit_Mstar)
        x0, y0, inc, Mstar, PA, vlsr = params
        if 0.5 < abs(x0):
            return -np.inf
        if 0.5 < abs(y0):
            return -np.inf
        if not 0.0 < Mstar < 10.0:
            return -np.inf
        if not 0.0 < inc < 90.:
            return -np.inf
        if not -90.0 < PA < 450.:
            return -np.inf
        if not 0.0 < vlsr < 1e4:
            return -np.inf
        return 0.0

    def _unpack_theta(self, theta, theta_fixed, fit_Mstar):
        """Unpack the model parameters."""
        if fit_Mstar:
            x0, y0, Mstar, PA, vlsr = theta
            inc = theta_fixed
        else:
            x0, y0, inc, PA, vlsr = theta
            Mstar = theta_fixed
        return x0, y0, inc, Mstar, PA, vlsr
