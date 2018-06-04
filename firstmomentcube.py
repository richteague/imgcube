"""
Class for fitting the first moment maps.
"""

import functions
import numpy as np
from cube import imagecube
from functions import random_p0


class firstmomentcube(imagecube):

    def __init__(self, path, mstar=None, inc=None, dist=None, vlsr=None,
                 clip=None):
        """Read in the first moment map."""
        imagecube.__init__(self, path, absolute=False, kelvin=False, clip=clip)
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
        if vlsr is None:
            print("WARNING: No systemic velocity specified.")
            self.vlsr = np.nanmedian(self.data)
        else:
            self.vlsr = vlsr

    def fit_keplerian(self, p0=None, fit_Mstar=True, beam=True, r_min=None,
                      r_max=None, nwalkers=128, nburnin=200, nsteps=50,
                      scatter=1e-2, error=None, plot_walkers=True,
                      plot_corner=True, plot_fit=True, return_samples=False):
        """Fit a Keplerian rotation profile to the first moment map."""

        # Load up emcee.
        try:
            import emcee
        except:
            raise ValueError("Cannot find emcee.")

        # Find the starting positions.
        if p0 is None:
            print("WARNING: No starting values provided. May not converge.")
            free_theta = self.mstar if fit_Mstar else self.inc
            p0 = [0.0, 0.0, free_theta, 45., self.vlsr]
        p0 = random_p0(np.squeeze(p0), scatter, nwalkers)

        # Make sure the error is across the whole image.
        error = 0.1 if error is None else error
        error = np.ones(self.data.shape) * error
        if error.shape != self.data.shape:
            raise ValueError("RMS doesn't match data.shape.")

        # Run the sampler.
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
            functions.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)
        if plot_fit:
            self._plot_best_fit(samples, theta_fixed, fit_Mstar, beam)

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
        model = self._get_masked_model(theta=theta, theta_fixed=theta_fixed,
                                       fit_Mstar=fit_Mstar, beam=beam,
                                       r_min=r_min, r_max=r_max)
        mask = np.logical_and(np.isfinite(self.data), np.isfinite(model))
        lnx2 = np.power((self.data[mask] - model[mask]) / error[mask], 2)
        lnx2 -= np.log(error[mask]**2 * np.sqrt(2 * np.pi))
        lnx2 = -0.5 * np.nansum(lnx2)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _get_masked_model(self, theta, theta_fixed, fit_Mstar, beam=True,
                          r_min=None, r_max=None):
        """Return a masked model rotation profile in [km/s]."""
        params = self._unpack_theta(theta, theta_fixed, fit_Mstar)
        x0, y0, inc, _, PA, _ = params
        rvals = self.disk_coordinates(x0=x0, y0=y0, inc=inc, PA=PA)[0]
        r_max = self.xaxis.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min
        model = self._get_model(params, beam)
        mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        return np.where(mask, model, np.nan)

    def _get_model(self, params, beam):
        """Return the model rotation profile in [km/s]."""
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
        if not 0.0 < Mstar < 5.0:
            return -np.inf
        if not 0.0 < inc < 90.0:
            return -np.inf
        if not 0.0 < PA < 360.:
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

    def _plot_best_fit(self, samples, theta_fixed, fit_Mstar, beam, r_min=None,
                       r_max=None):
        """Plot the best fit moment map."""
        model = self._get_masked_model(theta=np.median(samples, axis=0),
                                       theta_fixed=theta_fixed,
                                       fit_Mstar=fit_Mstar, beam=beam,
                                       r_min=r_min, r_max=r_max)
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.contourf(self.xaxis, self.yaxis, model, 30, cmap=cm.bwr)
        cb = plt.colorbar(im, pad=0.02, ticks=np.arange(10))
        cb.set_label('Line of Sight Velocity (km/s)', rotation=270,
                     labelpad=15)
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.3)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')

        from functions import plotbeam
        plotbeam(bmaj=self.bmaj, bmin=self.bmin, bpa=self.bpa, ax=ax)

    def plot_first_moment(self, ax=None, levels=None):
        """Plot the first moment map."""
        import matplotlib.cm as cm
        from functions import plotbeam
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        if levels is None:
            levels = np.nanpercentile(self.data, [2, 98])
            levels = np.linspace(levels[0], levels[1], 30)
        im = ax.contourf(self.xaxis, self.yaxis, self.data, levels,
                         cmap=cm.bwr)
        cb = plt.colorbar(im, pad=0.02, ticks=np.arange(10))
        cb.set_label('Line of Sight Velocity (km/s)', rotation=270,
                     labelpad=15)
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.3)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        plotbeam(bmaj=self.bmaj, bmin=self.bmin, bpa=self.bpa, ax=ax)
