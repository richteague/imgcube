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
            print("WARNING: No systemic velocity [km/s] specified.")
            self.vlsr = np.nanmedian(self.data)
        else:
            self.vlsr = vlsr
            if self.vlsr > 1e3:
                print("WARNING: systemic velocity in [m/s], not [km/s].")

    def fit_keplerian_psi(self, p0=None, fit_Mstar=True, beam=True, r_min=None,
                          r_max=None, nwalkers=128, nburnin=200, nsteps=50,
                          scatter=1e-2, error=None, plot_walkers=True,
                          plot_corner=True, plot_fit=True,
                          return_samples=False):
        """
        Fit a Keplerian rotation profile to the first moment map including the
        height of the emission surface above the midplane. Best for 8th
        moment maps rather than first.
        """

        # Load up emcee.
        try:
            import emcee
        except:
            raise ValueError("Cannot find emcee.")

        # Warning about the no bounds.
        if r_max is None:
            print("WARNING: No r_max specified which may cause trouble.")

        # Find the starting positions.
        if p0 is None:
            print("WARNING: No starting values provided - may not converge.")
            free_theta = self.mstar if fit_Mstar else self.inc
            p0 = [0., 0., free_theta, self._estimate_PA(),
                  self.vlsr * 1e3, 8., 0.0]
            print("\t Have chosen:"), p0
            print("\t Can include them with the p0 argument.")
        p0 = random_p0(np.squeeze(p0), scatter, nwalkers)

        # Make sure the error is across the whole image.
        error = 0.1 if error is None else error
        error = np.ones(self.data.shape) * error
        if error.shape != self.data.shape:
            raise ValueError("RMS doesn't match data.shape.")

        # Run the sampler.
        theta_fixed = self.inc if fit_Mstar else self.mstar
        args = (theta_fixed, fit_Mstar, error, beam, r_min, r_max)
        sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1],
                                        self._ln_probability, args=args)
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -nsteps:]
        samples = samples.reshape(-1, samples.shape[-1])

        # Allows for PA to be negative.
        samples[:, 3] = (samples[:, 3] + 360.) % 360.

        # Diagnosis plots.
        labels = [r'$x_0$', r'$y_0$', r'$M_{\star}$' if fit_Mstar else r'$i$',
                  r'${\rm PA}$', r'$v_{\rm LSR}$', r'$\varphi$',
                  r'${\rm side}$']
        if plot_walkers:
            functions.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)
        if plot_fit:
            self._plot_best_fit(samples, theta_fixed, fit_Mstar, beam,
                                r_min, r_max)
            self._plot_residual(samples, theta_fixed, fit_Mstar, beam,
                                r_min, r_max)

        # Return the fits.
        if return_samples:
            return samples
        return np.percentile(samples, [16, 50, 84], axis=0)

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

        # Warning about the no bounds.
        if r_max is None:
            print("WARNING: No r_max specified which may cause trouble.")

        # Find the starting positions.
        if p0 is None:
            print("WARNING: No starting values provided - may not converge.")
            free_theta = self.mstar if fit_Mstar else self.inc
            p0 = [0., 0., free_theta, self._estimate_PA(), self.vlsr * 1e3]
            print("\t Have chosen:"), p0
            print("\t Can include them with the p0 argument.")
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

        # Allows for PA to be negative.
        samples[:, 3] = (samples[:, 3] + 360.) % 360.

        # Diagnosis plots.
        labels = [r'$x_0$', r'$y_0$', r'$M_{\star}$' if fit_Mstar else r'$i$',
                  r'${\rm PA}$', r'$v_{\rm LSR}$']
        if plot_walkers:
            functions.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)
        if plot_fit:
            self._plot_best_fit(samples, theta_fixed, fit_Mstar, beam,
                                r_min, r_max)
            self._plot_residual(samples, theta_fixed, fit_Mstar, beam,
                                r_min, r_max)

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
                                   beam=True, r_min=r_min, r_max=r_max)

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
        rvals = self.disk_coordinates(x0=params[0], y0=params[1],
                                      inc=params[2], PA=params[4])[0]
        r_max = self.xaxis.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min
        model = self._get_model(params, beam)
        mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        return np.where(mask, model, np.nan)

    def _get_model(self, params, beam):
        """Return the model rotation profile in [km/s]."""
        try:
            x0, y0, inc, Mstar, PA, vlsr = params
            vkep = self._keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA,
                                           mstar=Mstar, dist=self.dist,
                                           vlsr=vlsr) / 1e3
        except:
            x0, y0, inc, Mstar, PA, vlsr, psi, side = params
            vkep = self._keplerian_profile_psi(x0=x0, y0=y0, inc=inc, PA=PA,
                                               mstar=Mstar, dist=self.dist,
                                               vlsr=vlsr, psi=psi)
            vkep = vkep[0 if side >= 0 else 1] / 1e3
        if beam:
            return self._convolve_image(vkep, self._beamkernel())
        return vkep

    def _ln_prior(self, theta, theta_fixed, fit_Mstar):
        """Log-priors for the MCMC fit."""

        # Unpack the free parameters.
        params = self._unpack_theta(theta, theta_fixed, fit_Mstar)
        x0, y0, inc, Mstar, PA, vlsr, psi, side = params

        # Conditions.
        if 0.5 < abs(x0):
            return -np.inf
        if 0.5 < abs(y0):
            return -np.inf
        if not 0.0 < Mstar < 5.0:
            return -np.inf
        if not 0.0 < inc < 90.0:
            return -np.inf
        if not -360. < PA < 360.:
            return -np.inf
        if not 0.0 < vlsr < 1e4:
            return -np.inf
        if not 0.0 <= psi < 45.:
            return -np.inf
        if not -1.0 <= side <= 1.0:
            return -np.inf
        return 0.0

    def _unpack_theta(self, theta, theta_fixed, fit_Mstar):
        """Unpack the model parameters."""
        if fit_Mstar:
            try:
                x0, y0, Mstar, PA, vlsr, psi, side = theta
            except:
                x0, y0, Mstar, PA, vlsr = theta
                psi = 0.0
                side = 0.0
            inc = theta_fixed
        else:
            try:
                x0, y0, inc, PA, vlsr, psi, side = theta
            except:
                x0, y0, inc, PA, vlsr = theta
                psi = 0.0
                side = 0.0
            Mstar = theta_fixed
        return x0, y0, inc, Mstar, PA, vlsr, psi, side

    def _estimate_PA(self, clip=5):
        """Estimate the PA of the disk."""
        t = np.where(self.data <= np.nanpercentile(self.data, [clip]),
                     self.disk_coordinates(0.0, 0.0, self.inc, 0.0)[1], np.nan)
        return np.nanmean((np.degrees(t) + 360.) % 360.) + 45.

    # == Plotting Functions == #

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
        if r_max is None:
            ax.set_xlim(self.xaxis.max(), self.xaxis.min())
            ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        else:
            ax.set_xlim(1.1 * r_max, -1.1 * r_max)
            ax.set_ylim(-1.1 * r_max, 1.1 * r_max)
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')

        if beam:
            from functions import plotbeam
            plotbeam(bmaj=self.bmaj, bmin=self.bmin, bpa=self.bpa, ax=ax)

    def _plot_residual(self, samples, theta_fixed, fit_Mstar, beam, r_min=None,
                       r_max=None):
        """Plot the residuals for the best fit model."""
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        median = np.median(samples, axis=0)
        model = self._get_masked_model(theta=median,
                                       theta_fixed=theta_fixed,
                                       fit_Mstar=fit_Mstar, beam=beam,
                                       r_min=r_min, r_max=r_max)

        # Find the bounds of the model.
        residual = (self.data - model)
        residual /= np.sign(self.data - median[-1] / 1e3)
        vmax = min(np.nanmax(abs(residual)), 0.5)
        vmin = -vmax
        tick = np.floor(vmax * 10. / 2.5) / 10.
        tick = np.arange(np.floor(vmin * tick) / tick, vmax+1, tick)

        # Plot the figure.
        fig, ax = plt.subplots()
        im = ax.contourf(self.xaxis - median[0], self.yaxis - median[1],
                         residual, levels=np.linspace(vmin, vmax, 30),
                         cmap=cm.RdBu_r, extend='both', vmin=vmin, vmax=vmax)
        cb = plt.colorbar(im, pad=0.02, ticks=tick)
        cb.set_label(r'${\rm Obsevations - Model \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)

        # Plot the disk center.
        ax.scatter(0.0, 0.0, color='k', marker='x')

        # Set up the axes.
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.3)
        if r_max is None:
            ax.set_xlim(self.xaxis.max(), self.xaxis.min())
            ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        else:
            ax.set_xlim(1.1 * r_max, -1.1 * r_max)
            ax.set_ylim(-1.1 * r_max, 1.1 * r_max)
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')

        if beam:
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
                         cmap=cm.RdBu_r, extend='both')
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
