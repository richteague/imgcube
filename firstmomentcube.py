"""
Class for fitting the first moment maps.
"""

import functions
import numpy as np
from cube import imagecube


class firstmomentcube(imagecube):

    def __init__(self, path, mstar=None, inc=None, dist=None, vlsr=None,
                 clip=None, suppress_warnings=True):
        """Read in the first moment map."""
        imagecube.__init__(self, path, absolute=False, kelvin=False, clip=clip,
                           suppress_warnings=suppress_warnings)
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

    def fit_keplerian(self, p0=None, fit_mstar=True, beam=True, r_min=None,
                      r_max=None, z_type='thin', nearest=None, nwalkers=None,
                      nburnin=300, nsteps=300, scatter=1e-2, error=None,
                      plot_walkers=True, plot_corner=True, plot_bestfit=True,
                      return_samples=False, return_sampler=False):
        """
        Fit a Keplerian rotation profile to a first / ninth moment map.

        - Input -

        p0
        fit_mstar
        beam
        r_min
        r_max
        z_type
        nearest
        """

        # Load up emcee.
        try:
            import emcee
        except:
            raise ValueError("Cannot find emcee.")

        # Check the input parameters.
        z_type = z_type.lower()
        if z_type not in ['thin', 'conical', 'flared']:
            raise ValueError("Can only fit 'thin', 'conical' or 'flared'.")

        # Warning about the bounds.
        if r_max is None and self.verbose:
            print("WARNING: No r_max specified which may cause trouble.")

        # Find the starting positions.
        if p0 is None:
            if self.verbose:
                print("WARNING: Estimating starting values. May not work.")
            free_theta = self.mstar if fit_mstar else self.inc
            p0 = [0.0, 0.0, free_theta, self._estimate_PA(), self.vlsr * 1e3]
            if z_type == 'conical':
                p0 += [8.0]
                if nearest is None:
                    p0 += [0.0]
            elif z_type == 'flared':
                p0 += [0.3, 1.25]
                if nearest is None:
                    p0 += [0.0]
            if self.verbose:
                print("Have chosen:", p0)
        ndim = len(p0)
        nwalkers = 2 * ndim if nwalkers is None else nwalkers
        p0 = self._random_p0(np.squeeze(p0), scatter, nwalkers)

        # Define the labels.
        labels = [r'$x_0$', r'$y_0$', r'$M_{\star}$' if fit_mstar else r'$i$',
                  r'${\rm PA}$', r'$v_{\rm LSR}$']
        if z_type == 'conical':
            labels += [r'$\psi$']
        elif z_type == 'flared':
            labels += [r'$z\,/\,r$', r'$\phi$']
        if z_type != 'thin' and nearest is None:
            labels += [r'${\rm tilt}$']
        assert len(labels) == ndim, "Mismatch in p0 and labels."

        # Make sure there are errors
        if self.verbose and error is None:
            print("WARNING: No error specified. Assuming 0.1 km/s.")
        error = 0.1 if error is None else error
        error = np.ones(self.data.shape) * error
        if error.shape != self.data.shape:
            raise ValueError("RMS doesn't match data.shape.")

        # Run the MCMC.
        theta_fixed = self.inc if fit_mstar else self.mstar
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprobability,
                                        args=(theta_fixed, fit_mstar, error,
                                              beam, r_min, r_max, z_type,
                                              nearest))
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -int(nsteps):]
        samples = samples.reshape(-1, samples.shape[-1])
        samples[:, 3] = (samples[:, 3] + 360.) % 360.

        # Plotting here.
        if plot_walkers:
            functions.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)
        if plot_bestfit:
            self._plot_bestfit(samples, theta_fixed, fit_mstar, error, beam,
                               r_min, r_max, z_type, nearest)

        # Return the fits.
        if return_sampler:
            return sampler
        if return_samples:
            return samples
        return np.percentile(samples, [16, 50, 84], axis=0)

    # == emcee Functions == #

    def _lnprobability(self, theta, theta_fixed, fit_mstar, error, beam, r_min,
                       r_max, z_type, nearest):
        """Log-likelihood function for a thin disk."""

        # Unpack the standard variables.

        x0, y0 = theta[0], theta[1]
        if fit_mstar:
            mstar = theta[2]
            inc = self.inc
        else:
            inc = theta[2]
            mstar = self.mstar
        PA = theta[3]
        vlsr = theta[4]
        if z_type != 'thin' and nearest is None:
            tilt = theta[-1]
        else:
            tilt = .1 if nearest == 'north' else -.1

        # Check their priors.

        if 0.5 < abs(x0):
            return -np.inf
        if 0.5 < abs(y0):
            return -np.inf
        if not 0.0 < mstar < 5.0:
            return -np.inf
        if not 0.0 < inc < 90.0:
            return -np.inf
        if not -360. < PA < 360.:
            return -np.inf
        if not 0.0 < vlsr < 1e4:
            return -np.inf
        if not -0.3 <= tilt <= 0.3:
            return -np.inf

        # Model specific values.

        if z_type == 'conical':
            psi = theta[5]
            if not 0.0 <= psi <= 45.:
                return -np.inf
            params = [psi]

        elif z_type == 'flared':
            aspect_ratio = theta[5]
            flaring_angle = theta[6]
            if not 0.0 <= aspect_ratio <= 0.5:
                return -np.inf
            if not 0.0 <= flaring_angle <= 2.0:
                return -np.inf
            params = [aspect_ratio, flaring_angle]

        else:
            params = None

        # Make the (convolved) model.

        vkep = self._get_model(x0=x0, y0=y0, inc=inc, PA=PA, vlsr=vlsr,
                               mstar=mstar, z_type=z_type, params=params,
                               tilt=tilt, r_min=r_min, r_max=r_max, beam=beam)

        # Calculate chi-squared and return.

        mask = np.logical_and(np.isfinite(self.data), np.isfinite(vkep))
        lnx2 = np.power((self.data[mask] - vkep[mask]) / error[mask], 2)
        lnx2 -= np.log(error[mask]**2 * np.sqrt(2 * np.pi))
        lnx2 = -0.5 * np.nansum(lnx2)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _get_model(self, x0, y0, inc, PA, vlsr, mstar, z_type, params, tilt,
                   r_min, r_max, beam):
        """Return the Keplerian rotation model."""
        nearest = 'north' if tilt >= 0 else 'south'
        vkep = self.keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA, vlsr=vlsr,
                                      mstar=mstar, z_type=z_type,
                                      params=params, r_min=r_min, r_max=r_max,
                                      nearest=nearest)
        if beam:
            mask = np.isfinite(vkep)
            vkep = self._convolve_image(vkep / 1e3, self._beamkernel())
            vkep = np.where(mask, vkep, np.nan)
            return vkep
        return vkep / 1e3

    def _estimate_PA(self, clip=95):
        """Estimate the PA of the disk."""
        mask = self.data >= np.nanpercentile(self.data, [clip])
        angles = np.where(mask, self.disk_coords()[1], np.nan)
        return np.nanmean(np.degrees(angles))

    def _random_p0(self, p0, scatter, nwalkers):
        """Get the starting positions."""
        p0 = np.squeeze(p0)
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    # == Plotting Functions == #

    def _plot_bestfit(self, samples, theta_fixed, fit_mstar, error, beam,
                      r_min, r_max, z_type, nearest):
        """Plot the best fit moment map."""

        theta = np.median(samples, axis=0)

        x0, y0 = theta[0], theta[1]
        if fit_mstar:
            mstar = theta[2]
            inc = self.inc
        else:
            inc = theta[2]
            mstar = self.mstar
        PA = theta[3]
        vlsr = theta[4]
        if z_type != 'thin' and nearest is None:
            tilt = theta[-1]
        else:
            tilt = 1.0 if nearest == 'north' else -1.0
        if z_type == 'conical':
            params = [theta[5]]
        elif z_type == 'flared':
            params = [theta[5], theta[6]]
        else:
            params = None

        model = self._get_model(x0=x0, y0=y0, inc=inc, PA=PA, vlsr=vlsr,
                                mstar=mstar, z_type=z_type, params=params,
                                tilt=tilt, r_min=r_min, r_max=r_max, beam=beam)

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        im = ax.contourf(self.xaxis, self.yaxis, model, 30, cmap=cm.bwr)
        cb = plt.colorbar(im, pad=0.02, ticks=np.arange(-20, 20))
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
