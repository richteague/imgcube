"""
Class for fitting the first moment maps.
"""

import functions
import numpy as np
from cube import imagecube


class firstmomentcube(imagecube):

    def __init__(self, path, mstar=None, inc=None, dist=None, vlsr=None,
                 resample=0, clip=None, collapse='quadratic', excludepix=None,
                 verbose=True, suppress_warnings=True, error=None, **kwargs):
        """Read in the first moment map."""

        # Base class.
        imagecube.__init__(self, path, absolute=False, resample=resample,
                           kelvin=False, clip=clip, verbose=verbose,
                           suppress_warnings=suppress_warnings)

        # Collapse the cube if necessary.
        if self.data.ndim == 3:
            self.data = np.where(np.isfinite(self.data), self.data, 0.0)
            if excludepix is not None:
                pix = np.atleast_1d(excludepix)
                if pix.size == 1:
                    mask = self.data >= pix[0]
                elif pix.size == 2:
                    mask = np.logical_and(self.data <= pix[0],
                                          self.data >= pix[1])
                self.data = np.where(mask, self.data, 0.0)
            self.data, self.error = self._collapse_cube(method=collapse,
                                                        **kwargs)
        else:
            if error is None and self.verbose:
                print("WARNING: No error specified. Assuming 0.1 km/s.")
            self.error = error if error is not None else 0.1

        # Populate the parameters.
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
                print("WARNING: Converting VLSR in [m/s] to [km/s].")
                self.vlsr *= 1e-3

        # Calculate the uncertainties.
        self.error = np.where(np.isfinite(self.error), self.error, 1e10)
        self.ivar = self._calculate_ivar([0.0, 0.0, 0.0, 0.0])

    def fit_keplerian(self, p0=None, fit_mstar=True, beam=True, r_min=None,
                      r_max=None, z_type='thin', nearest=None, nwalkers=None,
                      nburnin=300, nsteps=300, scatter=1e-2,
                      plot_walkers=True, plot_corner=True, plot_bestfit=True,
                      plot_residual=True, return_samples=False,
                      return_sampler=False, optimize=True, **kwargs):
        """
        Fit a Keplerian rotation profile to a first or ninth moment map. Using
        a ninth moment map is better as it will be less contaminated from the
        far side of the disk. However, for low inclination disks observed with
        a low spectral resolution the first moment map be the only option.

        You can also consider a range of emission surface geometries which will
        be necessary to explain the emission morphology for molecular lines
        which arise from the upper layers in the disk such as 12CO. Available
        functions are: 'thin', a geometrically thin disk; 'conical', a double
        cone model as in Rosenfeld et al. (2013) defined by an opening angle
        between the emission surface and the midplane, and 'flared', where the
        surface is a power-law function normalised at r = 1". If a 3D geometry
        is chosen then you can set which side of the disk is nearest to the
        observer, either 'north' or 'south'. If this is not given then it will
        be left as a free parameter 'tilt' where 'tilt' > 0 if the northern
        side is nearest and <= 0 for the southern side.

        - Input -

        p0              : Starting positions for the MCMC. If none is specified
                          then they are populated with reasonable guesses.
        fit_mstar       : To break the Mstar*sin(i) degeneracy, only one of the
                          two properties can be fit. If `fit_mstar = True` then
                          the inclination is held fixed, otherwise the stellar
                          mass is.
        beam            : Include a convolution of the synthesied beam. This is
                          not strictly correct but provides a better estimate
                          than none at all.
        r_min           : [optional] Minimum radius (in disk coordaintes) to
                          fit (arcsec).
        r_max           : Maximum radius (in disk coordaintes) to fit (arcsec).
                          Due to issues with convolution and NaNs, this should
                          be less than the noisy edges of the data.
        z_type          : Type of 3D geometry to assume.
        nearest         : [optional] Which side of the disk is closest to the
                          observer if known.
        nwalkers        : Number of walkers to use, by default is 2 * len(p0).
        nburnin         : Number of steps taken to burn in the walkers.
        nsteps          : Number of steps used to sample the posterior.
        scatter         : Scatter used to disperse the p0 values.
        error           : Error on the data in (km/s).
        plot_walkers    : Plot the sampling for each parameter.
        plot_corner     : Plot the covariances of the posteriors.
        plot_bestfit    : Plot the best-fit model.
        return_samples  : Return all the samples from the MCMC chains.
        return_sampler  : Return the emcee sampler.

        - Returns -

        percentiles     : [default] The [16, 50, 84]th percentiles of the
                          posterior distributions.
        samples         : [return_samples = True] All samples of the posterior
                           distribution. Doesn't include burnin samples.
        sampler         : [return_sampler = True] The emcee sampler.
        """

        # Load up emcee.
        try:
            import emcee
        except:
            raise ValueError("Cannot find emcee.")

        # Initial starting positions.
        p0, theta_fixed = self._guess_p0(p0=p0, fit_mstar=fit_mstar,
                                         z_type=z_type, nearest=nearest)
        self.ivar = self._calculate_ivar(p0=p0, r_min=r_min, r_max=r_max)

        # Optimize positions if requested.
        args = (theta_fixed, fit_mstar, beam, z_type, nearest)
        if optimize:
            p0 = self._optimize_p0(p0, *args)
            self.ivar = self._calculate_ivar(p0=p0, r_min=r_min, r_max=r_max)
        elif self.verbose:
            print("No optimization calls requested.")

        # Set up the MCMC parameters.
        ndim = len(p0)
        nwalkers = 2 * ndim if nwalkers is None else nwalkers
        init = np.squeeze(p0)
        p0 = self._random_p0(init, scatter, nwalkers)

        # Make sure all starting positions are valid.
        mask = np.ones(len(p0), dtype=bool)
        lp = np.empty(len(p0))
        while np.any(mask):
            p0[mask] = self._random_p0(init, scatter, mask.sum())
            lp[mask] = [self._lnprobability(p00, *args) for p00 in p0[mask]]
            mask = ~np.isfinite(lp)

        # Define the labels for plotting.
        labels = [r'$x_0$', r'$y_0$', r'$M_{\star}$' if fit_mstar else r'$i$',
                  r'${\rm PA}$', r'$v_{\rm LSR}$']
        if z_type == 'conical':
            labels += [r'$\psi$']
        elif z_type == 'flared':
            labels += [r'$z\,/\,r$', r'$\phi$']
        if z_type != 'thin' and nearest is None:
            labels += [r'${\rm tilt}$']
        if len(labels) != ndim:
            raise ValueError("Mismatch in p0 and labels.")

        # Run the MCMC.
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprobability,
                                        args=args)
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -int(nsteps):]
        samples = samples.reshape(-1, samples.shape[-1])
        samples[:, 3] = samples[:, 3] % 360.

        # Plotting the results.
        if plot_walkers:
            functions.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)
        if plot_bestfit:
            self._plot_bestfit(samples, theta_fixed, fit_mstar, beam,
                               z_type, nearest)
        if plot_residual:
            self._plot_bestfit(samples, theta_fixed, fit_mstar, beam,
                               z_type, nearest, residual=True)

        # Return the fits.
        if return_sampler:
            return sampler
        if return_samples:
            return samples
        return np.percentile(samples, [16, 50, 84], axis=0)

    def _guess_p0(self, p0=None, fit_mstar=True, z_type='thin', nearest=None):
        """Returns estimated [p0, theta_fixed] based on the data."""

        # Check the input parameters.
        z_type = z_type.lower()
        if z_type not in ['thin', 'conical', 'flared']:
            raise ValueError("Can only fit 'thin', 'conical' or 'flared'.")
        theta_fixed = self.inc if fit_mstar else self.mstar

        # Find the starting positions.
        if p0 is None:
            if self.verbose:
                print("WARNING: Estimating starting values. May not work.")
            free_theta = self.mstar if fit_mstar else self.inc
            p0 = [0.0, 0.0, free_theta, self._estimate_PA(),
                  np.nanmedian(self.data) * 1e3]
            if z_type == 'conical':
                p0 += [8.0]
                if nearest is None:
                    p0 += [0.0]
            elif z_type == 'flared':
                p0 += [0.2, 1.25]
                if nearest is None:
                    p0 += [0.0]
            if self.verbose:
                print("Estimated starting positions:")
                print(p0)
        return p0, theta_fixed

    def _calculate_ivar(self, p0, r_min=None, r_max=None):
        """Calculate the inverse variance including radius mask."""
        ivar = np.power(self.error, -2.0)
        r_min = 0.0 if r_min is None else r_min
        r_max = np.inf if r_max is None else r_max
        rvals = self.disk_coords(p0[0], p0[1], self.inc, p0[3])[0]
        mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        ivar[~mask] = 0.0
        return ivar

    def _optimize_p0(self, p0, *args):
        """Optimize the starting positions."""

        from scipy.optimize import minimize

        # Define negative log-likelihood function.
        def nlnL(theta):
            return -self._lnprobability(theta, *args)

        # Simple bounds for the minimization.
        bounds = [(None, None) for _ in p0]
        bounds[0] = (-0.5, 0.5)
        bounds[1] = (-0.5, 0.5)
        bounds[2] = (0.3, 3.0)
        bounds[3] = (0.0, 360.0)
        bounds[4] = (self.vlsr * 1e3 - 1e3, self.vlsr * 1e3 + 1e3)

        res = minimize(nlnL, x0=p0, method='TNC', bounds=bounds,
                       options={'maxiter': 100000, 'ftol': 1e-5})
        p0 = res.x
        p0[3] = p0[3] % 360.

        if res.success:
            print("Optimized starting positions:")
            print(p0)
        elif self.verbose:
            print("WARNING: scipy.optimize did not converge.")
            print("\t Paramaters may not be optical.\n")
            print res
        return p0

    # == emcee Functions == #

    def _lnprobability(self, theta, theta_fixed, fit_mstar, beam, z_type,
                       nearest):
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
        if abs(vlsr - self.vlsr * 1e3) > 1e3:
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
                               tilt=tilt, beam=beam)

        # Calculate chi-squared and return.
        lnx2 = np.power((self.data - vkep), 2) * self.ivar
        lnx2 = -0.5 * np.sum(lnx2)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _get_model(self, x0, y0, inc, PA, vlsr, mstar, z_type, params, tilt,
                   beam):
        """Return the Keplerian rotation model."""
        nearest = 'north' if tilt >= 0 else 'south'
        vkep = self.keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA, vlsr=vlsr,
                                      mstar=mstar, z_type=z_type,
                                      dist=self.dist, params=params,
                                      nearest=nearest)
        if beam:
            vkep = self._convolve_image(vkep, self._beamkernel())
        return vkep / 1e3

    def _random_p0(self, p0, scatter, nwalkers):
        """Get the starting positions."""
        p0 = np.squeeze(p0)
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    # == Collapsing Functions == #

    def _collapse_cube(self, method='ninth', **kwargs):
        """
        Collapse the cube based on the method:
        first / average : Intensity weighted average velocity.
        ninth / max     : Velocity of the peak pixel in each spectrum.
        quadratic       : Parabolic fit to the pixel and neighbouring pixels.
        """

        if method == 'first' or method == 'average':
            vlos, uncertainty = self._collapse_cube_first()
        elif method == 'ninth' or method == 'max':
            vlos, uncertainty = self._collapse_cube_ninth()
        elif method == 'quadratic':
            vlos, uncertainty = self._collapse_cube_quad()
        return vlos, uncertainty

    def _collapse_cube_first(self):
        """Traditional first moment map."""
        vel = self.velax[:, None, None] * np.ones(self.data.shape)
        weights = 1e-10 * np.random.random(vel.size).reshape(vel.shape)
        weights = np.where(self.data != 0.0, self.data, weights)
        vlos = np.average(vel, axis=0, weights=weights)
        uncertainty = np.ones(vlos.shape) * self.chan
        return vlos / 1e3, uncertainty / 1e3

    def _collapse_cube_ninth(self):
        """Velocity coordinate of maximum velocity."""
        vlos = np.take(self.velax, np.argmax(self.data, axis=0))
        uncertainty = np.ones(vlos.shape) * self.chan
        return vlos / 1e3, uncertainty / 1e3

    def _collapse_cube_quad(self, **kwargs):
        """Parabolic fit to the pixel of peak intensity."""
        from bettermoments import quadratic
        kwargs['linewidth'] = kwargs.pop('linewidth', None)
        if kwargs['linewidth'] is not None:
            kwargs['linewidth'] /= self.chan
        rms = np.nanstd([self.data[:5], self.data[-5:]])
        vlos, uncertainty = quadratic(self.data, x0=self.velax[0],
                                      dx=self.chan, uncertainty=rms,
                                      **kwargs)[:2]
        return vlos / 1e3, uncertainty / 1e3

    # == Plotting Functions == #

    def _plot_bestfit(self, samples, theta_fixed, fit_mstar, beam, z_type,
                      nearest, residual=False):
        """Plot the best fit moment map."""

        if samples.ndim == 2:
            theta = np.median(samples, axis=0)
        else:
            theta = samples

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
                                tilt=tilt, beam=beam)

        if residual:
            model -= self.data
            levels = np.where(self.ivar != 0.0, model, np.nan)
            levels = np.nanpercentile(levels, [2, 98])
            levels = max(abs(levels[0]), abs(levels[1]))
            levels = np.array([-levels, levels])
            ticks = None
        else:
            levels = np.where(self.ivar != 0.0, self.data, np.nan)
            levels = np.nanpercentile(levels, [2, 98])
            ticks = np.arange(-20, 20)

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        levels = np.linspace(levels[0], levels[1], 30)
        im = ax.contourf(self.xaxis, self.yaxis, model,
                         levels, cmap=cm.RdBu_r, extend='both')
        ax.contour(self.xaxis, self.yaxis, self.ivar, [0], colors='k')
        cb = plt.colorbar(im, pad=0.02, ticks=ticks)
        cb.set_label('Line of Sight Velocity (km/s)', rotation=270,
                     labelpad=15)
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.3)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')

        if beam:
            from functions import plotbeam
            plotbeam(bmaj=self.bmaj, bmin=self.bmin, bpa=self.bpa, ax=ax)

        return model

    def plot_first_moment(self, ax=None, levels=None):
        """Plot the first moment map."""
        import matplotlib.cm as cm
        from functions import plotbeam
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        if levels is None:
            levels = np.percentile(self.data, [2, 98]) - self.vlsr
            levels = max(abs(levels[0]), abs(levels[1]))
            levels = self.vlsr + np.linspace(-levels, levels, 30)
        im = ax.contourf(self.xaxis, self.yaxis, self.data, levels,
                         cmap=cm.RdBu_r, extend='both')
        cb = plt.colorbar(im, pad=0.02, ticks=np.arange(100))
        cb.set_label('Line of Sight Velocity (km/s)', rotation=270,
                     labelpad=15)
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.3)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        plotbeam(bmaj=self.bmaj, bmin=self.bmin, bpa=self.bpa, ax=ax)

    def plot_uncertainties(self, ax=None, levels=None):
        import matplotlib.cm as cm
        from functions import plotbeam
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        if levels is None:
            levels = np.arange(0, 2.1, 0.05)
        im = ax.contourf(self.xaxis, self.yaxis, self.error, levels,
                         cmap=cm.RdBu_r, extend='both')
        cb = plt.colorbar(im, pad=0.02, ticks=np.arange(0, 2.1, 0.1))
        cb.set_label('Uncertainty (km/s)', rotation=270,
                     labelpad=15)
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.3)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        plotbeam(bmaj=self.bmaj, bmin=self.bmin, bpa=self.bpa, ax=ax)
