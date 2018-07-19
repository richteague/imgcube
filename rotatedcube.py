"""
Class for rotated cubes (with their major axis aligned with the xaxis.).
This has the functionaility to infer the emission surface following Pinte et
al (2018) with improvements.

TODO: Check how to recover the far side of the disk.
"""

import emcee
import celerite
import numpy as np
from cube import imagecube
import functions
from functions import offsetSHO
from functions import gaussian
from functions import sort_arrays
from functions import random_p0
from functions import running_stdev
from functions import Matern32_model
from functions import plotscale
from functions import percentiles_to_errors
from detect_peaks import detect_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
import scipy.constants as sc


class rotatedcube(imagecube):

    def __init__(self, path, inc=None, mstar=None, dist=None, x0=0.0, y0=0.0,
                 verbose=True, clip=None):
        """Read in the rotated image cube."""

        # Initilize the class.
        imagecube.__init__(self, path, absolute=False, kelvin=True, clip=clip)
        self.verbose = verbose

        # Get the deprojected pixel values assuming a thin disk.
        self.x0, self.y0 = x0, y0
        if inc is None:
            raise ValueError("WARNING: No inclination specified.")
        self.inc = inc
        if not 0 <= self.inc <= 90:
            raise ValueError("Inclination must be 0 <= i <= 90.")
        if dist is None:
            raise ValueError("WARNING: No distance specified.")
        self.dist = dist
        if mstar is None:
            raise ValueError("WARNING: No stellar mass specified.")
        self.mstar = mstar
        self.rdisk, self.tdisk = self.disk_coordinates(x0, y0, self.inc, 0.0)

        # Define the surface.
        self.rbins, self.rvals = self._radial_sampling()
        self.zvals = np.zeros(self.rvals.size)

        return

    # == Spectral Deprojection == #

    def get_deprojected_spectra(self, vrot, rbins=None, rpnts=None,
                                include_height=True, method='bin', PA_min=None,
                                PA_max=None, exclude_PA=False):
        """
        Return the deprojected spectra using the provided rotation profile.
        """

        # Populate variables.
        if method.lower() not in ['bin', 'gp']:
            raise ValueError("Method must be ['bin', 'GP']")

        # Deprojected pixel coordinates.
        if include_height:
            rvals, tvals = self.disk_coordinates_3D()
        else:
            rvals, tvals = self.disk_coordinates(self.x0, self.y0, self.inc)
        rvals, tvals = rvals.flatten(), tvals.flatten()
        if rbins is None and rvals is None:
            print("WARNING: No radial sampling set, this will take a while.")
        rbins, rpnts = self._radial_sampling(rbins=rbins, rvals=rpnts)

        # Flatten the data to [velocity, nxpix * nypix].
        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        if dvals.shape != (self.velax.size, self.nxpix * self.nypix):
            raise ValueError("Wrong data shape.")

        # Deproject the spectra.
        deprojected = []
        for r in range(1, rbins.size):

            # Get spectra and deproject.
            mask = self._get_mask(r_min=rbins[r-1], r_max=rbins[r],
                                  PA_min=PA_min, PA_max=PA_max,
                                  exclude_PA=exclude_PA).flatten()
            spectra, angles = dvals[:, mask].T, tvals[mask]
            spectra = self._deproject_spectra(spectra, angles, vrot[r-1])

            # Collapse to a single spectrum.
            if method == 'bin':
                deprojected += [np.nanmean(spectra, axis=0)]
            else:
                noise = np.nanstd(self.data[0]) * np.ones(spectra.shape)
                velax = self.velax[None, :] * np.ones(spectra.shape)
                x, y, _ = Matern32_model(velax.flatten(), spectra.flatten(),
                                         noise.flatten(), oversample=False)
                deprojected += [interp1d(x, y, bounds_error=False,
                                fill_value='extrapolate')(self.velax)]
        return np.squeeze(deprojected)

    def _deprojected_width(self, vrot, spectra, angles, resample=True):
        """Return the width of the deprojected line profile."""
        x, y = self._deprojected_spectrum(spectra, angles, vrot, resample)
        Tb = np.max(y)
        dV = np.trapz(y, x) / Tb / np.sqrt(np.pi)
        x0 = x[y.argmax()]
        p0 = [Tb, dV, x0]
        try:
            return curve_fit(gaussian, x, y, p0=p0, maxfev=10000)[0][1]
        except:
            return 1e50

    def _deprojected_spectrum(self, spectra, angles, vrot, resample=True):
        """Collapsed deprojected spectrum."""
        deprojected = self._deproject_spectra(spectra, angles, vrot)
        if resample:
            return self.velax, np.nanmean(deprojected, axis=0)
        velax = self.velax[None, :] * np.ones(deprojected.shape)
        return sort_arrays(velax.flatten(), deprojected.flatten())

    def _deproject_spectra(self, spectra, angles, vrot):
        """Deproject all the spectra to a common systemic velocity."""
        # deprojected = [interp1d(self.velax - vrot * np.cos(angle), spectrum,
        #                         fill_value='extrapolate')(self.velax)
        deprojected = [interp1d(self.velax - vrot * np.cos(angle), spectrum,
                                fill_value=np.nan,
                                bounds_error=False)(self.velax)
                       for spectrum, angle in zip(spectra, angles)]
        return np.squeeze(deprojected)

    # == Temperature Profiles == #

    def get_temperature_profile(self, spectra, method='peak', fit=True,
                                errors=False, verbose=True, mu=28.,
                                emcee_kwargs={}, plot_walkers=False,
                                plot_corner=False):
        """
        Return the temperature profile from the spectra.

        - Inputs -

        spectra:        Array of the spectra to fit. Should all be on the
                        attached velocity axis.
        method:         Method to measure the temperature: 'peak' uses the line
                        peak, 'width', the line width and 'combined' uses both.
        fit:            Whether the peak and width are determined by fitting an
                        analytical form or numerically.
        errors:         Whether to return uncertainties on the derived values.
                        If True, use emcee to sample the posterior
                        distribution, otherwise use curve_fit which is much
                        faster.
        verbose:        Print out which number spectrum is being fit.
        mu:             Mean molecular weight of the emitting molecule.
        emcee_kwargs:   Dictionary of parameters used for emcee: nwalkers,
                        nburnin, nsteps, scatter.
        plot_walkers:   Plot the walkers for emcee runs?
        plot_corner:    Plot the corner plot for emcee runs?

        - Returns -

        temperature:    The derived temperature (and uncertainty if requested)
                        for each spectrum. If method != 'combined' then only a
                        single temperature will be returned.
        """

        # Check inputs.
        if self.bunit != 'k':
            print("WARNING: Intensity in Jy/beam, not K.")
        if method.lower() not in ['width', 'peak', 'combined']:
            raise ValueError("method must be ['width', 'peak', 'combined'].")
        if type(fit) is not bool:
            raise ValueError("fit must be True or False.")
        if fit and method == 'combined':
            raise ValueError("Must have fit=True for method='combine'.")

        # Make sure the spectra array shape is correct.
        spectra = np.atleast_2d(spectra)
        if spectra.shape[0] == self.velax.size:
            spectra = spectra.T
        if spectra.shape[1] != self.velax.size:
            raise ValueError("Wrong shape of spectra arrays.")

        # Loop through each spectrum.
        temp = []
        for s, spectrum in enumerate(spectra):
            if verbose:
                print("Running spectrum %d / %d." % (s+1, spectra.shape[0]))
            if fit:
                if errors:
                    if method == 'combined':
                        temp += [self._emcee_combined(spectrum, mu,
                                                      emcee_kwargs,
                                                      plot_walkers,
                                                      plot_corner)]
                    else:
                        temp += [self._emcee_individual(spectrum, mu,
                                                        emcee_kwargs,
                                                        plot_walkers,
                                                        plot_corner)]
                else:
                    if method == 'combined':
                        temp += [self._curvefit_combined(spectrum, mu)]
                    else:
                        temp += [self._curvefit_individual(spectrum, mu)]
            else:
                if method == 'peak':
                    temp += [self._numerical_peak(spectrum)]
                elif method == 'width':
                    temp += [self._numerical_width(spectrum)]
        return np.squeeze(temp)

    def _numerical_peak(self, spectrum):
        """Returns peak of the spectrum."""
        return np.nanmax(spectrum)

    def _numerical_width(self, spectrum):
        """Returns the Doppler width of spectrum assuming a Gaussian."""
        area = np.trapz(spectrum, self.velax)
        peak = self._numerical_peak(spectrum)
        return area / peak / np.sqrt(np.pi)

    def _curvefit_individual(self, spectrum, mu=28.):
        """Fit a Gaussian where Tb and dV are independent."""
        def gaussian_combined(x, x0, Tkin, Tb):
            dV = np.sqrt(2. * sc.k * Tkin / mu / sc.m_p)
            return gaussian(x, x0, dV, Tb)
        p0 = [self.velax[spectrum.argmax()],
              self._numerical_peak(spectrum),
              self._numerical_peak(spectrum)]
        popt, _ = curve_fit(gaussian, self.velax, spectrum,
                            p0=p0, maxfev=1000)
        return popt[1], popt[2]

    def _curvefit_combined(self, spectrum, mu=28.):
        """Fit a Gaussian where Tb and dV are coupled."""
        def gaussian_combined(x, x0, Tkin):
            dV = np.sqrt(2. * sc.k * Tkin / mu / sc.m_p)
            return gaussian(x, x0, dV, Tkin)
        p0 = [self.velax[spectrum.argmax()], self._numerical_peak(spectrum)]
        popt, _ = curve_fit(gaussian_combined, self.velax, spectrum,
                            p0=p0, maxfev=10000)
        return popt[1]

    def _ln_emcee(self, theta, spectrum, uncertainty, mu=28.):
        """Log-probability function for emcee."""

        # Unpack properties.
        try:
            x0, Tkin, Tb = theta
        except:
            x0, Tb = theta
            Tkin = Tb
        dV = np.sqrt(2. * sc.k * Tkin / mu / sc.m_p)

        # Uninformative priors.
        if not self.velax[0] <= x0 <= self.velax[-1]:
            return -np.inf
        if not 0 < Tb < 3. * np.nanmax(spectrum):
            return -np.inf
        if not 0 < dV < (self.velax[-1] - self.velax[0]) / 3.0:
            return -np.inf

        # Log-likelihood.
        model = gaussian(self.velax, x0, dV, Tb)
        lnx2 = np.power((spectrum - model) / uncertainty, 2)
        lnx2 += np.log(2. * np.pi * np.power(uncertainty, 2))
        return -0.5 * np.nansum(lnx2)

    def _emcee_combined(self, spectrum, mu=28., emcee_kwargs={},
                        plot_walkers=False, plot_corner=False):
        """Fit a Gaussian where Tb and dV are coupled using emcee."""

        # Populate parameters.
        nwalkers = emcee_kwargs.pop('nwalkers', 32)
        nburnin = emcee_kwargs.pop('nburnin', 50)
        nsteps = emcee_kwargs.pop('nsteps', 50)
        scatter = emcee_kwargs.pop('scatter', 1e-2)

        # Starting positions.
        p0 = [self.velax[spectrum.argmax()], self._numerical_peak(spectrum)]
        p0 = random_p0(p0, scatter, nwalkers)

        # Calculate the RMS of the spectrum.
        uncertainty = self._numerical_width(spectrum)
        uncertainty = abs(self.velax - uncertainty) / uncertainty
        uncertainty = np.nanstd(spectrum[uncertainty >= 3.0])

        # Run emcee.
        sampler = emcee.EnsembleSampler(nwalkers, 2, self._ln_emcee_single,
                                        args=(uncertainty, mu))
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -nsteps:]
        samples = samples.reshape(-1, samples.shape[-1])

        # Plot diagnostics if appropriate.
        labels = [r'${\rm x_0}$', r'${\rm T_{ex}}$']
        if plot_walkers:
            functions.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        return percentiles_to_errors(percentiles)

    def _emcee_individual(self, spectrum, mu=28., emcee_kwargs={},
                          plot_walkers=False, plot_corner=False):
        """Fit a Gaussian where Tb and dV are independent using emcee."""
        # Populate parameters.
        nwalkers = emcee_kwargs.pop('nwalkers', 32)
        nburnin = emcee_kwargs.pop('nburnin', 50)
        nsteps = emcee_kwargs.pop('nsteps', 50)
        scatter = emcee_kwargs.pop('scatter', 1e-2)

        # Starting positions.
        p0 = [self.velax[spectrum.argmax()],
              self._numerical_peak(spectrum),
              self._numerical_peak(spectrum)]
        p0 = random_p0(p0, scatter, nwalkers)

        # Calculate the RMS of the spectrum.
        uncertainty = self._numerical_width(spectrum)
        uncertainty = abs(self.velax - uncertainty) / uncertainty
        uncertainty = np.nanstd(spectrum[uncertainty >= 3.0])

        # Run emcee.
        sampler = emcee.EnsembleSampler(nwalkers, 3, self._ln_emcee_single,
                                        args=(uncertainty, mu))
        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -nsteps:]
        samples = samples.reshape(-1, samples.shape[-1])

        # Plot diagnostics if appropriate.
        labels = [r'${\rm x_0}$', r'${\rm T_{kin}}$', r'${\rm T_{B}}$']
        if plot_walkers:
            functions.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        return percentiles_to_errors(percentiles)

    # == Rotation Profiles == #

    def get_rotation_profile(self, include_height=True, rbins=None, rpnts=None,
                             resample=True, PA_min=None, PA_max=None,
                             exclude_PA=False, method='dV', nwalkers=32,
                             nburnin=100, nsteps=100, plot_walkers=False,
                             plot_corner=False, verbose=False):
        """
        Return the rotation profile by deprojecting the spectra. Two methods
        are available: 'dV' and 'GP'.

        'dV' - minimizing the width. This works by assuming a Gaussian line
        profile for the deprojected line profile and finding the rotation
        velocity which minimizes this. This approach is fast, however does not
        allow for uncertainties to be calculated. It also has the implicity
        assumption that the deprojected line profile is Gaussian.

        'GP' - finding the smoothest model. This approach models the
        deprojected line profile as a Gaussian Process and tries to find the
        rotation velocity which results in the 'smoothest' model. This allows
        us to relax the assumption of a Gaussian line profile and also return
        uncertainties on the derived rotation velocity.

        - Inputs -

        include_height: Deproject the pixels taking into account the emission
                        surface.
        rbins / rpnts:  Provide the radial grid in [arcsec] which you want to
                        bin the spectra into. By default this will span the
                        entire radius range.
        resample:       For method='dV', average the points back down to the
                        original resolution. This will speed up the fitting.
        PA_min:         Minimum (relative) position angle to include.
        PA_max:         Maximum (relative) position angle to include.
        exclude_PA:     Exclude, rather than include PA_min < PA < PA_mask.
        method:         Which method to use, either 'dV' or 'GP'.
        nwalkers:       Number of walkers to use in the MCMC fitting.
        nburnin:        Number of steps to use for burning in the walkers.
        nsteps:         Number of steps to use to sample the posterior.
        plot_walkers:   Plot the samples to check for convergence.
        plot_corner:    Plot the corner plot to check for covariance.
        verbose:        Print out how far you are through the fitting.

        - Output -

        rpnts:          The bin centres of the radial grid.
        v_rot:          If method='dV' then this is just v_rot in [m/s]. If
                        method='GP' this is the [16, 50, 84]th percentiles of
                        the posterior distribution for the GP model.
        """

        # Check that the method is working.
        if method.lower() not in ['dv', 'gp']:
            raise ValueError("Must specify method: 'dV' or 'GP'.")
        if method.lower() == 'gp' and resample:
            resample = False

        # Deprojected pixel coordinates.
        if include_height:
            rvals, tvals = self.disk_coordinates_3D()
        else:
            rvals, tvals = self.disk_coordinates(self.x0, self.y0, self.inc)
        rvals, tvals = rvals.flatten(), tvals.flatten()
        if rbins is None and rvals is None:
            print("WARNING: No radial sampling set, this will take a while.")
        rbins, rpnts = self._radial_sampling(rbins=rbins, rvals=rpnts)

        # Flatten the data to [velocity, nxpix * nypix].
        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        if dvals.shape != (self.velax.size, self.nxpix * self.nypix):
            raise ValueError("Wrong data shape.")

        # Cycle through each annulus and apply the method.
        v_rot = []
        for r in range(1, rbins.size):

            if verbose:
                print("Running %d / %d..." % (r, rbins.size-1))

            mask = self._get_mask(r_min=rbins[r-1], r_max=rbins[r],
                                  PA_min=PA_min, PA_max=PA_max,
                                  exclude_PA=exclude_PA,
                                  x0=self.x0, y0=self.y0,
                                  inc=self.inc).flatten()
            spectra, angles = dvals[:, mask].T, tvals[mask]

            if method.lower() == 'dv':
                v_rot += [self._get_vrot_from_width(spectra, angles, resample)]
            else:
                try:
                    vkep = self._projected_vkep(rbins[r-1:r+1].mean())
                    v_rot += [self._get_vrot_from_GP(spectra, angles, nwalkers,
                                                     nburnin, nsteps,
                                                     plot_walkers, plot_corner,
                                                     vkep)]
                except:
                    v_rot += [np.zeros((3, 4))]
        return rpnts, np.squeeze(v_rot)

    def _projected_vkep(self, radius, theta=None):
        """Return the projected Keplerian rotation at the given radius ["]."""
        try:
            import scipy.constants as sc
        except:
            raise ValueError("Cannot find scipy.constants.")
        vkep = sc.G * self.mstar * self.msun
        vkep = np.sqrt(vkep / radius / self.dist / sc.au)
        vkep *= 1.0 if theta is None else np.cos(theta)
        return vkep * np.sin(np.radians(self.inc))

    def _get_vrot_from_width(self, spectra, angles, resample=True):
        """Calculate rotation velocity by minimizing the linewidth."""
        vrot, vlsr = self._estimate_vrot(spectra, angles)
        args = (spectra, angles, resample)
        res = minimize(self._deprojected_width, vrot, args=args,
                       method='L-BFGS-B')
        return abs(res.x[0])

    def _estimate_vrot(self, spectra, angles):
        """Estimate the rotation velocity from fitting a SHO to peaks."""
        vpeaks = np.take(self.velax, np.argmax(spectra, axis=1))
        p0 = [0.5 * (np.max(vpeaks) - np.min(vpeaks)), np.mean(vpeaks)]
        try:
            popt, _ = curve_fit(offsetSHO, angles, vpeaks, p0=p0, maxfev=10000)
        except:
            popt = p0
        return np.squeeze(popt)

    def _get_p0(self, spectra, angles, nwalkers, scatter=3e-2):
        """
        Return starting positions for the GP approach. As a guide for the
        Gaussian Process model, rho ~ 3*dV, sigma ~ RMS.
        """

        # Guess the rotation velocity and systemic velocity.
        vrot, vlsr = self._estimate_vrot(spectra, angles)

        # Check that the rotation velocity is positive.
        if vrot < 0.0:
            print("WARNING: Negative rotation velocity found.")
            print("\t Check blue shifted side aligned with East.")

        # Derive properties of the line.
        x, y = self._deprojected_spectrum(spectra, angles, vrot)
        dV = np.trapz(y, x) / y.max() / np.sqrt(2. * np.pi)
        rms = np.nanvar(self.data[0])

        # Include some scatter and return.
        p0 = np.array([vrot, rms, np.log(rms), np.log(dV)])
        return random_p0(p0, scatter, nwalkers), vrot

    def _get_vrot_from_GP(self, spectra, angles, nwalkers=32,
                          nburnin=100, nsteps=50, plot_walkers=False,
                          plot_corner=False, vkep=None, threads=1.0):
        """Calculate rotation velocity by modelling lines as GPs."""
        p0, estimated_vrot = self._get_p0(spectra, angles, nwalkers)
        vkep = estimated_vrot if vkep is None else vkep

        if threads <= 1.0:
            args = (spectra, angles, False, vkep)
            sampler = emcee.EnsembleSampler(nwalkers, 4,
                                            self._log_probability_M32,
                                            args=args)
        else:
            raise NotImplementedError("This just slows it all down...")

        sampler.run_mcmc(p0, nburnin + nsteps)
        samples = sampler.chain[:, -nsteps:]
        samples = samples.reshape(-1, samples.shape[-1])

        # Diagnosis plots.
        labels = [r'${\rm v_{rot}}$', r'${\rm \sigma_{rms}}$',
                  r'${\rm ln(\sigma)}$', r'${\rm ln(\rho)}$']
        if plot_walkers:
            functions.plot_walkers(sampler.chain.T, nburnin, labels)
        if plot_corner:
            functions.plot_corner(samples, labels)

        # Return values.
        return np.percentile(samples, [16, 50, 84], axis=0)

    def _log_probability_M32(self, theta, spectra, angles, resample, vkep):
        """Log-probability function for the Gaussian Processes approach."""

        # Unpack the free parameters.
        vrot, noise, lnsigma, lnrho = theta

        # Uninformative priorsbut don't stray too far from the expected value.
        if abs(vrot - vkep) / vkep > 0.3:
            return -np.inf
        if noise <= 0.0:
            return -np.inf
        if not -5.0 < lnsigma < 10.:
            return -np.inf
        if not 0.0 <= lnrho <= 10.:
            return -np.inf

        # Generate the Gaussian Process model and return log-likelihood.
        x, y = self._deprojected_spectrum(spectra, angles, vrot, resample)

        # Remove pesky points.
        mask = np.percentile(x, [25, 75])
        mask = np.logical_and(x > mask[0], x < mask[1])
        x, y = x[mask], y[mask]

        k_noise = celerite.terms.JitterTerm(log_sigma=np.log(noise))
        k_line = celerite.terms.Matern32Term(log_sigma=lnsigma, log_rho=lnrho)
        kernel = k_noise + k_line
        gp = celerite.GP(kernel, mean=np.nanmean(y), fit_mean=True)
        try:
            gp.compute(x)
        except:
            return -np.inf
        ll = gp.log_likelihood(y, quiet=True)
        return ll if np.isfinite(ll) else -np.inf

    def fit_rotation_curve(self, rvals, vrot, dvrot=None, beam_clip=2.0,
                           fit_mstar=True, verbose=True, save=True):
        """Find the best fitting stellar mass for the rotation profile."""
        if beam_clip:
            mask = rvals > float(beam_clip) * self.bmaj
        else:
            mask = rvals > 0.0

        # Defining functions to let curve_fit do its thing.
        from scipy.optimize import curve_fit
        if fit_mstar:
            def vkep(rvals, mstar):
                return functions._keplerian(rvals, self.inc, mstar, self.dist)
            p0 = self.mstar
        else:
            def vkep(rvals, inc):
                return functions._keplerian(rvals, inc, self.mstar, self.dist)
            p0 = self.inc
        p, c = curve_fit(vkep, rvals[mask], vrot[mask], p0=p0, maxfev=10000,
                         sigma=dvrot[mask] if dvrot is not None else None)

        # Print, save and return the best-fit values.
        if fit_mstar:
            if verbose:
                print("Best-fit: Mstar = %.2f +\- %.2f Msun." % (p, c[0]))
            if save:
                self.mstar = p[0]
        else:
            if verbose:
                print("Best-fit inc: %.2f +\- %.2f degrees." % (p, c[0]))
            if save:
                self.inc = p[0]
        return p[0], c[0, 0]

    def _keplerian_mstar(self, rvals, mstar):
        """Keplerian rotation with stellar mass as free parameter."""
        vkep = np.sqrt(sc.G * mstar * self.msun / rvals / sc.au / self.dist)
        return vkep * np.sin(np.radians(self.inc))

    def _keplerian_inc(self, rvals, inc):
        """Keplerian rotation with inclination as free parameter."""
        vkep = sc.G * self.mstar * self.msun / rvals / sc.au / self.dist
        return np.sqrt(vkep * np.sin(np.radians(inc)))

    # == Spatial Deprojections == #

    def disk_coordinates_3D(self, niter=5):
        """
        Deprojected pixel coordinates in [arcsec, radians] taking account of
        the raised emission surface. Note that PA is relative to the eastern
        major axis.
        """
        rpix = None
        for _ in range(niter):
            rpix, tpix = self._disk_coordinates_3D_iteration(rpix)
        return rpix, tpix

    def _disk_coordinates_3D_iteration(self, rpix=None):
        """Return radius and position angle of each pixel."""
        if rpix is None:
            rpix = np.hypot(self.xaxis[None, :], self.yaxis[:, None])
        inc = np.radians(self.inc)
        zpix = self.emission_surface(rpix)
        zpix *= 1.0 if self.tilt == 'north' else -1.0
        ypix = np.ones(rpix.shape) * self.yaxis[:, None] / np.cos(inc)
        xpix = np.ones(rpix.shape) * self.xaxis[None, :]
        ypix -= zpix * np.tan(inc)
        return np.hypot(ypix, xpix), np.arctan2(ypix, xpix)

    # == Emission surface. == #

    def emission_surface(self, radii):
        """Linearlly interpolate the emission surface ["]."""
        if np.isnan(self.zvals[0]):
            idx = np.isfinite(self.zvals).argmax()
            rim = interp1d([0.0, self.rvals[idx]], [0.0, self.zvals[idx]])
            self.zvals[:idx] = rim(self.rvals[:idx])
        if np.isnan(self.zvals[-1]):
            idx = np.isnan(self.zvals).argmax()
            self.zvals[idx:] = 0.0
        return interp1d(self.rvals, self.zvals, bounds_error=False,
                        fill_value='extrapolate')(radii)

    def set_emission_surface_analytical(self, func='conical', theta=None):
        """
        Define the emission surface as an analytical function.

        - Input Variables -

        func:       Analytical function to use for the surface.
        theta:      Variables for the given function.

        - Possible Functions -

        powerlaw:   Power-law function: z = z_0 * (r / 1.0 arcsec)^z_q where
                    theta = [z_0, z_q].
        conical:    Flat, constant angle surface: z = r * tan(psi) + z_0, where
                    theta = [psi, z_0] where psi in [degrees].

        """
        theta = np.atleast_1d(theta)
        if func.lower() == 'powerlaw':
            if len(theta) != 2:
                raise ValueError("theta = [z_0, z_q].")
            self.zvals = theta[0] * np.power(self.rvals, theta[1])
        elif func.lower() == 'conical':
            if not 1 <= len(theta) < 3:
                raise ValueError("theta = [psi, (z_0)].")
            z0 = theta[1] if len(theta) == 2 else 0.0
            self.zvals = self.rvals * np.tan(np.radians(theta[0])) + z0
        else:
            raise ValueError("func must be 'powerlaw' or 'conical'.")
        return

    def set_emission_surface_data(self, nsigma=1.0, method='GP'):
        """Set the emission surface to that from the data."""
        r, z, _ = self.get_emission_surface_data(nsigma=nsigma, method=method)
        self.rvals, self.zvals = r, z

    def get_emission_surface_data(self, nsigma=1.0, method='GP', rbins=None,
                                  rvals=None):
        """
        Use the method in Pinte et al. (2018) to infer the emission surface.

        - Input Variables -

        x0, y0:     Coordinates [arcseconds] of the centre of the disk.
        inc         Inclination [degrees] of the disk.
        nsigma:     Clipping value used when removing background.

        - Output -

        coords:     A [3 x N] array where N is the number of successfully found
                    ellipses. Each ellipse yields a (r, z, Tb) trio. Distances
                    are in [au] (coverted using the provided distance) and the
                    brightness temperature in [K].
        """

        # Define the radial gridding.
        if rvals is None and rbins is None:
            rvals, rbins = self.rvals, self.rbins
        elif rvals is None:
            rvals = np.average([rbins[1:], rbins[:-1]], axis=0)
        elif rbins is None:
            dr = 0.5 * np.diff(rvals).mean()
            rbins = np.linspace(rvals[0]-dr, rvals[-1]+dr, rvals.size+1)
        clipped_data = self.data

        # Apply masking to the data.
        if nsigma > 0.0:
            r, I, dI = self.radial_profile(collapse='sum')
            mask = np.logical_and(I != 0.0, dI != 0.0)
            mask = nsigma * np.nanmean(dI[mask][-10])
            rsky = self.disk_coordinates(self.x0, self.y0, self.inc, 0.0)[0]
            mask = interp1d(r, I, fill_value='extrapolate')(rsky) >= mask
            mask = np.ones(clipped_data.shape) * mask[None, :, :]
            clipped_data = np.where(mask, clipped_data, 0.0)

            # Use the radial brightness temperature profile as another mask.
            r, Tb, dTb = self.radial_profile(collapse='max', beam_factor=False)
            clip = interp1d(r, Tb - nsigma * Tb, fill_value='extrapolate')
            r = self.disk_coordinates(self.x0, self.y0, self.inc, 0.0)[0]
            clipped_data = np.where(self.data >= clip(r), clipped_data, 0.0)

        # Calculate the emission surface and bin appropriately.
        r, z, Tb = self._get_emission_surface(clipped_data, self.x0, self.y0,
                                              self.inc, r_max=1.41*rbins[-1])
        if method.lower() not in ['gp', 'binned', 'raw']:
            raise ValueError("method must be 'gp', 'binned' or None.")
        if method.lower() == 'gp':
            r, z = sort_arrays(r, z)
            dz = running_stdev(z, window=(self.bmaj / np.nanmean(np.diff(r))))
            r, z, dz = Matern32_model(r, z, dz, jitter=True, return_var=True)
            z = interp1d(r, z, fill_value=np.nan, bounds_error=False)(rvals)
            dz = interp1d(r, dz, fill_value=np.nan, bounds_error=False)(rvals)
        elif method.lower() == 'binned':
            ridxs = np.digitize(r, rbins)
            dz = [np.nanstd(z[ridxs == rr]) for rr in range(1, rbins.size)]
            z = [np.nanmean(z[ridxs == rr]) for rr in range(1, rbins.size)]
            z, dz = np.squeeze(z), np.squeeze(dz)
        else:
            r, z = sort_arrays(r, z)
            dz = running_stdev(z, window=(self.bmaj / np.nanmean(np.diff(r))))
        return rvals, z, dz

    def plot_emission_surface(self, ax=None):
        """Plot the currently stored emission surface."""
        try:
            import matplotlib.pyplot as plt
        except:
            raise ValueError("Cannot find matplotlib.")
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(self.rvals, self.emission_surface(self.rvals),
                    fmt='-o', mew=0, color='k', ms=2)
        ax.set_xlim(0.0, self.rvals[self.zvals > 0.0].max()+self.bmaj)
        ax.set_ylabel(r'Height (arcsec)')
        ax.set_xlabel(r'Radius (arcsec)')
        plotscale(self.bmaj, dx=0.1, dy=0.9, ax=ax)

    def _get_emission_surface(self, data, x0, y0, inc, r_max=None):
        """Find the emission surface [r, z, dz] values."""

        coords = []
        tilt = []
        r_max = abs(self.xaxis).max() if r_max is None else r_max
        for c, channel in enumerate(data):

            # Avoid empty channels.
            if np.nanmax(channel) <= 0.0:
                continue

            # Cycle through the columns in the channel.
            for xidx in range(self.nxpix):

                # Skip rows if appropriate.
                if abs(self.xaxis[xidx] - x0) > r_max:
                    continue
                if np.nanmax(channel[:, xidx]) <= 0.0:
                    continue

                # Find the indices of the two largest peaks.
                yidx = detect_peaks(channel[:, xidx])
                if len(yidx) < 2:
                    continue
                pidx = channel[yidx, xidx].argsort()[::-1]
                yidx = yidx[pidx][:2]

                # Convert indices to polar coordinates.
                x = self.xaxis[xidx]
                yf, yn = self.yaxis[yidx]
                yc = 0.5 * (yf + yn)
                dy = max(yf - yc, yn - yc) / np.cos(np.radians(inc))
                r = np.hypot(x - x0, dy)
                z = abs(yc - y0) / np.sin(np.radians(inc))

                # Add coordinates to list. Apply some filtering.
                if np.isnan(r) or np.isnan(z) or z > r / 2.:
                    continue

                # Include the brightness temperature.
                Tb = channel[yidx[0], xidx]

                # Include the coordinates to the list.
                coords += [[r, z, Tb]]

                # Measure the tilt of the emission surface (north / south).
                tilt += [np.sign(yc - y0)]

        # Use the sign to tell if the surface is 'north' or 'south' tilted.
        self.tilt = 'north' if np.sign(np.nanmean(tilt)) > 0 else 'south'
        if self.verbose:
            print("Found offsets in the %s direction." % self.tilt)
        return np.squeeze(coords).T


def keplerian_rotation(rvals, theta, *args):
    """Call a specific function."""
    instance, name = args
    return getattr(instance, name)(rvals, theta)
