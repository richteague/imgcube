"""
Class for rotated cubes (with their major axis aligned with the xaxis.).
This has the functionaility to infer the emission surface following Pinte et
al (2018) with improvements.
"""


import functions
import numpy as np
from cube import imagecube
from detect_peaks import detect_peaks
from scipy.interpolate import interp1d
import scipy.constants as sc


class rotatedcube(imagecube):

    def __init__(self, path, inc=None, mstar=None, dist=None, x0=0.0, y0=0.0,
                 clip=None, kelvin=True, suppress_warnings=False):
        """Read in the rotated image cube."""

        # Initilize the class.
        imagecube.__init__(self, path, absolute=False, kelvin=kelvin,
                           clip=clip, suppress_warnings=suppress_warnings)
        if not kelvin and self.verbose:
            print("WARNING: Not using Kelvin.")

        # Get the deprojected pixel values assuming a thin disk.
        self.PA = 270.
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
        self.rdisk, self.tdisk = self.disk_coords(self.x0, self.y0, self.inc)

        # Define the surface.
        self.nearest = 'north'
        self.tilt = 1.0
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

        try:
            from eddy.eddy import ensemble
        except:
            raise ValueError("Cannot find the eddy package.")

        if method.lower() not in ['bin', 'gp']:
            raise ValueError("Method must be ['bin', 'GP']")

        # Deprojected pixel coordinates.

        rvals, tvals = self.disk_coords(self.x0, self.y0, self.inc,
                                        z_type='func',
                                        params=self.emission_surface,
                                        nearest=self.nearest)
        rvals, tvals = rvals.flatten(), tvals.flatten()
        if rbins is None and rvals is None and self.verbose:
            print("WARNING: No radial sampling set, this will take a while.")
        rbins, rpnts = self._radial_sampling(rbins=rbins, rvals=rpnts)
        if rpnts.size != vrot.size:
            raise ValueError("Wrong number of rotation velocities (vrot).")

        # Flatten the data to [velocity, nxpix * nypix].

        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        if dvals.shape != (self.velax.size, self.nxpix * self.nypix):
            raise ValueError("Wrong data shape.")

        # Deproject the spectra.

        deprojected = []
        for r in range(1, rbins.size):

            # Get spectra and deproject.

            mask = self.get_mask(r_min=rbins[r-1], r_max=rbins[r],
                                 PA_min=PA_min, PA_max=PA_max,
                                 exclude_PA=exclude_PA, x0=self.x0, y0=self.y0,
                                 inc=self.inc, z_type='func',
                                 params=self.emission_surface,
                                 nearest=self.nearest).flatten()
            spectra, theta = dvals[:, mask].T, tvals[mask]
            annulus = ensemble(spectra=spectra, theta=theta, velax=self.velax,
                               suppres_warnings=0 if self.verbose else 1)

            # Collapse to a single spectrum.

            if method == 'bin':
                deprojected += [annulus.deprojected_spectrum(vrot)]
            else:
                spectra = annulus.deprojecrted_spectra(vrot)
                noise = np.nanstd(self.data[0]) * np.ones(spectra.shape)
                velax = self.velax[None, :] * np.ones(spectra.shape)
                x, y, _ = functions.Matern32_model(velax.flatten(),
                                                   spectra.flatten(),
                                                   noise.flatten(),
                                                   oversample=False)
                deprojected += [interp1d(x, y, bounds_error=False,
                                fill_value='extrapolate')(self.velax)]
        return np.squeeze(deprojected)

    # == Rotation Profiles == #

    def get_rotation_profile(self, rbins=None, rpnts=None, resample=True,
                             PA_min=None, PA_max=None, exclude_PA=False,
                             method='dV', **kwargs):
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
        resample:       Average the points back down to the original
                        resolution. This will speed up the fitting but should
                        be used with caution.
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

        # Populate variables.

        try:
            from eddy.eddy import ensemble
        except:
            raise ValueError("Cannot find the eddy package.")
        if method.lower() not in ['dv', 'gp']:
            raise ValueError("Must specify method: 'dV' or 'GP'.")
        if method.lower() == 'gp' and resample:
            if self.verbose:
                print("WARNING: Resampling with GP method not advised.")

        # Deprojected pixel coordinates.

        rvals, tvals = self.disk_coords(self.x0, self.y0, self.inc, 90.,
                                        z_type='func',
                                        params=self.emission_surface,
                                        nearest=self.nearest)
        rvals, tvals = rvals.flatten(), tvals.flatten()
        if rbins is None and rvals is None and self.verbose:
            print("WARNING: No radial sampling set, this will take a while.")
        rbins, rpnts = self._radial_sampling(rbins=rbins, rvals=rpnts)

        # Flatten the data to [velocity, nxpix * nypix].

        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        if dvals.shape != (self.velax.size, self.nxpix * self.nypix):
            raise ValueError("Wrong data shape.")

        # Cycle through each annulus and apply the method.

        v_rot = []
        for r in range(1, rbins.size):

            if self.verbose:
                print("Running %d / %d..." % (r, rbins.size-1))

            # Get the annulus of points.

            mask = self.get_mask(r_min=rbins[r-1], r_max=rbins[r],
                                 PA_min=PA_min, PA_max=PA_max,
                                 exclude_PA=exclude_PA, x0=self.x0, y0=self.y0,
                                 inc=self.inc, z_type='func',
                                 params=self.emission_surface,
                                 nearest=self.nearest).flatten()
            spectra, theta = dvals[:, mask].T, tvals[mask]
            annulus = ensemble(spectra=spectra, theta=theta, velax=self.velax,
                               suppress_warnings=0 if self.verbose else 1)

            # Infer the rotation velocity.

            v_kep = self._projected_vkep(rpnts[r-1:r+1].mean())
            if method.lower() == 'dv':
                v_rot += [annulus.get_vrot_dV()]
            else:
                try:
                    v_rot += [annulus.get_vrot_GP(vref=v_kep, **kwargs)]
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

    # == Emission surface. == #

    def emission_surface(self, radii):
        """Returns the height at the given radius for the stored height."""
        if np.isnan(self.zvals[0]):
            idx = np.isfinite(self.zvals).argmax()
            rim = interp1d([0.0, self.rvals[idx]], [0.0, self.zvals[idx]])
            self.zvals[:idx] = rim(self.rvals[:idx])
        if np.isnan(self.zvals[-1]):
            idx = np.isnan(self.zvals).argmax()
            self.zvals[idx:] = 0.0
        return interp1d(self.rvals, self.zvals, bounds_error=False,
                        fill_value='extrapolate')(radii)

    def set_emission_surface_analytical(self, z_type='conical', params=[13.]):
        """
        Define the emission surface as an analytical function.

        - Input Variables -

        z_typr:     Analytical function to use for the surface.
        params:     Variables for the given function.

        - Possible Functions -

        flared:     Power-law function: z = z_0 * (r / 1.0 arcsec)^z_q where
                    theta = [z_0, z_q].
        conical:    Flat, constant angle surface: z = r * tan(psi) + z_0, where
                    theta = [psi, z_0] where psi in [degrees].

        """
        params = np.atleast_1d(params)
        if z_type.lower() == 'flared':
            if len(params) != 2:
                raise ValueError("theta = [z_0, z_q].")
            self.zvals = params[0] * np.power(self.rvals, params[1])
        elif z_type.lower() == 'conical':
            if not 1 <= len(params) < 3:
                raise ValueError("theta = [psi, (z_0)].")
            z0 = params[1] if len(params) == 2 else 0.0
            self.zvals = self.rvals * np.tan(np.radians(params[0])) + z0
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
        if rbins is None and rvals is None and self.verbose:
            print("WARNING: No radial sampling set, this will take a while.")
        rbins, rvals = self._radial_sampling(rbins=rbins, rvals=rvals)
        clipped_data = self.data

        # Apply masking to the data.
        if nsigma > 0.0:
            r, I, dI = self.radial_profile(collapse='sum')
            rsky = self.disk_coords(self.x0, self.y0, self.inc)[0]

            # Estimate the RMS.
            mask = np.logical_and(I != 0.0, dI != 0.0)
            mask = nsigma * np.nanmean(dI[mask][-10:])

            # Mask all points below nsigma * RMS.
            mask = interp1d(r, I, fill_value='extrapolate')(rsky) >= mask
            mask = np.ones(clipped_data.shape) * mask[None, :, :]
            clipped_data = np.where(mask, clipped_data, 0.0)

            # Mask all points below <Tb> - nsigma * d<Tb>.
            r, Tb, dTb = self.radial_profile(collapse='max', beam_factor=False)
            clip = interp1d(r, Tb - nsigma * Tb, fill_value='extrapolate')
            clipped_data = np.where(self.data >= clip(rsky), clipped_data, 0.0)

        # Calculate the emission surface and bin appropriately.
        r, z, Tb = self._get_emission_surface(clipped_data, self.x0, self.y0,
                                              self.inc, r_max=1.41*rbins[-1])
        idxs = np.argsort(r)
        r, z, Tb = r[idxs], z[idxs], Tb[idxs]

        if method.lower() not in ['gp', 'binned', 'raw']:
            raise ValueError("method must be 'gp', 'binned' or None.")

        if method.lower() == 'gp':
            window = self.bmaj / np.nanmean(np.diff(r))
            dz = functions.running_stdev(z, window=window)
            r, z, dz = functions.Matern32_model(r, z, dz, jitter=True,
                                                return_var=True)
            z = interp1d(r, z, fill_value=np.nan, bounds_error=False)(rvals)
            dz = interp1d(r, dz, fill_value=np.nan, bounds_error=False)(rvals)

        elif method.lower() == 'binned':
            ridxs = np.digitize(r, rbins)
            dz = [np.nanstd(z[ridxs == rr]) for rr in range(1, rbins.size)]
            z = [np.nanmean(z[ridxs == rr]) for rr in range(1, rbins.size)]
            z, dz = np.squeeze(z), np.squeeze(dz)

        else:
            dz = functions.running_stdev(z, window=window)
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
        functions.plotscale(self.bmaj, dx=0.1, dy=0.9, ax=ax)

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

        # Use the sign to tell if the closest surface is 'north' or 'south'.
        self.nearest = 'north' if np.sign(np.nanmean(tilt)) > 0 else 'south'
        self.tilt = 1.0 if self.nearest == 'north' else -1.0
        if self.verbose:
            print("Found the %s side is the closest." % self.nearest)
        return np.squeeze(coords).T
