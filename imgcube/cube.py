import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc


class imagecube:

    # Disk specific units.
    msun = 1.988e30
    fwhm = 2. * np.sqrt(2 * np.log(2))

    def __init__(self, path, kelvin=False, clip=None, resample=1, verbose=None,
                 suppress_warnings=True, absolute=False):
        """
        Load up a FITS image cube.

        Args:
            path (str): Relative path to the FITS cube.
            kelvin (Optional[bool/str]): Convert the brightness units to [K].
                If True, use the full Planck law, or if 'RJ' use the
                Rayleigh-Jeans approximation. This is not as accurate but does
                not suffer as much in the low intensity regime.
            clip (Optional[float]): Clip the image cube down to a FOV spanning
                (2 * clip) in [arcseconds].
            resample (Optional[int]): Resample the data spectrally, averaging
                over `resample` number of channels.
            verbose (Optional[bool]): Print out warning messages messages.
            suppress_warnings (Optional[bool]): Suppress warnings from other
                Python pacakges (for example numpy). If this is selected then
                verbose will be set to False unless specified.
            absolute (Optional[bool]): If True, use absolute coordinates using
                Astropy's WCS. This is not tested and is not compatible with
                most of the functions.

        Returns:
            None)
        """

        # Suppres warnings.
        if suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore")
            self.verbose = False if verbose is None else verbose
        else:
            self.verbose = True if verbose is None else verbose

        # Read in the data and header.
        self.path = os.path.expanduser(path)
        self.fname = self.path.split('/')[-1]
        self.data = np.squeeze(fits.getdata(self.path))
        self.data = np.where(np.isfinite(self.data), self.data, 0.0)
        self.header = fits.getheader(path)

        # Generate the cube axes.
        self.absolute = absolute
        if self.absolute and self.verbose:
            print("Returning absolute coordinate values are not tested.")
            print("WARNING: self.dpix will be strange.")
        self.xaxis = self._readpositionaxis(a=1)
        self.yaxis = self._readpositionaxis(a=2)
        self.nxpix = self.xaxis.size
        self.nypix = self.yaxis.size
        self.dpix = np.mean([abs(np.diff(self.xaxis)),
                             abs(np.diff(self.yaxis))])

        # Spectral axis. Make sure velocity is increasing.
        try:
            self.velax = self._readvelocityaxis()
            self.chan = np.mean(np.diff(self.velax))
            self.freqax = self._readfrequencyaxis()
            if self.chan < 0.0:
                self.data = self.data[::-1]
                self.velax = self.velax[::-1]
                self.freqax = self.freqax[::-1]
                self.chan *= -1.0
        except KeyError:
            self.velax = None
            self.chan = None
            self.freqax = None

        # Get the beam properties of the beam. If a CASA beam table is found,
        # take the median values. If neither is specified, assume that the
        # pixel size is the beam size.
        self._readbeam()

        # Convert brightness to Kelvin if appropriate. If kelvin = 'RJ' then
        # use the Rayleigh-Jeans approximation. If the approximation is not
        # used then the non-linearity of the conversion means the noise is
        # horrible.
        self.nu = self._readrestfreq()
        self.bunit = self.header['bunit'].lower()
        if self.bunit != 'k' and kelvin:
            if self.verbose:
                print("WARNING: Converting to Kelvin.")
            if type(kelvin) is str:
                if kelvin.lower() in ['rj', 'rayleigh-jeans']:
                    if self.verbose:
                        print("\t Using the Rayleigh-Jeans approximation.")
                    self.data = self._jybeam_to_Tb_RJ()
            else:
                self.data = self._jybeam_to_Tb()
            self.bunit = 'k'

        # Clip the clube down to a smaller field of view.
        if clip is not None:
            self._clip_cube(clip)

        # Resample the data by a factor by a factor of N.
        if resample <= 0:
            raise ValueError("'resample' must be equal to or larger than 0.")
        elif resample > 1:
            N = int(resample)
            data = [np.average(self.data[i*N:(i+1)*N], axis=0)
                    for i in range(int(self.data.shape[0] / N))]
            self.data = np.squeeze(data)
            velax = [np.average(self.velax[i*N:(i+1)*N])
                     for i in range(self.data.shape[0])]
            self.velax = np.squeeze(velax)
            self.chan = np.diff(self.velax).mean()
            if self.velax.size != self.data.shape[0]:
                raise ValueError("Mistmatch in data and velax shapes.")

        return

    # == Coordinate Deprojection == #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                    z1=0.0, phi=0.0, tilt=0.0, frame='polar'):
        """
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is parameterized as a powerlaw
        profile: z(r) = z0 * (r / 1")^psi. For a razor thin disk, z0 = 0.0,
        while for a conical disk, as described in Rosenfeld et al. (2013),
        psi = 1.0. A correction term, z' = z1 * (r / 1")^phi can be included
        to replicate the downward curve of the emission surface in the outer
        disk.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            tilt (Optional[float]): Value between -1 and 1, describing the
                rotation of the disk. For negative values, the disk is rotating
                clockwise on the sky.
            frame (Optional[str]): Frame of reference for the returned
                coordinates. Either 'polar' or 'cartesian'.

        Returns:
            c1 (ndarryy): Either r (cylindrical) or x depending on the frame.
            c2 (ndarray): Either theta or y depending on the frame.
            c3 (ndarray): Height above the midplane, z.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cartesian', 'polar']:
            raise ValueError("frame must be 'cartesian' or 'polar'.")

        # Define the emission surface function. This approach should leave
        # some flexibility for more complex emission surface parameterizations.

        def func(r):
            z = z0 * np.power(r, psi) + z1 * np.power(r, phi)
            if z0 >= 0.0:
                return np.clip(z, a_min=0.0, a_max=None)
            return np.clip(z, a_min=None, a_max=0.0)

        # Calculate the pixel values.

        if frame == 'cartesian':
            c1, c2 = self._get_flared_cart_coords(x0, y0, inc, PA, func, tilt)
            c3 = func(np.hypot(c1, c2))
        else:
            c1, c2 = self._get_flared_polar_coords(x0, y0, inc, PA, func, tilt)
            c3 = func(c1)
        return c1, c2, c3

    def get_annulus(self, r_min, r_max, PA_min=None, PA_max=None,
                    exclude_PA=False, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                    psi=1.0, z1=0.0, phi=1.0, tilt=0.0, beam_spacing=True,
                    return_theta=True, as_ensemble=False,
                    suppress_warnings=True, remove_empty=True,
                    sort_spectra=True, **kwargs):
        """
        Return an annulus (or section of), of spectra and their polar angles.
        Can select spatially independent pixels within the annulus, however as
        this is random, each draw will be different.

        Args:
            r_min (float): Minimum midplane radius of the annulus in [arcsec].
            r_max (float): Maximum midplane radius of the annulus in [arcsec].
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If True, exclude the provided polar
                angle range rather than include.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            tilt (Optional[float]): Value between -1 and 1, describing the
                rotation of the disk. For negative values, the disk is rotating
                clockwise on the sky.
            beam_spacing (Optional[bool/float]): If True, randomly sample the
                annulus such that each pixel is at least a beam FWHM apart. A
                number can also be used in place of a boolean which will
                describe the number of beam FWHMs to separate each sample by.
            as_ensemble (Optional[bool]): If true, return an ensemble instance
                from `eddy`. Requires `eddy` to be installed.

        Returns:
            spectra (ndarray): The spectra from each pixel in the annulus.
            theta (ndarray): The midplane polar angles in [radians] of each of
                the returned spectra.
            ensemble (annulus instance): An `eddy` annulus instance if
                as_ensemble == True.
        """

        dvals = self.data.copy()
        if dvals.ndim == 3:
            dvals = dvals.reshape(self.data.shape[0], -1)
        else:
            dvals = np.atleast_2d(dvals.flatten())

        mask = self.get_mask(r_min=r_min, r_max=r_max, exclude_r=False,
                             PA_min=PA_min, PA_max=PA_max,
                             exclude_PA=exclude_PA, x0=x0, y0=y0, inc=inc,
                             PA=PA, z0=z0, psi=psi, z1=z1, phi=phi, tilt=tilt)
        mask = mask.flatten()

        rvals, tvals, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                           psi=psi, z1=z1, phi=phi, tilt=tilt)
        rvals, tvals = rvals.flatten(), tvals.flatten()
        dvals, rvals, tvals = dvals[:, mask].T, rvals[mask], tvals[mask]

        # Apply the beam sampling.

        if beam_spacing:

            # Order the data in increase position angle.

            idxs = np.argsort(tvals)
            dvals, tvals = dvals[idxs], tvals[idxs]

            # Calculate the sampling rate.

            sampling = float(beam_spacing) * self.bmaj
            sampling /= np.mean(rvals) * np.median(np.diff(tvals))
            sampling = np.floor(sampling).astype('int')

            # If the sampling rate is above 1, start at a random location in
            # the array and sample at this rate, otherwise don't sample. This
            # happens at small radii, for example.

            if sampling > 1:
                start = np.random.randint(0, tvals.size)
                tvals = np.concatenate([tvals[start:], tvals[:start]])
                dvals = np.vstack([dvals[start:], dvals[:start]])
                tvals, dvals = tvals[::sampling], dvals[::sampling]
            elif self.verbose:
                print("WARNING: Unable to downsample the data.")

        # Return the values in the requested form.

        if as_ensemble:
            try:
                from eddy.fit_annulus import annulus
            except ImportError:
                raise ImportError("Please install eddy.")
            suppress_warnings = kwargs.pop('suppress_warnings', True)
            remove_empty = kwargs.pop('remove_empty', True)
            sort_spectra = kwargs.pop('sort_spectra', True)
            return annulus(spectra=dvals, theta=tvals, velax=self.velax,
                           suppress_warnings=suppress_warnings,
                           remove_empty=remove_empty,
                           sort_spectra=sort_spectra)
        return dvals, tvals

    def disk_to_sky(self, coords, frame='polar', x0=0.0, y0=0.0, inc=0.0,
                    PA=0.0, z0=0.0, psi=1.0, z1=0.0, phi=0.0, tilt=0.0,
                    return_idx=False):
        """
        For a given disk midplane coordinate, either (r, theta) or (x, y),
        return interpolated sky coordiantes in (x, y) for plotting.

        Args:
            coords (list): Midplane coordaintes to find in (x, y) in [arcsec,
                arcsec] or (r, theta) in [arcsec, deg].
            frame (Optional[str]): Frame of input coordinates, either
                'cartesian' or 'polar'.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            tilt (Optional[float]): Value between -1 and 1, describing the
                rotation of the disk. For negative values, the disk is rotating
                clockwise on the sky.
            return_idx (Optional[bool]): If True, return the indices of the
                nearest pixels rather than the interpolated values.

        Returns:
            x (float/int): Either the sky plane x-coordinate in [arcsec] or the
                index of the closest pixel.
            y (float/int): Either the sky plane y-coordinate in [arcsec] or the
                index of the closest pixel.
        """

        # Import the necessary module.

        try:
            from scipy.interpolate import griddata
        except Exception:
            raise ValueError("Can't find 'scipy.interpolate.griddata'.")

        # Make sure input coords are cartesian.

        frame = frame.lower()
        if frame not in ['polar', 'cartesian']:
            raise ValueError("frame must be 'polar' or 'cartesian'.")
        coords = np.atleast_2d(coords)
        if coords.shape[0] != 2 and coords.shape[1] == 2:
            coords = coords.T
        if coords.shape[0] != 2:
            raise ValueError("coords must be of shape [2 x N].")
        if frame == 'polar':
            xdisk = coords[0] * np.cos(np.radians(coords[1]))
            ydisk = coords[0] * np.sin(np.radians(coords[1]))
        else:
            xdisk, ydisk = coords

        # Grab disk coordinates and sky coordinates to interpolate between.

        xdisk_grid, ydisk_grid = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                                  z0=z0, psi=psi, z1=z1,
                                                  phi=phi, tilt=tilt,
                                                  frame='cartesian')[:2]
        xdisk_grid, ydisk_grid = xdisk_grid.flatten(), ydisk_grid.flatten()
        xsky_grid, ysky_grid = self._get_cart_sky_coords()[:2]
        xsky_grid, ysky_grid = xsky_grid.flatten(), ysky_grid.flatten()

        xsky = griddata((xdisk_grid, ydisk_grid), xsky_grid, (xdisk, ydisk),
                        method='nearest' if return_idx else 'linear',
                        fill_value=np.nan)
        ysky = griddata((xdisk_grid, ydisk_grid), ysky_grid, (xdisk, ydisk),
                        method='nearest' if return_idx else 'linear',
                        ffill_value=np.nan)

        # Return the values or calculate the indices.

        if not return_idx:
            xsky = xsky if xsky.size > 1 else xsky[0]
            ysky = ysky if ysky.size > 1 else ysky[0]
            return xsky, ysky
        xidx = np.array([abs(self.xaxis - x).argmin() for x in xsky])
        yidx = np.array([abs(self.yaxis - y).argmin() for y in ysky])
        xidx = xidx if xidx.size > 1 else xidx[0]
        yidx = yidx if yidx.size > 1 else yidx[0]
        return xidx, yidx

    def _estimate_PA(self, clip=95):
        """Estimate the PA in [deg] of the disk."""
        mask = self.data >= np.nanpercentile(self.data, [clip])
        angles = np.where(mask, self.disk_coords()[1], np.nan)
        return np.nanmean(np.degrees(angles)) % 360.

    @staticmethod
    def _rotate_coords(x, y, PA):
        """Rotate (x, y) by PA [deg]."""
        x_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        y_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coords(x, y, inc):
        """Deproject (x, y) by inc [deg]."""
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """Return caresian sky coordinates in [arcsec, arcsec]."""
        return np.meshgrid(self.xaxis - x0, self.yaxis - y0)

    def _get_polar_sky_coords(self, x0, y0):
        """Return polar sky coordinates in [arcsec, radians]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        return np.hypot(y_sky, x_sky), np.arctan2(x_sky, y_sky)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = imagecube._rotate_coords(y_sky, x_sky, -PA)
        return imagecube._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_polar_coords(self, x0, y0, inc, PA, func, tilt):
        """Return polar coordinates of surface in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_mid, t_mid = self._get_midplane_polar_coords(x0, y0, inc, PA)
        for _ in range(5):
            y_tmp = func(r_mid) * np.sign(tilt) * np.tan(np.radians(inc))
            y_tmp = y_mid - y_tmp
            r_mid = np.hypot(y_tmp, x_mid)
            t_mid = np.arctan2(y_tmp, x_mid)
        return r_mid, t_mid

    def _get_flared_cart_coords(self, x0, y0, inc, PA, func, tilt):
        """Return cartesian coordinates of surface in [arcsec, arcsec]."""
        r_mid, t_mid = self._get_flared_polar_coords(x0, y0, inc,
                                                     PA, func, tilt)
        return r_mid * np.cos(t_mid), r_mid * np.sin(t_mid)

    def clip_velocity(self, vmin=None, vmax=None):
        """Clip the cube between (including) the defined velocity ranges."""
        if self.velax is None:
            raise AttributeError("Cannot clip a 2D cube.")
        vmin = vmin if vmin is not None else self.velax.min()
        vmax = vmax if vmax is not None else self.velax.max()
        mask = np.logical_and(self.velax >= vmin, self.velax <= vmax)
        self.data = self.data[mask]
        self.velax = self.velax[mask]
        self.freqax = self.freqax[mask]

    def clip_frequency(self, fmin=None, fmax=None):
        """Clip the cube between (including) the defined frequency ranges."""
        fmin = fmin if fmin is not None else self.freqax.min()
        fmax = fmax if fmax is not None else self.freqax.max()
        mask = np.logical_and(self.freqax >= fmin, self.freqax <= fmax)
        self.data = self.data[mask]
        self.velax = self.velax[mask]
        self.freqax = self.freqax[mask]

    # == Radial Profiles == #

    def radial_profile(self, rpnts=None, rbins=None, x0=0.0, y0=0.0, inc=0.0,
                       PA=0.0, z0=0.0, psi=1.0, z1=0.0, phi=1.0, tilt=0.0,
                       PA_min=None, PA_max=None, exclude_PA=False,
                       beam_spacing=False, collapse='max', clip_values=None,
                       statistic='mean', uncertainty='stddev'):
        """
        Returns a radial profile of the data. If the data is 3D, then it is
        collapsed along the spectral axis with some provided function.

        Args:
            rpnts (ndarray): Bin centers in [arcsec].
            rbins (ndarray): Bin edges in [arcsec].
                NOTE: Only `rpnts` or `rbins` needs to be specified.
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            tilt (Optional[float]): Value between -1 and 1, describing the
                rotation of the disk. For negative values, the disk is rotating
                clockwise on the sky.
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If True, exclude the provided polar
                angle range rather than include.
            collapse (Optional[str]): Method used to collapse 3D data. Must be
                'max' to take the maximum value, 'sum' to sum along the
                spectral axis or 'int' to integrate along the spectral axis.
            clip_values (Optional[float/iterable]): Clip the data values. If a
                single value is given, clip all values below this, if two
                values are given, clip values between them.
            statistic (Optional[str]): Statistic to use to determin the bin
                value, either 'mean' or 'median'.
            uncertainty (Optional[str]): Measure of the bin uncertainty. Either
                'std' for the standard deviation or 'percentiles' for the 16th
                to 84th percentile range about the median.

        Returns:
            x (ndarray): Bin centers [arcsec].
            y (ndarray): Bin statistics.
            dy (ndarray): Bin uncertainties.
        """

        # Check variables are OK.

        statistic = statistic.lower()
        if statistic not in ['mean', 'median']:
            raise ValueError("Must choose statistic: mean or median.")
        uncertainty = uncertainty.lower()
        if uncertainty not in ['stddev', 'percentiles']:
            raise ValueError("Must choose uncertainty: stddev or percentiles.")

        # Define the points to sample the radial profile at.

        rbins, x = self._radial_sampling(rbins=rbins, rvals=rpnts)

        # Collapse and bin the data.

        dvals = self._collapse_cube(method=collapse, clip_values=clip_values)
        rvals, tvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                        psi=psi, z1=z1, phi=phi, tilt=tilt)[:2]
        rvals, tvals, dvals = rvals.flatten(), tvals.flatten(), dvals.flatten()

        if PA_min is not None or PA_max is not None:
            mask = self.get_mask(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                 z1=z1, phi=phi, tilt=tilt, PA_min=PA_min,
                                 PA_max=PA_max, exclude_PA=exclude_PA)
            mask = mask.flatten()
            rvals, tvals, dvals = rvals[mask], tvals[mask], dvals[mask]

        # Radially bin the data.

        ridxs = np.digitize(rvals, rbins)
        if statistic == 'mean':
            y = np.array([np.nanmean(dvals[ridxs == r])
                          for r in range(rbins.size - 1)])
        else:
            y = np.array([np.nanmedian(dvals[ridxs == r])
                          for r in range(rbins.size - 1)])

        if uncertainty == 'stddev':
            dy = np.array([np.nanstd(dvals[ridxs == r])
                           for r in range(rbins.size - 1)])
        else:
            dy = np.array([np.nanpercentile(dvals[ridxs == r], [16, 50, 84])
                           for r in range(rbins.size - 1)])
            dy = np.array([dy[1] - dy[0], dy[2] - dy[1]])
        return x, y, dy

    def _collapse_cube(self, method='max', clip_values=None):
        """Collapse the cube to a 2D image using the requested method."""
        if self.data.ndim > 2:
            to_avg = self._clipped_noise(clip_values=clip_values, fill=0.0)
            if method.lower() not in ['max', 'sum', 'int']:
                raise ValueError("Must choose collpase method: max, sum, int.")
            if method.lower() == 'max':
                to_avg = np.nanmax(to_avg, axis=0)
            elif method.lower() == 'sum':
                to_avg = np.nansum(to_avg, axis=0)
            else:
                to_avg = np.trapz(to_avg, dx=abs(self.chan), axis=0)
        else:
            to_avg = self.data.copy()
        return to_avg

    def _clipped_noise(self, clip_values=None, fill=0.0):
        """Returns a clipped self.data."""
        if clip_values is None:
            return self.data.copy()
        clip_values = np.atleast_1d(clip_values)
        if clip_values.size == 1:
            clip_values = np.insert(clip_values, 0, -1e10)
        mask = np.logical_and(self.data >= clip_values[0],
                              self.data <= clip_values[1])
        return np.where(mask, fill, self.data.copy())

    def _estimate_RMS(self, N=5):
        """Estimate the noise from the first and last N channels."""
        return np.nanstd([self.data[:int(N)], self.data[-int(N):]])

    def _radial_sampling(self, rbins=None, rvals=None):
        """Return default radial sampling if none are specified."""
        if rbins is not None and rvals is not None:
            raise ValueError("Specify only 'rbins' or 'rvals', not both.")
        if rvals is not None:
            dr = np.diff(rvals)[0] * 0.5
            rbins = np.linspace(rvals[0] - dr, rvals[-1] + dr, len(rvals) + 1)
        if rbins is not None:
            rvals = np.average([rbins[1:], rbins[:-1]], axis=0)
        else:
            rbins = np.arange(0, self.xaxis.max(), 0.25 * self.bmaj)[1:]
            rvals = np.average([rbins[1:], rbins[:-1]], axis=0)
        return rbins, rvals

    # == Functions to deal the synthesized beam. == #

    def _readbeam(self):
        """Reads the beam properties from the header."""
        try:
            if self.header.get('CASAMBM', False):
                beam = fits.open(self.path)[1].data
                beam = np.median([b[:3] for b in beam.view()], axis=0)
                self.bmaj, self.bmin, self.bpa = beam
            else:
                self.bmaj = self.header['bmaj'] * 3600.
                self.bmin = self.header['bmin'] * 3600.
                self.bpa = self.header['bpa']
        except:
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0
            self.beamarea = self.dpix**2.0

    def _calculate_beam_area_str(self):
        """Beam area in steradians."""
        omega = np.radians(self.bmin / 3600.)
        omega *= np.radians(self.bmaj / 3600.)
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    def _calculate_beam_area_pix(self):
        """Beam area in pix^2."""
        omega = self.bmin * self.bmaj / np.power(self.dpix, 2)
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    @property
    def beam_per_pix(self):
        """Number of beams per pixel."""
        return self._calculate_beam_area_pix() / self.dpix**2

    @property
    def beam(self):
        """Returns the beam parameters in ["], ["], [deg]."""
        return self.bmaj, self.bmin, self.bpa

    def _beamkernel(self, bmaj=None, bmin=None, bpa=None, nbeams=1.0):
        """Returns the 2D Gaussian kernel for convolution."""
        from astropy.convolution import Kernel
        if bmaj is None and bmin is None and bpa is None:
            bmaj = self.bmaj
            bmin = self.bmin
            bpa = self.bpa
        bmaj /= self.dpix * self.fwhm
        bmin /= self.dpix * self.fwhm
        bpa = np.radians(bpa)
        if nbeams > 1.0:
            bmin *= nbeams
            bmaj *= nbeams
        return Kernel(self._gaussian2D(bmin, bmaj, bpa + 90.).T)

    def _gaussian2D(self, dx, dy, PA=0.0):
        """2D Gaussian kernel in pixel coordinates."""
        xm = np.arange(-4*np.nanmax([dy, dx]), 4*np.nanmax([dy, dx])+1)
        x, y = np.meshgrid(xm, xm)
        x, y = self._rotate_coords(x, y, PA)
        k = np.power(x / dx, 2) + np.power(y / dy, 2)
        return np.exp(-0.5 * k) / 2. / np.pi / dx / dy

    def _convolve_image(self, image, kernel, fast=True):
        """Convolve the image with the provided kernel."""
        if fast:
            from astropy.convolution import convolve_fft
            return convolve_fft(image, kernel)
        from astropy.convolution import convolve
        return convolve(image, kernel)

    def convolve_cube(self, bmaj=None, bmin=None, bpa=None, nbeams=1.0,
                      fast=True, data=None):
        """Convolve the cube with a 2D Gaussian beam."""
        if data is None:
            data = self.data
        kernel = self._beamkernel(bmaj=bmaj, bmin=bmin, bpa=bpa, nbeams=nbeams)
        convolved_cube = [self._convolve_image(c, kernel, fast) for c in data]
        return np.squeeze(convolved_cube)

    def plotbeam(self, ax, x0=0.125, y0=0.125, **kwargs):
        """
        Plot the sythensized beam on the provided axes.

        Args:
            ax (matplotlib axes instance): Axes to plot the FWHM.
            x0 (float): Relative x-location of the marker.
            y0 (float): Relative y-location of the marker.
            kwargs (dic): Additional kwargs for the style of the plotting.
        """
        from matplotlib.patches import Ellipse
        beam = Ellipse(ax.transLimits.inverted().transform((x0, y0)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=False, hatch=kwargs.pop('hatch', '////////'),
                       lw=kwargs.pop('linewidth', kwargs.pop('lw', 1)),
                       color=kwargs.pop('color', kwargs.pop('c', 'k')),
                       zorder=kwargs.pop('zorder', 1000), **kwargs)
        ax.add_patch(beam)

    def plotFWHM(self, ax, x0=0.125, y0=0.125, major=True,
                 align='center', **kwargs):
        """
        Plot the synthesized beam FWHM on the provided axes.

        Args:
            ax (matplotlib axes instance): Axes to plot the FWHM.
            x0 (float): Relative x-location of the marker.
            y0 (float): Relative y-location of the marker.
            major (bool): If True, plot the beam major axis, otherwise the
                minor axis.
            align (str): How to align the marker with respect to the provided
                x0 value. Must be 'center' (default), 'left' or 'right'.
            kwargs (dic): Additional kwargs for the style of the plotting.
        """
        x0, y0 = ax.transLimits.inverted().transform((x0, y0))
        dx = 0.5 * self.bmaj if major else 0.5 * self.bmin
        if align not in ['left', 'center', 'right']:
            raise ValueError("align must be 'left', 'center' or 'right'.")
        if align.lower() == 'left':
            x0 += dx
        elif align.lower() == 'right':
            x0 -= dx
        ax.errorbar(x0, y0, xerr=dx, fmt=' ',
                    color=kwargs.pop('color', kwargs.pop('c', 'k')),
                    capthick=kwargs.pop('capthick', 1.5),
                    capsize=kwargs.pop('capsize', 1.25), **kwargs)

    # == Spectra Functions == #

    def integrated_spectrum(self, r_min=None, r_max=None, clip_values=None):
        """
        Return an integrated spectrum in [Jy]. Will convert images in Tb to
        Jy/beam using the full Planck law. This may cause some noise issues for
        low SNR data.

        Args:
            r_min (Optional[float]): Inner radius in [arcsec] of the area to
                integrate over. Note this is just a circular mask.
            r_max (Optional[float]): Outer radius in [arcsec] of the area to
                integrate over. Note this is just a circular mask.
            clip_values (Optional[float/iterable]): Clip the data values. If a
                single value is given, clip all values below this, if two
                values are given, clip values between them.

        Returns:
            flux (ndarray): Array of the integrated flux in [Jy] values along
                the attached velocity axis.
        """
        if self.data.ndim != 3:
            raise ValueError("Cannot make a spectrum from a 2D image.")
        mask = np.ones(self.data.shape)
        if r_max is not None or r_min is not None:
            rvals = np.hypot(self.xaxis[None, :], self.yaxis[:, None])
            if r_max is not None:
                mask = np.where(rvals[None, :, :] <= r_max, mask, 0)
            if r_min is not None:
                mask = np.where(rvals[None, :, :] >= r_min, mask, 0)
        to_sum = self._clipped_noise(clip_values=clip_values)
        if self.bunit.lower() == 'k':
            to_sum = self._Tb_to_jybeam(data=to_sum)
        to_sum /= self._calculate_beam_area_pix()
        return np.array([np.nansum(c) for c in to_sum * mask])

    # == Rotation Functions == #

    def keplerian_profile(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                          psi=1.0, z1=0.0, phi=1.0, tilt=0.0, mstar=1.0,
                          dist=100., vlsr=0.0):
        """Return a Keplerian rotation profile (for the near side) in [m/s]."""
        rvals, tvals, zvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                               z0=z0, psi=psi, z1=z1, phi=phi,
                                               tilt=tilt)
        vrot = sc.G * mstar * self.msun * np.power(rvals * dist * sc.au, 2.0)
        vrot *= np.power(np.hypot(rvals, zvals) * sc.au * dist, -3.0)
        return np.sqrt(vrot) * np.cos(tvals) * np.sin(np.radians(inc)) + vlsr

    def keplerian_curve(self, rpnts, mstar, dist, inc=90.0, z0=0.0, psi=1.0,
                        z1=0.0, phi=1.0):
        """
        Return a Keplerian rotation profile [m/s] at rpnts [arcsec].

        Args:
            rpnts (ndarray/float): Radial locations in [arcsec] to calculate
                the Keplerian rotation curve at.
            mstar (float): Mass of the central star in [Msun].
            dist (float): Distance to the source in [pc].
            inc (Optional[float]): Inclination of the source in [deg]. If not
                provided, will return the unprojected value.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.

        Returns:
            vkep (ndarray/float): Keplerian rotation curve [m/s] at the
                specified radial locations.
        """
        rpnts = np.squeeze(rpnts)
        zpnts = z0 * np.power(rpnts, psi) + z1 * np.power(rpnts, phi)
        r_m, z_m = rpnts * dist * sc.au, zpnts * dist * sc.au
        vkep = sc.G * mstar * self.msun * np.power(r_m, 2.0)
        vkep = np.sqrt(vkep / np.power(np.hypot(r_m, z_m), 3.0))
        return vkep * np.sin(np.radians(inc))

    # == Functions to write a Keplerian mask for CLEANing. == #

    def CLEAN_mask(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
                   z1=0.0, phi=1.0, tilt=0.0, mstar=1.0, dist=100.,
                   r_max=None, r_min=None, vlsr=0.0, dV0=500., dVq=-0.4,
                   nbeams=0.0, fname=None, fast=True, return_mask=False):
        """
        Create a mask suitable for CLEANing the data. The flared emission
        surface is described with the usual geometrical parameters (see
        disk_coords for more). A radial profile for the line width is also
        included such that

            dV = dV0 * (r / 1")**dVq

        providing a little more flexibility to alter the mask shape in the
        outer regions of the disk.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            tilt (Optional[float]): Value between -1 and 1, describing the
                rotation of the disk. For negative values, the disk is rotating
                clockwise on the sky.
            mstar (Optional[float]): Mass of the central starr in [Msun].
            dist (Optional[float]): Distance to the source in [pc].
            r_min (Optional[float]): Inner radius of the disk in [arcsec].
            r_max (Optional[float]): Outer radius of the disk in [arcsec].

        Additionally we can change the local linewidth as a function of radius.
        This is simply a power-law function,



        and allows a little more flexibility in changing the width of the mask
        as a function of radius. Typical values would be dV0 ~ 250. and
        dVq ~ 0.3.

        - Inputs -

        rbins:      Bin edges in [arcsec] for the binning.
        rpnts:      Bin centers in [arcsec] for the binning.
                    Note: Only specify either rpnts or rbins.
        x0, y0:     Source centre offset in [arcsec].
        inc, PA:    Source inlination and position angle, both in [degrees].
        z_type:     Type of emission surface to assume. Must be 'thin',
                    'conical' or 'flared'. See disk_coords() for more details.
        nearest:    Which side of the disk is closer to the observer. Only
                    necessary if z_type is not 'thin'.
        params:     Parameters for the emission surface. See disk_coords() for
                    more details.
        mstar:      Stellar mass in [Msun].
        dist:       Distance to the source in [pc].
        r_min:      Minimum radius of the mask in [arcsec].
        r_max:      Maximum radius of the mask in [arcsec].
        vlsr:       Systemic velocity of the system in [m/s]. Can also be a
                    list of values to allow for multiple lines / hyperfine
                    transitions.
        dV0, dVq:   Properties of the linewidth power-law profile.
        nbeams:     Number of beams to convolve the mask with.
        fast:       When convolving, whether to use the FFT method.
        fname:      Filename for the saved FITS file. By default it changes the
                    end from '*.fits' to '*.mask.fits'. Note it will overwrite
                    any other files of the same name.
        return_mask: If true, return the mask as a 3D array rather than saving.

        - Outputs -

        Coming Soon.
        """

        # Allow for multiple hyperfine components.
        vlsr = np.atleast_1d(vlsr)

        # Loop over all the systemic velocities.
        mask = [self._keplerian_mask(x0=x0, y0=y0, inc=inc, PA=PA,
                                     z0=0.0, psi=psi, z1=z1, phi=phi,
                                     tilt=tilt, mstar=mstar, r_max=r_max,
                                     r_min=r_min, dist=dist, vlsr=v, dV=dV0,
                                     dVq=dVq) for v in vlsr]
        mask = np.where(np.nansum(mask, axis=0) > 0, 1, 0)
        if mask.shape != self.data.shape:
            raise ValueError("Mask shape is not the same as the data.")

        # Include the beam smearing.
        if nbeams > 0.0:
            mask = self.convolve_cube(nbeams=nbeams, data=mask*1e2, fast=fast)
            mask = np.where(mask >= 1e-2, 1, 0)

        # Return the mask if requested.
        if return_mask:
            return mask

        # Otherwise, save as a new FITS cube.
        if np.diff(self._readvelocityaxis()).mean() < 0:
            mask = mask[::-1]

        if fname is None:
            fname = self.path.replace('.fits', '.mask.fits')
        hdu = fits.open(self.path)
        hdu[0].data = mask
        hdu[0].scale('int16')
        try:
            hdu.writeto(fname.replace('.fits', '') + '.fits',
                        overwrite=True, output_verify='fix')
        except TypeError:
            hdu.writeto(fname.replace('.fits', '') + '.fits',
                        clobber=True, output_verify='fix')

    def _dV_profile(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
                    z1=0.0, phi=1.0, tilt=0.0, dV=450., dVq=0.0):
        """Returns a deprojected linewidth profile for a given geometry."""
        if dVq == 0.0:
            return dV * np.ones((self.nypix, self.nxpix))
        rdisk = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                 z1=z1, phi=phi, tilt=tilt)[0]
        return dV * np.power(rdisk, dVq)

    def _keplerian_mask(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
                        z1=0.0, phi=1.0, tilt=0.0, mstar=1.0, r_max=None,
                        r_min=None, dist=100, vlsr=0.0, dV=250., dVq=0.0):
        """Generate the Keplerian mask as a cube. dV is FWHM of line."""
        mask = np.ones(self.data.shape) * self.velax[:, None, None]
        r_min = 0.0 if r_min is None else r_min
        r_max = 1e5 if r_max is None else r_max
        dV = self._dV_profile(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                              z1=z1, phi=phi, tilt=tilt, dV=dV, dVq=dVq)

        # Rotation of the front side of the disk.
        v1 = self.keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                    psi=psi, z1=z1, phi=phi, tilt=tilt,
                                    mstar=mstar, dist=dist, vlsr=vlsr)
        rr = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                              z1=z1, phi=phi, tilt=tilt)[0]
        v1 = np.where(np.logical_and(rr >= r_min, rr <= r_max), v1, 1e20)
        v1 = abs(mask - np.ones(self.data.shape) * v1[None, :, :])
        if z0 == 0.0 and z1 == 0.0:
            return np.where(v1 <= dV, 1.0, 0.0)

        # Rotation of the far side of the disk if appropriate.
        v2 = self.keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA, z0=-z0,
                                    psi=psi, z1=-z1, phi=phi, tilt=tilt,
                                    mstar=mstar, dist=dist, vlsr=vlsr)
        rr = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=-z0, psi=psi,
                              z1=-z1, phi=phi, tilt=tilt)[0]
        v2 = np.where(np.logical_and(rr >= r_min, rr <= r_max), v2, 1e20)
        v2 = abs(mask - np.ones(self.data.shape) * v2[None, :, :])
        return np.where(np.logical_or(v1 <= dV, v2 <= dV), 1.0, 0.0)

    # == Masking Functions == #

    def get_mask(self, r_min=None, r_max=None, exclude_r=False, PA_min=None,
                 PA_max=None, exclude_PA=False, x0=0.0, y0=0.0, inc=0.0,
                 PA=0.0, z0=0.0, psi=1.0, z1=0.0, phi=1.0, tilt=0.0):
        """
        Returns a 2D mask for pixels in the given region.

        Args:
            r_min (float): Minimum midplane radius of the annulus in [arcsec].
            r_max (float): Maximum midplane radius of the annulus in [arcsec].
            exclude_r (Optional[float]): If True, exclude the provided radial
                rangle rather than including it.
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If True, exclude the provided polar
                angle range rather than including it.
            x0 (Optional[float]): Source right ascension offset (arcsec).
            y0 (Optional[float]): Source declination offset (arcsec).
            inc (Optional[float]): Source inclination (degrees).
            PA (Optional[float]): Source position angle (degrees). Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to z0.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            tilt (Optional[float]): Value between -1 and 1, describing the
                rotation of the disk. For negative values, the disk is rotating
                clockwise on the sky.

        Returns:
            mask (ndarray): A 2D mask.
        """
        rvals, tvals, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                           psi=psi, z1=z1, phi=phi, tilt=tilt,
                                           frame='polar')
        r_min = rvals.min() if r_min is None else r_min
        r_max = rvals.max() if r_max is None else r_max
        r_mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        r_mask = ~r_mask if exclude_r else r_mask
        PA_min = tvals.min() if PA_min is None else np.radians(PA_min)
        PA_max = tvals.max() if PA_max is None else np.radians(PA_max)
        PA_mask = np.logical_and(tvals >= PA_min, tvals <= PA_max)
        PA_mask = ~PA_mask if exclude_PA else PA_mask
        return r_mask * PA_mask

    # == Functions to read the data cube axes. == #

    def _clip_cube(self, clip):
        """Clip the cube to +\- clip arcseconds from the origin."""
        if self.absolute:
            raise ValueError("Cannot clip with absolute coordinates.")
        xa = abs(self.xaxis - clip).argmin()
        xb = abs(self.xaxis + clip).argmin()
        ya = abs(self.yaxis - clip).argmin()
        yb = abs(self.yaxis + clip).argmin()
        if self.data.ndim == 3:
            self.data = self.data[:, yb:ya, xa:xb]
        else:
            self.data = self.data[yb:ya, xa:xb]
        self.xaxis = self.xaxis[xa:xb]
        self.yaxis = self.yaxis[yb:ya]
        self.nxpix = self.xaxis.size
        self.nypix = self.yaxis.size

    def _readspectralaxis(self, a):
        """Returns the spectral axis in [Hz] or [m/s]."""
        a_len = self.header['naxis%d' % a]
        a_del = self.header['cdelt%d' % a]
        a_pix = self.header['crpix%d' % a]
        a_ref = self.header['crval%d' % a]
        return a_ref + (np.arange(a_len) - a_pix + 1.0) * a_del

    def _readpositionaxis(self, a=1):
        """Returns the position axis in [arcseconds]."""
        if a not in [1, 2]:
            raise ValueError("'a' must be in [0, 1].")
        a_len = self.header['naxis%d' % a]
        a_del = self.header['cdelt%d' % a]
        if a == 1 and self.absolute:
            a_del /= np.cos(np.radians(self.header['crval2']))
        a_pix = self.header['crpix%d' % a]
        a_ref = self.header['crval%d' % a]
        if not self.absolute:
            a_ref = 0.0
            a_pix -= 0.5
        axis = a_ref + (np.arange(a_len) - a_pix + 1.0) * a_del
        if self.absolute:
            return axis
        return 3600 * axis

    def _readrestfreq(self):
        """Read the rest frequency."""
        try:
            nu = self.header['restfreq']
        except KeyError:
            try:
                nu = self.header['restfrq']
            except KeyError:
                nu = self.header['crval3']
        return nu

    def _readvelocityaxis(self):
        """Wrapper for _velocityaxis and _spectralaxis."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype%d' % a].lower():
            specax = self._readspectralaxis(a)
            nu = self._readrestfreq()
            velax = (nu - specax) * sc.c / nu
        else:
            velax = self._readspectralaxis(a)
        return velax

    def _readfrequencyaxis(self):
        """Returns the frequency axis in [Hz]."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype3'].lower():
            return self._readspectralaxis(a)
        return self._readrestfreq() * (1.0 - self._readvelocityaxis() / sc.c)

    def restframe_frequency(self, vlsr=0.0):
        """Return the rest frame frequency."""
        return self.nu * (1. - (self.velax - vlsr) / 2.998e8)

    def _background_Tb(self, Tcmb=2.73):
        """Return the background brightness temperature for the CMB."""
        Tbg = 2. * sc.h * np.power(self.nu, 3) / np.power(sc.c, 2)
        return Tbg / (np.exp(sc.h * self.nu / sc.k / Tcmb) - 1.0)

    def _jybeam_to_Tb(self, data=None):
        """Return data converted from Jy/beam to K using full Planck law."""
        data = self.data if data is None else data
        Tb = 1e-26 * abs(data) / self._calculate_beam_area_str()
        Tb = 2. * sc.h * np.power(self.nu, 3) / Tb / np.power(sc.c, 2)
        Tb = sc.h * self.nu / sc.k / np.log(Tb + 1.0)
        return np.where(data >= 0.0, Tb, -Tb)

    def _jybeam_to_Tb_RJ(self, data=None):
        """Jy/beam to K conversion."""
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / self.nu**2 / 2. / sc.k
        return jy2k * data / self._calculate_beam_area_str()

    def _Tb_to_jybeam(self, data=None):
        """K to Jy/beam conversion."""
        data = self.data if data is None else data
        Fv = 2. * sc.h * np.power(self.nu, 3) * np.power(sc.c, -2)
        Fv /= np.exp(sc.h * self.nu / sc.k / abs(data)) - 1.0
        Fv *= self._calculate_beam_area_str() / 1e-26
        return np.where(data >= 0.0, Fv, -Fv)
