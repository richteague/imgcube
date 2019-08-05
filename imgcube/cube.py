import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc


class imagecube:
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
        dx0 (Optional[float]): Recenter the image to this right ascencion
            offset [arcsec].
        dy0 (Optional[float]): Recenter the image to this declination
            offset [arcsec].
    """

    # Disk specific units.
    msun = 1.988e30
    fwhm = 2.35482004503
    disk_coords_niter = 20

    def __init__(self, path, kelvin=False, clip=None, resample=1, verbose=None,
                 suppress_warnings=True, dx0=0.0, dy0=0.0):

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
        self.xaxis = self._readpositionaxis(a=1)
        self.yaxis = self._readpositionaxis(a=2)
        self.nxpix = self.xaxis.size
        self.nypix = self.yaxis.size
        self.dpix = np.mean([abs(np.diff(self.xaxis)),
                             abs(np.diff(self.yaxis))])

        # Recenter the image if requested.
        if (dx0 != 0.0) or (dy0 != 0.0):
            self.shift_center(dx0=dx0, dy0=dy0, save=True)

        # Spectral axis. Make sure velocity is increasing.
        self.nu = self._readrestfreq()
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
        try:
            self.bunit = self.header['bunit'].lower()
        except:
            self.header['bunit'] = 'jy/beam'
            self.bunit = 'jy/beam'
        if self.bunit != 'k' and kelvin:
            if self.verbose:
                print("WARNING: Converting to Kelvin.")
            if isinstance(kelvin, str):
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
                    z1=0.0, phi=0.0, w_i=0.0, w_r=1.0, w_t=0.0, z_func=None,
                    w_func=None, extend=2., oversample=1, frame='cylindrical'):
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
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            z_func (Optional[callable]): User-defined function returning z in
                [arcsec] at a given radius in [arcsec].
            w_func (Optional[callable]): User-defined function returning z_warp
                in [arcsec] at a given radius in [arcsec].
            frame (Optional[str]): Frame of reference for the returned
                coordinates. Either 'polar' or 'cartesian'.
            z_func (Optional[function]): A function which provides z(r). Note
                that no checking will occur to make sure this is a valid
                function.
            extend (Optional[float]): Factor to extend the axis of the
                attached cube for the modelling.
            oversample (Optional[float]): Rescale the number of pixels along
                each axis. A larger number gives a better result, but at the
                cost of computation time.

        Returns:
            c1 (ndarryy): Either r (cylindrical) or x depending on the frame.
            c2 (ndarray): Either theta or y depending on the frame.
            c3 (ndarray): Height above the midplane, z.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Define the emission surface function. Either use the simple double
        # power-law profile or the user-provied function.

        if z_func is None:
            def z_func(r):
                z = z0 * np.power(r, psi) + z1 * np.power(r, phi)
                if z0 >= 0.0:
                    return np.clip(z, a_min=0.0, a_max=None)
                return np.clip(z, a_min=None, a_max=0.0)
        if w_func is None:
            def w_func(r, t):
                warp = np.radians(w_i) * np.exp(-0.5 * (r / w_r)**2)
                return r * np.tan(warp * np.sin(t - np.radians(w_t)))

        # Calculate the pixel values.
        r, t, z = self._get_flared_coords(x0, y0, inc, PA, z_func, w_func)
        if frame == 'cylindrical':
            return r, t, z
        return r * np.cos(t), r * np.sin(t), z

    def get_annulus(self, r_min, r_max, PA_min=None, PA_max=None,
                    exclude_PA=False, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                    psi=1.0, z1=0.0, phi=1.0, w_i=0.0, w_r=1.0, w_t=0.0,
                    z_func=None, w_func=None, beam_spacing=True,
                    return_theta=True, as_annulus=True,
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
            z_func (Optional[function]): A function which provides z(r). Note
                that no checking will occur to make sure this is a valid
                function.
            beam_spacing (Optional[bool/float]): If True, randomly sample the
                annulus such that each pixel is at least a beam FWHM apart. A
                number can also be used in place of a boolean which will
                describe the number of beam FWHMs to separate each sample by.
            annulus (Optional[bool]): If true, return an annulus instance
                from `eddy`. Requires `eddy` to be installed.

        Returns:
            spectra (ndarray): The spectra from each pixel in the annulus.
            ``theta`` (ndarray): The midplane polar angles in [radians] of each
            of the returned spectra. ensemble (annulus instance): An ``eddy``
            annulus instance if ``as_ensemble == True``.
        """

        dvals = self.data.copy()
        if dvals.ndim == 3:
            dvals = dvals.reshape(self.data.shape[0], -1)
        else:
            dvals = np.atleast_2d(dvals.flatten())

        mask = self.get_mask(r_min=r_min, r_max=r_max, exclude_r=False,
                             PA_min=PA_min, PA_max=PA_max,
                             exclude_PA=exclude_PA, x0=x0, y0=y0, inc=inc,
                             PA=PA, z0=z0, psi=psi, z1=z1, phi=phi, w_i=w_i,
                             w_r=w_r, w_t=w_t, z_func=z_func, w_func=w_func)
        mask = mask.flatten()

        rvals, tvals, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                           psi=psi, z1=z1, phi=phi, w_i=w_i,
                                           w_r=w_r, w_t=w_t, z_func=z_func,
                                           w_func=w_func)
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

        if as_annulus:
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

    def get_vlos(self, r_min, r_max, PA_min=None, PA_max=None,
                 exclude_PA=False, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                 psi=1.0, z1=0.0, phi=1.0, tilt=0.0, beam_spacing=True,
                 options=None):
        """
        Wrapper for the ``get_vlos`` function in ``eddy.fit_annulus``.

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
            options (Optional[dict]): Dictionary of options for ``get_vlos``.

        Returns:
            TBD
        """
        annulus = self.get_annulus(r_min=r_min, r_max=r_max, PA_min=PA_min,
                                   PA_max=PA_max, exclude_PA=exclude_PA, x0=x0,
                                   y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                   z1=z1, phi=phi, tilt=tilt,
                                   beam_spacing=beam_spacing, as_ensemble=True)
        options = {} if options is None else options
        return annulus.get_vlos(**options)

    def sky_to_disk(self, coords, frame_in='polar', frame_out='polar', x0=0.0,
                    y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0, z1=0.0, phi=0.0,
                    tilt=0.0):
        """
        For ther given sky coordinates, either (r, theta) or (x, y),
        return the interpolated disk coordiantes in either (r, theta) or (x, y)
        for plotting.

        Args:
            coords (list): Midplane coordaintes to find in (x, y) in [arcsec,
                arcsec] or (r, theta) in [arcsec, deg].
            frame_in (Optional[str]): Frame of input coordinates, either
                'cartesian' or 'polar'.
            frame_out (Optional[str]): Frame of the output coordinates, either
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

        Returns:
            x (float): The disk frame x-coordinate in [arcsec].
            y (float): The disk frame y-coordinate in [arcsec].
        """

        # Import the necessary module.

        try:
            from scipy.interpolate import griddata
        except Exception:
            raise ValueError("Can't find 'scipy.interpolate.griddata'.")

        # Check the input and output frames.

        frame_in = frame_in.lower()
        frame_out = frame_out.lower()
        for frame in [frame_in, frame_out]:
            if frame not in ['polar', 'cartesian']:
                raise ValueError("Frame must be 'polar' or 'cartesian'.")

        # Make sure input coords are cartesian.

        coords = np.atleast_2d(coords)
        if coords.shape[0] != 2 and coords.shape[1] == 2:
            coords = coords.T
        if coords.shape[0] != 2:
            raise ValueError("coords must be of shape [2 x N].")
        if frame_in == 'polar':
            xsky = coords[0] * np.cos(np.radians(coords[1]))
            ysky = coords[0] * np.sin(np.radians(coords[1]))
        else:
            xsky, ysky = coords

        # Convert to disk coordinates.

        xdisk, ydisk, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                           psi=psi, z1=z1, phi=phi, tilt=tilt,
                                           frame='cartesian')
        x_pix = (np.ones(xdisk.shape) * self.xaxis[None, ::-1]).flatten()[::5]
        y_pix = (np.ones(ydisk.shape) * self.yaxis[:, None]).flatten()[::5]
        x_int = griddata((x_pix, y_pix), xdisk.flatten()[::5], (xsky, ysky),
                         method='nearest')
        y_int = griddata((x_pix, y_pix), ydisk.flatten()[::5], (xsky, ysky),
                         method='nearest')

        # Convert to output frame.

        if frame_out == 'cartesian':
            return x_int, y_int
        r_int, t_int = np.hypot(x_int, y_int), np.arctan2(y_int, x_int)
        return r_int, np.degrees(t_int)

    def disk_to_sky(self, coords, frame_in='cylindrical', x0=0.0, y0=0.0,
                    inc=0.0, PA=0.0, z0=0.0, psi=1.0, z1=0.0, phi=0.0, w_i=0.0,
                    w_r=1.0, w_t=0.0, return_idx=False):
        """
        For a given disk midplane coordinate, either (r, theta) or (x, y),
        return interpolated sky coordiantes in (x, y) for plotting. The input
        needs to be a list like:

            coords = ([r0, t0], [r1, t1], ..., [rN, tN])

        If you have an array of values, rvals and tvals then,

            coords = np.vstack([rvals, tvals]).T

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

        frame_in = frame_in.lower()
        if frame_in not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")
        coords = np.atleast_2d(coords)
        if coords.shape[1] != 2:
            raise ValueError("coords must be of shape [N x 3].")
        if frame_in == 'cylindrical':
            xdisk = coords[:, 0] * np.cos(np.radians(coords[:, 1]))
            ydisk = coords[:, 0] * np.sin(np.radians(coords[:, 1]))
        else:
            xdisk, ydisk = coords.T

        # Grab disk coordinates and sky coordinates to interpolate between.

        xdisk_grid, ydisk_grid = self.disk_coords(x0=x0, y0=y0,
                                                  inc=inc, PA=PA,
                                                  z0=z0, psi=psi,
                                                  z1=z1, phi=phi,
                                                  w_i=w_i, w_r=w_r, w_t=w_t,
                                                  frame='cartesian')[:2]
        xdisk_grid, ydisk_grid = xdisk_grid.flatten(), ydisk_grid.flatten()
        xsky_grid, ysky_grid = self._get_cart_sky_coords(x0=x0, y0=y0)[:2]
        xsky_grid, ysky_grid = xsky_grid.flatten(), ysky_grid.flatten()

        xsky = griddata((xdisk_grid, ydisk_grid), xsky_grid, (xdisk, ydisk),
                        method='nearest' if return_idx else 'linear',
                        fill_value=np.nan)
        ysky = griddata((xdisk_grid, ydisk_grid), ysky_grid, (xdisk, ydisk),
                        method='nearest' if return_idx else 'linear',
                        fill_value=np.nan)

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
        x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
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
        x_rot, y_rot = imagecube._rotate_coords(x_sky, y_sky, PA)
        return imagecube._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_coords(self, x0, y0, inc, PA, z_func, w_func):
        """Return cylindrical coordinates of surface in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
        for _ in range(self.disk_coords_niter):
            z_tmp = z_func(r_tmp) + w_func(r_tmp, t_tmp)
            y_tmp = y_mid + z_tmp * np.tan(np.radians(inc))
            r_tmp = np.hypot(y_tmp, x_mid)
            t_tmp = np.arctan2(y_tmp, x_mid)
        return r_tmp, t_tmp, z_func(r_tmp)

    def shift_center(self, dx0=0.0, dy0=0.0, data=None, save=True):
        """Shift the source center by (dx0 [arcsec], dy0 [arcsec])."""
        from scipy.ndimage import shift
        y0, x0 = -dy0 / self.dpix, dx0 / self.dpix
        data = data if data is not None else self.data
        to_shift = np.where(np.isfinite(data), data, 0.0)
        if to_shift.ndim == 2:
            to_shift = np.array([to_shift])
        shifted = np.array([shift(c, [y0, x0]) for c in to_shift])
        if not save:
            return shifted
        if data.ndim == 2:
            shifted = shifted[0]
        self.data = shifted

    def rotate_image(self, PA, data=None, save=True):
        """Rotat the image PA [degrees] anticlockwise about the center."""
        from scipy.ndimage import rotate
        PA -= 90.0
        data = data if data is not None else self.data
        to_rotate = np.where(np.isfinite(data), data, 0.0)
        if to_rotate.ndim == 2:
            to_rotate = np.array([to_rotate])
        rotated = np.array([rotate(c, PA, reshape=False) for c in to_rotate])
        if not save:
            return rotated
        if data.ndim == 2:
            rotated = rotated[0]
        self.data = rotated

    def _get_flared_cart_coords_forward(self, x0, y0, inc, PA, func, extend=2,
                                        oversample=0.5, gridded=True):
        """
        Return cartestian coordinates of surface in [arcsec, radians]. A
        forward modelling approach which is slower, but can better account for
        non-parametric emission surfaces.

        Args:
            x0 (float): Source center x-axis offset in [arcsec].
            y0 (float): Source center y-axis offset in [arcsec].
            inc (float): Inclination of the disk in [deg]. Differences in
                positive and negative values dictate the tilt of the disk.
            PA (float): Position angle of the disk in [deg].
            func (function): Function returning the height of the emission
                surface in [arcsec] when provided a midplane radius in
                [arcsec].
            extend (optional[float]): Factor to extend the axis of the
                attached cube for the modelling.
            oversample (optional[float]): Rescale the number of pixels along
                each axis. A larger number gives a better result, but at the
                cost of computation time.

        Returns:
            x_obs (ndarray): Disk emission surface x coordinates.
            y_obs (ndarray): Disk emission surface y coordinates.
            z_obs (ndarray): Disk emission surface z coordinates.
        """

        # Disk coordinates.
        x_disk = np.linspace(extend*self.xaxis[0], extend*self.xaxis[-1],
                             int(self.nxpix*oversample))[::-1]
        y_disk = np.linspace(extend*self.yaxis[0], extend*self.yaxis[-1],
                             int(self.nypix*oversample))
        x_disk, y_disk = np.meshgrid(x_disk, y_disk)
        try:
            z_disk = func(np.hypot(x_disk, y_disk), np.arctan2(y_disk, x_disk))
        except TypeError:
            z_disk = func(np.hypot(x_disk, y_disk))
            z_disk = np.where(z_disk < 0.0, 0.0, z_disk)

        # Incline the disk.
        inc, PA = np.radians(inc), np.radians(PA + 90.0)
        x_inc = x_disk
        y_inc = y_disk * np.cos(inc) - z_disk * np.sin(inc)
        z_inc = y_disk * np.sin(inc) + z_disk * np.cos(inc)

        # Remove shadowed pixels.
        mask = np.ones(y_inc.shape).astype('bool')
        if inc < 0.0:
            y_inc = np.maximum.accumulate(y_inc, axis=0)
            mask[1:] = np.diff(y_inc, axis=0) != 0.0
        else:
            y_inc = np.minimum.accumulate(y_inc[::-1], axis=0)[::-1]
            mask[:-1] = np.diff(y_inc, axis=0) != 0.0

        # Rotate the disk.
        x_rot = x_inc * np.cos(PA) + y_inc * np.sin(PA)
        y_rot = y_inc * np.cos(PA) - x_inc * np.sin(PA)
        z_rot = z_inc

        # Shift the disk.
        x_rot += x0
        y_rot += y0

        # Return the pixel values for user-deprojection.
        if not gridded:
            return x_rot, y_rot, z_rot

        # Interpolate back onto the sky grid.
        from scipy.interpolate import griddata
        x_obs = griddata((x_rot.flatten(), y_rot.flatten()), x_disk.flatten(),
                         (self.xaxis[None, :], self.yaxis[:, None]))
        y_obs = griddata((x_rot.flatten(), y_rot.flatten()), y_disk.flatten(),
                         (self.xaxis[None, :], self.yaxis[:, None]))
        z_obs = griddata((x_rot.flatten(), y_rot.flatten()), z_disk.flatten(),
                         (self.xaxis[None, :], self.yaxis[:, None]))
        return x_obs, y_obs, z_obs

    def _get_flared_polar_coords_forward(self, x0, y0, inc, PA, func,
                                         extend=2.0, oversample=0.5):
        """As _get_flared_cart_coords_forward, returning polar coordinates."""
        coords = self._get_flared_cart_coords_forward(x0=x0, y0=y0, inc=inc,
                                                      PA=PA, func=func,
                                                      extend=extend,
                                                      oversample=oversample)
        x_obs, y_obs, z_obs = coords
        return np.hypot(x_obs, y_obs), np.arctan2(y_obs, -x_obs), z_obs

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
                       PA=0.0, z0=0.0, psi=1.0, z1=0.0, phi=1.0, w_i=0.0,
                       w_r=1.0, w_t=0.0, z_func=None, w_func=None,
                       PA_min=None, PA_max=None, exclude_PA=False,
                       beam_spacing=False, data=None, collapse='max',
                       clip_values=None, statistic='mean',
                       uncertainty='stddev', **kwargs):
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
            PA_min (Optional[float]): Minimum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            PA_max (Optional[float]): Maximum polar angle of the segment of the
                annulus in [degrees]. Note this is the polar angle, not the
                position angle.
            exclude_PA (Optional[bool]): If True, exclude the provided polar
                angle range rather than include.
            data (Optional[ndarray]): Data to use to create the profile, if not
                the attached data array.
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

        rbins, x = self.radial_sampling(rbins=rbins, rvals=rpnts)

        # Collapse and bin the data.
        if data is None:
            dvals = self._collapse_cube(method=collapse,
                                        clip_values=clip_values)
        else:
            if data.ndim != 2:
                raise ValueError("If providing own data, must be 2D!")
            dvals = data.flatten()
        rvals, tvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                        psi=psi, z1=z1, phi=phi, w_i=w_i,
                                        w_r=w_r, w_t=w_t, z_func=z_func,
                                        w_func=w_func)[:2]
        rvals, tvals, dvals = rvals.flatten(), tvals.flatten(), dvals.flatten()

        if PA_min is not None or PA_max is not None:
            mask = self.get_mask(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                 z1=z1, phi=phi, w_i=w_i, w_r=w_r, w_t=w_t,
                                 z_func=z_func, w_func=w_func, PA_min=PA_min,
                                 PA_max=PA_max, exclude_PA=exclude_PA)
            mask = mask.flatten()
            rvals, tvals, dvals = rvals[mask], tvals[mask], dvals[mask]

        # Radially bin the data.

        ridxs = np.digitize(rvals, rbins)
        if statistic == 'mean':
            y = np.array([np.nanmean(dvals[ridxs == r])
                          for r in range(1, rbins.size)])
        else:
            y = np.array([np.nanmedian(dvals[ridxs == r])
                          for r in range(1, rbins.size)])

        if uncertainty == 'stddev':
            dy = np.array([np.nanstd(dvals[ridxs == r])
                           for r in range(1, rbins.size)])
        else:
            dy = np.array([np.nanpercentile(dvals[ridxs == r], [16, 50, 84])
                           for r in range(1, rbins.size)])
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
        sx, fx = np.percentile(np.arange(self.xaxis.size),
                               [25, 75]).astype('int')
        sy, fy = np.percentile(np.arange(self.yaxis.size),
                               [25, 75]).astype('int')
        return np.nanstd([self.data[:int(N), sy:fy, sx:fx],
                          self.data[-int(N):, sy:fy, sx:fx]])

    def radial_sampling(self, rbins=None, rvals=None, spacing=0.25):
        """
        Return bins and bin center values. If the desired bin edges are known,
        will return the bin edges and vice versa. If neither are known will
        return default binning with the desired spacing.

        Args:
            rbins (optional[list]): List of bin edges.
            rvals (optional[list]): List of bin centers.
            spacing (optional[float]): Spacing of bin centers in units of beam
                major axis.

        Returns:
            list, list: List of bin edges and bin centers.
        """
        if rbins is not None and rvals is not None:
            raise ValueError("Specify only 'rbins' or 'rvals', not both.")
        if rvals is not None:
            dr = np.diff(rvals)[0] * 0.5
            rbins = np.linspace(rvals[0] - dr, rvals[-1] + dr, len(rvals) + 1)
        if rbins is not None:
            rvals = np.average([rbins[1:], rbins[:-1]], axis=0)
        else:
            rbins = np.arange(0, self.xaxis.max(), spacing * self.bmaj)[1:]
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
        except Exception:
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0
            self.beamarea = self.dpix**2.0

    def _calculate_beam_area_arcsec(self):
        """Beam area in square arcseconds."""
        omega = self.bmin * self.bmaj
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    def _calculate_beam_area_str(self):
        """Beam area in steradians."""
        omega = np.radians(self.bmin / 3600.)
        omega *= np.radians(self.bmaj / 3600.)
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    @property
    def pix_per_beam(self):
        """Number of pixels per beam."""
        return self._calculate_beam_area_arcsec() / np.power(self.dpix, 2.0)

    @property
    def beam_per_pix(self):
        """Number of beams per pixel."""
        return 1. / self.pix_per_beam

    @property
    def beam_area_arcsec(self):
        """Beam area in square arcseconds."""
        return self._calculate_beam_area_arcsec()

    @property
    def beam_area_str(self):
        """Beam area in steradians."""
        return self._calculate_beam_area_str()

    @property
    def beam(self):
        """Returns the beam parameters in [arcsec], [arcsec], and [degrees]."""
        return self.bmaj, self.bmin, self.bpa

    def _beamkernel(self, bmaj=None, bmin=None, bpa=None, nbeams=1.0):
        """Returns the 2D Gaussian kernel for convolution."""
        from astropy.convolution import Gaussian2DKernel
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
        return Gaussian2DKernel(bmin, bmaj, theta=np.radians(bpa))

    @staticmethod
    def _convolve_image(image, kernel, fast=True):
        """Convolve the image with the provided kernel."""
        if fast:
            from astropy.convolution import convolve_fft
            return convolve_fft(image, kernel)
        from astropy.convolution import convolve
        return convolve(image, kernel, boundary='extend')

    def convolve_cube(self, bmaj=None, bmin=None, bpa=None, nbeams=1.0,
                      fast=True, data=None):
        """
        Convolve the cube with a 2D Gaussian beam.

        Args:
            bmaj (Optional[float]): FWHM of the Gaussian's major axis in
                [arcsec]. Default is to use the beam major axis.
            bmin (Optional[float]): FWHM of the Gaussian's minor axis in
                [arcsec]. Default is to use the beam minor axis.
            bpia (Optional[float]): Position angle of the Gaussian in
                [degrees]. Default is to use the beam position angle.
            nbeams (Optional[float]): Number of beams to convolve the image by.
                This is simple a multiplicative factor for ``bmaj`` and
                ``bmin`` if they are not provided.
            fast (Optional[bool]): Whether to use the FFT method for the
                convolution. Default is ``True``.
            data (Optional[array]):
        """
        data = self.data if data is None else data
        if data.ndim == 2:
            data = np.array([data])
        bmaj = self.bmaj if bmaj is None else bmaj
        bmin = self.bmin if bmin is None else bmin
        bpa = self.bpa if bpa is None else bpa
        k = self._beamkernel(bmaj=bmaj, bmin=bmin, bpa=bpa, nbeams=nbeams)
        convolved_cube = [imagecube._convolve_image(c, k, fast) for c in data]
        return np.squeeze(convolved_cube)

    def add_correlated_noise(self, rms, bmaj, bmin=None, bpa=0.0, nchan=2):
        """Add the output of correlated_nosie() directly to self.data."""
        self.data += self.correlated_noise(rms=rms, bmaj=bmaj, bmin=bmin,
                                           bpa=bpa, nchan=nchan)

    def correlated_noise(self, rms, bmaj, bmin=None, bpa=0.0, nchan=2):
        """
        Generate a 3D cube of spatially and spectrall correlated noise,
        following function from Ryan Loomis. TODO: Allow for a user-defined
        kernel for the spectral convolution.

        Args:
            rms (float): Desired RMS of the noise.
            bmaj (float): Beam major axis for the spatial convolution in
                [arcsec].
            bmin (optional[float]): Beam minor axis for the spatial convolution
                in [arcsec]. If no value is provided we assume a circular beam.
            bpa (optional[float]): Position angle of the beam, east of north in
                [degrees]. This is not required for a circular beam.
            nchan (optional[int]): Width of Hanning kernel for spectral
                convolution. By default this is 2.

        Returns:
            noise (ndarray[float]): An array of noise with the same shape of
                the data with a standard deviation provided.
        """

        # Default to circular beam.
        bmin = bmaj if bmin is None else bmin

        # Make random noise.
        noise = np.random.normal(size=self.data.size).reshape(self.data.shape)

        # Convolve it along the channels.
        kernel = np.hanning(nchan + 2)
        kernel /= np.sum(kernel)
        if np.isfinite(kernel).all() and self.data.ndim == 3:
            noise = np.array([[np.convolve(noise[:, i, j], kernel, mode='same')
                               for i in range(noise.shape[1])]
                              for j in range(noise.shape[2])]).T

        # Convolve it spatially.
        if bmaj > 0.0:
            kernel = self._beamkernel(bmaj=bmaj, bmin=bmin, bpa=bpa)
            if self.data.ndim == 3:
                noise = np.array([self._convolve_image(c, kernel)
                                  for c in noise])
            else:
                noise = self._convolve_image(noise, kernel)

        # Rescale the noise.
        return noise * rms / np.std(noise)

    # == Plotting Functions == #

    @property
    def extent(self):
        """Extent for imshow."""
        return [self.xaxis[0], self.xaxis[-1], self.yaxis[0], self.yaxis[-1]]

    @property
    def BuRd(self):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        colors = np.vstack((c1, np.ones(4), c2))
        return mcolors.LinearSegmentedColormap.from_list('eddymap', colors)

    def plotbeam(self, ax, x0=0.1, y0=0.1, **kwargs):
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

    def plotaxes(self, ax, x0=0., y0=0., inc=0., PA=0., major=1.0, **kwargs):
        """
        Plot the major and minor axes on the provided axis.

        Args:
            ax (Matplotlib axes): Axes instance to plot onto.
            x0 (optional[float]): Relative x-location of the center [arcsec].
            y0 (optional[float]): Relative y-location of the center [arcsec].
            inc (optional[float]): Inclination of the disk in [degrees].
            PA (optional[float]): Position angle of the disk in [degrees].
            major (optional[float]): Size of the major axis line in [arcsec].
        """

        # Default parameters plotting values.
        ls = kwargs.pop('ls', kwargs.pop('linestyle', '--'))
        lw = kwargs.pop('lw', kwargs.pop('linewidth', 0.5))
        lc = kwargs.pop('c', kwargs.pop('color', 'k'))
        ms = kwargs.pop('s', kwargs.pop('size', 1))
        zo = kwargs.pop('zorder', -2)
        m = kwargs.pop('marker', 'o')

        # Plotting.
        coords = [[0.5, t] for t in np.arange(-180, 180, 90)]
        x, y = self.disk_to_sky(coords, x0=x0, y0=y0, inc=inc, PA=PA)

        for i in range(2):
            ax.plot(x[i::2], y[i::2], ls=ls, lw=lw, color=lc, zorder=zo)
        if dotted:
            for xx, yy in zip(x, y):
                ax.scatter(xx, yy, s=ms, marker=m, color=lc, zorder=zo, lw=lw)

    def plot_surface(self, ax=None, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                     psi=0.0, z1=0.0, phi=1.0, tilt=0.0, r_min=0.0, r_max=None,
                     ntheta=9, nrad=10, check_mask=True, **kwargs):
        """
        Overplot the emission surface onto an axis.

        Args:
            ax (Optional[AxesSubplot]): Axis to plot onto.
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
            r_min (Optional[float]): Inner radius to plot, default is 0.
            r_max (Optional[float]): Outer radius to plot.
            ntheta (Optional[int]): Number of theta contours to plot.
            nrad (Optional[int]): Number of radial contours to plot.
            check_mask (Optional[bool]): Mask regions which are like projection
                errors for highly flared surfaces.

        Returns:
            ax (AxesSubplot): Axis with the contours overplotted.
        """

        # Dummy axis to overplot.
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.subplots()[1]

        # Front half of the disk.
        rf, tf, zf = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                      psi=psi, z1=z1, phi=phi, tilt=tilt)
        rf = np.where(zf >= 0.0, rf, np.nan)
        tf = np.where(zf >= 0.0, tf, np.nan)

        # Rear half of the disk.
        rb, tb, zb = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=-z0,
                                      psi=psi, z1=-z1, phi=phi, tilt=tilt)
        rb = np.where(zb <= 0.0, rb, np.nan)
        tb = np.where(zb <= 0.0, tb, np.nan)

        # Flat disk for masking.
        rr, tt, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA)

        # Make sure the bounds are OK.
        r_min = 0.0 if r_min is None else r_min
        r_max = rr.max() if r_max is None else r_max

        # Make sure the front side hides the rear.
        mf = np.logical_and(rf >= r_min, rf <= r_max)
        mb = np.logical_and(rb >= r_min, rb <= r_max)
        tf = np.where(mf, tf, np.nan)
        rb = np.where(~mf, rb, np.nan)
        tb = np.where(np.logical_and(np.isfinite(rb), mb), tb, np.nan)

        # For some geometries we want to make sure they're not doing funky
        # things in the outer disk when psi is large.
        if check_mask:
            mm = rr <= check_mask * r_max
            rf = np.where(mm, rf, np.nan)
            rb = np.where(mm, rb, np.nan)
            tf = np.where(mm, tf, np.nan)
            tb = np.where(mm, tb, np.nan)

        # Popluate the kwargs with defaults.
        lw = kwargs.pop('lw', kwargs.pop('linewidth', 0.5))
        zo = kwargs.pop('zorder', 10000)
        c = kwargs.pop('colors', kwargs.pop('c', 'k'))

        radii = np.linspace(0, r_max, int(nrad + 1))[1:]
        theta = np.linspace(-np.pi, np.pi, int(ntheta + 1))[:-1]

        # Do the plotting.
        ax.contour(self.xaxis, self.yaxis, rf, levels=radii, colors=c,
                   linewidths=lw, linestyles='-', zorder=zo, **kwargs)
        ax.contour(self.xaxis, self.yaxis, tf, levels=theta, colors=c,
                   linewidths=lw, linestyles='-', zorder=zo, **kwargs)
        ax.contour(self.xaxis, self.yaxis, rb, levels=radii, colors=c,
                   linewidths=lw, linestyles='--', zorder=zo, **kwargs)
        ax.contour(self.xaxis, self.yaxis, tb, levels=theta, colors=c,
                   linewidths=lw, linestyles='--', zorder=zo)
        return ax

    def polar_plot(self, rgrid=None, tgrid=None, x0=0.0, y0=0.0, inc=0.0,
                   PA=0.0, z0=0.0, psi=0.0, z1=0.0, phi=1.0, w_i=0.0, w_r=1.0,
                   w_t=0.0, z_func=None, w_func=None, data=None,
                   collapse='max', clip_values=None, griddata_kwargs=None,
                   ax=None, xaxis='radius', imshow_kwargs=None):
        """
        Plots the polar deprojection (using self.deproject_data_polar()) of the
        attached data. You can also specify your own data if you want to

        Args:
            rgrid (Optional[array]): Radial
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
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            z_func (Optional[callable]): User-defined function returning z in
                [arcsec] at a given radius in [arcsec].
            w_func (Optional[callable]): User-defined function returning z_warp
                in [arcsec] at a given radius in [arcsec].
            data (Optional[array]): Data to deproject, otherwise use the
                attached data. If providing data, it must already be collapsed
                to a 2D array.
            collapse (Optional[str]): Method used to collapse 3D data. Must be
                'max' to take the maximum value, 'sum' to sum along the
                spectral axis or 'int' to integrate along the spectral axis.
            clip_values (Optional[float/iterable]): Clip the data values. If a
                single value is given, clip all values below this, if two
                values are given, clip values between them.
            ax (Optional[Matplolib axis]): Axes to plot the data on.
            xaxis (Optional[str]): Which value to have along the x-axis, either
                'radius' or 'polar angle'.
            imshow_kwargs(Optional[dict]): Kwargs to be passed to imshow.

        Returns:
            ax (Matoplot axis): Axis on which the plot is plotted.

        """
        gridded = self.deproject_data_polar(rgrid=rgrid, tgrid=tgrid, x0=x0,
                                            y0=y0, inc=inc, PA=PA, z0=z0,
                                            psi=psi, z1=z1, phi=phi, w_i=w_i,
                                            w_r=w_r, w_t=w_t, z_func=z_func,
                                            w_func=w_func, data=data,
                                            collapse=collapse,
                                            clip_values=clip_values,
                                            griddata_kwargs=griddata_kwargs)
        rgrid, tgrid, dgrid = gridded

        import matplotlib.pyplot as plt

        xaxis = xaxis.lower()
        if xaxis not in ['radius', 'polar angle']:
            raise ValueError("'xaxis' must be 'radius' or 'polar angle'.")

        if ax is None:
            _, ax = plt.subplots()

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        _ = imshow_kwargs.pop('origin', None)
        _ = imshow_kwargs.pop('extent', None)
        aspect = imshow_kwargs.pop('aspect', 'auto')

        if xaxis != 'radius':
            dgrid = dgrid.T
            extent = [tgrid[0], tgrid[-1], rgrid[0], rgrid[-1]]
            ylabel, xlabel = 'Radius (arcsec)', 'Polar Angle (degrees)'
        else:
            extent = [rgrid[0], rgrid[-1], tgrid[0], tgrid[-1]]
            xlabel, ylabel = 'Radius (arcsec)', 'Polar Angle (degrees)'

        im = ax.imshow(dgrid, origin='lower', extent=extent, aspect=aspect,
                       **imshow_kwargs)
        cb = plt.colorbar(im, pad=0.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def deproject_data_polar(self, rgrid=None, tgrid=None, x0=0.0, y0=0.0,
                             inc=0.0, PA=0.0, z0=0.0, psi=0.0, z1=0.0, phi=1.0,
                             w_i=0.0, w_r=1.0, w_t=0.0, z_func=None,
                             w_func=None, data=None, collapse='max',
                             clip_values=None, griddata_kwargs=None):
        """
        Deproject the data into (r, theta) coordinates based on the geometrical
        properties provided.

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
            w_i (Optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (Optional[float]): Scale radius of the warp in [arcsec].
            w_t (Optional[float]): Angle of nodes of the warp in [degrees].
            z_func (Optional[callable]): User-defined function returning z in
                [arcsec] at a given radius in [arcsec].
            w_func (Optional[callable]): User-defined function returning z_warp
                in [arcsec] at a given radius in [arcsec].
            data (Optional[array]): Data to deproject, otherwise use the
                attached data. If providing data, it must already be collapsed
                to a 2D array.
            collapse (Optional[str]): Method used to collapse 3D data. Must be
                'max' to take the maximum value, 'sum' to sum along the
                spectral axis or 'int' to integrate along the spectral axis.
            clip_values (Optional[float/iterable]): Clip the data values. If a
                single value is given, clip all values below this, if two
                values are given, clip values between them.
            griddata_kwargs (Optional[dict]): Kwargs to be passed to gridddata.

        Returns:
            rgrid: Radial grid to deproject onto in [arcsec].
            tgrid: Polar angle grid to deproject onto in [degrees].
            dgrid: Deprojected data array.
        """

        # 1 - collapse the data.
        if data is None:
            dvals = self._collapse_cube(method=collapse,
                                        clip_values=clip_values).flatten()
        else:
            if data.ndim != 2:
                raise ValueError("If providing own data, must be 2D!")
            dvals = data.flatten()

        # 2 - define the grids. convert the polar grids to radians.
        rvals, tvals, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                           psi=psi, z1=z1, phi=phi, w_i=w_i,
                                           w_r=w_r, w_t=w_t, z_func=z_func,
                                           w_func=w_func)
        rvals, tvals = rvals.flatten(), tvals.flatten()
        if rvals.shape != dvals.shape:
            raise ValueError("Mismatch in 'rvals' and 'dvals' shapes.")
        if rgrid is None:
            rgrid = np.arange(self.bmaj, self.xaxis.max(), 2.*self.dpix)
        if tgrid is None:
            tgrid = np.linspace(-3.2, 3.2, 61)
        else:
            tgrid = np.radians(tgrid)

        # 3 - grid the data.
        from scipy.interpolate import griddata
        griddata_kwargs = {} if griddata_kwargs is None else griddata_kwargs
        method = griddata_kwargs.pop('method', 'nearest')
        dgrid = griddata(points=(rvals, tvals), values=dvals,
                         xi=(rgrid[None, :], tgrid[:, None]),
                         method=method, **griddata_kwargs)
        return rgrid, np.degrees(tgrid), dgrid

    # == Emission Height Functions == #

    def emission_height(self, inc, PA, x0=0.0, y0=0.0, chans=None,
                        threshold=0.95, smooth=[0.5, 0.5], **kwargs):
        """
        Infer the height of the emission surface following the method in Pinte
        et al. (2018a).

        Args:
            inc (float): Inclination of the source in [degrees].
            PA (float): Position angle of the source in [degrees].
            x0 (Optional[float]): Source center offset in x direction in
                [arcsec].
            y0 (Optional[float]): Source center offset in y direction in
                [arcsec].
            chans (Optional[list]): The lower and upper channel numbers to
                include in the fitting.
            threshold (Optional[float]): Fraction of the peak intensity at that
                radius to clip in calculating the data.
            smooth (Optional[list]): Kernel to smooth the profile with prior to
                measuring the peak pixel positions.

        Returns:
            r (ndarray): Deprojected midplane radius in [arcsec].
            z (ndarray): Deprojected emission height in [arcsec].
            Fnu (ndarray): Intensity of at that location.
        """

        # Extract the channels to use.
        if chans is None:
            chans = [0, self.velax.size]
        chans = np.atleast_1d(chans)
        chans[0] = max(chans[0], 0)
        chans[1] = min(chans[1], self.velax.size)
        data = self.data[chans[0]:chans[1]+1].copy()

        # Shift the images to center the image.
        if (x0 != 0.0) or (y0 != 0.0):
            data = self.shift_center(dx0=x0, dy0=y0, data=data, save=False)

        # Rotate the image so major axis is aligned with x-axis.
        if PA is not None:
            data = self.rotate_image(PA, data=data, save=False)

        # Make a radial profile of the peak values.
        rvals = self._get_midplane_polar_coords(0.0, 0.0, inc, 0.0)[0]
        rbins = np.arange(0, self.xaxis.max() + self.dpix, self.dpix)
        if threshold > 0.0:
            Tb = np.max(data, axis=0).flatten()
            ridxs = np.digitize(rvals.flatten(), rbins)
            avgTb = [np.mean(Tb[ridxs == r]) for r in range(1, rbins.size)]
            kernel = np.ones(np.ceil(self.bmaj / self.dpix).astype('int'))
            avgTb = np.convolve(avgTb, kernel / np.sum(kernel), mode='same')

        # Clip everything below this value.
        from scipy.interpolate import interp1d
        avgTb = interp1d(rbins[:-1], threshold * avgTb,
                         fill_value=np.nan, bounds_error=False)
        data = np.where(data >= avgTb(rvals), data, 0.0)

        # Find all the peaks. Save the (r, z, Tb) value. Here we convolve the
        # profile with a top-hat function to reduce some of the noise. We find
        # the two peaks and follow the method from Pinte et al. (2018a) to
        # calculate the height of the emission.
        #from detect_peaks import detect_peaks
        smooth = smooth / np.sum(smooth) if smooth is not None else [1.0]
        peaks = []
        for c_idx in range(data.shape[0]):
            for x_idx in range(data.shape[2]):
                x_c = self.xaxis[x_idx]
                mpd = kwargs.pop('mpd', 0.05 * abs(x_c))
                try:
                    profile = np.convolve(data[c_idx, :, x_idx],
                                          smooth, mode='same')
                    y_idx = detect_peaks(profile, mpd=mpd, **kwargs)
                    y_idx = y_idx[data[c_idx, y_idx, x_idx].argsort()]
                    y_f, y_n = self.yaxis[y_idx[-2:]]
                    y_c = 0.5 * (y_f + y_n)
                    r = np.hypot(x_c, (y_f - y_c) / np.cos(np.radians(inc)))
                    z = y_c / np.sin(np.radians(inc))
                    if z > 0.5 * r or z < 0:
                        raise ValueError()
                    Tb = data[c_idx, y_idx[-1], x_idx]
                except:
                    r, z, Tb = np.nan, np.nan, np.nan
                peaks += [[r, z, Tb]]
        peaks = np.squeeze(peaks)
        return peaks[~np.any(np.isnan(peaks), axis=1)].T

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
            ``flux`` (ndarray): Array of the integrated flux in [Jy] values
            along the attached velocity axis. ``uncertainty`` (ndarray): Array
            of the uncertainties on ``flux``.
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
        to_sum /= self.pix_per_beam
        return np.array([np.nansum(c) for c in to_sum * mask])

    def get_deprojected_spectrum(self, r_min, r_max, PA_min=None, PA_max=None,
                                 exclude_PA=False, x0=0.0, y0=0.0, inc=0.0,
                                 PA=0.0, z0=0.0, psi=1.0, z1=0.0, phi=1.0,
                                 tilt=0.0, beam_spacing=False, mstar=1.0,
                                 dist=100., vrot=None, vrad=0., resample=True):
        """
        Return the azimuthally averaged spectrum from an annulus described by
        r_min and r_max. The spectra can be deprojected assuming either
        Keplerian rotation or with the provided vrot and vrad values.

        Args:
            Coming...

        Returns:
            ``x`` (ndarray[float]): Spectral axis of the deprojected spectrum.
            ``y`` (ndarray[float]): Spectrum, either flux density or brightness
            temperature depending on the units of the cube. ``dy``
            (ndarray[float]): Uncertainty on each y value based on the scatter
            in the resampled bins.
        """
        annulus = self.get_annulus(r_min=r_min, r_max=r_max, PA_min=PA_min,
                                   PA_max=PA_max, exclude_PA=exclude_PA, x0=x0,
                                   y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                                   z1=z1, phi=phi, tilt=tilt, z_func=None,
                                   as_ensemble=True, beam_spacing=beam_spacing)
        if vrot is None:
            vrot = self.keplerian_curve(rpnts=np.average([r_min, r_max]),
                                        mstar=mstar, dist=dist, inc=inc, z0=z0,
                                        psi=psi, z1=z1, phi=phi)
        return annulus.deprojected_spectrum(vrot=vrot, vrad=vrad,
                                            resample=resample, scatter=True)

    # == Rotation Functions == #

    def keplerian_profile(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0,
                          psi=1.0, z1=0.0, phi=1.0, tilt=0.0, mstar=1.0,
                          dist=100., vlsr=0.0):
        """Return a Keplerian rotation profile (for the near side) in [m/s]."""
        rvals, tvals, zvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                               z0=z0, psi=psi, z1=z1, phi=phi)
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
                   z1=0.0, phi=1.0, mstar=1.0, dist=100.,
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
            vlsr (Optional[float]): Systemic velocity in [m/s].
            dV0 (Optional[float]): Doppler line width at 1 arcsec in [m/s].
            dVq (Optional[float]): Power-law exponent of the line width
                profile. A value of 0 will result in dV which is constant.
            nbeams (Optional[float]): The number of beams kernels to convolve
                the mask with. For example, nbeams=1 will conolve the mask with
                the attached beam, nbeams=2 will convolve it with a beam double
                the size.
            fname (Optional[str]): File name to save the mask to. If none is
                specified, will use the same path but with a ''.mask.fits'
                extension.
            fast (Optional[bool]): If True, use the fast convolve from Astropy.
            return_mask (Optional[bool]): If True, return the mask as an array,
                otherwise, save it to a FITS file.

            Returns:
                mask (ndarray): If return_mask is True, will return a mask
                matching the shape of the attached cube.
        """

        # Allow for multiple hyperfine components.
        vlsr = np.atleast_1d(vlsr)

        # Loop over all the systemic velocities.
        mask = [self._keplerian_mask(x0=x0, y0=y0, inc=inc, PA=PA,
                                     z0=0.0, psi=psi, z1=z1, phi=phi,
                                     mstar=mstar, r_max=r_max,
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
                                 z1=z1, phi=phi)[0]
        return dV * np.power(rdisk, dVq)

    def _keplerian_mask(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=1.0,
                        z1=0.0, phi=1.0, mstar=1.0, r_max=None,
                        r_min=None, dist=100, vlsr=0.0, dV=250., dVq=0.0):
        """Generate the Keplerian mask as a cube. dV is FWHM of line."""
        mask = np.ones(self.data.shape) * self.velax[:, None, None]
        r_min = 0.0 if r_min is None else r_min
        r_max = 1e5 if r_max is None else r_max
        dV = self._dV_profile(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                              z1=z1, phi=phi, dV=dV, dVq=dVq)

        # Rotation of the front side of the disk.
        v1 = self.keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                    psi=psi, z1=z1, phi=phi,
                                    mstar=mstar, dist=dist, vlsr=vlsr)
        rr = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0, psi=psi,
                              z1=z1, phi=phi)[0]
        v1 = np.where(np.logical_and(rr >= r_min, rr <= r_max), v1, 1e20)
        v1 = abs(mask - np.ones(self.data.shape) * v1[None, :, :])
        if z0 == 0.0 and z1 == 0.0:
            return np.where(v1 <= dV, 1.0, 0.0)

        # Rotation of the far side of the disk if appropriate.
        v2 = self.keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA, z0=-z0,
                                    psi=psi, z1=-z1, phi=phi,
                                    mstar=mstar, dist=dist, vlsr=vlsr)
        rr = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=-z0, psi=psi,
                              z1=-z1, phi=phi)[0]
        v2 = np.where(np.logical_and(rr >= r_min, rr <= r_max), v2, 1e20)
        v2 = abs(mask - np.ones(self.data.shape) * v2[None, :, :])
        return np.where(np.logical_or(v1 <= dV, v2 <= dV), 1.0, 0.0)

    def synthetic_obs(self, bmaj, bmin=None, bpa=0.0, rms=None, chan=None,
                      nchan=None, rescale='auto', spectral_response=None,
                      save=False):
        """
        Generate synthetic observations by convolving the data spatially and
        spectrally and adding correlated noise.

        Args:
            bmaj (float): Beam major axis in [arcsec].
            bmin (optional[float]): Beam minor axis in [arcsec].
            bpa (optional[float]): Beam position angle in [degrees].
            rms (optional[float]): RMS noise of the noise.
            chan (optional[float]): Channel size (m/s) of the resulting data.
            nchan (optional[int]): Number of channels of the resulting data. If
                this would extend beyond the attached velocity then these edge
                channels are ignored.
            rescale (optional[int]): Rescaling factor for the pixels. If
                ``rescale='auto'`` then the pixels will be rescaled so there's
                5 pixels per bmin.
            spectral_response (optional[str]): Type of spectral response to
                include. ``'hanning'`` will include a triangle kernel, while
                ``'averageX'``, where ``'X'`` is a number will use a simple
                running average of ``X`` channels.
            save (optional[bool/str]): If True, save the data as a new cube.
                You may also provide a path to save to (noting that this will
                overwrite anything).
        """

        # Check the input values.
        bmin = bmaj if bmin is None else bmin
        assert bmaj >= bmin, "bmaj >= bmin"
        bpa = np.radians(bpa % 360.0)

        # Copy the data and rescale it.
        data = self.data.copy()
        if type(rescale) in [int, float]:
            if rescale < 1:
                raise ValueError("rescale ({}) must be > 1.".format(rescale))
        elif rescale == 'auto':
            rescale = bmin / self.dpix / 5.0
        if rescale:
            from scipy.ndimage import zoom
            data = np.array([zoom(d, 1. / rescale) for d in data])

        # Rescale the axes to take care of silly things.
        npix = data.shape[1]
        axis = np.linspace(self.yaxis[0], self.yaxis[-1], npix)
        span = axis.max() - axis.min()
        shift = (self.dpix * rescale * data.shape[1] - span) / 2
        axis = np.linspace(axis[0] - shift, axis[-1] - shift, npix)
        dpix = np.diff(axis).mean()

        # Resample the data in velocity space.
        # Includes an interpolation of the data to resample.
        if chan is not None:
            if chan < self.chan:
                raise ValueError("chan ({}) must be larger than ".format(chan)
                                 + "input channel ({}).".format(self.chan))
            vspan = self.velax.max() - self.velax.min()
            if nchan is None:
                nchan = vspan // chan
            start = 0.5 * (vspan - nchan * chan)
            velax = np.arange(0, nchan * chan, chan) + start

            # Make sure the new axis doesn't overshoot the data.
            if velax[0] < self.velax[0]:
                velax = velax[abs(velax - self.velax[0]).argmin():]
            if velax[-1] > self.velax[-1]:
                velax = velax[:abs(velax - self.velax[-1]).argmin()]
            nchan = velax.size

            # Make a high-resolution cube to average down.
            from scipy.interpolate import CubicSpline
            velaxh = np.linspace(self.velax[0], self.velax[-1],
                                 self.velax.size*10)
            data_high = CubicSpline(self.velax, data, axis=0)(velaxh)
            data = []
            vbins = np.linspace(velax[0]-0.5*chan, velax[-1]+0.5*chan, nchan+1)
            for v1, v2 in zip(vbins[:-1], vbins[1:]):
                idxa = abs(velaxh - v1).argmin()
                idxb = abs(velaxh - v2).argmin() + 1
                data += [np.mean(data_high[idxa:idxb], axis=0)]
            data = np.array(data)

        else:
            velax = self.velax

        # Spatially convolve the data.
        from astropy.convolution import convolve, Gaussian2DKernel
        beam = Gaussian2DKernel(bmin/dpix/self.fwhm, bmaj/dpix/self.fwhm, bpa)
        data = np.array([convolve(c, beam, boundary='extend') for c in data])

        # Include a spectral response.
        if spectral_response is not None:
            if spectral_response == 'hanning':
                kernel = np.array([0.25, 0.5, 0.25])
            if 'average' in spectral_response:
                try:
                    kernel = int(spectral_response.replace('average', ''))
                except:
                    kernel = 2
                kernel = np.ones(kernel) / kernel
            data = np.array([[np.convolve(data[:, i, j], kernel, mode='same')
                              for i in range(data.shape[1])]
                             for j in range(data.shape[2])]).T

        # Add the noise.
        if rms is not None:
            noise = np.random.normal(size=data.size).reshape(data.shape)
            noise = np.array([convolve(c, beam, boundary='wrap') for c in noise])
            if spectral_response is not None:
                noise = np.array([[np.convolve(noise[:, i, j], kernel, mode='same')
                                   for i in range(noise.shape[1])]
                                  for j in range(noise.shape[2])]).T
            data += noise * rms / np.std(noise)

        # Save the cube.
        if save:
            # Open and attach data.
            hdu = fits.PrimaryHDU()
            hdu.data = data

            # Right-Ascension axis.
            hdu.header['CTYPE1'] = 'RA---SIN'
            hdu.header['CDELT1'] = -dpix / 3600.
            hdu.header['CRPIX1'] = data.shape[1] / 2 + 1
            hdu.header['CRVAL1'] = self.yaxis[0] / 3600.
            hdu.header['CUNIT1'] = 'deg'

            # Declination axis.
            hdu.header['CTYPE2'] = 'DEC--SIN'
            hdu.header['CDELT2'] = dpix / 3600.
            hdu.header['CRPIX2'] = data.shape[1] / 2 + 1
            hdu.header['CRVAL2'] = self.yaxis[0] / 3600.
            hdu.header['CUNIT2'] = 'deg'

            # Velocity axis.
            hdu.header['CTYPE3'] = 'VELO-LSR'
            hdu.header['CDELT3'] = chan
            hdu.header['CRPIX3'] = 1
            hdu.header['CRVAL3'] = velax[0]
            hdu.header['CUNIT3'] = 'm/s'

            # Other.
            hdu.header['BUNIT'] = self.header['BUNIT']
            hdu.header['RESTFREQ'] = self.header['RESTFREQ']
            hdu.header['BMAJ'] = bmaj / 3600.
            hdu.header['BMIN'] = bmin / 3600.
            hdu.header['BPA'] = np.degrees(bpa)

            # Save the file.
            if type(save) is not str:
                fname = '.{:.2f}arcsec'.format(np.mean([bmaj, bmin]))
                if rms is not None:
                    fname += '.{:.2f}rms'.format(rms)
                fname = self.path.replace('.fits', fname + '.fits')
            else:
                fname = save
            hdu.writeto(fname.replace('.fits', '') + '.fits',
                        overwrite=True, output_verify='fix')

        return axis, velax, data

    # == Masking Functions == #

    def get_mask(self, r_min=None, r_max=None, exclude_r=False, PA_min=None,
                 PA_max=None, exclude_PA=False, x0=0.0, y0=0.0, inc=0.0,
                 PA=0.0, z0=0.0, psi=1.0, z1=0.0, phi=1.0, w_i=0.0, w_r=1.0,
                 w_t=0.0, z_func=None, w_func=None):
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
            z_func (Optional[callable]): A function which returns the emission
                height in [arcsec] for a midplane radius in [arcsec]. If
                provided, will be used in place of the parametric emission
                surface.

        Returns:
            mask (ndarray): A 2D mask.
        """
        rvals, tvals, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                           psi=psi, z1=z1, phi=phi, w_i=w_i,
                                           w_r=w_r, w_t=w_t, z_func=z_func,
                                           w_func=w_func, frame='cylindrical')
        r_min = np.nanmin(rvals) if r_min is None else r_min
        r_max = np.nanmax(rvals) if r_max is None else r_max
        r_mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        r_mask = ~r_mask if exclude_r else r_mask
        PA_min = np.nanmin(tvals) if PA_min is None else np.radians(PA_min)
        PA_max = np.nanmax(tvals) if PA_max is None else np.radians(PA_max)
        PA_mask = np.logical_and(tvals >= PA_min, tvals <= PA_max)
        PA_mask = ~PA_mask if exclude_PA else PA_mask
        return r_mask * PA_mask

    # == Functions to read the data cube axes. == #

    def _clip_cube(self, radius):
        """Clip the cube to +\- clip arcseconds from the origin."""
        xa = abs(self.xaxis - radius).argmin()
        if self.xaxis[xa] < radius:
            xa -= 1
        xb = abs(self.xaxis + radius).argmin()
        if -self.xaxis[xb] < radius:
            xb += 1
        xb += 1
        ya = abs(self.yaxis + radius).argmin()
        if -self.yaxis[ya] < radius:
            ya -= 1
        yb = abs(self.yaxis - radius).argmin()
        if self.yaxis[yb] < radius:
            yb += 1
        yb += 1
        if self.data.ndim == 3:
            self.data = self.data[:, ya:yb, xa:xb]
        else:
            self.data = self.data[ya:yb, xa:xb]
        self.xaxis = self.xaxis[xa:xb]
        self.yaxis = self.yaxis[ya:yb]
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
        a_pix = self.header['crpix%d' % a]
        a_ref = self.header['crval%d' % a]
        a_ref = 0.0
        a_pix -= 0.5
        axis = a_ref + (np.arange(a_len) - a_pix + 1.0) * a_del
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
            velax = (self.nu - specax) * sc.c
            velax /= self.nu
        else:
            velax = self._readspectralaxis(a)
        return velax

    def _readfrequencyaxis(self):
        """Returns the frequency axis in [Hz]."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype3'].lower():
            return self._readspectralaxis(a)
        return self._readrestfreq() * (1.0 - self._readvelocityaxis() / sc.c)

    def velocity_resolution(self, dnu):
        """Convert spectral resolution in [Hz] to [m/s]."""
        v0 = self.restframe_frequency_to_velocity(self.nu)
        v1 = self.restframe_frequency_to_velocity(self.nu + dnu)
        vA = max(v0, v1) - min(v0, v1)
        v1 = self.restframe_frequency_to_velocity(self.nu - dnu)
        vB = max(v0, v1) - min(v0, v1)
        return np.mean([vA, vB])

    def spectral_resolution(self, dV):
        """Convert velocity resolution [m/s] to [Hz]."""
        nu = self.velocity_to_restframe_frequency(velax=[-dV, 0.0, dV])
        return np.mean([abs(nu[1] - nu[0]), abs(nu[2] - nu[1])])

    def velocity_to_restframe_frequency(self, velax=None, vlsr=0.0):
        """Return the rest-frame frequency [Hz] of the given velocity [m/s]."""
        velax = self.velax if velax is None else np.squeeze(velax)
        return self.nu * (1. - (velax - vlsr) / 2.998e8)

    def restframe_frequency_to_velocity(self, nu, vlsr=0.0):
        """Return the velocity [m/s] of the given rest-frame frequency [Hz]."""
        return 2.998e8 * (1. - nu / self.nu) + vlsr

    def _background_Tb(self, Tcmb=2.73):
        """Return the background brightness temperature for the CMB."""
        Tbg = 2. * sc.h * np.power(self.nu, 3) / np.power(sc.c, 2)
        return Tbg / (np.exp(sc.h * self.nu / sc.k / Tcmb) - 1.0)

    def _jybeam_to_Tb(self, data=None, nu=None):
        """Jy/beam to K conversion."""
        nu = self.nu if nu is None else nu
        data = self.data if data is None else data
        Tb = 1e-26 * abs(data) / self._calculate_beam_area_str()
        Tb = 2. * sc.h * np.power(nu, 3) / Tb / np.power(sc.c, 2)
        Tb = sc.h * nu / sc.k / np.log(Tb + 1.0)
        return np.where(data >= 0.0, Tb, -Tb)

    def _jybeam_to_Tb_RJ(self, data=None, nu=None):
        """Jy/beam to K conversion."""
        nu = self.nu if nu is None else nu
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return jy2k * data / self._calculate_beam_area_str()

    def _Tb_to_jybeam(self, data=None, nu=None):
        """K to Jy/beam conversion in Rayleigh–Jeans approximation."""
        nu = self.nu if nu is None else nu
        data = self.data if data is None else data
        Fv = 2. * sc.h * np.power(nu, 3) * np.power(sc.c, -2)
        Fv /= np.exp(sc.h * nu / sc.k / abs(data)) - 1.0
        Fv *= self._calculate_beam_area_str() / 1e-26
        return np.where(data >= 0.0, Fv, -Fv)

    def _Tb_to_jybeam_RJ(self, data=None, nu=None):
        """K to Jy/beam conversion in Rayleigh–Jeans approximation."""
        nu = self.nu if nu is None else nu
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return data * self._calculate_beam_area_str() / jy2k

    def spiral_coords(self, r_p, t_p, m=None, r_min=None, r_max=None,
                      mstar=1.0, T0=20.0, Tq=-0.5, dist=100., clockwise=True,
                      frame_out='cartesian'):
        """
        Spiral coordinates from Bae & Zhaohuan (2018a). In order to recover the
        linear spirals from Rafikov (2002), use m >> 1.

        Args:
            r_p (float): Orbital radius of the planet in [arcsec].
            t_p (float): Polar angle of planet relative to the red-shifted
                major axis of the disk in [radians].
            m (optional[int]): Azimuthal wavenumber of the spiral. If not
                specified, will assume the dominant term based on the rotation
                and temperature profiles.
            r_min (optional[float]): Inner radius of the spiral in [arcsec].
            r_max (optional[float]): Outer radius of the spiral in [arcsec].
            mstar (optioanl[float]): Stellar mass of the central star in [Msun]
                to calculate the rotation profile.
            T0 (optional[float]): Gas temperature in [K] at 1 arcsec.
            Tq (optional[float]): Exoponent of the radial gas temperature
                profile.
            dist (optional[float]): Source distance in [pc] used to scale
                [arcsec] to [au] in the calculation of the rotation profile.
            clockwise (optional[bool]): Direction of the spiral.
            frame_out (optional[str]): Coordinate frame of the returned values,
                either 'cartesian' or 'cylindrical'.

        Returns:
            ndarray:
                Coordinates of the spiral in either cartestian or cylindrical
                frame.
        """

        # Define the radial grid in [arcsec].
        r_min = 0.1 if r_min is None else r_min
        r_max = self.xaxis.max() if r_max is None else r_max
        rvals = np.arange(r_min, r_max, 0.1 * self.dpix)
        clockwise = 1.0 if clockwise is True else -1.0

        # Define the physical properties as a function of radius. SI units.
        omega = np.sqrt(sc.G * mstar * 1.988e30 * (rvals * sc.au * dist)**-3)
        tgas = T0 * np.power(rvals, Tq)
        cs = np.sqrt(sc.k * tgas / 2.37 / sc.m_p)
        H = cs / omega

        # Define the dominant wave number if not defined.
        if m is None:
            m = 0.5 * (r_p * dist * sc.au / H)[abs(rvals - r_p).argmin()]
        m = np.round(m)
        rmn = r_p * dist * sc.au * (1.0 - 1.0 / m)**(2./3.)
        rmp = r_p * dist * sc.au * (1.0 + 1.0 / m)**(2./3.)

        # Integrate the equation numerically.
        x = rvals * dist * sc.au
        y = omega * np.sqrt(abs((1 - (rvals / r_p)**(3./2.))**2 - m**-2.)) / cs
        idx_n = abs(rvals * sc.au * dist - rmn).argmin()
        idx_p = abs(rvals * sc.au * dist - rmp).argmin()
        phi = np.ones(rvals.size) * t_p

        for i, r in enumerate(x):
            phi[i] = t_p - np.sign(r - r_p) * np.pi / 4. / m
            if r <= rmn:
                phi[i] -= clockwise * np.trapz(y[i:idx_n+1][::-1],
                                               x=x[i:idx_n+1][::-1])
            elif r >= rmp:
                phi[i] -= clockwise * np.trapz(y[idx_p:i+1],
                                               x=x[idx_p:i+1])
            else:
                phi[i] = np.nan

        # Return the spirals.
        if frame_out == 'cylindrical':
            return rvals, phi
        return rvals * np.cos(phi), rvals * np.sin(phi)


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/
                                        blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            m1 = np.hstack((dx, 0)) <= 0
            m2 = np.hstack((0, dx)) > 0
            ire = np.where(m1 & m2)[0]
        if edge.lower() in ['falling', 'both']:
            m1 = np.hstack((dx, 0)) < 0
            m2 = np.hstack((0, dx)) >= 0
            ife = np.where(m1 & m2)[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        thing = np.unique(np.hstack((indnan, indnan-1, indnan+1)))
        ind = ind[np.in1d(ind, thing, invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind
