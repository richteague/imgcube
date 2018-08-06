"""
Default FITS cube class. Reads in the data and generate the axes.

Things still to do:

    1 - slight difference in coordinates between CASA and here.

"""

import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc
from astropy.convolution import Kernel
from astropy.convolution import convolve
from astropy.convolution import convolve_fft
from functions import percentiles_to_errors


class imagecube:

    msun = 1.988e30
    fwhm = 2. * np.sqrt(2 * np.log(2))

    def __init__(self, path, absolute=False, kelvin=True, clip=None,
                 suppress_warnings=True):
        """Load up an image cube."""

        # Suppres warnings.
        if suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore")
            self.verbose = False
        else:
            self.verbose = True

        # Read in the data and header.
        self.path = os.path.expanduser(path)
        self.fname = self.path.split('/')[-1]
        self.data = np.squeeze(fits.getdata(self.path))
        self.header = fits.getheader(path)

        # Generate the cube axes.
        self.absolute = absolute
        if self.absolute and self.verbose:
            print("Returning absolute coordinate values.")
            print("WARNING: self.dpix will be strange.")
        self.xaxis = self._readpositionaxis(a=1)
        self.yaxis = self._readpositionaxis(a=2)
        self.nxpix = self.xaxis.size
        self.nypix = self.yaxis.size
        self.dpix = np.mean([abs(np.diff(self.xaxis)),
                             abs(np.diff(self.yaxis))])

        # Spectral axis.
        self.velax = self._readvelocityaxis()
        self.chan = np.mean(np.diff(self.velax))

        # Get the beam properties of the beam.
        try:
            self.bmaj = self.header['bmaj'] * 3600.
            self.bmin = self.header['bmin'] * 3600.
            self.bpa = self.header['bpa']
            self.beamarea = self._calculate_beam_area_pix()
        except:
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0
            self.beamarea = self.dpix**2.0

        # Convert brightness to Kelvin if appropriate.
        self.nu = self._readrestfreq()
        self.bunit = self.header['bunit'].lower()
        if self.bunit == 'k':
            self.jy2k = 1.0
        else:
            self.jy2k = self._jy2k()
        if kelvin:
            if self.data.ndim == 2 and self.verbose:
                print("WARNING: Converting to Kelvin.")
            self.data *= self.jy2k
            self.bunit = 'k'

        # Clip the clube down to a smaller field of view.
        if clip is not None:
            self._clip_cube(clip)

        return

    # == Coordinate Deprojection == #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, frame='polar',
                    z_type='thin', params=None, nearest='north', get_z=False):
        """
        Return the disk coordinates given the specified deprojection values.

        - Input -

        x0, y0  :   Disk offset in [arcsec].
        inc     :   Inclination of disk in [degrees].
        PA      :   Position angle of the disk in [degrees].
        frame   :   Coordinates returned, either 'cartesian' or 'polar'.
        z_type  :   Type of surface: 'thin', 'conical', 'flared', 'func'.
        params  :   Parameters need for the specified surface. Coming soon.
        nearest :   Which side of the disk is nearest to the observer.
        get_z   :   Return the z value.

        - Output -

        x, y    :   Disk coordinates if frame='cartesian'.
        r, t    :   Disk coordinates if frame='polar'.
        """

        # Check the input variables.
        frame = frame.lower()
        if frame not in ['cartesian', 'polar']:
            raise ValueError("frame must be 'cartesian' or 'polar'.")
        z_type = z_type.lower()
        if z_type not in ['thin', 'conical', 'flared', 'func']:
            raise ValueError("Unknown z_type value.")
        nearest = nearest.lower()
        if nearest not in ['north', 'south']:
            raise ValueError("Either 'north' or 'south' must be closer.")
        tilt = 1.0 if nearest == 'north' else -1.0

        # Rescale for the PA definition (major axis east of north).
        PA -= 45.

        # Geometrically thin disk.
        if z_type == 'thin':
            if frame == 'cartesian':
                c1, c2 = self._get_midplane_cart_coords(x0, y0, inc, PA)
            else:
                c1, c2 = self._get_midplane_polar_coords(x0, y0, inc, PA)
            if get_z:
                return c1, c2, np.zeros(c1.shape)
            return c1, c2

        # If not thin then must have some vertical extent.
        # Define the height functions here to pass to the other functions.
        if z_type == 'conical':
            def func(r):
                return r * np.tan(np.radians(params))
        elif z_type == 'flared':
            def func(r):
                return params[0] * np.power(r, params[1])
        else:
            func = params
        assert callable(func)

        # Geometrically thick disk.
        if frame == 'cartesian':
            c1, c2 = self._get_flared_cart_coords(x0, y0, inc, PA, func, tilt)
            if get_z:
                return c1, c2, func(np.hypot(c1, c2))
        else:
            c1, c2 = self._get_flared_polar_coords(x0, y0, inc, PA, func, tilt)
            if get_z:
                return c1, c2, func(c1)
        return c1, c2

        # Calculate the height.

    def _rotate_coords(self, x, y, PA):
        """Rotate (x, y) by PA [deg]."""
        x_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        y_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        return x_rot, y_rot

    def _deproject_coords(self, x, y, inc):
        """Deproject (x, y) by inc [deg]."""
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """Return caresian sky coordinates in [arcsec, arcsec]."""
        return np.meshgrid(self.xaxis[::-1] + x0, self.yaxis + y0)

    def _get_polar_sky_coords(self, x0, y0):
        """Return polar sky coordinates in [arcsec, radians]."""
        x_sky, y_sky = self._get_cartesian_sky_coords(x0, y0)
        return np.hypot(y_sky, x_sky), np.arctan2(y_sky, x_sky)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = self._rotate_coords(x_sky, y_sky, PA)
        return self._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_polar_coords(self, x0, y0, inc, PA, func, tilt):
        """Return polar coordinates of surface in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_mid, t_mid = self._get_midplane_polar_coords(x0, y0, inc, PA)
        for _ in range(5):
            y_tmp = y_mid - func(r_mid) * tilt * np.tan(np.radians(inc))
            r_mid, t_mid = np.hypot(y_tmp, x_mid), np.arctan2(y_tmp, x_mid)
        return r_mid, t_mid

    def _get_flared_cart_coords(self, x0, y0, inc, PA, func, tilt):
        """Return cartesian coordinates of surface in [arcsec, arcsec]."""
        r_mid, t_mid = self._get_flared_polar_coords(x0, y0, inc,
                                                     PA, func, tilt)
        return r_mid * np.cos(t_mid), r_mid * np.sin(t_mid)

    # == Radial Profiles == #

    def radial_profile(self, rpnts=None, rbins=None, x0=0.0, y0=0.0, inc=0.0,
                       PA=0.0, z_type='thin', nearest='north', params=None,
                       collapse='max', statistic='mean', PA_min=None,
                       PA_max=None, exclude_PA=False, beam_factor=False,
                       clip_values=None):
        """
        Returns the azimuthally averaged intensity profile. If the data is 3D,
        then it is collapsed along the spectral axis.

        - Input -

        rpnts:              Bin centers in [arcsec] for the binning.
        rbins:              Bin edges in [arcsec] for the binning.
                            Note: Only specify either rpnts or rbins.
        x0, y0:             Source centre offset in [arcsec].
        inc, PA:            Inclination and position angle of the disk, both in
                            [degrees].
        collapse:           Method to collapse the cube: 'max', maximum value
                            along the spectral axis; 'sum', sum along the
                            spectral axis; 'int', integrated along the spectral
                            axis.
        statistic:          Return either the mean and standard deviation for
                            each annulus with 'mean' or the 16th, 50th and 84th
                            percentiles with 'percentiles'.
        PA_mask:            Only include values within [PA_min, PA_max].
        excxlude_PA_mask:   Exclude the values within [PA_min, PA_max]
        beam_factor:        Include the number of beams averaged over in the
                            calculation of the uncertainty.
        clip_values:        Clip values. If a single value is specified, clip
                            all absolute values below this, otherwise, if two
                            values are specified, clip values between these.

        - Output -

        pnts:               Array of bin centers.
        y:                  Array of the bin means or medians.
        dy:                 Array of uncertainties in the bin.
        """

        # Collapse the data to a 2D image if necessary.
        to_avg = self._collapse_cube(collapse).flatten()

        # Define the points to sample the radial profile at.
        rbins, rpnts = self._radial_sampling(rbins=rbins, rvals=rpnts)
        rvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z_type=z_type,
                                 params=params, nearest=nearest)[0].flatten()

        # Apply the masks.
        mask = self.get_mask(r_min=rbins[0], r_max=rbins[-1], PA_min=PA_min,
                             PA_max=PA_max, exclude_PA=exclude_PA, x0=x0,
                             y0=y0, inc=inc, PA=PA, z_type=z_type,
                             params=params, nearest=nearest).flatten()

        if mask.size != to_avg.size:
            raise ValueError("Mask and data sizes do not match.")
        if clip_values is not None:
            clip_values = np.squeeze([clip_values])
            if clip_values.size == 1:
                mask *= abs(to_avg) >= clip_values
            else:
                mask *= np.logical_or(to_avg <= clip_values[0],
                                      to_avg >= clip_values[1])
        rvals, to_avg = rvals[mask], to_avg[mask]

        # Apply the averaging.
        ridxs = np.digitize(rvals, rbins)
        if statistic.lower() not in ['mean', 'percentiles']:
            raise ValueError("Must choose statistic: mean or percentiles.")
        if statistic.lower() == 'mean':
            y = [np.nanmean(to_avg[ridxs == r]) for r in range(1, rbins.size)]
            dy = [np.nanstd(to_avg[ridxs == r]) for r in range(1, rbins.size)]
            y, dy = np.squeeze(y), np.squeeze(dy)
        else:
            y = [np.nanpercentile(to_avg[ridxs == r], [16, 50, 84])
                 for r in range(1, rbins.size)]
            y = percentiles_to_errors(y)
            y, dy = y[0], y[1:]

        # Include the correction for the number of beams averaged over.
        if beam_factor:
            n_beams = 2. * np.pi * rpnts / self.bmaj
            PA_min = -np.pi if PA_min is None else PA_min
            PA_max = np.pi if PA_max is None else PA_max
            if PA_min != -np.pi or PA_max != np.pi:
                arc = (PA_max - PA_min) / 2. / np.pi
                arc = max(0.0, min(arc, 1.0))
                if exclude_PA:
                    n_beams *= 1. - arc
                else:
                    n_beams *= arc
            dy /= np.sqrt(n_beams)
        return rpnts, y, dy

    def _collapse_cube(self, method='max'):
        """Collapse the cube to a 2D image using the requested method."""
        if self.data.ndim > 2:
            if method.lower() not in ['max', 'sum', 'int']:
                raise ValueError("Must choose collpase method: max, sum, int.")
            if method.lower() == 'max':
                to_avg = np.amax(self.data, axis=0)
            elif method.lower() == 'sum':
                to_avg = np.nansum(self.data, axis=0)
            else:
                to_avg = np.where(np.isfinite(self.data), self.data, 0.0)
                to_avg = np.trapz(to_avg, self.velax, axis=0)
        else:
            to_avg = self.data.copy()
        return to_avg.flatten()

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
            return convolve_fft(image, kernel)
        return convolve(image, kernel)

    def convolve_cube(self, bmaj=None, bmin=None, bpa=None, nbeams=1.0,
                      fast=True, data=None):
        """Convolve the cube with a 2D Gaussian beam."""
        if data is None:
            data = self.data
        kernel = self._beamkernel(bmaj=bmaj, bmin=bmin, bpa=bpa, nbeams=nbeams)
        convolved_cube = [self._convolve_image(c, kernel, fast) for c in data]
        return np.squeeze(convolved_cube)

    def plotbeam(self, ax, dx=0.125, dy=0.125, **kwargs):
        """Plot the sythensized beam on the provided axes."""
        from matplotlib.patches import Ellipse
        beam = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=False, hatch=kwargs.get('hatch', '////////'),
                       lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                       color=kwargs.get('color', kwargs.get('c', 'k')),
                       zorder=kwargs.get('zorder', 1000))
        ax.add_patch(beam)

    # == Rotation Functions == #

    def keplerian_profile(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z_type='thin',
                          nearest='north', params=None,  mstar=1.0, r_max=None,
                          r_min=None, dist=100., vlsr=0.0):
        """Return a Keplerian rotation profile (for the near side) in [m/s]."""
        rvals, tvals, zvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                               z_type=z_type, nearest=nearest,
                                               params=params, get_z=True,
                                               frame='polar')
        v_rot = sc.G * mstar * self.msun * np.power(rvals * dist * sc.au, 2.0)
        v_rot /= np.power(np.hypot(rvals, zvals) * sc.au * dist, 3.0)
        vrot = np.sqrt(v_rot) * np.cos(tvals) * np.sin(np.radians(inc)) + vlsr
        r_min = rvals.min() if r_min is None else r_min
        r_max = rvals.max() if r_max is None else r_max
        mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        return np.where(mask, vrot, np.nan)

    # == Functions to write a Keplerian mask for CLEANing. == #

    def CLEAN_mask(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z_type='thin',
                   nearest='north', params=None, mstar=1.0, r_max=None,
                   r_min=None, dist=100., vlsr=0.0, dV=250., nbeams=0.0,
                   dVq=0.0, fname=None, fast=True, return_mask=False):
        """
        Save a CASA readable mask using the spectral information.
        More details to be written soon...
        """

        raise NotImplementedError("Work in progress...")

        # Allow for multiple surfaces.
        vlsr = np.atleast_1d(vlsr)
        # params = [0.0]
        psis = [0.0] if psi == 0.0 else np.arange(0.0, psi, 5.0)
        mask = [self._keplerian_mask(x0=x0, y0=y0, inc=inc, PA=PA, mstar=mstar,
                                     rout=rout, rin=rin, dist=dist, vlsr=v,
                                     dV=dV, dVq=dVq, psi=p)
                for v in vlsr for p in psis]
        mask = np.nansum(mask, axis=0)
        mask = np.where(mask > 0, 1, 0)
        if mask.shape != self.data.shape:
            raise ValueError("Mask shape is not the same as the data.")

        # Include the beam smearing.
        if nbeams > 0.0:
            mask = self.convolve_cube(nbeams=nbeams, data=mask*1e2, fast=fast)

        # Return the mask if requested.
        if return_mask:
            return mask

        # Otherwise, save as a new FITS cube.
        if fname is None:
            fname = self.path.replace('.fits', '.mask.fits')
        hdu = fits.open(self.path)
        hdu[0].data = mask
        hdu[0].scale('int16')  # Might need to be int32?
        try:
            hdu.writeto(fname.replace('.fits', '') + '.fits',
                        overwrite=True, output_verify='fix')
        except TypeError:
            hdu.writeto(fname.replace('.fits', '') + '.fits',
                        clobber=True, output_verify='fix')

    def _dV_profile(self, x0, y0, inc, PA, dV, dVq=0.0):
        """Return a radial linewidth profile."""
        if dVq == 0.0:
            return dV
        rdisk = self.disk_coordinates(x0, y0, inc, PA)[0]
        return dV * np.power(rdisk, dVq)

    def _keplerian_mask(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, mstar=1.0,
                        rout=None, rin=None, dist=100, vlsr=0.0, dV=250.,
                        psi=0.0, dVq=0.0):
        """Generate the Keplerian mask as a cube. dV is FWHM of line."""
        mask = np.ones(self.data.shape) * self.velax[:, None, None]
        dV = self._dV_profile(x0, y0, inc, PA, dV, dVq)

        # Flat disk.
        if psi == 0.0:
            vkep = self._keplerian_profile(x0=x0, y0=y0, inc=inc, PA=PA,
                                           mstar=mstar, rout=rout, rin=rin,
                                           dist=dist, vlsr=vlsr)
            vkep = abs(mask - np.ones(self.data.shape) * vkep[None, :, :])
            return np.where(vkep <= dV, 1., 0.)

        # Flared disk.
        vkep = self._keplerian_profile_psi(x0=x0, y0=y0, inc=inc, PA=PA,
                                           mstar=mstar, rout=rout, rin=rin,
                                           dist=dist, vlsr=vlsr, psi=psi)
        vkep1 = abs(mask - np.ones(self.data.shape) * vkep[0][None, :, :])
        vkep2 = abs(mask - np.ones(self.data.shape) * vkep[1][None, :, :])
        return np.where(np.logical_or(vkep1 <= dV, vkep2 <= dV), 1., 0.)

    # == Masking Functions == #

    def get_mask(self, r_min=None, r_max=None, PA_min=None, PA_max=None,
                 exclude_r=False, exclude_PA=False, x0=0.0, y0=0.0, inc=0.0,
                 PA=0.0, z_type='thin', params=None, nearest='north'):
        """Returns a 2D mask for pixels in the given region."""
        rvals, tvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                        z_type=z_type, params=params,
                                        nearest=nearest, frame='polar')
        r_min = rvals.min() if r_min is None else r_min
        r_max = rvals.max() if r_max is None else r_max
        PA_min = tvals.min() if PA_min is None else PA_min
        PA_max = tvals.max() if PA_max is None else PA_max
        r_mask = np.logical_and(rvals >= r_min, rvals <= r_max)
        PA_mask = np.logical_and(tvals >= PA_min, tvals <= PA_max)
        r_mask = ~r_mask if exclude_r else r_mask
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

    def _readspectralaxis(self):
        """Returns the spectral axis in [Hz] or [m/s]."""
        a_len = self.header['naxis3']
        a_del = self.header['cdelt3']
        a_pix = self.header['crpix3']
        a_ref = self.header['crval3']
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
        if 'freq' in self.header['ctype3'].lower():
            specax = self._readspectralaxis()
            nu = self._readrestfreq()
            velax = (nu - specax) * sc.c / nu
        else:
            velax = self._readspectralaxis()
        return velax

    def _jy2k(self):
        """Jy/beam to K conversion."""
        jy2k = 1e-26 * sc.c**2 / self.nu**2 / 2. / sc.k
        return jy2k / self._calculate_beam_area_str()
