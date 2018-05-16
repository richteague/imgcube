"""
Class to read in the image cubes produced with CASA or LIME. Also implements
some convenience functions to aid analysis including:

convolve_cube:      Convolve the data with a 2D Gaussian beam.
radial_profiles:    Azimuthally average the cube assuming some geometrical
                    parameters.
rotate_cube:        Rotate the cube around a given pivot pixel.
write_mask:         Generate a Keplerian mask used for CLEANing with CASA.

"""

import numpy as np
from astropy.io import fits
import scipy.constants as sc
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Kernel
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit, minimize
from prettyplots.prettyplots import sort_arrays, running_stdev
from prettyplots.gaussianprocesses import Matern32_model
from detect_peaks import detect_peaks
import warnings
warnings.filterwarnings("ignore")


class imagecube:

    msun = 1.989e30
    fwhm = 2. * np.sqrt(2. * np.log(2))

    def __init__(self, path, T_B=False, offset=True, mask=None,
                 x0=0.0, y0=0.0, inc=0.0, PA=0.0, vlsr=0.0, dist=1.0):
        """
        Read in a CASA produced image.
        """

        # Load the data and mask if appropriate.
        self.path = path
        self.data = np.squeeze(fits.getdata(path))
        if mask is not None:
            self.mask = np.squeeze(fits.getdata(mask))
        else:
            self.mask = np.ones(self.data.shape)
        self.data *= np.where(self.mask, 1, 0)

        # Flip the data array to make sure it is correct.
        '''
        if self.data.ndim == 2:
            self.data = self.data[::-1, ::-1]
        else:
            self.data = self.data[:, ::-1, ::-1]
        '''

        # Read in the data axes.
        self.header = fits.getheader(path)
        self.bunit = self.header['bunit']
        self.velax = self._readvelocityaxis(path)
        self.chan = np.mean(np.diff(self.velax))
        self.nu = self._readrestfreq()
        self.xaxis = self._readpositionaxis(path, 1, offset)
        self.yaxis = self._readpositionaxis(path, 2, offset)
        self.nxpix = int(self.xaxis.size)
        self.nypix = int(self.yaxis.size)
        self.dpix = self._pixelscale()
        self.xaxis += 2.0 * self.dpix
        self.yaxis -= 1.5 * self.dpix

        # Attempt to interpret beam parameters in [arcsec].
        try:
            self.bmaj = self.header['bmaj'] * 3600.
            self.bmin = self.header['bmin'] * 3600.
            self.bpa = self.header['bpa']
        except KeyError:
            self.bmaj = abs(self.dpix)
            self.bmin = abs(self.dpix)
            self.bpa = 0.0

        # Convert to brightness temperature [K].
        if T_B:
            self.data *= self.Tb

        # Assign already-known parameters.
        self.x0 = x0
        self.y0 = y0
        self.inc = inc
        self.PA = PA
        self.vlsr = vlsr
        self.dist = dist

        return

    def convolve_cube(self, bmin, bmaj=None, bpa=0.0, fast=True, save=False,
                      name=None):
        """
        Convolve the cube with a 2D Gaussian function.

        - Input Variables -

        bmin:       Minor axis of the beam in [arcseconds].
        bmaj:       Major axis of the beam in [arcseconds]. If not specified,
                    the beam is assumed to  be circular.
        bpa:        Position angle of the beam in [degrees]. Describes the
                    anti-clockwise angle between the major axis and the x-axis.
        fast:       Boolean describing whether to use fast fourier transforms.
        save:       Boolean describing whether the convolved cube should be
                    saved as a new output file with the extension
                    'convolved.fits' and the beam parameters included in the
                    header as usual.
        name:       Filename for the output file if the previous extension is
                    not desired.
        """

        # Fill the beam parameters and generate the kernel.
        if bmaj is None:
            bmaj = bmin
        if bmin < bmaj:
            temp = bmin
            bmin = bmaj
            bmaj = temp
        kern = self._beamkernel(bmin=bmin, bmaj=bmaj, bpa=bpa)

        # Apply the convolution on a channel-by-channel basis.
        if fast:
            cube_conv = np.array([convolve_fft(c, kern) for c in self.data])
        else:
            cube_conv = np.array([convolve(c, kern) for c in self.data])
        self.data = cube_conv

        # If appropriate, save the convolved cube.
        if save or name is not None:
            name = self._savecube(self.data, extension='.convolved', name=name)
            self._annotate_header(name, 'bmin', (bmin / 3600.))
            self._annotate_header(name, 'bmaj', (bmaj / 3600.))
            self._annotate_header(name, 'bpa', bpa)
            self.bmin = bmin
            self.bmaj = bmaj
            self.bpa = bpa
        return

    def radial_profile(self, rpnts=None, rbins=None, x0=None, y0=None,
                       inc=None, PA=None, collapse='max', statistic='mean',
                       PA_mask=None, exclude_PA_mask=False, beam_factor=False,
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

        # Define how to collapse the data.
        if self.data.ndim > 2:
            if collapse.lower() not in ['max', 'sum', 'int']:
                raise ValueError("Must choose collpase method: max, sum, int.")
            if collapse.lower() == 'max':
                to_avg = np.amax(self.data, axis=0)
            elif collapse.lower() == 'sum':
                to_avg = np.nansum(self.data, axis=0)
            else:
                to_avg = np.where(np.isfinite(self.data), self.data, 0.0)
                to_avg = np.trapz(to_avg, self.velax, axis=0)
        else:
            to_avg = self.data.copy()
        to_avg = to_avg.flatten()

        # Define the points to sample the radial profile at.
        if rbins is not None and rpnts is not None:
            raise ValueError("Specify either 'rbins' or 'rpnts'.")
        if rpnts is not None:
            dr = np.diff(rpnts)[0] * 0.5
            rbins = np.linspace(rpnts[0] - dr, rpnts[-1] + dr, len(rpnts) + 1)
        if rbins is not None:
            rpnts = np.average([rbins[1:], rbins[:-1]], axis=0)
        else:
            rbins = np.arange(0., self.xaxis.max(), 0.25 * self.bmaj)
            rpnts = np.average([rbins[1:], rbins[:-1]], axis=0)

        # Apply the deprojection.
        x0 = x0 if x0 is not None else self.x0
        y0 = y0 if y0 is not None else self.y0
        inc = inc if inc is not None else self.inc
        PA = PA if PA is not None else self.PA
        rvals, tvals = self.disk_coordinates(x0, y0, inc, PA, dist=1.0)
        rvals, tvals = rvals.flatten(), tvals.flatten()

        # Mask out values which are not within the specified PA region.
        if PA_mask is not None:
            PA_mask = np.squeeze([PA_mask])
            if len(PA_mask) != 2:
                raise ValueError("PA_mask but be the PA interval to bin.")
            mask = np.logical_and(tvals >= PA_mask[0], tvals <= PA_mask[1])
            if exclude_PA_mask:
                mask = np.where(mask, False, True)
            rvals, to_avg = rvals[mask], to_avg[mask]

        # Clip values below a certain value.
        if clip_values is not None:
            clip_values = np.squeeze([clip_values])
            if clip_values.size == 1:
                mask = abs(to_avg) >= clip_values
            else:
                mask = np.logical_or(to_avg <= clip_values[0],
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
            dy /= np.sqrt(2. * np.pi * rpnts / self.bmaj)
        return rpnts, y, dy

    def deprojectspectra(self, data=None, **kwargs):
        """
        Write a .fits file with the spectrally deprojected spectra. Required
        variables are:

        data:       Data to deproject, by default is the attached cube.
        dx:         RA offset of source centre in [arcsec].
        dy:         Dec offset of source centre in [arcsec].
        inc:        Inclination of the disk in [degrees]. Must be positive.
        pa:         Position angle of the disk in [degrees]. This is measured
                    anticlockwise from north to the blue-shifted major axis.
                    This may result in a 180 discrepancy with some literature
                    values.
        qkep        Power law index for the velocity profile. If none is
                    specified (default), then velocity is assumed to be
                    Keplerian (qkep = -0.5), otherwise a powerlaw form is used.
                    The pivot radius will always be 1 arcsecond.
        rout:       Outer radius of the disk in [arcsec]. If not specified then
                    all pixels will be deprojected. If a value is given, then
                    only pixels within that radius will be shifted, all others
                    will be masked and returned as zero.
        mstar:      Stellar mass in [Msun].
        dist:       Distance of the source in [parsec].
        save:       Save the shifted cube or not [bool].
        return:     Return the shifted data or not [bool].
        name:       Output name of the cube. By default this is the image name
                    but with the '.specdeproj' extension before '.fits'.
        """
        if data is None:
            data = self.data
        else:
            if data.shape != self.data.shape:
                raise ValueError("Unknown data shape.")

        if kwargs.get('qkep', None) is None:
            vproj = self._keplerian(image=True, **kwargs)
        else:
            vproj = self._powerlawkep(image=True, **kwargs)
            print vproj.shape
        shifted = np.zeros(data.shape)
        if shifted[0].shape != vproj.shape:
            raise ValueError("Mismatch in velocity and data array shapes.")
        for i in range(shifted.shape[2]):
            for j in range(shifted.shape[1]):
                if vproj[j, i] > 1e10:
                    pix = np.zeros(self.velax.size)
                else:
                    pix = interp1d(self.velax - vproj[j, i], data[:, j, i],
                                   fill_value=0.0, bounds_error=False,
                                   assume_sorted=True)
                    pix = pix(self.velax)
                shifted[:, j, i] = pix

        if kwargs.get('save', True):
            extension = '.specdeproj'
            if kwargs.get('qkep', None) is not None:
                extension += '.nonKep'
            self._savecube(shifted, extension=extension, **kwargs)
        else:
            return shifted

    def _annotate_header(self, filename, key, value):
        """Update / add the header key and value."""
        with fits.open(filename, 'update') as f:
            f[0].header[key] = value
        return

    def get_radius(self, radius, dr=0.1, **kwargs):
        """Return the azimuthally averaged value at r +/- dr [arcsec]."""
        rvals, _ = self._deproject(**kwargs)
        rpnts = rvals.flatten()
        if self.data.ndim == 3:
            ppnts = self.data.reshape(self.data.shape[0], -1).T
        else:
            ppnts = self.data.flatten()
        if kwargs.get('brightness', True):
            ppnts *= self.Tb
        ridxs = np.digitize(rpnts, [radius - dr, radius + dr])
        pavgs = np.nanmean(ppnts[ridxs == 1], axis=0)
        pstds = np.nanstd(ppnts[ridxs == 1], axis=0)
        return pavgs, pstds

    def restore_data(self, **kwargs):
        """Restore the self.data to the original data."""
        self.data = np.squeeze(fits.getdata(self.path))
        self.data = np.swapaxes(self.data, -2, -1)
        try:
            self.bmaj = self.header['bmaj'] * 3600.
            self.bmin = self.header['bmin'] * 3600.
            self.bpa = self.header['bpa']
        except KeyError:
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0
        if kwargs.get('return', False):
            return self.data

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

    @property
    def Tb(self):
        """Calculate the Jy/beam to K conversion."""
        if self.bunit == 'K':
            return 1.0
        return 1e-26 * sc.c**2 / self.nu**2 / 2. / sc.k / self._beamarea()

    @property
    def beamperpix(self):
        """Number of beams per pixel."""
        Abeam = np.pi * np.radians(self.bmin / 3600.)
        Abeam *= np.radians(self.bmaj / 3600.) / 4. / np.log(2.)
        Apixel = np.radians(self.dpix / 3600.)**2
        return Apixel / Abeam

    def _beamarea(self):
        """Beam area in steradians."""
        omega = np.radians(self.bmin / 3600.)
        omega *= np.radians(self.bmaj / 3600.)
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    def _pixelscale(self):
        """Returns the average pixel scale of the image."""
        return np.mean([abs(np.mean(np.diff(self.xaxis))),
                        abs(np.mean(np.diff(self.yaxis)))])

    def _spectralaxis(self, fn):
        """Returns the spectral axis in [Hz]."""
        a_len = fits.getval(fn, 'naxis3')
        a_del = fits.getval(fn, 'cdelt3')
        a_pix = fits.getval(fn, 'crpix3')
        a_ref = fits.getval(fn, 'crval3')
        return a_ref + (np.arange(a_len) - a_pix + 0.5) * a_del

    def _savecube(self, newdata, extension='', name=None):
        """Save a new .fits file with the appropriate data."""
        hdu = fits.open(self.path)
        hdu[0].data = newdata
        if name is None:
            name = self.path.replace('.fits', '%s.fits' % extension)
        name = name.replace('.fits', '') + '.fits'
        hdu.writeto(name, overwrite=True, output_verify='fix+warn')
        return name

    def _annotateheader(self, hdr, **kwargs):
        """Include the model parameters in the header."""
        hdr['INC'] = kwargs['inc'], 'disk inclination [degrees].'
        hdr['PA'] = kwargs['pa'], 'disk position angle [degrees].'
        hdr['MSTAR'] = kwargs['mstar'], 'source mass [Msun].'
        hdr['DV'] = kwargs['dV'], 'intrinsic linewidth [km/s].'
        hdr['DX'] = kwargs.get('dx', 0.0), 'RA offset [arcsec].'
        hdr['DY'] = kwargs.get('dy', 0.0), 'Dec offset [arcsec].'
        for v, v0 in enumerate(np.array(kwargs.get('vlsr', 0.0)).flatten()):
            hdr['VSYS%d' % v] = v0, 'systemic velocity [km/s].'
        return hdr

    def write_mask(self, inc=0.0, PA=0.0, mstar=1.0, rout=None, dist=1.0,
                   dV=500., vlsr=0.0, psi=0.0, x0=0.0, y0=0.0, nbeams=0.0,
                   fast=True, return_mask=False, name=None, rin=None,
                   single_layer=False):
        """
        Write a .fits file of the mask. Imporant variables are:

        - Input -

        name:           Output name of the mask. By default it is the image
                        name but with the '.mask' extension before '.fits'.
        inc:            Inclination of the disk in [degrees].
        pa:             Position angle of the disk in [degrees]. This is
                        measured anticlockwise from north to the blue-shifted
                        major axis. This may result in a 180 discrepancy with
                        some literature values.
        mstar:          Mass of the central star [solar masses].
        rout:           Outer radius of the disk in [arcsec].
        dist:           Distance of the source in [parsec].
        dV:             Expected line width of the source in [m/s].
        vlsr:           Systemic velocity of the source in [m/s]. This can be a
                        list of velocities and the mask will be the summation
                        of masks centred at each velocity.
        psi:            Angle between the emission surface and the midplane. If
                        not zero then will mask at two inclinations to mimic
                        the splitting expected for molecular lines.
        dx:             RA offset of source centre in [arcsec].
        dy:             Dec offset of source centre in [arcsec].
        nbeams:         Number of beams to convolve the mask with.
        fast:           Use FFT in the convolution. Default is True.
        single_layer:   Only use a single psi value. This can lead to missing
                        emission that originates closer towards the midplane.

        - Returns -

        mask:           Boolean mask of where emission is to be expected.
        """

        # Define the different masks to sum over. This includes different
        # flaring angles and source velocities (for hyperfine components).
        if psi <= 3.0 or single_layer:
            psis = np.atleast_1d(psi)
        else:
            psis = (psi - np.unique(np.arange(0.0, psi, 3.0)))[::-1]

        # Generate the mask, looping through various parameters.
        mask = [self._mask(inc=inc, PA=PA, x0=x0, y0=y0, vlsr=v0, dV=dV,
                           dist=dist, mstar=mstar, psi=p, rout=rout, rin=rin)
                for v0 in np.atleast_1d(vlsr) for p in psis]
        mask = np.where(np.sum(mask, axis=0) > 0, 1, 0)

        # Apply the beam convolution.
        if nbeams > 0.0:
            kern = self._beamkernel(nbeams=nbeams)
            if fast:
                mask = np.array([convolve_fft(c, kern) for c in mask])
            else:
                mask = np.array([convolve(c, kern) for c in mask])
            mask = np.where(mask > 0.01, 1, 0)

        # Return the mask.
        if return_mask:
            return mask

        # Replace the data, swapping axes as appropriate.
        hdu = fits.open(self.path)
        hdu[0].data = mask
        if name is None:
            name = self.path.replace('.fits', '.mask.fits')
        hdu[0].scale('int32')

        # Try to remove the multiple beams.
        hdu[0].header['CASAMBM'] = False
        # Save the mask.
        try:
            hdu.writeto(name.replace('.fits', '') + '.fits',
                        overwrite=True, output_verify='fix')
        except TypeError:
            hdu.writeto(name.replace('.fits', '') + '.fits',
                        clobber=True, output_verify='fix')

    def _mask(self, inc=0.0, PA=0.0, x0=0.0, y0=0.0, vlsr=0.0, dV=500.,
              dist=1.0, mstar=1.0, psi=0.0, rout=None, rin=None):
        """
        Returns the Keplerian mask.

        - Input -

        inc:        Inclination of the disk in degrees.
        PA:         Position angle of the disk in degrees, measured as the
                    angle between the blue-shifted major-axis and North in an
                    anticlockwise direction.
        x0, y0:     Relative offsets of the centre of the disk in [arcsec].
        vlsr:       Velocity of the source in [m/s].
        dV:         Expected intrinsic linewidth in [m/s].
        dist:       Distance to the source in [pc].
        mstar:      Mass of the central star in [Msun].
        psi:        Flaring angle [deg] between the emission surface and
                    the midplane.
        rout:       Outer radius of the mask in [au].
        rin:        Inner radius of the mask in [au].

        - Returns =

        mask:       Integer mask of regions expected to have emission.
        """

        vkep = self._keplerian(mstar=mstar, inc=inc, PA=PA, x0=x0, y0=y0,
                               dist=dist, psi=psi, rout=rout, rin=rin,
                               vfill=1e20)
        vdat = (self.velax - vlsr)[:, None, None] * np.ones(self.data.shape)
        if psi == 0.0:
            vkep = vkep[None, :, :] * np.ones(self.data.shape)
            mask = np.where(abs(vkep - vdat) <= dV, 1, 0)
        else:
            vkep_near = vkep[0][None, :, :] * np.ones(self.data.shape)
            vkep_far = vkep[1][None, :, :] * np.ones(self.data.shape)
            mask_near = np.where(abs(vkep_near - vdat) <= dV, 1, 0)
            mask_far = np.where(abs(vkep_far - vdat) <= dV, 1, 0)
            mask = np.logical_or(mask_near, mask_far)
        return mask

    def _keplerian(self, mstar=1.0, inc=0.0, PA=0.0, x0=0.0, y0=0.0, dist=1.0,
                   psi=0.0, rout=None, rin=None, vfill=np.nan):
        """
        Returns the projected Keplerian velocity in [m/s].

        - Input -

        mstar:      Mass of the central star in [Msun].
        inc:        Inclination of the disk in [deg].
        PA:         Position angle of the disk in [deg], measured as the angle
                    between the blue-shifted major-axis and North in an
                    anticlockwise direction.
        x0, y0:     Relative offsets of the centre of the disk in [arcsec].
        dist:       Distance to the source in [pc].
        psi:        Flaring angle [deg] between the emission surface and the
                    midplane. Examples include 15 degrees for 12CO or 9 degrees
                    for 13CO.
        rout:       Outer radius of the disk in [au]. Outside this radius the
                    returned velocity will be `vfill`.
        rin:        Inner radius of the disk in [au]. Inside this radius the
                    returned velocity will be `vfill`.
        vfill:      Value to fill pixels outside the disk.

        - Returns -

        vkep:       Project Keplerian velocity in [m/s].
        """

        # Calculate the Keplerian rotation across the whole disk.
        x_sky, y_sky = np.meshgrid(self.xaxis[::-1] + 2.5 * self.dpix - x0,
                                   self.yaxis - 0.5 * self.dpix - y0)
        x_rot, y_rot = self._rotate(x_sky, y_sky, PA + 90.)
        t_pos, t_neg = self._solve_quadratic(x_rot, y_rot, inc, psi)
        y_disk = y_rot / np.cos(np.radians(inc))
        y_near = y_disk + t_pos * np.sin(np.radians(inc))
        y_far = y_disk + t_neg * np.sin(np.radians(inc))
        r_near, r_far = np.hypot(x_rot, y_near), np.hypot(x_rot, y_far)
        t_near, t_far = np.arctan2(y_near, x_rot), np.arctan2(y_far, x_rot)

        # Projected Keplerian velocities.
        v_near = np.sqrt(sc.G * mstar * self.msun / r_near / sc.au / dist)
        v_near *= np.sin(np.radians(inc)) * np.cos(t_near)
        v_far = np.sqrt(sc.G * mstar * self.msun / r_far / sc.au / dist)
        v_far *= np.sin(np.radians(inc)) * np.cos(t_far)

        # Mask pixels outside the disk and return.
        if rout is None:
            rout = np.nanmax([r_near, r_far])
        else:
            rout /= dist
        if rin is None:
            rin = 0.0
        else:
            rin /= dist
        v_far = np.where(np.logical_and(r_far >= rin, r_far <= rout),
                         v_far, vfill)
        v_near = np.where(np.logical_and(r_near >= rin, r_near <= rout),
                          v_near, vfill)

        # If there is no flaring, return just the midplane velocity.
        if psi == 0.0:
            return v_far
        return v_far, v_near

    def disk_coordinates(self, x0=None, y0=None, inc=None, PA=None, dist=1.0):
        """
        Convert pixel coordinates to disk polar coordinates.
        Returns r [au], theta [radians].
        TODO: why do we need the corrections?
        """

        x0 = x0 if x0 is not None else self.x0
        y0 = y0 if y0 is not None else self.y0
        inc = inc if inc is not None else self.inc
        PA = PA if PA is not None else self.PA

        x, y = np.meshgrid(self.xaxis[::-1] - self.dpix - x0, self.yaxis - y0)
        x_sky, y_sky = x * dist, y * dist
        x_rot, y_rot = self._rotate(x_sky, y_sky, PA + 90.)
        x_dep, y_dep = self._incline(x_rot, y_rot, inc)
        return np.hypot(x_dep, y_dep), np.arctan2(y_dep, x_dep)

    def _rotate(self, x, y, PA):
        '''Clockwise rotation about origin by PA [deg].'''
        x_rot = x * np.cos(np.radians(PA)) + y * np.sin(np.radians(PA))
        y_rot = y * np.cos(np.radians(PA)) - x * np.sin(np.radians(PA))
        return x_rot, y_rot

    def _incline(self, x, y, inc):
        '''Incline the image by angle inc [deg] about x-axis.'''
        return x, y / np.cos(np.radians(inc))

    def _beamkernel(self, bmaj=None, bmin=None, bpa=None, nbeams=1.0):
        """Returns the 2D Gaussian kernel. CHECK ROTATION."""
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
        x, y = self._rotate(x, y, PA)
        k = np.power(x / dx, 2) + np.power(y / dy, 2)
        return np.exp(-0.5 * k) / 2. / np.pi / dx / dy

    def _velocityaxis(self, fn):
        """Return velocity axis in [m/s]."""
        a_len = fits.getval(fn, 'naxis3')
        a_del = fits.getval(fn, 'cdelt3')
        a_pix = fits.getval(fn, 'crpix3')
        a_ref = fits.getval(fn, 'crval3')
        return (a_ref + (np.arange(a_len) - a_pix + 1.0) * a_del)

    def _readvelocityaxis(self, fn):
        """Wrapper for _velocityaxis and _spectralaxis."""
        if 'freq' in fits.getval(fn, 'ctype3').lower():
            specax = self._spectralaxis(fn)
            try:
                nu = fits.getval(fn, 'restfreq')
            except KeyError:
                try:
                    nu = fits.getval(fn, 'restfrq')
                except KeyError:
                    nu = np.mean(specax)
            velax = (nu - specax) * sc.c / nu
            if velax[0] > velax[-1]:
                velax = velax[::-1]
            return velax
        else:
            return self._velocityaxis(fn)

    def _solve_quadratic(self, x, y, inc_in, psi_in):
        """Solve Eqn.(5) from Rosenfeld et al. (2013)."""
        inc, psi = np.radians(inc_in), np.radians(psi_in)
        a = np.cos(2.*inc) + np.cos(2.*psi)
        b = -4. * np.sin(psi)**2 * y * np.tan(inc)
        c = -2. * np.sin(psi)**2 * (x**2 + np.power(y / np.cos(inc), 2))
        t_p = -b + np.sqrt(b**2 - 4 * a * c) / 2. / a
        t_n = -b - np.sqrt(b**2 - 4 * a * c) / 2. / a
        return t_p, t_n

    def _readpositionaxis(self, fn, a=1, offset=True):
        """Returns the position axis in ["]."""
        if a not in [1, 2]:
            raise ValueError("'a' must be in [0, 1].")
        a_len = fits.getval(fn, 'naxis%d' % a)
        a_del = fits.getval(fn, 'cdelt%d' % a)
        a_pix = fits.getval(fn, 'crpix%d' % a)
        if offset:
            a_pix = 0.5 * a_len
        return 3600. * ((np.arange(1, a_len+1) - a_pix + 1.0) * a_del)

    def plotbeam(self, ax, dx=0.15, dy=0.15, **kwargs):
        """Plot the synthesized beam."""
        beam = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=False, hatch=kwargs.get('hatch', '////////'),
                       lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                       color=kwargs.get('color', kwargs.get('c', 'k')))
        ax.add_patch(beam)
        return

    def emission_surface(self, x0=None, y0=None, inc=None, clip=0.0,
                         method='GP', r_max=None, rbins=None, dist=1.0):
        """
        Derive the emission surface following Pinte et al. (2018), however use
        a Gaussian Process model rather than binning the data.

        - Input -

        x0, y0:             Source centre offset in [arcsec].
        inc:                Inclination of the disk in [degrees].
        clip:               Threshold used for clipping points. Only points
                            above <y> - clip * <dy> will be considered in the
                            emisison profile where <dy> is the noise on the
                            azimuthally averaged intensity profile.
        method:             Method for creating the radial profile. 'GP' will
                            use a Gaussian Process using a Matern 3/2 kernel.
                            'binning' will bin the points, 'raw' will return
                            the clipped pixels from _get_emission_surface().
        dist:               Distance to the source in [pc]. A dist = 1 will
                            return the values in [arcsec].
        """

        # Measure the radial intensity (/brightness) profile.
        x, y, dy = self.radial_profile(collapse='max', beam_factor=True)

        # Infer the emission surface.
        x0 = x0 if x0 is not None else self.x0
        y0 = y0 if y0 is not None else self.y0
        inc = inc if inc is not None else self.inc
        emission_surface = self._get_emission_surface(x0, y0, inc, r_max=r_max)

        # Clip the data based on radius and intensity.
        r_max = r_max if r_max is not None else self.xaxis.max()
        clip = interp1d(x, y - clip * dy, fill_value='extrapolate')
        mask = emission_surface[2] >= clip(emission_surface[0])
        mask *= emission_surface[0] <= r_max
        r, z, Tb = emission_surface[:, mask]

        # Make the radial profile.
        if method.lower() not in ['gp', 'binned', 'raw']:
            raise ValueError("method must be 'gp', 'binned' or None.")
        if method.lower() == 'gp':
            r, z = sort_arrays(r, z)
            dz = running_stdev(z, window=(self.bmaj / np.nanmean(np.diff(r))))
            r, z, dz = Matern32_model(r, z, dz, jitter=True, return_var=True)
        elif method.lower() == 'binned':
            if rbins is None:
                rbins = np.arange(0, r_max, 0.25 * self.bmaj)
            ridxs = np.digitize(r, rbins)
            dz = [np.nanstd(z[ridxs == rr]) for rr in range(1, rbins.size)]
            z = [np.nanmean(z[ridxs == rr]) for rr in range(1, rbins.size)]
            z, dz = np.squeeze(z), np.squeeze(dz)
            r = np.average([rbins[1:], rbins[:-1]], axis=0)
        else:
            r, z = sort_arrays(r, z)
            dz = running_stdev(z, window=(self.bmaj / np.nanmean(np.diff(r))))
        return r * dist, z * dist, dz * dist

    def _get_emission_surface(self, x0, y0, inc, r_max=None, mph=0.0, mpd=0.0):
        """
        Infer a list of [r, z, Tb] coordiantes for the emission surface.

        - Input Variables -

        x0, y0:     Coordinates [arcseconds] of the centre of the disk.
        inc         Inclination [degrees] of the disk.
        r_max:      Maximum radius to apply fitting to.
        mph:        Minimum peak height in [K]. If None, use all pixels.
        mpd:        Minimum distance between peaks in [arcseconds].

        - Output -

        coords:     A [3 x N] array where N is the number of successfully found
                    ellipses. Each ellipse yields a (r, z, Tb) trio. Distances
                    are in [au] (coverted using the provided distance) and the
                    brightness temperature in [K].
        """

        coords = []
        r_max = self.xaxis.max() if r_max is None else r_max
        for c, channel in enumerate(self.data):

            # Avoid empty channels.
            if np.nanmax(channel) < mph:
                continue

            # Cycle through the columns in the channel.
            for xidx in range(self.nxpix):

                # Skip rows if appropriate.
                if abs(self.xaxis[xidx] - x0) > r_max:
                    continue
                if np.nanmax(channel[:, xidx]) < mph:
                    continue

                # Find the indices of the two largest peaks.
                yidx = detect_peaks(channel[:, xidx], mph=mph, mpd=mpd)
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
        return np.squeeze(coords).T

    def rotation_profile(self, rpnts=None, rbins=None, x0=None, y0=None,
                         inc=None, PA=None, method='dV', resample=True,
                         verbose=True):
        """
        Calculate the rotation profile from the spectra width.
        """

        # Find the pixel coordinates.
        x0 = x0 if x0 is not None else self.x0
        y0 = y0 if y0 is not None else self.y0
        inc = inc if inc is not None else self.inc
        PA = PA if PA is not None else self.PA
        rvals, tvals = self.disk_coordinates(x0, y0, inc, PA, dist=1.0)
        rvals, tvals = rvals.flatten(), tvals.flatten()

        # Flatten the data to [velocity, nxpix * nypix].
        dvals = self.data.copy().reshape(self.data.shape[0], -1)
        if dvals.shape != (self.velax.size, self.nxpix * self.nypix):
            raise ValueError("Wrong data shape.")

        # Calculate the radial binning.
        if rpnts is not None and rbins is not None:
            raise ValueError("Only specify either 'rpnts' or 'rbins'.")
        if rpnts is None and rbins is None:
            rbins = np.arange(0, self.xaxis.max(), 0.25 * self.bmaj)
            rpnts = np.average([rbins[1:], rbins[:-1]], axis=0)
        elif rpnts is None:
            rpnts = np.average([rbins[1:], rbins[:-1]], axis=0)
        else:
            dr = 0.5 * np.average(np.diff(rpnts))
            rbins = np.linspace(rpnts[0] - dr, rpnts[-1] + dr, rpnts.size + 1)

        # Make sure the method is selected.
        if method.lower() not in ['dv', 'gp']:
            raise ValueError("Method must be 'dV' or 'GP'.")

        # Cycle through each annulus and apply the method.
        v_rot = []
        ridxs = np.digitize(rvals, rbins)
        for r in range(1, rbins.size):

            if verbose:
                print("Running %d / %d." % (r, rpnts.size))

            spectra = dvals[:, ridxs == r].T
            angles = tvals[ridxs == r]

            if method.lower() == 'dv':
                v_rot += [self._vrot_from_dV(spectra, angles, resample)]
            elif method.lower() == 'gp':
                v_rot += [self._vrot_from_GP(spectra, angles, resample)]

        return rpnts, np.squeeze(v_rot)

    def _vrot_from_dV(self, spectra, angles, resample=True):
        """
        Calculate v_rot through minimizing the width.
        """
        vrot, _ = self._estimate_vrot(spectra, angles)
        args = (spectra, angles, resample)
        res = minimize(self._deprojected_width, vrot, args=args,
                       method='Nelder-Mead')
        return abs(res.x[0]) if res.success else np.nan

    def _estimate_vrot(self, spectra, angles):
        """Estimate the rotation velocity from fitting a SHO to peaks."""
        vpeaks = np.take(self.velax, np.argmax(spectra, axis=1))
        p0 = [0.5 * (np.max(vpeaks) - np.min(vpeaks)), np.mean(vpeaks)]
        try:
            popt, _ = curve_fit(offsetSHO, angles, vpeaks, p0=p0, maxfev=10000)
        except:
            popt = p0
        return np.squeeze(popt)

    def _deprojected_width(self, vrot, spectra, angles, resample=True):
        """Width of the deprojected line profile."""

        # Deproject the spectrum.
        x, y = self._deprojected_spectrum(spectra, angles, vrot, resample)

        # Estimate starting positions for the Gaussian fit.
        Tb = np.max(y)
        dV = np.trapz(y, x) / Tb / np.sqrt(np.pi)
        x0 = x[y.argmax()]
        p0 = [Tb, dV, x0]

        # Return the Gaussian fit.
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
        """Deproject all the spectra given the rotation velocity."""
        deprojected = [interp1d(self.velax - vrot * np.cos(angle), spectrum,
                                fill_value='extrapolate')(self.velax)
                       for spectrum, angle in zip(spectra, angles)]
        return np.squeeze(deprojected)


def gaussian(x, x0, dx, A):
    """Gaussian function with Doppler width."""
    return A * np.exp(-np.power((x-x0) / dx, 2))


def offsetSHO(theta, A, y0):
    """Simple harmonic oscillator with an offset."""
    return A * np.cos(theta) + y0


def percentiles_to_errors(pcnts):
    """Covert [16, 50, 84]th percentiles to [y, -dy, +dy]."""
    pcnts = np.squeeze([pcnts])
    if pcnts.ndim > 1:
        if pcnts.shape[1] != 3 and pcnts.shape[0] == 3:
            pcnts = pcnts.T
        if pcnts.shape[1] != 3:
            raise TypeError("Must provide a Nx3 or 3xN array.")
        return np.array([[p[1], p[1]-p[0], p[2]-p[1]] for p in pcnts]).T
    return np.squeeze([pcnts[1], pcnts[1]-pcnts[0], pcnts[2]-pcnts[1]])
