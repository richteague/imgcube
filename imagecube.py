"""
Class to read in the image cubes produced with CASA. Incorporates some simple
functions which help analyse the data, for example creating a Keplerian mask
for cleaning or spectrally deprojecting each pixel to a common VLSR.

TODO:
    1 - Include a better way to annotate the headers. Maybe some form of dict.
"""

import numpy as np
from astropy.io import fits
import scipy.constants as sc
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Kernel
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")


class imagecube:

    msun = 1.989e30
    fwhm = 2. * np.sqrt(2. * np.log(2))

    def __init__(self, path):
        """Read in a CASA produced image."""
        self.path = path
        self.data = np.squeeze(fits.getdata(path))
        self.header = fits.getheader(path)
        self.velax = self._readvelocityaxis(path)
        self.chan = np.mean(np.diff(self.velax))
        self.nu = self._readrestfreq()
        self.xaxis = self._readpositionaxis(path, 1)
        self.yaxis = self._readpositionaxis(path, 2)
        self.nxpix = int(self.xaxis.size)
        self.nypix = int(self.yaxis.size)
        self.dpix = self._pixelscale()
        self.bmaj = self.header['bmaj'] * 3600.
        self.bmin = self.header['bmin'] * 3600.
        self.bpa = self.header['bpa']
        return

    def azimithallyaverage(self, data=None, rpnts=None, **kwargs):
        """
        Azimuthally average a cube. Variables are:

        data:       Data to deproject, by default is the attached cube.
        rpnts:      Points to average at in [arcsec]. If none are given, assume
                    beam spaced points across the radius of the image.
        deproject:  If the cube first be deprojected before averaging [bool].
                    By default this is true. If so, the fields required by
                    self.deprojectspectra are necessary.
        """

        # Choose the data to azimuthally average.
        if data is None:
            data = self.data
        else:
            if data.shape != self.data.shape:
                raise ValueError("Unknown data shape.")

        # Define the points to sample the radial profile at.
        if rpnts is None:
            rbins = np.arange(0., self.xaxis.max(), self.bmaj)
        else:
            dr = np.diff(rpnts)[0] * 0.5
            rbins = np.linspace(rpnts[0] - dr, rpnts[-1] + dr, len(rpnts) + 1)
        nbin = rbins.size

        # Apply the deprojection if required.
        if kwargs.get('deproject', True):
            data = self.deprojectspectra(data=data, save=False, **kwargs)

        # Apply the averaging.
        rvals, _ = self._deproject(**kwargs)
        ridxs = np.digitize(rvals.ravel(), rbins)
        data = data.reshape((data.shape[0], -1)).T
        avg = [np.nanmean(data[ridxs == r], axis=0) for r in range(1, nbin)]
        return np.squeeze(avg)

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

    def writemask(self, **kwargs):
        """
        Write a .fits file of the mask. Imporant variables are:

        name:       Output name of the mask. By default it is the image name
                    but with the '.mask' extension before '.fits'.
        inc:        Inclination of the disk in [degrees]. Must be positive.
        pa:         Position angle of the disk in [degrees]. This is measured
                    anticlockwise from north to the blue-shifted major axis.
                    This may result in a 180 discrepancy with some literature
                    values.
        mstar:      Mass of the central star [solar masses].
        rout:       Outer radius of the disk in [arcsec].
        dist:       Distance of the source in [parsec].
        dV:         Expected line width of the source in [km/s].
        vlsr:       Systemic velocity of the source in [km/s]. This can be a
                    list of velocities and the mask will be the summation of
                    masks centred at each velocity.
        dx:         RA offset of source centre in [arcsec].
        dy:         Dec offset of source centre in [arcsec].
        nbeams:     Number of beams to convolve the mask with. Default is 1.
        fast:       Use FFT in the convolution. Default is True.
        """

        vlsr = np.array(kwargs.pop('vlsr', 0.0)).flatten()
        mask = np.sum([self._mask(vlsr=v0, **kwargs) for v0 in vlsr], axis=0)
        kwargs['vlsr'] = vlsr

        # Apply the beam convolution.
        if kwargs.get('nbeams', 0.0) > 0.0:
            kern = self._beamkernel(**kwargs)
            if kwargs.get('fast', True):
                mask = np.array([convolve_fft(c, kern) for c in mask])
            else:
                mask = np.array([convolve(c, kern) for c in mask])
        mask = np.where(mask > 1e-10, 1, 0)

        # Replace the data, swapping axes as appropriate.
        # I'm not sure why this works but it does...
        hdu = fits.open(self.path)
        hdu[0].data = np.swapaxes(mask, 1, 2)
        if kwargs.get('name', None) is None:
            name = self.path.replace('.fits', '.mask.fits')
        else:
            name = kwargs.get('name')
        hdu[0].scale('int32')
        hdu[0].header = self._annotateheader(hdu[0].header, **kwargs)
        hdu.writeto(name.replace('.fits', '') + '.fits',
                    overwrite=True, output_verify='fix')
        if kwargs.get('return', False):
            return mask

    def _readrestfreq(self):
        """Read the rest frequency."""
        try:
            nu = self.header['restfreq']
        except KeyError:
            nu = self.header['restfrq']
        return nu

    @property
    def Tb(self):
        """Calculate the Jy/beam to K conversion."""
        omega = np.pi * np.radians(self.header['bmin'])
        omega *= np.radians(self.header['bmaj']) / 4. / np.log(2.)
        return 1e-26 * sc.c**2 / self.nu**2 / 2. / sc.k / omega

    @property
    def beamperpix(self):
        """Number of beams per pixel."""
        Abeam = np.pi * np.radians(self.header['bmin'])
        Abeam *= np.radians(self.header['bmaj']) / 4. / np.log(2.)
        Apixel = np.radians(self.dpix / 3600.)**2
        return Apixel / Abeam

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

    def _savecube(self, newdata, extension='', **kwargs):
        """Save a new .fits file with the appropriate data."""
        hdu = fits.open(self.path)
        hdu[0].data = np.swapaxes(newdata, 1, 2)
        name = kwargs.get('name', None)
        if kwargs.get('name', None) is None:
            name = self.path.replace('.fits', '%s.fits' % extension)
        # hdu[0].scale('int32')
        hdu.writeto(name.replace('.fits', '') + '.fits',
                    overwrite=True, output_verify='fix')
        return

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

    def _beamkernel(self, **kwargs):
        """Returns the 2D Gaussian kernel."""
        bmaj = self.bmaj / self.dpix / self.fwhm
        bmin = self.bmin / self.dpix / self.fwhm
        bpa = np.radians(self.bpa)
        if kwargs.get('nbeams', 1.0) > 1.0:
            bmin *= kwargs.get('nbeams', 1.0)
            bmaj *= kwargs.get('nbeams', 1.0)
        return Kernel(self._gaussian2D(bmin, bmaj, pa=bpa))

    def _mask(self, **kwargs):
        """Returns the Keplerian mask."""
        rsky, tsky = self._diskpolar(**kwargs)
        vkep = self._keplerian(**kwargs)
        vdat = self.velax - kwargs.get('vlsr', 2.89) * 1e3
        vdat = vdat[:, None, None] * np.ones(self.data.shape)
        dV = 0.5 * (kwargs.get('dV', .3) * 1e3 + self.chan)
        return np.where(abs(vkep - vdat) <= dV, 1, 0)

    def _keplerian(self, **kwargs):
        """Returns the projected Keplerian velocity [m/s]."""
        rsky, tsky = self._diskpolar(**kwargs)
        vkep = np.sqrt(sc.G * kwargs.get('mstar', 0.7) * self.msun / rsky)
        vkep *= np.sin(np.radians(kwargs.get('inc', 6.))) * np.cos(tsky)
        rout = kwargs.get('rout', 4) * sc.au * kwargs.get('dist', 1.0)
        vkep = np.where(rsky <= rout, vkep, kwargs.get('vfill', 1e20))
        if kwargs.get('image', False):
            return vkep[0]
        return vkep

    def _powerlawkep(self, **kwargs):
        """Returns the projected velocity profile [m/s]."""
        rsky, tsky = self._deproject(**kwargs)
        dist = kwargs.get('dist', 1.0)
        mstar = kwargs.get('mstar', 0.7)
        qkep = kwargs.get('qkep', -0.5)
        rout = kwargs.get('rout', 4.)
        inc = np.radians(kwargs.get('inc', 6.))
        vproj = np.sqrt(sc.G * mstar * self.msun / dist / sc.au)
        vproj *= np.power(rsky, qkep) * np.sin(inc) * np.cos(tsky)
        vproj = np.where(rsky <= rout, vproj, kwargs.get('vfill', 1e20))
        if kwargs.get('image', False):
            return vproj
        return vproj[None, :, :] * np.ones(self.data.shape)

    def _diskpolar(self, **kwargs):
        """Returns the polar coordinates of the sky in [m] and [rad]."""
        rsky, tsky = self._deproject(**kwargs)
        rsky *= kwargs.get('dist', 1.0) * sc.au
        rsky = rsky[None, :, :] * np.ones(self.data.shape)
        tsky = tsky[None, :, :] * np.ones(self.data.shape)
        return rsky, tsky

    def _deproject(self, **kwargs):
        """Returns the deprojected pixel values, (r, theta)."""
        inc, pa = kwargs.get('inc', 0.0), kwargs.get('pa', 0.0)
        dx, dy = kwargs.get('dx', 0.0), kwargs.get('dy', 0.0)
        x_sky = self.xaxis[None, :] * np.ones(self.nypix)[:, None] - dx
        y_sky = self.yaxis[:, None] * np.ones(self.nxpix)[None, :] - dy
        x_rot, y_rot = self._rotate(x_sky, y_sky, np.radians(pa))
        x_dep, y_dep = self._incline(x_rot, y_rot, np.radians(inc))
        return np.hypot(x_dep, y_dep), np.arctan2(y_dep, x_dep)

    def _velocityaxis(self, fn):
        """Return velocity axis in [km/s]."""
        a_len = fits.getval(fn, 'naxis3')
        a_del = fits.getval(fn, 'cdelt3')
        a_pix = fits.getval(fn, 'crpix3')
        a_ref = fits.getval(fn, 'crval3')
        return (a_ref + (np.arange(a_len) - a_pix + 0.5) * a_del)

    def _readvelocityaxis(self, fn):
        """Wrapper for _velocityaxis and _spectralaxis."""
        if fits.getval(fn, 'ctype3').lower() == 'freq':
            specax = self._spectralaxis(fn)
            try:
                nu = fits.getval(fn, 'restfreq')
            except KeyError:
                nu = fits.getval(fn, 'restfrq')
            return (nu - specax) * sc.c / nu
        else:
            return self._velocityaxis(fn)

    def _readpositionaxis(self, fn, a=1):
        """Returns the position axis in ["]."""
        if a not in [1, 2]:
            raise ValueError("'a' must be in [0, 1].")
        a_len = fits.getval(fn, 'naxis%d' % a)
        a_del = fits.getval(fn, 'cdelt%d' % a)
        a_pix = fits.getval(fn, 'crpix%d' % a)
        return 3600. * ((np.arange(1, a_len+1) - a_pix + 0.5) * a_del)

    def _rotate(self, x, y, t):
        '''Rotation by angle t [rad].'''
        x_rot = x * np.cos(t) + y * np.sin(t)
        y_rot = y * np.cos(t) - x * np.sin(t)
        return x_rot, y_rot

    def _incline(self, x, y, i):
        '''Incline the image by angle i [rad].'''
        return x, y / np.cos(i)

    def _gaussian2D(self, dx, dy, pa=0.0):
        """2D Gaussian kernel in pixel coordinates."""
        xm = np.arange(-4*max(dy, dx), 4*max(dy, dx)+1)
        x, y = np.meshgrid(xm, xm)
        x, y = self._rotate(x, y, pa)
        k = np.power(x / dx, 2) + np.power(y / dy, 2)
        return np.exp(-0.5 * k) / 2. / np.pi / dx / dy
