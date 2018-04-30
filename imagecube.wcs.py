"""
Class to deal with the image cubes provided by CASA.
Using Astropy WCS to help.
"""

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle


class imagecube:

    def __init__(self, path, brightness=False, verbose=True):
        """Load up the image cube."""

        # Location variables.
        self.path = path
        self.fn = path.split('/')[-1]
        self.dir = path.replace(self.fn, '')
        self.verbose = verbose

        # Load up the data.
        self.hdu = fits.open(path)[0]
        self.data = np.squeeze(self.hdu.data)
        self.header = self.hdu.header
        self.nv, self.ny, self.nx = self.data.shape

        # Set up the coordiante system.
        self.coords = WCS(self.header)

        x0 = self.data.shape[-1] / 2.
        y0 = self.data.shape[-1] / 2.

        self.phase_center = self.coords.celestial.wcs_pix2world(x0, y0, 1)
        if self.verbose:
            print("Image center: %s %s" % (self.fmt_hms(self.phase_center[0]),
                                           self.fmt_dms(self.phase_center[1])))

        # Define the axes.


        print dx[0], dx[-1]
        return

    def fmt_hms(self, angle):
        """String formatting HMS."""
        return '%dh%1.dm%.3fs' % (Angle('%sd' % angle).hms)

    def fmt_dms(self, angle):
        """String formatting dms."""
        return '%dd%dm%.3fs' % (Angle('%sd' % angle).dms[0],
                                abs(Angle('%sd' % angle).dms[1]),
                                abs(Angle('%sd' % angle).dms[2]))
