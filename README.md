# imagecube

[![Documentation Status](https://readthedocs.org/projects/imgcube/badge/?version=latest)](https://imgcube.readthedocs.io/en/latest/?badge=latest)

A class with functions for analysing observations of protoplanetary disks (although in practice any disk would work). Everything is developed in a relatively ad-hoc manner and is therefore poorly tested or documented. Hopefully over time this will change.


### Installation

This should be a relatively painless by cloning the respository, moving into the main directory and then in the command line

```bash
pip install .
```

This should install the dependencies which are definitely needed: [`astropy`](http://www.astropy.org/) and `scipy`.

I would also recommend [`eddy`](https://github.com/richteague/eddy) if you want to do any analysis of annuli of spectra. Note that there is a lot of overlap between `eddy` and `imagecube` in terms of functionality.

### Contribution

If you use `imagecube` and notice anything wrong or something that could be improved, please [open an issue](https://github.com/richteague/imgcube/issues/new).

### Alternatives

I would also recommend [`spectral-cube`](https://github.com/radio-astro-tools/spectral-cube) if you want another suite of functions for manipulating FITS data with extensive documentation.
