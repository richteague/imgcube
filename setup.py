#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="imgcube",
    version="0.1",
    author="Richard Teague",
    author_email="rteague@umich.edu",
    description=("Analyze and maniuplate FITS cubes of protoplanetary disks."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richteague/imgcube",
    packages=["imgcube"],
    license="MIT",
    install_requires=["scipy", "numpy", "matplotlib", "astropy", "astro-eddy"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
