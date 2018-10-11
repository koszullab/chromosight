#!/usr/bin/env python3
# coding: utf-8

"""Loop detector

Detect loops in Hi-C contact maps with pattern matching.

"""

from setuptools import setup, find_packages

CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Artistic License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Visualization",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]

name = "declooptor"

MAJOR = 0
MINOR = 1
MAINTENANCE = 0
VERSION = f"{MAJOR}.{MINOR}.{MAINTENANCE}"

LICENSE = "Artistic"

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().splitlines()

with open("serpentine/version.py", "w") as f:
    f.write("__version__ = '{}'\n".format(VERSION))

setup(
    name=name,
    author="axel.cournac@pasteur.fr",
    description=__doc__,
    version=VERSION,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=["demos"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    entry_points={"console_scripts": ["declooptor=declooptor.detector:_main"]},
)
