#!/usr/bin/env python3
# coding: utf-8

"""Detect loops (and other patterns) in Hi-C contact maps.
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

name = "chromosight"

MAJOR = 0
MINOR = 1
MAINTENANCE = 0
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MAINTENANCE)

LICENSE = "GPLv3"

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().splitlines()

with open("chromosight/version.py", "w") as f:
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
    entry_points={
        "console_scripts": ["chromosight=chromosight.cli.chromosight:main"]
    },
)
