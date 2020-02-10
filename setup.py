#!/usr/bin/env python3
# coding: utf-8

"""Detect loops (and other patterns) in Hi-C contact maps."""

from setuptools import setup, find_packages
import codecs

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Artistic License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Visualization",
    "Operating System :: OS Independent",
]

name = "chromosight"

MAJOR = 0
MINOR = 4
MAINTENANCE = 0
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MAINTENANCE)

LICENSE = "GPLv3"

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().splitlines()

with open("chromosight/version.py", "w") as f:
    f.write("__version__ = '{}'\n".format(VERSION))

with codecs.open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=name,
    author="axel.cournac@pasteur.fr",
    long_description_content_type="text/markdown",
    description=__doc__,
    long_description=LONG_DESCRIPTION,
    version=VERSION,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    url="https://github.com/koszullab/chromosight",
    package_data={"chromosight": ("kernels/*",)},
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=REQUIREMENTS,
    include_package_data=True,
    entry_points={
        "console_scripts": ["chromosight=chromosight.cli.chromosight:main"]
    },
)
