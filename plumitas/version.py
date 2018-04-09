from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 1  # use '' for first of series, number for 1 and above
_version_extra = 'dev3'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3.6",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "plumitas: Pythonic analysis and visualization of Plumed output files"
# Long description will go up on the pypi page
long_description = """

Plumitas
========
Pythonic analysis and visualization of Plumed output files. For more
info, please go to the repository README_.

.. _README: https://github.com/UWPRG/plumitas/blob/master/README.md

License
=======
``plumitas`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2018--, Josh Smith, The University of Washington
Department of Chemical Engineering.
"""

NAME = "plumitas"
MAINTAINER = "Josh Smith"
MAINTAINER_EMAIL = "jsmith07@uw.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/UWPRG/plumitas"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Pfaendtner Research Group"
AUTHOR_EMAIL = "jsmith07@uw.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'plumitas': [pjoin('data', '*')]}
REQUIRES = ["numpy"]
