#!/usr/bin/env python

import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    sys.exit('Sorry, Python < 3.5 is not supported')

with open("README.md") as readme_file:
    readme = readme_file.read()

REQUIREMENTS = [
    "progressbar2",
    "setuptools_scm"
]

setup(
    name="datasheet",
    version="0.3.1",
    use_scm_version=True,
    description="Datasheet grabber",
    long_description=readme,
    author="Sean Leavey",
    author_email="sean.leavey@ligo.org",
    url="https://git.ligo.org/sean-leavey/datasheet",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            '%s = datasheet.__main__:main' % "datasheet"
        ]
    },
    install_requires=REQUIREMENTS,
    setup_requires=['setuptools_scm'],
    license="GPLv3",
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6"
    ]
)
