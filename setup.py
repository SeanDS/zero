#!/usr/bin/env python3

from setuptools import setup

import circuit

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "scipy",
    "matplotlib",
    "progressbar",
    "appdirs",
    "tabulate",
    "sphinx-bootstrap-theme"
]

setup(
    name="circuit.py",
    version=circuit.__version__,
    description=circuit.DESCRIPTION,
    long_description=readme,
    author="Sean Leavey",
    author_email="sean.leavey@ligo.org",
    url="https://git.ligo.org/sean-leavey/circuit",
    packages=[
        "circuit"
    ],
    package_data={
        "circuit": ["circuit.conf.dist", "library.conf.dist"]
    },
    entry_points={
        'console_scripts': [
            '%s = circuit.__main__:main' % circuit.PROGRAM
        ]
    },
    install_requires=requirements,
    license="GPLv3",
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5"
    ]
)
