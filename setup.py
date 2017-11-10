#!/usr/bin/env python3

from setuptools import setup

import version

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "progressbar"
]

setup(
    name="electronics.py",
    version=version.VERSION,
    description=version.DESCRIPTION,
    long_description=readme,
    author="Sean Leavey",
    author_email="electronics@attackllama.com",
    url="https://github.com/SeanDS/electronics.py",
    packages=[
        "electronics"
    ],
    entry_points={
        'console_scripts': [
            '%s = electronics:main' % version.PROGRAM
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
