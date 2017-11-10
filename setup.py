#!/usr/bin/env python3

from setuptools import setup

from electronics import __version__, PROG, DESC

with open("README.md") as readme_file:
    readme = readme_file.read()

__version__ = electronics.__version__

requirements = [
    "progressbar"
]

setup(
    name="electronics.py",
    version=__version__,
    description=DESC,
    long_description=readme,
    author="Sean Leavey",
    author_email="electronics@attackllama.com",
    url="https://github.com/SeanDS/electronics.py",
    packages=[
        "electronics"
    ],
    entry_points={
        'console_scripts': [
            '%s = electronics:main' % PROG
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
