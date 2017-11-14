#!/usr/bin/env python3

from setuptools import setup

import electronics

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "scipy",
    "matplotlib",
    "progressbar",
    "appdirs",
    "tabulate",
    "sphinx-autodoc-typehints",
    "sphinx-bootstrap-theme"
]

setup(
    name="electronics.py",
    version=electronics.__version__,
    description=electronics.DESCRIPTION,
    long_description=readme,
    author="Sean Leavey",
    author_email="electronics@attackllama.com",
    url="https://github.com/SeanDS/electronics.py",
    packages=[
        "electronics"
    ],
    package_data={
        "electronics": ["electronics.conf.dist", "library.conf.dist"],
    },
    entry_points={
        'console_scripts': [
            '%s = electronics.__main__:main' % electronics.PROGRAM
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
