from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

REQUIREMENTS = [
    "numpy",
    "scipy",
    "matplotlib",
    "progressbar2",
    "appdirs",
    "tabulate",
    "setuptools_scm",
    "ply"
]

# extra dependencies
EXTRAS = {
    "dev": [
        "pylint",
        "bandit",
        "sphinx",
        "sphinx_rtd_theme",
        "numpydoc",
        "nbsphinx",
        "graphviz"
    ]
}

setup(
    name="circuit",
    use_scm_version=True,
    description="Linear circuit simulator",
    long_description=README,
    author="Sean Leavey",
    author_email="sean.leavey@ligo.org",
    url="https://git.ligo.org/sean-leavey/circuit",
    packages=find_packages(),
    package_data={
        "circuit": ["circuit.conf.dist", "library.conf.dist"]
    },
    entry_points={
        'console_scripts': [
            '%s = circuit.__main__:main' % "circuit"
        ]
    },
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS,
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
