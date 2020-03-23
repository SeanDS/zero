from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

REQUIREMENTS = [
    "numpy >= 1.15.2",
    "scipy >= 1.1.0",
    "matplotlib >= 3.0.3",
    "requests >= 2.19.1",
    "progressbar2 >= 3.38.0",
    "tabulate >= 0.8.2",
    "setuptools_scm >= 3.1.0",
    "ply >= 3.11",
    "Click == 7.0",
    "quantiphy >= 2.5.0",
    "PyYAML >= 3.13",
    "graphviz >= 0.9",
]

# Extra dependencies.
EXTRAS = {
    "dev": [
        "pylint",
        "bandit",
        "sphinx",
        "sphinx-autobuild",
        "sphinx-click",
        "sphinx_rtd_theme",
        "sphinxcontrib-programoutput",
        "doc8",
        "numpydoc",
        "nbsphinx",
    ]
}

setup(
    name="zero",
    use_scm_version={
        "write_to": "zero/_version.py"
    },
    description="Linear circuit simulator",
    long_description=README,
    author="Sean Leavey",
    author_email="sean.leavey@ligo.org",
    url="https://git.ligo.org/sean-leavey/zero",
    packages=find_packages(),
    package_data={
        "zero.config": ["zero.yaml.dist", "zero.yaml.dist.default",
                        "components.yaml.dist", "components.yaml.dist.default"]
    },
    entry_points={
        "console_scripts": [
            "%s = zero.__main__:cli" % "zero"
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ]
)
