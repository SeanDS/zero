from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

REQUIREMENTS = [
    "numpy >= 1.15",
    "scipy >= 1.1",
    "matplotlib >= 3.0",
    "requests >= 2.19",
    "progressbar2 >= 3.38",
    "tabulate >= 0.8",
    "setuptools_scm >= 3.1",
    "ply >= 3.11",
    "click == 7.0",
    "PyYAML >= 3.13"
]

# Extra dependencies.
EXTRAS = {
    "dev": [
        "pylint",
        "bandit",
        "sphinx",
        "sphinx-click",
        "sphinx_rtd_theme",
        "numpydoc",
        "nbsphinx",
        "graphviz == 0.9"
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
