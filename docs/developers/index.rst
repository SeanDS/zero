.. include:: /defs.txt

Information for developers
==========================

Submission of small bug fixes and features is encouraged. For larger features, please contact the
author to discuss feasibility and structure.

Code style
~~~~~~~~~~

Follow `PEP 8`_ where possible.

Documentation style
~~~~~~~~~~~~~~~~~~~

Use `NumPy docstring format`_. Language and grammar should follow `Google style`_.

Development environment
~~~~~~~~~~~~~~~~~~~~~~~

A Visual Studio Code configuration file is provided in the project root when checked out via
``git``, which sets some code format settings which should be followed. This configuration file is
used automatically if the project is opened in Visual Studio Code from its root directory.

It may be useful to run |Zero| within a ``conda`` or ``pipenv`` environment to allow for separation
of dependencies from your system and from other projects. In both cases it is still recommended to
install |Zero| via ``pip``. For rapid development, it is highly recommended to make the project
`editable` so changes to project files reflect immediately in the library and CLI, and to install
the extra `dev` dependencies to allow you to build the documentation and run code linting tools:

.. code-block:: bash

   pip install -e .[dev]

Merge requests
~~~~~~~~~~~~~~

If you have code to submit for inclusion in |Zero|, please open a `merge request`_ on GitLab
targeting the ``develop`` branch. To keep the git repository's merge graph clean, ideally you should
make your changes on a branch with one of the following conventions depending on what kind of change
you make:

- ``feature/my-feature`` for new features
- ``hotfix/my-fix`` for bug fixes

Replace ``my-feature`` or ``my-fix`` with an appropriate short description. This naming scheme
roughly follows that presented in `A successful Git branching model`_.

Creating new releases
~~~~~~~~~~~~~~~~~~~~~

The steps below should be followed when creating a new release:

#. Ensure all tests pass and all bundled examples work as intended, and all documentation is
   up-to-date.
#. Create a new release branch from ``develop``, where ``x.x.x`` is the intended new version number:
   ``git checkout -b release/x.x.x develop``.
#. Update default user config and component library ``distributed_with`` keys to match the new
   intended version number.
#. Commit changes and checkout ``develop``.
#. Checkout ``develop`` branch then merge release without fast-forwarding:
   ``git merge --no-ff release/x.x.x``.
#. Checkout ``master`` branch then merge ``release/x.x.x`` without fast-forwarding:
   ``git merge --no-ff release/x.x.x``.
#. Tag the release on ``master`` with the version: ``git tag -a x.x.x``.
#. Delete the release branch: ``git branch -d release/x.x.x``.
#. Push all changes to ``master`` and ``develop`` and the new tag to origin.

Note that when a new tag is pushed to the `ligo.org` GitLab server, the CI runner automatically
creates and uploads a new PyPI release.

Updating PyPI (pip) package
---------------------------

This requires `twine <https://packaging.python.org/key_projects/#twine>`__ and the credentials for
the |Zero| PyPI project.

By default, the GitLab CI runner will deploy a PyPI package automatically whenever a new tag is
created. The instructions below are for when this must be done manually:

#. Go to the source root directory.
#. Checkout the ``master`` branch (so the release uses the correct tag).
#. Remove previously generated distribution files:
   ``rm -rf build dist``

#. Create new distribution files:
   ``python setup.py sdist bdist_wheel``

#. (Optional) Upload distribution files to PyPI test server, entering the required credentials when
   prompted:
   ``python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*``

   You can then check the package is uploaded properly by viewing the `Zero project on the PyPI test server`_.
   You can also check that it installs correctly with:
   ``pip install --index-url https://test.pypi.org/simple/ --no-deps zero``

   Note: even if everything installs correctly, the test package will not work correctly due to lack
   of dependencies (forced by the ``--no-deps`` flag, since they are not all available on the PyPI
   test server).
#. Upload distribution files to PyPI, entering the required credentials when prompted:
   ``python -m twine upload dist/*``

#. Verify everything is up-to-date on `PyPI <https://pypi.org/project/zero/>`__.

API documentation
~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    api/modules

.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _NumPy docstring format: https://numpydoc.readthedocs.io/en/latest/example.html
.. _Google style: https://developers.google.com/style/
.. _merge request: https://git.ligo.org/sean-leavey/zero/merge_requests
.. _A successful Git branching model: https://nvie.com/posts/a-successful-git-branching-model/
.. _Zero project on the PyPI test server: https://test.pypi.org/project/zero/
