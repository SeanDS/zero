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

Merge requests
~~~~~~~~~~~~~~

Please open a `merge request`_ on GitLab, targeting |Zero|'s `develop` branch. To keep the git
repository's merge graph clean, ideally you should make your changes on a branch with one of the
following conventions depending on what kind of change you make:

- ``feature/my-feature`` for new features
- ``fix/my-fix`` for bug fixes

Replace ``my-feature`` or ``my-fix`` with an appropriate short description. This naming scheme
roughly follows that presented in `A successful Git branching model`_.

Creating new releases
~~~~~~~~~~~~~~~~~~~~~

The steps below should be followed when creating a new release:

#. Ensure all tests pass and all bundled examples work as intended, and all documentation is
   up-to-date.
#. Create a new release branch from ``develop``, where ``x.x.x`` is the intended new version number:
   ``git checkout -b release/x.x.x develop``.
#. Update default user component library ``distributed_with`` key to match the new intended version
   number.
#. Commit changes and checkout ``develop``.
#. Checkout ``develop`` branch then merge release without fast-forwarding:
   ``git merge --no-ff release/x.x.x``.
#. Checkout ``master`` branch then merge ``release/x.x.x`` without fast-forwarding:
   ``git merge --no-ff release/x.x.x``.
#. Tag the release on ``master`` with the version: ``git tag -a x.x.x``.
#. Delete the release branch: ``git branch -d release/x.x.x``.
#. Push all changes to ``master`` and ``develop`` and the new tag to origin.

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
