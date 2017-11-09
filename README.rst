precession
==========

**Author** Davide Gerosa

**email** dgerosa@caltech.edu

**Copyright** Copyright (C) 2016 Davide Gerosa

**Licence** CC BY 4.0

**Version** 1.0.2

DYNAMICS OF SPINNING BLACK-HOLE BINARIES WITH PYTHON
====================================================

``precession`` is an open-source Python module to study the dynamics of
precessing black-hole binaries in the post-Newtonian regime. The code
provides a comprehensive toolbox to (i) study the evolution of the
black-hole spins along their precession cycles, (ii) perform
gravitational-wave driven binary inspirals using both orbit-averaged and
precession-averaged integrations, and (iii) predict the properties of
the merger remnant through fitting formulae obtained from numerical
relativity simulations. ``precession`` is a ready-to-use tool to add the
black-hole spin dynamics to larger-scale numerical studies such as
gravitational-wave parameter estimation codes, population synthesis
models to predict gravitational-wave event rates, galaxy merger trees
and cosmological simulations of structure formation. ``precession``
provides fast and reliable integration methods to propagate statistical
samples of black-hole binaries from/to large separations where they form
to/from small separations where they become detectable, thus linking
gravitational-wave observations of spinning black-hole binaries to their
astrophysical formation history. The code is also a useful tool to
compute initial parameters for numerical relativity simulations
targeting specific precessing systems.

This code is released to the community under the `Creative Commons
Attribution International
license <http://creativecommons.org/licenses/by/4.0>`__. Essentially,
you may use ``precession`` as you like but must make reference to our
work. When using ``precession`` in any published work, please cite the
paper describing its implementation:

-  *PRECESSION: Dynamics of spinning black-hole binaries with python.*
   D. Gerosa, M. Kesden. PRD 93 (2016)
   `124066 <http://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.124066>`__.
   `arXiv:1605.01067 <https://arxiv.org/abs/1605.01067>`__

``precession`` is an open-source code distributed under git
version-control system on

-  `github.com/dgerosa/precession <https://github.com/dgerosa/precession>`__

API documentation can be generated automatically in html format from the
code docstrings using ``pdoc``, and is uplodad to a dedicated branch of
the git repository

-  `dgerosa.github.io/precession <https://dgerosa.github.io/precession>`__

Further information and scientific results are available at:

-  `www.tapir.caltech.edu/~dgerosa/precession <http://www.tapir.caltech.edu/~dgerosa/precession>`__
-  `www.davidegerosa.com/precession <http://www.davidegerosa.com/precession>`__

INSTALLATION
------------

``precession`` works in python 2.x and has been tested on 2.7.10. It can
be installed through `pip <https://pypi.python.org/pypi/precession>`__:

::

    pip install precession

Prerequisites are ``numpy``, ``scipy`` and ``parmap``, which can be all
installed through pip. Information on all code functions are available
through Pyhton's built-in help system

::

    import precession
    help(precession.function)

Several tests and tutorial are available in the submodule
``precession.test``. A detailed description of the functionalies of the
code is provided in the scientific paper
`arXiv:1605.01067 <https://arxiv.org/abs/1605.01067>`__, where examples
are also presented.

RESULTS
-------

``precession`` has been used in the following published papers:

-  Gerosa and Sesana. MNRAS 446 (2015) 38-55.
   `arXiv:1405.2072 <https://arxiv.org/abs/1405.2072>`__
-  Kesden et al. PRL 114 (2015) 081103.
   `arXiv:1411.0674 <https://arxiv.org/abs/1411.0674>`__
-  Gerosa et al. MNRAS 451 (2015) 3941-3954.
   `arXiv:1503.06807 <https://arxiv.org/abs/1503.06807>`__
-  Gerosa et al. PRD 92 (2015) 064016.
   `arXiv:1506.03492 <https://arxiv.org/abs/1506.03492>`__
-  Gerosa et al. PRL 115 (2015) 141102.
   `arXiv:1506.09116 <https://arxiv.org/abs/1506.09116>`__
-  Trifiro' et al. PRD 93 (2016) 044071.
   `arXiv:1507.05587 <https://arxiv.org/abs/1507.05587>`__
-  Gerosa and Kesden. PRD 93 (2016) 124066.
   `arXiv:1605.01067 <https://arxiv.org/abs/1605.01067>`__
-  Gerosa and Moore. PRL 117 (2016) 011101.
   `arXiv:1606.04226 <https://arxiv.org/abs/1606.04226>`__
-  Rodriguez et al. APJL 832 (2016) L2
   `arXiv:1609.05916 <https://arxiv.org/abs/1609.05916>`__
-  Gerosa et al. CQG 34 (2017) 6, 064004
   `arXiv:1612.05263 <https://arxiv.org/abs/1612.05263>`__
-  Gerosa and Berti. PRD 95 (2017) 124046.
   `arXiv:1703.06223 <https://arxiv.org/abs/1703.06223>`__
-  Zhao et al. PRD 96 (2017) 024007.
   `arXiv:1705.02369 <https://arxiv.org/abs/1705.02369>`__
-  Wysocki et al.
   `arXiv:1709.01943 <https://arxiv.org/abs/1709.01943>`__

RELEASES
--------

|DOI| v1.0.0 (stable)

CREDITS
-------

The code is developed and maintained by `Davide
Gerosa <www.davidegerosa.com>`__. Please, report bugs to

::

    dgerosa@caltech.edu

I am happy to help you out!

**Thanks**: M. Kesden, U. Sperhake, E. Berti, R. O'Shaughnessy, A.
Sesana, D. Trifiro', A. Klein, J. Vosmera and X. Zhao.

.. |DOI| image:: https://zenodo.org/badge/21015/dgerosa/precession.svg
   :target: https://zenodo.org/badge/latestdoi/21015/dgerosa/precession
