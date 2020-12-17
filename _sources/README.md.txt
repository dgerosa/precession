precession
==========

- - -
- - -

> **WARNING**: We are currently developing a new, awesome version of `precession`. You can check out our progess on the `master` and `dev` branches. `precession_v1`, as described in [arXiv:1605.01067](https://arxiv.org/abs/1605.01067) is still fully functional, and we reccommend you to use it for now. You can install it via pip (`pip install precession`, see below). The source code is available in a [dedicated branch](https://github.com/dgerosa/precession/tree/precession_v1) of this repository called `precession_v1`. The documentation can be browsed at [this link](https://htmlpreview.github.io/?https://github.com/dgerosa/precession/blob/precession_v1/docs/index.html). Sorry for the inconvenience, we hope v2 will be out soon! Stay tuned. 

- - -
- - -

# DYNAMICS OF SPINNING BLACK-HOLE BINARIES WITH PYTHON



`precession` is an open-source Python module to study the dynamics of precessing
black-hole binaries in the post-Newtonian regime.  The code provides a
comprehensive toolbox to (i) study the evolution of the black-hole spins  along
their precession cycles, (ii) perform gravitational-wave driven binary inspirals
using both orbit-averaged and precession-averaged integrations, and (iii)
predict the properties of the merger remnant through fitting formulae obtained
from numerical relativity simulations. `precession` is a ready-to-use tool to
add  the black-hole spin dynamics to larger-scale numerical studies such as
gravitational-wave parameter estimation codes, population synthesis models to
predict gravitational-wave event rates, galaxy merger trees and  cosmological
simulations of structure formation. `precession` provides fast and reliable
integration methods to propagate statistical samples of black-hole binaries
from/to large separations where they form to/from small separations where they
become detectable, thus linking gravitational-wave observations of spinning
black-hole binaries to their astrophysical formation history. The code is also a
useful tool to compute initial parameters for numerical relativity simulations
targeting specific precessing systems.

This code is released to the community under the [Creative Commons Attribution
International license](http://creativecommons.org/licenses/by/4.0).
Essentially, you may use `precession` as you like but must make reference to
our work. When using `precession` in any published work, please cite the paper
describing its implementation:

- *PRECESSION: Dynamics of spinning black-hole binaries with python.*
D. Gerosa, M. Kesden. PRD 93 (2016)
[124066](http://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.124066).
[arXiv:1605.01067](https://arxiv.org/abs/1605.01067)

`precession` is an open-source code distributed under git version-control system on

- [github.com/dgerosa/precession](https://github.com/dgerosa/precession)

API documentation can be generated automatically in html format from the code
docstrings using `pdoc`, and is uplodad to a dedicated branch of the git
repository

- [dgerosa.github.io/precession](https://dgerosa.github.io/precession)

Further information and scientific results are available at:

- [www.tapir.caltech.edu/~dgerosa/precession](http://www.tapir.caltech.edu/~dgerosa/precession)
- [www.davidegerosa.com/precession](http://www.davidegerosa.com/precession)


### INSTALLATION

`precession` works in python 2.x and has been tested on 2.7.10. It can be
installed through [pip](https://pypi.python.org/pypi/precession):

    pip install precession

Prerequisites are `numpy`, `scipy` and `parmap`, which can be all installed
through pip. Information on all code functions are available through Pyhton's
built-in help system

    import precession
    help(precession.function)

Several tests and tutorial are available in the submodule `precession.test`. A
detailed description of the functionalies of the code is provided in the
scientific paper [arXiv:1605.01067](https://arxiv.org/abs/1605.01067), where
examples are also presented.


### RESULTS

`precession` has been used in the following published papers:

- Gerosa and Sesana. MNRAS 446 (2015) 38-55. [arXiv:1405.2072](https://arxiv.org/abs/1405.2072)
- Kesden et al. PRL 114 (2015) 081103. [arXiv:1411.0674](https://arxiv.org/abs/1411.0674)
- Gerosa et al. MNRAS 451 (2015) 3941-3954. [arXiv:1503.06807](https://arxiv.org/abs/1503.06807)
- Gerosa et al. PRD 92 (2015) 064016. [arXiv:1506.03492](https://arxiv.org/abs/1506.03492)
- Gerosa et al. PRL 115 (2015) 141102. [arXiv:1506.09116](https://arxiv.org/abs/1506.09116)
- Trifiro' et al. PRD 93 (2016) 044071. [arXiv:1507.05587](https://arxiv.org/abs/1507.05587)
- Gerosa and Kesden. PRD 93 (2016) 124066. [arXiv:1605.01067](https://arxiv.org/abs/1605.01067)
- Gerosa and Moore. PRL 117 (2016) 011101. [arXiv:1606.04226](https://arxiv.org/abs/1606.04226)
- Rodriguez et al. APJL 832 (2016) L2 [arXiv:1609.05916](https://arxiv.org/abs/1609.05916)
- Gerosa et al. CQG 34 (2017) 6, 064004 [arXiv:1612.05263](https://arxiv.org/abs/1612.05263)
- Gerosa and Berti. PRD 95 (2017) 124046. [arXiv:1703.06223](https://arxiv.org/abs/1703.06223)
- Zhao et al. PRD 96 (2017) 024007. [arXiv:1705.02369](https://arxiv.org/abs/1705.02369)
- Wysocki et al. PRD 97 (2018) 043014 [arXiv:1709.01943](https://arxiv.org/abs/1709.01943)
- Gerosa J.Phys.Conf.Ser. 957 (2018) 012014. [arXiv:1711.1003](https://arxiv.org/abs/1711.1003)
- Rodriguez et al. PRL 120 (2018) 151101. [arXiv:1712.0493](https://arxiv.org/abs/1712.0493)
- Gerosa et al. PRD 97 (2018) 104049. [arXiv:1802.04276](https://arxiv.org/abs/1802.04276)
- Gerosa et al. PRD 98 (2018) 084036. [arXiv:1808.02491](https://arxiv.org/abs/1808.02491)
- Varma et al. PRL 122 (2019) 011101. [arXiv:1809.09125](https://arxiv.org/abs/1809.09125)
- Gerosa et al. PRD 99 (2019) 103004. [arXiv:1902.00021](https://arxiv.org/abs/1902.00021)
- Gerosa et al. CQG 36 (2019) 105003. [arXiv:1811.05979](https://arxiv.org/abs/1811.05979)
- Tso et al. PRD 99 (2019) 124043 [arXiv:1807.00075](https://arxiv.org/abs/1807.00075)
- Gerosa and Berti. PRD 100 (2019) 041301. [arXiv:1906.05295](https://arxiv.org/abs/1906.05295)
- Varma et al. PRR 1 (2019) 033015. [arXiv:1905.09300](https://arxiv.org/abs/1905.09300)
- Wong and Gerosa. PRD 100 (2019) 083015. [arXiv:1909.06373](https://arxiv.org/abs/1909.06373)
- Phukon et al. PRD 100 (2019) 124008. [arXiv:1904.03985](https://arxiv.org/abs/1904.03985)
- Varma et al. [arXiv:2002.00296](https://arxiv.org/abs/2002.00296)
- Mould and Gerosa [arXiv:2003.02281](https://arxiv.org/abs/2003.02281)


### RELEASES

[![DOI](https://zenodo.org/badge/46057982.svg)](https://zenodo.org/badge/latestdoi/46057982)

*v1.0.0* Stable version released together with the first arxiv submission of [arXiv:1605.01067](https://arxiv.org/abs/1605.01067).

*v1.0.2* Clarifications on typos in Eq. (36) and (37) of [arXiv:1605.01067](https://arxiv.org/abs/1605.01067). See help(precession) for more information.

*v1.0.3* Python 3 now supported (hurray!). By default, `finalspin ` now returns more updated result by Hofmann, Barausse and Rezzolla 2016.



### CREDITS
The code is developed and maintained by [Davide Gerosa](www.davidegerosa.com).
Please, report bugs to

    dgerosa@caltech.edu

I am happy to help you out!

**Thanks**: M. Kesden, U. Sperhake, E. Berti, R. O'Shaughnessy, A. Sesana, D.
Trifiro', A. Klein, J. Vosmera and X. Zhao.

