
[![DOI](https://zenodo.org/badge/46057982.svg)](https://zenodo.org/badge/latestdoi/46057982)

## precession

`precession` is a Python module to study the dynamics of precessing black-hole binaries using multi-timescale methods. The code provides a comprehensive toolbox to (i) capture the black-hole dynamics on the spin-precession timescale in closed form, (ii) average generic quantities over a precession period, (iii) numerically integrate the binary inspiral using both orbit- and precession-averaged approximations, (v) evaluate spin-precession estimators to be used in gravitational-wave astronomy, and (vi) estimate the remnant properties. Key applications include propagating gravitational-wave posterior samples as well as population-synthesis predictions of astrophysical nature.

The current version (v2) of `precession` is described in 
- *Efficient multi-timescale dynamics of precessing black-hole binaries.*
D. Gerosa, G. Fumagalli, M. Mould, G. Cavallotto, D. Padilla Monroy, D. Gangardt, V. De Renzis.
[PRD 108 (2023) 024042](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.024042)
[arXiv:2304.04801](https://arxiv.org/abs/2304.04801)

The implementation of eccentricity is described in
- *Spin-eccentricity interplay in merging binary black holes.*
G. Fumagalli, D. Gerosa.
[PRD 108 (2023) 124055](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.124055).
[arXiv:2310.16893](https://arxiv.org/abs/2310.16893)

The previous implementation (v1) is described in
- *PRECESSION: Dynamics of spinning black-hole binaries with python.*
D. Gerosa, M. Kesden.
[PRD 93 (2016) 124066](http://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.124066).
[arXiv:1605.01067](https://arxiv.org/abs/1605.01067)

Note that v2 and v1 are *not* backward compatible; they are different codes. Unless you are maintaining a legacy pipeline, we highly reccommend using the new code. It is faster, more accurate, and provides more functionalities. 

`precession` is released under the MIT licence. You may use `precession` as you like but should acknowledge our work. When using the code in any published work, please cite the papers above. The codes have been used in a variety of studies in gravitational-wave and astronomy black-hole binary dynamics, follow the citations to those papers for more. 

The code is distributed under git version control at
- [github.com/dgerosa/precession](https://github.com/dgerosa/precession)

The documentation (v2) is available 
- [dgerosa.github.io/precession](https://dgerosa.github.io/precession)

The v1 documentation is archived at [this link](https://htmlpreview.github.io/?https://github.com/dgerosa/precession/blob/precession_v1/docs/index.html).

Installing the code is as easy as

    pip install precession

A short tutorial is provided in [the documentation](https://dgerosa.github.io/precession) together with a detailed list of all functions.

The main code deals with quasi-circular black-hole binaries and is accessible via `import precession`. The functionalities to study eccentric binaries are in the `eccentricity` submodule, which we reccommend importing as `import precession.eccentricity as precession`.   

The code is developed and maintained by [Davide Gerosa](www.davidegerosa.com). Please report bugs and suggestions using github.

#### Change log
- *v2.1.0* Extension to eccentric orbits. See [arXiv:2508.04801](https://arxiv.org/abs/2304.04801)
- *v2.0.0* New code, not backward compatible. See [arXiv:2304.04801](https://arxiv.org/abs/2304.04801)
- *v1.0.3* Python 3 support. Updated final-spin formula.
- *v1.0.0* First public release. See [arXiv:1605.01067](https://arxiv.org/abs/1605.01067).

(Third-level versions not explicitly indicated refer to patches for minor typos/bug fixes) 



