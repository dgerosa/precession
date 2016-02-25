precession
==========

**Author** Davide Gerosa

**email** d.gerosa@damtp.cam.ac.uk

**Copyright** Copyright (C) 2016 Davide Gerosa

**Licence** CC BY 4.0

**Version** 0.0.0.36


# DYNAMICS OF SPINNING BLACK-HOLE BINARIES WITH PYTHON

`precession` is a Python module to study the dynamics of precessing black-hole
binaries in the post-Newtonian regime. The code includes tools to study the
precessional dynamics, integrators to perform orbit-averaged and
precession-averaged post-Newtonian inspirals, and implementation of the fitting
formulae to predict the properties of the black-hole remnant.

This code is released to the community under the [Creative Commons Attribution
International license](http://creativecommons.org/licenses/by/4.0).
Essentially, you may use `precession` as you like but must make reference to
our work. When using precession in any published work, please cite the paper
describing its implementation:

- *Precession. Dynamics of spinning black-hole binaries with Python.* 
Davide Gerosa. Submitted to... arXiv:...

`precession` is an open-source code distributed under git version-control system on

- [github.com/dgerosa/precession](github.com/dgerosa/precessions)

API documentation can be generated automatically in html format from the code docstrings using pdoc, and is is uplodad in a dedicated branch of the git repository

- [dgerosa.github.io/precession](dgerosa.github.io/precession)

Further information and scientific results on the results are available at:

- [www.damtp.cam.ac.uk/user/dg438/spinprecession](www.damtp.cam.ac.uk/user/dg438/spinprecession) 
- [www.davidegerosa.com/spinprecession](www.davidegerosa.com/spinprecession)


### INSTALLATION
 
`precession` works in python 2.x and has been tested on 2.7.10. It can be
installed through pip:

    pip install precession

Prerequisites are `numpy`, `scipy` and `parmap`, which can be all installed
through pip. Information on all code functions are available through Pyhton's
built-in help system

    import precession
    help(precession.function)

Several tests and tutorial are available in the submodule `precession.test`.


### THANKS
The code is developed and maintained by [Davide Gerosa](www.davidegerosa.com). 
Please, report bugs to

    d.gerosa@damtp.cam.ac.uk

I'm happy to help you out! 

I'd like to thank E. Berti, M. Kesden, U. Sperhake, R. O'Shaughnessy, D.
Trifiro' and J. Vosmera for the help received in interpreting the physical
results and implementing some of the algorithms.
