precession
==========

**Author** Davide Gerosa

**email** d.gerosa@damtp.cam.ac.uk

**Copyright** Copyright (C) 2016 Davide Gerosa

**Licence** CC BY 4.0

**Version** 0.9.0


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

- *Precession. Dynamics of spinning black-hole binaries with Python.* 
D. Gerosa, M. Kesden. [arXiv:1604.xxxxx](https://arxiv.org/abs/1604.xxxxx)

`precession` is an open-source code distributed under git version-control system on

- [github.com/dgerosa/precession](https://github.com/dgerosa/precession)

API documentation can be generated automatically in html format from the code
docstrings using `pdoc`, and is uplodad to a dedicated branch of the git
repository      

- [dgerosa.github.io/precession](https://dgerosa.github.io/precession)

Further information and scientific results on the results are available at:

- [www.damtp.cam.ac.uk/user/dg438/precession](http://www.damtp.cam.ac.uk/user/dg438/precession) 
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
scientific paper [arXiv:1604.xxxxx](https://arxiv.org/abs/1604.xxxxx), where
examples are also presented. 


### THINGS TO KEEP IN MIND

1. **Units**. All quantities in the code are specified in units where c=G=1.
Moreover, the binary total mass M=m1+m2 is  set to 1, and everything else is
computed accordingly. In practice, this means that e.g. the binary separation r
is actually r/M. If you are trying to use `precession` from a code with
different units (lal?), you should just pass r/M instead of your cgs or SI r.
Equivalently, the angular momentum L, the spins Si and the total angular
momentum J are actually L/M^2, Si/M^2 and J/M^2.

2. **Don't go too close to the limits (i)**. This is a code to study double-spin
precession and, in general, it won't behave nicely if you are too close to
aligned, hence non-precessing, configurations. These configurations are given by
the limits Jmin and Jmax of the total angular momentum J, and/or by
sin(theta_i)=0. I strongly recommend to always set a tolerance, for instance,

        Jmin,Jmax=precession.J_lim(q,S1,S2,r) for J in
        numpy.linspace(Jmin+1e-6,Jmax-1e-6,100): 
            do things...

3. **Don't go too close to the limits (ii)**. For the same reason, some
quantities cannot be computed efficiently for binaries which are very close to a
spin-orbit resonance. For instance, the computation of the angle alpha may be
inaccurate for binaries very close to xi_min and xi_max as returned by
xi_allowed.

4. **Checkpointing**. Checkpointing is implemented in some functions for
computational efficiency. Temporary data are stored in a local directory and
will be read in if available. To delete all previous data run

        precession.empty_temp()

    By default, data are stored in a local directory called
    `precession_checkpoints`, which is created when needed. You can change it
    setting

        precession.storedir=[path]

5. **Parallelization**. Some parts of the code are parallelized using the
`parmap` module. Instructions on code parallelization are set by the global
variable CPUs: (i) `CPUs=1` enforces a serial computation; (ii) `CPUs=integer`
specifies the number of parallel processes; (iii) `CPUs=0` (default) autodetects
the number of core in the current machine.

    You can set this variable using

        precession.CPUs = [integer]

6. **The equal-mass limit**. The equal-mass q=1 limit requires some extra care.
If q=1 the total-spin magnitude S cannot be used to parametrize the precession
cycle and the angle varphi needs to be tracked explicitly. The q=1 case is
implemented in the code: inputs and outputs of some of the functions are
actually specified as cos(varphi), even though for simplicity we still call them
**S**. In case of precession-averaged integrations to/from infinity, kappa_inf
becomes degenerate with xi and an initial value of S is required.
Please, refer to the documentation below for details. The generic unequal-mass
part of the code works fine up to q<0.995. To run higher values of q we
recommend setting q=1 and learn the relevant parts of the code.

7. **Stalling**. When performing precession-averaged evolutions, some binaries
may occasionally stall and take longer to run. This is due to the first step
attempted by the ODE integrator. This is a minor issue and  only happens to
roughly one binary in a million or so. If you really want to fix this, you
should play with the `h0` optional paramenter in scipy's odeint function.


### CREDITS
The code is developed and maintained by [Davide Gerosa](www.davidegerosa.com). 
Please, report bugs to

    d.gerosa@damtp.cam.ac.uk

I'm happy to help you out! 

**Thanks**: E. Berti, M. Kesden, U. Sperhake, R. O'Shaughnessy, D.
Trifiro', A. Klein, J. Vosmera and X. Zhao.
