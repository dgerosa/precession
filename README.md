precession 
==========

**Author** Davide Gerosa

**email** d.gerosa@damtp.cam.ac.uk

**Copyright** Copyright (C) 2015 Davide Gerosa

**Licence** CC by-nc-sa 3.0

**Version** 0.0.0.36


# DYNAMICS OF PRECESSING BLACK-HOLE BINARIES

Detailed documentation and results from the code are available at:

- [www.damtp.cam.ac.uk/user/dg438/spinprecession](www.damtp.cam.ac.uk/user/dg438/spinprecession) 
- [www.davidegerosa.com/spinprecession](www.davidegerosa.com/spinprecession)

This code is released to the community under the [Creative Commons Attribution
4.0 International license](http://creativecommons.org/licenses/by/4.0).
Essentially, you may use `precession` as you like but must make reference to
our work. When using precession in any published work, please cite the paper
describing its implementation:

ADD REFERENCE TO CODE PAPER HERE!

`precession` is an open-source code distributed under git version-control system on

[github.com/dgerosa/precession](github.com/dgerosa/precessions)

API documentation can be generated automatically in html format from the code docstrings using pdoc, and is is uplodad in a dedicated branch of the git repository

[dgerosa.github.io/precession](dgerosa.github.io/precession)


### INSTALLATION
 
`precession` works in python 2.x and has been tested on 2.7.10. It can be
installed through pip:

    pip install precession

Prerequisites are `numpy`, `scipy` and `parmap`, which can be all installed
through pip.


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
        numpy.linspace(Jmin+1e-4,Jmax-1e-4,100): 
            do things...

3. **Don't go too close to the limits (ii)**. For the same reason, some
quantities cannot be computed efficiently for binaries which are very close to a
spin-orbit resonance (which indeed does not precess at all!). For instance,  the
computation of the angle alpha is somewhat unstable close to xi_min and xi_max
as returned by xi_allowed. Richard O'Shaughnessy found that a tolerance of 2e-3
on xi works well.


4. **Checkpointing**. Checkpointing is implemented in some functions for
computational efficiency. Temporary data are stored in a local directory and
will be read in if available. To delete all previous data run

        precession.empty_temp()

    By default, data are stored in a local directory called `checkpoints`, which
    is created when needed. You can change it setting

        precession.storedir=[path]

5. **Parallelization**. Some parts of the code are parallelized using the
`parmap` module. Instructions on code parallelization are set by the global
variable CPUs - `CPUs=1`: no parallelization will be used; - `CPUs=integer`: to
specify the actual number of cores to be used; - `CPUs=0` (default): all CPUs in
the current machine will be used.

    You can set this variable using

        precession.CPUs = [integer]

6. **The equal-mass limit**. The equal-mass q=1 limit requires some extra care.
If q=1 the total-spin magnitude S cannot be used to parametrize the precession
cycle and the angle varphi needs to be tracked explicitly. The q=1 case is
implemented in the code: inputs and outputs of some of the functions are
actually specified in cos(varphi), even if for simplicity we still call them
**S**. In case of precession-averaged integrations to/from infinity, kappa_inf
becomes degenerate with xi and a required initial value of S is required.
Please, refer to the documentation below for details. The generic unequal-mass
part of the code works fine up to q<0.9999. To run higher values of q we
recommend setting q=1.

7. **Stalling**. When performing precession-averaged evolutions, some binaires
may occasionally stall and take longer to run. This is due to the first step
attempted by the ODE integrator. This is a minor issue and  only happens to
roughly one binary in a million or so. If you really want to fix this, you
should play with the h0 optional paramenter in scipy's odeint function.


### THANKS
The code is developed and maintained by [Davide Gerosa](www.davidegerosa.com). 
Please, report bugs to

    d.gerosa@damtp.cam.ac.uk

I'm happy to help you out! 

I'd like to thank E. Berti, M. Kesden, U. Sperhake, R. O'Shaughnessy, D.
Trifiro' and J. Vosmera for the help received in interpreting the physical
results and implementing some of the algorithms.
