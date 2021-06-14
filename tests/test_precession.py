
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')


import pytest
import numpy as np
import precession
from functools import wraps

#
# def thatsilly():
#     """
#     Test the test procedure.
#     """
#
#     assert True

#
#
# class generictest:
#     """
#     Parameters
#     ----------
#     func : function
#         The function to test. It should output a single value.
#     args : dict
#         A dictionary of arguments for func. The keys are the argument names.
#         Must be ordered as in the definition of func.
#     output_to_compare :
#         The value to compare the output to.
#     args_to_repeat : list of str (optional)
#         Which arguments of func to repeat to get vector output.
#     """
#
#     def __init__(self, func, args, output_to_compare, args_to_repeat='all', multiple=2):
#
#         self.func = func
#         self.args = args
#         self.multiple = multiple
#         self.output_to_compare = np.array(output_to_compare)
#         if args_to_repeat == 'all':
#             self.args_to_repeat = list(self.args.keys())
#         elif args_to_repeat is None:
#             self.args_to_repeat = []
#
#     def test_single(self):
#         return np.allclose(self.func(**self.args), self.output_to_compare)
#
#     def test_multiple(self):
#
#         _args = self.args.copy()
#         for arg in self.args_to_repeat:
#             _args[arg] = np.repeat(_args[arg], self.multiple)
#
#         _output_to_compare = np.reshape(
#                                 np.repeat(self.output_to_compare, self.multiple, axis=0),
#                                         (self.output_to_compare.shape[0],self.multiple))
#
#         return np.allclose(self.func(**_args), _output_to_compare)
#
#     def __call__(self):
#
#         return self.test_single() and self.test_multiple()


def both(testfunction,multiple=3):
    """
    testfunction is the function that performes the test and it's called "test_codefunction" where codefunction is a function of the precession code.
    """

    @wraps(testfunction)
    def wrapper():
        input, output = testfunction()

        # Extract the codefunction
        codefunction = eval("precession."+testfunction.__name__.split("test_")[1:][0])
        # Make sure output is ready for reshaping
        output = np.array(output)
        # Inflate the inputs
        _input = input.copy()
        for arg in _input:
            _input[arg] = np.repeat(_input[arg], multiple)
        # Inflate the outputs
        _output = np.reshape( np.repeat(output, multiple, axis=0), (output.shape[0],multiple) )

        # Test on a single entry
        checksingle = np.allclose(codefunction(**input), output)

        # Test on multiple entries
        checkmultiple = np.allclose(codefunction(**_input), _output)

        # Actual test for pytest
        assert checksingle
        assert checkmultiple

    return wrapper


@both
def test_eval_m1():
    return {"q":0.8}, [0.55555556]

@both
def test_eval_m2():
    return {"q":0.8}, [0.44444444]

@both
def test_masses():
    return {"q":0.8}, [[0.55555556],[0.44444444]]

@both
def test_eval_q():
    return {"m1":36, "m2":29}, [0.80555556]

@both
def test_eval_eta():
    return {"q":1}, [0.25]

@both
def test_eval_S1():
    return {"q":0.8, "chi1":1}, [0.30864198]

@both
def test_eval_S2():
    return {"q":0.8, "chi2":1}, [0.19753086]

@both
def test_spinmags():
    return {"q":0.8, "chi1":1, "chi2":1}, [[0.30864198],[0.19753086]]

@both
def test_eval_L():
    return {"q":0.8, "r":10}, [0.7808093]

@both
def test_eval_v():
    return {"r":10}, [0.31622777]


### There needs to be tests for all these functions, multiple ones for some functions.
# eval_L(r, q)
# eval_v(r)
# eval_r(L=None, u=None, q=None)
# Jlimits_LS1S2(r, q, chi1, chi2)
# kappadiscriminant_coefficients(u, chieff, q, chi1, chi2)
# kapparesonances(u, chieff, q, chi1, chi2)
# kappainfresonances(chieff, q, chi1, chi2)
# Jresonances(r, chieff, q, chi1, chi2)
# Jlimits(r=None, chieff=None, q=None, chi1=None, chi2=None, enforce=False)
# kappainflimits(chieff=None, q=None, chi1=None, chi2=None, enforce=False)
# chiefflimits_definition(q, chi1, chi2)
# chieffdiscriminant_coefficients(kappa, u, q, chi1, chi2)
# chieffresonances(J, r, q, chi1, chi2)
# anglesresonances(J=None, r=None, chieff=None, q=None, chi1=None, chi2=None)
# chiefflimits(J=None, r=None, q=None, chi1=None, chi2=None, enforce=False)
# Slimits_S1S2(q, chi1, chi2)
# Slimits_LJ(J, r, q)
# Slimits_LJS1S2(J, r, q, chi1, chi2)
# Scubic_coefficients(kappa, u, chieff, q, chi1, chi2)
# Ssroots(J, r, chieff, q, chi1, chi2, precomputedroots=None)
# Slimits_plusminus(J, r, chieff, q, chi1, chi2)
# Satresonance(J=None, kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None)
# Slimits(J=None, r=None, chieff=None, q=None, chi1=None, chi2=None, enforce=False)
# limits_check(S=None, J=None, r=None, chieff=None, q=None, chi1=None, chi2=None)
# eval_chieff(theta1=None, theta2=None, S=None, varphi=None, J=None, r=None, q=None, chi1=None, chi2=None)
# effectivepotential_plus(S, J, r, q, chi1, chi2)
# effectivepotential_minus(S, J, r, q, chi1, chi2)
# eval_varphi(S, J, r, chieff, q, chi1, chi2, cyclesign=-1)
# eval_costheta1(S, J, r, chieff, q, chi1, chi2)
# eval_theta1(S, J, r, chieff, q, chi1, chi2)
# eval_costheta2(S, J, r, chieff, q, chi1, chi2)
# eval_theta2(S, J, r, chieff, q, chi1, chi2)
# eval_costheta12(theta1=None, theta2=None, deltaphi=None, S=None, q=None, chi1=None, chi2=None)
# eval_theta12(theta1=None, theta2=None, deltaphi=None, S=None, q=None, chi1=None, chi2=None)
# eval_cosdeltaphi(S, J, r, chieff, q, chi1, chi2)
# eval_deltaphi(S, J, r, chieff, q, chi1, chi2, cyclesign=-1)
# eval_costhetaL(S, J, r, q, chi1, chi2)
# eval_thetaL(S, J, r, q, chi1, chi2)
# eval_J(theta1=None, theta2=None, deltaphi=None, kappa=None, r=None, q=None, chi1=None, chi2=None)
# eval_S(theta1, theta2, deltaphi, q, chi1, chi2)
# eval_kappa(J, r, q)
# eval_u(r, q)
# eval_kappainf(theta1inf, theta2inf, q, chi1, chi2)
# eval_costheta1inf(kappainf, chieff, q, chi1, chi2)
# eval_theta1inf(kappainf, chieff, q, chi1, chi2)
# eval_costheta2inf(kappainf, chieff, q, chi1, chi2)
# eval_theta2inf(kappainf, chieff, q, chi1, chi2)
# morphology(J, r, chieff, q, chi1, chi2, simpler=False)
# eval_cyclesign(dSdt=None, deltaphi=None, varphi=None, Lvec=None, S1vec=None, S2vec=None)
# conserved_to_angles(S, J, r, chieff, q, chi1, chi2, cyclesign=+1)
# angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, full_output=False)
# angles_to_asymptotic(theta1inf, theta2inf, q, chi1, chi2)
# asymptotic_to_angles(kappainf, chieff, q, chi1, chi2)
# vectors_to_conserved(Lvec, S1vec, S2vec, q, full_output=False)
# vectors_to_angles(Lvec, S1vec, S2vec)
# conserved_to_Jframe(S, J, r, chieff, q, chi1, chi2, cyclesign=1)
# angles_to_Jframe(theta1, theta2, deltaphi, r, q, chi1, chi2)
# angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2)
# conserved_to_Lframe(S, J, r, chieff, q, chi1, chi2, cyclesign=1)
# conserved_to_inertial(S, J, r, chieff, q, chi1, chi2, cyclesign=1)
# angles_to_inertial(theta1, theta2, deltaphi, r, q, chi1, chi2)
# derS_prefactor(r, chieff, q)
# dSsdtsquared(S, J, r, chieff, q, chi1, chi2)
# dSsdt(S, J, r, chieff, q, chi1, chi2, cyclesign=1)
# dSdt(S, J, r, chieff, q, chi1, chi2)
# elliptic_parameter(Sminuss, Spluss, S3s)
# elliptic_amplitude(S, Sminuss, Spluss)
# elliptic_characheristic(Sminuss, Spluss, J, L, sign)
# time_normalization(Spluss, S3s, r, chieff, q)
# eval_tau(J, r, chieff, q, chi1, chi2, precomputedroots=None)
# Soft(t, J, r, chieff, q, chi1, chi2, precomputedroots=None)
# tofS(S, J, r, chieff, q, chi1, chi2, cyclesign=1, precomputedroots=None)
# Ssampling(J, r, chieff, q, chi1, chi2, N=1)
# Ssav_mfactor(m)
# Ssav(J, r, chieff, q, chi1, chi2)
# Ssrootsinf(theta1inf, theta2inf, q, chi1, chi2)
# Ssavinf(theta1inf, theta2inf, q, chi1, chi2)
# rhs_precav(u, kappa, chieff, q, chi1, chi2)
# integrator_precav(kappainitial, uinitial, ufinal, chieff, q, chi1, chi2)
# inspiral_precav(theta1=None, theta2=None, deltaphi=None, S=None, J=None, kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None, requested_outputs=None)
# precession_average(J, r, chieff, q, chi1, chi2, func, *args, method='quadrature', Nsamples=1e4)
# rupdown(q, chi1, chi2)
# omegasq_aligned(r, q, chi1, chi2, which)
# widenutation(q, chi1, chi2)
# rhs_orbav(v, allvars, q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula=False)
# integrator_orbav(Lhinitial, S1hinitial, S2hinitial, vinitial, vfinal, q, chi1, chi2, quadrupole_formula=False)
# inspiral_orbav(theta1=None, theta2=None, deltaphi=None, S=None, Lh=None, S1h=None, S2h=None, J=None, kappa=None, r=None, u=None, chieff=None, q=None, chi1=None, chi2=None, quadrupole_formula=False, requested_outputs=None)
# inspiral_hybrid(theta1=None, theta2=None, deltaphi=None, S=None, J=None, kappa=None, r=None, rswitch=None, u=None, uswitch=None, chieff=None, q=None, chi1=None, chi2=None, requested_outputs=None)
# inspiral(*args, which=None, **kwargs)
# frequency_prefactor(J, r, chieff, q, chi1, chi2)
# azimuthalangle_prefactor(J, r, chieff, q, chi1, chi2, precomputedroots=None)
# eval_OmegaL(S, J, r, chieff, q, chi1, chi2)
# eval_alpha(J, r, chieff, q, chi1, chi2, precomputedroots=None)
# eval_phiL(S, J, r, chieff, q, chi1, chi2, cyclesign=1, precomputedroots=None)
# chip_terms(theta1, theta2, q, chi1, chi2)
# eval_chip_heuristic(theta1, theta2, q, chi1, chi2)
# eval_chip_generalized(theta1, theta2, deltaphi, q, chi1, chi2)
# eval_chip_asymptotic(theta1, theta2, q, chi1, chi2)
# eval_chip_averaged(theta1=None, theta2=None, deltaphi=None, J=None, r=None, chieff=None, q=None, chi1=None, chi2=None, method='quadrature', Nsamples=1e4)
# _integrand(S, J, r, chieff, q, chi1, chi2)
# eval_chip(theta1=None, theta2=None, deltaphi=None, J=None, r=None, chieff=None, q=None, chi1=None, chi2=None, which="averaged", method='quadrature', Nsamples=1e4)
# gwfrequency_to_pnseparation(theta1, theta2, deltaphi, f, q, chi1, chi2, M_msun)
# pnseparation_to_gwfrequency(theta1, theta2, deltaphi, r, q, chi1, chi2, M_msun)
