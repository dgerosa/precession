
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


def both(testfunction, multiple=5):
    """
    testfunction is the function that performes the test and it's called "test_codefunction" where codefunction is a function of the precession code.
    """

    @wraps(testfunction)
    def wrapper():

        #input, output = testfunction
        testargs = testfunction()
        if len(testargs) == 2:
            input, output = testargs
            input_repeat = list(input.keys())
        elif len(testargs) == 3:
            input, output, input_repeat = testargs
            if input_repeat is None:
                input_repeat = []
            elif input_repeat == 'all':
                input_repeat = list(input.keys())
            elif type(input_repeat) is str:
                input_repeat = [input_repeat]

        # Extract the codefunction
        cfs = testfunction.__name__.split("_")
        assert cfs[0] == 'test'
        if cfs[1].isdigit():
            codefunction = "_".join(cfs[2:])
        else:
            codefunction = "_".join(cfs[1:])

        codefunction = eval("precession."+codefunction)

        # Inflate the inputs
        _input = input.copy()
        for arg in input_repeat:
            #_input[arg] = np.repeat(_input[arg], multiple, axis=-1)
            #_input[arg] = np.squeeze(np.tile(_input[arg], multiple).reshape(multiple, np.size(input[arg])))
            _input[arg] = np.squeeze(np.repeat([_input[arg]], multiple, axis=0))

        # Random seed for functions which use resampling
        np.random.seed(42)
        returns = codefunction(**input)

        np.random.seed(42)
        _returns = codefunction(**_input)

        # If codefunction returns a dictionary, convert it to a list
        if type(output) is dict:
            assert (type(returns) is dict) and (type(_returns) is dict)

            # This bit can change depending on whether output has same keys as returns
            # and if so, whether same ordering can be assumed or not
            # E.g., for key in keys can be replaced with np.array(list(returns.values))
            #assert sorted(output.keys()) == sorted(returns.keys())
            #returns = list(returns.values())
            #_returns = list(_returns.values())
            #output = list(output.values())
            # Make sure returns are same order as output
            keys = sorted(output.keys())
            returns = [returns[key] for key in keys]
            _returns = [_returns[key] for key in keys]
            output = [np.array(output[key]) for key in keys]

        # Make sure output is ready for reshaping
        #output = np.array(output)
        # Inflate the outputs
        #_output = np.reshape( np.repeat(output, multiple, axis=0), (output.shape[0],multiple) )
        #_output = np.tile(output, multiple)

        # arrays in output dictionary can have different shapes
        _output = []
        for par in output:
            _output.append(np.squeeze(np.repeat([par], multiple, axis=0)))

        # Test on a single entry
        #checksingle = np.allclose(returns, output)
        # Test on multiple entries
        #checkmultiple = np.allclose(_returns, _output)

        # Actual test for pytest
        #assert checksingle
        #assert checkmultiple

        for which in [[returns, output], [_returns, _output]]:
            for r, o in zip(*which):
                assert np.allclose(r, o)

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
    return {"r":10, "q":0.8}, [0.7808093]

@both
def test_eval_v():
    return {"r":10}, [0.31622777]

@both
def test_1_eval_r():
    return {"L":1 ,"q":0.8}, [16.4025]

@both
def test_2_eval_r():
    return {"u":0.5 ,"q":0.8}, [16.4025]

@both
def test_Jlimits_LS1S2():
    # See Fig 5 in arxiv:1506.03492
    return {"r":10 ,"q":0.8, "chi1":1, "chi2":1}, [[0.27463646],[1.28698214]]

@both
def test_kappadiscriminant_coefficients():
    return {"u":0.5 ,"chieff":0.5, "q":0.8, "chi1":1, "chi2":1}, [[-2.22902511e+03],[ 3.19940568e+03],[-1.79577102e+03],[ 4.89088064e+02],[-6.38228469e+01],[ 3.11722082e+00]]

@both
def test_kapparesonances():
    return {"u":0.5 ,"chieff":0.5, "q":0.8, "chi1":1, "chi2":1}, [[0.28276221],[0.38587938]]

@both
def test_kappainfresonances():
    return {"chieff":0.5, "q":0.8, "chi1":1, "chi2":1}, [[0.22839506],[0.28395062]]

@both
def test_Jresonances():
    # See Fig 5 in arxiv:1506.03492
    return {"r":10, "chieff":0.5, "q":0.8, "chi1":1, "chi2":1}, [[1.03459125],[1.12552698]]

@both
def test_1_Jlimits():
    # Should be like test_Jlimits_LS1S2
    return {"r":10 ,"q":0.8, "chi1":1, "chi2":1}, [[0.27463646],[1.28698214]]

@both
def test_2_Jlimits():
    # Should be like test_Jresonances
    return {"r":10, "chieff":0.5, "q":0.8, "chi1":1, "chi2":1}, [[1.03459125],[1.12552698]]

@both
def test_1_kappainflimits():
    return {"q":0.8, "chi1":1, "chi2":1}, [[-0.50617284],[ 0.50617284]]

@both
def test_2_kappainflimits():
    return {"chieff":0.5, "q":0.8, "chi1":1, "chi2":1}, [[0.22839506],[0.28395062]]

@both
def test_chiefflimits_definition():
    return {"q":0.8, "chi1":1, "chi2":1}, [[-1],[1]]

@both
def test_chiefflimits_definition():
    return {"q":0.8, "chi1":1, "chi2":1}, [[-1],[1]]

@both
def test_chieffdiscriminant_coefficients():
    return {"kappa":0, "u":0.5, "q":0.8, "chi1":1, "chi2":1}, [[1.67772160e+01], [7.18727648e+01], [1.65274010e+01], [-2.22873533e+00], [-5.73039852e-01], [-9.48560958e-03], [2.68503758e-03]]

@both
def test_chieffresonances():
    # See Fig 4 in arxiv:1506.03492
    return {"J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [[0.16035695],[0.43413573]]

@both
def test_1_anglesresonances():
    # See Fig 5 in arxiv:1506.03492
    return {"J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [[1.2517679], [1.60205348], [0], [1.39849931], [0.7036308], [np.pi]]

@both
def test_2_anglesresonances():
    # See Fig 5 in arxiv:1506.03492
    return {"J":0.25, "r":10, "q":0.2, "chi1":1, "chi2":1}, [[3.0860461], [1.41114207], [np.pi], [2.9563055], [0.05030985], [np.pi]]

@both
def test_3_anglesresonances():
    # See Fig 5 in arxiv:1506.03492
    return {"r":10, "chieff":0.5, "q":0.8, "chi1":1, "chi2":1}, [[1.27123854], [0.71342048], [np.pi], [0.90362362], [1.21157995], [0]]

@both
def test_1_chiefflimits():
    # Should be like test_chiefflimits_definition
    return {"q":0.8, "chi1":1, "chi2":1}, [[-1],[1]]

@both
def test_2_chiefflimits():
    # Should be like test_chieffresonances
    return {"J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [[0.16035695],[0.43413573]]

@both
def test_Slimits_S1S2():
    return {"q":0.8, "chi1":1, "chi2":1}, [[0.11111111],[0.50617284]]

@both
def test_Slimits_LJ():
    return {"J":1, "r":10, "q":0.8}, [[0.2191907], [1.7808093]]

@both
def test_Slimits_LJS1S2():
    # See Fig 4 in arxiv:1506.03492
    return {"J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [[0.2191907], [0.50617284]]

@both
def test_Scubic_coefficients():
    return {"kappa":0, "u":0.5, "chieff":0.5, "q":0.8, "chi1":1, "chi2":1}, [[6.48000000e-01], [6.74375309e-01], [1.47249383e-01], [1.02484377e-04]]

@both
def test_1_Ssroots():
    return {"J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1}, [[0.08748025], [0.19301536], [0.01050669]]

#TODO: the following test fails because of some array inflating in @both
# This works fine:
#    precomputedroots=Ssroots(J=1,r=10,chieff=0.3,q=0.8,chi1=1,chi2=1)
#    print(Ssroots(J=None,r=None,chieff=None,q=None,chi1=None,chi2=None,precomputedroots=precomputedroots))

# @both
# def test_2_Ssroots():
#     precomputedroots=precession.Ssroots(J=1,r=10,chieff=0.3,q=0.8,chi1=1,chi2=1)
#     return {"J":None, "r":None, "chieff":None, "q":None, "chi1":None, "chi2":None, "precomputedroots":precomputedroots}, [[0.08748025], [0.19301536], [0.01050669]]

@both
def test_Slimits_plusminus():
    # See Fig 4 in arxiv:1506.03492
    return {"J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1}, [[0.2957706], [0.43933514]]

@both
def test_1_Satresonance():
    # See Fig 4 in arxiv:1506.03492. precession_v1, resonant_finder with more=True
    return {"J":1, "r":10, "chieff":0.43413573, "q":0.8, "chi1":1, "chi2":1}, [0.26925273]

@both
def test_2_Satresonance():
    return {"J":1, "u":0.64036123, "chieff":0.43413573, "q":0.8, "chi1":1, "chi2":1}, [0.26925273]

@both
def test_3_Satresonance():
    return {"kappa":0.24995658, "r":10, "chieff":0.43413573, "q":0.8, "chi1":1, "chi2":1}, [0.26925273]

@both
def test_4_Satresonance():
    return {"kappa":0.24995658, "u":0.64036123, "chieff":0.43413573, "q":0.8, "chi1":1, "chi2":1}, [0.26925273]


@both
def test_1_Slimits():
    # Should be like test_Slimits_S1S2
    return {"q":0.8, "chi1":1, "chi2":1}, [[0.11111111],[0.50617284]]

@both
def test_2_Slimits():
    # Should be like test_Slimits_LJ
    return {"J":1, "r":10, "q":0.8}, [[0.2191907], [1.7808093]]

@both
def test_3_Slimits():
    # Should be like test_Slimits_LJS1S2
    return {"J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [[0.2191907], [0.50617284]]

@both
def test_4_Slimits():
    # Should be like test_Slimits_plusminus
    return {"J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1}, [[0.2957706], [0.43933514]]

@both
def test_1_eval_chieff():
    return {"theta1":np.pi/8, "theta2":np.pi/4, "q":0.6, "chi1":1, "chi2":1}, [0.84258975]

@both
def test_2_eval_chieff():
    # See Fig 2 in arxiv:1506.03492.
    return {"S":0.4, "varphi":np.pi/4, "J":2.34, "r":100, "q":0.6, "chi1":1, "chi2":1}, [-0.16650103]

@both
def test_effectivepotential_plus():
    # See Fig 5 in arxiv:1506.03492.
    return {"S":0.4, "J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [0.34933852]

@both
def test_effectivepotential_minus():
    # See Fig 5 in arxiv:1506.03492.
    return {"S":0.4, "J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [0.22470033]

@both
def test_eval_varphi():
    # See Fig 2 in arxiv:1506.03492.
    return {"S":0.4, "J":2.34, "r":100, "chieff":-0.05, "q":0.6, "chi1":1, "chi2":1, "cyclesign":-1}, [1.66785314]

@both
def test_eval_costheta1():
    return {"S":0.4, "J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1}, [0.22948025]

@both
def test_eval_theta1():
    return {"S":0.4, "J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1}, [1.33925268]

@both
def test_eval_costheta2():
    return {"S":0.4, "J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1}, [0.38814969]

@both
def test_eval_theta2():
    return {"S":0.4, "J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1}, [1.1721733]

@both
def test_1_eval_costheta12():
    return {"S":0.4, "q":0.8, "chi1":1, "chi2":1}, [0.21095]

@both
def test_2_eval_costheta12():
    return {"theta1":1.33925268, "theta2":1.1721733, "deltaphi":1.43450291}, [0.21095]

@both
def test_1_eval_theta12():
    return {"S":0.4, "q":0.8, "chi1":1, "chi2":1}, [1.3582496]

@both
def test_2_eval_theta12():
    return {"theta1":1.33925268, "theta2":1.1721733, "deltaphi":1.43450291}, [1.3582496]

@both
def test_eval_cosdeltaphi():
    return {"S":0.4, "J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1}, [0.13587184]

@both
def test_eval_deltaphi():
    return {"S":0.4, "J":1, "r":10, "chieff":0.3, "q":0.8, "chi1":1, "chi2":1, "cyclesign":-1}, [1.43450291]

@both
def test_eval_costhetaL():
    return {"S":0.4, "J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [0.92830808]

@both
def test_eval_thetaL():
    return {"S":0.4, "J":1, "r":10, "q":0.8, "chi1":1, "chi2":1}, [0.38096012]

@both
def test_1_eval_J():
    return {"theta1":np.pi/8, "theta2":np.pi/4, "deltaphi":np.pi/3, "r":100, "q":0.6, "chi1":1, "chi2":1}, [2.81246294]

@both
def test_2_eval_J():
    return {"kappa":0.24995658, "r":10, "q":0.8}, [1.]

@both
def test_eval_S():
    return {"theta1":np.pi/8, "theta2":np.pi/4, "deltaphi":np.pi/3, "q":0.6, "chi1":1, "chi2":1}, [0.50891976]

@both
def test_eval_kappa():
    return {"J":1, "r":10, "q":0.8}, [0.24995658]

@both
def test_eval_u():
    return {"r":16.4025 ,"q":0.8}, [0.5]

@both
def test_eval_kappainf():
    return {"theta1inf":np.pi/8 , "theta2inf":np.pi/4, "q":0.6, "chi1":1, "chi2":1}, [0.46032733]

@both
def test_eval_costheta1inf():
    return {"kappainf":0.46032733 , "chieff":0.84258975, "q":0.6, "chi1":1, "chi2":1}, [np.cos(np.pi/8)]

@both
def test_eval_theta1inf():
    return {"kappainf":0.46032733 , "chieff":0.84258975, "q":0.6, "chi1":1, "chi2":1}, [np.pi/8]

@both
def test_eval_costheta2inf():
    return {"kappainf":0.46032733 , "chieff":0.84258975, "q":0.6, "chi1":1, "chi2":1}, [np.cos(np.pi/4)]

@both
def test_eval_theta2inf():
    return {"kappainf":0.46032733 , "chieff":0.84258975, "q":0.6, "chi1":1, "chi2":1}, [np.pi/4]


# TODO
# limits_check(S=None, J=None, r=None, chieff=None, q=None, chi1=None, chi2=None)


### There needs to be tests for all these functions, multiple ones for some functions.
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



#
# # For precav the precession-timescale parameters are not testable due to resampling
# # You can just get rid of those parameters from the output dictionary and the decorator handles it
# @both
# def test_inspiral_precav():
#     input = {'theta1': np.pi/3,
#              'theta2': np.pi/4,
#              'deltaphi': np.pi/5,
#              'r': [100, 10],
#              'q': 0.9,
#              'chi1': 0.9,
#              'chi2': 0.9}
#     output = {#'theta1': np.array([[1.06093077, 0.48512929]]),
#               #'theta2': np.array([[0.76645558, 1.28725517]]),
#               #'deltaphi': np.array([[ 0.56240114, -0.8766089 ]]),
#               #'S': np.array([[0.43577611, 0.3958433 ]]),
#               'J': np.array([[2.781612  , 1.10229372]]),
#               'kappa': np.array([[0.30523421, 0.3764109 ]]),
#               'r': np.array([[100,  10]]),
#               'u': np.array([[0.20055556, 0.63421235]]),
#               'chieff': np.array([0.53829289]),
#               'q': np.array([0.9]),
#               'chi1': np.array([0.9]),
#               'chi2': np.array([0.9])}
#     input_repeat = ['theta1', 'theta2', 'deltaphi', 'r', 'q', 'chi1', 'chi2']
#     return input, output
#
#
# @both
# def test_inspiral_orbav():
#     input = {'theta1': np.pi/3,
#              'theta2': np.pi/4,
#              'deltaphi': np.pi/5,
#              'r': [100, 10],
#              'q': 0.9,
#              'chi1': 0.9,
#              'chi2': 0.9}
#     output = {'t': ([[      0.        , 8033766.36903445]]),
#  'theta1': ([[1.04719755, 0.3598033 ]]),
#  'theta2': ([[0.78539816, 1.3560499 ]]),
#  'deltaphi': ([[0.62831853, 0.44602418]]),
#  'S': ([[0.43406977, 0.39482006]]),
#  'Lh': ([[ 0.12291091,  0.        ,  0.99241771],
#         [ 0.04774722, -0.24924823,  0.96749448]]),
#  'S1h': ([[-0.7717029 ,  0.21260216,  0.5993955 ],
#         [ 0.10108719,  0.10443983,  0.99329492]]),
#  'S2h': ([[-0.56469899, -0.2624718 ,  0.78244719],
#         [-0.26706902,  0.85727799,  0.45588938]]),
#  'J': ([[2.781612  , 1.10248973]]),
#  'kappa': ([[0.30523421, 0.37668498]]),
#  'r': ([[100,  10]]),
#  'u': ([[0.20055556, 0.63421235]]),
#  'chieff': ([[0.53829289, 0.53655487]]),
#  'q': ([0.9]),
#  'chi1': ([0.9]),
#  'chi2': ([0.9])}
#     return input, output
