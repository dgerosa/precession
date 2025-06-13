from importlib.machinery import SourceFileLoader
import ast
import inspect
import functools
import re
import warnings
import numpy as np
import scipy.special
import scipy.integrate
import scipy.spatial.transform
from itertools import repeat
import precession
from precession import *

"""
This module dynamically wraps functions from `precession.py`, replacing the `r` parameter
with `a` (semi-major axis) and `e` (eccentricity). All functions listed in `__all__`
are available for use and are documented below.
"""


def eccentricize(func):
    sig = inspect.signature(func)

    # Replace 'r' with 'a' and 'e' in the function signature
    new_params = []
    for name, param in sig.parameters.items():
        if name == 'r':
            default = None if param.default is not inspect.Parameter.empty else inspect.Parameter.empty
            new_params.append(inspect.Parameter('a', kind=param.kind, default=default))
            new_params.append(inspect.Parameter('e', kind=param.kind, default=default))
        else:
            new_params.append(param)
    new_sig = sig.replace(parameters=new_params)

    # Update the docstring
    old_doc = func.__doc__
    if old_doc:
        lines = old_doc.split('\n')
        new_lines = []
        skip_next = False

        for line in lines:
            stripped = line.strip()
            
            # Replace 'r:' parameter block with a/e
            if stripped.startswith('r: float, optional (default: None)'):
                indent = line[:line.find('r:')]
                new_lines.append(f"{indent}a: float, optional (default: None)")
                new_lines.append(f"{indent}    Semi-major axis.")
                new_lines.append(f"{indent}e: float, optional (default: None)")
                new_lines.append(f"{indent}    Eccentricity: 0<=e<=1")
                skip_next = True 
                continue
            elif stripped.startswith('r: '):
                indent = line[:line.find('r:')]
                new_lines.append(f"{indent}a: float")
                new_lines.append(f"{indent}    Semi-major axis.")
                new_lines.append(f"{indent}e: float")
                new_lines.append(f"{indent}    Eccentricity: 0<=e<=1")
                skip_next = True
                continue
            if skip_next:
                skip_next = False
                continue

            line = re.sub(r'r=r,', 'a=a,e=e,', line)
            line = re.sub(r',r,', ',a,e,', line)
            new_lines.append(line)

        new_doc = '\n'.join(new_lines)
    else:
     new_doc = None



    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound_new = new_sig.bind(*args, **kwargs)
        bound_new.apply_defaults()

        a = bound_new.arguments.pop('a')
        e = bound_new.arguments.pop('e')
        r = a * (1 - e**2)

        arguments = dict(bound_new.arguments)
        arguments['r'] = r

        bound_orig = sig.bind(**arguments)
        bound_orig.apply_defaults()

        return func(*bound_orig.args, **bound_orig.kwargs)

    wrapper.__signature__ = new_sig
    wrapper.__doc__ = new_doc
    return wrapper



# Load the original functions from precession.py
circ = precession 
#functions = ['roots_vec', 'norm_nested', 'normalize_nested', 'dot_nested', 'scalar_nested', 'rotate_nested', 'sample_unitsphere', 'isotropic_angles', 'tiler', 'affine', 'inverseaffine', 'wraproots', 'ellippi', 'ismonotonic', 'eval_m1', 'eval_m2', 'eval_q', 'eval_eta', 'eval_S1', 'eval_S2', 'eval_chi1', 'eval_chi2', 'eval_L', 'eval_v', 'eval_r', 'eval_u', 'eval_chieff', 'eval_deltachi', 'eval_deltachiinf', 'eval_costheta1', 'eval_theta1', 'eval_costheta2', 'eval_theta2', 'eval_costheta12', 'eval_theta12', 'eval_cosdeltaphi', 'eval_deltaphi', 'eval_costhetaL', 'eval_thetaL', 'eval_J', 'eval_kappa', 'eval_S', 'eval_cyclesign', 'conserved_to_angles', 'angles_to_conserved', 'vectors_to_angles', 'vectors_to_Jframe', 'vectors_to_Lframe', 'angles_to_Lframe', 'angles_to_Jframe', 'conserved_to_Lframe', 'conserved_to_Jframe', 'vectors_to_conserved', 'kappadiscriminant_coefficients', 'kappalimits_geometrical', 'kapparesonances', 'kapparescaling', 'kappalimits', 'chiefflimits', 'deltachilimits_definition', 'anglesresonances', 'deltachicubic_coefficients', 'deltachicubic_rescaled_coefficients', 'deltachiroots', 'deltachilimits_rectangle', 'deltachilimits_plusminus', 'deltachilimits', 'deltachirescaling', 'deltachiresonance', 'elliptic_parameter', 'deltachitildeav', 'deltachitildeav2',  'dchidt2_RHS', 'eval_tau', 'deltachioft', 'tofdeltachi', 'deltachisampling', 'intertial_ingredients', 'eval_OmegaL', 'eval_phiL', 'eval_alpha', 'morphology', 'chip_terms', 'eval_chip_heuristic', 'eval_chip_generalized', 'eval_chip_averaged', 'eval_chip_rms', 'eval_chip', 'eval_nutation_freq', 'eval_bracket_omega', 'eval_delta_omega', 'eval_delta_theta', 'eval_bracket_theta', 'rupdown', 'updown_endpoint', 'angleresonances_endpoint', 'omegasq_aligned', 'widenutation_separation', 'widenutation_condition', 'rhs_precav', 'integrator_precav', 'inspiral_precav', 'precession_average', 'rhs_orbav', 'integrator_orbav', 'inspiral_orbav', 'inspiral_hybrid', 'inspiral', 'gwfrequency_to_pnseparation', 'pnseparation_to_gwfrequency', 'remnantmass', 'remnantspin', 'reminantspindirection', 'remnantkick']
functions = ['eval_L',
 'eval_v',
 'eval_u',
 'eval_cosdeltaphi',
 'eval_deltaphi',
 'eval_costhetaL',
 'eval_thetaL',
 'eval_J',
 'eval_kappa',
 'eval_S',
 'conserved_to_angles',
 'angles_to_conserved',
 'angles_to_Lframe',
 'angles_to_Jframe',
 'conserved_to_Lframe',
 'conserved_to_Jframe',
 'kappalimits_geometrical',
 'kapparesonances',
 'kapparescaling',
 'kappalimits',
 'anglesresonances',
 'deltachilimits_plusminus',
 'deltachilimits',
 'deltachirescaling',
 'deltachiresonance',
 'dchidt2_RHS',
 'eval_tau',
 'deltachioft',
 'tofdeltachi',
 'deltachisampling',
 'intertial_ingredients',
 'eval_OmegaL',
 'eval_phiL',
 'eval_alpha',
 'morphology',
 'eval_chip_averaged',
 'eval_chip_rms',
 'eval_chip',
 'eval_nutation_freq',
 'eval_bracket_omega',
 'eval_delta_omega',
 'eval_delta_theta',
 'eval_bracket_theta',
 'omegasq_aligned',
 'widenutation_condition',
 'precession_average',
 ]
# Apply eccentricize and expose the new functions
for name in functions:
    func = getattr(circ, name)
    sig = inspect.signature(func)
    if 'r' in sig.parameters:
        globals()[name] = eccentricize(func)
    else:
        # No 'r' param, keep original function as is
        globals()[name] = func


def eval_a(L=None,u=None,uc=None,e=None,q=None):
    """
    Semi-major axis of the binary. Valid inputs are either  (L,e,q) or (u,e,q) or (uc,q).

    Call
    ----
    a = eval_a(a=a, L=L,u=u,e=e,q=q)

    Parameters
    ----------
    
    L: float, optional (default: None)
        Magnitude of the Newtonian orbital angular momentum.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    uc: float, optional (default: None)
        Circular compactified separation uc=u(e=0).    
    e: float (default: 0)
        Binary eccentricity: 0<=e<=1.     
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.

    Returns
    -------
    a: float
        Binary semi-major axis.
    """
   
    if L is not None and u is None and q is not None:
        L = np.atleast_1d(L)
        a = (L * (1+q)**2)**2 / (q**2 *(1-e**2))

    elif L is None and u is not None and q is not None:
        u = np.atleast_1d(u)
        a = (1+q)**4/(4*q**2*(1-e**2)*u**2)

    elif L is None and u is None and q is not None and uc is not None :
         eta=eval_eta(q)
         a = 1/4 * (uc)**(-2) * (eta)**(-2)
    else:
        raise TypeError("Provide either  (L,e,q) or (u,e,q) or (uc, q)")
    return a


def eval_e(L=None,u=None,uc=None,a=None,q=None):
    """
    Orbital eccentricity of the binary. Valid inputs are either (L,a,q) or (u,uc,q).

    Call
    ----
    e = eval_e(L=L, a=a, q=q)
    e = eval_e(u=u, uc=uc, q=q)

    Parameters
    ----------

    L: float, optional (default: None)
        Magnitude of the Newtonian orbital angular momentum.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    uc: float, optional (default: None)
        Circular compactified separation uc=u(e=0).   
    a: float, optional (default: None)
        Binary semi-major axis.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.

    Returns
    -------
    e: float
        Binary eccentricity.
    """
   
    if L is not None and u is None and q is not None:
        q = np.atleast_1d(q)
        L = np.atleast_1d(L)
        pre_E=(1-( ((L**2 * (1+q)**4) / (a*q**2)))) 
        pre_E=np.where(pre_E<0, 0, pre_E)
        e =np.sqrt(pre_E) 
   

    elif L is None and u is not None and q is not None and uc is not None:
        u= np.atleast_1d(u)
        uc= np.atleast_1d(uc)
        e =np.sqrt(1-np.float64(uc)**2/np.float64(u)**2)

    elif L is None and u is not None and q is not None and a is not None:
        u= np.atleast_1d(u)
        a= np.atleast_1d(a)
        e =np.sqrt(-1-4*q-4*q**3-q**4+q**2*(-6+4*a*u**2))/(2*np.sqrt(a)*q*u)

    else:
        raise TypeError("Provide either (L,a,q) or (u, uc,q).")    
    
    return e


def ddchidt_prefactor(a, e, chieff, q):
    """
    etamerical prefactor to the ddeltachi/dt derivative.
    
    Parameters
    ----------
    a: float
        Semi-major axis.
    e: float
        Eccentricity: 0<=e<=1.    
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    
    Returns
    -------
    mathcalA: float
        Prefactor in the ddeltachi/dt equation.
    
    Examples
    --------
    ``mathcalA = precession.ddchidt_prefactor(a,e,chieff,q)``
    """

    a = np.atleast_1d(a).astype(float)
    e = np.atleast_1d(e).astype(float)
    chieff = np.atleast_1d(chieff).astype(float)
    q = np.atleast_1d(q).astype(float)
    r = a * (1 - e**2)
    mathcalA = (3/2)*((1+q)**(-1/2))*(r**(-11/4))*(1-(chieff/r**0.5))*(1 - e**2)**(3/2)

    return mathcalA        


def vectors_to_conserved(Lvec, S1vec, S2vec, a, e , q,full_output=False):
    """
    Convert vectors (L,S1,S2) to conserved quanties (deltachi,kappa,chieff).
    
    Parameters
    ----------
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    full_output: boolean, optional (default: False)
        Return additional outputs.
    
    Returns
    -------
    chieff: float
        Effective spin.
    cyclesign: integer, optional
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``deltachi,kappa,chieff = precession.vectors_to_conserved(Lvec,S1vec,S2vec,q)``
    ``deltachi,kappa,chieff,cyclesign = precession.vectors_to_conserved(Lvec,S1vec,S2vec,q,full_output=True)``
    """

    L = norm_nested(Lvec)
    S1 = norm_nested(S1vec)
    S2 = norm_nested(S2vec)

    if a is None and e is not  None:
        a = eval_a(L=L,e=e,q=q)
    elif e is None and a is not None:
        e = eval_e(L=L,a=a,q=q)    
 
    chi1 = eval_chi1(q,S1)
    chi2 = eval_chi2(q,S2)

    theta1,theta2,deltaphi = vectors_to_angles(Lvec, S1vec, S2vec)

    deltachi, kappa, chieff, cyclesign= angles_to_conserved(theta1, theta2, deltaphi, a, e, q, chi1, chi2, full_output=True)

    if full_output:
        return np.stack([deltachi, kappa, chieff, cyclesign])

    else:
        return np.stack([deltachi, kappa, chieff])

def implicit(u,uc):
    """LHS of eq. (28) in Fumagalli & Gerosa, arXiv:2310.16893. This is used to compute the implicit function u(uc) in the inspiral_precav function.
    Parameters 
    ----------
    u: float
        Compactified separation 1/(2L).
    uc: float
        Circular compactified separation 1/(2L(e=0)).
    """
    return uc * u**(37/84) * (u**2 / uc**2 - 1 )**(121/532) * (u**2 / uc**2 -(121/425))**(145/532)

def inspiral_precav(theta1=None, theta2=None, deltaphi=None,deltachi=None, kappa=None, a=None, e=None, u=None, uc=None, chieff=None, q=None, chi1=None, chi2=None, requested_outputs=None,  enforce=False, **odeint_kwargs):
    """
    The integration range must be specified using either (a,e) or (uc, u), (and not both). These need to be arrays with lenght >=1, where e.g. a[0] corresponds to the initial condition and a[1:] corresponds to the location where outputs are returned. Do not go to past time infinity with e!=0.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    The initial conditions must be specified in terms of one an only one of the following:
        - theta1,theta2, and deltaphi.
        - kappa, chieff.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'a', 'e', 'u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'].
    The flag enforce allows checking the consistency of the input variables.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.

     Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
    r: float, optional (default: None)
        Binary separation.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    requested_outputs: list, optional (default: None)
        Set of outputs.
    enforce: boolean, optional (default: False)
        If True raise errors, if False raise warnings.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,a=a,e=e,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,uc=uc,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(kappa,a=a,e=e,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(kappa,uc=uc,u=u,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, kappa, a, e, uc, u, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if   k == 4 or k == 6 :  # Either  (a,e) or (uc, uc)
                    inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
                
    theta1, theta2, deltaphi,  kappa, a, e, uc, u, chieff, q, chi1, chi2 = inputs
    # This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'a', 'e', 'uc','u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'])
    # If in doubt, return everything
    if requested_outputs is None:
        requested_outputs = alloutputs
    
    def _compute(theta1, theta2, deltaphi, kappa, a, e, uc, u, chieff, q, chi1, chi2):
        
        # Make sure you have q, chi1, and chi2.
        if q is None or chi1 is None or chi2 is None:
            raise TypeError("Please provide q, chi1, and chi2.")
        ## User pass (a, e0) -> return uc and u [from u recover e]
        if a is not None and e is not None and uc is None and u is None:

            assert np.logical_or(ismonotonic(a, '<='), ismonotonic(a, '>=')), 'a must be monotonic'
            if e !=0:
                def solve(uc, c0, e):
                     return scipy.optimize.brentq(lambda u : implicit(u,uc) - c0, 100*uc/(1-e**2)**0.5, uc, xtol=1e-15)
                uc_vals =eval_u(a, tiler(0, a), tiler(q, a))
                u0 = eval_u(a[0], e,q)
                c0 = implicit(u0,uc_vals[0]) 
                us=[]
                eb=e
                for uc in uc_vals: 
                   u=solve(uc,c0,eb)
                   eb=np.sqrt(1-np.float64(uc)**2/np.float64(u)**2)
                   us.append(u)
                u=np.asarray(us)
                uc = eval_u(a, tiler(0, a), tiler(q, a))
            else:
                u = eval_u(a,tiler(e, a), tiler(q, a))
                uc=u
         ## User pass (uc, u0) -> return a, e0, u [from u recover e]   
        elif a is None and e is None and u is not None and uc is not None and  np.shape(uc) >  np.shape(u):
           # print('dentro')
            assert np.logical_or(ismonotonic(uc, '<='), ismonotonic(uc, '>=')), 'uc must be monotonic'
            if uc[0] != u:
                def solve(uc, c0, e):
                     return scipy.optimize.brentq(lambda u : implicit(u,uc) - c0, 100*uc/(1-e**2)**0.5, uc, xtol=1e-15)
                c0 = implicit(u,uc[0]) 
                us=[]
                eb=eval_e(u=u, uc=uc[0], q=q)
                for ucs in uc: 
                   u=solve(ucs,c0,eb)
                   eb=np.sqrt(1-np.float64(ucs)**2/np.float64(u)**2)
                   us.append(u)
                u=np.asarray(us)
                a=eval_a(uc=uc, q=tiler(q, uc))
                e=eval_e(u=u[0], uc=uc[0], q=q)
            else:
               u=uc
               a=eval_a(uc=uc, q=tiler(q, uc)) 
               e=0

        else:
            raise TypeError("Please provide either (a,e0) or (uc, u0).  Do not work at infinity.")
     
        assert np.sum(u == 0) <= 1 and np.sum(u[1:-1] == 0) == 0, "There can only be one a=np.inf location, either at the beginning or at the end."
       
        if theta1 is not None and theta2 is not None and deltaphi is not None and kappa is None and chieff is None:
                 deltachi, kappa, chieff = angles_to_conserved(theta1=theta1, theta2=theta2, deltaphi=deltaphi,a=a[0],e=e, q=q,chi1=chi1, chi2=chi2)
        # User provides kappa, chieff, and maybe deltachi.
        elif theta1 is None and theta2 is None and deltaphi is None and kappa is not None and chieff is not None:
            pass

        else:
            raise TypeError("Please provide one and not more of the following: (theta1,theta2,deltaphi), (kappa,chieff).")

        if enforce: # Enforce limits
            chieffmin, chieffmax = chiefflimits(q, chi1, chi2)
            assert chieff >= chieffmin and chieff <= chieffmax,  "Unphysical initial conditions [inspiral_precav]."+str(theta1)+" "+str(theta2)+" "+str(deltaphi)+" "+str(kappa)+" "+str( chieffmin)+" "+str(chieffmax )+" "+str(chieff)+" "+str(q)+" "+str(chi1)+" "+str(chi2)
            kappamin,kappamax = kappalimits(a=a[0],e=e,u=u[0], chieff=chieff, q=q, chi1=chi1, chi2=chi2)
            assert kappa >= kappamin and kappa <= kappamax, "kappa Unphysical initial conditions [inspiral_precav]."+str(theta1)+" "+str(theta2)+" "+str(deltaphi)+" "+str(kappa)+" "+str(kappamin)+" "+str(kappamax)+" "+str(chieff)+" "+str(q)+" "+str(chi1)+" "+str(chi2)

        # Actual integration.
        
        kappa = np.squeeze(integrator_precav(kappa, u, chieff, q, chi1, chi2,**odeint_kwargs))
        deltachiminus = None
        deltachiplus = None
        deltachi3 = None
        deltachi=None
        theta1=None
        theta2=None
        deltaphi=None
        
        e=eval_e(a=a, u=u, q=tiler(q, a))  
        # Roots along the evolution
        if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi', 'deltachi', 'deltachiminus', 'deltachiplus', 'deltachi3']):
            deltachiminus,deltachiplus,deltachi3 = deltachiroots(kappa, u, tiler(chieff,u), tiler(q,u),tiler(chi1,u),tiler(chi2,u))
                    
            if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi', 'deltachi']):
                #print(kappa, a,e,u)
                deltachi = deltachisampling(kappa, a,e, tiler(chieff,u), tiler(q,u),tiler(chi1,u),tiler(chi2,u), precomputedroots=np.stack([deltachiminus,deltachiplus,deltachi3]))
                # Compute the angles. Assign random cyclesign
                if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi']):
                    theta1,theta2,deltaphi = conserved_to_angles(deltachi, kappa, a,e, tiler(chieff,u), tiler(q,u),tiler(chi1,u),tiler(chi2,u), cyclesign = np.random.choice([-1, 1], u.shape))

        return theta1, theta2, deltaphi, deltachi, kappa, a, e, uc,u, deltachiminus, deltachiplus, deltachi3, chieff, q, chi1, chi2
    # Here I force dtype=object buse the outputs have different shapes
    allresults = np.array(list(map(_compute, theta1, theta2, deltaphi, kappa, a,e, uc, u, chieff, q, chi1, chi2)), dtype=object).T
   
    # Return only requested outputs (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs, requested_outputs)

    # Store into a dictionary
    outcome = {}

    for k, v in zip(alloutputs[wantoutputs], allresults[wantoutputs]):
        outcome[k] = np.squeeze(np.stack(v))

        # For the constants of motion...
        if k == 'chieff' or k == 'q' or k == 'chi1' or k == 'chi2':  # Constants of motion
            outcome[k] = np.atleast_1d(outcome[k])
        #... and everything else
        else:
            outcome[k] = np.atleast_2d(outcome[k])

    return outcome



def rhs_orbav(allvars, a, q, m1, m2, eta, chi1, chi2, S1, S2, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3]):
    """
    Right-hand side of the systems of ODEs describing orbit-averaged inspiral. The equations are reported in Sec 4A of Gerosa and Kesden, arXiv:1605.01067. The format is d[allvars]/dv=RHS where allvars=[Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz,t], h indicates unit vectors, v is the orbital velocity, and t is time.
    This is an internal function used by the ODE integrator and is not array-compatible.
    
    Parameters
    ----------
    allvars: array
        Packed ODE input variables.
    a: float
        Newtonian orbital velocity.
    q: float
        Mass ratio: 0<=q<=1.
    m1: float
        Mass of the primary (heavier) black hole.
    m2: float
        Mass of the secondary (lighter) black hole.
    eta: float
        Symmetric mass ratio 0<=eta<=1/4.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    S1: float
        Magnitude of the primary spin.
    S2: float
        Magnitude of the secondary spin.
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    
    Returns
    -------
    RHS: float
        Right-hand side.
    
    Examples
    --------
    ``RHS = precession.rhs_orbav(allvars,a,q,m1,m2,eta,chi1,chi2,S1,S2,PNorderpre=[0,0.5],PNorderrad=[0,1,1.5,2,2.5,3])``
    """

    # Unpack inputs
  
    Lh = allvars[0:3]
    S1h = allvars[3:6]
    S2h = allvars[6:9]
    delta_lambda = allvars[9]
    Ph= np.array([np.cos(delta_lambda), np.sin(delta_lambda), 0])

    e = np.sqrt(allvars[10])
    t = allvars[11]

    
    u=np.squeeze(eval_u(a, e,q))
    


    # Angles 
    ct1 = np.dot(S1h, Lh)
    ct2 = np.dot(S2h, Lh)
    ct12 = np.dot(S1h, S2h)
    
    cphi1 = np.dot(Ph, S1h)
    cphi2= np.dot(Ph, S2h)
    c2phi1=2*cphi1**2-1
    c2phi2=2*cphi2**2-1
    c2phi=np.cos(2*(np.arccos(cphi1)+np.arccos(cphi2)))
  

   
    
   
    # Spin precession for S1
    Omega1=  (0 in PNorderpre) *(16 * ((1 + -1 * (e)**(2)))**(3/2) * Lh * (4 + 3 * q) * (u)**(5) * \
    (eta)**(6))+(0.5 in PNorderpre) * (-32 * ((1 + -1 * (e)**(2)))**(3/2) * (q)**(-1) * (u)**(6) * \
    (eta)**(6) * (-1 * q * S2 * S2h + 3 * Lh * ((1 + q))**(2) * (ct1 * q * S1 + ct2 * S2) * eta))
    dS1hdt = np.cross(Omega1, S1h)

    # Spin precession for S2
    Omega2 =  (0 in PNorderpre) *(16 * ((1 + -1 * (e)**(2)))**(3/2) * Lh * (q)**(-1) * (3 + 4 * q) * \
    (u)**(5) * (eta)**(6))+(0.5 in PNorderpre) * (-32 * ((1 + -1 * (e)**(2)))**(3/2) * (q)**(-2) * (u)**(6) * \
    (eta)**(6) * (-1 * (q)**(2) * S1 * S1h + 3 * Lh * ((1 + q))**(2) * (ct1 * q * S1 + ct2 * S2) * eta))
    dS2hdt = np.cross(Omega2, S2h)


    # Conservation of angular momentum
    OmegaL=  (0 in PNorderpre) * (32 * ((1 + -1 * (e)**(2)))**(3/2) * (q)**(-1) * (q * (4 + 3 * q) * \
    S1 * S1h + (3 + 4 * q) * S2 * S2h) * (u)**(6) * (eta)**(6))-(0.5 in PNorderpre) * (192 * ((1 + -1 * \
    (e)**(2)))**(3/2) * (q)**(-2) * ((1 + q))**(2) * (ct1 * q * S1 + ct2 * S2) * (q * S1 * S1h + S2 * \
    S2h) * (u)**(7) * (eta)**(7))
    dLhdt = np.cross(OmegaL, Lh)

    #pn terms for dudt :') from Klein et al. 2018  arXiv:1801.08542
    zero_pnterm_u= 1024/5 * ((1 + -1 * (e)**(2)))**(3/2) * (8 + 7 * (e)**(2)) * (eta)**(9)

    one_pnterm_u = -256/315 * ((1 + -1 * (e)**(2)))**(3/2) * (eta)**(11) * (24 * (743 + \
    924 * eta) + (5 * (e)**(4) * (-9021 + 6832 * eta) + 8 * (e)**(2) * \
    (-18444 + 18403 * eta)))

    oneptfive_pnterm_u = 2048 * ((1 + -1 * (e)**(2)))**(3/2) * ((128/5 + (388/5 * (e)**(2) + \
    (49/5 * (e)**(4) + (-49/720 * (e)**(6) + (-109/5760 * (e)**(8) + \
    -2567/2304000 * (e)**(10)))))) * np.pi + (1/15 * (904 + (2224 * \
    (e)**(2) + 297 * (e)**(4))) * (ct1 * S1 + ct2 * S2) + 2/15 * (300 + \
    (958 * (e)**(2) + 157 * (e)**(4))) * (q)**(-1) * (ct1 * (q)**(2) * S1 \
    + ct2 * S2))) * (eta)**(12)
    
    two_pnterm_u=512/2835 * ((1 + -1 * (e)**(2)))**(3/2) * (eta)**(13) * (-1/10 * \
    (q)**(-2) * ((1 + q))**(2) * (-176 + (160 * q + (48 * (e)**(2) * (-10 \
    + 9 * q) + (e)**(4) * (-69 + 63 * q)))) * ((q)**(2) * (S1)**(2) + \
    (S2)**(2)) + (-3/5 * (28 + (76 * (e)**(2) + 11 * (e)**(4))) * \
    (q)**(-2) * ((1 + q))**(2) * ((q)**(2) * (S1)**(2) + (2 * ct12 * \
    q * S1 * S2 + (S2)**(2))) + (1/20 * (968 + (2616 * (e)**(2) + 381 * \
    (e)**(4))) * (q)**(-2) * ((1 + q))**(2) * ((q)**(2) * (S1)**(2) + (2 \
    * ct1 * ct2 * q * S1 * S2 + (S2)**(2))) + (1/10 * (q)**(-2) * ((1 + \
    q))**(2) * (24 * (e)**(2) * (-55 + 54 * q) + (8 * (-61 + 60 * q) + 3 \
    * (e)**(4) * (-64 + 63 * q))) * ((ct1)**(2) * (q)**(2) * (S1)**(2) + \
    (ct2)**(2) * (S2)**(2)) + (3/10 * c2phi * (e)**(2) * (149 + 31 * \
    (e)**(2)) * (q)**(-2) * ((1 + q))**(2) * ((-1 + (ct1)**(2)) * \
    (q)**(2) * (S1)**(2) + (-2 * (ct12 + -1 * ct1 * ct2) * q * S1 * \
    S2 + (-1 + (ct2)**(2)) * (S2)**(2))) + (3/10 * (e)**(2) * (149 + 31 * \
    (e)**(2)) * (-1 + q) * (q)**(-2) * ((1 + q))**(2) * (c2phi1 * (-1 + \
    (ct1)**(2)) * (q)**(2) * (S1)**(2) + c2phi2 * (-1 + (ct2)**(2)) * \
    (S2)**(2)) + (-1/10 * ((1 + -1 * (e)**(2)))**(-1/2) * (-32 + (-532 * \
    (e)**(2) + (859 * (e)**(4) + 130 * (e)**(6)))) * (-1 + ((1 + -1 * \
    (e)**(2)))**(1/2)) * (-5 + 2 * eta) + (-85/6 * (e)**(2) * ((-1 + \
    (e)**(2)))**(-1) * (1 + 17 * eta) + 1/7560 * (e)**(6) * (161643 + \
    (-617988 * eta + 525952 * (eta)**(2))))))))))) * (8 * (34103 + (122949 \
    * eta + 59472 * (eta)**(2))) + (12 * (e)**(2) * (-513446 + (-1041522 * \
    eta + 1032101 * (eta)**(2))) + 3 * (e)**(4) * (2095517 + (-8842605 * eta \
    + 5826072 * (eta)**(2)))))
    
    twoptfive_pnterm_u= 1/141750 * ((1 + -1 * (e)**(2)))**(3/2) * np.pi * (eta)**(14) * \
    (-11059200 * (4159 + 15876 * eta) + (-1843200 * (e)**(2) * (-623013 + \
    904016 * eta) + (50 * (e)**(8) * (-30227745 + 3401956 * eta) + (-9600 * \
    (e)**(6) * (-8437609 + 8101664 * eta) + (-57600 * (e)**(4) * \
    (-25148607 + 24142172 * eta) + (e)**(10) * (114231477 + 30218416 * \
    eta))))))

    three_pnterm_u=  16384 * ((1 + -1 * (e)**(2)))**(3/2) * (eta)**(15) * \
    (16447322263/21829500 + (-54784/525 * np.euler_gamma + (512/15 * \
    (np.pi)**(2) + ((-56198689/34020 + 902/15 * (np.pi)**(2)) * eta \
    + (541/140 * (eta)**(2) + (-1121/81 * (eta)**(3) + ((e)**(2) * ((1 + -1 \
    * (e)**(2)))**(-1) * (1/378000 * (91284763 + -11682688 * ((1 + ((1 + \
    -1 * (e)**(2)))**(1/2)))**(-1)) + ((19505077/5040 + -595/8 * \
    (np.pi)**(2)) * eta + -48569/12 * (eta)**(2))) + ((e)**(6) * \
    (-28913792717/6468000 + (-4922/175 * np.euler_gamma + ((np.pi)**(2) * \
    (46/5 + -369/80 * eta) + (107275139/30240 * eta + (5155951/1512 * \
    (eta)**(2) + -44338/15 * (eta)**(3)))))) + (-1/2661120 * (e)**(8) * \
    (730533171 + (-735596048 * eta + (-58925504 * (eta)**(2) + 332817408 * \
    (eta)**(3)))) + ((e)**(2) * (247611308999/87318000 + (-392048/525 * \
    np.euler_gamma + (-2828420479/680400 * eta + (1070903/315 * (eta)**(2) + \
    (-392945/324 * (eta)**(3) + (np.pi)**(2) * (3664/15 + 477/4 * \
    eta)))))) + (1/1512000 * ((1 + -1 * (e)**(2)))**(-1/2) * (-1 + ((1 + \
    -1 * (e)**(2)))**(1/2)) * (32 * (19954466 + (75 * (-19748 + 861 * \
    (np.pi)**(2)) * eta + -1990800 * (eta)**(2))) + (600 * (e)**(8) * \
    (979185 + (-1020722 * eta + 314888 * (eta)**(2))) + (-60 * (e)**(2) * \
    (57250248 + (35 * (-4387744 + 12669 * (np.pi)**(2)) * eta + \
    39729760 * (eta)**(2))) + (15 * (e)**(4) * (-200752728 + (5 * \
    (-60816544 + 677607 * (np.pi)**(2)) * eta + 53136160 * (eta)**(2))) \
    + 30 * (e)**(6) * (334770238 + (5 * (-85449820 + 7749 * \
    (np.pi)**(2)) * eta + 122099600 * (eta)**(2))))))) + ((e)**(4) * \
    (-93304/175 * np.euler_gamma + ((np.pi)**(2) * (872/5 + -53131/960 * \
    eta) + 1/174636000 * (-709913331003 + (1140975546095 * eta + \
    (1274078796375 * (eta)**(2) + -947869014400 * (eta)**(3)))))) + \
    (1284/175 * (1/4050 * (e)**(2) * (988200 + (-16992900 * (e)**(2) + \
    (153995525 * (e)**(4) + (-840554750 * (e)**(6) + 3569058808 * \
    (e)**(8))))) * np.log(2) + (-243 * ((e)**(2) + (-39/4 * (e)**(4) + \
    (2735/64 * (e)**(6) + (25959/512 * (e)**(8) + -638032239/409600 * \
    (e)**(10))))) * np.log(3) + (-48828125/5184 * ((e)**(6) + (-83/8 * \
    (e)**(8) + 12637/256 * (e)**(10))) * np.log(5) + \
    -4747561509943/33177600 * (e)**(10) * np.log(7)))) + 214/525 * \
    (256 + (1832 * (e)**(2) + (1308 * (e)**(4) + 69 * (e)**(6)))) * \
    np.log((1 + ((1 + -1 * (e)**(2)))**(-1/2)) * ((16 * u * eta + -16 * \
    (e)**(2) * u * eta))**(-1)))))))))))))))

    #pn terms for dedt :') from Klein et al. 2018  arXiv:1801.08542
    zero_pnterm_e= -512/15 * (e)**(2) * ((1 + -1 * (e)**(2)))**(3/2) * (304 + 121 * \
    (e)**(2)) * (eta)**(9)

    one_pnterm_e = 256/315 * ((1 + -1 * (e)**(2)))**(3/2) * (eta)**(11) * (8 * (e)**(2) \
    * (8451 + 28588 * eta) + (12 * (e)**(4) * (-59834 + 54271 * eta) + \
    (e)**(6) * (-125361 + 93184 * eta)))

    oneptfive_pnterm_e = -2048 * ((1 + -1 * (e)**(2)))**(3/2) * (-1/16588800 * (e)**(2) * \
    (-4357324800 + (-6601236480 * (e)**(2) + (-557959680 * (e)**(4) + \
    (-598080 * (e)**(6) + (1161732 * (e)**(8) + 5971 * (e)**(10)))))) * \
    np.pi + (1/45 * (e)**(2) * (19688 + (28256 * (e)**(2) + 2367 * \
    (e)**(4))) * (ct1 * S1 + ct2 * S2) + 1/15 * (e)**(2) * (4344 + (8090 \
    * (e)**(2) + 835 * (e)**(4))) * (q)**(-1) * (ct1 * (q)**(2) * S1 + \
    ct2 * S2))) * (eta)**(12)
    
    two_pnterm_e=-128/945 * ((1 + -1 * (e)**(2)))**(3/2) * (eta)**(13) * (-504 * \
    (q)**(-2) * ((1 + q))**(2) * (80 + (15 * (e)**(6) * (-83 + 74 * q) + \
    (70 * (e)**(4) * (-191 + 170 * q) + 8 * (e)**(2) * (-1023 + 938 * \
    q)))) * ((q)**(2) * (S1)**(2) + (S2)**(2)) + (-252 * (-80 + (15688 * \
    (e)**(2) + (25270 * (e)**(4) + 2355 * (e)**(6)))) * (q)**(-2) * ((1 + \
    q))**(2) * ((q)**(2) * (S1)**(2) + (2 * ct12 * q * S1 * S2 + \
    (S2)**(2))) + (252 * (-80 + (44984 * (e)**(2) + (71790 * (e)**(4) + \
    6705 * (e)**(6)))) * (q)**(-2) * ((1 + q))**(2) * ((q)**(2) * \
    (S1)**(2) + (2 * ct1 * ct2 * q * S1 * S2 + (S2)**(2))) + (504 * \
    (q)**(-2) * ((1 + q))**(2) * (80 + (45 * (e)**(6) * (-75 + 37 * q) + \
    (30 * (e)**(4) * (-1203 + 1190 * q) + 8 * (e)**(2) * (-2809 + 2814 * \
    q)))) * ((ct1)**(2) * (q)**(2) * (S1)**(2) + (ct2)**(2) * (S2)**(2)) \
    + (63 * c2phi * (e)**(2) * (88432 + (161872 * (e)**(2) + 16521 * \
    (e)**(4))) * (q)**(-2) * ((1 + q))**(2) * ((-1 + (ct1)**(2)) * \
    (q)**(2) * (S1)**(2) + (-2 * (ct12 + -1 * ct1 * ct2) * q * S1 * \
    S2 + (-1 + (ct2)**(2)) * (S2)**(2))) + (126 * (e)**(2) * (q)**(-2) * \
    ((1 + q))**(2) * (-44224 + (44208 * q + (16 * (e)**(2) * (-5061 + \
    5056 * q) + (e)**(4) * (-8265 + 8256 * q)))) * (c2phi1 * (-1 + \
    (ct1)**(2)) * (q)**(2) * (S1)**(2) + c2phi2 * (-1 + (ct2)**(2)) * \
    (S2)**(2)) + (-2016 * (e)**(2) * ((1 + -1 * (e)**(2)))**(1/2) * (2672 \
    + (6963 * (e)**(2) + 565 * (e)**(4))) * (-5 + 2 * eta) + (32 * \
    (e)**(2) * (-952397 + (801495 * eta + 284256 * (eta)**(2))) + (6 * \
    (e)**(8) * (1262181 + (-1448284 * eta + 919520 * (eta)**(2))) + (24 * \
    (e)**(4) * (-3113989 + (-3495771 * eta + 4059279 * (eta)**(2))) + 4 * \
    (e)**(6) * (23283055 + (-39171801 * eta + 21405048 * \
    (eta)**(2)))))))))))))
    
    twoptfive_pnterm_e=-8192 * (e)**(2) * ((1 + -1 * (e)**(2)))**(3/2) * np.pi * \
    (eta)**(14) * (1/315 * (-167073 + -610144 * eta) + ((e)**(10) * \
    (3311197679/18579456000 + 62161997121736/2125526925319875 * eta) + \
    (1/16588800 * (e)**(8) * (-45904599 + 6209264 * eta) + (-5/12096 * \
    (e)**(4) * (-12204489 + 11870488 * eta) + (-1/5040 * (e)**(2) * \
    (-29712813 + 44912932 * eta) + -1/2903040 * (e)**(6) * (-652241337 + \
    604953772 * eta))))))

    three_pnterm_e= -16384 * ((1 + -1 * (e)**(2)))**(3/2) * (eta)**(15) * (-1460336/23625 \
    * (-1 + ((1 + -1 * (e)**(2)))**(1/2)) + ((e)**(8) * \
    (120660628321/12936000 + (-20437/175 * np.euler_gamma + ((np.pi)**(2) \
    * (191/5 + -6519/320 * eta) + (-91818931/5040 * eta + (2495471/126 * \
    (eta)**(2) + -11792069/1215 * (eta)**(3)))))) + ((e)**(4) * \
    (64020009407/21829500 + (-9525568/1575 * np.euler_gamma + \
    ((np.pi)**(2) * (89024/45 + -15727/96 * eta) + (770214901/12600 * \
    eta + (-80915371/7560 * (eta)**(2) + -86910509/9720 * (eta)**(3)))))) + \
    ((e)**(10) * (302322169/887040 + (-1921387/5040 * eta + (41179/108 * \
    (eta)**(2) + -386792/1215 * (eta)**(3)))) + (-1/13097700 * (e)**(2) * \
    (-184965635913 + (21896493696 * np.euler_gamma + (-10022243847 * eta + \
    (43127784315 * (eta)**(2) + (3287953900 * (eta)**(3) + -291060 * \
    (np.pi)**(2) * (24608 + 9153 * eta)))))) + (1/756000 * ((1 + -1 * \
    (e)**(2)))**(1/2) * (1500 * (e)**(8) * (560883 + (-579382 * eta + \
    179032 * (eta)**(2))) + (16 * (e)**(2) * (-136278294 + (25 * \
    (-22107964 + 174783 * (np.pi)**(2)) * eta + 114097200 * (eta)**(2))) \
    + (3 * (e)**(4) * (5721323968 + (25 * (-612728912 + 1977717 * \
    (np.pi)**(2)) * eta + 3819138400 * (eta)**(2))) + (e)**(6) * \
    (21285782452 + (25 * (-1059132440 + 52521 * (np.pi)**(2)) * eta + \
    7563561600 * (eta)**(2)))))) + ((e)**(6) * (-4588588/1575 * np.euler_gamma \
    + ((np.pi)**(2) * (42884/45 + -295559/960 * eta) + 1/174636000 * \
    (-3501037251219 + (3387807843805 * eta + (6097745236275 * (eta)**(2) + \
    -4313976836400 * (eta)**(3)))))) + (428/1575 * (e)**(2) * (-1/2700 * \
    (-17647200 + (481982400 * (e)**(2) + (-6453060300 * (e)**(4) + \
    (46781466075 * (e)**(6) + (-262622893260 * (e)**(8) + 1732248932266 * \
    (e)**(10)))))) * np.log(2) + (-6561 * (1 + (-49/4 * (e)**(2) + \
    (4369/64 * (e)**(4) + (214449/512 * (e)**(6) + (-623830739/81920 * \
    (e)**(8) + 76513915569/1638400 * (e)**(10)))))) * np.log(3) + \
    (48828125/1769472 * (e)**(4) * (-27648 + (337536 * (e)**(2) + \
    (-1908084 * (e)**(4) + 6631171 * (e)**(6)))) * np.log(5) + \
    4747561509943/4915200 * (e)**(8) * (-20 + 259 * (e)**(2)) * \
    np.log(7)))) + 107/1575 * (e)**(2) * (24608 + (89024 * (e)**(2) + \
    (42884 * (e)**(4) + 1719 * (e)**(6)))) * np.log((1 + ((1 + -1 * \
    (e)**(2)))**(-1/2)) * ((16 * u * eta + -16 * (e)**(2) * u * \
    eta))**(-1))))))))))

    
    dydt = (zero_pnterm_u* (0 in PNorderrad) *u**9  +(1 in PNorderrad)*one_pnterm_u*u**11\
        + (1.5 in PNorderrad)*oneptfive_pnterm_u *u**12 + (2 in PNorderrad)*two_pnterm_u * u**13 \
        + (2.5 in PNorderrad)*twoptfive_pnterm_u *u**14 + (3 in PNorderrad)*three_pnterm_u *u**15) 
        
    de2dt = (zero_pnterm_e* (0 in PNorderrad) *u**8  +(1 in PNorderrad)*one_pnterm_e*u**10\
        + (1.5 in PNorderrad)*oneptfive_pnterm_e *u**11 + (2 in PNorderrad)*two_pnterm_e * u**12 \
        + (2.5 in PNorderrad)*twoptfive_pnterm_e *u**13 + (3 in PNorderrad)*three_pnterm_e *u**14)#/(2*e)
    #print(de2dt)

    dadt=2*a*(2*dydt*eta*np.sqrt(a)*(1-e**2)**(3/2)-e*de2dt)/(-1+e**2)

    # precession of line of periastron  from Klein et al. 2018  arXiv:1801.08542 [Eq. 17b]
    ddelta_lambdadt = 96 * ((1-e**2))**(3/2) * (u)**(5) * (eta)**(5) * ((1 + 12  * (u)**(2) * (eta)**(2)))**(-1)

    # Integrate in a, not in time
    dtda = 1/dadt
    dLhda = dLhdt*dtda
    dS1hda = dS1hdt*dtda
    dS2hda = dS2hdt*dtda
    ddelta_lambdada = ddelta_lambdadt*dtda

    de2da = de2dt*dtda

    # Pack outputs
    
    return np.concatenate([dLhda, dS1hda, dS2hda,  [ddelta_lambdada], [de2da], [dtda]])


def integrator_orbav(Lhinitial, S1hinitial, S2hinitial, delta_lambda, a ,e, q, chi1, chi2, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3], **odeint_kwargs):
    """
    Integration of the systems of ODEs describing orbit-averaged inspirals.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    Lhinitial: array
        Initial direction of the orbital angular momentum, unit vector.
    S1hinitial: array
        Initial direction of the primary spin, unit vector.
    S2hinitial: array
        Initial direction of the secondary spin, unit vector.
    a: float
       Bianary semi-major axis.
    e : float
       Eccentricty: 0<=e<1. .
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    ODEsolution: array
        Solution of the ODE.
    
    Examples
    --------
    ``ODEsolution = precession.integrator_orbavintegrator_orbav(Lhinitial,S1hinitial,S2hinitial,v,q,chi1,chi2)``
    """

    Lhinitial = np.atleast_2d(Lhinitial).astype(float)
    S1hinitial = np.atleast_2d(S1hinitial).astype(float)
    S2hinitial = np.atleast_2d(S2hinitial).astype(float)
    delta_lambda = np.atleast_1d(delta_lambda).astype(float)
    a = np.atleast_2d(a).astype(float)
    e = np.atleast_1d(e).astype(float)
    q = np.atleast_1d(q).astype(float)
    delta_lambda = np.atleast_1d(delta_lambda).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)

    # Defaults for the integrators, can be changed by the user
    if 'mxstep' not in odeint_kwargs: odeint_kwargs['mxstep']=5000000
    if 'rol' not in odeint_kwargs: odeint_kwargs['rtol']=1e-13
    if 'aol' not in odeint_kwargs: odeint_kwargs['atol']=1e-13
    odeint_kwargs['full_output']=0 # This needs to be forced for compatibility with the rest of the code

    def _compute(Lhinitial, S1hinitial, S2hinitial,delta_lambda, a,e, q, chi1, chi2):

        # I need unit vectors
        assert np.isclose(np.linalg.norm(Lhinitial), 1)
        assert np.isclose(np.linalg.norm(S1hinitial), 1)
        assert np.isclose(np.linalg.norm(S2hinitial), 1)
       

        # Pack inputs
    
        ic = np.concatenate([Lhinitial, S1hinitial, S2hinitial,[delta_lambda], [e], [0]])

        # Compute these quantities here instead of inside the RHS for speed
        m1 = eval_m1(q).item()
        m2 = eval_m2(q).item()
        S1 = eval_S1(q, chi1).item()
        S2 = eval_S2(q, chi2).item()
        eta = eval_eta(q).item()

        # solve_ivp implementation. Didn't really work.
        #ODEsolution = scipy.integrate.solve_ivp(rhs_orbav, (vinitial, vfinal), ic, method='LSODA', t_eval=(vinitial, vfinal), dense_output=True, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula),rtol=1e-12,atol=1e-12)
        #ODEsolution = scipy.integrate.solve_ivp(rhs_orbav, (vinitial, vfinal), ic, t_eval=(vinitial, vfinal), dense_output=True, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula))

        ODEsolution = scipy.integrate.odeint(rhs_orbav, ic, a, args=(q, m1, m2, eta, chi1, chi2, S1, S2, PNorderpre, PNorderrad), **odeint_kwargs)#, printmessg=0,rtol=1e-10,atol=1e-10)#,tcrit=sing)
        return ODEsolution

    ODEsolution = (list(map(_compute, Lhinitial, S1hinitial, S2hinitial, delta_lambda, a, e, q, chi1, chi2)))

    return ODEsolution

def inspiral_orbav(theta1=None, theta2=None, deltaphi=None, Lh=None, S1h=None, S2h=None, delta_lambda=0, deltachi=None, kappa=None, a=None, e=None, uc=None, u=None, chieff=None, q=None, chi1=None, chi2=None, cyclesign=+1, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3,3.5], requested_outputs=None, **odeint_kwargs):
    """
    Perform precession-averaged inspirals. The variables q, chi1, and chi2 must always be provided.
    The integration range must be specified using either (a,e) or (uc,u) (and not both). These need to be arrays with lenght >=1, where e.g. a[0] corresponds to the initial condition and a[1:] corresponds to the location where outputs are returned.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and uc of shape (M,N).
    The initial conditions must be specified in terms of one an only one of the following:
        - Lh, S1h, and S2h
        - theta1,theta2, and deltaphi.
        - deltachi, kappa, chieff, cyclesign.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'].
    The flag enforce allows checking the consistency of the input variables.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    Lh: array, optional (default: None)
        Direction of the orbital angular momentum, unit vector.
    S1h: array, optional (default: None)
        Direction of the primary spin, unit vector.
    S2h: array, optional (default: None)
        Direction of the secondary spin, unit vector.
    deltachi: float, optional (default: None)
        Weighted spin difference.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
    a: float, optional (default: None)
        Binary semi-major axis.
    e: float, optional (default: None)
        Eccentricity 0<=e<1.    
    uc: float, optional (default: None)
        Circualr compactified separation 1/(2L(e=0)).
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: +1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    requested_outputs: list, optional (default: None)
        Set of outputs.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral_orbav(Lh=Lh,S1h=S1h,S2h=S2h,a=a,e=e,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(Lh=Lh,S1h=S1h,S2h=S2h,uc=uc,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,a=a,e=e,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,uc=uc,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(deltachi=deltachi,kappa=kappa,a=a,e=e,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """
    

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, Lh, S1h, S2h, delta_lambda, deltachi, kappa, a, e, uc, u, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if k == 3 or k == 4 or k == 5 or k == 9 or k == 11:  # Lh, S1h, S2h, a or uc
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, Lh, S1h, S2h, delta_lambda, deltachi, kappa, a, e, uc, u, chieff, q, chi1, chi2= inputs

    def _compute(theta1, theta2, deltaphi, Lh, S1h, S2h, delta_lambda, deltachi, kappa, a, e, uc, u, chieff, q, chi1, chi2, cyclesign):

        if q is None or chi1 is None or chi2 is None:
            raise TypeError("Please provide q, chi1, and chi2.")

        if a is not None and uc is None:
            assert np.logical_or(ismonotonic(a, '<='), ismonotonic(a, '>=')), 'a must be monotonic'
            uc = eval_u(a=a, e=tiler(0, a), q=tiler(q, a))  # Convert a in uc to uc 
            
        elif a is None and uc is not None:
            assert np.logical_or(ismonotonic(uc, '<='), ismonotonic(uc, '>=')), 'uc must be monotonic'
            a = eval_a(uc=uc, q=tiler(q, uc))  # Convert uc in a to a 
        else:
            raise TypeError("Please provide either a or uc.")

        # User provides Lh, S1h, and S2h
        if Lh is not None and S1h is not None and S2h is not None and theta1 is None and theta2 is None and deltaphi is None and deltachi is None and kappa is None and chieff is None:
            pass

        # User provides theta1, theta2, and deltaphi.
        elif Lh is None and S1h is None and S2h is None and theta1 is not None and theta2 is not None and deltaphi is not None and deltachi is None and kappa is None and chieff is None:
            Lh, S1h, S2h = angles_to_Lframe(theta1, theta2, deltaphi, a[0] ,e, q, chi1, chi2)
            


        # User provides deltachi, kappa, and chieff.
        elif Lh is None and S1h is None and S2h is None and theta1 is None and theta2 is None and deltaphi is None and deltachi is not None and kappa is not None and chieff is not None:
             #cyclesign=+1 by default
            Lh, S1h, S2h = conserved_to_Lframe(deltachi, kappa, a[0], e, chieff, q, chi1, chi2, cyclesign=cyclesign)
        else:
            raise TypeError("Please provide one and not more of the following: (Lh,S1h,S2h), (theta1,theta2,deltaphi), (deltachi,kappa,chieff).")

        # Make sure vectors are normalized
        Lh = Lh/np.linalg.norm(Lh)
        S1h = S1h/np.linalg.norm(S1h)
        S2h = S2h/np.linalg.norm(S2h)

        

        # Integration
        evaluations = integrator_orbav(Lh, S1h, S2h, delta_lambda, a, e**2, q, chi1, chi2, PNorderpre=PNorderpre, PNorderrad=PNorderrad,**odeint_kwargs)[0].T
        # For solve_ivp implementation
        #evaluations = np.squeeze(ODEsolution.item().sol(v))

        # Returned output is
        # Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, delta_lambda, e, (t)
        Lh = evaluations[0:3, :].T
        S1h = evaluations[3:6, :].T
        S2h = evaluations[6:9, :].T
        delta_lambda = evaluations[9, :].T
        e = np.sqrt((evaluations[10, :].T))
        t = evaluations[11, :].T

        # Renormalize. The normalization is not enforced by the integrator, it is only maintaied within etamerical accuracy.
        #Lh = Lh/np.linalg.norm(Lh)
        #S1h = S1h/np.linalg.norm(S1h)
        #S2h = S2h/np.linalg.norm(S2h)

        S1 = eval_S1(q, chi1)
        S2 = eval_S2(q, chi2)
        L = eval_L(a,e, tiler(q, a))
        Lvec = (L*Lh.T).T
        S1vec = S1*S1h
        S2vec = S2*S2h
        theta1, theta2, deltaphi = vectors_to_angles(Lvec, S1vec, S2vec)
        deltachi, kappa, chieff, cyclesign = vectors_to_conserved(Lvec, S1vec, S2vec, a, e, tiler(q,a), full_output=True)
        u= eval_u(a, e, tiler(q, a))  # Convert a in u to u using the tiler function
        return t, theta1, theta2, deltaphi, Lh, S1h, S2h, delta_lambda , deltachi, kappa, a, e, uc, u, chieff, q, chi1, chi2, cyclesign

    # This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['t', 'theta1', 'theta2', 'deltaphi', 'Lh', 'S1h', 'S2h', 'delta_lambda','deltachi', 'kappa', 'a', 'e','uc','u', 'chieff', 'q', 'chi1', 'chi2', 'cyclesign'])


    if cyclesign ==+1 or cyclesign==-1:
        cyclesign=np.atleast_1d(tiler(cyclesign,q))
    
    # Here I force dtype=object because the outputs have different shapes
    allresults = np.array(list(map(_compute, theta1, theta2, deltaphi, Lh, S1h, S2h, delta_lambda, deltachi, kappa, a, e, uc, u, chieff, q, chi1, chi2, cyclesign)), dtype=object).T

    # Handle the outputs.
    # Return all
    if requested_outputs is None:
        requested_outputs = alloutputs
    # Return only those requested (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs, requested_outputs)

    # Store into a dictionary
    outcome = {}
    for k, v in zip(alloutputs[wantoutputs], allresults[wantoutputs]):
        outcome[k] = np.squeeze(np.stack(v))

        if k == 'q' or k == 'chi1' or k == 'chi2':  # Constants of motion (chieff is not enforced!)
            outcome[k] = np.atleast_1d(outcome[k])
        else:
            outcome[k] = np.atleast_2d(outcome[k])

    return outcome
def test():
    """
    Test function to check if the module is working correctly.e
    """
    print("Precession module is working correctly.")
    # You can add more tests here if needed.

# Ensure tiler, eval_u, and ismonotonic are imported or defined

def inspiral_hybrid(theta1=None, theta2=None, deltaphi=None, deltachi=None, kappa=None, a=None, aswitch=None, e=None, uc=None, ucswitch=None,u=None, chieff=None, q=None, chi1=None, chi2=None, requested_outputs=None,**odeint_kwargs):
    """
    Perform hybrid inspirals, i.e. evolve the binary at large separation with a pression-averaged evolution and at small separation with an orbit-averaged evolution, matching the two. The variables q, chi1, and chi2 must always be provided. The integration range must be specified using either a or uc (and not both); provide also ucswitch and aswitch consistently.
    Either a of uc needs to be arrays with lenght >=1, where e.g. a[0] corresponds to the initial condition and a[1:] corresponds to the location where outputs are returned. It does not work at past time infinity if e !=0.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    The initial conditions must be specified in terms of one an only one of the following:
        - theta1,theta2, and deltaphi (but note that deltaphi is not necessary if integrating from infinite separation).
        - kappa, chieff.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'a','e', 'uc','u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'].
    The flag enforce allows checking the consistency of the input variables.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    deltachi: float, optional (default: None)
        Weighted spin difference.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
    a: float, optional (default: None)
        Binary semi-major axis.
    e: float, optional (default: None)
        Binary eccentricity: 0<=e<1 .
    aswitch: float, optional (default: None)
        Matching separation between the precession- and orbit-averaged chunks.
    eswitch: float, optional (default: None)
        Matching separation between the precession- and orbit-averaged chunks.    
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    uc: float, optional (default: None)
        Compactified circular separation 1/(2L).    
    uswitch: float, optional (default: None)
        Matching compactified separation between the precession- and orbit-averaged chunks.
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    requested_outputs: list, optional (default: None)
        Set of outputs.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral_hybrid(theta1=theta1,theta2=theta2,deltaphi=deltaphi,a=a,e=e,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_hybrid(theta1=theta1,theta2=theta2,deltaphi=deltaphi,uc=uc,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_hybrid(kappa,a=a,e=e,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_hybrid(kappa,uc=uc,u=u,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    # Outputs available in both orbit-averaged and precession-averaged evolutions
    alloutputs = np.array(['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'a','e', 'uc','u','chieff', 'q', 'chi1', 'chi2'])
    if requested_outputs is None:
        requested_outputs = alloutputs
        # Return only those requested (in1d return boolean array)
    wantoutputs = np.intersect1d(alloutputs, requested_outputs)

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, deltachi, kappa, a, aswitch, e, uc, ucswitch, u, chieff, q, chi1, chi2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(tiler(None, np.atleast_1d(q))))
        else:
            if k == 5 or k ==8 :  # Either u or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, deltachi, kappa, a, aswitch, e, uc, ucswitch, u, chieff, q, chi1, chi2 = inputs

    def _compute(theta1, theta2, deltaphi, deltachi, kappa, a, aswitch, e, uc,ucswitch, u, chieff, q, chi1, chi2):
        estart = None  # Ensure estart is always defined
        ## User pass (uc, u0, ucswitch) -> return a,eswictch, uswitch
        
        if a is None and aswitch is None and uc is not None and ucswitch is not None and u is not None:
            a = eval_a(uc=uc, q=tiler(q, uc))
            e0= eval_e(u=u,uc=uc[0], q=q)
            estart=np.copy(e0)
            aswitch = eval_a(uc=ucswitch, q=tiler(q, ucswitch))
            uc0=uc[0]

            def solve(uc, c0, e):
                     return scipy.optimize.brentq(lambda u : implicit(u,uc) - c0, 100*uc/(1-e**2)**0.5, uc, xtol=1e-15)
            c0 = implicit(u,uc[0]) 
            uswitch=solve(ucswitch,c0,e0)
            eswitch=np.sqrt(1-np.float64(ucswitch)**2/np.float64(uswitch)**2)
      

        ## User pass (a, e0, aswitch) -> return eswitch uswitch and  ucswitch
        elif a is not None and aswitch is not None and e is not None and uc is None and ucswitch is None and u is None:
            e0=e
            estart=np.copy(e)
            u = eval_u(a=a[0], e=e,q=q)
            uc0=eval_u(a=a[0],e=0, q=q)
            ucswitch= eval_u(a=aswitch, e=0, q=q)
        else:
             raise TypeError("Please provide either a or uc.")

        def _2_step(uc0, u, ucswitch,e0):
            def solve(uc, c0, e):
                     return scipy.optimize.brentq(lambda u : implicit(u,uc) - c0, 100*uc/(1-e**2)**0.5, uc, xtol=1e-15)
            c0 = implicit(u,uc0) 
            uswitch=solve(ucswitch,c0,e0)
            eswitch=np.sqrt(1-np.float64(ucswitch)**2/np.float64(uswitch)**2)
            return uswitch, eswitch
        
        uswitch, eswitch= _2_step(uc0, u, ucswitch,e0)
        print("uswitch, eswitch", uswitch, eswitch)

        forward = ismonotonic(a, ">=")
        backward = ismonotonic(a, "<=")

        assert np.logical_or(forward, backward), "a must be monotonic"
        assert aswitch > np.min(a) and aswitch < np.max(a), "The switching condition must to be within the range spanned by a."

        alarge = a[a >= aswitch]
        asmall = a[a < aswitch]


        # Integrating forward: precession-averaged first, then orbit-averaged
        if forward:
            inspiral_first = inspiral_precav
            afirst = np.append(alarge, aswitch)
            inspiral_second = inspiral_orbav
            asecond = np.append(aswitch, asmall)

        # Integrating backward: orbit-averaged first, then precession-averaged
        elif backward:
            inspiral_first = inspiral_orbav
            afirst = np.append(asmall, aswitch)
            inspiral_second = inspiral_precav
            asecond = np.append(aswitch, alarge)

        # First chunk of the evolution
        evolution_first = inspiral_first(theta1=theta1, theta2=theta2, deltaphi=deltaphi, deltachi=deltachi, kappa=kappa, a=afirst,e=estart, chieff=chieff, q=q, chi1=chi1, chi2=chi2, requested_outputs=alloutputs,**odeint_kwargs)

        # Second chunk of the evolution
        evolution_second = inspiral_second(theta1=np.squeeze(evolution_first['theta1'])[-1], theta2=np.squeeze(evolution_first['theta2'])[-1], deltaphi=np.squeeze(evolution_first['deltaphi'])[-1], a=asecond,e=eswitch, q=q, chi1=chi1, chi2=chi2, requested_outputs=alloutputs,**odeint_kwargs)

        # Store outputs
        evolution_full = {}
        for k in wantoutputs:
            # Quantities that vary in both the precession-averaged and the orbit-averaged evolution
            if k in ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'a', 'e','uc','u']:
                evolution_full[k] = np.atleast_2d(np.append(evolution_first[k][:, :-1], evolution_second[k][:, 1:]))
            # Quantities that vary only on the orbit-averaged evolution
            if k in ['chieff']:
                if forward:
                    evolution_full[k] = np.atleast_2d(np.append(tiler(evolution_first[k][:], afirst[:-1]), evolution_second[k][:, 1:]))
                elif backward:
                    evolution_full[k] = np.atleast_2d(np.append(evolution_first[k][:, :-1], tiler(evolution_second[k][:], asecond[1:])))
            # Quanties that do not vary
            if k in ['q', 'chi1', 'chi2']:
                evolution_full[k] = evolution_second[k]

        return evolution_full

    allresults = list(map(_compute, theta1, theta2, deltaphi, deltachi, kappa, a, aswitch, e, uc,ucswitch, u, chieff, q, chi1, chi2))
    evolution_full = {}
    for k in allresults[0].keys():
        evolution_full[k] = np.concatenate(list(evolution_full[k] for evolution_full in allresults))

    return evolution_full




all_fun=functions+['eval_a', 'eval_e', 'ddchidt_prefactor', 'vectors_to_conserved', 'inspiral_precav','rhs_orbav', 'integrator_orbav', 'inspiral_orbav','tesat0' ,'inspiral_hybrid','implicit']
__all__ = all_fun
