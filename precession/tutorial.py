'''
# DYNAMICS OF PRECESSING BLACK-HOLE BINARIES
precession.tutorial is a submodule containing various tests to illustate the features of precession.

PUT MORE MORE

'''

import sys,os
import precession  # My module to compute stuff on the precession timescale

__author__ = precession.__author__
__email__ = precession.__email__
__copyright__ = precession.__copyright__
__license__ = precession.__license__
__version__ = precession.__version__


def parameter_selection():
    '''
    Selection of consistent parameters to describe the BH spin orientations, both at finite and infinitely large separations. Compute some quantities which characterize the spin-precession dynamics, such as morphology, precessional period and resonant angles, 

    **Run using**

        import precession.tutorial
        precession.tutorial.parameter_selection()
    '''
    print parameter_selection.__doc__

    print "\n *Parameter selection at finite separation*"
    q=0.8   # Must be q<=1. Check documentation for q=1.
    chi1=1. # Must be chi1<=1
    chi2=1. # Must be chi2<=1
    print "We study a binary with:"
    print "\t mass ratio q= %.2f" %q
    print "\t dimensionless spins chi1=%.2f and chi2=%.2f" %(chi1,chi2)
    M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2) # Total-mass units M=1
    print "\t spin magnitudes S1=%.2f and S2=%.2f" %(S1,S2)
    r=100*M # Must be r>10M for PN to be valid
    print "\t separation is r=%.2f" %r
    xi_min,xi_max=precession.xi_lim(q,S1,S2)
    Jmin,Jmax=precession.J_lim(q,S1,S2,r)
    Sso_min,Sso_max=precession.Sso_limits(S1,S2)
    print "The geometrical limits on xi,J and S are"
    print "\t %.2f <= xi <= %.2f" %(xi_min,xi_max)
    print "\t %.2f <= J <= %.2f" %(Jmin,Jmax)
    print "\t %.2f <= S <= %.2f" %(Sso_min,Sso_max)
    J= (Jmin+Jmax)/2.
    print "We select a value of J=%.2f within the limits." %J
    St_min,St_max=precession.St_limits(J,q,S1,S2,r)
    print "This constraints the range of S to"
    print "\t %.2f <= S <= %.2f" %(St_min,St_max)
    xi_low,xi_up=precession.xi_allowed(J,q,S1,S2,r)
    print "The allowed values of xi can be found extremizing the effective potentials"
    print "\t %.2f <= xi <= %.2f" %(xi_low,xi_up)
    xi=(xi_low+xi_up)/2.
    print "We select a value of xi=%.2f within the limits." %xi
    test=(J>=min(precession.J_allowed(xi,q,S1,S2,r)) and J<=max(precession.J_allowed(xi,q,S1,S2,r)))
    print "Is our couple (xi,J) consistent?", test
    Sb_min,Sb_max=precession.Sb_limits(xi,J,q,S1,S2,r)
    print "S oscillates between S-=%.2f and S+=%.2f" %(Sb_min,Sb_max)
    S=(Sb_min+Sb_max)/2.
    print "We select a value of S=%.2f between S- and S+" %S
    t1,t2,dp,t12=precession.parametric_angles(S,J,xi,q,S1,S2,r)
    print "The angles describing the spin orientations are"
    print "\t (theta1,theta2,DeltaPhi)=(%.2f,%.2f,%.2f)" %(t1,t2,dp)
    xi,J,S = precession.from_the_angles(t1,t2,dp,q,S1,S2,r)
    print "From the angles one can go back to"
    print "\t (xi,J,S)=(%.2f,%.2f,%.2f)" %(xi,J,S)
    
    print "\n *Features of spin precession*"
    t1_dp0,t2_dp0,t1_dp180,t2_dp180=precession.resonant_finder(xi,q,S1,S2,r)
    print "The spin-orbit resonances are located at"
    print "\t (theta1,theta2)=("+str(t1_dp0)+","+str(t2_dp0)+") for DeltaPhi=0"
    print "\t (theta1,theta2)=("+str(t1_dp180)+","+str(t2_dp180)+") for DeltaPhi=pi"
    tau = precession.precession_period(xi,J,q,S1,S2,r)
    print "We integrate dt/dS to calculate the precessional period tau="+str(tau)
    alpha = precession.alphaz(xi,J,q,S1,S2,r)
    print "We integrate Omega*dt/dS to find alpha="+str(alpha)
    morphology = precession.find_morphology(xi,J,q,S1,S2,r)
    if morphology==-1: labelm="Librating about DeltaPhi=0"
    elif morphology==1: labelm="Librating about DeltaPhi=pi"    
    elif morphology==0: labelm="Circulating"
    print "The precessional morphology of this binary is: "+labelm
    sys.stdout = os.devnull # Ignore warnings
    phase,xi_transit_low,xi_transit_up=precession.phase_xi(J,q,S1,S2,r)
    sys.stdout = sys.__stdout__ # Restore warnings
    if phase==-1: labelp="a single DeltaPhi~pi phase"
    elif phase==2: labelp="two DeltaPhi~pi phases and a Circulating phase"    
    elif phase==3: labelp="a DeltaPhi~0, a Circulating, and a DeltaPhi~pi phase"
    print "The coexisintg phases are: "+labelp
    print "Indeed, the current morphology ("+labelm+") is part of those."
    
    print "\n *Parameter selection at infinitely separation*"
    print "We study a binary with:"
    print "\t mass ratio q="+str(q)
    print "\t dimensionless spins chi1="+str(chi1)+" and chi2="+str(chi2)
    print "\t at infinitely large separation"
    kappainf_min,kappainf_max=precession.kappainf_lim(S1,S2)
    print "The geometrical limits on xi and kappa_inf are"    
    print "\t "+str(xi_min)+" <= xi <= "+str(xi_max)
    print "\t "+str(kappainf_min)+" <= kappa_inf <= "+str(kappainf_max)
    print "We select a value of xi="+str(xi)+" within the limits."
    kappainf_low,kappainf_up=precession.kappainf_allowed(xi,q,S1,S2)
    print "This constraints the range of kappa_inf to"
    print "\t "+str(kappainf_low)+" <= kappa_inf <= "+str(kappainf_up)
    kappainf=(kappainf_low+kappainf_up)/2.
    print "We select a value of kappa_inf="+str(kappainf)+" within the limits."
    test=(xi>=min(precession.xiinf_allowed(kappainf,q,S1,S2)) and xi<=max(precession.xiinf_allowed(kappainf,q,S1,S2)))
    print "Is our couple (xi,kappa_inf) consistent?", test 
    t1_inf,t2_inf=precession.thetas_inf(xi,kappainf,q,S1,S2)
    print "The asymptotic (constant) values of theta1 and theta2 are"
    print "\t (theta1_inf,theta2_inf)=("+str(t1_inf)+","+str(t2_inf)+") for DeltaPhi=0"
    xi,kappainf = precession.from_the_angles_inf(t1_inf,t2_inf,q,S1,S2)
    print "From the angles one can go back to"
    print "\t (xi,kappa_inf)=("+str(xi)+","+str(kappainf)+")"
   
   
#parameter_selection()