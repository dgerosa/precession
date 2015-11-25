__author__ = "Davide Gerosa"
__email__ = "d.gerosa@damtp.cam.ac.uk"

import sys,os
import precession  # My module to compute stuff on the precession timescale

# This is a tiny tutorial to learn using my code precession.py


q=0.8
chi1=1.
chi2=1.
print "The main parameters here are q="+str(q)+" chi1="+str(chi1)+" and chi2="+str(chi2)

M,m1,m2,S1,S2 = precession.get_fixed(q,chi1,chi2)
print "The magnitude of the two spins in total mass units (M="+str(M)+") are S1="+str(S1)+" S2="+str(S2)

r=100*M
print "The separation is r="+str(r)

Jmin,Jmax = precession.J_lim(q,S1,S2,r)
xi_min,xi_max=precession.xi_lim(q,S1,S2)
Sso_min,Sso_max=precession.Sso_limits(S1,S2)

print "The geometrical limits in xi,J and S are"
print "\t "+str(xi_min)+" <= xi <= "+str(xi_max)
print "\t "+str(Jmin)+" <= J <= "+str(Jmax)
print "\t "+str(Sso_min)+" <= S <= "+str(Sso_max)

J= (Jmin+Jmax)/2.
print "Let me select a value of J="+str(J)+" within the limits."

St_min,St_max = precession.St_limits(J,q,S1,S2,r)
print "This constraints the range of S to"
print "\t "+str(St_min)+" <= S <= "+str(St_max)

xi_low,xi_up = precession.xi_allowed(J,q,S1,S2,r)
print "The allowed values of xi can be found extrmizing the effective potentials"
print "\t "+str(xi_low)+" <= xi <= "+str(xi_up)

xi=(xi_low+xi_up)/2.
print "Let me select a value of xi="+str(xi)+" within the limits."

Sb_min,Sb_max = precession.Sb_limits(xi,J,q,S1,S2,r)
print "I have a consisten couple (J,xi). S oscillates between S-="+str(Sb_min)+"and S+="+str(Sb_max)

S=(Sb_min+Sb_max)/2.
print "Let me select a value of S="+str(S)+" between S- and S+"

theta1,theta2,deltaphi,theta12 = precession.parametric_angles(S,J,xi,q,S1,S2,r)
print "The angles describing the spin orientations with respect to the orbital plane are"
print "\t theta1="+str(theta1)
print "\t theta2="+str(theta2)
print "\t DeltaPhi="+str(deltaphi)

xi,J,S = precession.from_the_angles(theta1,theta2,deltaphi,q,S1,S2,r)
print "Of course, from the angles one can go back to"
print "\t xi="+str(xi)
print "\t J="+str(J)
print "\t S="+str(S)

tau = precession.precession_period(xi,J,q,S1,S2,r)
print "One can integrate dt/dS to find the precessional period tau="+str(tau)

alpha = precession.alphaz(xi,J,q,S1,S2,r)
print "One can also integrate Omega*dt/dS to find alpha="+str(alpha)

morphology = precession.find_morphology(xi,J,q,S1,S2,r)
if morphology==-1:
    labelm="Librating about DeltaPhi=0"
elif morphology==1:
    labelm="Librating about DeltaPhi=pi"    
elif morphology==0:
    labelm="Circulating"
print "The precessional morphology of this binary is: "+labelm

sys.stdout = os.devnull # Ignore warnings
phase,xi_transit_low,xi_transit_up=precession.phase_xi(J,q,S1,S2,r)
sys.stdout = sys.__stdout__ # Restore warnings
if phase==-1:
    labelp="a single DeltaPhi~pi phase"
elif phase==2:
    labelp="two DeltaPhi~pi phases and a circulating phase"    
elif phase==3:
    labelp="a DeltaPhi~0, a circulating, and a DeltaPhi~pi phase"
print "At this value of J the coexisintg phases are: "+labelp+". Indeed, the current morphology ("+labelm+") is part of those."


r_final=10*M
J_vals = precession.Jofr(xi,J,[r,r_final],q,S1,S2)
print "I can integrate dJ/dL from the initial condition J="+str(J_vals[0])+" specified at r="+str(r)+" down to r="+str(r_final)+"  and I find J="+str(J_vals[-1])

kappa_inf = precession.kappa_backwards(xi,J,r,q,S1,S2)
print "I can also integrate dJ/dL backwards to r=infinity from the same initial conditions finding kappa="+str(kappa_inf)



