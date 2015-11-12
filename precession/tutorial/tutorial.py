__author__ = "Davide Gerosa"
__email__ = "d.gerosa@damtp.cam.ac.uk"

import precession  # My module to compute stuff on the precession timescale

# This is a tiny tutorial to learn using my code precession.py

q=0.8
chi1=1.
chi2=1.
print "The main parameters here are q="+str(q)+" chi1="+str(chi1)+" and chi2="+str(chi2)

M,m1,m2,S1,S2 = precession.get_fixed(q,chi1,chi2)
print "The spins in total mass units are S1="+str(S1)+" S2="+str(S2)

r=100*M
print "The separation is r="+str(r)

Jmin,Jmax = precession.J_lim(q,S1,S2,r)
print "The limits in J are J_min="+str(Jmin)+" J_max="+str(Jmax)

J= (Jmin+Jmax)/2.
print "I selected a value of J="+str(J)+" between J_min and J_max"

St_min,St_max = precession.St_limits(J,q,S1,S2,r)
print "For this J, the limits in S are S_min="+str(St_min)+" S_max="+str(St_max)

xi_low,xi_up = precession.xi_allowed(J,q,S1,S2,r)
print "For this J, the limits in xi are xi_min="+str(xi_low)+" xi_max="+str(xi_up)

xi=(xi_low+xi_up)/2.
print "I selected a value of xi="+str(xi)+" between xi_min and xi_max"

Sb_min,Sb_max = precession.Sb_limits(xi,J,q,S1,S2,r)
print "For this J AND xi, the limits in S are S-="+str(Sb_min)+" S+="+str(Sb_max)

S=(Sb_min+Sb_max)/2.
print "I selected a value of S="+str(S)+" between S- and S+"

theta1,theta2,deltaphi,theta12 = precession.parametric_angles(S,J,xi,q,S1,S2,r)
print "The usual spin angles are theta1="+str(theta1)+" theta2="+str(theta2)+" DeltaPhi="+str(deltaphi)

xi,J,S = precession.from_the_angles(theta1,theta2,deltaphi,q,S1,S2,r)
print "Of course, from the angles I can go back to xi="+str(xi)+" J="+str(J)+" and S="+str(S)

morphology = precession.find_morphology(xi,J,q,S1,S2,r)
if morphology==-1:
    morph="Librating about DeltaPhi=0"
elif morphology==1:
    morph="Librating about DeltaPhi=pi"    
elif morphology==0:
    morph="Circulating"
print "The morphology is: "+str(morph)

tau = precession.precession_period(xi,J,q,S1,S2,r)
print "I can integrate dt/dS to find tau="+str(tau)

alpha = precession.alphaz(xi,J,q,S1,S2,r)
print "I can integrate Omega*dt/dS to find alpha="+str(alpha)

r_final=10*M
J_vals = precession.Jofr(xi,J,[r,r_final],q,S1,S2)
print "I can integrate dJ/dL from the initial condition J="+str(J_vals[0])+" specified at r="+str(r)+" down to r="+str(r_final)+"  and I find J="+str(J_vals[-1])

kappa_inf = precession.kappa_backwards(xi,J,r,q,S1,S2)
print "I can also integrate dJ/dL backwards to r=infinity from the same initial conditions finding kappa="+str(kappa_inf)



