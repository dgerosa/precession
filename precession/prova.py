import numpy as np
import precession as circ
print(circ.eval_L(r=10,q=0.8))
import precession.eccentricity as ecc
print(ecc.eval_L(a=10,e=0.1,q=0.8). circ.eval_L(r=10,q=0.8))
a=np.linspace(30,10,50)
e=0
q=0.9
t1,t2,dw= circ.isotropic_angles(N=1)
chi1=0.1
chi2=0.2

Lh, S1h, S2h=ecc.angles_to_L_newframe(t1, t2, dw, a[0],e, q, chi1, chi2)
Lh = Lh/np.linalg.norm(Lh)
S1h = S1h/np.linalg.norm(S1h)
S2h = S2h/np.linalg.norm(S2h)

