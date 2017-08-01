'''
This submodule provides practical examples to illustrate the main features of
`precession`. Examples are illustrated in 

- *Precession. Dynamics of spinning black-hole binaries with Python.* 
D. Gerosa, M. Kesden. [arXiv:1605.01067](https://arxiv.org/abs/1605.01067)

This submodule has to be loaded separately:

    import precession.test

'''


import sys,os
import precession 
import numpy
import random
from matplotlib import use #Useful when working on SSH
use('Agg') 
from matplotlib import rc #Set plot defaults
font = {'family':'serif','serif':['cmr10'],'weight' : 'medium','size' : 20}
rc('font', **font)
rc('text',usetex=True)
rc('figure',max_open_warning=1000)
import pylab
import matplotlib
import time
import multiprocessing


def minimal():
    '''
    A minimal working example to perform a BH binary inspiral

    **Run using**

        import precession.test
        precession.test.minimal()
    '''

    t0=time.time() 
    q=0.75    # Mass ratio
    chi1=0.5  # Primary's spin magnitude
    chi2=0.95 # Secondary's spin magnitude
    print "Take a BH binary with q=%.2f, chi1=%.2f and chi2=%.2f" %(q,chi1,chi2)
    sep=numpy.logspace(10,1,10) # Output separations
    t1= numpy.pi/3.  # Spin orientations at r_vals[0]
    t2= 2.*numpy.pi/3.
    dp= numpy.pi/4.
    M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2)
    t1v,t2v,dpv=precession.evolve_angles(t1,t2,dp,sep,q,S1,S2)    
    print "Perform BH binary inspiral"
    print "log10(r/M) \t theta1 \t theta2 \t deltaphi"
    for r,t1,t2,dp in zip(numpy.log10(sep),t1v,t2v,dpv):
        print "%.0f \t\t %.3f \t\t %.3f \t\t %.3f" %(r,t1,t2,dp)
    t=time.time()-t0
    print "Executed in %.3fs" %t


def parameter_selection():
    
    '''
    Selection of consistent parameters to describe the BH spin orientations, both at finite and infinitely large separation. Compute some quantities which characterize the spin-precession dynamics, such as morphology, precessional period and resonant angles.
    All quantities are given in total-mass units c=G=M=1.

    **Run using**

        import precession.test
        precession.test.parameter_selection()
    '''

    print "\n *Parameter selection at finite separations*"
    q=0.8   # Must be q<=1. Check documentation for q=1.
    chi1=1. # Must be chi1<=1
    chi2=1. # Must be chi2<=1
    M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2) # Total-mass units M=1
    print "We study a binary with\n\tq=%.3f  m1=%.3f  m2=%.3f\n\tchi1=%.3f  S1=%.3f\n\tchi2=%.3f  S2=%.3f" %(q,m1,m2,chi1,S1,chi2,S2)
    r=100*M # Must be r>10M for PN to be valid
    print "at separation\n\tr=%.3f" %r
    xi_min,xi_max=precession.xi_lim(q,S1,S2)
    Jmin,Jmax=precession.J_lim(q,S1,S2,r)
    Sso_min,Sso_max=precession.Sso_limits(S1,S2)
    print "The geometrical limits on xi,J and S are\n\t%.3f<=xi<=%.3f\n\t%.3f<=J<=%.3f\n\t%.3f<=S<=%.3f" %(xi_min,xi_max,Jmin,Jmax,Sso_min,Sso_max)
    J= (Jmin+Jmax)/2.
    print "We select a value of J\n\tJ=%.3f " %J
    St_min,St_max=precession.St_limits(J,q,S1,S2,r)
    print "This constrains the range of S to\n\t%.3f<=S<=%.3f" %(St_min,St_max)
    xi_low,xi_up=precession.xi_allowed(J,q,S1,S2,r)
    print "The allowed range of xi is\n\t%.3f<=xi<=%.3f" %(xi_low,xi_up)
    xi=(xi_low+xi_up)/2.
    print "We select a value of xi\n\txi=%.3f" %xi
    test=(J>=min(precession.J_allowed(xi,q,S1,S2,r)) and J<=max(precession.J_allowed(xi,q,S1,S2,r)))
    print "Is our couple (xi,J) consistent?", test
    Sb_min,Sb_max=precession.Sb_limits(xi,J,q,S1,S2,r)
    print "S oscillates between\n\tS-=%.3f\n\tS+=%.3f" %(Sb_min,Sb_max)
    S=(Sb_min+Sb_max)/2.
    print "We select a value of S between S- and S+\n\tS=%.3f" %S
    t1,t2,dp,t12=precession.parametric_angles(S,J,xi,q,S1,S2,r)
    print "The angles describing the spin orientations are\n\t(theta1,theta2,DeltaPhi)=(%.3f,%.3f,%.3f)" %(t1,t2,dp)
    xi,J,S = precession.from_the_angles(t1,t2,dp,q,S1,S2,r)
    print "From the angles one can recovery\n\t(xi,J,S)=(%.3f,%.3f,%.3f)" %(xi,J,S)
    
    print "\n *Features of spin precession*"
    t1_dp0,t2_dp0,t1_dp180,t2_dp180=precession.resonant_finder(xi,q,S1,S2,r)
    print "The spin-orbit resonances for these values of J and xi are\n\t(theta1,theta2)=(%.3f,%.3f) for DeltaPhi=0\n\t(theta1,theta2)=(%.3f,%.3f) for DeltaPhi=pi" %(t1_dp0,t2_dp0,t1_dp180,t2_dp180)
    tau = precession.precession_period(xi,J,q,S1,S2,r)
    print "We integrate dt/dS to calculate the precessional period\n\ttau=%.3f" %tau
    alpha = precession.alphaz(xi,J,q,S1,S2,r)
    print "We integrate Omega*dt/dS to find\n\talpha=%.3f" %alpha
    morphology = precession.find_morphology(xi,J,q,S1,S2,r)
    if morphology==-1: labelm="Librating about DeltaPhi=0"
    elif morphology==1: labelm="Librating about DeltaPhi=pi"    
    elif morphology==0: labelm="Circulating"
    print "The precessional morphology is: "+labelm
    sys.stdout = os.devnull # Ignore warnings
    phase,xi_transit_low,xi_transit_up=precession.phase_xi(J,q,S1,S2,r)
    sys.stdout = sys.__stdout__ # Restore warnings
    if phase==-1: labelp="a single DeltaPhi~pi phase"
    elif phase==2: labelp="two DeltaPhi~pi phases, a Circulating phase"    
    elif phase==3: labelp="a DeltaPhi~0, a Circulating, a DeltaPhi~pi phase"
    print "The coexisting phases are: "+labelp
    
    print "\n *Parameter selection at infinitely large separation*"
    print "We study a binary with\n\tq=%.3f  m1=%.3f  m2=%.3f\n\tchi1=%.3f  S1=%.3f\n\tchi2=%.3f  S2=%.3f" %(q,m1,m2,chi1,S1,chi2,S2)
    print "at infinitely large separation"
    kappainf_min,kappainf_max=precession.kappainf_lim(S1,S2)
    print "The geometrical limits on xi and kappa_inf are\n\t%.3f<=xi<=%.3f\n\t %.3f<=kappa_inf<=%.3f" %(xi_min,xi_max,kappainf_min,kappainf_max)
    print "We select a value of xi\n\txi=%.3f" %xi
    kappainf_low,kappainf_up=precession.kappainf_allowed(xi,q,S1,S2)
    print "This constrains the range of kappa_inf to\n\t%.3f<=kappa_inf<=%.3f" %(kappainf_low,kappainf_up)
    kappainf=(kappainf_low+kappainf_up)/2.
    print "We select a value of kappa_inf\n\tkappa_inf=%.3f" %kappainf
    test=(xi>=min(precession.xiinf_allowed(kappainf,q,S1,S2)) and xi<=max(precession.xiinf_allowed(kappainf,q,S1,S2)))
    print "Is our couple (xi,kappa_inf) consistent?", test 
    t1_inf,t2_inf=precession.thetas_inf(xi,kappainf,q,S1,S2)
    print "The asymptotic (constant) values of theta1 and theta2 are\n\t(theta1_inf,theta2_inf)=(%.3f,%.3f)" %(t1_inf,t2_inf)
    xi,kappainf = precession.from_the_angles_inf(t1_inf,t2_inf,q,S1,S2)
    print "From the angles one can recovery\n\t(xi,kappa_inf)=(%.3f,%.3f)" %(xi,kappainf)
    

def spin_angles():
    
    '''
    Binary dynamics on the precessional timescale. The spin angles
    theta1,theta2, DeltaPhi and theta12 are computed and plotted against the
    time variable, which is obtained integrating dS/dt. The morphology is also
    detected as indicated in the legend of the plot. Output is saved in
    ./spin_angles.pdf.

    **Run using**

        import precession.test
        precession.test.spin_angles()
    '''

    fig=pylab.figure(figsize=(6,6))      # Create figure object and axes
    ax_t1=fig.add_axes([0,1.95,0.9,0.5]) # first (top)
    ax_t2=fig.add_axes([0,1.3,0.9,0.5])  # second
    ax_dp=fig.add_axes([0,0.65,0.9,0.5]) # third
    ax_t12=fig.add_axes([0,0,0.9,0.5])   # fourth (bottom)

    q=0.7    # Mass ratio. Must be q<=1.
    chi1=0.6 # Primary spin. Must be chi1<=1
    chi2=1.  # Secondary spin. Must be chi2<=1
    M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2) # Total-mass units M=1
    r=20*M   # Separation. Must be r>10M for PN to be valid
    J=0.94   # Magnitude of J: Jmin<J<Jmax as given by J_lim
    xi_vals=[-0.41,-0.3,-0.22] # Effective spin: xi_low<xi<xi_up as given by xi_allowed

    for xi,color in zip(xi_vals,['blue','green','red']): # Loop over three binaries

        tau = precession.precession_period(xi,J,q,S1,S2,r) # Period
        morphology = precession.find_morphology(xi,J,q,S1,S2,r) # Morphology
        if morphology==-1: labelm="${\\rm L}0$"
        elif morphology==1: labelm="${\\rm L}\\pi$"   
        elif morphology==0: labelm="${\\rm C}$"
        Sb_min,Sb_max=precession.Sb_limits(xi,J,q,S1,S2,r) # Limits in S
        S_vals = numpy.linspace(Sb_min,Sb_max,1000) # Create array, from S- to S+
        S_go=S_vals # First half of the precession cycle: from S- to S+
        t_go=map(lambda x: precession.t_of_S(S_go[0],x, Sb_min,Sb_max,xi,J,q,S1,S2,r,0,sign=-1.),S_go) # Compute time values. Assume t=0 at S-      
        t1_go,t2_go,dp_go,t12_go=zip(*[precession.parametric_angles(S,J,xi,q,S1,S2,r) for S in S_go]) # Compute the angles.
        dp_go=[-dp for dp in dp_go] # DeltaPhi<=0 in the first half of the cycle 
        S_back=S_vals[::-1] # Second half of the precession cycle: from S+ to S-
        t_back=map(lambda x: precession.t_of_S(S_back[0],x, Sb_min,Sb_max, xi,J,q,S1,S2,r,t_go[-1],sign=1.),S_back) # Compute time, start from the last point of the first half t_go[-1]
        t1_back,t2_back,dp_back,t12_back=zip(*[precession.parametric_angles(S,J,xi,q,S1,S2,r) for S in S_back]) # Compute the angles. DeltaPhi>=0 in the second half of the cycle

        for ax,vec_go,vec_back in zip([ax_t1,ax_t2,ax_dp,ax_t12], [t1_go,t2_go,dp_go,t12_go], [t1_back,t2_back,dp_back,t12_back]): # Plot all curves
            ax.plot([t/tau for t in t_go],vec_go,c=color,lw=2,label=labelm)
            ax.plot([t/tau for t in t_back],vec_back,c=color,lw=2)

        # Options for nice plotting
        for ax in [ax_t1,ax_t2,ax_dp,ax_t12]:
            ax.set_xlim(0,1)
            ax.set_xlabel("$t/\\tau$")
            ax.set_xticks(numpy.linspace(0,1,5))
        for ax in [ax_t1,ax_t2,ax_t12]:
            ax.set_ylim(0,numpy.pi)
            ax.set_yticks(numpy.linspace(0,numpy.pi,5))
            ax.set_yticklabels(["$0$","$\\pi/4$","$\\pi/2$","$3\\pi/4$","$\\pi$"])
        ax_dp.set_ylim(-numpy.pi,numpy.pi)
        ax_dp.set_yticks(numpy.linspace(-numpy.pi,numpy.pi,5))
        ax_dp.set_yticklabels(["$-\\pi$","$-\\pi/2$","$0$","$\\pi/2$","$\\pi$"])
        ax_t1.set_ylabel("$\\theta_1$")
        ax_t2.set_ylabel("$\\theta_2$")
        ax_t12.set_ylabel("$\\theta_{12}$")
        ax_dp.set_ylabel("$\\Delta\\Phi$")
        ax_t1.legend(loc='lower right',fontsize=18) # Fill the legend with the precessional morphology

    fig.savefig("spin_angles.pdf",bbox_inches='tight') # Save pdf file


def phase_resampling():
    
    '''
    Precessional phase resampling. The magnidute of the total spin S is sampled
    according to |dS/dt|^-1, which correspond to a flat distribution in t(S).
    Output is saved in ./phase_resampling.pdf and data stored in
    `precession.storedir'/phase_resampling_.dat


    **Run using**

        import precession.test
        precession.test.phase_resampling()
    '''

    fig=pylab.figure(figsize=(6,6))      #Create figure object and axes
    ax_tS=fig.add_axes([0,0,0.6,0.6])    #bottom-left
    ax_td=fig.add_axes([0.65,0,0.3,0.6]) #bottom-right
    ax_Sd=fig.add_axes([0,0.65,0.6,0.3]) #top-left

    q=0.5    # Mass ratio. Must be q<=1.
    chi1=0.3 # Primary spin. Must be chi1<=1
    chi2=0.9 # Secondary spin. Must be chi2<=1
    M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2) # Total-mass units M=1
    r=200.*M # Separation. Must be r>10M for PN to be valid
    J=3.14   # Magnitude of J: Jmin<J<Jmax as given by J_lim
    xi=-0.01 # Effective spin: xi_low<xi<xi_up as given by xi_allowed
    Sb_min,Sb_max=precession.Sb_limits(xi,J,q,S1,S2,r) # Limits in S
    tau=precession.precession_period(xi,J,q,S1,S2,r)   # Precessional period
    d=2000   # Size of the statistical sample

    precession.make_temp() # Create store directory, if necessary
    filename=precession.storedir+"/phase_resampling.dat" # Output file name
    if not os.path.isfile(filename): # Compute and store data if not present
        out=open(filename,"w")
        out.write("# q chi1 chi2 r J xi d\n") # Write header
        out.write( "# "+' '.join([str(x) for x in (q,chi1,chi2,r,J,xi,d)])+"\n")

        # S and t values for the S(t) plot
        S_vals=numpy.linspace(Sb_min,Sb_max,d)
        t_vals=numpy.array([abs(precession.t_of_S(Sb_min,S,Sb_min,Sb_max,xi,J,q,S1,S2,r)) for S in S_vals])
        # Sample values of S from |dt/dS|. Distribution should be flat in t.
        S_sample=numpy.array([precession.samplingS(xi,J,q,S1,S2,r) for i in range(d)])
        t_sample=numpy.array([abs(precession.t_of_S(Sb_min,S,Sb_min,Sb_max,xi,J,q,S1,S2,r)) for S in S_sample])
        # Continuous distributions (normalized)
        S_distr=numpy.array([2.*abs(precession.dtdS(S,xi,J,q,S1,S2,r)/tau) for S in S_vals])
        t_distr=numpy.array([2./tau for t in t_vals])

        out.write("# S_vals t_vals S_sample t_sample S_distr t_distr\n")
        for Sv,tv,Ss,ts,Sd,td in zip(S_vals,t_vals,S_sample,t_sample,S_distr,t_distr):
            out.write(' '.join([str(x) for x in (Sv,tv,Ss,ts,Sd,td)])+"\n")
        out.close()
    else:  # Read
        S_vals,t_vals,S_sample,t_sample,S_distr,t_distr=numpy.loadtxt(filename,unpack=True)

    # Rescale all time values by 10^-6, for nicer plotting
    tau*=1e-6; t_vals*=1e-6; t_sample*=1e-6; t_distr/=1e-6

    ax_tS.plot(S_vals,t_vals,c='blue',lw=2)  # S(t) curve
    ax_td.plot(t_distr,t_vals,lw=2.,c='red') # Continous distribution P(t)
    ax_Sd.plot(S_vals,S_distr,lw=2.,c='red') # Continous distribution P(S)
    ax_td.hist(t_sample,bins=60,range=(0,tau/2.),normed=True,histtype='stepfilled', color="blue",alpha=0.4,orientation="horizontal") # Histogram P(t)
    ax_Sd.hist(S_sample,bins=60,range=(Sb_min,Sb_max),normed=True,histtype='stepfilled', color="blue",alpha=0.4) # Histogram P(S)

    # Options for nice plotting
    ax_tS.set_xlim(Sb_min,Sb_max)
    ax_tS.set_ylim(0,tau/2.)
    ax_tS.set_xlabel("$S/M^2$")
    ax_tS.set_ylabel("$t/(10^6 M)$")
    ax_td.set_xlim(0,0.5)  
    ax_td.set_ylim(0,tau/2.)
    ax_td.set_xlabel("$P(t)$")
    ax_td.set_yticklabels([])
    ax_Sd.set_xlim(Sb_min,Sb_max)
    ax_Sd.set_ylim(0,20)  
    ax_Sd.set_xticklabels([])
    ax_Sd.set_ylabel("$P(S)$")

    fig.savefig("phase_resampling.pdf",bbox_inches='tight') # Save pdf file


def PNwrappers():
    
    '''
    Wrappers of the PN integrators. Here we show how to perform orbit-averaged,
    precession-averaged and hybrid PN inspirals using the various wrappers
    implemented in the code. We also show how to estimate the final mass, spin
    and recoil of the BH remnant following a merger.


    **Run using**

        import precession.test
        precession.test.PNwrappers()
    '''

    q=0.9      # Mass ratio. Must be q<=1.
    chi1=0.5   # Primary spin. Must be chi1<=1
    chi2=0.5   # Secondary spin. Must be chi2<=1
    print "We study a binary with\n\tq=%.3f, chi1=%.3f, chi2=%.3f" %(q,chi1,chi2)
    M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2) # Total-mass units M=1
    ri=1000*M  # Initial separation.
    rf=10.*M   # Final separation.
    rt=100.*M  # Intermediate separation for hybrid evolution.
    r_vals=numpy.logspace(numpy.log10(ri),numpy.log10(rf),10) # Output requested
    t1i=numpy.pi/4.; t2i=numpy.pi/4.; dpi=numpy.pi/4. # Initial configuration
    xii,Ji,Si=precession.from_the_angles(t1i,t2i,dpi,q,S1,S2,ri)
    print "Configuration at ri=%.0f\n\t(xi,J,S)=(%.3f,%.3f,%.3f)\n\t(theta1,theta2,deltaphi)=(%.3f,%.3f,%.3f)" %(ri,xii,Ji,Si,t1i,t2i,dpi)

    print " *Orbit-averaged evolution*"
    print "Evolution ri=%.0f --> rf=%.0f" %(ri,rf)
    Jf,xif,Sf=precession.orbit_averaged(Ji,xii,Si,r_vals,q,S1,S2)
    print "\t(xi,J,S)=(%.3f,%.3f,%.3f)" %(xif[-1],Jf[-1],Sf[-1])
    t1f,t2f,dpf=precession.orbit_angles(t1i,t2i,dpi,r_vals,q,S1,S2)
    print "\t(theta1,theta2,deltaphi)=(%.3f,%.3f,%.3f)" %(t1f[-1],t2f[-1],dpf[-1])
    Jvec,Lvec,S1vec,S2vec,Svec=precession.Jframe_projection(xii,Si,Ji,q,S1,S2,ri)
    Lxi,Lyi,Lzi=Lvec; S1xi,S1yi,S1zi=S1vec; S2xi,S2yi,S2zi=S2vec  
    Lx,Ly,Lz,S1x,S1y,S1z,S2x,S2y,S2z=precession.orbit_vectors(Lxi,Lyi,Lzi,S1xi,S1yi,S1zi,S2xi,S2yi,S2zi,r_vals,q)
    print "\t(Lx,Ly,Lz)=(%.3f,%.3f,%.3f)\n\t(S1x,S1y,S1z)=(%.3f,%.3f,%.3f)\n\t(S2x,S2y,S2z)=(%.3f,%.3f,%.3f)" %(Lx[-1],Ly[-1],Lz[-1],S1x[-1],S1y[-1],S1z[-1],S2x[-1],S2y[-1],S2z[-1])
    
    print " *Precession-averaged evolution*"  
    print "Evolution ri=%.0f --> rf=%.0f" %(ri,rf)
    Jf=precession.evolve_J(xii,Ji,r_vals,q,S1,S2)
    print "\t(xi,J,S)=(%.3f,%.3f,-)" %(xii,Jf[-1])
    t1f,t2f,dpf=precession.evolve_angles(t1i,t2i,dpi,r_vals,q,S1,S2)
    print "\t(theta1,theta2,deltaphi)=(%.3f,%.3f,%.3f)" %(t1f[-1],t2f[-1],dpf[-1])
    print "Evolution ri=%.0f --> infinity" %ri
    kappainf=precession.evolve_J_backwards(xii,Jf[-1],rf,q,S1,S2)
    print "\tkappainf=%.3f" %kappainf    
    Jf=precession.evolve_J_infinity(xii,kappainf,r_vals,q,S1,S2)
    print "Evolution infinity --> rf=%.0f" %rf 
    print "\tJ=%.3f" %Jf[-1] 

    print " *Hybrid evolution*"  
    print "Prec.Av. infinity --> rt=%.0f & Orb.Av. rt=%.0f --> rf=%.0f" %(rt,rt,rf)
    t1f,t2f,dpf=precession.hybrid(xii,kappainf,r_vals,q,S1,S2,rt)
    print "\t(theta1,theta2,deltaphi)=(%.3f,%.3f,%.3f)" %(t1f[-1],t2f[-1],dpf[-1])
    
    print " *Properties of the BH remnant*"  
    Mfin=precession.finalmass(t1f[-1],t2f[-1],dpf[-1],q,S1,S2)    
    print "\tM_f=%.3f" %Mfin
    chifin=precession.finalspin(t1f[-1],t2f[-1],dpf[-1],q,S1,S2)
    print "\tchi_f=%.3f, S_f=%.3f" %(chifin,chifin*Mfin**2)
    vkick=precession.finalkick(t1f[-1],t2f[-1],dpf[-1],q,S1,S2)
    print "\tvkick=%.5f" %(vkick) # Geometrical units c=1


def compare_evolutions():
    
    '''
    Compare precession averaged and orbit averaged integrations. Plot the
    evolution of xi, J, S and their relative differences between the two
    approaches. Since precession-averaged estimates of S require a random
    sampling, this plot will look different every time this routine is executed.
    Output is saved in ./spin_angles.pdf.
    
    **Run using**

        import precession.test
        precession.test.compare_evolutions()
    '''

    fig=pylab.figure(figsize=(6,6)) # Create figure object and axes
    L,Ws,Wm,G=0.85,0.15,0.3,0.03    # Sizes
    ax_Sd=fig.add_axes([0,0,L,Ws])              # bottom-small    
    ax_S=fig.add_axes([0,Ws,L,Wm])              # bottom-main
    ax_Jd=fig.add_axes([0,Ws+Wm+G,L,Ws])        # middle-small
    ax_J=fig.add_axes([0,Ws+Ws+Wm+G,L,Wm])      # middle-main
    ax_xid=fig.add_axes([0,2*(Ws+Wm+G),L,Ws])   # top-small
    ax_xi=fig.add_axes([0,Ws+2*(Ws+Wm+G),L,Wm]) # top-main

    q=0.8      # Mass ratio. Must be q<=1.
    chi1=0.6   # Primary spin. Must be chi1<=1
    chi2=1.    # Secondary spin. Must be chi2<=1
    M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2) # Total-mass units M=1
    ri=100.*M  # Initial separation.
    rf=10.*M   # Final separation.
    r_vals=numpy.linspace(ri,rf,1001) # Output requested
    Ji=2.24    # Magnitude of J: Jmin<J<Jmax as given by J_lim
    xi=-0.5    # Effective spin: xi_low<xi<xi_up as given by xi_allowed

    Jf_P=precession.evolve_J(xi,Ji,r_vals,q,S1,S2) # Pr.av. integration
    Sf_P=[precession.samplingS(xi,J,q,S1,S2,r) for J,r in zip(Jf_P[0::10],r_vals[0::10])] # Resample S (reduce output for clarity)
    Sb_min,Sb_max= zip(*[precession.Sb_limits(xi,J,q,S1,S2,r) for J,r in zip(Jf_P,r_vals)]) # Envelopes
    S=numpy.average([precession.Sb_limits(xi,Ji,q,S1,S2,ri)]) # Initialize S
    Jf_O,xif_O,Sf_O=precession.orbit_averaged(Ji,xi,S,r_vals,q,S1,S2) # Orb.av. integration

    Pcol,Ocol,Dcol='blue','red','green'
    Pst,Ost='solid','dashed'
    ax_xi.axhline(xi,c=Pcol,ls=Pst,lw=2)         # Plot xi, pr.av. (constant)
    ax_xi.plot(r_vals,xif_O,c=Ocol,ls=Ost,lw=2)  # Plot xi, orbit averaged
    ax_xid.plot(r_vals,(xi-xif_O)/xi*1e11,c=Dcol,lw=2) # Plot xi deviations (rescaled)
    ax_J.plot(r_vals,Jf_P,c=Pcol,ls=Pst,lw=2)    # Plot J, pr.av.
    ax_J.plot(r_vals,Jf_O,c=Ocol,ls=Ost,lw=2)    # Plot J, orb.av
    ax_Jd.plot(r_vals,(Jf_P-Jf_O)/Jf_O*1e3,c=Dcol,lw=2) # Plot J deviations (rescaled)
    ax_S.scatter(r_vals[0::10],Sf_P,facecolor='none',edgecolor=Pcol) # Plot S, pr.av. (resampled)
    ax_S.plot(r_vals,Sb_min,c=Pcol,ls=Pst,lw=2)  # Plot S, pr.av. (envelopes)
    ax_S.plot(r_vals,Sb_max,c=Pcol,ls=Pst,lw=2)  # Plot S, pr.av. (envelopes)
    ax_S.plot(r_vals,Sf_O,c=Ocol,ls=Ost,lw=2)    # Plot S, orb.av (evolved)
    ax_Sd.plot(r_vals[0::10],(Sf_P-Sf_O[0::10])/Sf_O[0::10],c=Dcol,lw=2) # Plot S deviations

    # Options for nice plotting
    for ax in [ax_xi,ax_xid,ax_J,ax_Jd,ax_S,ax_Sd]:
        ax.set_xlim(ri,rf)
        ax.yaxis.set_label_coords(-0.16, 0.5)
        ax.spines['left'].set_lw(1.5)
        ax.spines['right'].set_lw(1.5)
    for ax in [ax_xi,ax_J,ax_S]:
        ax.spines['top'].set_lw(1.5)
    for ax in [ax_xid,ax_Jd,ax_Sd]:
        ax.axhline(0,c='black',ls='dotted')
        ax.spines['bottom'].set_lw(1.5)
    for ax in [ax_xid,ax_J,ax_Jd,ax_S]: ax.set_xticklabels([])
    ax_xi.set_ylim(-0.55,-0.45)
    ax_J.set_ylim(0.4,2.3)
    ax_S.set_ylim(0.24,0.41)
    ax_xid.set_ylim(-0.2,1.2)
    ax_Jd.set_ylim(-3,5.5)
    ax_Sd.set_ylim(-0.7,0.7)
    ax_xid.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax_Jd.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax_S.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax_Sd.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax_xi.xaxis.set_ticks_position('top')
    ax_xi.xaxis.set_label_position('top')
    ax_Sd.set_xlabel("$r/M$")
    ax_xi.set_xlabel("$r/M$")
    ax_xi.set_ylabel("$\\xi$")
    ax_J.set_ylabel("$J/M^2$")
    ax_S.set_ylabel("$S/M^2$")
    ax_xid.set_ylabel("$\\Delta\\xi/\\xi \;[10^{-11}]$")
    ax_Jd.set_ylabel("$\\Delta J/J \;[10^{-3}]$")
    ax_Sd.set_ylabel("$\\Delta S / S$")

    fig.savefig("compare_evolutions.pdf",bbox_inches='tight') # Save pdf file


def timing():
    
    '''
    This examples compare the numerical performance of `precession.orbit_angles`
    and `precession.evolve_angles`. Computation is performed twice, first using
    all the available CPUs and then explicitely disabling the code
    parallelization.
    
    **Run using**

        import precession.test
        precession.test.timing()
    '''

    BHsample=[] #  Construct a sample of BH binaries
    N=100
    for i in range(N):
        q=random.uniform(0,1)
        chi1=random.uniform(0,1)
        chi2=random.uniform(0,1)
        M,m1,m2,S1,S2=precession.get_fixed(q,chi1,chi2)
        t1=random.uniform(0,numpy.pi)
        t2=random.uniform(0,numpy.pi)
        dp=random.uniform(0,2.*numpy.pi)
        BHsample.append([q,S1,S2,t1,t2,dp])
    q_vals,S1_vals,S2_vals,t1i_vals,t2i_vals,dpi_vals=zip(*BHsample) # Traspose python list

    ri=1e4*M      # Initial separation
    rf=10*M        # Final separation
    r_vals=[ri,rf] # Intermediate output separations not needed here

    print " *Integrating a sample of N=%.0f BH binaries from ri=%.0f to rf=%.0f using %.0f CPUs*" %(N,ri,rf,multiprocessing.cpu_count()) # Parallel computation used by default
    t0=time.time() 
    precession.orbit_angles(t1i_vals,t2i_vals,dpi_vals,r_vals,q_vals,S1_vals,S2_vals)  
    t=time.time()-t0
    print "Orbit-averaged: parallel integrations\n\t total time t=%.3fs\n\t time per binary t/N=%.3fs" %(t,t/N)
    t0=time.time()
    precession.evolve_angles(t1i_vals,t2i_vals,dpi_vals,r_vals,q_vals,S1_vals,S2_vals)    
    t=time.time()-t0
    print "Precession-averaged: parallel integrations\n\t total time t=%.3fs\n\t time per binary t/N=%.3fs" %(t,t/N)

    precession.empty_temp() # Remove previous checkpoints
    precession.CPUs=1       # Force serial computation
    print " *Integrating a sample of N=%.0f BH binaries from ri=%.0f to rf=%.0f using %.0f CPU*" %(len(BHsample),ri,rf,precession.CPUs)
    t0=time.time()
    precession.orbit_angles(t1i_vals,t2i_vals,dpi_vals,r_vals,q_vals,S1_vals,S2_vals)  
    t=time.time()-t0
    print "Orbit-averaged: serial integrations\n\t total time t=%.3fs\n\t time per binary t/N=%.3fs" %(t,t/N)
    t0=time.time()
    precession.evolve_angles(t1i_vals,t2i_vals,dpi_vals,r_vals,q_vals,S1_vals,S2_vals)    
    t=time.time()-t0
    print "Precession-averaged: serial integrations\n\t total time t=%.3fs\n\t time per binary t/N=%.3fs" %(t,t/N)
    precession.empty_temp() # Remove previous checkpoints
    
    
def all():
    '''
    Run all tests in this submodule
    
    **Run using**

        import precession.test
        precession.test.all()
    '''
    
    print "\n**** Execution precession.test.minimal\n"
    minimal()
    print "\n**** Execution precession.test.parameter_selection\n"
    parameter_selection()
    print "\n**** Execution precession.test.spin_angles\n"
    spin_angles()
    print "\n**** Execution precession.test.phase_resampling\n"
    phase_resampling()
    print "\n**** Execution precession.test.PNwrappers\n"
    PNwrappers()
    print "\n**** Execution precession.test.compare_evolutions\n"
    compare_evolutions()
    print "\n**** Execution precession.test.timing\n"
    timing()






