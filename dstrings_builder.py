'''
This is a semi-automatic docstrings builder for the precession code.

Usage:
python dstrings_builder.py <name of function>

For each function docstrings, the developer needs to provide the intro blurb and the "Call" line. This code will then try to fill the "Parameters" and "Returns" description.
'''



import sys
import precession
import pyperclip

fun = sys.argv[1]

def descr(varname,vardef=None):

    # This is a lookup table
    lookup={}
    lookup['MISSING']=["COULD NOT BUILD","FILL MANUALLY"]
    lookup['q']=["float","Mass ratio: 0<=q<=1"]
    lookup['m1']=["float","Mass of the primary (heavier) black hole"]
    lookup['m2']=["float","Mass of the secondary (lighter) black hole"]
    lookup['chi1']=["float","Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1"]
    lookup['chi2']=["float","Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1"]
    lookup['r']=["float","Binary separation"]
    lookup['L']=["float","Magnitude of the Newtonian orbital angular momentum"]
    lookup['S1']=["float","Magnitude of the primary spin"]
    lookup['S2']=["float","Magnitude of the secondary spin"]
    lookup['xi']=["float","Effective spin"]
    lookup['J']=["float","Magnitude of the total angular momentum"]
    lookup['S']=["float","Magnitude of the total spin"]
    lookup['Ssq']=["float","Squared magnitude of the total spin"]

    lookup['kappa']=["float","Regularized angular momentum (J^2-L^2)/(2L)"]
    lookup['u']=["float","Compactified separation 1/(2L)"]
    lookup['varphi']=["float","Generalized nutation coordinate (Eq 9 in arxiv:1506.03492)."]
    lookup['sign']=["integer","Sign, either +1 or -1"]
    lookup['theta1']=["float","Angle between orbital angular momentum and primary spin"]
    lookup['theta2']=["float","Angle between orbital angular momentum and secondary spin"]
    lookup['theta12']=["float","Angle between the two spins"]
    lookup['deltaphi']=["float","Angle between the projections of the two spins onto the orbital plane"]

    lookup['costheta1']=["float","Cosine of the angle between orbital angular momentum and primary spin"]
    lookup['costheta2']=["float","Cosine of the angle between orbital angular momentum and secondary spin"]
    lookup['costheta12']=["float","Cosine of the angle between the two spins"]
    lookup['cosdeltaphi']=["float","Cosine of the angle between the projections of the two spins onto the orbital plane"]


    lookup['theta1inf']=["float","Asymptotic value of the angle between orbital angular momentum and primary spin"]
    lookup['theta2inf']=["float","Asymptotic value of the angle between orbital angular momentum and secondary spin"]
    lookup['costheta1inf']=["float","Cosine of the asymptotic angle between orbital angular momentum and primary spin"]
    lookup['costheta2inf']=["float","Cosine of the asymptotic angle between orbital angular momentum and secondary spin"]


    lookup['kappainf']=["float","Asymptotic value of the regularized momentum kappa"]
    lookup['t']=["float","Time"]
    lookup['m']=["float","Parameter of elliptic function(s)"]
    lookup['Lvec']=["array","Cartesian vector of the orbital angular momentum"]
    lookup['S1vec']=["array","Cartesian vector of the primary spin"]
    lookup['S2vec']=["array","Cartesian vector of the secondary spin"]
    lookup['Lh']=["array","Direction of the orbital angular momentum, unit vector"]
    lookup['S1h']=["array","Direction of the primary spin, unit vector"]
    lookup['S2h']=["array","Direction of the secondary spin, unit vector"]
    lookup['eta']=["float","Symmetric mass ratio 0<=eta<=1/4"]
    lookup['v']=["float","Newtonian orbital velocity"]
    lookup['Jmin']=["float","Minimum value of the total angular momentum J"]
    lookup['Jmax']=["float","Maximum value of the total angular momentum J"]
    lookup['ximin']=["float","Minimum value of the effective spin xi"]
    lookup['ximax']=["float","Maximum value of the effective spin xi"]
    lookup['Smin']=["float","Minimum value of the total spin S"]
    lookup['Smax']=["float","Maximum value of the total spin S"]

    lookup['coeff6']=["float","Coefficient to the x^6 term in polynomial"]
    lookup['coeff5']=["float","Coefficient to the x^5 term in polynomial"]
    lookup['coeff4']=["float","Coefficient to the x^4 term in polynomial"]
    lookup['coeff3']=["float","Coefficient to the x^3 term in polynomial"]
    lookup['coeff2']=["float","Coefficient to the x^2 term in polynomial"]
    lookup['coeff1']=["float","Coefficient to the x^1 term in polynomial"]
    lookup['coeff0']=["float","Coefficient to the x^0 term in polynomial"]
    lookup['coeff']=["float","Coefficient"]

    lookup['thetaL']=["float","Angle betwen orbital angular momentum and total angular momentum"]
    lookup['costhetaL']=["float","Cosine of the angle betwen orbital angular momentum and total angular momentum"]
    lookup['morph']=["string","Spin morphology"]

    lookup['simpler']=["boolean","If True simplifies output"]
    lookup['N']=["integer","Number of samples"]
    lookup['vec']=["array","Vector in Cartesian coomponents"]
    lookup['dS2dt2']=["float","Squared time derivative of the squared total spin."]
    lookup['dS2dt']=["float","Time derivative of the squared total spin"]
    lookup['dSdt']=["float","Time derivative of the total spin"]

    lookup['Sminus2']=["float","Lowest physical root, if present, of the effective potential equation"]
    lookup['Splus2']=["float","Largest physical root, if present, of the effective potential equation"]
    lookup['S32']=["float","Spurious root of the effective potential equation"]
    lookup['tau']=["float","Nutation period"]

    lookup['Sminus2inf']=["float","Asymptotic value of the lowest physical root, if present, of the effective potential equation"]
    lookup['Splus2inf']=["float","Asymptotic value of the largest physical root, if present, of the effective potential equation"]
    lookup['S32inf']=["float","Asymptotic value of the spurious root of the effective potential equation"]
    lookup['RHS']=["float","Right-hand side"]
    lookup['outputs']=["dictionary","Set of outputs"]
    lookup['requested_outputs']=["list","Set of outputs"]

    lookup['r_udp']=["float","Outer orbital separation in the up-down instability."]
    lookup['r_udm']=["float","Inner orbital separation in the up-down instability."]
    lookup['r_wide']=["float","Orbital separation where wide nutations becomes possible."]
    lookup['omegasq']=["float","Squared frequency."]
    lookup['which']=["string","Select function behavior."]

    lookup['allvars']=["array","Packed ODE input variables."]
    lookup['precomputedroots']=["array","Output of S2roots."]
    lookup['ODEsolution']=["array of scipy OdeSolution objects", "Solution of the ODE. Key method is .sol(t)"]

    lookup['kappainitial']=["float","Initial value of the regularized momentum kappa"]
    lookup['uinitial']=["float","Initial value of the compactified separation 1/(2L)"]
    lookup['ufinal']=["float","Final value of the compactified separation 1/(2L)"]


    if varname in lookup:
        pass
    else:
        varname="MISSING"

    dstrings=varname+": "
    dstrings+=lookup[varname][0]
    if vardef is not None:
        dstrings+=', optional (default: '+vardef+')'

    dstrings+="\n\t"
    dstrings+=lookup[varname][1]
    if dstrings[-1]!=".":
        dstrings+="."
    dstrings+="\n"

    return dstrings


# A tab is four spaces in standard python
realtab='    '

if True:
    docs='\"\"\"\n'
    #with open("precession/precession.py") as file:
    #    sourcecode=file.readlines()
    #

    # Grab the current docstrings
    sourcecode = eval("precession."+fun+".__doc__").split('\n')
    sourcecode = [line.strip() for line in sourcecode]
    if sourcecode[0]=='':
        sourcecode=sourcecode[1:]

    intro=None
    for i,line in enumerate(sourcecode):
        if line =='' and intro is None:
            intro="\n".join(sourcecode[0:i])
            docs+=intro
            if docs[-1]!=".":
                docs+="."
            docs+='\n\n'




        if fun in line and "=" in line and "(" in line and ")" in line:
            # Remove all the space
            line= line.replace(' ','').replace('\t','')
            docs+="Call\n----\n"
            docs+=line.replace('=',' = ',1)
            docs+='\n'

            # Select string in between parentheses
            inputs = line.split('(')[1].split(')')[0].split(',')

            docs+="\nParameters\n----------\n"

            # Loop over inputs
            for var in inputs:
                varname = var.split('=')[0]
                try:
                    vardef = var.split('=')[1]
                except:
                    vardef= None
                #print(varname,vardef)
                docs+=descr(varname,vardef)

            # Select before equal sign
            outputs = line.split('=')[0].split(',')

            docs+="\nReturns\n-------\n"
            # Loop over inputs
            for var in outputs:
                docs+=descr(var)



    # Remove last new line
    #docs=docs.rstrip()
    # Indent everything
    docs=realtab+docs.replace('\n','\n'+realtab)
    docs+='\"\"\"\n'

    print(docs) # To screen

    pyperclip.copy(docs) # Copy to clipboard
