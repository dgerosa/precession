'''
This is a semi-automatic docstrings builder for the precession code.

Usage:
python dstrings_builder.py <name of function>

For each function docstrings, the developer needs to provide the intro blurb and the "Examples" line. This code will then try to fill the "Parameters" and "Returns" description.
'''



import sys,os
import pyperclip

# Load package from path, not the pip installation (if any)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import precession
#print(precession.__version__)

fun = sys.argv[1]

def descr(varname,vardef=None):

    # This is a lookup table
    lookup={}
    lookup['MISSING']=["COULD NOT BUILD","FILL MANUALLY"]
    lookup['q']=["float","Mass ratio: 0<=q<=1"]
    lookup['m1']=["float","Mass of the primary (heavier) black hole"]
    lookup['m2']=["float","Mass of the secondary (lighter) black hole"]
    lookup['chi1']=["float","Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1"]
    lookup['chi2']=["float","Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1"]
    lookup['r']=["float","Binary separation"]
    lookup['L']=["float","Magnitude of the Newtonian orbital angular momentum"]
    lookup['S1']=["float","Magnitude of the primary spin"]
    lookup['S2']=["float","Magnitude of the secondary spin"]
    lookup['chieff']=["float","Effective spin"]
    lookup['J']=["float","Magnitude of the total angular momentum"]
    lookup['S']=["float","Magnitude of the total spin"]
    lookup['Ssq']=["float","Squared magnitude of the total spin"]
    lookup['kappa']=["float","Regularized angular momentum (J^2-L^2)/(2L)"]
    lookup['u']=["float","Compactified separation 1/(2L)"]
    lookup['varphi']=["float","Generalized nutation coordinate (Eq 9 in arxiv:1506.03492)."]
    lookup['sign']=["integer","Sign, either +1 or -1"]
    lookup['cyclesign']=["integer","Sign (either +1 or -1) to cover the two halves of a precesion cycle"]
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
    lookup['phi']=["float","Amplitude of elliptic function(s)"]
    lookup['n']=["float","Characheristic of elliptic function(s)"]

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
    lookup['kappamin']=["float","Minimum value of the regularized angular momentum kappa"]
    lookup['kappamax']=["float","Maximum value of the regularized angular momentum kappa"]
    lookup['kappainfmin']=["float","Minimum value of the asymptotic angular momentum kappainf"]
    lookup['kappainfmax']=["float","Maximum value of the asymptotic angular momentum kappainf"]
    lookup['chieffmin']=["float","Minimum value of the effective spin chieff"]
    lookup['chieffmax']=["float","Maximum value of the effective spin chieff"]
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
    lookup['enforce']=["boolean","If True raise errors, if False raise warnings"]
    lookup['N']=["integer","Number of samples"]
    lookup['vec']=["array","Vector in Cartesian coomponents"]
    lookup['dSsdts']=["float","Squared time derivative of the squared total spin."]
    lookup['dSsdt']=["float","Time derivative of the squared total spin"]
    lookup['dSdt']=["float","Time derivative of the total spin"]
    lookup['Sminuss']=["float","Lowest physical root, if present, of the effective potential equation"]
    lookup['Spluss']=["float","Largest physical root, if present, of the effective potential equation"]
    lookup['S3s']=["float","Spurious root of the effective potential equation"]
    lookup['tau']=["float","Nutation period"]
    lookup['Sminussinf']=["float","Asymptotic value of the lowest physical root, if present, of the effective potential equation"]
    lookup['Splussinf']=["float","Asymptotic value of the largest physical root, if present, of the effective potential equation"]
    lookup['S3sinf']=["float","Asymptotic value of the spurious root of the effective potential equation"]
    lookup['RHS']=["float","Right-hand side"]
    lookup['outputs']=["dictionary","Set of outputs"]
    lookup['requested_outputs']=["list","Set of outputs"]
    lookup['r_udp']=["float","Outer orbital separation in the up-down instability."]
    lookup['r_udm']=["float","Inner orbital separation in the up-down instability."]
    lookup['r_wide']=["float","Orbital separation where wide nutations becomes possible."]
    lookup['omegasq']=["float","Squared frequency."]
    lookup['which']=["string","Select function behavior."]
    lookup['allvars']=["array","Packed ODE input variables."]
    lookup['ODEsolution']=["array of scipy OdeSolution objects", "Solution of the ODE. Key method is .sol(t)"]
    lookup['kappainitial']=["float","Initial value of the regularized momentum kappa"]
    lookup['uinitial']=["float","Initial value of the compactified separation 1/(2L)"]
    lookup['ufinal']=["float","Final value of the compactified separation 1/(2L)"]
    lookup['Lhinitial']=["array","Initial direction of the orbital angular momentum, unit vector"]
    lookup['S1hinitial']=["array","Initial direction of the primary spin, unit vector"]
    lookup['S2hinitial']=["array","Initial direction of the secondary spin, unit vector"]
    lookup['vinitial']=["float","Initial value of the newtonian orbital velocity"]
    lookup['vfinal']=["float","Final value of the newtonian orbital velocity"]
    lookup['mathcalC0']=["float","Prefactor in the OmegaL equation"]
    lookup['mathcalCplus']=["float","Prefactor in the OmegaL equation"]
    lookup['mathcalCminus']=["float","Prefactor in the OmegaL equation"]
    lookup['mathcalC0prime']=["float","Prefactor in the PhiL equation"]
    lookup['mathcalCplusprime']=["float","Prefactor in the PhiL equation"]
    lookup['mathcalCminusprime']=["float","Prefactor in the PhiL equation"]
    lookup['mathcalT']=["float","Prefactor in the tau equation"]

    lookup['alpha']=['float', "Azimuthal angle spanned by L about J during an entire cycle"]
    lookup['phiL']=['float', "Azimuthal angle spanned by L about J"]
    lookup['OmegaL']=['float', "Precession frequency of L about J"]
    lookup['full_output']=['boolean', "Return additional outputs"]
    lookup['chipterm1']=['float', "Term in effective precessing spin chip"]
    lookup['chipterm2']=['float', "Term in effective precessing spin chip"]
    lookup['chip']=['float', "Effective precessing spin chip"]
    lookup['Nsamples']=['integer', "Number of Monte Carlo samples"]
    lookup['rswitch']=["float","Matching separation between the precession- and orbit-averaged chunks"]
    lookup['uswitch']=["float","Matching compactified separation between the precession- and orbit-averaged chunks"]
    lookup['M_msun']=["float","Total mass of the binary in solar masses"]
    lookup['f']=["float","Gravitational-wave frequency in Hz"]
    lookup['theta1atmin']=["float","Value of the angle theta1 at the resonance that minimizes either J or chieff, depending on the input"]
    lookup['theta1atmax']=["float","Value of the angle theta1 at the resonance that maximizes either J or chieff, depending on the input"]
    lookup['theta2atmin']=["float","Value of the angle theta2 at the resonance that minimizes either J or chieff, depending on the input"]
    lookup['theta2atmax']=["float","Value of the angle theta2 at the resonance that maximizes either J or chieff, depending on the input"]
    lookup['deltaphiatmin']=["float","Value of the angle deltaphi at the resonance that minimizes either J or chieff, depending on the input"]
    lookup['deltaphiatmax']=["float","Value of the angle deltaphi at the resonance that maximizes either J or chieff, depending on the input"]
    lookup['precomputedroots']=["array","Pre-computed output of Ssroots for computational efficiency"]
    lookup['deltaphiatmax']=["float","Value of the angle deltaphi at the resonance that maximizes either J or chieff, depending on the input"]

    lookup['mfin']=["float","Mass of the black-hole remnant"]
    lookup['chifin']=["float","Spin of the black-hole remnant"]
    lookup['vk']=["float","Kick of the black-hole remnant (magnitude)"]
    lookup['vk_array']=["array","Kick of the black-hole remnant (in a frame aligned with L)"]
    lookup['superkick']=['boolean', "Switch kick terms on and off"]
    lookup['hangupkick']=['boolean', "Switch kick terms on and off"]
    lookup['crosskick']=['boolean', "Switch kick terms on and off"]
    lookup['kms']=['boolean', "Return velocities in km/s"]
    lookup['maxphase']=['boolean', "Maximize over orbital phase at merger"]

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


foundone=False

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

        # This is the first occurrence
        if not foundone:

            # Remove all the space
            line= line.replace(' ','').replace('\t','')

            # Select string in between parentheses
            inputs = line.split('(')[1].split(')')[0].split(',')

            docs+="Parameters\n----------\n"

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
            outputs = line.replace("`","").split('=')[0].split(',')

            docs+="\nReturns\n-------\n"
            # Loop over inputs
            for var in outputs:
                docs+=descr(var)

            docs+="\nExamples\n--------\n``"
            docs+=line.replace('precession.',"").replace('=',' = precession.',1).replace("`","")
            docs+='``\n'


            foundone=True


        else:
            newdocs =''
            for j, dline in enumerate(docs.split('\n')):
                if fun not in dline:
                    newdocs+=dline+"\n"

                if fun in dline:
                    newdocs+=dline+"\n"

                    line= line.replace(' ','').replace('\t','')
                    line = line.replace('=',' = ',1)
                    newdocs+=line+"\n"

                newoutputs = line.replace(' ','').split('=')[0].split(',')

            firstnewoutput=True
            for newoutput in newoutputs:
                if newoutput not in outputs:
                    if firstnewoutput:
                        newdocs+="Other parameters\n-------\n"
                        firstnewoutput=False
                    #print(newoutput)
                    newdocs+=descr(newoutput)

                #for var in outputs:
                #    docs+=descr(var)


            foundone=True

            docs=newdocs

# Remove last new line
#docs=docs.rstrip()
# Indent everything

docs=realtab+docs.replace('\n','\n'+realtab).replace('\t',realtab)
docs+='\"\"\"\n'

print(docs) # To screen

pyperclip.copy(docs) # Copy to clipboard
