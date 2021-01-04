import sys

fun = sys.argv[1]

def descr(varname,vardef=None):


    lookup={}
    lookup['MISSING']=["COULD NOT BUILD","FILL MANUALLY"]
    lookup['q']=["float","Mass ratio: 0<=q<=1"]
    lookup['m1']=["float","Mass of the primary (heavier) black hole"]
    lookup['m2']=["float","Mass of the secondary (lighter) black hole"]
    lookup['chi1']=["float","Dimensionless spin of the primary (heavier) black hole: 0<=chi1<= 1"]
    lookup['chi2']=["float","Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1"]
    lookup['r']=["float","Binary separation"]
    lookup['L']=["float","Magnitude of the Newtonian orbital angular momentum"]
    lookup['xi']=["float","Effective spin"]
    lookup['J']=["float","Magnitude of the total angular momentum"]
    lookup['S']=["float","Magnitude of the total spin"]
    lookup['kappa']=["float","Regularized angular momentum (J^2-L^2)/(2L)"]
    lookup['u']=["float","Compactified separation 1/(2L)"]
    lookup['varphi']=["float","Generalized nutation coordinate (Eq 9 in arxiv:1506.03492)."]
    lookup['sign']=["integer","Sign, either +1 or -1"]
    lookup['theta1']=["float","Angle between orbital angular momentum and primary spin"]
    lookup['theta2']=["float","Angle between orbital angular momentum and secondary spin"]
    lookup['deltaphi']=["float","Angle between the projections of the two spins onto the orbital plane"]
    lookup['theta1inf']=["float","Asymptotic value of the angle between orbital angular momentum and primary spin"]
    lookup['theta2inf']=["float","Asymptotic value of the angle between orbital angular momentum and secondary spin"]
    lookup['kappainf']=["float","Asymptotic value of the regularized momentum kappa"]
    lookup['t']=["float","Time"]
    lookup['m']=["float","Parameter of elliptic function(s)"]
    lookup['Lvec']=["array","Cartesian vector of the orbital angular momentum"]
    lookup['S1vec']=["array","Cartesian vector of the primary spin"]
    lookup['S2vec']=["array","Cartesian vector of the secondary spin"]
    lookup['Lh']=["array","Direction of the orbital angular momentum, unit vector"]
    lookup['S1h']=["array","Direction of the primary spin, unit vector"]
    lookup['S2h']=["array","Direction of the secondary spin, unit vector"]


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



docs=""
with open("precession/precession.py") as file:
    sourcecode=file.readlines()
    for i,line in enumerate(sourcecode):
        if fun in line:
            if '----' in sourcecode[i-1]:

                # Remove all the space
                line= line.replace(' ','')

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
                outputs = line.split('=')[0].split(',')

                docs+="\nReturns\n-------\n"
                # Loop over inputs
                for var in outputs:
                    docs+=descr(var)
# Remove last new line
docs=docs.rstrip()
# Indent everything
docs='\t'+docs.replace('\n','\n\t')

print(docs)
