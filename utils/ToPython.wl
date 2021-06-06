(* ::Package:: *)

BeginPackage["ToPython`"]
 ToPython::usage = "ToPython[expression,numpystring] converts Mathematica expression to a Numpy compatible expression.
 because Numpy can be imported in several ways, numpystring is a string that will be added to appended to function names, e.g., Cos->numpy.cos"
Begin["Private`"]
ToPython[x_,numpyprefix_:"numpy"]:=Module[{expression=x,greekrule,PythonForm,numpypre=numpyprefix,lp,rp,a,b},
(*FUNCTION TO CONVERT MATHEMATICA EXPRESSION TO NUMPY;
----------------------------------------------------;
INPUT ARGUMENTS;
x: your mathematica expression, it can be numbers, literals, complexes or lists;
numpy\[LetterSpace]prefix: string defining your Numpy import prefix, e.g.:
if your used "import numpy as np", your prefix should be the string "np"
if your used "from numpy import *", your prefix should be the empty string ""
;
OUTPUT;
the Numpy python-ready expression (to be copied as a string);
!The formatted expression will be copied ot your clipboard, ready to paste on Python!;
------------------------------------------------------;
Not tested for every possible combination; use at your risk, by Gustavo Wiederhecker*)
If[numpyprefix=="",sep="",sep="."];(*if no prefix is included, the "." separator is not used*)
lp="(";
rp=")";
PythonForm[Rational[a_,b_]]:=PythonForm[a]<>"/"<>PythonForm[b];
PythonForm[Complex[a_,b_]]:="complex"<>lp<>PythonForm[a]<>","<>PythonForm[b]<>rp;
PythonForm[Times[a_,b_]]:=PythonForm[a]<>" * "<>PythonForm[b];
PythonForm[Plus[a_,b_]]:=lp<>PythonForm[a]<>" + "<>PythonForm[b]<>rp;
PythonForm[h_[args__]]:=numpypre<>sep<>ToLowerCase[PythonForm[h]]<>lp<>PythonForm[args]<>rp;
PythonForm[Power[a_,b_]]:=lp<>PythonForm[a]<>rp<>"**"<>lp<>PythonForm[b]<>rp;
PythonForm[a_ListQ]:=numpypre<>sep<>"array"<>StringReplace[ToString[a],{"{"-> "[","}"-> "]"}];
PythonForm[Arg]=numpypre<>sep<>"angle";
(*Some functions that are note defined in numpy*)
PythonForm[Csc]:="1/"<>numpypre<>sep<>"sin";
PythonForm[Sec]:="1/"<>numpypre<>sep<>"cos";
PythonForm[Cot]:="1/"<>numpypre<>sep<>"tan";
PythonForm[Csch]:="1/"<>numpypre<>sep<>"sinh";
PythonForm[Sech]:="1/"<>numpypre<>sep<>"cosh";
PythonForm[Coth]:="1/"<>numpypre<>sep<>"tanh";
(*Handling arrays*)
PythonForm[List[args__]]:=numpypre<>sep<>"array"<>lp<>"["<>Table[PythonForm[{args}[[ii]]]<>",",{ii,1,Length@{args}}]<>"]"<>rp;
(*Pi and E*)
PythonForm[\[Pi]]=numpypre<>sep<>"pi";
PythonForm[E]=numpypre<>sep<>"e";
(*real numbers, engineering notation*)
PythonForm[r_Real]:=Block[{a=MantissaExponent[r]},If[r>=0,ToString[N[a[[1]],6]]<>"e"<>ToString[a[[2]]],"("<>ToString[N[a[[1]],6]]<>"e"<>ToString[a[[2]]]<>")"]];
(*Greek characters*)
greekrule={"\[Alpha]"->"alpha","\[Beta]"->"beta","\[Gamma]"->"gamma","\[Delta]"->"delta","\[CurlyEpsilon]"->"curlyepsilon","\[Zeta]"->"zeta","\[Eta]"->"eta","\[Theta]"->"theta","\[Iota]"->"iota","\[Kappa]"->"kappa","\[Lambda]"->"lambda","\[Mu]"->"mu","\[Nu]"->"nu","\[Xi]"->"xi","\[Omicron]"->"omicron","\[Pi]"->"pi","\[Rho]"->"rho","\[FinalSigma]"->"finalsigma","\[Sigma]"->"sigma","\[Tau]"->"tau","\[Upsilon]"->"upsilon","\[CurlyPhi]"->"curlyphi","\[Chi]"->"chi","\[Psi]"->"psi","\[Omega]"->"omega","\[CapitalAlpha]"->"Alpha","\[CapitalBeta]"->"Beta","\[CapitalGamma]"->"Gamma","\[CapitalDelta]"->"Delta","\[CapitalEpsilon]"->"CurlyEpsilon","\[CapitalZeta]"->"Zeta","\[CapitalEta]"->"Eta","\[CapitalTheta]"->"Theta","\[CapitalIota]"->"Iota","\[CapitalKappa]"->"Kappa","\[CapitalLambda]"->"Lambda","\[CapitalMu]"->"Mu","\[CapitalNu]"->"Nu","\[CapitalXi]"->"Xi","\[CapitalOmicron]"->"Omicron","\[CapitalPi]"->"Pi","\[CapitalRho]"->"Rho","\[CapitalSigma]"->"Sigma","\[CapitalTau]"->"Tau","\[CapitalUpsilon]"->"Upsilon","\[CapitalPhi]"->"CurlyPhi","\[CapitalChi]"->"Chi","\[CapitalPsi]"->"Psi","\[CapitalOmega]"->"Omega"};
(*Everything else*)
PythonForm[allOther_]:=StringReplace[ToString[allOther,FortranForm],greekrule];
(*Copy results to clipboard*)
CopyToClipboard[PythonForm[expression]];
PythonForm[expression]]
End[]
EndPackage[]
