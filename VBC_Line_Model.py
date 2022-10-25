from xspec import *

import numpy as np
import scipy as sc
import math as mp
from cmath import sqrt as Sqrt
from cmath import atan as Arctan
from scipy import integrate
from scipy import special
#even though these are imported in this module, you will still need to import
#them in your actual running environment too.

#lagroots=[0.070539889692,0.3721268180016,0.9165821024833,1.7073065310283,
#	2.7491992553094,4.048925313851,5.6151749708616,7.459017453671,9.594392869581,
#	12.038802546964,14.814293442631,17.948895520519,21.47878824029,25.451702793187,
#	29.932554631701,35.01343424048,40.833057056729,47.619994047347,55.810795750064,
#	66.524416525616]
#lagweights=[0.1687468018511,0.2912543620061,0.266686102867,
#	0.1660024532695,0.074826064668792,0.02496441730928,0.006202550844572,
#	0.0011449623864769,0.0015574177302781,0.000154014408652,0.000010864863665,
#	0.0000053301209095,0.000000175798179051,0.00000003725502402,0.00000004767529251,
#	0.000000003372844243,0.000000000001155014,0.000000000000001539,
#	0.000000000000005286, 0.000000000000000165]

lagroots, lagweights = special.roots_laguerre(20)

def r_min_1b( xi ):
	if xi <= 0:
		return  1.0 / (1.0 + xi)
	else:
		w = 2.0 / ( 1 - xi**2 ) ;
		q = np.cbrt( 27./4 * w**2 * (2-w) * ( 1.0 - np.sqrt( 1.0 - 16./27 * (w - 2.0 ) / w ) ))
		s = np.sqrt(  w * ( w * ( 1./4 + 1/q ) - 2/q  ) + q/3 ) / 2

		if w >= np.sqrt( 8 ): # r_3
			return w / 4 + s + np.sqrt( w * ( -1/(4*s) + w * ( 3/16. + w / (32*s) ) ) - s**2 )
		else:   # r_1
			return w / 4 - s + np.sqrt( w * (  1/(4*s) + w * ( 3./16 - w / (32*s) ) ) - s**2  )

def rminfunc(xi):
    if xi <= 0:
        return 1 / (1 + xi)
    else:
        w = 2/(1-xi**2)
        d1 = 27*(w**2 - 0.5 * w**3)
        d0 = 3*w**2 - 6*w
        Q = np.cbrt(0.5*d1 + 0.5*np.sqrt(d1**2 - 4*d0**3))
        S = 0.5*np.sqrt(0.25*w**2 + (Q+d0/Q)/3)
        if w >= np.sqrt(8):
            return 0.25*w + S + 0.5*np.sqrt(0.75*w**2 - 4*S**2 - (8*w-w**3)/(8*S))
        else:
            return 0.25*w - S + 0.5*np.sqrt(0.75*w**2 - 4*S**2 + (8*w-w**3)/(8*S))

# Integrand of our profile model.
# The order of the parameters is important!!!
# The integration algorithms from scipy require the integration variable be the
# FIRST parameter in the define function.
# numpy's version of exp() is used here since PyXspec inputs an array of channel's
# when evaluating the model, so using numpy is faster since it is optimized for
# array math.
def Lintegrand(r, xi, taustar, l0):
	mu = -xi/(1 - (1/r)) #convert xi to mu to keep tau in simple form

	zt = Sqrt( ((1-mu**2) * r**2) - 1 )
	gamma = ( 1 - (r * (1 - mu**2)) ) / zt
	# cmath.sqrt is used because this quantity is imaginary for all r

	if mu >= 0:
		t = Arctan( 1/ zt ) - Arctan( gamma/mu )
	else:
		t = Arctan( 1/ zt ) + Arctan( gamma/np.abs(mu) ) + np.pi
	# cmath.atan is used because the results are imaginary due to zt
	#being imaginary.
	tau = taustar * t / zt
	tau = tau.real
	# tau is mathematically real because t/zt has zero imaginary component.
	# It's still a cmath complex float, so we have to remove the imaginary part.

	dmudxi = 0.5*r/(r-1)
	dPdr = 1/l0
	optExp = np.exp(-tau)
	l0Exp = np.exp(-(r-1)/l0)
	return dmudxi*dPdr*optExp*l0Exp

def hintegrand(u, xi, taustar, l0):
	r = u*l0 + r_min_1b(xi);

	mu = -xi/(1 - (1/r)) #convert xi to mu to keep tau in simple form

	zt = Sqrt( ((1-mu**2) * r**2) - 1 )
		# cmath.sqrt is used because this quantity is imaginary for all r
	gamma = ( 1 - (r * (1 - mu**2)) ) / zt

	if mu >= 0:
		t = Arctan( 1/ zt ) - Arctan( gamma/mu )
	else:
		t = Arctan( 1/ zt ) + Arctan( gamma/np.abs(mu) ) + np.pi
	# cmath.atan is used because the results are imaginary due to zt
	#being imaginary.
	tau = taustar * t / zt
	tau = tau.real
	# tau is mathematically real because t/ztfunc has zero imaginary component.
	# It's still a cmath complex float, so we have to remove the imaginary part.

	return 0.5*(r/(r-1))*np.exp(-tau)


def VBCLineSP(engs, params, flux):
	# PARAMETER NAMES AND ORDER ARE REQUIRED BY PyXspec DOCUMENTATION!!! DO NOT CHANGE
	# Implementation of VBC model into PyXspec using quadrature functions from
	# the scipy library.
	#
	# Algorithm of the function works as follows:
	# 1) For each energy bin, convert the value to xi space
	# 2) Calculate integrate value for each xi bin
	# 3) Calculate area under L to normalize the model function to have unit area
	#    before instrumental normalization
	# 4) If xi[i] > 1 and xi[i+1] > 1, set flux[i] = zero.
	#    This is both for systematic reasons and a priori. We do not expect
	#    velocities greater than terminal speed, so it does not make sense for
	#	 there to be flux there. This is corroborated by the model's behavior.
	#    For |xi| > 1, we have to integrate arctan in the regions (-I inf, -I)
	#    and (I, I inf), which are its branch cuts.
	# 5) If xi[i] > 1 but xi[i+1] < 1, set flux[i] = (0.5 * Deltaxi/Larea)L(xi[i+1]).
	#    Justification is that in the integration, the value of L(xi[i]) would be
	#    zero based on the discussion in Step 3.
	# 6) If xi[i] < 1 but xi[i+1] > 1, set flux[i] = (0.5 * Deltaxi/Larea)L(xi[i]).
	#    Justification is same as Step 4 but now L(xi[i+1])=0
	# 7) If xi[i] < 1 but xi[i+1] < 1, set flux[i] = (0.5 * Deltaxi/Larea)(Lvals[i] + Lvals[i+1]).
	#    NOTE: THE MINUS SIGN IS TO ACCOUNT FOR THE VALUES OF xi RUNNING IN DECREASING
	#    ORDER COMPARED TO THE ENERGY ARRAY!
	#
	# Parameters from the params array:
	# taustar = params[0]
	# l0 = params[1]
	# vinf = params[2]
	# lambp = params[3]
	c = 3e5 #km/s
	hc = 12.398 #keV A


	xi = np.array([((hc/engs[i]) - params[3]) * c / (params[3] * params[2])
		for i in range(len(engs))])
	Deltaxi = xi[0]-xi[1]

	Lvals = [integrate.quad(Lintegrand, rminfunc(xival), np.inf,\
		args=(xival, params[0], params[1]))[0] if np.abs(xival) < 1 else 0. for xival in xi]
	Larea = 0.5 * (Lvals[(len(Lvals)-1)] + Lvals[0])
	for i in range(1, len(xi)-1):
		Larea += Lvals[i]
	Larea = Deltaxi * Larea

	for i in range(len(xi)-1):
		if np.abs(xi[i]) > 1 and np.abs(xi[i+1]) > 1:
			flux[i] = 0.0
		elif (np.abs(xi[i]) > 1 and np.abs(xi[i+1]) < 1):
			flux[i] = (0.5 * Deltaxi/Larea) * Lvals[i+1]
		elif (np.abs(xi[i]) < 1 and np.abs(xi[i+1]) > 1):
			flux[i] = (0.5 * Deltaxi/Larea) * Lvals[i]
		else:
			flux[i] = (0.5 * Deltaxi/Larea) * (Lvals[i] + Lvals[i+1])

def VBCLineGL(engs, params, flux):
	# PARAMETER NAMES AND ORDER ARE REQUIRED BY PyXspec DOCUMENTATION!!! DO NOT CHANGE
	# Implementation of VBC model into PyXspec using a Gauss-Lagauerre quadrature
	#
	# Algorithm of the function works as follows:
	# 1) For each energy bin, convert the value to xi space
	# 2) Calculate integrate value for each xi bin
	# 3) Calculate area under the integrand using a 20-point Gauss-Laguerre
	#    quadrature
	# 4) If xi[i] > 1 and xi[i+1] > 1, set flux[i] = zero.
	#    This is both for systematic reasons and a priori. We do not expect
	#    velocities greater than terminal speed, so it does not make sense for
	#	 there to be flux there. This is corroborated by the model's behavior.
	#    For |xi| > 1, we have to integrate arctan in the regions (-I inf, -I)
	#    and (I, I inf), which are its branch cuts.
	# 5) If xi[i] > 1 but xi[i+1] < 1, set flux[i] = (0.5 * Deltaxi/Larea)L(xi[i+1]).
	#    Justification is that in the integration, the value of L(xi[i]) would be
	#    zero based on the discussion in Step 3.
	# 6) If xi[i] < 1 but xi[i+1] > 1, set flux[i] = (0.5 * Deltaxi/Larea)L(xi[i]).
	#    Justification is same as Step 4 but now L(xi[i+1])=0
	# 7) If xi[i] < 1 but xi[i+1] < 1, set flux[i] = (0.5 * Deltaxi/Larea)(Lvals[i] + Lvals[i+1]).
	#    NOTE: THE MINUS SIGN IS TO ACCOUNT FOR THE VALUES OF xi RUNNING IN DECREASING
	#    ORDER COMPARED TO THE ENERGY ARRAY!
	#
	# Parameters from the params array:
	# taustar = params[0]
	# l0 = params[1]
	# vinf = params[2]
	# lambp = params[3]
	c = 3e5 #km/s
	hc = 12.398 #keV A
	Lvals = []

	xi = np.array([((hc/engs[i]) - params[3]) * c / (params[3] * params[2])\
		for i in range(len(engs))])
	Deltaxi = xi[0]-xi[1]

	for i in range(len(xi)):
		intSum = 0
		if abs(xi[i])<1:
			for j in range(len(lagroots)):
				intSum += lagweights[j] * hintegrand(lagroots[j], xi[i], params[0], params[1])
			Lvals.append(np.exp(-(r_min_1b(xi[i]) - 1.0)/params[1])*intSum)
		else:
			Lvals.append(0.)

	Larea = 0.5 * (Lvals[(len(Lvals)-1)] + Lvals[0])
	for i in range(1, len(xi)-1):
		Larea += Lvals[i]
	Larea = Deltaxi * Larea

	for i in range(len(xi)-1):
		if np.abs(xi[i]) > 1 and np.abs(xi[i+1]) > 1:
			flux[i] = 0.0
		elif (np.abs(xi[i]) > 1 and np.abs(xi[i+1]) < 1):
			flux[i] = (0.5 * Deltaxi/Larea) * Lvals[i+1]
		elif (np.abs(xi[i]) < 1 and np.abs(xi[i+1]) > 1):
			flux[i] = (0.5 * Deltaxi/Larea) * Lvals[i]
		else:
			flux[i] = (0.5 * Deltaxi/Larea) * (Lvals[i] + Lvals[i+1])


VBCLineInfo = ("taustar   \"\"  1  0  1e-4  5.  10.  0.01",\
        "l0   Rstar  1  1e-8  1e-4  5.  10.  0.01",\
        "vinf   km/s  2450  0  1e-4  3000  5000  0.01",\
        "lambp   A  6.648  0  1e-4  50.  100.  0.01",)
AllModels.addPyMod(VBCLine, VBCLineInfo, 'add')
AllModels.addPyMod(VBCtest, VBCLineInfo, 'add')
