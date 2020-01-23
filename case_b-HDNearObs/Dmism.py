#!/usr/bin/python3

import sys
modelName = sys.argv[1]
pathToFunctions = '/home/lamd/gwes/' + modelName + '/'
sys.path.insert(0, pathToFunctions) # to call functions in myFunctions.py
#sys.path.insert(0,'/home/lamd/testPython/lib/python3/site-packages')
import myFunctions as myFun
import numpy as np
import scipy
import math
from scipy.sparse import eye

# Set general variables
processRank = sys.argv[2]
rank = int(processRank)
dataTypes = sys.argv[3]
nbObsPts = int(sys.argv[4])

# Load calibration data
#obsData = np.reshape(np.loadtxt('/home/lamd/gwes/obsDataWithNoise_' + processRank + '.txt'),(-1, 1)) # with inflated noise
obsData = np.reshape(np.loadtxt('/home/lamd/gwes/obsData4OF_' + processRank + '.txt'),(-1, 1)) # without noise
#obsData_st = np.tile(obsData[0:10], (36,1))
#obsData_st = np.repeat(obsData[0:10], 36)
#obsData_tr = obsData[10:]
#obsData_st = obsData[0:360]
#obsData_tr = obsData[360:]
#newObsData = np.concatenate((obsData_st, obsData_tr))
#nbOfData = newObsData.shape[0]
nbOfData = obsData.shape[0]

# Load simulated data
simDataEns = np.loadtxt('/home/lamd/gwes/ens_of_simulatedData.txt')
simData_ini = np.reshape(simDataEns[:, rank], (-1, 1))
#simData_ini = np.reshape(np.loadtxt('/home/lamd/gwes/simData_' + processRank + '.txt'), (-1, 1)) # without noise
#simData_ini = np.reshape(np.loadtxt('/home/lamd/gwes/simDataWithNoise_' + processRank + '.txt'), (-1, 1)) # with inflated noise 
#simData_ini_st = np.tile(simData_ini[0:10], (36,1))
#simData_ini_st = np.repeat(simData_ini[0:10], 36)
#simData_ini_tr = simData_ini[10:]
#simData_ini_st = simData_ini[0:360]
#simData_ini_tr = simData_ini[360:]
#newSimData_ini = np.concatenate((simData_ini_st, simData_ini_tr))


## Compute the ensemble of measurement error covariance matrix

#ensOfOrigObsData = np.loadtxt('/home/lamd/gwes/ens_of_origObsData_withRegularNoise.txt') # original ensemble
#
## Assuming uncorrelated observation data noise
#inv_obsErrCovar = np.eye(nbOfData)
#varVector = np.reshape(np.var(ensOfOrigObsData, axis=1), (-1,1))
#inv_obsErrCovar[range(nbOfData), range(nbOfData)] = np.reshape(1/varVector, nbOfData)

inv_obsErrCovar = np.eye(nbOfData)
#np.savetxt('inv_obsErrCovar.txt', inv_obsErrCovar, fmt="%.4e")

# Compute initial value of objective function
#objFun = myFun.computeObjFun(newObsData, inv_obsErrCovar, newSimData_ini, dataTypes) # value is a list
objFun = myFun.computeObjFun(obsData, inv_obsErrCovar, simData_ini, dataTypes) # value is a list
#print(objFun)
OF_tot = float(format(objFun[0], '.4e'))

#TODO: if conditions with other combination of data types
if dataTypes == "h+q":
	OF_h = float(format(objFun[1], '.4e'))
	OF_q = float(format(objFun[2], '.4e'))

# Write OF values computed using head data only
	with open('Dmism_h_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_h)
		g.close()
	# Write OF values computed using head data only
	with open('Dmism_q_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_q)
		g.close()

elif dataTypes == "dh+q":
	OF_dh = float(format(objFun[1], '.4e'))
	OF_q = float(format(objFun[2], '.4e'))

	# Write OF values computed using head data only
	with open('objFun_dh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_dh)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_q_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_q)
		g.close()

elif dataTypes == "h+dh":
	OF_ssH = float(format(objFun[1], '.4e'))
	OF_h = float(format(objFun[2], '.4e'))
	OF_dh = float(format(objFun[3], '.4e'))

	# Write OF values computed using head data only
	with open('objFun_ssH_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_ssH)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_h_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_h)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_dh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_dh)
		g.close()

elif dataTypes == "h":
	OF_ssH = float(format(objFun[1], '.4e'))
	OF_h = float(format(objFun[2], '.4e'))

	# Write OF values computed using head data only
	with open('Dmism_ssH_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_ssH)
		g.close()
	# Write OF values computed using head data only
	with open('Dmism_h_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_h)
		g.close()

elif dataTypes == "h+dh+vdh":
	OF_h = float(format(objFun[1], '.4e'))
	OF_dh = float(format(objFun[2], '.4e'))
	OF_vdh = float(format(objFun[3], '.4e'))

	# Write OF values computed using head data only
	with open('objFun_h_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_h)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_dh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_dh)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_vdh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_vdh)
		g.close()

elif dataTypes == "h+vdh":
	OF_h = float(format(objFun[1], '.4e'))
	OF_vdh = float(format(objFun[2], '.4e'))

	# Write OF values computed using head data only
	with open('objFun_h_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_h)
		g.close()

	# Write OF values computed using head data only
	with open('objFun_vdh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_vdh)
		g.close()

elif dataTypes == "h+vdh+q":
	OF_h = float(format(objFun[1], '.4e'))
	OF_vdh = float(format(objFun[2], '.4e'))
	OF_q = float(format(objFun[3], '.4e'))

	# Write OF values computed using head data only
	with open('/home/lamd/gwes/objFun_h_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_h)
		g.close()
	# Write OF values computed using head data only
	with open('/home/lamd/gwes/objFun_vdh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_vdh)
		g.close()
	# Write OF values computed using head data only
	with open('/home/lamd/gwes/objFun_q_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_q)
		g.close()

elif dataTypes == "h+dh+vdh+q":
	OF_h = float(format(objFun[1], '.4e'))
	OF_dh = float(format(objFun[2], '.4e'))
	OF_vdh = float(format(objFun[3], '.4e'))
	OF_q = float(format(objFun[4], '.4e'))

	# Write OF values computed using head data only
	with open('objFun_h_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_h)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_dh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_dh)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_vdh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_vdh)
		g.close()
	# Write OF values computed using head data only
	with open('/home/lamd/gwes/objFun_q_' + processRank + '.txt', 'w') as q:
		q.write("%e\n" % OF_q)
		q.close()
	
elif dataTypes == "h+dh+q":
	OF_h = float(format(objFun[1], '.4e'))
	OF_dh = float(format(objFun[2], '.4e'))
	OF_q = float(format(objFun[3], '.4e'))

	# Write OF values computed using head data only
	with open('objFun_h_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_h)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_dh_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_dh)
		g.close()
	# Write OF values computed using head data only
	with open('objFun_q_' + processRank + '.txt', 'w') as g:
		g.write("%e\n" % OF_q)
		g.close()
	

## Write initial value of the objective function to file
#with open('objFun_' + processRank + '.txt', 'w') as f:
#	f.write("%e" % objFun)
#	f.close()

# Write to file listing all objective function values during the optimization
with open('DmismValues_' + processRank + '.txt', 'w') as g:
	g.write("%e\n" % OF_tot)
	g.close()

## Write initial Levenberg-Marquardt parameter to file 
##LevMarq = 10**math.floor(math.log10(OF_tot/(2*nbOfData)))
#LevMarq = 10 # or 100
#
#with open('LevMarq_' + processRank + '.txt', 'w') as f:
#	f.write("%f" % LevMarq)
#	f.close()
#
## Write to file listing all objective function values during the optimization
#with open('LevMarqValues_' + processRank + '.txt', 'w') as g:
#	g.write("%f\n" % LevMarq)
#	g.close()
#
