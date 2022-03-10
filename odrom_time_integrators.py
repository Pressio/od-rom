
import numpy as np
from scipy import linalg
from myio import load_basis_from_binary_file
import time, math

# --------------------------------
# RK2
# --------------------------------
def odrom_rk2(odProblem, yhat, nSteps, dt, observer=None):

  romRhs = np.zeros_like(yhat)
  k1     = np.zeros_like(yhat)
  k2     = np.zeros_like(yhat)
  yhat0  = np.zeros_like(yhat)
  half = 0.5

  time = 0.
  for step in range(1, nSteps+1):
    if step % 10 == 0:
      stateNorm = linalg.norm(yhat, check_finite=False)
      print("step {} of {}, romStateNorm = {}".format(step, nSteps, stateNorm))
      if math.isnan(stateNorm): break

    if observer!= None:
      observer(step-1, yhat)

    # step 1
    odProblem.reconstructFomState(yhat)
    odProblem.computeFomVelocity(time)
    odProblem.projectFomVelo(romRhs)
    k1[:] = dt * romRhs

    # step 2
    yhat0 = yhat + k1
    odProblem.reconstructFomState(yhat0)
    odProblem.computeFomVelocity(time+dt)
    odProblem.projectFomVelo(romRhs)
    k2[:] = dt * romRhs

    yhat[:] = yhat + k2*half + k1*half
    time += dt

# --------------------------------
# RK4
# --------------------------------
def odrom_rk4(odProblem, yhat, nSteps, dt, observer=None):

  romRhs = np.zeros_like(yhat)
  k1     = np.zeros_like(yhat)
  k2     = np.zeros_like(yhat)
  k3     = np.zeros_like(yhat)
  k4     = np.zeros_like(yhat)
  yhat0  = np.zeros_like(yhat)

  half = 0.5
  two  = 2.
  oneOverSix = 1./6.

  time = 0.
  for step in range(1, nSteps+1):
    if step % 10 == 0:
      stateNorm =linalg.norm(yhat, check_finite=False)
      print("step {} of {}, romStateNorm = {}".format(step, nSteps, stateNorm))
      if math.isnan(stateNorm): break

    if observer!= None:
      observer(step-1, yhat)

    # step 1
    odProblem.reconstructFomState(yhat)
    odProblem.computeFomVelocity(time)
    odProblem.projectFomVelo(romRhs)
    k1[:] = dt * romRhs

    # step 2
    yhat0 = yhat + half*k1
    odProblem.reconstructFomState(yhat0)
    odProblem.computeFomVelocity(time+half*dt)
    odProblem.projectFomVelo(romRhs)
    k2[:] = dt * romRhs

    # step 3
    yhat0 = yhat + half*k2
    odProblem.reconstructFomState(yhat0)
    odProblem.computeFomVelocity(time+half*dt)
    odProblem.projectFomVelo(romRhs)
    k3[:] = dt * romRhs

    # step 4
    yhat0 = yhat + k3
    odProblem.reconstructFomState(yhat0)
    odProblem.computeFomVelocity(time+dt)
    odProblem.projectFomVelo(romRhs)
    k4[:] = dt * romRhs

    yhat[:] = yhat + (k1+two*k2+two*k3+k4)*oneOverSix

    time += dt

# --------------------------------
# SSPRK3
# --------------------------------
def odrom_ssprk3(odProblem, yhat, nSteps, dt, observer=None):

  romRhs = np.zeros_like(yhat)
  yhat0  = np.zeros_like(yhat)
  yhat1  = np.zeros_like(yhat)

  two = 2.
  oneOverThree  = 1./3.
  oneOverFour   = 1./4.
  threeOverFour = 3./4.

  time = 0.
  for step in range(1, nSteps+1):
    if step % 10 == 0:
      stateNorm =linalg.norm(yhat, check_finite=False)
      print("step {} of {}, romStateNorm = {}".format(step, nSteps, stateNorm))
      if math.isnan(stateNorm): break

    if observer!= None:
      observer(step-1, yhat)

    odProblem.reconstructFomState(yhat)
    odProblem.computeFomVelocity(time)
    odProblem.projectFomVelo(romRhs)
    yhat0[:] = yhat + dt * romRhs

    odProblem.reconstructFomState(yhat0)
    odProblem.computeFomVelocity(time+dt)
    odProblem.projectFomVelo(romRhs)
    yhat1[:] = threeOverFour*yhat + oneOverFour*yhat0 + oneOverFour*dt*romRhs

    odProblem.reconstructFomState(yhat1)
    odProblem.computeFomVelocity(time+0.5*dt)
    odProblem.projectFomVelo(romRhs)
    yhat[:] = oneOverThree*(yhat + two*yhat1 + two*dt*romRhs)

    time += dt
