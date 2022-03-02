
import numpy as np
from scipy import linalg
from myio import load_basis_from_binary_file
import time, math

class OdRomFull:
  def __init__(self, podDir, fomTotalDofs, modesDicIn, \
               physDim, numDofsPerCell, \
               partInfoDir):

    # physical dimensions and dofs/cell
    self.physDim_ = physDim
    self.ndpc_    = numDofsPerCell

    # modesDicIn: dictionary such that:
    # key = tileId, value = num of modes in tile
    self.modesDic_ = modesDicIn
    self.nTiles_   = len(modesDicIn.keys())

    # count all mode count within each tile
    self.totalModesCount_ = np.sum(list(self.modesDic_.values()))
    print("self.totalModesCount_ = {}".format(self.totalModesCount_))

    # phis_: dictionary such that:
    # key = tileId, value = local bases
    self.phis_ = {}

    self.fomState_ = np.zeros(fomTotalDofs)
    self.fomVelo_  = np.zeros(fomTotalDofs)
    self.fullStateRows_ = {}

    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]
      myPhi = load_basis_from_binary_file(podDir + "/lsv_state_p_" + str(tileId) )[:,0:myK]
      self.phis_[tileId] = myPhi
      srVecFile = partInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
      self.fullStateRows_[tileId] = np.loadtxt(srVecFile, dtype=int)

  def createRomState(self):
    return np.zeros(self.totalModesCount_)

  def viewFomState(self):
    return self.fomState_

  def reconstructMemberFomStateFullMeshOrdering(self, romStateIn):
    romState_i = 0
    fomState_i = 0
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]
      myPhi = self.phis_[tileId]
      myRomStateSlice = romStateIn[romState_i:romState_i+myK]
      tmpy  = np.dot(myPhi, myRomStateSlice)

      for j,it in enumerate(self.fullStateRows_[tileId]):
        self.fomState_[it] = tmpy[j]
      romState_i += myK
      fomState_i += len(tmpy)

  def projectMemberFomVelo(self, romState):
    romState_i = 0
    fomVelo_i = 0
    for tileId in range(self.nTiles_):
      myK = self.modesDic_[tileId]
      myPhi = self.phis_[tileId]
      n = myPhi.shape[0]
      myRhsSlice = self.fomVelo_[fomVelo_i:fomVelo_i+n]
      yhattmp    = np.dot(myPhi.T, myRhsSlice)
      romState[romState_i:romState_i+myK] = np.copy(yhattmp)
      romState_i += myK
      fomVelo_i += n

  def reconstructMemberFomState(self, romStateIn):
    romState_i = 0
    fomState_i = 0
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]
      myPhi = self.phis_[tileId]
      myRomStateSlice = romStateIn[romState_i:romState_i+myK]
      tmpy  = np.dot(myPhi, myRomStateSlice)
      self.fomState_[fomState_i:fomState_i+len(tmpy)] = np.copy(tmpy)
      romState_i += myK
      fomState_i += len(tmpy)

  # --------------------------------
  # SSPRK3
  # --------------------------------
  def runSSPRK3(self, workDir, yhat, fomApp, nSteps, dt, observer=None):
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

      self.reconstructMemberFomState(yhat)
      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      yhat0[:] = yhat + dt * romRhs

      self.reconstructMemberFomState(yhat0)
      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      yhat1[:] = threeOverFour*yhat + oneOverFour*yhat0 + oneOverFour*dt*romRhs

      self.reconstructMemberFomState(yhat1)
      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      yhat[:] = oneOverThree*(yhat + two*yhat1 + two*dt*romRhs)

      time += dt

  # --------------------------------
  # RK4
  # --------------------------------
  def runRK4(self, workDir, yhat, fomApp, nSteps, dt, observer=None):
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
      self.reconstructMemberFomState(yhat)
      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      k1[:] = dt * romRhs

      # step 2
      yhat0 = yhat + half*k1
      self.reconstructMemberFomState(yhat0)
      fomApp.velocity(self.fomState_, time+half*dt, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      k2[:] = dt * romRhs

      # step 3
      yhat0 = yhat + half*k2
      self.reconstructMemberFomState(yhat0)
      fomApp.velocity(self.fomState_, time+half*dt, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      k3[:] = dt * romRhs

      # step 4
      yhat0 = yhat + k3
      self.reconstructMemberFomState(yhat0)
      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      k4[:] = dt * romRhs

      yhat[:] = yhat + (k1+two*k2+two*k3+k4)*oneOverSix

      time += dt

  # --------------------------------
  # RK2
  # --------------------------------
  def runRK2(self, workDir, yhat, fomApp, nSteps, dt, observer=None):
    romRhs = np.zeros_like(yhat)
    k1     = np.zeros_like(yhat)
    k2     = np.zeros_like(yhat)
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

      # if step % 50 == 0:
      #   self.reconstructMemberFomStateFullMeshOrdering(yhat)
      #   np.savetxt(workDir+"/y_rec_"+str(step)+".txt", self.fomState_)

      if observer!= None:
        observer(step-1, yhat)

      # step 1
      self.reconstructMemberFomState(yhat)
      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      k1[:] = dt * romRhs

      # step 2
      yhat0 = yhat + half*k1
      self.reconstructMemberFomState(yhat0)
      fomApp.velocity(self.fomState_, time+half*dt, self.fomVelo_)
      self.projectMemberFomVelo(romRhs)
      k2[:] = dt * romRhs

      yhat[:] = yhat + k2*1.

      time += dt
