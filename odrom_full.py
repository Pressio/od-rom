
import numpy as np
from scipy import linalg

# -------------------------------------------------------------------
def load_basis_from_binary_file(lsvFile):
  nr, nc  = np.fromfile(lsvFile, dtype=np.int64, count=2)
  M = np.fromfile(lsvFile, offset=np.dtype(np.int64).itemsize*2)
  M = np.reshape(M, (nr, nc), order='F')
  return M

# -------------------------------------------------------------------
class OdRomFull:
  def __init__(self, podDir, fomStateSize, fomVeloSize, modesIn):
    self.modes_    = modesIn
    self.nTiles_   = len(modesIn.keys())
    self.fomState_ = np.zeros(fomStateSize)
    self.fomVelo_  = np.zeros(fomVeloSize)

    self.totalModesCount_ = 0
    for k,v in self.modes_.items():
      self.totalModesCount_ += int(v)

    self.phis_      = {}
    for tileId in range(self.nTiles_):
      myK   = self.modes_[tileId]
      myPhi = load_basis_from_binary_file(podDir + "/lsv_state_p_" + str(tileId) )[:,0:myK]
      self.phis_[tileId] = myPhi

  def createRomState(self):
    return np.zeros(self.totalModesCount_)

  def viewFomStateFullMesh(self):
    return self.fomState_

  def reconstructMemberFomState(self, romStateIn):
    romStateStart = 0
    fomStateStart = 0
    for tileId in range(self.nTiles_):
      myK = self.modes_[tileId]
      myPhi = self.phis_[tileId]
      myRomStateSlice = romStateIn[romStateStart:romStateStart+myK]
      tmpy  = np.dot(myPhi, myRomStateSlice)
      self.fomState_[fomStateStart:fomStateStart+len(tmpy)] = np.copy(tmpy)
      romStateStart += myK
      fomStateStart += len(tmpy)

  def projectMemberFomVelo(self, romState):
    romStateStart = 0
    fomVeloStart = 0
    for tileId in range(self.nTiles_):
      myK = self.modes_[tileId]
      myPhi = self.phis_[tileId]
      n = myPhi.shape[0]
      myRhsSlice = self.fomVelo_[fomVeloStart:fomVeloStart+n]
      yhattmp    = np.dot(myPhi.T, myRhsSlice)
      romState[romStateStart:romStateStart+myK] = np.copy(yhattmp)
      romStateStart += myK
      fomVeloStart += n

  def run(self, workDir, yhat, fomApp, nSteps, dt):
    romRhs = np.zeros_like(yhat)
    yhat0  = np.zeros_like(yhat)
    yhat1  = np.zeros_like(yhat)

    two = 2.
    oneOverThree  = 1./3.
    oneOverFour   = 1./4.
    threeOverFour = 3./4.

    time = 0.
    for step in range(1, nSteps+1):
      print("norm = {}".format(linalg.norm(yhat)))
      if step % 50 == 0:
        print("step = ", step, "/", nSteps)

      self.reconstructMemberFomState(yhat)
      if step % 50 == 0:
        np.savetxt(workDir+"/y_rec_"+str(step)+".txt", self.fomState_)

      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs);
      yhat0[:] = yhat + dt * romRhs

      self.reconstructMemberFomState(yhat0)
      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs);
      yhat1[:] = threeOverFour*yhat + oneOverFour*yhat0 + oneOverFour*dt*romRhs

      self.reconstructMemberFomState(yhat1)
      fomApp.velocity(self.fomState_, time, self.fomVelo_)
      self.projectMemberFomVelo(romRhs);
      yhat[:] = oneOverThree*(yhat + two*yhat1 + two*dt*romRhs)

      time += dt
