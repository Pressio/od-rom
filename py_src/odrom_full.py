
import time, math
import numpy as np
from scipy import linalg
from .myio import load_basis_from_binary_file

class OdRomFull:
  def __init__(self, \
               fomObj, physDim, numDofsPerCell, \
               partInfoDir, modesDicIn, podDir, \
               refStateFullMeshOrdering,  refStateForOdRomAlgo):

    # physical dimensions and dofs/cell
    self.physDim_ = physDim
    self.ndpc_    = numDofsPerCell

    self.fomObj_ = fomObj
    self.refStateFullMeshOrdering_ = refStateFullMeshOrdering
    self.refStateForOdRomAlgo_     = refStateForOdRomAlgo

    self.modesDic_ = modesDicIn
    self.nTiles_   = len(modesDicIn.keys())
    # count all mode count within each tile
    self.totalModesCount_ = np.sum(list(self.modesDic_.values()))
    print("self.totalModesCount_ = {}".format(self.totalModesCount_))

    fomTotalDofs   = fomObj.totalDofStencilMesh()
    self.fomState_ = np.zeros(fomTotalDofs)
    self.fomVelo_  = np.zeros(fomTotalDofs)

    # phis_: key = tileId, value = local bases
    self.phis_ = {}

    # fullStateRows_:
    #   key = tileId
    #   value = indices of rows in the full state that belong to me
    self.fullStateRows_ = {}
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]

      phiFile = podDir + "/lsv_state_p_" + str(tileId)
      myPhi = load_basis_from_binary_file(phiFile)[:,0:myK]
      self.phis_[tileId] = myPhi

      srVecFile = partInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
      self.fullStateRows_[tileId] = np.loadtxt(srVecFile, dtype=int)

  # -------------------------------------------------------------------
  def createRomState(self):
    return np.zeros(self.totalModesCount_)

  # -------------------------------------------------------------------
  def viewFomState(self):
    return self.fomState_

  # -------------------------------------------------------------------
  def computeFomVelocity(self, time):
    self.fomObj_.velocity(self.fomState_, time, self.fomVelo_)

  # -------------------------------------------------------------------
  def reconstructFomStateFullMeshOrdering(self, romStateIn):
    romState_i = 0
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]
      myPhi = self.phis_[tileId]
      myRomStateSlice = romStateIn[romState_i:romState_i+myK]
      tmpy  = np.dot(myPhi, myRomStateSlice)

      for j,it in enumerate(self.fullStateRows_[tileId]):
        self.fomState_[it] = tmpy[j]
      romState_i += myK

    if self.refStateFullMeshOrdering_.any() != None:
      self.fomState_ += self.refStateFullMeshOrdering_

  # -------------------------------------------------------------------
  def reconstructFomState(self, romStateIn):
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

    if self.refStateForOdRomAlgo_.any() != None:
      self.fomState_ += self.refStateForOdRomAlgo_

  # -------------------------------------------------------------------
  def projectFomVelo(self, romState):
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
