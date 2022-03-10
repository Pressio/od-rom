
import numpy as np
from scipy import linalg
from myio import load_basis_from_binary_file
import time, math

class OdRomMaskedGappy:
  def __init__(self,
               fomObj, physDim, numDofsPerCell, \
               partInfoDir, modesDicIn, sampleMeshPath,\
               podDir, projectorDir, \
               refState, fullMeshTotalDofs):

    self.physDim_  = physDim
    self.ndpc_     = numDofsPerCell

    self.fomObj_ = fomObj
    self.refState_ = refState
    self.modesDic_ = modesDicIn
    self.nTiles_   = len(modesDicIn.keys())
    self.totalModesCount_ = np.sum(list(self.modesDic_.values()))
    print("self.totalModesCount_ = {}".format(self.totalModesCount_))

    self.fomState_ = np.zeros(fullMeshTotalDofs)
    self.fomVelo_  = np.zeros(fullMeshTotalDofs)
    self.fullStateRows_ = {}
    self.fullVeloRows_  = {}
    self.projs_ = {}
    self.phis_ = {}
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]

      phiFile = podDir + "/lsv_state_p_" + str(tileId)
      self.phis_[tileId] = load_basis_from_binary_file(phiFile)[:,0:myK]

      myProjFile = projectorDir+'/projector_p_'+str(tileId)+'.txt'
      self.projs_[tileId] = np.loadtxt(myProjFile)

      file1 = partInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
      self.fullStateRows_[tileId] = np.loadtxt(file1, dtype=int)

      file2 = sampleMeshPath + "/sample_mesh_gids_p_"+str(tileId)+".txt"
      gids = np.loadtxt(file2, dtype=int)
      tmp = np.zeros(len(gids)*numDofsPerCell, dtype=int)
      for j in range(numDofsPerCell):
        tmp[j::numDofsPerCell] = gids*numDofsPerCell + j
      self.fullVeloRows_[tileId] = np.sort(tmp)

  # -------------------------------------------------------------------
  def createRomState(self):
    return np.zeros(self.totalModesCount_)

  # -------------------------------------------------------------------
  def viewFomStateOnFullMesh(self):
    return self.fomState_

  # -------------------------------------------------------------------
  def computeFomVelocity(self, time):
    self.fomObj_.velocity(self.fomState_, time, self.fomVelo_)

  # -------------------------------------------------------------------
  def reconstructFomState(self, romStateIn):
    romState_i = 0
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]
      myPhi = self.phis_[tileId]
      myRomStateSlice = romStateIn[romState_i:romState_i+myK]
      tmpy  = np.dot(myPhi, myRomStateSlice)

      for j,it in enumerate(self.fullStateRows_[tileId]):
        self.fomState_[it] = tmpy[j]
      romState_i += myK

    if self.refState_.any() != None:
      self.fomState_ += self.refState_

  # -------------------------------------------------------------------
  def reconstructFomStateFullMeshOrdering(self, romStateIn):
    self.reconstructFomState(romStateIn)

  # -------------------------------------------------------------------
  def projectFomVelo(self, romState):
    romState_i = 0
    for tileId in range(self.nTiles_):
      myK = self.modesDic_[tileId]
      myProjector = self.projs_[tileId]
      n = myProjector.shape[0]
      myRhsSlice = self.fomVelo_[self.fullVeloRows_[tileId]]
      yhattmp    = np.dot(myProjector.T, myRhsSlice)
      romState[romState_i:romState_i+myK] = np.copy(yhattmp)
      romState_i += myK
