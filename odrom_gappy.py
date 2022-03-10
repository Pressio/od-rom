
import numpy as np
from scipy import linalg
from myio import load_basis_from_binary_file
import time, math

def make_bases_on_stenci_mesh(partInfoDir, tileId, sampleMeshPath, podDir,\
                              numModes, numDofsPerCell):

   # need to store the bases on stencil mesh
   # so first, load on bases full local mesh
   phiFile = podDir + "/lsv_state_p_" + str(tileId)
   myPhi   = load_basis_from_binary_file(phiFile)[:,0:numModes]

   # load indices such that we can extract phi on stencil mesh
   myCellGids   = np.loadtxt(partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt",dtype=int)
   myStMeshGids = np.loadtxt(sampleMeshPath + "/stencil_mesh_gids_p_"+str(tileId)+".dat", dtype=int)
   myStCount    = len(myStMeshGids)

   commonElem  = set(myStMeshGids).intersection(myCellGids)
   commonElem  = np.sort(list(commonElem))
   mylocalinds = np.searchsorted(myCellGids, commonElem)
   mySlicedPhi = np.zeros((myStCount*numDofsPerCell, numModes), order='F')
   for j in range(numDofsPerCell):
     mySlicedPhi[j::numDofsPerCell, :] = myPhi[numDofsPerCell*mylocalinds + j, :]

   return mySlicedPhi


class OdRomGappy:
  def __init__(self,
               # fomObj operates on a reduced mesh
               # because this is odrom with real smaple mesh
               fomObj, physDim, numDofsPerCell, \
               partInfoDir, modesDicIn, sampleMeshPath,\
               podDir, projectorDir, \
               refStateFullMeshOrdering, refStateForOdRomAlgo,\
               fullMeshTotalDofs):

    # physical dimensions and dofs/cell
    self.physDim_ = physDim
    self.ndpc_    = numDofsPerCell

    self.fomObj_ = fomObj
    self.refStateFullMeshOrdering_ = refStateFullMeshOrdering
    self.refStateForOdRomAlgo_ = refStateForOdRomAlgo

    self.modesDic_ = modesDicIn
    self.nTiles_   = len(modesDicIn.keys())
    # count all mode count within each tile
    self.totalModesCount_ = np.sum(list(self.modesDic_.values()))
    print("self.totalModesCount_ = {}".format(self.totalModesCount_))

    self.fomStateFullMesh_    = np.zeros(fullMeshTotalDofs)
    self.fomStateStencilMesh_ = np.zeros(fomObj.totalDofStencilMesh())
    self.fomVelo_             = np.zeros(fomObj.totalDofSampleMesh())

    self.projs_ = {}
    self.phis_ = {}
    self.phisOnFullMesh_ = {}
    self.fullStateRows_ = {}
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]

      # phi on full mesh
      phiFile = podDir + "/lsv_state_p_" + str(tileId)
      self.phisOnFullMesh_[tileId] = load_basis_from_binary_file(phiFile)[:,0:myK]

      # phi on stencil mesh
      self.phis_[tileId] = make_bases_on_stenci_mesh(partInfoDir, tileId, \
                                                     sampleMeshPath, podDir,\
                                                     myK, numDofsPerCell)

      # projectors
      self.projs_[tileId] = np.loadtxt(projectorDir+'/projector_p_'+str(tileId)+'.txt')

      srVecFile = partInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
      self.fullStateRows_[tileId] = np.loadtxt(srVecFile, dtype=int)

  # -------------------------------------------------------------------
  def createRomState(self):
    return np.zeros(self.totalModesCount_)

  # -------------------------------------------------------------------
  def viewFomStateOnFullMesh(self):
    return self.fomStateFullMesh_

  # -------------------------------------------------------------------
  def computeFomVelocity(self, time):
    self.fomObj_.velocity(self.fomStateStencilMesh_, time, self.fomVelo_)

  # -------------------------------------------------------------------
  def reconstructFomStateFullMeshOrdering(self, romStateIn):
    romState_i = 0
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]
      myPhi = self.phisOnFullMesh_[tileId]
      myRomStateSlice = romStateIn[romState_i:romState_i+myK]
      tmpy  = np.dot(myPhi, myRomStateSlice)

      for j,it in enumerate(self.fullStateRows_[tileId]):
        self.fomStateFullMesh_[it] = tmpy[j]
      romState_i += myK

    if self.refStateFullMeshOrdering_.any() != None:
      self.fomStateFullMesh_ += self.refStateFullMeshOrdering_

  # -------------------------------------------------------------------
  def projectFomVelo(self, romState):
    romState_i = 0
    fomVelo_i = 0
    for tileId in range(self.nTiles_):
      myK = self.modesDic_[tileId]
      myProjector = self.projs_[tileId]
      n = myProjector.shape[0]
      myRhsSlice = self.fomVelo_[fomVelo_i:fomVelo_i+n]
      yhattmp    = np.dot(myProjector.T, myRhsSlice)
      romState[romState_i:romState_i+myK] = np.copy(yhattmp)
      romState_i += myK
      fomVelo_i += n

  # -------------------------------------------------------------------
  def reconstructFomState(self, romStateIn):
    romState_i = 0
    fomState_i = 0
    for tileId in range(self.nTiles_):
      myK   = self.modesDic_[tileId]
      myPhi = self.phis_[tileId]
      myRomStateSlice = romStateIn[romState_i:romState_i+myK]
      tmpy  = np.dot(myPhi, myRomStateSlice)
      self.fomStateStencilMesh_[fomState_i:fomState_i+len(tmpy)] = np.copy(tmpy)
      romState_i += myK
      fomState_i += len(tmpy)

    if self.refStateForOdRomAlgo_.any() != None:
      self.fomStateStencilMesh_ += self.refStateForOdRomAlgo_
