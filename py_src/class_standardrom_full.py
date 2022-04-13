
import time, math, logging
import numpy as np
from scipy import linalg
from .fncs_myio import load_basis_from_binary_file

class StandardRomFull:
  def __init__(self, \
               fomObj, physDim, numDofsPerCell, \
               numModes, podDir, refState):

    logger = logging.getLogger(__name__)

    # physical dimensions and dofs/cell
    self.physDim_ = physDim
    self.ndpc_    = numDofsPerCell

    self.fomObj_ = fomObj
    self.refState_ = refState

    self.numModes_ = numModes
    logger.info("self.totalModesCount_ = {}".format(self.numModes_))

    fomTotalDofs   = fomObj.totalDofStencilMesh()
    self.fomState_ = np.zeros(fomTotalDofs)
    self.fomVelo_  = np.zeros(fomTotalDofs)

    phiFile = podDir + "/lsv_state_p_0"
    self.phi_ = load_basis_from_binary_file(phiFile)[:,0:numModes]

  # -------------------------------------------------------------------
  def createRomState(self):
    return np.zeros(self.numModes_)

  # -------------------------------------------------------------------
  def viewFomState(self):
    return self.fomState_

  # -------------------------------------------------------------------
  def computeFomVelocity(self, time):
    self.fomObj_.velocity(self.fomState_, time, self.fomVelo_)

  # -------------------------------------------------------------------
  def reconstructFomStateFullMeshOrdering(self, romStateIn):
    self.fomState_  = np.dot(self.phi_, romStateIn)

    if self.refState_.any() != None:
      self.fomState_ += self.refState_

  # -------------------------------------------------------------------
  def reconstructFomState(self, romStateIn):
    self.reconstructFomStateFullMeshOrdering(romStateIn)

  # -------------------------------------------------------------------
  def projectFomVelo(self, romState):
    romState[:] = np.dot(self.phi_.T, self.fomVelo_)
