
import numpy as np
import sys
from .myio import write_matrix_to_bin_omit_shape

# -------------------------------------------------------------------
class FomObserver:
  def __init__(self, stateDofsCount, samplingFreqState, samplingFreqVelocity, numSteps):
    self.f_     = [int(samplingFreqState), int(samplingFreqVelocity)]
    self.count_ = [int(0), int(0)]

    totalStateSnaps = int(numSteps/samplingFreqState)
    totalRhsSnaps   = int(numSteps/samplingFreqVelocity)
    self.sM_ = np.zeros((totalStateSnaps,stateDofsCount), order='C')
    self.vM_ = np.zeros((totalRhsSnaps,  stateDofsCount), order='C')

  def __call__(self, step, sIn, vIn):
    if step % self.f_[0] == 0:
      self.sM_[self.count_[0], :] = np.copy(sIn)
      self.count_[0] += 1

    if step % self.f_[1] == 0:
      self.vM_[self.count_[1], :] = np.copy(vIn)
      self.count_[1] += 1

  def write(self, outDir):
    # note that we don't need to tranpose here before writing and don't write shape
    write_matrix_to_bin_omit_shape(outDir+"/fom_snaps_state", self.sM_, False)
    write_matrix_to_bin_omit_shape(outDir+"/fom_snaps_rhs",   self.vM_, False)
    #np.savetxt(outDir+"/fom_snaps_state.txt", self.sM_)
    #np.savetxt(outDir+"/fom_snaps_rhs.txt",   self.vM_)

# -------------------------------------------------------------------
class RomObserver:
  def __init__(self, samplingFreqState, numSteps, modesPerTile):
    self.f_     = int(samplingFreqState)
    self.count_ = int(0)

    if numSteps % samplingFreqState !=0:
      sys.exit("RomObserver: numSteps not divisible by samplingFreqState")

    totNumModes = np.sum(list(modesPerTile.values()))
    totalStateSnaps = int(numSteps/samplingFreqState)
    self.sM_ = np.zeros((totalStateSnaps, totNumModes), order='C')

  def __call__(self, step, romState):
    if step % self.f_ == 0:
      self.sM_[self.count_, :] = np.copy(romState)
      self.count_ += 1

  def write(self, outDir):
    # note that final False, Flase is to indicate
    # we don't need to transpose here before writing and don't write shape
    write_matrix_to_bin_omit_shape(outDir+"/rom_snaps_state", self.sM_, False)
