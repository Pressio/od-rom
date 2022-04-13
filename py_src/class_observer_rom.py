
import numpy as np
import sys
from .fncs_myio import write_matrix_to_bin_omit_shape

class RomObserver:
  def __init__(self, samplingFreqState, numSteps, modesPerTile, dt):
    self.f_     = int(samplingFreqState)
    self.count_ = int(0)
    self.dt_    = dt

    if numSteps % samplingFreqState !=0:
      sys.exit("RomObserver: numSteps not divisible by samplingFreqState")

    totalStateSnaps = int(numSteps/samplingFreqState) + 1
    totNumModes = np.sum(list(modesPerTile.values()))
    self.sM_ = np.zeros((totalStateSnaps, totNumModes), order='C')
    self.stateSnapsTimes_ = np.zeros((totalStateSnaps, 2))

  def __call__(self, step, romState):
    if step % self.f_ == 0:
      self.sM_[self.count_, :] = np.copy(romState)
      self.stateSnapsTimes_[self.count_, 0] = step
      self.stateSnapsTimes_[self.count_, 1] = step*self.dt_
      self.count_ += 1

  def write(self, outDir):
    # note that final False, Flase is to indicate
    # we don't need to transpose here before writing and don't write shape
    write_matrix_to_bin_omit_shape(outDir+"/rom_snaps_state", self.sM_, False)
    np.savetxt(outDir+"/rom_snaps_state_steps_and_times.txt", self.stateSnapsTimes_)
