
import numpy as np
import sys, logging
from .fncs_myio import write_matrix_to_bin_omit_shape

class FomObserver:
  def __init__(self, stateDofsCount, \
               samplingFreqState, samplingFreqVelocity, \
               numSteps, dt):
    logger = logging.getLogger(__name__)

    self.dt_    = dt
    self.f_     = [int(samplingFreqState), int(samplingFreqVelocity)]
    self.count_ = [int(0), int(0)]

    if numSteps % samplingFreqState != 0:
      logger.error("numSteps not divisible by samplingFreqState")
      sys.exit(1)

    if numSteps % samplingFreqVelocity != 0:
      logger.error("numSteps not divisible by samplingFreqVelocity")
      sys.exit(1)

    totalStateSnaps = int(numSteps/samplingFreqState)+1
    totalRhsSnaps   = int(numSteps/samplingFreqVelocity)+1
    self.sM_ = np.zeros((totalStateSnaps,stateDofsCount), order='C')
    self.vM_ = np.zeros((totalRhsSnaps,  stateDofsCount), order='C')
    self.stateSnapsTimes_ = np.zeros((totalStateSnaps, 2))
    self.velocSnapsTimes_ = np.zeros((totalRhsSnaps,   2))

  def __call__(self, step, sIn, vIn):
    if step % self.f_[0] == 0:
      self.sM_[self.count_[0], :] = np.copy(sIn)
      self.stateSnapsTimes_[self.count_[0], 0] = step
      self.stateSnapsTimes_[self.count_[0], 1] = step*self.dt_
      self.count_[0] += 1

    if step % self.f_[1] == 0:
      self.vM_[self.count_[1], :] = np.copy(vIn)
      self.velocSnapsTimes_[self.count_[1], 0] = step
      self.velocSnapsTimes_[self.count_[1], 1] = step*self.dt_
      self.count_[1] += 1

  def write(self, outDir):
    # note that we don't need to tranpose here before writing and don't write shape
    write_matrix_to_bin_omit_shape(outDir+"/fom_snaps_state", self.sM_, False)
    write_matrix_to_bin_omit_shape(outDir+"/fom_snaps_rhs",   self.vM_, False)
    np.savetxt(outDir+"/fom_snaps_state_steps_and_times.txt", self.stateSnapsTimes_)
    np.savetxt(outDir+"/fom_snaps_rhs_steps_and_times.txt",   self.velocSnapsTimes_)
