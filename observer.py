
import numpy as np
from myio import write_matrix_to_bin

class FomObserver:
  def __init__(self, N, sf, vf, numSteps):
    self.f_     = [int(sf), int(vf)]
    self.count_ = [int(0), int(0)]

    totalStateSnaps = int(numSteps/sf)
    self.sM_ = np.zeros((totalStateSnaps,N), order='F')
    totalRhsSnaps = int(numSteps/vf)
    self.vM_ = np.zeros((totalRhsSnaps,N), order='F')

  def __call__(self, step, sIn, vIn):
    if step % self.f_[0] == 0:
      self.sM_[self.count_[0], :] = np.copy(sIn)
      self.count_[0] += 1

    if step % self.f_[1] == 0:
      self.vM_[self.count_[1], :] = np.copy(vIn)
      self.count_[1] += 1

  def write(self, outDir):
    # note that we don't need to tranpose here before writing and don't write shape
    write_matrix_to_bin(outDir+"/fom_snaps_state", self.sM_, False, False)
    write_matrix_to_bin(outDir+"/fom_snaps_rhs",   self.vM_, False, False)
