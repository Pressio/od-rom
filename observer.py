
import numpy as np
from myio import write_matrix_to_bin

# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
class RomObserver:
  def __init__(self, sf, numSteps, modesPerTile):
    self.f_     = int(sf)
    self.count_ = int(0)

    totNumModes = np.sum(list(modesPerTile.values()))
    totalStateSnaps = int(numSteps/sf)
    self.sM_ = np.zeros((totalStateSnaps, totNumModes), order='F')

  def __call__(self, step, romState):
    if step % self.f_ == 0:
      self.sM_[self.count_, :] = np.copy(romState)
      self.count_ += 1

  def write(self, outDir):
    # note that final False, Flase is to indicate
    # we don't need to transpose here before writing and don't write shape
    write_matrix_to_bin(outDir+"/rom_snaps_state", self.sM_, False, False)
