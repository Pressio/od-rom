
from argparse import ArgumentParser
import sys, os
import numpy as np

from py_src.fncs_myio import *
from py_src.fncs_miscellanea import get_run_id, str2bool
from py_src.fncs_to_extract_from_yaml_input import *
from py_src.fncs_to_extract_from_mesh_info_file import *

#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  parser   = ArgumentParser()
  parser.add_argument("--wdir",    dest="workdir", required=True)
  parser.add_argument("--fomdir",  dest="fomdir",  required=True)
  parser.add_argument("--poddir",  dest="poddir",  required=True)
  parser.add_argument("--userefstate", dest="userefstate", type=str2bool)
  args    = parser.parse_args()
  workDir = args.workdir
  fomDir  = args.fomdir
  podDir  = args.poddir
  useRefState = args.userefstate

  # load fom states inside fomdir
  fomMeshDir     = find_meshdir_from_input_file(fomDir)
  fomTotCells    = find_total_cells_from_info_file(fomMeshDir)
  numDofsPerCell = find_numdofspercell_from_input_file(fomDir)
  totFomDofs     = fomTotCells*numDofsPerCell
  fomStates      = load_fom_state_snapshot_matrix([fomDir], totFomDofs, \
                                                  numDofsPerCell, False)
  if useRefState:
    refState = np.loadtxt(fomDir + "/initial_state.txt")
    for j in range(fomStates.shape[1]):
      fomStates[:,j] -= refState
  assert(fomStates.flags['F_CONTIGUOUS'])

  # read number of modes to use
  numModes = np.loadtxt(workDir+"/modes.txt", dtype=int)
  print("modes = {}".format(numModes))

  # project and reconstruct the fom states
  fomStatesReconstructed = np.zeros_like(fomStates, order='F')
  phiFile = podDir + "/lsv_state_p_0"
  phi     = load_basis_from_binary_file(phiFile)[:,0:numModes]
  tmpY    = np.dot(phi.transpose(), fomStates)
  fomStatesReconstructed = np.dot(phi, tmpY)

  # compute space-time errorw
  stErrs = []
  fomSflat = fomStates.flatten('F')
  approxSflat = fomStatesReconstructed.flatten('F')
  error    = approxSflat-fomSflat
  stErrs.append( np.linalg.norm(error)/np.linalg.norm(fomSflat) )
  stErrs.append( np.linalg.norm(error, ord=1)/np.linalg.norm(fomSflat, ord=1) )
  stErrs.append( np.linalg.norm(error, ord=np.inf)/np.linalg.norm(fomSflat, ord=np.inf) )
  np.savetxt(workDir+"/errors_space_time.txt", np.array(stErrs))

  # each column in fomStates and fomStatesReconstructed contains one time step
  numSteps = fomStates.shape[1]
  error    = fomStates-fomStatesReconstructed

  # fill error matrix errMat with different metrics
  # each row is a time step, each col is a type of norm/metric
  errMat = np.zeros((numSteps, 3))
  if useRefState:
    # if using ref state, col 0 (init condi) is zero by definition, cannot divide by zero
    errMat[0, :] = 0.
    errMat[1:, 0] = np.linalg.norm(error[:, 1:], axis=0)/np.linalg.norm(fomStates[:,1:], axis=0)
    errMat[1:, 1] = np.linalg.norm(error[:, 1:], ord=1, axis=0)/np.linalg.norm(fomStates[:,1:], ord=1, axis=0)
    errMat[1:, 2] = np.linalg.norm(error[:, 1:], ord=np.inf, axis=0)/np.linalg.norm(fomStates[:,1:], ord=np.inf, axis=0)
  else:
    errMat[:, 0] = np.linalg.norm(error, axis=0)/np.linalg.norm(fomStates, axis=0)
    errMat[:, 1] = np.linalg.norm(error, ord=1, axis=0)/np.linalg.norm(fomStates, ord=1, axis=0)
    errMat[:, 2] = np.linalg.norm(error, ord=np.inf, axis=0)/np.linalg.norm(fomStates, ord=np.inf, axis=0)

  np.savetxt(workDir+"/errors_in_time.txt", errMat)
