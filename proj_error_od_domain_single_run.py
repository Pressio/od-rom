
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
  parser.add_argument("--infodir", dest="infodir",  required=True)
  parser.add_argument("--userefstate", dest="userefstate", type=str2bool)
  args    = parser.parse_args()
  workDir = args.workdir
  fomDir  = args.fomdir
  podDir  = args.poddir
  infoDir = args.infodir
  useRefState = args.userefstate

  # load fom states
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

  # load num modes used for each partition
  modesPerTile = np.loadtxt(workDir+"/modes_per_tile.txt", dtype=int)
  totalModes = np.sum(modesPerTile)
  numTiles = len(modesPerTile)
  print("modesPerTile = {}".format(modesPerTile))

  # calculate
  fomStatesReconstructed = np.zeros_like(fomStates, order='F')
  romState_i = 0
  for tileId in range(numTiles):
    myK = modesPerTile[tileId]
    phiFile = podDir + "/lsv_state_p_" + str(tileId)
    myPhi   = load_basis_from_binary_file(phiFile)[:,0:myK]
    srVecFile       = infoDir + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
    myFullStateRows = np.loadtxt(srVecFile, dtype=int)

    mySliceOfFomSnaps = fomStates[myFullStateRows, :]
    tmpY   = np.dot(myPhi.transpose(), mySliceOfFomSnaps)
    tmpRec = np.dot(myPhi, tmpY)

    for j,it in enumerate(myFullStateRows):
      fomStatesReconstructed[it, :] = tmpRec[j, :]

    # update the index so that the span of the romState
    # in the next tile is correct
    romState_i += myK

  # compute space-time errorw
  stErrs = []
  fomSflat = fomStates.flatten('F')
  approxSflat = fomStatesReconstructed.flatten('F')
  error = approxSflat-fomSflat
  stErrs.append( np.linalg.norm(error)/np.linalg.norm(fomSflat) )
  stErrs.append( np.linalg.norm(error, ord=1)/np.linalg.norm(fomSflat, ord=1) )
  stErrs.append( np.linalg.norm(error, ord=np.inf)/np.linalg.norm(fomSflat, ord=np.inf) )
  np.savetxt(workDir+"/errors_space_time.txt", np.array(stErrs))

  # compute error at each time step
  # each column in fomStates or approximation contains one step
  numSteps = fomStates.shape[1]
  error    = fomStates-fomStatesReconstructed
  print(error.shape)

  # fill error matrix errMat with different metrics
  # each row is a time step, each col is a type of norm/metric
  errMat = np.zeros((numSteps, 3))
  if useRefState:
    # if using ref state, col 0 (i.e. init condi) is zero by definition
    # so we cannot devide by zero
    errMat[0, :] = 0.
    errMat[1:, 0] = np.linalg.norm(error[:, 1:], axis=0)/np.linalg.norm(fomStates[:,1:], axis=0)
    errMat[1:, 1] = np.linalg.norm(error[:, 1:], ord=1, axis=0)/np.linalg.norm(fomStates[:,1:], ord=1, axis=0)
    errMat[1:, 2] = np.linalg.norm(error[:, 1:], ord=np.inf, axis=0)/np.linalg.norm(fomStates[:,1:], ord=np.inf, axis=0)
  else:
    errMat[:, 0] = np.linalg.norm(error, axis=0)/np.linalg.norm(fomStates, axis=0)
    errMat[:, 1] = np.linalg.norm(error, ord=1, axis=0)/np.linalg.norm(fomStates, ord=1, axis=0)
    errMat[:, 2] = np.linalg.norm(error, ord=np.inf, axis=0)/np.linalg.norm(fomStates, ord=np.inf, axis=0)

  np.savetxt(workDir+"/errors_in_time.txt", errMat)
