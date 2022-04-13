
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

  #-----------------------------------
  # load fom states 
  #-----------------------------------
  fomMeshDir     = find_meshdir_from_input_file(fomDir)
  fomTotCells    = find_total_cells_from_info_file(fomMeshDir)
  numDofsPerCell = find_numdofspercell_from_input_file(fomDir)
  totFomDofs     = fomTotCells*numDofsPerCell
  rawFomStates   = load_fom_state_snapshot_matrix([fomDir], totFomDofs, \
                                                  numDofsPerCell, False)
  assert(rawFomStates.flags['F_CONTIGUOUS'])

  fomStatesPossiblyCentered = rawFomStates.copy()
  if useRefState:
    refState = np.loadtxt(fomDir + "/initial_state.txt")
    for j in range(fomStatesPossiblyCentered.shape[1]):
      fomStatesPossiblyCentered[:,j] -= refState

  #-----------------------------------
  # load num modes for each partition
  #-----------------------------------
  modesPerTile = np.loadtxt(workDir+"/modes_per_tile.txt", dtype=int)
  totalModes = np.sum(modesPerTile)
  numTiles = len(modesPerTile)
  print("modesPerTile = {}".format(modesPerTile))

  #-----------------------------------
  # project and reconstruct the fom states
  #-----------------------------------
  fomStatesReconstructed = np.zeros_like(fomStatesPossiblyCentered, order='F')
  for tileId in range(numTiles):
    myK = modesPerTile[tileId]
    phiFile = podDir + "/lsv_state_p_" + str(tileId)
    myPhi   = load_basis_from_binary_file(phiFile)[:,0:myK]
    srVecFile       = infoDir + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
    myFullStateRows = np.loadtxt(srVecFile, dtype=int)

    mySliceOfFomSnaps = fomStatesPossiblyCentered[myFullStateRows, :]
    tmpY   = np.dot(myPhi.transpose(), mySliceOfFomSnaps)
    tmpRec = np.dot(myPhi, tmpY)

    for j,it in enumerate(myFullStateRows):
      fomStatesReconstructed[it, :] = tmpRec[j, :]

  if useRefState:
    refState = np.loadtxt(fomDir + "/initial_state.txt")
    for j in range(fomStatesReconstructed.shape[1]):
      fomStatesReconstructed[:,j] += refState

  #-----------------------------------
  # compute space-time error
  #-----------------------------------
  stErrs = []
  fomSflat = rawFomStates.flatten('F')
  approxSflat = fomStatesReconstructed.flatten('F')
  error = approxSflat-fomSflat
  stErrs.append( np.linalg.norm(error)/np.linalg.norm(fomSflat) )
  stErrs.append( np.linalg.norm(error, ord=1)/np.linalg.norm(fomSflat, ord=1) )
  stErrs.append( np.linalg.norm(error, ord=np.inf)/np.linalg.norm(fomSflat, ord=np.inf) )
  np.savetxt(workDir+"/errors_space_time.txt", np.array(stErrs))

  #-----------------------------------
  # compute error at each time step
  #-----------------------------------
  # each column in rawFomStates or approximation contains one step
  numSteps = rawFomStates.shape[1]
  error    = rawFomStates-fomStatesReconstructed
  # fill error matrix errMat
  # each row is a time step, each col is a norm/metric
  errMat = np.zeros((numSteps, 3))
  errMat[:, 0] = np.linalg.norm(error, axis=0)/np.linalg.norm(rawFomStates, axis=0)
  errMat[:, 1] = np.linalg.norm(error, ord=1, axis=0)/np.linalg.norm(rawFomStates, ord=1, axis=0)
  errMat[:, 2] = np.linalg.norm(error, ord=np.inf, axis=0)/np.linalg.norm(rawFomStates, ord=np.inf, axis=0)
  np.savetxt(workDir+"/errors_in_time.txt", errMat)
