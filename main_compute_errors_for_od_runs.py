
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, subprocess, pathlib, logging
import numpy as np
from scipy import linalg as scipyla

from py_src.fncs_banners_and_prints import *

from py_src.fncs_miscellanea import \
  get_run_id, find_all_partitions_info_dirs, \
  make_modes_per_tile_dic_with_const_modes_count

from py_src.fncs_myio import *

from py_src.fncs_fom_run_dirs_detection import \
  find_fom_test_dir_with_target_id

from py_src.fncs_to_extract_from_mesh_info_file import *
from py_src.fncs_to_extract_from_yaml_input import *

# -------------------------------------------------------------------
def find_all_odrom_dirs(workDir):
  myl = [workDir+'/'+d for d in os.listdir(workDir) if "odrom_" in d[:7]]
  myl = sorted(myl, key=get_run_id)
  return myl

# -------------------------------------------------------------------
def compute_and_save_errors(outDir, fomStates, approximation):
  assert(fomStates.shape == approximation.shape)

  # compute error at each time step
  # each column in fomStates or approximation contains one step
  numSteps = fomStates.shape[1]
  errMat   = np.zeros((numSteps, 3))
  error = fomStates-approximation
  errMat[:, 0] = np.linalg.norm(error, axis=0)/np.linalg.norm(fomStates, axis=0)
  errMat[:, 1] = np.linalg.norm(error, ord=1, axis=0)/np.linalg.norm(fomStates, ord=1, axis=0)
  errMat[:, 2] = np.linalg.norm(error, ord=np.inf, axis=0)/np.linalg.norm(fomStates, ord=np.inf, axis=0)
  np.savetxt(outDir+"/errors_in_time.txt", errMat)

  # compute space-time errorw
  stErrs = []
  fomSflat = fomStates.flatten('F')
  approxSflat = approximation.flatten('F')
  error    = approxSflat-fomSflat
  stErrs.append( np.linalg.norm(error)/np.linalg.norm(fomSflat) )
  stErrs.append( np.linalg.norm(error, ord=1)/np.linalg.norm(fomSflat, ord=1) )
  stErrs.append( np.linalg.norm(error, ord=np.inf)/np.linalg.norm(fomSflat, ord=np.inf) )
  np.savetxt(outDir+"/errors_space_time.txt", np.array(stErrs))

# -------------------------------------------------------------------
def check_if_error_files_already_present(targetDir):
  b1 = os.path.exists(targetDir+"/errors_space_time.txt")
  b2 = os.path.exists(targetDir+"/errors_in_time.txt")
  return b1 and b2

# -------------------------------------------------------------------
def compute_errors_for_odrom_dir(workDir, romDir):
  logging.debug("\033[0;37;46mromDir     = {} {}".format(os.path.basename(romDir), color_resetter()))

  # check if the rom dir already contains errors, if so exit
  if check_if_error_files_already_present(romDir):
    logging.info("errors already computed, leaving dir")
    return

  # find the fom test dir that matches the target rom dir id
  myRunId = get_run_id(romDir)
  fomTestDir = find_fom_test_dir_with_target_id(workDir, myRunId)
  logging.debug("\033[0;37;46mfomTestDir = {} {}".format(os.path.basename(fomTestDir), color_resetter()))

  # -------------------------------------
  # load fom snapshots
  # -------------------------------------
  fomMeshDir     = find_meshdir_from_input_file(fomTestDir)
  fomTotCells    = find_total_cells_from_info_file(fomMeshDir)
  numDofsPerCell = find_numdofspercell_from_input_file(fomTestDir)
  totFomDofs     = fomTotCells*numDofsPerCell
  fomStates      = load_fom_state_snapshot_matrix([fomTestDir], totFomDofs, \
                                                  numDofsPerCell, False)
  assert(fomStates.flags['F_CONTIGUOUS'])

  # -------------------------------------
  # load rom states
  # -------------------------------------
  modesPerTile = np.loadtxt(romDir+"/modes_per_tile.txt", dtype=int)
  totalModes = np.sum(modesPerTile)
  numTiles   = len(modesPerTile)
  romStates  = load_rom_state_snapshot_matrix(romDir, totalModes)
  assert(romStates.flags['F_CONTIGUOUS'])

  # need to be careful because the rom and fom can have a different
  # sampling frequency and different time step sizes.
  # We need to figure out what are the times collected by the fom and rom
  # and then intersect to find the corresponding states that we need
  # to compare to compute the errors
  fomStatesCollectionTimes = np.loadtxt(fomTestDir+"/fom_snaps_state_steps_and_times.txt", dtype=float)[:,1]
  logging.debug("fomStatesCollectionTimes = {}".format(fomStatesCollectionTimes))
  logging.info("")
  assert(len(fomStatesCollectionTimes) == fomStates.shape[1])
  maxFomTime = np.max(fomStatesCollectionTimes)

  romStatesCollectionTimes = np.loadtxt(romDir+"/rom_snaps_state_steps_and_times.txt", dtype=float)[:,1]
  logging.debug("romStatesCollectionTimes = {}".format(romStatesCollectionTimes))
  logging.info("")
  romStatesCollectionTimesToUse = romStatesCollectionTimes
  # need to figure out if the rom run failed:
  # to do so, we find the largest time that the fom reached and see where
  # it is inside romStatesCollectionTimes. The same time should be the last entry
  # in romStatesCollectionTimes, if this is not the case, then the ROM run failed
  romRunFailed = False
  indexInRomTimes = int(np.where(romStatesCollectionTimes==maxFomTime)[0])
  if indexInRomTimes != len(romStatesCollectionTimes)-1:
    romRunFailed = True
    romStatesCollectionTimesToUse = romStatesCollectionTimesToUse[:indexInRomTimes]
  logging.debug("romStatesCollectionTimesToUse = {}".format(romStatesCollectionTimesToUse))
  print("")

  # find which col indices from fom and rom states I need to keep
  # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
  fomColIndicesToKeep = np.where(np.in1d(fomStatesCollectionTimes, romStatesCollectionTimesToUse))[0]
  logging.debug("fomColIndicesToKeep = {}".format(fomColIndicesToKeep))
  if len(fomColIndicesToKeep) == 0:
    logging.error("fomColIndicesToKeep cannot be empty, something wrong, terminating")
    sys.exit()

  romColIndicesToKeep = np.where(np.in1d(romStatesCollectionTimesToUse, fomStatesCollectionTimes))[0]
  logging.debug("romColIndicesToKeep = {}".format(romColIndicesToKeep))
  if len(romColIndicesToKeep) == 0:
    logging.error("romColIndicesToKeep cannot be empty, something wrong, terminating")
    sys.exit()

  # now that we have the target col indices, select only those states
  selectedFomStates = fomStates[:, fomColIndicesToKeep]
  selectedRomStates = romStates[:, romColIndicesToKeep]

  fomStateReconstructed = np.zeros_like(selectedFomStates, order='F')
  partitioningInfo      = find_partition_info_path_from_input_file(romDir)
  # here we want to ALWAYS use the FULL pod modes within each tile
  # so that we can reconstruct on the FULL mesh
  podDir = find_state_full_pod_modes_path_from_input_file(romDir)
  romState_i = 0
  for tileId in range(numTiles):
    myK = modesPerTile[tileId]
    phiFile         = podDir + "/lsv_state_p_" + str(tileId)
    myPhi           = load_basis_from_binary_file(phiFile)[:,0:myK]
    srVecFile       = partitioningInfo+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
    myFullStateRows = np.loadtxt(srVecFile, dtype=int)

    # myslice of romStates
    myRomStatesSlice = selectedRomStates[romState_i:romState_i+myK, :]
    tmpy = np.dot(myPhi, myRomStatesSlice)
    for j,it in enumerate(myFullStateRows):
      fomStateReconstructed[it, :] = tmpy[j, :]

    # update the index so that the span of the romState
    # in the next tile is correct
    romState_i += myK

  # reference state
  refState = np.zeros(totFomDofs)
  if find_if_using_ic_as_reference_state_from_yaml(romDir):
    refState = np.loadtxt(fomTestDir + "/initial_state.txt")
    for j in range(fomStateReconstructed.shape[1]):
      fomStateReconstructed[:,j] += refState

  compute_and_save_errors(romDir, selectedFomStates, fomStateReconstructed)

# -------------------------------------------------------------------
def setLogger():
  dateFmt = '%Y-%m-%d'
  logFmt2 = '%(levelname)-8s: [%(name)s] %(message)s'
  logging.basicConfig(format=logFmt2, encoding='utf-8', level=logging.DEBUG)

#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  setLogger();
  banner_driving_script_info(os.path.basename(__file__))

  parser   = ArgumentParser()
  parser.add_argument("--wdir", dest="workdir", required=True)
  args     = parser.parse_args()
  workDir  = args.workdir

  # make sure the workdir exists
  if not os.path.exists(workDir):
    sys.exit("Working dir {} does not exist, terminating".format(workDir))

  banner_import_problem()
  scenario = read_scenario_from_dir(workDir)
  problem  = read_problem_name_from_dir(workDir)
  module   = importlib.import_module(problem)
  check_and_print_problem_summary(problem, module)
  logging.info("")

  for dirIt in find_all_odrom_dirs(args.workdir):
    compute_errors_for_odrom_dir(args.workdir, dirIt)
    logging.info("")
