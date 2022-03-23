
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import random, subprocess
import matplotlib.pyplot as plt
import re, os, time, yaml
import numpy as np
from scipy import linalg as scipyla
from decimal import Decimal
from scipy import optimize as sciop

# local imports
from myio import *
# from odrom_full import *
# from odrom_gappy import *
# from odrom_masked_gappy import *
# from odrom_time_integrators import *
# from standardrom_full import *

# -------------------------------------------------------------------
def get_run_id(runDir):
  return int(runDir.split('_')[-1])

# -------------------------------------------------------------------
def find_fom_test_dir_with_id(workDir, targetid):
  myl = [workDir+'/'+d for d in os.listdir(workDir) \
                 if "fom_test" in d and get_run_id(d)==targetid]
  assert(len(myl)==1)
  return myl[0]

# -------------------------------------------------------------------
def find_all_standard_rom_dirs(workDir):
  myl = [workDir+'/'+d for d in os.listdir(workDir) \
         if "standardrom" in d]
  myl = sorted(myl, key=get_run_id)
  return myl

# -------------------------------------------------------------------
def find_all_odrom_nohr_dirs(workDir):
  myl = [workDir+'/'+d for d in os.listdir(workDir) \
         if "odrom_full" in d]
  myl = sorted(myl, key=get_run_id)
  return myl

# -------------------------------------------------------------------
def find_partition_info_path_from_input_file(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["partioningInfo"]

# -------------------------------------------------------------------
def find_state_pod_modes_path_from_input_file(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["basesDir"]

# -------------------------------------------------------------------
def find_state_sampling_frequency_from_input_file(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["stateSamplingFreq"]

# -------------------------------------------------------------------
def find_numdofspercell_from_input_file(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["numDofsPerCell"]

# -------------------------------------------------------------------
def find_meshdir_from_input_file(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["meshDir"]

# -------------------------------------------------------------------
def using_ic_as_reference_state(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["usingIcAsRefState"]

# -------------------------------------------------------------------
def find_dimensionality_from_info_file(workDir):
  reg = re.compile(r'dim.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_num_cells_from_info_file(workDir, ns):
  reg = re.compile(r''+ns+'.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_total_cells_from_info_file(workDir):
  dims = find_dimensionality_from_info_file(workDir)
  if dims == 1:
    return find_num_cells_from_info_file(workDir, "nx")
  elif dims==2:
    nx = find_num_cells_from_info_file(workDir, "nx")
    ny = find_num_cells_from_info_file(workDir, "ny")
    return nx*ny
  else:
    sys.exit("Invalid dims = {}".format(dims))

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
  np.savetxt(outDir+"/space_time_errors.txt", np.array(stErrs))


# -------------------------------------------------------------------
def compute_errors_for_standard_rom_dir(workDir, romDir):
  # extract the runid from the rom dir
  myRunId = get_run_id(romDir)

  # find the fom test dir that matches the id
  fomTestDir = find_fom_test_dir_with_id(workDir, myRunId)
  print(fomTestDir)

  # load fom states
  fomMeshDir     = find_meshdir_from_input_file(fomTestDir)
  fomTotCells    = find_total_cells_from_info_file(fomMeshDir)
  numDofsPerCell = find_numdofspercell_from_input_file(fomTestDir)
  totFomDofs     = fomTotCells*numDofsPerCell

  fomStates = load_fom_state_snapshot_matrix([fomTestDir], totFomDofs, \
                                             numDofsPerCell, False)
  assert(fomStates.flags['F_CONTIGUOUS'])

  # ensure that the sampling frequency of the state
  # is the same in rom and fom runs
  fomSampFreq = find_state_sampling_frequency_from_input_file(fomTestDir)
  romSampFreq = find_state_sampling_frequency_from_input_file(romDir)
  if fomSampFreq != romSampFreq:
    sys.exit("mismatching sampling freq: fomSampFreq != romSampFreq:")

  # load num of modes used
  numModes = int(np.loadtxt(romDir+"/rom_dofs_count.txt"))
  print("numModes = {}".format(numModes))

  # load pods
  podDir  = find_state_pod_modes_path_from_input_file(romDir)
  phiFile = podDir + "/lsv_state_p_0"
  myPhi = load_basis_from_binary_file(phiFile)[:,0:numModes]
  print("myPhi.shape = {}".format(myPhi.shape))
  fomNumDofs = myPhi.shape[0]

  refState = np.zeros(fomNumDofs)
  if using_ic_as_reference_state(romDir):
    refState = np.loadtxt(fomTestDir + "/initial_state.txt")

  # load rom states
  romStateSnaps = load_rom_state_snapshot_matrix(romDir, numModes)
  assert(romStateSnaps.flags['F_CONTIGUOUS'])

  # reconstruct all fom snaps
  fomStateReconstructed = np.dot(myPhi, romStateSnaps)
  for j in range(fomStateReconstructed.shape[1]):
    fomStateReconstructed[:,j] += refState
  print("fomStateReconstructed.shape = {}".format(fomStateReconstructed.shape))

  compute_and_save_errors(romDir, fomStates, fomStateReconstructed)
  #plt.plot(fomStateReconstructed.flatten('F')[160000:162000], '--k')
  #plt.plot(fomStates.flatten('F')[160000:162000], '-r')
  #plt.show()


# -------------------------------------------------------------------
def compute_errors_for_odrom_nohr_dir(workDir, romDir):
  # extract the runid from the rom dir
  myRunId = get_run_id(romDir)

  # find the fom test dir that matches the id
  fomTestDir = find_fom_test_dir_with_id(workDir, myRunId)
  print(fomTestDir)

  # load fom states
  fomMeshDir     = find_meshdir_from_input_file(fomTestDir)
  fomTotCells    = find_total_cells_from_info_file(fomMeshDir)
  numDofsPerCell = find_numdofspercell_from_input_file(fomTestDir)
  totFomDofs     = fomTotCells*numDofsPerCell
  fomStates = load_fom_state_snapshot_matrix([fomTestDir], totFomDofs, \
                                             numDofsPerCell, False)
  assert(fomStates.flags['F_CONTIGUOUS'])

  # ensure that the sampling frequency of the state
  # is the same in rom and fom runs
  fomSampFreq = find_state_sampling_frequency_from_input_file(fomTestDir)
  romSampFreq = find_state_sampling_frequency_from_input_file(romDir)
  if fomSampFreq != romSampFreq:
    sys.exit("mismatching sampling freq: fomSampFreq != romSampFreq:")

  # load modes used for each partition
  modesPerTile = np.loadtxt(romDir+"/modes_per_tile.txt", dtype=int)
  totalModes = np.sum(modesPerTile)
  numTiles = len(modesPerTile)
  print("modesPerTile = {}".format(modesPerTile))

  # load rom states
  romStateSnaps = load_rom_state_snapshot_matrix(romDir, totalModes)
  assert(romStateSnaps.flags['F_CONTIGUOUS'])

  fomStateReconstructed = np.zeros_like(fomStates, order='F')
  partitioningInfo      = find_partition_info_path_from_input_file(romDir)
  podDir                = find_state_pod_modes_path_from_input_file(romDir)
  romState_i = 0
  for tileId in range(numTiles):
    myK = modesPerTile[tileId]
    phiFile = podDir + "/lsv_state_p_" + str(tileId)
    myPhi   = load_basis_from_binary_file(phiFile)[:,0:myK]
    srVecFile       = partitioningInfo+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
    myFullStateRows = np.loadtxt(srVecFile, dtype=int)

    # myslice of romStates
    myRomStatesSlice = romStateSnaps[romState_i:romState_i+myK, :]
    tmpy = np.dot(myPhi, myRomStatesSlice)
    for j,it in enumerate(myFullStateRows):
      fomStateReconstructed[it, :] = tmpy[j, :]

    # update the index so that the span of the romState
    # in the next tile is correct
    romState_i += myK

  # reference state
  refState = np.zeros(totFomDofs)
  if using_ic_as_reference_state(romDir):
    refState = np.loadtxt(fomTestDir + "/initial_state.txt")
  for j in range(fomStateReconstructed.shape[1]):
    fomStateReconstructed[:,j] += refState

  compute_and_save_errors(romDir, fomStates, fomStateReconstructed)

#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  parser   = ArgumentParser()
  parser.add_argument("--wdir", "--workdir", dest="workdir", required=True)
  args     = parser.parse_args()
  workDir  = args.workdir

  # find all standard rom dirs
  for dirIt in find_all_standard_rom_dirs(args.workdir):
    print(dirIt)
    compute_errors_for_standard_rom_dir(args.workdir, dirIt)

  # find all odrom dirs without hyperreduction
  for dirIt in find_all_odrom_nohr_dirs(args.workdir):
    print(dirIt)
    compute_errors_for_odrom_nohr_dir(args.workdir, dirIt)
