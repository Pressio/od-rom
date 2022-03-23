
import numpy as np
import sys, os, re, yaml

# -------------------------------------------------------------------
def write_scenario_to_file(scenarioId, outDir):
  f = open(outDir+"/scenario_id.txt", "w")
  f.write(str(scenarioId))
  f.close()

# -------------------------------------------------------------------
def read_scenario_from_dir(dirFrom):
  return int(np.loadtxt(dirFrom+"/scenario_id.txt"))

# -------------------------------------------------------------------
def write_problem_name_to_file(problemName, outDir):
  f = open(outDir+"/problem.txt", "w")
  f.write(problemName)
  f.close()

# -------------------------------------------------------------------
def read_problem_name_from_dir(dirFrom):
  with open (dirFrom+"/problem.txt", "r") as myfile:
    data=myfile.readlines()
  assert(len(data)==1)
  return data[0]

# -------------------------------------------------------------------
def write_matrix_to_bin_omit_shape(fileName, M, transposeBeforeWriting):
  fileo = open(fileName, "wb")
  if transposeBeforeWriting:
    MT = np.transpose(M)
    MT.tofile(fileo)
  else:
    M.tofile(fileo)
  fileo.close()

# -------------------------------------------------------------------
def write_dic_to_yaml_file(filePath, dicToWrite):
  with open(filePath, 'w') as yaml_file:
    yaml.dump(dicToWrite, yaml_file, \
              default_flow_style=False, \
              sort_keys=False)

# -------------------------------------------------------------------
def load_basis_from_binary_file(lsvFile):
  nr, nc  = np.fromfile(lsvFile, dtype=np.int64, count=2)
  M = np.fromfile(lsvFile, offset=np.dtype(np.int64).itemsize*2)
  M = np.reshape(M, (nr, nc), order='F')
  return M

# -------------------------------------------------------------------
def load_fom_state_snapshot_matrix(dataDirs, fomTotDofs, \
                                   numDofsPerCell, \
                                   subtractInitialCondition):
  M = np.zeros((0, fomTotDofs))
  for targetDirec in dataDirs:
    print("reading data from {}".format(targetDirec))

    data = np.fromfile(targetDirec+"/fom_snaps_state")

    numTimeSteps = int(np.size(data)/fomTotDofs)
    D  = np.reshape(data, (numTimeSteps, fomTotDofs))
    if subtractInitialCondition:
      print("subtracting initial state")
      IC = np.loadtxt(targetDirec+"/initial_state.txt")
      for i in range(D.shape[0]):
        D[i,:] = D[i,:] - IC

    M = np.append(M, D, axis=0)

  print("state snapshots: shape  : ", M.T.shape)
  print("state snapshots: min/max: ", np.min(M), np.max(M))
  return M.T

# -------------------------------------------------------------------
def load_fom_rhs_snapshot_matrix(dataDirs, fomTotDofs, numDofsPerCell):
  M = np.zeros((0, fomTotDofs))
  for targetDirec in dataDirs:
    print("reading data from {}".format(targetDirec))

    data = np.fromfile(targetDirec+"/fom_snaps_rhs")
    numTimeSteps = int(np.size(data)/fomTotDofs)
    D = np.reshape(data, (numTimeSteps, fomTotDofs))
    M = np.append(M, D, axis=0)

  print("rhs snapshots: shape    : ", M.T.shape)
  print("rhs snapshots: min/max  : ", np.min(M), np.max(M))
  return M.T

# -------------------------------------------------------------------
def load_rom_state_snapshot_matrix(runDir, totalNumModes):
  print("reading data from {}".format(runDir))
  data = np.fromfile(runDir+"/rom_snaps_state")
  numTimeSteps = int(np.size(data)/totalNumModes)
  M = np.reshape(data, (numTimeSteps, totalNumModes))
  #print("A= ", M.flags['C_CONTIGUOUS'])
  print("rom state snapshots: shape  : ", M.T.shape)
  print("rom state snapshots: min/max: ", np.min(M), np.max(M))
  return M.T
