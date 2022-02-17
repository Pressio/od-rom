
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import random, subprocess
import matplotlib.pyplot as plt
import re, os, time, yaml
import numpy as np
from numpy import linalg as LA
import pressiodemoapps as pda
from scipy import linalg
from odrom_full import *

#==============================================================
# functions
#==============================================================

def get_run_id(runDir):
  return int(runDir.split('_')[-1])

# -------------------------------------------------------------------
def get_tile_id(stringIn):
  return int(stringIn.split('_')[-1])

# -------------------------------------------------------------------
def get_set_id(stringIn):
  return int(stringIn.split('_')[-1])

# -------------------------------------------------------------------
def find_sample_mesh_count_from_info_file(workDir):
  reg = re.compile(r'sampleMeshSize.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_stencil_mesh_count_from_info_file(workDir):
  reg = re.compile(r'stencilMeshSize.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

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
def dir_path_to_partition_info(workDir, npx, npy, style):
  return workDir + "/od_info_"+str(npx)+"x"+str(npy)+"_"+style

# -------------------------------------------------------------------
def dir_path_to_partition_based_full_mesh(workDir, partitioningKeyword):
  return workDir + "/partition_based_"+partitioningKeyword+"_full_mesh"

# -------------------------------------------------------------------
def dir_path_to_pod_data(workDir, partitioningKeyword, setId):
  return workDir + "/partition_based_"+partitioningKeyword+"_full_pod_set_"+str(setId)

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
def write_matrix_to_bin(fileName, M, writeShape, transposeBeforeWriting):
  fileo = open(fileName, "wb")
  # write to beginning of file the extents of the matrix
  if writeShape:
    r=np.int64(M.shape[0])
    np.array([r]).tofile(fileo)
    c=np.int64(M.shape[1])
    np.array([c]).tofile(fileo)
  if transposeBeforeWriting:
    MT = np.transpose(M)
    MT.tofile(fileo)
  else:
    M.tofile(fileo)
  fileo.close()

# -------------------------------------------------------------------
def load_basis_from_binary_file(lsvFile):
  nr, nc  = np.fromfile(lsvFile, dtype=np.int64, count=2)
  M = np.fromfile(lsvFile, offset=np.dtype(np.int64).itemsize*2)
  M = np.reshape(M, (nr, nc), order='F')
  return M

# -------------------------------------------------------------------
def inviscid_flux_string_to_stencil_size(stringIn):
  if stringIn == "FirstOrder":
    return 3
  elif stringIn == "Weno3":
    return 5
  elif stringIn == "Weno5":
    return 7
  else:
    sys.exit("Invalid scheme detected {}".format(scheme))
    return None

# -------------------------------------------------------------------
def inviscid_flux_string_to_enum(stringIn):
  if stringIn == "FirstOrder":
    return pda.InviscidFluxReconstruction.FirstOrder
  elif stringIn == "Weno3":
    return pda.InviscidFluxReconstruction.Weno3
  elif stringIn == "Weno5":
    return pda.InviscidFluxReconstruction.Weno5
  else:
    sys.exit("Invalid string")

# ----------------------------------------------------------------
def make_fom_mesh_if_needed(args, module):
  pdaMeshDir = args.pdadir + "/meshing_scripts"

  meshArgs = None
  if args.problem == "2d_swe":
    # for swe we need both x and y
    assert(len(args.mesh) == 2)

    # figure out what stencil size we need
    schemeStr = module.base_dic[args.scenario]['fom']['inviscidFluxReconstruction']
    stencilSize = inviscid_flux_string_to_stencil_size(schemeStr)

    nx, ny = args.mesh[0], args.mesh[1]
    outDir = args.workdir + "/full_mesh" + str(nx) + "x" + str(ny)
    if os.path.exists(outDir):
      print('Mesh {} already exists'.format(outDir))
    else:
      print('Generating mesh {}'.format(outDir))
      meshArgs = ("python3", pdaMeshDir + '/create_full_mesh.py',\
                  "-n", str(nx), str(ny),\
                  "--outDir", outDir,\
                  "--bounds", "-5.0", "5.0", "-5.0", "5.0",
                  "-s", str(stencilSize))
      popen  = subprocess.Popen(meshArgs, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
    return outDir

  else:
    sys.exit("make_fom_mesh_if_needed: invalid problem = {}".format(problemName))

# -------------------------------------------------------------------
def write_dic_to_yaml_file(filePath, dicToWrite):
  with open(filePath, 'w') as yaml_file:
    yaml.dump(dicToWrite, yaml_file, \
              default_flow_style=False, \
              sort_keys=False)

# -------------------------------------------------------------------
class FomObserver:
  def __init__(self, N, sf, vf, numSteps):
    self.f_     = [sf, vf]
    self.count_ = [0,0]

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
def run_single_fom_if_needed(runDir, appObj, dic):
  if not os.path.exists(runDir):
    os.makedirs(runDir)
    print("Doing FOM run for {}".format(runDir))

    # write to yaml the dic to save info for later
    inputFile = runDir + "/input.yaml"
    write_dic_to_yaml_file(inputFile, dic)

    # extrac params
    odeScheme         = dic['odeScheme']
    dt                = dic['dt']
    finalTime         = dic['finalTime']
    stateSamplingFreq = dic['stateSamplingFreq']
    rhsSamplingFreq   = dic['velocitySamplingFreq']

    numSteps = int(finalTime/dt)

    # run
    yn = appObj.initialCondition()
    np.savetxt(runDir+'/initial_state.txt', yn)
    numDofs = len(yn)

    obsO = FomObserver(numDofs, stateSamplingFreq, rhsSamplingFreq, numSteps)
    if odeScheme in ["RungeKutta4", "RK4"]:
      pda.advanceRK4(appObj, yn, dt, numSteps, observer=obsO)
    elif odeScheme == "SSPRK3":
      pda.advanceSSP3(appObj, yn, dt, numSteps, observer=obsO)
    else:
      sys.exit("run_single_fom: invalid ode scheme = {}".format(odeScheme))

    obsO.write(runDir)
    np.savetxt(runDir+'/final_state.txt', yn)

  else:
    print("FOM run {} already exists".format(runDir))

# -------------------------------------------------------------------
def do_multiple_foms(kind, workDir, problem, module, scenario, fomMesh):
  assert(kind in ["train", "test"])

  param_values = None
  if kind == "train":
    param_values = module.train_points[scenario]
  else:
    param_values = module.test_points[scenario]

  # fom object is loaded in same way for all problems
  fomMeshObj = pda.load_cellcentered_uniform_mesh(fomMesh)

  if problem == "2d_swe":
    probId  = pda.Swe2d.SlipWall

    fomDic    = module.base_dic[scenario]['fom']
    schemeStr = fomDic['inviscidFluxReconstruction']
    schemeEnu = inviscid_flux_string_to_enum(schemeStr)

    for k,v in param_values.items():
      gravity, coriolis, pulse  = 9.8, -3.0, 0.125
      if scenario == 1:
        coriolis = v
      else:
        sys.exit("invalid scenario = {}".format(scenario))

      runDir = workDir + "/fom_"+kind+"_"+str(k)
      appObj = pda.create_problem(fomMeshObj, probId, schemeEnu, gravity, coriolis, pulse)
      fomDic['gravity']  = gravity
      fomDic['coriolis'] = coriolis
      fomDic['pulse']    = pulse
      run_single_fom_if_needed(runDir, appObj, fomDic)

  else:
    sys.exit("invalid problem = {}".format(problemName))


# -------------------------------------------------------------------
def make_uniform_partitions_2d(workDir, module, scenario, fullMeshPath):
  file_path = pathlib.Path(__file__).parent.absolute()

  for pIt in module.odrom_partitioning_topol[scenario]:
    nTilesX, nTilesY = pIt[0], pIt[1]
    outDir = dir_path_to_partition_info(workDir, nTilesX, nTilesY, "uniform")

    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for {}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(file_path)+'/partition_uniform.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTilesX), str(nTilesY),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      print(output)


# -------------------------------------------------------------------
def make_full_mesh_for_odrom_using_partition_based_indexing(workDir, pdaDir, module):
  '''
  # for doing FULL od-rom without HR, for performance reasons,
  # we cannot use the same full mesh used in the fom.
  # We need to make a new full mesh with a new indexing
  # that is consistent with the paritions
  '''
  # find in workDir the fullMesh used there, and ensure unique
  fomFullMeshes = [workDir+'/'+d for d in os.listdir(workDir) \
                   # we need to find only dirs that being with this
                   # otherwise we pick up other things
                   if "full_mesh" in os.path.basename(d)[0:9]]
  assert(len(fomFullMeshes)==1)
  fomFullMesh = fomFullMeshes[0]
  totalCells = find_total_cells_from_info_file(fomFullMesh)

  # find all partitions directories
  partsInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "od_info_" in d]

  # for each partitioning, make the mesh
  for partitionInfoDirIt in partsInfoDirs:
    partitionStringIdentifier = os.path.basename(partitionInfoDirIt)[8:]
    outDir = dir_path_to_partition_based_full_mesh(workDir, partitionStringIdentifier)
    if os.path.exists(outDir):
      print('Partition-based full mesh dir {} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # to actually make the mesh, I need to make an array of gids
      # which in this case is the full gids
      gids = np.arange(0, totalCells)
      np.savetxt(outDir+'/sample_mesh_gids.dat', gids, fmt='%8i')

      print('Generating sample mesh in:')
      print(' {}'.format(outDir))
      meshScriptsDir = pdaDir + "/meshing_scripts"
      args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
              "--fullMeshDir", fomFullMesh,
              "--sampleMeshIndices", outDir+'/sample_mesh_gids.dat',
              "--outDir", outDir,
              "--useTilingFrom", partitionInfoDirIt)
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      print(output)


# -------------------------------------------------------------------
def load_fom_state_snapshot_matrix(dataDirs, numTotDofs, numDofsPerCell):
  M = np.zeros((0, numTotDofs))
  for targetDirec in dataDirs:
    print("reading data from {}".format(targetDirec))

    data = np.fromfile(targetDirec+"/fom_snaps_state")
    numTimeSteps = int(np.size(data)/numTotDofs)
    D = np.reshape(data, (numTimeSteps, numTotDofs))
    M = np.append(M, D, axis=0)

  print("state snapshots: shape  : ", M.T.shape)
  print("state snapshots: min/max: ", np.min(M), np.max(M))
  return M.T

# -------------------------------------------------------------------
def load_fom_rhs_snapshot_matrix(dataDirs, numTotDofs, numDofsPerCell):
  M = np.zeros((0, numTotDofs))
  for targetDirec in dataDirs:
    print("reading data from {}".format(targetDirec))

    data = np.fromfile(targetDirec+"/fom_snaps_rhs")
    numTimeSteps = int(np.size(data)/numTotDofs)
    D = np.reshape(data, (numTimeSteps, numTotDofs))
    M = np.append(M, D, axis=0)

  print("rhs snapshots: shape    : ", M.T.shape)
  print("rhs snapshots: min/max  : ", np.min(M), np.max(M))
  return M.T

# -------------------------------------------------------------------
def do_svd_py(mymatrix, lsvFile, svaFile):
  timing = np.zeros(1)
  start = time.time()
  U,S,_ = linalg.svd(mymatrix, full_matrices=False, lapack_driver='gesdd')
  end = time.time()
  elapsed = end - start
  timing[0] = elapsed
  #print("elapsed ", elapsed)

  #singular values
  print("Writing sing values to file: {}".format(svaFile))
  np.savetxt(svaFile, S)

  assert(U.flags['F_CONTIGUOUS'])

  # left singular vectors
  fileo = open(lsvFile, "wb")
  # write to beginning of file the extents of the matrix
  print("  writing POD modes to file: {}".format(lsvFile))
  r=np.int64(U.shape[0])
  np.array([r]).tofile(fileo)
  c=np.int64(U.shape[1])
  np.array([c]).tofile(fileo)
  '''
  NOTE: tofile write an array in rowwise, REGARDLESS of the layout of the matrix.
  So here we need to pass U.T to tofile so that tofile writes U in the proper
  way required format for how we read these later
  '''
  UT = np.transpose(U)
  UT.tofile(fileo)
  fileo.close()

  #outDir = os.path.dirname(lsvFile)
  #np.savetxt(lsvFile+'.txt', U[:,:3])
  # np.savetxt(outDir+'/timings.txt', timing)

# -------------------------------------------------------------------
def compute_partition_based_pod(workDir, setId, dataDirs, module):
  '''
  compute pod for both state and rhs using fom train data
  '''

  # find in workDir the fullMesh used there, and ensure unique
  fomFullMeshes = [workDir+'/'+d for d in os.listdir(workDir) \
                   # we need to find only dirs that being with this
                   # otherwise we pick up other things
                   if "full_mesh" in os.path.basename(d)[0:9]]
  assert(len(fomFullMeshes)==1)
  fomFullMesh = fomFullMeshes[0]
  totCells = find_total_cells_from_info_file(fomFullMesh)
  totFomDofs = totCells*module.numDofsPerCell

  fomStateSnaps = load_fom_state_snapshot_matrix(dataDirs, totFomDofs, module.numDofsPerCell)
  fomRhsSnaps   = load_fom_rhs_snapshot_matrix(dataDirs,   totFomDofs, module.numDofsPerCell)
  print("pod: fomStateSnaps.shape = ", fomStateSnaps.shape)
  print("pod:   fomRhsSnaps.shape = ", fomRhsSnaps.shape)
  print("")
  #np.savetxt(workDir+'/FULLsnaps.txt', fomStateSnaps)

  # now that we have the FOM data loaded  for a target set of runs,
  # loop over all partitions and compute POD
  # find in workDir all directories with info about partitions
  partsInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "od_info_" in d]

  for partitionInfoDirIt in partsInfoDirs:
    partitionStringIdentifier = os.path.basename(partitionInfoDirIt)[8:]
    tiles = np.loadtxt(partitionInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    outDir = dir_path_to_pod_data(workDir, partitionStringIdentifier, setId)

    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # loop over each tile
      for tileId in range(nTilesX*nTilesY):
        # I need to compute POD for both STATE and RHS
        # using FOM data LOCAL to myself, so need to load
        # which rows of the FOM state I own and slice accordingly
        rowsFiles = partitionInfoDirIt + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
        myRowsInFullState = np.loadtxt(rowsFiles, dtype=int)

        # with loaded row indices I can slice the FOM STATE and RHS snapshots
        # to get only the data that belongs to me.
        myStateSlice = fomStateSnaps[myRowsInFullState, :]
        myRhsSlice   = fomRhsSnaps[myRowsInFullState, :]
        print("pod: tileId={}".format(tileId))
        print("  stateSlice.shape={}".format(myStateSlice.shape))
        print("  rhsSlice.shape  ={}".format(myRhsSlice.shape))
        #np.savetxt(outDir+'/snaps_p_'+str(tileId)+'.txt', myStateSlice)

        # svd
        lsvFile = outDir + '/lsv_state_p_'+str(tileId)
        svaFile = outDir + '/sva_state_p_'+str(tileId)
        do_svd_py(myStateSlice, lsvFile, svaFile)

        lsvFile = outDir + '/lsv_rhs_p_'+str(tileId)
        svaFile = outDir + '/sva_rhs_p_'+str(tileId)
        do_svd_py(myRhsSlice, lsvFile, svaFile)

# -------------------------------------------------------------------
def compute_cumulative_energy(svalues, target):
  # convert percentage to decimal
  target = float(target)/100.

  sSq = np.square(svalues)
  den = np.sum(sSq)
  rsum = 0.
  for i in range(0, len(svalues)):
    rsum += sSq[i]
    ratio = (rsum/den)
    if ratio >= target:
      return i
  return len(svalues)

# -------------------------------------------------------------------
def find_modes_per_tile_from_target_energy(podDir, energy):
  result = {}
  # find all sing values files
  singValsFiles = [podDir+'/'+f for f in os.listdir(podDir) \
                   if "sva_state" in f]
  # sort based on the tile id
  singValsFiles = sorted(singValsFiles, key=get_tile_id)

  for it in singValsFiles:
    singValues = np.loadtxt(it)
    tileId = get_tile_id(it)
    K = compute_cumulative_energy(singValues, energy)
    if K ==0:
      result[tileId] = max(K, 1)

  totalModesCount = 0
  for k,v in result.items():
    totalModesCount += int(v)

  return totalModesCount, result

# -------------------------------------------------------------------
def run_full_od_galerkin(workDir, problem, module, scenario, fomMeshPath):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  for partInfoDirIt in partInfoDirs:
    partitionStringIdentifier = os.path.basename(partInfoDirIt)[8:]
    tiles = np.loadtxt(partInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    nTiles = nTilesX*nTilesY

    # for each topology, find full mesh, should be unique
    topoString = str(nTilesX)+"x"+str(nTilesY)
    meshDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                if topoString in d and "full_mesh" in d]
    assert(len(meshDirs)==1)
    romMeshDir = meshDirs[0]
    # fom object is loaded in same way for all problems
    romMeshObj = pda.load_cellcentered_uniform_mesh(romMeshDir)

    # find all FULL pod dirs
    # note that we might have more than one because
    # there might be multiple sets being handled
    podDirs = [workDir+'/'+d for d in os.listdir(workDir) \
               if topoString in d and "full_pod_set" in d]
    for podDirIt in podDirs:
      # find what setId I am doing
      setId = get_set_id(podDirIt)

      for energyIt in module.odrom_energies[scenario]:
        romSizeOverAllPartitions, modesPerTileDic = find_modes_per_tile_from_target_energy(podDirIt, energyIt)

        # solve odrom at each test point needed
        param_values = module.test_points[scenario]
        if problem == "2d_swe":
          probId    = pda.Swe2d.SlipWall
          romDic    = module.base_dic[scenario]['odrom']
          schemeStr = romDic['inviscidFluxReconstruction']
          schemeEnu = inviscid_flux_string_to_enum(schemeStr)
          for k,v in param_values.items():
            gravity, coriolis, pulse  = 9.8, -3.0, 0.125
            if scenario == 1:
              coriolis = v
            else:
              sys.exit("invalid scenario = {}".format(scenario))

            outDir = workDir + "/odrom_full_"+partitionStringIdentifier+"_"+str(energyIt)+"_set_"+str(setId)
            if os.path.exists(outDir):
              print('{} already exists'.format(outDir))
            else:
              print(outDir)
              os.system('mkdir -p ' + outDir)
              appObj = pda.create_problem(romMeshObj, probId, schemeEnu, gravity, coriolis, pulse)
              romDic['gravity']  = gravity
              romDic['coriolis'] = coriolis
              romDic['pulse']    = pulse
              romDic['energy']   = energyIt
              romDic['partioningInfo'] = partInfoDirIt
              inputFile = outDir + "/input.yaml"
              write_dic_to_yaml_file(inputFile, romDic)

              # make ROM initial condition
              meshObjIc   = pda.load_cellcentered_uniform_mesh(fomMeshPath)
              appObjIc    = pda.create_problem(meshObjIc, probId, schemeEnu, gravity, coriolis, pulse)
              fomIc       = appObj.initialCondition()
              romState    = np.zeros(romSizeOverAllPartitions)
              romStateStart = 0
              for tileId in range(nTiles):
                myK = modesPerTileDic[tileId]
                myFullPhi       = load_basis_from_binary_file(podDirIt + "/lsv_state_p_" + str(tileId) )[:,0:myK]
                myStateRowsFile = partInfoDirIt+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
                myStateRows     = np.loadtxt(myStateRowsFile, dtype=int)
                myFomIcSlice    = fomIc[myStateRows]
                tmpyhat         = np.dot(myFullPhi.transpose(), myFomIcSlice)
                romState[romStateStart:romStateStart+myK] = np.copy(tmpyhat)
                romStateStart += myK

              stencilDofsCount = find_stencil_mesh_count_from_info_file(romMeshDir)*module.numDofsPerCell
              sampleDofsCount  = find_sample_mesh_count_from_info_file(romMeshDir)*module.numDofsPerCell
              odRomObj = OdRomFull(podDirIt, stencilDofsCount, sampleDofsCount, modesPerTileDic)
              odRomObj.reconstructMemberFomState(romState)
              yRIC = odRomObj.viewFomStateFullMesh()
              np.savetxt(outDir+"/y_rec_ic.txt", yRIC)

              appObjForRom = pda.create_problem(romMeshObj, probId, schemeEnu, gravity, coriolis, pulse)
              dt        = romDic['dt']
              finalTime = romDic['finalTime']
              numSteps = int(finalTime/dt)
              odRomObj.run(outDir, romState, appObjForRom, numSteps, dt)

              odRomObj.reconstructMemberFomState(romState)
              yRecon = odRomObj.viewFomStateFullMesh()
              np.savetxt(workDir+"/y_rec_final.txt", yRecon)


#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  parser   = ArgumentParser()
  parser.add_argument("--wdir", "--workdir",
                      dest="workdir",
                      required=True)
  parser.add_argument("--pdadir",
                      dest="pdadir",
                      required=True)
  parser.add_argument("--problem",
                      dest="problem",
                      required=True)
  parser.add_argument("--scenario", "-s",
                      dest="scenario",
                      type=int,
                      required=True)
  parser.add_argument("--mesh",
                      nargs=2, \
                      dest="mesh", \
                      type=int, \
                      required=True)

  args     = parser.parse_args()
  workDir  = args.workdir
  pdaDir   = args.pdadir
  problem  = args.problem
  scenario = args.scenario

  if not os.path.exists(workDir):
    print("Working dir {} does not exist, creating it".format(workDir))
    os.system('mkdir -p ' + workDir)

  # write scenario id, problem to file
  write_scenario_to_file(scenario, workDir)
  write_problem_name_to_file(problem, workDir)

  print("========================")
  print("Importing problem module")
  print("========================")
  module = importlib.import_module(problem)
  print(module)
  print("")

  print("========================")
  print("Making full mesh ")
  print("========================")
  fomMeshPath = make_fom_mesh_if_needed(args, module)
  print("")

  print("========================")
  print("Train runs")
  print("========================")
  do_multiple_foms("train", workDir, problem, module, scenario, fomMeshPath)
  print("")

  print("========================")
  print("Test runs")
  print("========================")
  do_multiple_foms("test", workDir, problem, module, scenario, fomMeshPath)
  print("")

  print("========================")
  print("Creating partitions")
  print("========================")
  for partitioningStyleIt in module.odrom_partitioning_style[scenario]:
    if module.dimensionality == 2:
      if partitioningStyleIt == "uniform":
        make_uniform_partitions_2d(workDir, module, scenario, fomMeshPath)
    else:
      sys.exit("Invalid dimensionality = {}".format(module.dimensionality))
  print("")

  print("=====================================")
  print("Creating partition-based full meshes ")
  print("=====================================")
  # for doing FULL od-rom without HR, for performance reasons,
  # we cannot use the same full mesh used in the fom.
  # We need to make a new full mesh with a new indexing
  # that is consistent with the partitions
  make_full_mesh_for_odrom_using_partition_based_indexing(workDir, pdaDir, module)
  print("")

  print("=====================================")
  print("Computing partition-based POD ")
  print("=====================================")
  for setId, trainIndices in module.odrom_basis_sets[scenario].items():
    print("")
    print("------------------------")
    print("Handling setId = {}".format(setId))
    print("------------------------")

    trainDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                 if "train" in d and get_run_id(d) in trainIndices]
    compute_partition_based_pod(workDir, setId, trainDirs, module)
  print("")

  print("=====================================")
  print("Doing FULL od-galerkin if needed ")
  print("=====================================")
  # check if GalerkinFull is present in the list of algos wanted
  if "GalerkinFull" in module.odrom_algos[scenario]:
    run_full_od_galerkin(workDir, problem, module, scenario, fomMeshPath)
  print("")


  #print(module.train_points[args.scenario])
  sys.exit()









  partitions    = [[6,6]]
  smPercentages = [1.0]
  basisCounts   = [10]

  sampFrequency  = {'state': 2, 'rhs': 2 }
  timeStepSize   = 0.005
  finalTime      = 5.0
  numSteps       = int(finalTime/timeStepSize)
  train_values   = [-3.0, -0.5]
  test_values    = [-3.5]
  numDofsPerCell = 3
  print("numSteps = {}".format(numSteps))

  probId = pda.Swe2d.SlipWall #pda.Euler2d.RayleighTaylor
  scheme = None
  if args.stencilSize == 3:
    scheme = pda.InviscidFluxReconstruction.FirstOrder
  elif args.stencilSize == 5:
    scheme = pda.InviscidFluxReconstruction.Weno3
  elif args.stencilSize == 7:
    scheme = pda.InviscidFluxReconstruction.Weno5
  else:
    sys.exit("Invalid stencil size {}".format(args.stencilSize))

  #===============
  # generate and load full mesh
  meshPath   = make_full_mesh_if_needed(workDir, args)
  meshObj    = pda.load_cellcentered_uniform_mesh(meshPath)
  nx, ny     = extract_mesh_size(meshPath)
  totFomDofs = nx*ny*numDofsPerCell
  assert(nx == args.meshSize[0])
  assert(ny == args.meshSize[1])

  #===============
  # train foms
  for i,paramValueIt in enumerate(train_values):
    print("Running fom_train_{} for {}".format(i, paramValueIt))
    appObj = pda.create_problem(meshObj, probId, scheme, 9.8, paramValueIt)
    run_fom(workDir, "train", i, appObj, timeStepSize, numSteps, sampFrequency)
  print("")

  #===============
  # test foms
  for i,paramValueIt in enumerate(test_values):
    print("Running fom_test_{} for  {}".format(i, paramValueIt))
    appObj = pda.create_problem(meshObj, probId, scheme, 9.8, paramValueIt)
    run_fom(workDir, "test", i, appObj, timeStepSize, numSteps, sampFrequency)
  print("")

  print("===================")
  print("Creating partitions")
  print("===================")
  make_partitions(workDir, partitions, nx, ny, numDofsPerCell)
  print("")

  print("===================")
  print("Doing POD")
  print("===================")
  compute_full_pod_in_each_tile(workDir, partitions, totFomDofs, numDofsPerCell)
  print("")

  # print("===================")
  # print("Sample meshes")
  # print("===================")
  # compute_sample_mesh(workDir, pdaDir, meshPath, partitions, basisCounts,\
  #                     smPercentages, totFomDofs, numDofsPerCell)
  # print("")

  # print("===================")
  # print("Doing projectors")
  # print("===================")
  # compute_gappy_projector(workDir, partitions, basisCounts, numDofsPerCell)
  # print("")

  # print("===================")
  # print("Doing masked ODROM")
  # print("===================")
  # parts  = partitions[0]
  # nTiles = parts[0]*parts[1]
  # K      = basisCounts[0]
  # pct    = smPercentages[0]
  # smMeshDir = [workDir+'/'+d for d in os.listdir(workDir) if "_sample_mesh" in d][0]
  # smMeshObj = pda.load_cellcentered_uniform_mesh(smMeshDir+"/sm")

  # for i,paramValueIt in enumerate(test_values):
  #   print("Running od_rom_test_{} for  {}".format(i, paramValueIt))

  #   fomObjForIC = pda.create_problem(meshObj, probId, scheme, 9.8, paramValueIt)
  #   fomIc       = fomObjForIC.initialCondition()
  #   romState    = np.zeros(K*nTiles)
  #   partInfoDir = dir_path_to_partition_info(workDir, parts[0], parts[1])
  #   podDataDir  = dir_path_to_pod_data(workDir, parts[0], parts[1])
  #   for tileId in range(nTiles):
  #     myFullPhi       = load_basis_from_binary_file(podDataDir + "/lsv_state_p_" + str(tileId) )[:,0:K]
  #     myStateRowsFile = partInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
  #     myStateRows     = np.loadtxt(myStateRowsFile, dtype=int)
  #     myFomIcSlice    = fomIc[myStateRows]
  #     tmpyhat         = np.dot(myFullPhi.transpose(), myFomIcSlice)
  #     romState[tileId*K:tileId*K+K] = np.copy(tmpyhat)

  #   fomObj = pda.create_problem(smMeshObj, probId, scheme, 9.8, paramValueIt)
  #   sampleMeshSize  = fomObj.totalDofSampleMesh()
  #   stencilMeshSize = fomObj.totalDofStencilMesh()

  #   odRomObj = OdRom(workDir, totFomDofs, sampleMeshSize, \
  #                    stencilMeshSize, numDofsPerCell, parts, K, pct)
  #   odRomObj.reconstructMemberFomStateFullMesh(romState)
  #   yRIC = odRomObj.viewFomStateFullMesh()
  #   np.savetxt(workDir+"/y_rec_ic.txt", yRIC)

  #   odRomObj.run(romState, fomObj, numSteps, timeStepSize)

  #   odRomObj.reconstructMemberFomStateFullMesh(romState)
  #   yRecon = odRomObj.viewFomStateFullMesh()
  #   np.savetxt(workDir+"/y_rec_final.txt", yRecon)

  #   # fomCoords = np.loadtxt(meshPath+'/coordinates.dat', dtype=float)
  #   # x = np.reshape(fomCoords[:,1], (ny,nx))
  #   # y = np.reshape(fomCoords[:,2], (ny,nx))
  #   # fig = plt.figure()
  #   # ax = plt.gca()
  #   # rho  = np.reshape(fomRIC[0::4], (ny,nx))
  #   # ax.contourf(x, y, rho, 35)
  #   # ax.set_aspect(aspect=1.)
  #   # plt.show()
