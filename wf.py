
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import random, subprocess
import matplotlib.pyplot as plt
import re, os, time, yaml
import numpy as np
from scipy import linalg as scipyla
from decimal import Decimal

# pda module
import pressiodemoapps as pda

# local imports
from myio import *
from legendre_bases import LegendreBases2d
from observer import FomObserver, RomObserver
from odrom_full import *

#==============================================================
# functions
#==============================================================

def get_run_id(runDir):
  return int(runDir.split('_')[-1])

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
def path_to_partition_info_dir(workDir, npx, npy, style):
  return workDir + "/od_info_"+str(npx)+"x"+str(npy)+"_"+style

# -------------------------------------------------------------------
def string_identifier_from_partition_info_dir(infoDir):
  return os.path.basename(infoDir)[8:]

# -------------------------------------------------------------------
def path_to_partition_based_full_mesh_dir(workDir, partitioningKeyword):
  return workDir + "/partition_based_"+partitioningKeyword+"_full_mesh"

# -------------------------------------------------------------------
def path_to_pod_data_dir(workDir, partitioningKeyword, setId):
  return workDir + "/partition_based_"+partitioningKeyword+"_full_pod_set_"+str(setId)

# -------------------------------------------------------------------
def path_to_poly_bases_data_dir(workDir, partitioningKeyword, \
                                order, energy=None, setId=None):
  result = workDir + "/partition_based_"+partitioningKeyword+"_full_poly_order_"+str(order)
  if energy != None:
    result += "_"+str(energy)
  if setId != None:
    result += "_set_"+str(setId)
  return result

# ----------------------------------------------------------------
def make_fom_mesh_if_not_existing(workDir, problem, \
                                  module, scenario, \
                                  pdaDir, meshSize):
  assert( len(meshSize)== module.dimensionality)

  meshArgs = ("python3", \
              pdaDir + "/meshing_scripts/create_full_mesh.py")

  outDir = workDir + "/full_mesh" + str(meshSize[0])
  if len(meshSize) == 1:
    meshArgs += ( "-n", str(meshSize[0]) )
  if len(meshSize) == 2:
    outDir += "x" + str(meshSize[1])
    meshArgs += ( "-n", str(meshSize[0]), str(meshSize[1]) )
  if len(meshSize) == 3:
    outDir += "x" + str(meshSize[1]) + "x" + str(meshSize[2])
    meshArgs += ( "-n", str(meshSize[0]), str(meshSize[1]), str(meshSize[2]) )

  meshArgs += ("--outDir", outDir)

  # problem-specific function to fill args for FOM mesh generation
  meshArgs += module.tuple_args_for_fom_mesh_generation(scenario)

  # now, generate mesh if needed
  if os.path.exists(outDir):
    print('{} already exists'.format(outDir))
  else:
    print('Generating mesh {}'.format(outDir))
    popen  = subprocess.Popen(meshArgs, stdout=subprocess.PIPE);
    popen.wait()
    output = popen.stdout.read();

# ----------------------------------------------------------------
def find_full_mesh_and_ensure_unique(workDir):
  # This is because the mesh is specified
  # via command line argument and must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory

  fomFullMeshes = [workDir+'/'+d for d in os.listdir(workDir) \
                   # we need to find only dirs that BEGIN with
                   # this string otherwise we pick up other things
                   if "full_mesh" == os.path.basename(d)[0:9]]
  if len(fomFullMeshes) != 1:
    sys.exit("Error: detected multiple full meshes in a single working dir. \
You can only have a single FULL mesh the working directory.")

  return fomFullMeshes[0]

# -------------------------------------------------------------------
def run_single_fom(runDir, appObj, dic):
  # write to yaml the dic to save info for later
  inputFile = runDir + "/input.yaml"
  write_dic_to_yaml_file(inputFile, dic)

  # extrac params
  odeScheme         = dic['odeScheme']
  dt                = float(dic['dt'])
  stateSamplingFreq = int(dic['stateSamplingFreq'])
  rhsSamplingFreq   = int(dic['velocitySamplingFreq'])
  finalTime         = float(dic['finalTime'])
  numSteps          = int(round(Decimal(finalTime)/Decimal(dt), 8))
  print("numSteps = ", numSteps)

  # run
  yn = appObj.initialCondition()
  np.savetxt(runDir+'/initial_state.txt', yn)
  numDofs = len(yn)

  start = time.time()
  obsO = FomObserver(numDofs, stateSamplingFreq, rhsSamplingFreq, numSteps)
  if odeScheme in ["RungeKutta4", "RK4", "rungekutta4", "rk4"]:
    pda.advanceRK4(appObj, yn, dt, numSteps, observer=obsO)
  elif odeScheme in ["SSPRK3", "ssprk3"]:
    pda.advanceSSP3(appObj, yn, dt, numSteps, observer=obsO)
  else:
    sys.exit("run_single_fom: invalid ode scheme = {}".format(odeScheme))

  elapsed = time.time() - start
  print("elapsed = {}".format(elapsed))
  f = open(runDir+"/timing.txt", "w")
  f.write(str(elapsed))
  f.close()

  obsO.write(runDir)
  np.savetxt(runDir+'/final_state.txt', yn)


# -------------------------------------------------------------------
def run_foms(workDir, problem, module, scenario, \
             testOrTrainString, fomMesh):

  assert(testOrTrainString in ["train", "test"])

  # load the list of parameter values to run FOM for
  param_values = None
  if testOrTrainString == "train":
    param_values = module.train_points[scenario]
  else:
    param_values = module.test_points[scenario]

  # fom mesh object is loaded in same way for ALL problems
  fomMeshObj = pda.load_cellcentered_uniform_mesh(fomMesh)

  # now we need to create the problem obj and run
  # but how we do this is specific to each problem
  for k,val in param_values.items():
    fomObj = None

    # get the dic with base parameters for the FOM
    fomDic   = module.base_dic[scenario]['fom'].copy()
    coeffDic = module.base_dic[scenario]['physicalCoefficients'].copy()

    # create problem using in-module function
    fomObj = module.create_problem_for_scenario(scenario, fomMeshObj, \
                                                coeffDic, fomDic, val)

    # the train/test simulation time might differ, ensure we pick the right one
    finalTime = fomDic['finalTimeTrain'] if testOrTrainString == "train" \
      else fomDic['finalTimeTest']

    # set the final time in dic
    del fomDic['finalTimeTrain']
    del fomDic['finalTimeTest']
    fomDic['finalTime'] = float(finalTime)

    # run FOM run for current fomDic
    runDir = workDir + "/fom_"+testOrTrainString+"_"+str(k)
    if not os.path.exists(runDir):
      os.makedirs(runDir)
      print("Doing FOM run for {}".format(runDir))
      run_single_fom(runDir, fomObj, fomDic)
    else:
      print("FOM run {} already exists".format(runDir))

# -------------------------------------------------------------------
def make_uniform_partitions_2d(workDir, module, scenario, fullMeshPath):
  '''
  tile a 2d mesh using uniform partitions if possible
  Info for target tiling layout is extracted from the scenario.
  '''
  this_file_path = pathlib.Path(__file__).parent.absolute()

  for pIt in module.odrom_partitioning_topol[scenario]:
    nTilesX, nTilesY = pIt[0], pIt[1]

    outDir = path_to_partition_info_dir(workDir, nTilesX, nTilesY, "uniform")
    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for {}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(this_file_path)+'/partition_uniform.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTilesX), str(nTilesY),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      print(output)


# -------------------------------------------------------------------
def make_full_mesh_for_odrom_using_partition_based_indexing(workDir, pdaDir, \
                                                            module, fomMesh):
  '''
  for FULL od-rom without HR, for performance reasons,
  we do not/should not use the same full mesh used in the fom.
  We need to make a new full mesh with a new indexing
  that is consistent with the partitions and allows continguous storage
  of the state and rhs within each tile
  '''
  totalCells = find_total_cells_from_info_file(fomMesh)

  # find all existing partitions directories inside the workDir
  partitionInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                       if "od_info_" in d]

  # for each one, make the mesh with correct indexing
  for partitionInfoDirIt in partitionInfoDirs:
    # I need to extract an identifier from the direc so that I can
    # use this string to uniquely create a corresponding directory
    # where to store the new mesh
    stringIdentifier = string_identifier_from_partition_info_dir(partitionInfoDirIt)
    outDir = path_to_partition_based_full_mesh_dir(workDir, stringIdentifier)
    if os.path.exists(outDir):
      print('Partition-based full mesh dir {} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # to make the mesh, I need to make an array of gids
      # which in this case is the full gids
      gids = np.arange(0, totalCells)
      np.savetxt(outDir+'/sample_mesh_gids.dat', gids, fmt='%8i')

      print('Generating sample mesh in:')
      print(' {}'.format(outDir))
      meshScriptsDir = pdaDir + "/meshing_scripts"
      args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
              "--fullMeshDir", fomMesh,
              "--sampleMeshIndices", outDir+'/sample_mesh_gids.dat',
              "--outDir", outDir,
              "--useTilingFrom", partitionInfoDirIt)
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      print(output)


# -------------------------------------------------------------------
def do_svd_py(mymatrix, lsvFile, svaFile):
  timing = np.zeros(1)
  start = time.time()
  U,S,_ = scipyla.svd(mymatrix, full_matrices=False, lapack_driver='gesdd')
  end = time.time()
  elapsed = end - start
  timing[0] = elapsed
  #print("elapsed ", elapsed)

  #singular values
  #print("Writing sing values to file: {}".format(svaFile))
  np.savetxt(svaFile, S)

  assert(U.flags['F_CONTIGUOUS'])

  # left singular vectors
  fileo = open(lsvFile, "wb")
  # write to beginning of file the extents of the matrix
  #print("  writing POD modes to file: {}".format(lsvFile))
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
def replicate_bases_for_multiple_dofs(M, numDofsPerCell):
  if numDofsPerCell == 1:
    return M

  elif numDofsPerCell == 2:
    K1 = np.shape(M)[1]
    K2 = np.shape(M)[1]
    Phi = np.zeros((M.shape[0]*numDofsPerCell, K1+K2))
    Phi[0::2,    0: K1]       = M
    Phi[1::2,   K1: K1+K2]    = M
    return Phi

  elif numDofsPerCell == 3:
    K1 = np.shape(M)[1]
    K2 = np.shape(M)[1]
    K3 = np.shape(M)[1]
    Phi = np.zeros((M.shape[0]*numDofsPerCell, K1+K2+K3))
    Phi[0::3,    0: K1]       = M
    Phi[1::3,   K1: K1+K2]    = M
    Phi[2::3,K1+K2: K1+K2+K3] = M
    return Phi

  elif numDofsPerCell == 4:
    K1 = np.shape(M)[1]
    K2 = np.shape(M)[1]
    K3 = np.shape(M)[1]
    K4 = np.shape(M)[1]
    Phi = np.zeros((M.shape[0]*numDofsPerCell, K1+K2+K3+K4))
    Phi[0::4,       0: K1]       = M
    Phi[1::4,     K1 : K1+K2]    = M
    Phi[2::4,   K1+K2: K1+K2+K3] = M
    Phi[3::4,K1+K2+K3: K1+K2+K3+K4] = M
    return Phi
  else:
    sys.exit("Invalid numDofsPerCell")

# -------------------------------------------------------------------
def compute_poly_bases_same_order_all_tiles(fomMesh, outDir, \
                                            partInfoDir, \
                                            targetOrder, \
                                            numDofsPerCell):

  fomCellsXcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,1]
  fomCellsYcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,2]
  tiles = np.loadtxt(partInfoDir+"/topo.txt")
  nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

  polyObj = LegendreBases2d("totalOrder")
  modesPerTile = {}
  for tileId in range(nTilesX*nTilesY):
    myFile = partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myRows = np.loadtxt(myFile, dtype=int)

    myX, myY = fomCellsXcoords[myRows], fomCellsYcoords[myRows]
    lsvFile = outDir + '/lsv_state_p_'+str(tileId)

    U0 = polyObj(targetOrder, myX, myY)
    #U0 = create_legendre_basis_max_order(myX, myY, targetOrder)
    U  = replicate_bases_for_multiple_dofs(U0, numDofsPerCell)
    U,_ = np.linalg.qr(U, mode='reduced')

    fileo = open(lsvFile, "wb")
    r=np.int64(U.shape[0])
    np.array([r]).tofile(fileo)
    c=np.int64(U.shape[1])
    np.array([c]).tofile(fileo)
    UT = np.transpose(U)
    UT.tofile(fileo)
    fileo.close()

    modesPerTile[tileId] = U.shape[1]

  np.savetxt(outDir+"/modes_per_tile.txt", \
             np.array(list(modesPerTile.values())), \
             fmt="%5d")

# -------------------------------------------------------------------
def compute_poly_bases_to_match_pod(fomMesh, outDir, \
                                    partInfoDir, \
                                    podModesPerTileToMatch,
                                    numDofsPerCell):

  fomCellsXcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,1]
  fomCellsYcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,2]
  tiles = np.loadtxt(partInfoDir+"/topo.txt")
  nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

  polyObj = LegendreBases2d("totalOrder")
  modesPerTile = {}
  for tileId in range(nTilesX*nTilesY):
    myFile = partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myRows = np.loadtxt(myFile, dtype=int)

    myX, myY = fomCellsXcoords[myRows], fomCellsYcoords[myRows]
    lsvFile = outDir + '/lsv_state_p_'+str(tileId)

    # the order we want has to be found such that the total modes
    # matches the pod modes in that tile, but we need to account
    # also for the fact that we might have multiple dofs/cell
    target = int(podModesPerTileToMatch[tileId]/numDofsPerCell)
    order = polyObj.findClosestOrderToMatchTargetBasesCount(target)
    print("order = {}".format(order))

    U0 = polyObj(order, myX, myY)
    U  = replicate_bases_for_multiple_dofs(U0, numDofsPerCell)
    U,_ = np.linalg.qr(U,mode='reduced')

    print("tile = {}, podModesCount = {}, polyModesCount = {}".format(tileId, \
                                                                      podModesPerTileToMatch[tileId],
                                                                      U.shape[1]))
    fileo = open(lsvFile, "wb")
    r=np.int64(U.shape[0])
    np.array([r]).tofile(fileo)
    c=np.int64(U.shape[1])
    np.array([c]).tofile(fileo)
    UT = np.transpose(U)
    UT.tofile(fileo)
    fileo.close()

    modesPerTile[tileId] = U.shape[1]

  np.savetxt(outDir+"/modes_per_tile.txt", \
             np.array(list(modesPerTile.values())), \
             fmt="%5d")

# -------------------------------------------------------------------
def compute_partition_based_state_and_rhs_pod(workDir, setId, dataDirs, module, fomMesh):
  '''
  compute pod for both state and rhs using fom train data
  '''
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell

  # only load snapshots once
  fomStateSnapsFullDomain = load_fom_state_snapshot_matrix(dataDirs, totFomDofs, \
                                                           module.numDofsPerCell)
  fomRhsSnapsFullDomain   = load_fom_rhs_snapshot_matrix(dataDirs,   totFomDofs, \
                                                         module.numDofsPerCell)
  print("pod: fomStateSnapsFullDomain.shape = ", fomStateSnapsFullDomain.shape)
  print("pod:   fomRhsSnapsFullDomain.shape = ", fomRhsSnapsFullDomain.shape)
  print("")

  # with the FOM data loaded for a target setId (i.e. set of runs)
  # loop over all partitions and compute local POD.
  # To do this, we find in workDir all directories with info about partitions
  # which identifies all possible partitions
  partsInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "od_info_" in d]

  for partitionInfoDirIt in partsInfoDirs:
    # I need to extract an identifier from the direc so that I can
    # use this string to uniquely create a corresponding directory
    # where to store the POD data
    stringIdentifier = string_identifier_from_partition_info_dir(partitionInfoDirIt)
    tiles = np.loadtxt(partitionInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

    outDir = path_to_pod_data_dir(workDir, stringIdentifier, setId)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # loop over each tile
      for tileId in range(nTilesX*nTilesY):
        # I need to compute POD for both STATE and RHS
        # using FOM data LOCAL to myself, so need to load
        # which rows of the FOM state I own and use to slice
        myFile = partitionInfoDirIt + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
        myRowsInFullState = np.loadtxt(myFile, dtype=int)

        # after loading the row indices, slice the FOM STATE and RHS snapshots
        # to get only the data that belongs to me.
        myStateSlice = fomStateSnapsFullDomain[myRowsInFullState, :]
        myRhsSlice   = fomRhsSnapsFullDomain[myRowsInFullState, :]
        print("pod: tileId={}: stateSlice.Shape={}, rhsSlice.shape={}".format(tileId, \
                                                                              myStateSlice.shape,\
                                                                              myRhsSlice.shape))

        # compute svd
        lsvFile = outDir + '/lsv_state_p_'+str(tileId)
        svaFile = outDir + '/sva_state_p_'+str(tileId)
        do_svd_py(myStateSlice, lsvFile, svaFile)

        lsvFile = outDir + '/lsv_rhs_p_'+str(tileId)
        svaFile = outDir + '/sva_rhs_p_'+str(tileId)
        do_svd_py(myRhsSlice, lsvFile, svaFile)

# -------------------------------------------------------------------
def compute_cumulative_energy(svalues, targetPercentage):
  if targetPercentage == 100.:
    return len(svalues)
  else:
    # convert percentage to decimal
    target = float(targetPercentage)/100.
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
def compute_total_modes_across_all_tiles(modesPerTileDic):
  return np.sum(list(modesPerTileDic.values()))

# -------------------------------------------------------------------
def find_modes_per_tile_from_target_energy(podDir, energy):
  def get_tile_id(stringIn):
    return int(stringIn.split('_')[-1])

  modesPerTileDic = {}
  # find all sing values files
  singValsFiles = [podDir+'/'+f for f in os.listdir(podDir) \
                   if "sva_state" in f]
  # sort based on the tile id
  singValsFiles = sorted(singValsFiles, key=get_tile_id)

  for it in singValsFiles:
    singValues = np.loadtxt(it)
    tileId = get_tile_id(it)
    K = compute_cumulative_energy(singValues, energy)
    modesPerTileDic[tileId] = max(K, 5)

  return modesPerTileDic


# -------------------------------------------------------------------
def make_od_rom_initial_condition(workDir, appObjForIc, \
                                  partitionInfoDir, \
                                  basesDir, modesPerTileDic, \
                                  romSizeOverAllPartitions):
  nTiles   = len(modesPerTileDic.keys())
  fomIc    = appObjForIc.initialCondition()
  romState = np.zeros(romSizeOverAllPartitions)
  romStateSpanStart = 0
  for tileId in range(nTiles):
    myK             = modesPerTileDic[tileId]
    myPhi           = load_basis_from_binary_file(basesDir + "/lsv_state_p_" + str(tileId) )[:,0:myK]
    myStateRowsFile = partitionInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
    myStateRows     = np.loadtxt(myStateRowsFile, dtype=int)
    myFomIcSlice    = fomIc[myStateRows]
    tmpyhat         = np.dot(myPhi.transpose(), myFomIcSlice)
    romState[romStateSpanStart:romStateSpanStart+myK] = np.copy(tmpyhat)
    romStateSpanStart += myK
  return romState

# -------------------------------------------------------------------
def run_full_od_galerkin_for_all_test_values(workDir, problem, module, \
                                             scenario, fomMeshPath, \
                                             partInfoDir, basesDir, \
                                             energyValue,
                                             polyOrder, \
                                             modesPerTileDic, \
                                             romMeshObj, \
                                             nTiles, setId, \
                                             basesKind):

  # this is odrom WITHOUT HR, so the following should hold:
  stencilDofsCount = romMeshObj.stencilMeshSize()*module.numDofsPerCell
  sampleDofsCount  = romMeshObj.sampleMeshSize()*module.numDofsPerCell
  assert(stencilDofsCount == sampleDofsCount)
  fomTotalDofs = stencilDofsCount

  romSizeOverAllPartitions = compute_total_modes_across_all_tiles(modesPerTileDic)

  fomMeshObj = pda.load_cellcentered_uniform_mesh(fomMeshPath)
  partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDir)

  param_values = module.test_points[scenario]
  for k, val in param_values.items():

    outDir = workDir + "/odrom_full_"+partitionStringIdentifier+"_"+basesKind
    if polyOrder != None:
      outDir += "_order_"+str(polyOrder)

    if energyValue != None:
      outDir += "_"+str(energyValue)

    if setId != None:
      outDir += "_set_"+str(setId)

    outDir += "_"+str(k)

    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print("Running odrom in {}".format(os.path.basename(outDir)))
      os.system('mkdir -p ' + outDir)
      romRunDic    = module.base_dic[scenario]['odrom'].copy()
      coeffDic     = module.base_dic[scenario]['physicalCoefficients'].copy()
      appObjForIc  = None
      appObjForRom = None

      appObjForIc  = module.create_problem_for_scenario(scenario, fomMeshObj, \
                                                        coeffDic, romRunDic, val)
      appObjForRom = module.create_problem_for_scenario(scenario, romMeshObj,
                                                        coeffDic, romRunDic, val)
      # these should hold
      assert(appObjForIc  != None)
      assert(appObjForRom != None)

      # write things to run directory
      f = open(outDir+"/rom_dofs_count.txt", "w")
      f.write(str(romSizeOverAllPartitions))
      f.close()

      np.savetxt(outDir+"/modes_per_tile.txt", \
                 np.array(list(modesPerTileDic.values())),
                 fmt="%5d")

      if basesKind == "using_pod_bases":
        romRunDic['energy'] = energyValue
      if basesKind == "using_poly_bases":
        romRunDic['polyOrder'] = polyOrder

      romRunDic['basesDir'] = basesDir
      romRunDic['partioningInfo'] = partInfoDir
      inputFile = outDir + "/input.yaml"
      write_dic_to_yaml_file(inputFile, romRunDic)

      romState = make_od_rom_initial_condition(workDir, appObjForIc, \
                                               partInfoDir, basesDir, \
                                               modesPerTileDic, \
                                               romSizeOverAllPartitions)

      # construct odrom object
      odRomObj = OdRomFull(basesDir, fomTotalDofs, modesPerTileDic, \
                           module.dimensionality, module.numDofsPerCell, \
                           partInfoDir)
      # initial condition
      odRomObj.reconstructMemberFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_ic.txt", odRomObj.viewFomState())

      # time loop
      dt         = romRunDic['dt']
      finalTime  = romRunDic['finalTime']
      numSteps   = int(finalTime/dt)
      odeScheme  = romRunDic['odeScheme']

      # create observer
      stateSamplingFreq = int(romRunDic['stateSamplingFreq'])
      obsO = RomObserver(stateSamplingFreq, numSteps, modesPerTileDic)

      pTimeStart = time.time()
      if odeScheme in ["SSPRK3", "ssprk3"]:
        odRomObj.runSSPRK3(outDir, romState, appObjForRom, numSteps, dt, obsO)
      elif odeScheme in ["RungeKutta4", "RK4", "rk4"]:
        odRomObj.runRK4(outDir, romState, appObjForRom, numSteps, dt, obsO)
      elif odeScheme in ["RungeKutta2", "RK2", "rk2"]:
        odRomObj.runRK2(outDir, romState, appObjForRom, numSteps, dt, obsO)

      elapsed = time.time() - pTimeStart
      print("elapsed = {}".format(elapsed))
      f = open(outDir+"/timing.txt", "w")
      f.write(str(elapsed))
      f.close()

      # tell observer to write snapshots to file
      obsO.write(outDir)
      # reconstruct final state
      odRomObj.reconstructMemberFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_final.txt", odRomObj.viewFomState())


# -------------------------------------------------------------------
def run_full_od_galerkin_pod_bases(workDir, problem, module, \
                                   scenario, fomMeshPath, ):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in partInfoDirs:
    tiles = np.loadtxt(partInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    nTiles = nTilesX*nTilesY
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # for each one, find the associated full mesh with the indexiging
    # suitable for doing the ODROM. Note: each decomposition should
    # have a unique full mesh associated with it.
    topoString = str(nTilesX)+"x"+str(nTilesY)
    odMeshDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if topoString in d and "full_mesh" in d]
    assert(len(odMeshDirs)==1)
    odRomMeshDir = odMeshDirs[0]
    # the mesh found becomes the mesh to use for the odrom.
    # we can make a mesh object for this decomposition once
    # and used for all the runs below
    odRomMeshObj = pda.load_cellcentered_uniform_mesh(odRomMeshDir)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    howManySets = len(module.odrom_basis_sets[scenario].keys())
    for setId in range(howManySets):
      currPodDir = path_to_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3: over all target energies
      # ------
      for energyValue in module.odrom_energies[scenario]:
        modesPerTileDic = find_modes_per_tile_from_target_energy(currPodDir, energyValue)

        # -------
        # loop 4: over all test values
        # ------
        run_full_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                 scenario, fomMeshPath, \
                                                 partInfoDirIt, currPodDir, \
                                                 energyValue, None, \
                                                 modesPerTileDic, \
                                                 odRomMeshObj, nTiles, setId, \
                                                 "using_pod_bases")

# -------------------------------------------------------------------
def run_od_galerkin_same_poly_bases_in_all_tiles(workDir, problem, module, \
                                                 scenario, fomMeshPath, \
                                                 polyOrders):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in partInfoDirs:
    tiles  = np.loadtxt(partInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    nTiles = nTilesX*nTilesY
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # for each one, find the associated full mesh with the indexiging
    # suitable for doing the ODROM. Note: each decomposition should
    # have a unique full mesh associated with it.
    topoString = str(nTilesX)+"x"+str(nTilesY)
    odMeshDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if topoString in d and "full_mesh" in d]
    assert(len(odMeshDirs)==1)
    odRomMeshDir = odMeshDirs[0]
    # the mesh found becomes the mesh to use for the odrom.
    # we can make a mesh object for this decomposition once
    # and used for all the runs below
    odRomMeshObj = pda.load_cellcentered_uniform_mesh(odRomMeshDir)

    # -------
    # loop 2: over all target orders
    # ------
    for orderIt in polyOrders:
      polyBasesDir = path_to_poly_bases_data_dir(workDir, partitionStringIdentifier, orderIt)
      if not os.path.exists(polyBasesDir):
        os.system('mkdir -p ' + polyBasesDir)
        compute_poly_bases_same_order_all_tiles(fomMeshPath, polyBasesDir, \
                                                partInfoDirIt, orderIt,\
                                                module.numDofsPerCell)

      modesPerTileArr = np.loadtxt(polyBasesDir+"/modes_per_tile.txt", dtype=int)
      modesPerTile = {}
      if nTiles==1:
        modesPerTile[0] = int(modesPerTileArr)
      else:
        for i in np.arange(nTiles):
          modesPerTile[i] = int(modesPerTileArr[i])

      # -------
      # loop 3: over all test values
      # ------
      # for this case, we have a constant poly order in all tiles,
      # so no setId needed because bases do not depend on the train data at all
      setId = None
      run_full_od_galerkin_for_all_test_values(workDir, problem, module, \
                                               scenario, fomMeshPath, \
                                               partInfoDirIt, polyBasesDir, \
                                               None, orderIt, modesPerTile, \
                                               odRomMeshObj, nTiles, setId, \
                                               "using_poly_bases")

# -------------------------------------------------------------------
def run_od_poly_galerkin_find_order_matching_pod_modes_count(workDir, problem, \
                                                             module, scenario, \
                                                             fomMeshPath):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in partInfoDirs:
    tiles = np.loadtxt(partInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    nTiles = nTilesX*nTilesY
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # for each one, find the associated full mesh with the indexiging
    # suitable for doing the ODROM. Note: each decomposition should
    # have a unique full mesh associated with it.
    topoString = str(nTilesX)+"x"+str(nTilesY)
    odMeshDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if topoString in d and "full_mesh" in d]
    assert(len(odMeshDirs)==1)
    odRomMeshDir = odMeshDirs[0]
    # the mesh found becomes the mesh to use for the odrom.
    # we can make a mesh object for this decomposition once
    # and used for all the runs below
    odRomMeshObj = pda.load_cellcentered_uniform_mesh(odRomMeshDir)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    howManySets = len(module.odrom_basis_sets[scenario].keys())
    for setId in range(howManySets):
      currPodDir = path_to_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3: over all target POD energies
      # ------
      for energyValue in module.odrom_energies[scenario]:
        modesPerTilePod = find_modes_per_tile_from_target_energy(currPodDir, energyValue)

        # now that we know POD modes per tile, we create local poly bases
        # such that the order yields num of modes that matches POD bases
        polyBasesDir = path_to_poly_bases_data_dir(workDir, partitionStringIdentifier, \
                                                   -1, energyValue, setId)
        if not os.path.exists(polyBasesDir):
          os.system('mkdir -p ' + polyBasesDir)
          compute_poly_bases_to_match_pod(fomMeshPath, polyBasesDir, \
                                          partInfoDirIt, modesPerTilePod, \
                                          module.numDofsPerCell)

        polyModesPerTileArr = np.loadtxt(polyBasesDir+"/modes_per_tile.txt", dtype=int)
        polyModesPerTile = {}
        if nTiles==1:
          polyModesPerTile[0] = int(polyModesPerTileArr)
        else:
          for i in np.arange(nTiles):
            polyModesPerTile[i] = int(polyModesPerTileArr[i])

        # -------
        # loop 4: over all test values
        # ------
        run_full_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                 scenario, fomMeshPath, \
                                                 partInfoDirIt, polyBasesDir, \
                                                 energyValue, -1, polyModesPerTile, \
                                                 odRomMeshObj, nTiles, setId, \
                                                 "using_poly_bases")

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
  meshSize = args.mesh

  if not os.path.exists(workDir):
    print("Working dir {} does not exist, creating it".format(workDir))
    os.system('mkdir -p ' + workDir)
    print("")

  # write scenario id, problem to file
  write_scenario_to_file(scenario, workDir)
  write_problem_name_to_file(problem, workDir)

  print("========================")
  print("Importing problem module")
  print("========================")
  module = importlib.import_module(problem)

  try:    print("{}: dimensionality = {}".format(problem, module.dimensionality))
  except: sys.exit("Missing dimensionality in problem's module")

  try:    print("{}: numDofsPerCell = {}".format(problem, module.numDofsPerCell))
  except: sys.exit("Missing numDofsPerCell in problem's module")

  print(module)
  print("")

  print("========================")
  print("Make full mesh ")
  print("========================")
  make_fom_mesh_if_not_existing(workDir, problem, \
                                module, scenario, \
                                pdaDir, meshSize)
  print("")

  # before we move on, we need to ensure that in workDir
  # there is a unique FULL mesh. This is because the mesh is specified
  # via command line argument and must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory
  fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

  print("========================")
  print("FOM Train runs")
  print("========================")
  run_foms(workDir, problem, module, scenario, "train", fomMeshPath)
  print("")

  print("========================")
  print("FOM Test runs")
  print("========================")
  run_foms(workDir, problem, module, scenario, "test", fomMeshPath)
  print("")

  print("========================")
  print("Compute partitions")
  print("========================")
  # loop over all target styles for creating the tiles:
  # the simplest and currently only supported is uniform but we can add more.
  # E.g. might add a new method that use FOM train data so that is also why
  # the partitioning stage makes sense to be AFTER the FOM train runs are complete
  for partitioningStyleIt in module.odrom_partitioning_style[scenario]:
    if module.dimensionality == 2 and partitioningStyleIt == "uniform":
      make_uniform_partitions_2d(workDir, module, scenario, fomMeshPath)
    else:
      sys.exit("Invalid dimensionality or partitionStyle = {}".format(module.dimensionality, \
                                                                      partitioningStyleIt))
  print("")

  print("=====================================")
  print("Compute partition-based full meshes ")
  print("=====================================")
  # for FULL od-rom without HR, for performance reasons,
  # we don't/should not use the same full mesh used in the fom.
  # We need to make a new full mesh with a new indexing
  # that is consistent with the partitions and allows continguous storage
  # of the state and rhs within each tile
  make_full_mesh_for_odrom_using_partition_based_indexing(workDir, pdaDir, module, fomMeshPath)
  print("")

  print("=====================================")
  print("Compute partition-based POD")
  print("=====================================")
  '''
  The pod modes for each tile must be computed in two cases:
  1. if the target scenario explicitly has podGalerkin
  2. if scenario has PolyGalerkin on and the poly_order = -1
     because poly_order = -1 indicates that we compute the poly order
     in each tile such that we match as possible the number of local pod modes
  '''
  mustComputeTiledPodModes = False
  if "PodGalerkinFull" in module.odrom_algos[scenario]:
    mustComputeTiledPodModes = True
  if "PolyGalerkinFull" in module.odrom_algos[scenario] and \
     -1 in module.odrom_poly_order[scenario]:
    mustComputeTiledPodModes = True

  if mustComputeTiledPodModes:
    for setId, trainIndices in module.odrom_basis_sets[scenario].items():
      print("Handling setId = {}".format(setId))
      print("------------------------")
      trainDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "train" in d and get_run_id(d) in trainIndices]
      assert(len(trainDirs) == len(trainIndices))

      compute_partition_based_state_and_rhs_pod(workDir, setId, trainDirs, module, fomMeshPath)
    print("")

  print("=====================================")
  print("Running FULL poly od-galerkin ")
  print("=====================================")
  if "PolyGalerkinFull" in module.odrom_algos[scenario]:

    # check if -1 is is part of the order list, and if so that means
    # we need to use polyn bases such that in each tile we decide the order
    # based on the number of POD modes to have a fair comparison
    if -1 in module.odrom_poly_order[scenario]:
      run_od_poly_galerkin_find_order_matching_pod_modes_count(workDir, \
                                                               problem, module, \
                                                               scenario, \
                                                               fomMeshPath)

    # run the case involving each partition of same poly bases,
    # so we need to list all the orders we want that are != -1
    polyOrders = [i for i in module.odrom_poly_order[scenario] if i > 0]
    if polyOrders:
      run_od_galerkin_same_poly_bases_in_all_tiles(workDir, problem, module, \
                                                   scenario, fomMeshPath, polyOrders)

  print("=====================================")
  print("Running FULL pod od-galerkin ")
  print("=====================================")
  if "PodGalerkinFull" in module.odrom_algos[scenario]:
    run_full_od_galerkin_pod_bases(workDir, problem, module, scenario, fomMeshPath)

  print("")
