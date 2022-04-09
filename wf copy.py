
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import re, time, yaml, random, subprocess
import numpy as np
from scipy import linalg as scipyla
from decimal import Decimal
from scipy import optimize as sciop

# try:
#   import pressiotools.linalg as ptla
#   from pressiotools.samplemesh.withLeverageScores import computeNodes
# except ImportError:
#   raise ImportError("Unable to import classes from pressiotools")

try:
  import pressiodemoapps as pda
except ImportError:
  raise ImportError("Unable to import pressiodemoapps")

# local imports
from py_src.myio import *
from py_src.legendre_bases import LegendreBases1d, LegendreBases2d
from py_src.observer import FomObserver, RomObserver
from py_src.odrom_full import *
from py_src.odrom_gappy import *
from py_src.odrom_masked_gappy import *
from py_src.odrom_time_integrators import *
from py_src.standardrom_full import *

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#                    SPECIFIC FOR STANDARD ROM
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def compute_full_domain_state_pod(workDir, module, scenario, \
                                  setId, dataDirs, fomMesh):
  '''
  compute pod from state snapshots on the FULL domain
  '''

  outDir = path_to_full_domain_state_pod_data_dir(workDir, setId)
  if os.path.exists(outDir):
    print('{} already exists'.format(outDir))
  else:
    fomTotCells = find_total_cells_from_info_file(fomMesh)
    totFomDofs  = fomTotCells*module.numDofsPerCell
    # find from scenario if we want to subtract initial condition
    # from snapshots before doing pod.
    subtractInitialCondition = module.use_ic_reference_state[scenario]

    # load snapshots once
    fomStateSnapsFullDomain = load_fom_state_snapshot_matrix(dataDirs, totFomDofs, \
                                                             module.numDofsPerCell, \
                                                             subtractInitialCondition)
    print("pod: fomStateSnapsFullDomain.shape = ", fomStateSnapsFullDomain.shape)

    os.system('mkdir -p ' + outDir)
    lsvFile = outDir + '/lsv_state_p_0'
    svaFile = outDir + '/sva_state_p_0'
    do_svd_py(fomStateSnapsFullDomain, lsvFile, svaFile)
  print("")

# -------------------------------------------------------------------
def compute_full_domain_rhs_pod(workDir, module, scenario, \
                                setId, dataDirs, fomMesh):
  '''
  compute pod for rhs snapshosts on FULL domain
  '''
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell
  fomRhsSnapsFullDomain   = load_fom_rhs_snapshot_matrix(dataDirs, totFomDofs, \
                                                         module.numDofsPerCell)
  print("pod: fomRhsSnapsFullDomain.shape = ", fomRhsSnapsFullDomain.shape)

  outDir = path_to_full_domain_rhs_pod_data_dir(workDir, setId)
  if os.path.exists(outDir):
    print('{} already exists'.format(outDir))
  else:
    os.system('mkdir -p ' + outDir)
    lsvFile = outDir + '/lsv_rhs_p_0'
    svaFile = outDir + '/sva_rhs_p_0'
    do_svd_py(fomRhsSnapsFullDomain, lsvFile, svaFile)
  print("")

# -------------------------------------------------------------------
def find_modes_for_full_domain_from_target_energy(module, scenario, podDir, energy):
  singValues = np.loadtxt(podDir+'/sva_state_p_0')
  return compute_cumulative_energy(singValues, energy)

# -------------------------------------------------------------------
def compute_quad_projector_single_tile(fomTrainDirs, fomMesh, outDir, \
                                       partitionInfoDir, statePodDir, \
                                       sampleMeshDir, modesPerTileDic, \
                                       numDofsPerCell):
  fomTotCells      = find_total_cells_from_info_file(fomMesh)
  totFomDofs       = fomTotCells*numDofsPerCell
  fSnapsFullDomain = load_fom_rhs_snapshot_matrix(fomTrainDirs, totFomDofs, numDofsPerCell)

  myNumModes = modesPerTileDic[0]

  # load my phi on full mesh
  myPhiFile     = statePodDir + "/lsv_state_p_0"
  myPhiFullMesh = load_basis_from_binary_file(myPhiFile)[:,0:myNumModes]

  mySmMeshGids  = np.loadtxt(sampleMeshDir + "/sample_mesh_gids_p_0.txt", dtype=int)
  mySmCount     = len(mySmMeshGids)
  print("required = ", numDofsPerCell* mySmCount)
  print("snaps #  = ", fSnapsFullDomain.shape[1] )
  assert( numDofsPerCell* mySmCount <= fSnapsFullDomain.shape[1] )

  # phi on sample mesh
  myPhiSampleMesh = np.zeros((mySmCount*numDofsPerCell, myPhiFullMesh.shape[1]), order='F')
  for j in range(numDofsPerCell):
    myPhiSampleMesh[j::numDofsPerCell, :] = myPhiFullMesh[numDofsPerCell*mySmMeshGids + j, :]

  # get rhs snaps on sample mesh
  myfSnapsSampleMesh = np.zeros((mySmCount*numDofsPerCell, fSnapsFullDomain.shape[1]), order='F')
  for j in range(numDofsPerCell):
    myfSnapsSampleMesh[j::numDofsPerCell, :] = fSnapsFullDomain[numDofsPerCell*mySmMeshGids + j, :]

  print("myPhiSampleMesh.shape = ", myPhiSampleMesh.shape)
  print("myfSnapsSampleMesh.shape = ", myfSnapsSampleMesh.shape)
  print("fSnapsFullDomain.shape = ", fSnapsFullDomain.shape)

  # setup sequence of ls problem: minimize (Aw - b)
  # initialize weights (weights for each basis vector)
  W = np.zeros_like(myPhiSampleMesh)
  print(W.shape)
  for j in range(myPhiFullMesh.shape[1]):
    A = myPhiSampleMesh[:,j:j+1] * myfSnapsSampleMesh[:, :]
    print("A.shape = ", A.shape)
    b = myPhiFullMesh[:,j].transpose() @ fSnapsFullDomain[:, :]
    print("b.shape = ", b.shape)
    W[:,j], _ = sciop.nnls(A.T, b, maxiter=5000)

  mjop = myPhiSampleMesh * W
  # save mjop to file
  np.savetxt(outDir+'/projector_p_'+str(0)+'.txt', mjop)


# -------------------------------------------------------------------
def run_full_standard_galerkin_for_all_test_values(workDir, problem, \
                                                   module, scenario, \
                                                   fomMeshPath, basesDir, \
                                                   modeSettingPolicy, \
                                                   energyValue, numModes, setId):

  meshObj = pda.load_cellcentered_uniform_mesh(fomMeshPath)

  # this is rom WITHOUT HR, so the following should hold:
  stencilDofsCount = meshObj.stencilMeshSize()*module.numDofsPerCell
  sampleDofsCount  = meshObj.sampleMeshSize()*module.numDofsPerCell
  assert(stencilDofsCount == sampleDofsCount)
  fomTotalDofs = stencilDofsCount

  # loop over all test param values to do
  param_values = module.test_points[scenario]
  for k, val in param_values.items():

    # figure out the name of the output directory
    outDir = workDir + "/standardrom_full_modesSettingPolicy_" + modeSettingPolicy
    if energyValue != None:
      outDir += "_"+str(energyValue)
    if energyValue == None:
      outDir += "_"+str(numModes)
    if setId != None:
      outDir += "_set_"+str(setId)
    outDir += "_"+str(k)

    # check outdir, make and run if needed
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print("Running standard rom in {}".format(os.path.basename(outDir)))
      os.system('mkdir -p ' + outDir)
      romRunDic = module.base_dic[scenario]['odrom'].copy()
      coeffDic  = module.base_dic[scenario]['physicalCoefficients'].copy()
      appObj    = module.create_problem_for_scenario(scenario, meshObj, \
                                                     coeffDic, romRunDic, val)
      # write some info to run directory
      f = open(outDir+"/rom_dofs_count.txt", "w")
      f.write(str(numModes))
      f.close()

      if energyValue != None:
        romRunDic['energy'] = energyValue

      romRunDic['basesDir'] = basesDir
      romRunDic['numDofsPerCell'] = module.numDofsPerCell

      usingIcAsRefState = module.use_ic_reference_state[scenario]
      romRunDic['usingIcAsRefState'] = usingIcAsRefState

      # make ROM initial state
      romState = None
      if usingIcAsRefState:
        # dont need to do projection, romState is simply all zeros
        romState = np.zeros(numModes)
      else:
        myPhi = load_basis_from_binary_file(basesDir+"/lsv_state_p_0")[:,0:numModes]
        fomIc = appObj.initialCondition()
        romState = np.dot(myPhi.transpose(), fomIc)

      refState = appObj.initialCondition() \
        if usingIcAsRefState else np.array([None])

      # construct standard rom object
      romObj = StandardRomFull(appObj, module.dimensionality, \
                               module.numDofsPerCell, numModes, \
                               basesDir, refState)
      # initial condition
      romObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_ic.txt", romObj.viewFomState())

      # time loop
      dt         = romRunDic['dt']
      finalTime  = romRunDic['finalTime']
      numSteps   = int(finalTime/dt)
      odeScheme  = romRunDic['odeScheme']
      romRunDic['numSteps'] = numSteps

      # create observer
      stateSamplingFreq = int(module.base_dic[scenario]['stateSamplingFreqTest'])
      romRunDic['stateSamplingFreq'] = stateSamplingFreq
      # here I need to pass {0: numModes} because of API compatibility
      obsO = RomObserver(stateSamplingFreq, numSteps, {0: numModes})

      # write yaml to file
      inputFile = outDir + "/input.yaml"
      write_dic_to_yaml_file(inputFile, romRunDic)

      pTimeStart = time.time()
      if odeScheme in ["SSPRK3", "ssprk3"]:
        odrom_ssprk3(romObj, romState, numSteps, dt, obsO)
      elif odeScheme in ["RungeKutta4", "RK4", "rk4"]:
        odrom_rk4(romObj, romState, numSteps, dt, obsO)
      elif odeScheme in ["RungeKutta2", "RK2", "rk2"]:
        odrom_rk2(romObj, romState, numSteps, dt, obsO)

      elapsed = time.time() - pTimeStart
      print("elapsed = {}".format(elapsed))
      f = open(outDir+"/timing.txt", "w")
      f.write(str(elapsed))
      f.close()

      # tell observer to write snapshots to file
      obsO.write(outDir)
      # reconstruct final state
      romObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_final.txt", romObj.viewFomState())
      print("")

# -------------------------------------------------------------------
def run_standard_pod_galerkin_full(workDir, problem, module, \
                                   scenario, fomMeshPath):

  # -------
  # loop 2: over all POD computed from various sets of train runs
  # ------
  howManySets = len(module.basis_sets[scenario].keys())
  for setId in range(howManySets):
    currPodDir = path_to_full_domain_state_pod_data_dir(workDir, setId)

    # -------
    # loop 3: over all target energies
    # ------
    if not hasattr(module, 'standardrom_modes_setting_policies'):
      sys.exit("for standard galerkin, you need standardrom_modes_setting_policies in the problem")
    if scenario not in module.standardrom_modes_setting_policies:
      sys.exit("scenario = {} not valid key in module.standardrom_modes_setting_policies".format(scenario))

    for modeSettingIt_key, modeSettingIt_val in module.standardrom_modes_setting_policies[scenario].items():

      if modeSettingIt_key == 'userDefinedValue':
        for numModes in modeSettingIt_val:
          run_full_standard_galerkin_for_all_test_values(workDir, problem, module, \
                                                         scenario, fomMeshPath, \
                                                         currPodDir, modeSettingIt_key, \
                                                         None, numModes, setId)
      elif modeSettingIt_key == 'energyBased':
        for energyValue in modeSettingIt_val:
          numModes = find_modes_for_full_domain_from_target_energy(module, scenario, \
                                                                   currPodDir, energyValue)
          run_full_standard_galerkin_for_all_test_values(workDir, problem, module, \
                                                         scenario, fomMeshPath, \
                                                         currPodDir, modeSettingIt_key, \
                                                         energyValue, numModes, setId)

      else:
        sys.exit('run_standard_pod_galerkin_full: invalid modeSettingPolicy = {}'.format(modeSettingIt_key))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#                    SPECIFIC FOR OD-ROM
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# -------------------------------------------------------------------
def make_rectangular_uniform_partitions_1d(workDir, fullMeshPath, listOfTilings):
  '''
  tile a 1d mesh using uniform partitions if possible
  '''
  this_file_path = pathlib.Path(__file__).parent.absolute()

  for nTilesX in listOfTilings:
    outDir = path_to_partition_info_dir(workDir, nTilesX, 1, "rectangularUniform")
    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for \n{}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(this_file_path)+'/partition_uniform.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTilesX),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

# -------------------------------------------------------------------
def make_rectangular_uniform_partitions_2d(workDir, fullMeshPath, listOfTilings):
  '''
  tile a 2d mesh using uniform partitions if possible
  '''
  this_file_path = pathlib.Path(__file__).parent.absolute()

  for pIt in listOfTilings:
    nTilesX, nTilesY = pIt[0], pIt[1]

    outDir = path_to_partition_info_dir(workDir, nTilesX, nTilesY, "rectangularUniform")
    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for \n{}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(this_file_path)+'/partition_uniform.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTilesX), str(nTilesY),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

# -------------------------------------------------------------------
def make_concentric_uniform_partitions_2d(workDir, fullMeshPath, listOfTilings):
  this_file_path = pathlib.Path(__file__).parent.absolute()

  for nTiles in listOfTilings:
    outDir = path_to_partition_info_dir(workDir, nTiles, None, "concentricUniform")
    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for \n{}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(this_file_path)+'/partition_radial.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTiles),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

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

      print('Generating partition-based FULL mesh in:')
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
def compute_poly_bases_same_order_all_tiles(dimens, fomMesh, outDir, \
                                            partInfoDir, \
                                            targetOrder, \
                                            numDofsPerCell):

  assert(dimens in [1,2])

  fomCellsXcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,1]
  fomCellsYcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,2]
  nTiles = np.loadtxt(partInfoDir+"/ntiles.txt", dtype=int)
  #nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

  polyObj = LegendreBases2d("totalOrder") if dimens == 2 else LegendreBases1d("totalOrder")
  modesPerTile = {}
  for tileId in range(nTiles):
    myFile = partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myRows = np.loadtxt(myFile, dtype=int)

    myX, myY = fomCellsXcoords[myRows], fomCellsYcoords[myRows]
    lsvFile = outDir + '/lsv_state_p_'+str(tileId)

    if dimens == 1:
      U0 = polyObj(targetOrder, myX)
    else:
      U0 = polyObj(targetOrder, myX, myY)

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
def compute_poly_bases_to_match_pod(dimens, fomMesh, outDir, \
                                    partInfoDir, \
                                    podModesPerTileToMatch,
                                    numDofsPerCell):

  assert(dimens in [1,2])

  fomCellsXcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,1]
  fomCellsYcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,2]
  nTiles = np.loadtxt(partInfoDir+"/ntiles.txt", dtype=int)
  #nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

  polyObj = LegendreBases2d("totalOrder") if dimens == 2 else LegendreBases1d("totalOrder")
  modesPerTile = {}
  for tileId in range(nTiles):
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

    if dimens == 1:
      U0 = polyObj(order, myX)
    else:
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
def compute_total_modes_across_all_tiles(modesPerTileDic):
  return np.sum(list(modesPerTileDic.values()))

# -------------------------------------------------------------------
def find_modes_per_tile_from_target_energy(module, scenario, podDir, energy):
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
    modesPerTileDic[tileId] = max(K, module.odrom_min_num_modes_per_tile[scenario])

  return modesPerTileDic

# -------------------------------------------------------------------
def make_bases_on_sample_mesh(partInfoDir, tileId, sampleMeshPath, \
                              phiFullMesh, numDofsPerCell):

   myCellGids   = np.loadtxt(partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt",dtype=int)
   mySmMeshGids = np.loadtxt(sampleMeshPath + "/sample_mesh_gids_p_"+str(tileId)+".txt", dtype=int)
   mySmCount    = len(mySmMeshGids)

   commonElem  = set(mySmMeshGids).intersection(myCellGids)
   commonElem  = np.sort(list(commonElem))
   mylocalinds = np.searchsorted(myCellGids, commonElem)
   phiOnSampleMesh = np.zeros((mySmCount*numDofsPerCell, phiFullMesh.shape[1]), order='F')
   for j in range(numDofsPerCell):
     phiOnSampleMesh[j::numDofsPerCell, :] = phiFullMesh[numDofsPerCell*mylocalinds + j, :]

   return phiOnSampleMesh

# -------------------------------------------------------------------
def compute_quad_projector(fomTrainDirs, fomMesh, outDir, \
                           partitionInfoDir, statePodDir, \
                           sampleMeshDir, modesPerTileDic, \
                           numDofsPerCell):

  # load f snapshots
  fomTotCells      = find_total_cells_from_info_file(fomMesh)
  totFomDofs       = fomTotCells*module.numDofsPerCell
  fSnapsFullDomain = load_fom_rhs_snapshot_matrix(fomTrainDirs, totFomDofs, numDofsPerCell)

  nTiles = len(modesPerTileDic)
  for tileId in range(nTiles):
    myNumModes = modesPerTileDic[tileId]

    # load my phi on full mesh
    myPhiFile     = statePodDir + "/lsv_state_p_" + str(tileId)
    myPhiFullMesh = load_basis_from_binary_file(myPhiFile)[:,0:myNumModes]

    # restrict on sample mesh
    myPhiSampleMesh = make_bases_on_sample_mesh(partitionInfoDir, tileId, \
                                                sampleMeshDir, myPhiFullMesh,\
                                                numDofsPerCell)
    assert(myPhiSampleMesh.shape[1] == myPhiFullMesh.shape[1])

    # indexing info
    cellGidsFile   = partitionInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myCellGids     = np.loadtxt(cellGidsFile, dtype=int)
    sampleGidsFile = sampleMeshDir + "/sample_mesh_gids_p_"+str(tileId)+".txt"
    mySmMeshGids   = np.loadtxt(sampleGidsFile, dtype=int)
    mySmCount      = len(mySmMeshGids)
    print(numDofsPerCell* mySmCount)
    print(fSnapsFullDomain.shape[1])

    # get rhs snaps on sample mesh
    rowsFile = partitionInfoDir + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
    myRowsInFullState = np.loadtxt(rowsFile, dtype=int)
    myRhsSnaps  = fSnapsFullDomain[myRowsInFullState, :]
    assert( numDofsPerCell* mySmCount <= myRhsSnaps.shape[1] )

    commonElem  = set(mySmMeshGids).intersection(myCellGids)
    commonElem  = np.sort(list(commonElem))
    mylocalinds = np.searchsorted(myCellGids, commonElem)
    myfSnapsSampleMesh = np.zeros((mySmCount*numDofsPerCell, myRhsSnaps.shape[1]), order='F')
    print(myfSnapsSampleMesh.shape)
    print(len(mylocalinds))
    for j in range(numDofsPerCell):
      myfSnapsSampleMesh[j::numDofsPerCell, :] = myRhsSnaps[numDofsPerCell*mylocalinds + j, :]

    # setup sequence of ls problem: minimize (Aw - b)
    # initialize weights (weights for each basis vector)
    W = np.zeros_like(myPhiSampleMesh)
    print(W.shape)

    numModes = myPhiFullMesh.shape[1]
    for j in range(numModes):
      A = myPhiSampleMesh[:,j:j+1] * myfSnapsSampleMesh[:,:]
      b = myPhiFullMesh[:,j].transpose() @ myRhsSnaps[:, :]
      W[:,j],_ = sciop.nnls(A.T, b, maxiter=5000)

    mjop = myPhiSampleMesh * W
    np.savetxt(outDir+'/projector_p_'+str(tileId)+'.txt', mjop)


# -------------------------------------------------------------------
def compute_phi_on_stencil(outDir, partitionInfoDir, \
                           statePodDir, sampleMeshDir, \
                           modesPerTileDic, numDofsPerCell):

  nTiles = len(modesPerTileDic)
  maxNumRows = 0
  rowsPerTile = []
  for tileId in range(nTiles):
    myNumModes = modesPerTileDic[tileId]

    # load my full phi
    myPhiFile = statePodDir + "/lsv_state_p_" + str(tileId)
    myPhi     = load_basis_from_binary_file(myPhiFile)[:,0:myNumModes]

    # load indices such that we can extract phi on stencil mesh
    myCellGids   = np.loadtxt(partitionInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt",dtype=int)
    myStMeshGids = np.loadtxt(sampleMeshDir + "/stencil_mesh_gids_p_"+str(tileId)+".dat", dtype=int)
    myStCount    = len(myStMeshGids)

    commonElem  = set(myStMeshGids).intersection(myCellGids)
    commonElem  = np.sort(list(commonElem))
    mylocalinds = np.searchsorted(myCellGids, commonElem)
    mySlicedPhi = np.zeros((myStCount*numDofsPerCell, myNumModes), order='F')
    for j in range(numDofsPerCell):
      mySlicedPhi[j::numDofsPerCell, :] = myPhi[numDofsPerCell*mylocalinds + j, :]

    maxNumRows = max(maxNumRows, mySlicedPhi.shape[0])
    rowsPerTile.append(mySlicedPhi.shape[0])
    np.savetxt(outDir+'/phi_on_stencil_p_'+str(tileId)+'.txt', mySlicedPhi)

  np.savetxt(outDir+'/max_num_rows.txt', np.array([int(maxNumRows)]), fmt="%6d")
  np.savetxt(outDir+'/rows_per_tile.txt', np.array(rowsPerTile), fmt="%6d")

# -------------------------------------------------------------------
def compute_gappy_projector(outDir, partitionInfoDir, \
                            statePodDir, rhsPodDir, sampleMeshDir, \
                            modesPerTileDic, numDofsPerCell):

  nTiles = len(modesPerTileDic)
  maxNumRows = 0
  rowsPerTile = []
  for tileId in range(nTiles):
    myNumModes = modesPerTileDic[tileId]

    # load my full phi
    myFullPhiFile = statePodDir + "/lsv_state_p_" + str(tileId)
    myFullPhi     = load_basis_from_binary_file(myFullPhiFile)[:,0:myNumModes]

    # indexing info
    myFile1      = partitionInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myCellGids   = np.loadtxt(myFile1, dtype=int)
    myFile2      = sampleMeshDir + "/sample_mesh_gids_p_"+str(tileId)+".txt"
    mySmMeshGids = np.loadtxt(myFile2, dtype=int)
    mySmCount    = len(mySmMeshGids)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # WARNING:  might need to change this but this seems to
    # work better than other things
    rhsSingVals = np.loadtxt(rhsPodDir + "/sva_rhs_p_" + str(tileId))
    K = compute_cumulative_energy(rhsSingVals, 99.9999)
    #K = myNumModes*4 + 1
    print("tile::: ", K, myNumModes, mySmCount)
    if mySmCount*numDofsPerCell < K:
      print("Cannot have K > mySmCount*numDofsPerCell in tileId = {:>5}, adapting K".format(tileId))
      K = mySmCount*numDofsPerCell - 1

    # K should be larger than myNumModes
    if K < myNumModes:
      print("Cannot have K < myNumModes in tileId = {:>5}, adapting K".format(tileId))
      K = myNumModes + 1

    myFullRhsPodFile = rhsPodDir + "/lsv_rhs_p_" + str(tileId)
    myTheta = load_basis_from_binary_file(myFullRhsPodFile)[:,0:K]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # need to slice pod of rhs (i.e. theta) to get elements on my sample mesh cells
    # note that I need to do the following beacuse the POD modes saved are computed
    # using the FOM rhs local data, which is not in the same order as the indexing
    # for the odrom, since the indexing within each tile has changed.
    # So I cannot just a get a contiguous subview of the theta matrix, but
    # I need to do a bit more maninulation to figure out which row indices to get
    commonElem = set(mySmMeshGids).intersection(myCellGids)
    commonElem = np.sort(list(commonElem))
    mylocalinds = np.searchsorted(myCellGids, commonElem)
    mySlicedTheta = np.zeros((mySmCount*numDofsPerCell, myTheta.shape[1]), order='F')
    for j in range(numDofsPerCell):
      mySlicedTheta[j::numDofsPerCell, :] = myTheta[numDofsPerCell*mylocalinds + j, :]

    A = myFullPhi.transpose() @ myTheta
    projector = A @ linalg.pinv(mySlicedTheta)
    print(" tileId = {:>5}, projectorShape = {}".format(tileId, projector.T.shape))

    # write to file
    # here when writing w need to consider that project above is computed
    # such that it is short and wide, so to write it to file we need to
    # "view" it as flipped. the actual num rows is the cols and vice versa.
    numRows = np.int64(projector.shape[1])
    numCols = np.int64(projector.shape[0])
    fileo = open(outDir+'/projector_p_'+str(tileId), "wb")
    np.array([numRows]).tofile(fileo)
    np.array([numCols]).tofile(fileo)
    projector.tofile(fileo)
    fileo.close()
    np.savetxt(outDir+'/projector_p_'+str(tileId)+'.txt', projector.T)
    maxNumRows = max(maxNumRows, projector.T.shape[0])
    rowsPerTile.append(projector.T.shape[0])

  np.savetxt(outDir+'/max_num_rows.txt', np.array([int(maxNumRows)]), fmt="%6d")
  np.savetxt(outDir+'/rows_per_tile.txt', np.array(rowsPerTile), fmt="%6d")


# -------------------------------------------------------------------
def compute_gappy_projector_using_factor_of_state_pod_modes(outDir, partitionInfoDir, \
                                                            statePodDir, rhsPodDir, \
                                                            sampleMeshDir, \
                                                            modesPerTileDic, \
                                                            numDofsPerCell):

  nTiles = len(modesPerTileDic)
  maxNumRows = 0
  rowsPerTile = []
  for tileId in range(nTiles):
    myNumStatePodModes = modesPerTileDic[tileId]

    # load my full phi
    myFullPhiFile = statePodDir + "/lsv_state_p_" + str(tileId)
    myFullPhi     = load_basis_from_binary_file(myFullPhiFile)[:,0:myNumStatePodModes]

    # indexing info
    myFile1      = partitionInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myCellGids   = np.loadtxt(myFile1, dtype=int)
    myFile2      = sampleMeshDir + "/sample_mesh_gids_p_"+str(tileId)+".txt"
    mySmMeshGids = np.loadtxt(myFile2, dtype=int)
    mySmCount    = len(mySmMeshGids)

    K = myNumStatePodModes*3 + 1
    print("tile::: ", K, myNumStatePodModes, mySmCount)
    if mySmCount*numDofsPerCell < K:
      print("Cannot have K > mySmCount*numDofsPerCell in tileId = {:>5}, adapting K".format(tileId))
      K = mySmCount*numDofsPerCell - 1

    # K should be larger than myNumStatePodModes
    if K < myNumStatePodModes:
      print("Cannot have K < myNumStatePodModes in tileId = {:>5}, adapting K".format(tileId))
      K = myNumStatePodModes + 1

    myFullRhsPodFile = rhsPodDir + "/lsv_rhs_p_" + str(tileId)
    myTheta = load_basis_from_binary_file(myFullRhsPodFile)[:,0:K]

    # need to slice pod of rhs (i.e. theta) to get elements on my sample mesh cells
    # note that I need to do the following beacuse the POD modes saved are computed
    # using the FOM rhs local data, which is not in the same order as the indexing
    # for the odrom, since the indexing within each tile has changed.
    # So I cannot just a get a contiguous subview of the theta matrix, but
    # I need to do a bit more maninulation to figure out which row indices to get
    commonElem = set(mySmMeshGids).intersection(myCellGids)
    commonElem = np.sort(list(commonElem))
    mylocalinds = np.searchsorted(myCellGids, commonElem)
    mySlicedTheta = np.zeros((mySmCount*numDofsPerCell, myTheta.shape[1]), order='F')
    for j in range(numDofsPerCell):
      mySlicedTheta[j::numDofsPerCell, :] = myTheta[numDofsPerCell*mylocalinds + j, :]

    A = myFullPhi.transpose() @ myTheta
    projector = A @ linalg.pinv(mySlicedTheta)
    print(" tileId = {:>5}, projectorShape = {}".format(tileId, projector.T.shape))

    # write to file
    # here when writing w need to consider that project above is computed
    # such that it is short and wide, so to write it to file we need to
    # "view" it as flipped. the actual num rows is the cols and vice versa.
    numRows = np.int64(projector.shape[1])
    numCols = np.int64(projector.shape[0])
    fileo = open(outDir+'/projector_p_'+str(tileId), "wb")
    np.array([numRows]).tofile(fileo)
    np.array([numCols]).tofile(fileo)
    projector.tofile(fileo)
    fileo.close()
    np.savetxt(outDir+'/projector_p_'+str(tileId)+'.txt', projector.T)
    maxNumRows = max(maxNumRows, projector.T.shape[0])
    rowsPerTile.append(projector.T.shape[0])

  np.savetxt(outDir+'/max_num_rows.txt', np.array([int(maxNumRows)]), fmt="%6d")
  np.savetxt(outDir+'/rows_per_tile.txt', np.array(rowsPerTile), fmt="%6d")



# -------------------------------------------------------------------
def run_hr_od_galerkin_for_all_test_values(workDir, problem, \
                                           module, scenario, partInfoDir, \
                                           fomMeshPath, odromSampleMeshPath, \
                                           fullPodDir, projectorDir, phiOnStencilDir,\
                                           modeSettingPolicy, \
                                           energyValue, modesPerTileDic, \
                                           setId, smKeywordForDirName,
                                           algoNameForDirName):

  # store various things
  romSizeOverAllPartitions = compute_total_modes_across_all_tiles(modesPerTileDic)
  fomMeshObj = pda.load_cellcentered_uniform_mesh(fomMeshPath)
  partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDir)

  # the pda meshobj to use in the odrom has to use the sample mesh files
  # generated from the pda script create_sample_mesh which are inside pda_sm
  romMeshObj = pda.load_cellcentered_uniform_mesh(odromSampleMeshPath+"/pda_sm")

  # loop over all test param values to do
  for k, val in module.test_points[scenario].items():

    # figure out the name of the output directory
    outDir = workDir + "/odrom_"+algoNameForDirName+"_"+partitionStringIdentifier
    outDir += "_modesSettingPolicy_"+modeSettingPolicy

    if 'Energy' in modeSettingPolicy:
      outDir += "_"+str(energyValue)
    elif modeSettingPolicy == 'allTilesUseTheSameUserDefinedValue':
      # all tiles use same value so pick first
      outDir += "_"+str(modesPerTileDic[0])

    if setId != None:
      outDir += "_set_"+str(setId)
    outDir += "_"+smKeywordForDirName+"_"+str(k)

    # check outdir, make and run if needed
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print("Running odrom in {}".format(os.path.basename(outDir)))
      os.system('mkdir -p ' + outDir)

      romRunDic    = module.base_dic[scenario]['odrom'].copy()
      coeffDic     = module.base_dic[scenario]['physicalCoefficients'].copy()
      appObjForIc  = None
      appObjForRom = None

      # we need distinct problems for initial condition and running the rom
      # this is because the rom initial condition should ALWAYS be computed
      # using the full FOM, regardless if we do hr or full rom.
      # The problem object for running the odrom must be one with a
      # modified cell indexing to suit the odrom implementation
      appObjForIc  = module.create_problem_for_scenario(scenario, fomMeshObj, \
                                                        coeffDic, romRunDic, val)
      appObjForRom = module.create_problem_for_scenario(scenario, romMeshObj,
                                                        coeffDic, romRunDic, val)
      # these objects should be valid
      assert(appObjForIc  != None)
      assert(appObjForRom != None)

      # write some info to run directory
      f = open(outDir+"/rom_dofs_count.txt", "w")
      f.write(str(romSizeOverAllPartitions))
      f.close()
      np.savetxt(outDir+"/modes_per_tile.txt", \
                 np.array(list(modesPerTileDic.values())),
                 fmt="%5d")
      romRunDic['meshDir']         = odromSampleMeshPath
      romRunDic['energy']          = energyValue
      romRunDic['fullPodDir']      = fullPodDir
      romRunDic['projectorDir']    = projectorDir
      romRunDic['phiOnStencilDir'] = phiOnStencilDir
      romRunDic['partioningInfo']  = partInfoDir
      romRunDic['numDofsPerCell']  = module.numDofsPerCell
      romRunDic['numTiles']        = len(modesPerTileDic.keys())

      # make ROM initial state
      usingIcAsRefState = module.use_ic_reference_state[scenario]
      romRunDic['usingIcAsRefState'] = usingIcAsRefState
      romState = make_od_rom_initial_condition(workDir, appObjForIc, \
                                               partInfoDir, fullPodDir, \
                                               modesPerTileDic, \
                                               romSizeOverAllPartitions, \
                                               usingIcAsRefState)

      # note that here we set two reference states because
      # one is used for reconstructing the fom state wrt full mesh indexing
      # while the other is used for doing reconstructiong for odrom indexing
      refStateForFullMeshOrdering = appObjForIc.initialCondition() \
        if usingIcAsRefState else np.array([None])
      refStateForOdRomAlgo = appObjForRom.initialCondition() \
        if usingIcAsRefState else np.array([None])

      # construct odrom object
      fomFullMeshTotalDofs = fomMeshObj.stencilMeshSize()*module.numDofsPerCell
      print(fomFullMeshTotalDofs)
      odRomObj = OdRomGappy(appObjForRom, module.dimensionality, \
                            module.numDofsPerCell, partInfoDir, \
                            modesPerTileDic, odromSampleMeshPath, \
                            fullPodDir, projectorDir, phiOnStencilDir,\
                            refStateForFullMeshOrdering, \
                            refStateForOdRomAlgo, \
                            fomFullMeshTotalDofs)

      ## initial condition
      odRomObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_ic.txt", odRomObj.viewFomStateOnFullMesh())

      # time loop
      dt         = romRunDic['dt']
      finalTime  = romRunDic['finalTime']
      numSteps   = int(finalTime/dt)
      odeScheme  = romRunDic['odeScheme']
      romRunDic['numSteps'] = numSteps

      # create observer
      stateSamplingFreq = int(module.base_dic[scenario]['stateSamplingFreqTest'])
      romRunDic['stateSamplingFreq'] = stateSamplingFreq
      obsO = RomObserver(stateSamplingFreq, numSteps, modesPerTileDic)

      inputFile = outDir + "/input.yaml"
      write_dic_to_yaml_file(inputFile, romRunDic)

      pTimeStart = time.time()
      if odeScheme in ["SSPRK3", "ssprk3"]:
        odrom_ssprk3(odRomObj, romState, numSteps, dt, obsO)

      elif odeScheme in ["RungeKutta4", "RK4", "rk4"]:
        odrom_rk4(odRomObj, romState, numSteps, dt, obsO)

      elif odeScheme in ["RungeKutta2", "RK2", "rk2"]:
        odrom_rk2(odRomObj, romState, numSteps, dt, obsO)

      elapsed = time.time() - pTimeStart
      print("elapsed = {}".format(elapsed))
      f = open(outDir+"/timing.txt", "w")
      f.write(str(elapsed))
      f.close()

      np.savetxt(outDir+"/rom_state_final.txt", romState)

      # tell observer to write snapshots to file
      obsO.write(outDir)
      # reconstruct final state
      odRomObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_final.txt", odRomObj.viewFomStateOnFullMesh())

# -------------------------------------------------------------------
def run_masked_gappy_od_galerkin_for_all_test_values(workDir, problem, \
                                                     module, scenario, \
                                                     partInfoDir, \
                                                     fomMeshPath, \
                                                     odromSampleMeshPath, \
                                                     podDir, projectorDir, \
                                                     energyValue, \
                                                     modesPerTileDic, \
                                                     setId, smKeywordForDirName):
  # store various things
  romSizeOverAllPartitions = compute_total_modes_across_all_tiles(modesPerTileDic)
  meshObj = pda.load_cellcentered_uniform_mesh(fomMeshPath)
  partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDir)

  # loop over all test param values to do
  param_values = module.test_points[scenario]
  for k, val in param_values.items():

    # figure out the name of the output directory
    outDir = workDir + "/odrom_masked_gappy_"+partitionStringIdentifier
    if energyValue != None:
      outDir += "_"+str(energyValue)
    if setId != None:
      outDir += "_set_"+str(setId)
    outDir += "_"+smKeywordForDirName+"_"+str(k)

    # check outdir, make and run if needed
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print("Running odrom in {}".format(os.path.basename(outDir)))
      os.system('mkdir -p ' + outDir)

      romRunDic    = module.base_dic[scenario]['odrom'].copy()
      coeffDic     = module.base_dic[scenario]['physicalCoefficients'].copy()

      appObj = module.create_problem_for_scenario(scenario, meshObj, \
                                                  coeffDic, romRunDic, val)
      # write some info to run directory
      f = open(outDir+"/rom_dofs_count.txt", "w")
      f.write(str(romSizeOverAllPartitions))
      f.close()
      np.savetxt(outDir+"/modes_per_tile.txt", \
                 np.array(list(modesPerTileDic.values())),
                 fmt="%5d")

      romRunDic['energy'] = energyValue
      romRunDic['podDir'] = podDir
      romRunDic['projectorDir'] = projectorDir
      romRunDic['partioningInfo'] = partInfoDir
      romRunDic['numDofsPerCell'] = module.numDofsPerCell

      # make ROM initial state
      usingIcAsRefState = module.use_ic_reference_state[scenario]
      romRunDic['usingIcAsRefState'] = usingIcAsRefState
      romState = make_od_rom_initial_condition(workDir, appObj, \
                                               partInfoDir, podDir, \
                                               modesPerTileDic, \
                                               romSizeOverAllPartitions, \
                                               usingIcAsRefState)

      refState = appObj.initialCondition() \
        if usingIcAsRefState else np.array([None])

      fomFullMeshTotalDofs = meshObj.stencilMeshSize()*module.numDofsPerCell
      print(fomFullMeshTotalDofs)
      odRomObj = OdRomMaskedGappy(appObj, module.dimensionality, \
                                  module.numDofsPerCell, partInfoDir, \
                                  modesPerTileDic, odromSampleMeshPath, \
                                  podDir, projectorDir, \
                                  refState, fomFullMeshTotalDofs)

      ## initial condition
      odRomObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_ic.txt", odRomObj.viewFomStateOnFullMesh())

      # time loop
      dt         = romRunDic['dt']
      finalTime  = romRunDic['finalTime']
      numSteps   = int(finalTime/dt)
      odeScheme  = romRunDic['odeScheme']
      romRunDic['numSteps'] = numSteps

      # create observer
      stateSamplingFreq = int(module.base_dic[scenario]['stateSamplingFreqTest'])
      romRunDic['stateSamplingFreq'] = stateSamplingFreq
      obsO = RomObserver(stateSamplingFreq, numSteps, modesPerTileDic)

      inputFile = outDir + "/input.yaml"
      write_dic_to_yaml_file(inputFile, romRunDic)

      pTimeStart = time.time()
      if odeScheme in ["SSPRK3", "ssprk3"]:
        odrom_ssprk3(odRomObj, romState, numSteps, dt, obsO)
      elif odeScheme in ["RungeKutta4", "RK4", "rk4"]:
        odrom_rk4(odRomObj, romState, numSteps, dt, obsO)
      elif odeScheme in ["RungeKutta2", "RK2", "rk2"]:
        odrom_rk2(odRomObj, romState, numSteps, dt, obsO)

      elapsed = time.time() - pTimeStart
      print("elapsed = {}".format(elapsed))
      f = open(outDir+"/timing.txt", "w")
      f.write(str(elapsed))
      f.close()

      # tell observer to write snapshots to file
      obsO.write(outDir)
      # reconstruct final state
      odRomObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_final.txt", odRomObj.viewFomStateOnFullMesh())


# -------------------------------------------------------------------
def run_od_pod_masked_galerkin_gappy(workDir, problem, module, \
                                     scenario, fomMeshPath):
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in find_all_partitions_info_dirs(workDir):
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    howManySets = len(module.basis_sets[scenario].keys())
    for setId in range(howManySets):
      currStatePodDir = path_to_state_pod_data_dir(workDir, partitionStringIdentifier, setId)
      currRhsPodDir   = path_to_rhs_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3: over all target energies
      # ------
      for energyValue in module.odrom_energies[scenario]:
        modesPerTileDic = find_modes_per_tile_from_target_energy(module, scenario, currStatePodDir, energyValue)

        # -------
        # loop 4: over all samples meshes
        # ------
        for sampleMeshDirIt in find_all_sample_meshes_for_target_partition_info(workDir, partInfoDirIt):
          smKeyword = string_identifier_from_sample_mesh_dir(sampleMeshDirIt)
          print(smKeyword)

          # compute gappy projector for each tile if needed
          projectorDir = path_to_gappy_projector_dir(workDir, \
                                                     partitionStringIdentifier, \
                                                     setId, energyValue, \
                                                     smKeyword)
          if os.path.exists(projectorDir):
            print('{} already exists'.format(projectorDir))
          else:
            print('Generating {}'.format(projectorDir))
            os.system('mkdir -p ' + projectorDir)
            compute_gappy_projector(projectorDir, partInfoDirIt,\
                                    currStatePodDir, currRhsPodDir,\
                                    sampleMeshDirIt, modesPerTileDic,
                                    module.numDofsPerCell)

          # -------
          # loop 5: over all test values
          # ------
          run_masked_gappy_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                           scenario, partInfoDirIt, \
                                                           fomMeshPath, sampleMeshDirIt, \
                                                           currStatePodDir, projectorDir, \
                                                           energyValue, modesPerTileDic, \
                                                           setId, smKeyword)

# -------------------------------------------------------------------
def run_od_pod_galerkin_quad(workDir, problem, module, \
                             scenario, fomMeshPath):

  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in find_all_partitions_info_dirs(workDir):
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    for setId, trainIndices in module.basis_sets[scenario].items():
      currStateFullPodDir = path_to_state_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # find all train dirs for current setId
      trainDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "train" in d and get_run_id(d) in trainIndices]
      assert(len(trainDirs) == len(trainIndices))

      # -------
      # loop 3: over all target energies
      # ------
      for energyValue in module.odrom_energies[scenario]:
        modesPerTileDic = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                 currStateFullPodDir, energyValue)

        # -------
        # loop 4: over all samples meshes
        # ------
        for sampleMeshDirIt in find_all_sample_meshes_for_target_partition_info(workDir, partInfoDirIt):
          smKeyword = string_identifier_from_sample_mesh_dir(sampleMeshDirIt)

          # compute projector for each tile if needed
          projectorDir = path_to_quad_projector_dir(workDir, \
                                                    partitionStringIdentifier, \
                                                    setId, \
                                                    energyValue, \
                                                    smKeyword)
          if os.path.exists(projectorDir):
            print('{} already exists'.format(projectorDir))
          else:
            print('Generating {}'.format(projectorDir))
            os.system('mkdir -p ' + projectorDir)
            compute_quad_projector(trainDirs, fomMeshPath, \
                                   projectorDir, partInfoDirIt, \
                                   currStateFullPodDir, \
                                   sampleMeshDirIt, modesPerTileDic,
                                   module.numDofsPerCell)

          # compute phi on stencil for each tile if needed
          phiOnStencilDir = path_to_phi_on_stencil_dir(workDir, \
                                                       partitionStringIdentifier, \
                                                       setId, \
                                                       energyValue, \
                                                       smKeyword)
          print(phiOnStencilDir)
          if os.path.exists(phiOnStencilDir):
            print('{} already exists'.format(phiOnStencilDir))
          else:
            print('Generating {}'.format(phiOnStencilDir))
            os.system('mkdir -p ' + phiOnStencilDir)
            compute_phi_on_stencil(phiOnStencilDir, partInfoDirIt, \
                                   currStateFullPodDir, sampleMeshDirIt, \
                                   modesPerTileDic, module.numDofsPerCell)

          # -------
          # loop 5: over all test values
          # ------
          run_hr_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                 scenario, partInfoDirIt, \
                                                 fomMeshPath, sampleMeshDirIt, \
                                                 currStateFullPodDir, projectorDir,
                                                 phiOnStencilDir, \
                                                 energyValue, modesPerTileDic, \
                                                 setId, smKeyword, "quad")


# -------------------------------------------------------------------
def run_od_pod_galerkin_gappy(workDir, problem, module, \
                              scenario, fomMeshPath):

  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in find_all_partitions_info_dirs(workDir):
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)
    nTiles = np.loadtxt(partInfoDirIt+"/ntiles.txt", dtype=int)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    howManySets = len(module.basis_sets[scenario].keys())
    for setId in range(howManySets):
      currStateFullPodDir = path_to_state_pod_data_dir(workDir, partitionStringIdentifier, setId)
      currRhsFullPodDir   = path_to_rhs_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3: over all samples meshes
      # ------
      for sampleMeshDirIt in find_all_sample_meshes_for_target_partition_info(workDir, partInfoDirIt):
        smKeyword = string_identifier_from_sample_mesh_dir(sampleMeshDirIt)

        # -------
        # loop 4
        # ------
        for modeSettingIt_key, modeSettingIt_val in module.odrom_modes_setting_policies[scenario].items():

          if modeSettingIt_key == 'allTilesUseTheSameUserDefinedValue':
            for numModes in modeSettingIt_val:
              modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, numModes)
              # all tiles have same modes, so find what that is
              numOfModes = modesPerTileDic[0]

              projectorDir = path_to_gappy_projector_dir(workDir, "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         None, numOfModes, smKeyword)
              print(projectorDir)
              if os.path.exists(projectorDir):
                print('{} already exists'.format(projectorDir))
              else:
                print('Generating {}'.format(projectorDir))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStateFullPodDir, currRhsFullPodDir,\
                                                                        sampleMeshDirIt, modesPerTileDic,
                                                                        module.numDofsPerCell)

              # compute phi on stencil for each tile if needed
              phiOnStencilDir = path_to_phi_on_stencil_dir(workDir, partitionStringIdentifier, \
                                                           setId, modeSettingIt_key,\
                                                           None, numOfModes, smKeyword)

              print(phiOnStencilDir)
              if os.path.exists(phiOnStencilDir):
                print('{} already exists'.format(phiOnStencilDir))
              else:
                print('Generating {}'.format(phiOnStencilDir))
                os.system('mkdir -p ' + phiOnStencilDir)
                compute_phi_on_stencil(phiOnStencilDir, partInfoDirIt, \
                                       currStateFullPodDir, sampleMeshDirIt, \
                                       modesPerTileDic, module.numDofsPerCell)

              run_hr_od_galerkin_for_all_test_values(workDir, problem, module,
                                                     scenario, partInfoDirIt, \
                                                     fomMeshPath, sampleMeshDirIt, \
                                                     currStateFullPodDir, projectorDir,
                                                     phiOnStencilDir, modeSettingIt_key, \
                                                     None, modesPerTileDic, \
                                                     setId, smKeyword, "gappy")

          elif modeSettingIt_key == 'findMinValueAcrossTilesUsingEnergyAndUseInAllTiles':
            for energyValue in modeSettingIt_val:
              modesPerTileDicTmp = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                          currStateFullPodDir, energyValue)
              # find minimum value
              minMumModes = np.min(list(modesPerTileDicTmp.values()))
              modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, minMumModes)

              projectorDir = path_to_gappy_projector_dir(workDir, "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         energyValue, None, smKeyword)
              print(projectorDir)
              if os.path.exists(projectorDir):
                print('{} already exists'.format(projectorDir))
              else:
                print('Generating {}'.format(projectorDir))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStateFullPodDir, currRhsFullPodDir,\
                                                                        sampleMeshDirIt, modesPerTileDic,
                                                                        module.numDofsPerCell)

              # compute phi on stencil for each tile if needed
              phiOnStencilDir = path_to_phi_on_stencil_dir(workDir, partitionStringIdentifier, \
                                                           setId, modeSettingIt_key,\
                                                           energyValue, None, smKeyword)

              print(phiOnStencilDir)
              if os.path.exists(phiOnStencilDir):
                print('{} already exists'.format(phiOnStencilDir))
              else:
                print('Generating {}'.format(phiOnStencilDir))
                os.system('mkdir -p ' + phiOnStencilDir)
                compute_phi_on_stencil(phiOnStencilDir, partInfoDirIt, \
                                       currStateFullPodDir, sampleMeshDirIt, \
                                       modesPerTileDic, module.numDofsPerCell)

              run_hr_od_galerkin_for_all_test_values(workDir, problem, module,
                                                     scenario, partInfoDirIt, \
                                                     fomMeshPath, sampleMeshDirIt, \
                                                     currStateFullPodDir, projectorDir,
                                                     phiOnStencilDir, modeSettingIt_key, \
                                                     energyValue, modesPerTileDic, \
                                                     setId, smKeyword, "gappy")

          elif modeSettingIt_key == 'tileSpecificUsingEnergy':
            for energyValue in modeSettingIt_val:
              modesPerTileDic = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                       currStateFullPodDir, energyValue)
              projectorDir = path_to_gappy_projector_dir(workDir, "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         energyValue, None, smKeyword)
              print(projectorDir)
              if os.path.exists(projectorDir):
                print('{} already exists'.format(projectorDir))
              else:
                print('Generating {}'.format(projectorDir))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStateFullPodDir, currRhsFullPodDir,\
                                                                        sampleMeshDirIt, modesPerTileDic,
                                                                        module.numDofsPerCell)

              # compute phi on stencil for each tile if needed
              phiOnStencilDir = path_to_phi_on_stencil_dir(workDir, partitionStringIdentifier, \
                                                           setId, modeSettingIt_key,\
                                                           energyValue, None, smKeyword)

              print(phiOnStencilDir)
              if os.path.exists(phiOnStencilDir):
                print('{} already exists'.format(phiOnStencilDir))
              else:
                print('Generating {}'.format(phiOnStencilDir))
                os.system('mkdir -p ' + phiOnStencilDir)
                compute_phi_on_stencil(phiOnStencilDir, partInfoDirIt, \
                                       currStateFullPodDir, sampleMeshDirIt, \
                                       modesPerTileDic, module.numDofsPerCell)

              run_hr_od_galerkin_for_all_test_values(workDir, problem, module,
                                                     scenario, partInfoDirIt, \
                                                     fomMeshPath, sampleMeshDirIt, \
                                                     currStateFullPodDir, projectorDir,
                                                     phiOnStencilDir, modeSettingIt_key, \
                                                     energyValue, modesPerTileDic, \
                                                     setId, smKeyword, "gappy")

  sys.exit()


# -------------------------------------------------------------------
def compute_od_pod_projection_errors(workDir, problem, module, \
                                     scenario, fomMeshPath):

  fomTestDirs = find_all_fom_test_dirs(workDir)

  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in find_all_partitions_info_dirs(workDir):
    nTiles = np.loadtxt(partInfoDirIt+"/ntiles.txt", dtype=int)
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    howManySets = len(module.basis_sets[scenario].keys())
    for setId in range(howManySets):
      currPodDir = path_to_state_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3
      # ------
      for modeSettingIt_key, modeSettingIt_val \
          in module.odrom_modes_setting_policies[scenario].items():

        if modeSettingIt_key == 'allTilesUseTheSameUserDefinedValue':
          for numModes in modeSettingIt_val:
            modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, numModes)

            # assemble the name of output directory
            outDirRoot = workDir + "/od_proj_error_"+partitionStringIdentifier
            outDirRoot += "_using_pod_bases"
            outDirRoot += "_modesSettingPolicy_"+modeSettingIt_key
            # all tiles use same value so pick first
            outDirRoot += "_"+str(numModes)
            outDirRoot += "_set_"+str(setId)

            # loop over each FOM test dir and compute proj error
            for fomTestDirIt in find_all_fom_test_dirs(workDir):
              outDir = outDirRoot + "_" + str(get_run_id(fomTestDirIt))

              # check outdir, make and run if needed
              if os.path.exists(outDir):
                print('{} already exists'.format(outDir))
              else:
                print("Running od proj errro in {}".format(os.path.basename(outDir)))
                os.system('mkdir -p ' + outDir)

                np.savetxt(outDir+"/modes_per_tile.txt", \
                           np.array(list(modesPerTileDic.values())),
                           fmt="%5d")

                args = ("python3", 'od_proj_error.py', \
                        "--wdir", outDir, "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, "--infodir", partInfoDirIt,\
                        "--userefstate",  str(module.use_ic_reference_state[scenario]))
                popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read();


        elif modeSettingIt_key == 'tileSpecificUsingEnergy':
          for energyValue in modeSettingIt_val:
            modesPerTileDic = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                     currPodDir, energyValue)

            # assemble the name of output directory
            outDirRoot = workDir + "/od_proj_error_"+partitionStringIdentifier
            outDirRoot += "_using_pod_bases"
            outDirRoot += "_modesSettingPolicy_"+modeSettingIt_key
            outDirRoot += "_"+str(energyValue)
            outDirRoot += "_set_"+str(setId)

            # loop over each FOM test dir and compute proj error
            for fomTestDirIt in find_all_fom_test_dirs(workDir):
              outDir = outDirRoot + "_" + str(get_run_id(fomTestDirIt))

              # check outdir, make and run if needed
              if os.path.exists(outDir):
                print('{} already exists'.format(outDir))
              else:
                print("Running od proj errro in {}".format(os.path.basename(outDir)))
                os.system('mkdir -p ' + outDir)

                np.savetxt(outDir+"/modes_per_tile.txt", \
                           np.array(list(modesPerTileDic.values())),
                           fmt="%5d")

                args = ("python3", 'od_proj_error.py', \
                        "--wdir", outDir, "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, "--infodir", partInfoDirIt,\
                        "--userefstate",  str(module.use_ic_reference_state[scenario]))
                popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read();


        elif modeSettingIt_key == 'findMinValueAcrossTilesUsingEnergyAndUseInAllTiles':
          for energyValue in modeSettingIt_val:
            modesPerTileDicTmp = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                        currPodDir, energyValue)
            # find minimum value
            minMumModes = np.min(list(modesPerTileDicTmp.values()))
            modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, minMumModes)

            # assemble the name of output directory
            outDirRoot = workDir + "/od_proj_error_"+partitionStringIdentifier
            outDirRoot += "_using_pod_bases"
            outDirRoot += "_modesSettingPolicy_"+modeSettingIt_key
            outDirRoot += "_"+str(energyValue)
            outDirRoot += "_set_"+str(setId)

            # loop over each FOM test dir and compute proj error
            for fomTestDirIt in find_all_fom_test_dirs(workDir):
              outDir = outDirRoot + "_" + str(get_run_id(fomTestDirIt))

              # check outdir, make and run if needed
              if os.path.exists(outDir):
                print('{} already exists'.format(outDir))
              else:
                print("Running od proj errro in {}".format(os.path.basename(outDir)))
                os.system('mkdir -p ' + outDir)

                np.savetxt(outDir+"/modes_per_tile.txt", \
                           np.array(list(modesPerTileDic.values())),
                           fmt="%5d")

                args = ("python3", 'od_proj_error.py', \
                        "--wdir", outDir, "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, "--infodir", partInfoDirIt,\
                        "--userefstate",  str(module.use_ic_reference_state[scenario]))
                popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read();


        else:
          sys.exit('compute_od_pod_projection_errors: invalid modeSettingPolicy = {}'.format(modeSettingIt_key))



# -------------------------------------------------------------------
def run_od_poly_galerkin_same_order_in_each_tile(workDir, problem, \
                                                 module, scenario, \
                                                 fomMeshPath, polyOrders):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in partInfoDirs:
    nTiles  = np.loadtxt(partInfoDirIt+"/ntiles.txt", dtype=int)
    #nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    #nTiles = nTilesX*nTilesY
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
        compute_poly_bases_same_order_all_tiles(module.dimensionality, \
                                                fomMeshPath, polyBasesDir, \
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
                                               odRomMeshObj, setId, \
                                               "using_poly_bases")

# -------------------------------------------------------------------
def run_od_poly_galerkin_with_order_matching_pod_count(workDir, problem, \
                                                       module, scenario, \
                                                       fomMeshPath):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in partInfoDirs:
    nTiles = np.loadtxt(partInfoDirIt+"/ntiles.txt", dtype=int)
    #nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    #nTiles = nTilesX*nTilesY
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
        modesPerTilePod = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                 currPodDir, energyValue)

        # now that we know POD modes per tile, we create local poly bases
        # such that the order yields num of modes that matches POD bases
        polyBasesDir = path_to_poly_bases_data_dir(workDir, partitionStringIdentifier, \
                                                   -1, energyValue, setId)
        if not os.path.exists(polyBasesDir):
          os.system('mkdir -p ' + polyBasesDir)
          compute_poly_bases_to_match_pod(module.dimensionality, \
                                          fomMeshPath, polyBasesDir, \
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
                                                 odRomMeshObj, setId, \
                                                 "using_poly_bases")





# -------------------------------------------------------------------
def compute_sample_mesh_random_full_domain(workDir, module, scenario, pdaDir, fomMeshPath):
  # get list of RANDOM sample mesh cases from module
  sampleMeshesList = [it for it in module.sample_meshes[scenario]\
                      if "random" in it]
  print(sampleMeshesList)

  for sampleMeshCaseIt in sampleMeshesList:
    fractionOfCellsNeeded = sampleMeshCaseIt[1]
    # create name of directory where to store the sample mesh
    outDir = path_to_full_domain_sample_mesh_random(workDir, fractionOfCellsNeeded)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print('Generating RANDOM sample mesh in {}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      fomNumCells = find_total_cells_from_info_file(fomMeshPath)
      sampleMeshCount = int(fomNumCells * fractionOfCellsNeeded)
      sample_mesh_gids = random.sample(range(0, fomNumCells), sampleMeshCount)
      sample_mesh_gids = np.sort(sample_mesh_gids)
      print(" numCellsFullDomain = ", fomNumCells)
      print(" sampleMeshSize     = ", sampleMeshCount)
      np.savetxt(outDir+'/sample_mesh_gids_p_0.txt', sample_mesh_gids, fmt='%8i')

      print('Generating sample mesh in:')
      print(' {}'.format(outDir))
      owd = os.getcwd()
      meshScriptsDir = pdaDir + "/meshing_scripts"
      args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
              "--fullMeshDir", fomMeshPath,
              "--sampleMeshIndices", outDir+'/sample_mesh_gids_p_0.txt',
              "--outDir", outDir)
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      print(output)


# -------------------------------------------------------------------
def compute_sample_mesh_psampling_full_domain(workDir, module, scenario, pdaDir, fomMeshPath):
  # get list of sample mesh cases, filter only those having "psampling" in it
  sampleMeshesList = [it for it in module.sample_meshes[scenario]\
                      if "psampling" in it]
  print(sampleMeshesList)

  # -------
  # loop 2: over all setIds
  # ------
  howManySets = len(module.basis_sets[scenario].keys())
  for setId in range(howManySets):
    # for psampling I need to get the rhs modes
    currRhsPodDir = path_to_full_domain_rhs_pod_data_dir(workDir, setId)

    # -------
    # loop 3: over all target sample mesh cases
    # ------
    for sampleMeshCaseIt in sampleMeshesList:
      # extract the fraction, which must be in position 1
      assert(not isinstance(sampleMeshCaseIt[1], str))
      fractionOfCellsNeeded = sampleMeshCaseIt[1]

      # for psampling sample mesh, I need to use a certain dof/variable
      # for exmaple for swe, there is h,u,v so I need to know which one
      # to use to find the cells
      # this info is provided in the problem
      assert(isinstance(sampleMeshCaseIt[2], int))
      whichDofToUseForFindingCells = sampleMeshCaseIt[2]

      # name of where to generate files
      outDir = path_to_full_domain_sample_mesh_psampling(workDir, setId, fractionOfCellsNeeded)
      if os.path.exists(outDir):
        print('{} already exists'.format(outDir))
      else:
        print('Generating psampling sample mesh in: \n {}'.format(outDir))
        os.system('mkdir -p ' + outDir)

        fomNumCells = find_total_cells_from_info_file(fomMeshPath)
        mySampleMeshCount = int(fomNumCells * fractionOfCellsNeeded)
        rhsPodFile = currRhsPodDir + "/lsv_rhs_p_0"
        myRhsPod = load_basis_from_binary_file(rhsPodFile)[whichDofToUseForFindingCells::module.numDofsPerCell]
        if myRhsPod.shape[1] < mySampleMeshCount:
          print("Warning: psampling sample mesh for full domain")
          print("         not enough rhs modes, automatically reducing sample mesh count")
          mySampleMeshCount = myRhsPod.shape[1]-1

        Q,R,P = scipyla.qr(myRhsPod[:,0:mySampleMeshCount].transpose(), pivoting=True)
        mySampleMeshGidsWrtFullMesh = np.sort(P[0:mySampleMeshCount])
        np.savetxt(outDir+'/sample_mesh_gids_p_0.txt',\
                   mySampleMeshGidsWrtFullMesh, fmt='%8i')

        print('Generating sample mesh in:')
        print(' {}'.format(outDir))
        owd = os.getcwd()
        meshScriptsDir = pdaDir + "/meshing_scripts"
        args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
                "--fullMeshDir", fomMeshPath,
                "--sampleMeshIndices", outDir+'/sample_mesh_gids_p_0.txt',
                "--outDir", outDir)
        popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
        popen.wait()
        output = popen.stdout.read();
        print(output)


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
  # meshSize is optional because one could directly
  # specify it inside base_dic of the target problem
  parser.add_argument("--mesh",
                      nargs='+', \
                      dest="mesh", \
                      type=int, \
                      required=False)

  args     = parser.parse_args()
  workDir  = args.workdir
  pdaDir   = args.pdadir
  problem  = args.problem
  scenario = args.scenario

  if not os.path.exists(workDir):
    print("Working dir {} does not exist, creating it".format(workDir))
    os.system('mkdir -p ' + workDir)
    print("")

  # write scenario id, problem to file
  write_scenario_to_file(scenario, workDir)
  write_problem_name_to_file(problem, workDir)

  # -------------------------------------------------------------
  # -------------------------------------------------------------

  stage = "Partition domain"
  print("=================================================")
  print(stage)
  print("=================================================")
  triggers = ["PodOdGalerkinFull", "PodOdGalerkinGappy", "PodOdGalerkinQuad",\
              "PodOdGalerkinMasked", "LegendreOdGalerkinFull"]
  if any(x in triggers for x in module.algos[scenario]):
    for key, val in module.odrom_partitions[scenario].items():
      style, listOfTilings = key, val

      if module.dimensionality == 1:
        assert(style == "rectangularUniform")
        make_rectangular_uniform_partitions_1d(workDir, fomMeshPath, listOfTilings)

      if module.dimensionality == 2 and \
         style == "rectangularUniform":
        make_rectangular_uniform_partitions_2d(workDir, fomMeshPath, listOfTilings)

      if module.dimensionality == 2 and \
         style == "concentricUniform":
        make_concentric_uniform_partitions_2d(workDir, fomMeshPath, listOfTilings)
  else:
    print("skipping: " + stage)
  print("")

  # -------------------------------------------------------------
  # -------------------------------------------------------------

  print("=================================================")
  print("Running pod od projection error ")
  print("=================================================")
  if "PodOdProjectionError" in module.algos[scenario]:
    compute_od_pod_projection_errors(workDir, problem, module, scenario, fomMeshPath)
  print("")

  print("=================================================")
  print("Running FULL pod od-galerkin ")
  print("=================================================")
  if "PodOdGalerkinFull" in module.algos[scenario]:
    run_od_pod_galerkin_full(workDir, problem, module, scenario, fomMeshPath)
  print("")

  # print("=================================================")
  # print("Running Gappy pod od-galerkin ")
  # print("=================================================")
  # if "PodOdGalerkinGappy" in module.algos[scenario]:
  #   run_od_pod_galerkin_gappy(workDir, problem, module, scenario, fomMeshPath)
  # print("")

  sys.exit()

  # print("=================================================")
  # print("Running Masked Gappy pod od-galerkin ")
  # print("=================================================")
  # if "PodOdGalerkinMasked" in module.algos[scenario]:
  #   run_od_pod_masked_galerkin_gappy(workDir, problem, module, scenario, fomMeshPath)
  # print("")

  # print("=================================================")
  # print("Running Quad pod od-galerkin ")
  # print("=================================================")
  # if "PodOdGalerkinQuad" in module.algos[scenario]:
  #   run_od_pod_galerkin_quad(workDir, problem, module, scenario, fomMeshPath)
  # print("")


  '''
  anything below here must be revisited
  # poly odgalerkin worked before but need to be revised
  '''
  # print("=================================================")
  # print("Running FULL poly od-galerkin ")
  # print("=================================================")
  # if "PolyGalerkinFull" in module.odrom_algos[scenario]:
  #   # check if -1 is is part of the order list, and if so that means
  #   # we need to use polyn bases such that in each tile we decide the order
  #   # based on the number of POD modes to have a fair comparison
  #   if -1 in module.odrom_poly_order[scenario]:
  #     run_od_poly_galerkin_with_order_matching_pod_count(workDir, problem, \
  #                                                        module, scenario, \
  #                                                        fomMeshPath)

  #   # run the case involving each partition of same poly bases,
  #   # so we need to list all the orders we want that are != -1
  #   polyOrders = [i for i in module.odrom_poly_order[scenario] if i > 0]
  #   if polyOrders:
  #     run_od_poly_galerkin_same_order_in_each_tile(workDir, problem, module, \
  #                                                  scenario, fomMeshPath, polyOrders)
