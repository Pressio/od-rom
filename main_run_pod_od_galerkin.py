
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, subprocess
import numpy as np
from scipy import linalg as scipyla

try:
  import pressiodemoapps as pda
except ImportError:
  raise ImportError("Unable to import pressiodemoapps")

from py_src.banners_and_prints import *

from py_src.miscellanea import \
  find_full_mesh_and_ensure_unique,\
  get_run_id, \
  find_all_partitions_info_dirs, \
  make_modes_per_tile_dic_with_const_modes_count,\
  compute_total_modes_across_all_tiles

from py_src.myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir,\
  load_fom_state_snapshot_matrix, \
  load_fom_rhs_snapshot_matrix, \
  load_basis_from_binary_file, \
  write_dic_to_yaml_file

from py_src.directory_naming import \
  path_to_partition_info_dir, \
  path_to_state_pod_data_dir, \
  string_identifier_from_partition_info_dir, \
  path_to_partition_based_full_mesh_dir

from py_src.mesh_info_file_extractors import *

from py_src.make_od_rom_initial_condition import *
from py_src.odrom_full import *
from py_src.odrom_time_integrators import *
from py_src.observer import RomObserver

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
def run_full_od_galerkin_for_all_test_values(workDir, problem, \
                                             module, scenario, \
                                             fomMeshPath, partInfoDir, \
                                             romMeshObj, setId, \
                                             basesDir, basesKind, \
                                             modeSettingPolicy, \
                                             modesPerTileDic, \
                                             energyValue, polyOrder):

  # this is odrom WITHOUT HR, so the following should hold:
  stencilDofsCount = romMeshObj.stencilMeshSize()*module.numDofsPerCell
  sampleDofsCount  = romMeshObj.sampleMeshSize()*module.numDofsPerCell
  assert(stencilDofsCount == sampleDofsCount)
  fomTotalDofs = stencilDofsCount

  # store various things
  romSizeOverAllPartitions = compute_total_modes_across_all_tiles(modesPerTileDic)
  fomMeshObj = pda.load_cellcentered_uniform_mesh(fomMeshPath)
  partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDir)

  # loop over all test param values to do
  param_values = module.test_points[scenario]
  for k, val in param_values.items():

    # figure out the name of the output directory
    outDir = workDir + "/odrom_full_"+partitionStringIdentifier
    outDir += "_" + basesKind
    outDir += "_modesSettingPolicy_"+modeSettingPolicy

    if 'Energy' in modeSettingPolicy:
      outDir += "_"+str(energyValue)
    elif modeSettingPolicy == 'allTilesUseTheSameUserDefinedValue':
      # all tiles use same value so pick first
      outDir += "_"+str(modesPerTileDic[0])

    if polyOrder != None:
      outDir += "_order_"+str(polyOrder)

    if setId != None:
      outDir += "_set_"+str(setId)
    outDir += "_"+str(k)

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

      if basesKind == "using_pod_bases":
        romRunDic['energy'] = energyValue
      if basesKind == "using_poly_bases":
        romRunDic['polyOrder'] = polyOrder

      romRunDic['basesDir'] = basesDir
      romRunDic['partioningInfo'] = partInfoDir
      romRunDic['numDofsPerCell'] = module.numDofsPerCell

      # make ROM initial state
      usingIcAsRefState = module.use_ic_reference_state[scenario]
      romRunDic['usingIcAsRefState'] = usingIcAsRefState
      romState = make_od_rom_initial_condition(workDir, appObjForIc, \
                                               partInfoDir, basesDir, \
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
      odRomObj = OdRomFull(appObjForRom, module.dimensionality, \
                           module.numDofsPerCell, partInfoDir, \
                           modesPerTileDic, basesDir, \
                           refStateForFullMeshOrdering, \
                           refStateForOdRomAlgo)
      # initial condition
      odRomObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_ic.txt", odRomObj.viewFomState())

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

      # write yaml to file
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
      np.savetxt(outDir+"/y_rec_final.txt", odRomObj.viewFomState())
      print("")


# -------------------------------------------------------------------
def run_od_pod_galerkin_full(workDir, problem, module, \
                             scenario, fomMeshPath):

  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in find_all_partitions_info_dirs(workDir):
    nTiles = np.loadtxt(partInfoDirIt+"/ntiles.txt", dtype=int)
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # for each decomposition, find the corresponding full mesh with
    # the indexing suitable for ODROM. Each decomposition should
    # have a unique full mesh associated with it.
    #topoString = str(nTilesX)+"x"+str(nTilesY)
    odMeshDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if partitionStringIdentifier in d and "full_mesh" in d]
    assert(len(odMeshDirs)==1)
    odRomMeshDir = odMeshDirs[0]
    # the mesh directory just found becomes the one to use for the odrom.
    # we can make a mesh object right here and use for all the runs below
    odRomMeshObj = pda.load_cellcentered_uniform_mesh(odRomMeshDir)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    howManySets = len(module.basis_sets[scenario].keys())
    for setId in range(howManySets):
      currPodDir = path_to_state_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3
      # ------
      for modeSettingIt_key, modeSettingIt_val in module.odrom_modes_setting_policies[scenario].items():

        if modeSettingIt_key == 'allTilesUseTheSameUserDefinedValue':
          for numModes in modeSettingIt_val:
            modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, numModes)

            run_full_od_galerkin_for_all_test_values(workDir, problem, \
                                                     module, scenario, \
                                                     fomMeshPath, partInfoDirIt, \
                                                     odRomMeshObj, setId, \
                                                     currPodDir, "using_pod_bases",\
                                                     modeSettingIt_key, \
                                                     modesPerTileDic, \
                                                     None, None)

        elif modeSettingIt_key == 'tileSpecificUsingEnergy':
          for energyValue in modeSettingIt_val:
            modesPerTileDic = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                     currPodDir, energyValue)
            run_full_od_galerkin_for_all_test_values(workDir, problem, \
                                                     module, scenario, \
                                                     fomMeshPath, partInfoDirIt, \
                                                     odRomMeshObj, setId, \
                                                     currPodDir, "using_pod_bases",\
                                                     modeSettingIt_key, \
                                                     modesPerTileDic, \
                                                     energyValue, None)

        elif modeSettingIt_key == 'findMinValueAcrossTilesUsingEnergyAndUseInAllTiles':
          for energyValue in modeSettingIt_val:
            modesPerTileDicTmp = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                        currPodDir, energyValue)
            # find minimum value
            minMumModes = np.min(list(modesPerTileDicTmp.values()))
            modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, minMumModes)

            run_full_od_galerkin_for_all_test_values(workDir, problem, \
                                                     module, scenario, \
                                                     fomMeshPath, partInfoDirIt, \
                                                     odRomMeshObj, setId, \
                                                     currPodDir, "using_pod_bases",\
                                                     modeSettingIt_key, \
                                                     modesPerTileDic, \
                                                     energyValue, None)

        else:
          sys.exit('run_od_pod_galerkin_full: invalid modeSettingPolicy = {}'.format(modeSettingIt_key))



#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  parser   = ArgumentParser()
  parser.add_argument("--wdir", dest="workdir", required=True)
  parser.add_argument("--pdadir", dest="pdadir", required=True)
  args     = parser.parse_args()
  workDir  = args.workdir
  pdaDir   = args.pdadir

  # make sure the workdir exists
  if not os.path.exists(workDir):
    sys.exit("Working dir {} does not exist, terminating".format(workDir))

  # --------------------------------------
  banner_import_problem()
  # --------------------------------------
  scenario = read_scenario_from_dir(workDir)
  problem  = read_problem_name_from_dir(workDir)
  module   = importlib.import_module(problem)
  check_and_print_problem_summary(problem, module)
  print("")

  # before we move on, we need to ensure that in workDir
  # there is a unique FULL mesh. This is because the mesh is specified
  # via command line argument and must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory
  fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

  # --------------------------------------
  banner_make_full_meshes_with_partition_based_indexing()
  # --------------------------------------
  # needed if we are doing PodOdGalerkinFull or LegendreOdGalerkinFull
  # indeed, for od-rom without HR, for performance reasons,
  # we don't/should not use the same full mesh used in the fom.
  # We need to make a new full mesh with a new indexing
  # that is consistent with the partitions and allows continguous storage
  # of the state and rhs within each tile
  if "PodOdGalerkinFull"   in module.algos[scenario] or \
     "PolyOdGalerkinFully" in module.algos[scenario]:
    make_full_mesh_for_odrom_using_partition_based_indexing(workDir, \
                                                            pdaDir, \
                                                            module, \
                                                            fomMeshPath)
  else:
    print("skipping: " + stage)
  print("")

  # --------------------------------------
  banner_run_pod_od_galerkin()
  # --------------------------------------
  if "PodOdGalerkinFull" in module.algos[scenario]:
    run_od_pod_galerkin_full(workDir, problem, module, scenario, fomMeshPath)
  print("")
