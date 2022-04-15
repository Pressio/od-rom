
import numpy as np
import sys, os, time, logging

try:
  import pressiodemoapps as pda
except ImportError:
  raise ImportError("Unable to import pressiodemoapps")

from .fncs_myio import \
  write_dic_to_yaml_file

from .fncs_make_od_rom_initial_condition import *

from .fncs_miscellanea import \
  get_run_id, \
  find_all_partitions_info_dirs, \
  compute_total_modes_across_all_tiles

from .fncs_directory_naming import \
  string_identifier_from_partition_info_dir

from .class_odrom_gappy import *
from .class_odrom_gappy_masked import *
from .class_observer_rom import RomObserver
from .fncs_time_integration import *

# -------------------------------------------------------------------
def run_hr_od_galerkin_for_all_test_values(workDir, problem, \
                                           module, scenario, partInfoDir, \
                                           fomMeshPath, odromSampleMeshPath, \
                                           fullPodDir, projectorDir, phiOnStencilDir,\
                                           modeSettingPolicy, \
                                           energyValue, modesPerTileDic, \
                                           setId, smKeywordForDirName,
                                           algoNameForDirName):
  logger = logging.getLogger(__name__)

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
      logger.info('{} already exists'.format(os.path.basename(outDir)))
    else:
      logger.info("Running odrom in {}".format(os.path.basename(outDir)))
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
      logger.info(fomFullMeshTotalDofs)
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
      obsO = RomObserver(stateSamplingFreq, numSteps, modesPerTileDic, dt)

      inputFile = outDir + "/input.yaml"
      write_dic_to_yaml_file(inputFile, romRunDic)

      pTimeStart = time.time()
      if odeScheme in ["SSPRK3", "ssprk3"]:
        odrom_ssprk3(odRomObj, romState, numSteps, dt, obsO)

      elif odeScheme in ["RungeKutta4", "RK4", "rk4"]:
        odrom_rk4(odRomObj, romState, numSteps, dt, obsO)

      elif odeScheme in ["RungeKutta2", "RK2", "rk2"]:
        odrom_rk2(odRomObj, romState, numSteps, dt, obsO)

      # because of how RomObserver and the time integration are done,
      # we need to call it one time at the time to ensure
      # we observe/store the final rom state
      obsO(numSteps, romState)

      elapsed = time.time() - pTimeStart
      logger.info("elapsed = {}".format(elapsed))
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
                                                     modeSettingPolicy, \
                                                     energyValue, \
                                                     modesPerTileDic, \
                                                     setId, smKeywordForDirName):
  logger = logging.getLogger(__name__)

  # store various things
  romSizeOverAllPartitions = compute_total_modes_across_all_tiles(modesPerTileDic)
  meshObj = pda.load_cellcentered_uniform_mesh(fomMeshPath)
  partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDir)

  # loop over all test param values to do
  param_values = module.test_points[scenario]
  for k, val in param_values.items():

    # figure out the name of the output directory
    outDir = workDir + "/odrom_masked_gappy_"+partitionStringIdentifier
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
      logger.info('{} already exists'.format(os.path.basename(outDir)))
    else:
      logger.info("Running odrom in {}".format(os.path.basename(outDir)))
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
      romRunDic['fullPodDir'] = podDir
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
      logger.info(fomFullMeshTotalDofs)
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
      obsO = RomObserver(stateSamplingFreq, numSteps, modesPerTileDic, dt)

      inputFile = outDir + "/input.yaml"
      write_dic_to_yaml_file(inputFile, romRunDic)

      pTimeStart = time.time()
      if odeScheme in ["SSPRK3", "ssprk3"]:
        odrom_ssprk3(odRomObj, romState, numSteps, dt, obsO)
      elif odeScheme in ["RungeKutta4", "RK4", "rk4"]:
        odrom_rk4(odRomObj, romState, numSteps, dt, obsO)
      elif odeScheme in ["RungeKutta2", "RK2", "rk2"]:
        odrom_rk2(odRomObj, romState, numSteps, dt, obsO)

      # because of how RomObserver and the time integration are done,
      # we need to call it one time at the time to ensure
      # we observe/store the final rom state
      obsO(numSteps, romState)

      elapsed = time.time() - pTimeStart
      logger.info("elapsed = {}".format(elapsed))
      f = open(outDir+"/timing.txt", "w")
      f.write(str(elapsed))
      f.close()

      # tell observer to write snapshots to file
      obsO.write(outDir)
      # reconstruct final state
      odRomObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_final.txt", odRomObj.viewFomStateOnFullMesh())
