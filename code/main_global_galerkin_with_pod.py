
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import re, time, yaml, random, subprocess, logging
import numpy as np
from scipy import linalg as scipyla
from decimal import Decimal
from scipy import optimize as sciop

try:
  import pressiodemoapps as pda
except ImportError:
  raise ImportError("Unable to import pressiodemoapps")

from py_src.fncs_banners_and_prints import *

from py_src.fncs_miscellanea import \
  find_full_mesh_and_ensure_unique

from py_src.fncs_cumulative_energy import \
  compute_cumulative_energy

from py_src.fncs_myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir, \
  load_basis_from_binary_file, \
  write_dic_to_yaml_file

from py_src.fncs_directory_naming import \
  path_to_full_domain_state_pod_data_dir

from py_src.fncs_to_extract_from_mesh_info_file import *

from py_src.class_standardrom_full import *
from py_src.class_observer_rom import RomObserver
from py_src.fncs_time_integration import *

# -------------------------------------------------------------------
def find_modes_for_full_domain_from_target_energy(module, scenario, podDir, energy):
  singValues = np.loadtxt(podDir+'/sva_state_p_0')
  return compute_cumulative_energy(singValues, energy)


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
      logging.info('{} already exists'.format(outDir))
    else:
      logging.info("Running standard rom in {}".format(os.path.basename(outDir)))
      os.system('mkdir -p ' + outDir)
      romRunDic = module.base_dic[scenario]['rom'].copy()
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
      obsO = RomObserver(stateSamplingFreq, numSteps, {0: numModes}, dt)

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

      # because of how RomObserver and the time integration are done,
      # we need to call it one time at the time to ensure
      # we observe/store the final rom state
      obsO(numSteps, romState)

      elapsed = time.time() - pTimeStart
      logging.info("elapsed = {}".format(elapsed))
      f = open(outDir+"/timing.txt", "w")
      f.write(str(elapsed))
      f.close()

      # tell observer to write snapshots to file
      obsO.write(outDir)
      # reconstruct final state
      romObj.reconstructFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_final.txt", romObj.viewFomState())
  logging.info("")

# -------------------------------------------------------------------
def run_standard_pod_galerkin_full(workDir, problem, module, \
                                   scenario, fomMeshPath):

  # -------
  # loop: over all sets of train runs
  # ------
  howManySets = len(module.basis_sets[scenario].keys())
  for setId in range(howManySets):
    currPodDir = path_to_full_domain_state_pod_data_dir(workDir, setId)

    # -------
    # loop: over all mode setting policies
    # ------
    if not hasattr(module, 'standardrom_modes_setting_policies'):
      logging.error("for standard galerkin, you need standardrom_modes_setting_policies in the problem")
      sys.exit(1)
    if scenario not in module.standardrom_modes_setting_policies:
      logging.error("scenario = {} not valid key in module.standardrom_modes_setting_policies".format(scenario))
      sys.exit(1)

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
        logging.error('run_standard_pod_galerkin_full: invalid modeSettingPolicy = {}'.format(modeSettingIt_key))
        sys.exit(1)

# -------------------------------------------------------------------
def setLogger():
  dateFmt = '%Y-%m-%d' # %H:%M:%S'
  # logFmt1 = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
  logFmt2 = '%(levelname)-8s: [%(name)s] %(message)s'
  logging.basicConfig(format=logFmt2, encoding='utf-8', level=logging.DEBUG)

#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  setLogger()
  banner_driving_script_info(os.path.basename(__file__))

  parser   = ArgumentParser()
  parser.add_argument("--wdir", dest="workdir", required=True)
  args     = parser.parse_args()
  workDir  = args.workdir

  # make sure the workdir exists
  if not os.path.exists(workDir):
    logging.error("Working dir {} does not exist, terminating".format(workDir))
    sys.exit(1)

  banner_import_problem()
  scenario = read_scenario_from_dir(workDir)
  problem  = read_problem_name_from_dir(workDir)
  module   = importlib.import_module(problem)
  check_and_print_problem_summary(problem, module)
  logging.info("")

  if "GlobalGalerkinWithPodBases" in module.algos[scenario]:
    fomMeshPath = find_full_mesh_and_ensure_unique(workDir)
    banner_pod_standard_galerkin()
    run_standard_pod_galerkin_full(workDir, problem, module, scenario, fomMeshPath)
  else:
    logging.info("Nothing to do here")
