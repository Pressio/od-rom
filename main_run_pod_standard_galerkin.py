
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import re, time, yaml, random, subprocess
import numpy as np
from scipy import linalg as scipyla
from decimal import Decimal
from scipy import optimize as sciop

try:
  import pressiodemoapps as pda
except ImportError:
  raise ImportError("Unable to import pressiodemoapps")

from py_src.banners_and_prints import *

from py_src.miscellanea import \
  find_full_mesh_and_ensure_unique

from py_src.myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir, \
  load_basis_from_binary_file, \
  write_dic_to_yaml_file

from py_src.directory_naming import \
  path_to_full_domain_state_pod_data_dir

from py_src.mesh_info_file_extractors import *

from py_src.standardrom_full import *
from py_src.observer import RomObserver
from py_src.odrom_time_integrators import *

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

#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  parser   = ArgumentParser()
  parser.add_argument("--wdir", dest="workdir", required=True)
  args     = parser.parse_args()
  workDir  = args.workdir

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
  banner_pod_standard_galerkin()
  # --------------------------------------
  if "PodStandardGalerkinFull" in module.algos[scenario]:
    run_standard_pod_galerkin_full(workDir, problem, module, scenario, fomMeshPath)
  else:
    print("skipping: " + stage)
  print("")
