
from argparse import ArgumentParser
import sys, os, importlib, time, subprocess, logging
import math
import numpy as np
from decimal import Decimal

try:
  import pressiodemoapps as pda
except ImportError:
  raise ImportError("Unable to import pressiodemoapps")

from py_src.fncs_banners_and_prints import \
  banner_driving_script_info, \
  banner_import_problem, check_and_print_problem_summary, \
  banner_make_fom_mesh, banner_fom_train, banner_fom_test
from py_src.fncs_myio import *
from py_src.class_observer_fom import FomObserver

#==============================================================
# functions
#==============================================================

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
  meshArgs += module.custom_tuple_args_for_fom_mesh_generation(scenario)

  # now, generate mesh if needed
  if os.path.exists(outDir):
    logging.info('{} already exists'.format(outDir))
  else:
    logging.info('Generating mesh {}'.format(outDir))
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
    em = "I found multiple full meshes:\n"
    for it in fomFullMeshes:
      em += "   " + it + "\n"
    em += "inside the workDir = {} \n".format(workDir)
    em += "You can only have a single FULL mesh the working directory."
    logging.error(em)
    sys.exit(1)
  return fomFullMeshes[0]

# ----------------------------------------------------------------
def run_single_fom(runDir, appObj, dic):
  odeScheme         = dic['odeScheme']
  dt                = float(dic['dt'])
  stateSamplingFreq = int(dic['stateSamplingFreq'])
  rhsSamplingFreq   = int(dic['velocitySamplingFreq'])
  finalTime         = float(dic['finalTime'])
  numSteps          = int(round(Decimal(finalTime)/Decimal(dt), 8))
  logging.debug("numSteps = {}".format(numSteps))
  dic['numSteps'] = numSteps

  # write to yaml the dic to save info for later
  inputFile = runDir + "/input.yaml"
  write_dic_to_yaml_file(inputFile, dic)

  # run
  yn = appObj.initialCondition()
  np.savetxt(runDir+'/initial_state.txt', yn)
  numDofs = len(yn)

  start = time.time()
  obsO = FomObserver(numDofs, stateSamplingFreq, rhsSamplingFreq, numSteps, dt)

  if odeScheme in ["RungeKutta4", "RK4", "rungekutta4", "rk4"]:
    pda.advanceRK4(appObj, yn, dt, numSteps, observer=obsO)
  elif odeScheme in ["RungeKutta2", "RK2", "rungekutta2", "rk2"]:
    pda.advanceRK2(appObj, yn, dt, numSteps, observer=obsO)
  elif odeScheme in ["SSPRK3", "ssprk3"]:
    pda.advanceSSP3(appObj, yn, dt, numSteps, observer=obsO)
  else:
    logging.error("run_single_fom: invalid ode scheme = {}".format(odeScheme))
    sys.exit(1)

  # because of how the advance fncs are implemented, the state yn
  # after the time integration is the state at the last step
  # but we need to do one last observation to make sure it gets stored
  # in snapshots , because of the fomObserver works.
  tmpvelo = appObj.createVelocity()
  appObj.velocity(yn, numSteps*dt, tmpvelo)
  obsO(numSteps, yn, tmpvelo)

  elapsed = time.time() - start
  logging.debug("elapsed = {}".format(elapsed))
  f = open(runDir+"/timing.txt", "w")
  f.write(str(elapsed))
  f.close()

  obsO.write(runDir)
  np.savetxt(runDir+'/final_state.txt', yn)
  from scipy import linalg
  stateNorm = linalg.norm(yn, check_finite=False)
  if math.isnan(stateNorm):
    logging.error("Fom run failed, maybe check time step?")
    sys.exit(1)

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
    # copy dic for the FOM
    fomDic = module.base_dic[scenario]['fom'].copy()

    fomDic['numDofsPerCell'] = module.numDofsPerCell
    fomDic['meshDir'] = fomMesh

    # decide sampling freq of state
    stateSamplingFreq = 0
    if testOrTrainString == "train":
      stateSamplingFreq = fomDic['stateSamplingFreqTrain']
    else:
      stateSamplingFreq = module.base_dic[scenario]['stateSamplingFreqTest']
    del fomDic['stateSamplingFreqTrain']
    fomDic['stateSamplingFreq'] = int(stateSamplingFreq)

    # the train/test simulation time might differ, ensure we pick the right one
    finalTime = fomDic['finalTimeTrain'] if testOrTrainString == "train" \
      else fomDic['finalTimeTest']
    # set the final time in dic
    del fomDic['finalTimeTrain']
    del fomDic['finalTimeTest']
    fomDic['finalTime'] = float(finalTime)

    # create problem using in-module function
    coeffDic = module.base_dic[scenario]['physicalCoefficients'].copy()
    fomObj = module.create_problem_for_scenario(scenario, fomMeshObj, \
                                                coeffDic, fomDic, val)

    # run FOM for current fomDic
    runDir = workDir + "/fom_"+testOrTrainString+"_"+str(k)
    if not os.path.exists(runDir):
      os.makedirs(runDir)
      logging.info("Running FOM {}".format(runDir))
      run_single_fom(runDir, fomObj, fomDic)
    else:
      logging.info("{} already exists".format(runDir))

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

  # read args
  parser   = ArgumentParser()
  parser.add_argument("--wdir", dest="workdir", required=True)
  parser.add_argument("--problem", dest="problem", required=True)
  parser.add_argument("-s", dest="scenario", type=int,  required=True)
  parser.add_argument("--pdadir", dest="pdadir", required=True)
  args     = parser.parse_args()
  workDir  = args.workdir
  pdaDir   = args.pdadir
  problem  = args.problem
  scenario = args.scenario

  if not os.path.exists(workDir):
    logging.info("{} does not exist, creating it".format(workDir))
    os.system('mkdir -p ' + workDir)

  # write scenario id, problem to file
  write_scenario_to_file(scenario, workDir)
  write_problem_name_to_file(problem, workDir)

  banner_import_problem()
  module = importlib.import_module(problem)
  check_and_print_problem_summary(problem, module)
  print("")

  # verify that scenario is a valid key in the specialized dics in module.
  # use base_dic to check this because that dic should always be present.
  valid_scenarios_ids = list(module.base_dic.keys())
  if scenario not in valid_scenarios_ids:
    logging.error("Scenario = {} is invalid for the target problem".format(scenario))
    sys.exit(1)

  # --------------------------------------
  banner_make_fom_mesh()
  meshSize = module.base_dic[scenario]["fom"]["meshSize"]
  make_fom_mesh_if_not_existing(workDir, problem, module, \
                                scenario, pdaDir, meshSize)
  # before moving on, ensure that in workDir there is a UNIQUE FULL mesh.
  # Because the mesh must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory
  fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

  banner_fom_train()
  run_foms(workDir, problem, module, scenario, "train", fomMeshPath)

  banner_fom_test()
  run_foms(workDir, problem, module, scenario, "test", fomMeshPath)
