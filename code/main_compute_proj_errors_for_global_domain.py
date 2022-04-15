
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, subprocess, pathlib, logging
import numpy as np
from scipy import linalg as scipyla

try:
  import pressiodemoapps as pda
except ImportError:
  raise ImportError("Unable to import pressiodemoapps")

from py_src.fncs_banners_and_prints import *

from py_src.fncs_miscellanea import \
  find_full_mesh_and_ensure_unique, get_run_id, \
  find_modes_for_full_domain_from_target_energy

from py_src.fncs_myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir

from py_src.fncs_directory_naming import \
  path_to_full_domain_state_pod_data_dir

from py_src.fncs_fom_run_dirs_detection import \
  find_all_fom_test_dirs

from py_src.fncs_to_extract_from_mesh_info_file import *

# -------------------------------------------------------------------
def compute_projection_errors(workDir, problem, module, scenario):

  this_file_path = pathlib.Path(__file__).parent.absolute()

  # loop: over all sets of train runs
  howManySets = len(module.basis_sets[scenario].keys())
  for setId in range(howManySets):
    currPodDir = path_to_full_domain_state_pod_data_dir(workDir, setId)

    # loop: over all mode setting policies
    for modeSettingIt_key, modeSettingIt_val in module.standardrom_modes_setting_policies[scenario].items():

      # --------------------------------------------------
      if modeSettingIt_key == 'userDefinedValue':
      # --------------------------------------------------
        for numModes in modeSettingIt_val:

          # assemble the name of output directory
          outDirRoot = workDir + "/full_domain_proj_error_using_pod_bases"
          outDirRoot += "_modesSettingPolicy_"+modeSettingIt_key
          outDirRoot += "_"+str(numModes)
          outDirRoot += "_set_"+str(setId)

          # loop over each FOM test dir and compute proj error
          for fomTestDirIt in find_all_fom_test_dirs(workDir):
            outDir = outDirRoot + "_" + str(get_run_id(fomTestDirIt))
            if os.path.exists(outDir):
              logging.info('{} already exists'.format(os.path.basename(outDir)))
            else:
              logging.info("computing {}".format(os.path.basename(outDir)))
              os.system('mkdir -p ' + outDir)

              # num of modes is needed by the script below
              np.savetxt(outDir+"/modes.txt", np.array([numModes]), fmt="%5d")

              args = ("python3", \
                      str(this_file_path)+'/proj_error_global_domain_single_run.py', \
                      "--wdir", outDir, \
                      "--fomdir", fomTestDirIt, \
                      "--poddir", currPodDir, \
                      "--userefstate",  str(module.use_ic_reference_state[scenario]))
              popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
              popen.wait()
              output = popen.stdout.read();

      # --------------------------------------------------
      elif modeSettingIt_key == 'energyBased':
      # --------------------------------------------------
        for energyValue in modeSettingIt_val:
          numModes = find_modes_for_full_domain_from_target_energy(module, scenario, \
                                                                   currPodDir, energyValue)

          # assemble the name of output directory
          outDirRoot = workDir + "/full_domain_proj_error_using_pod_bases"
          outDirRoot += "_modesSettingPolicy_"+modeSettingIt_key
          outDirRoot += "_"+str(energyValue)
          outDirRoot += "_set_"+str(setId)

          # loop over each FOM test dir and compute proj error
          for fomTestDirIt in find_all_fom_test_dirs(workDir):
            outDir = outDirRoot + "_" + str(get_run_id(fomTestDirIt))
            if os.path.exists(outDir):
              logging.info('{} already exists'.format(os.path.basename(outDir)))
            else:
              logging.info("computing {}".format(os.path.basename(outDir)))
              os.system('mkdir -p ' + outDir)

              # num of modes is needed by the script below
              np.savetxt(outDir+"/modes.txt", np.array([numModes]), fmt="%5d")

              args = ("python3", \
                      str(this_file_path)+'/proj_error_global_domain_single_run.py', \
                      "--wdir", outDir, \
                      "--fomdir", fomTestDirIt, \
                      "--poddir", currPodDir, \
                      "--userefstate",  str(module.use_ic_reference_state[scenario]))
              popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
              popen.wait()
              output = popen.stdout.read();

      else:
        logging.error(__file__ + 'invalid modeSettingPolicy = {}'.format(modeSettingIt_key))
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

  if "PodStandardProjectionError" in module.algos[scenario]:
    banner_compute_full_domain_projection_error()
    compute_projection_errors(workDir, problem, module, scenario)
  else:
    logging.info("Nothing to do here")
