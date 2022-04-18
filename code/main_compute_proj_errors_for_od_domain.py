
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
  find_full_mesh_and_ensure_unique,\
  get_run_id, \
  find_all_partitions_info_dirs, \
  make_modes_per_tile_dic_with_const_modes_count, \
  find_modes_per_tile_from_target_energy

from py_src.fncs_myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir

from py_src.fncs_directory_naming import \
  path_to_state_pod_data_dir, \
  string_identifier_from_partition_info_dir

from py_src.fncs_fom_run_dirs_detection import \
  find_all_fom_test_dirs

from py_src.fncs_to_extract_from_mesh_info_file import *

# -------------------------------------------------------------------
def compute_od_pod_projection_errors(workDir, problem, module, scenario):

  this_file_path = pathlib.Path(__file__).parent.absolute()
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

        # --------------------------------------------------------------
        if modeSettingIt_key == 'allTilesUseTheSameUserDefinedValue':
        # --------------------------------------------------------------
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
                logging.info('{} already exists'.format(os.path.basename(outDir)))
              else:
                logging.info("od proj error in {}".format(os.path.basename(outDir)))
                os.system('mkdir -p ' + outDir)

                np.savetxt(outDir+"/modes_per_tile.txt", \
                           np.array(list(modesPerTileDic.values())),
                           fmt="%5d")

                args = ("python3", \
                        str(this_file_path)+'/proj_error_od_domain_single_run.py', \
                        "--wdir", outDir, \
                        "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, \
                        "--infodir", partInfoDirIt,\
                        "--userefstate", str(module.use_ic_reference_state[scenario]))
                popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read();

        # --------------------------------------------------------------
        elif modeSettingIt_key == 'tileSpecificUsingEnergy':
        # --------------------------------------------------------------
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
                logging.info('{} already exists'.format(os.path.basename(outDir)))
              else:
                logging.info("od proj error in {}".format(os.path.basename(outDir)))
                os.system('mkdir -p ' + outDir)

                np.savetxt(outDir+"/modes_per_tile.txt", \
                           np.array(list(modesPerTileDic.values())),
                           fmt="%5d")

                args = ("python3", \
                        str(this_file_path)+'/proj_error_od_domain_single_run.py', \
                        "--wdir", outDir, \
                        "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, \
                        "--infodir", partInfoDirIt,\
                        "--userefstate",  str(module.use_ic_reference_state[scenario]))
                popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read();

        # --------------------------------------------------------------
        elif modeSettingIt_key in ['findMinValueAcrossTilesUsingEnergyAndUseInAllTiles', \
                                   'findMaxValueAcrossTilesUsingEnergyAndUseInAllTiles']:
        # --------------------------------------------------------------
          for energyValue in modeSettingIt_val:
            modesPerTileDicTmp = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                        currPodDir, energyValue)
            numModesChosen = 0 
            if 'min' in modeSettingIt_key: 
              numModesChosen = np.min(list(modesPerTileDicTmp.values()))
            else:
              numModesChosen = np.max(list(modesPerTileDicTmp.values()))
            modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, numModesChosen)

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
                logging.info('{} already exists'.format(os.path.basename(outDir)))
              else:
                logging.info("od proj error in {}".format(os.path.basename(outDir)))
                os.system('mkdir -p ' + outDir)

                np.savetxt(outDir+"/modes_per_tile.txt", \
                           np.array(list(modesPerTileDic.values())),
                           fmt="%5d")

                args = ("python3", \
                        str(this_file_path)+'/proj_error_od_domain_single_run.py', \
                        "--wdir", outDir, \
                        "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, \
                        "--infodir", partInfoDirIt,\
                        "--userefstate",  str(module.use_ic_reference_state[scenario]))
                popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read();


        else:
          logging.error('compute_od_pod_projection_errors: invalid modeSettingPolicy = {}'.format(modeSettingIt_key))
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

  if "PodOdProjectionError" in module.algos[scenario]:
    banner_compute_od_projection_error()
    compute_od_pod_projection_errors(workDir, problem, module, scenario)
  else:
    logging.info("Nothing to do here")
