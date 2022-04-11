
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, subprocess, pathlib
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
  make_modes_per_tile_dic_with_const_modes_count

from py_src.myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir

from py_src.directory_naming import \
  path_to_state_pod_data_dir, \
  string_identifier_from_partition_info_dir

from py_src.fom_run_dirs_detection import \
  find_all_fom_test_dirs

from py_src.mesh_info_file_extractors import *

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

                args = ("python3", \
                        str(this_file_path)+'/py_src/proj_error_od.py', \
                        "--wdir", outDir, \
                        "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, \
                        "--infodir", partInfoDirIt,\
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

                args = ("python3", \
                        str(this_file_path)+'/py_src/proj_error_od.py', \
                        "--wdir", outDir, \
                        "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, \
                        "--infodir", partInfoDirIt,\
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

                args = ("python3", \
                        str(this_file_path)+'/py_src/proj_error_od.py', \
                        "--wdir", outDir, \
                        "--fomdir", fomTestDirIt, \
                        "--poddir", currPodDir, \
                        "--infodir", partInfoDirIt,\
                        "--userefstate",  str(module.use_ic_reference_state[scenario]))
                popen  = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()
                output = popen.stdout.read();


        else:
          sys.exit('compute_od_pod_projection_errors: invalid modeSettingPolicy = {}'.format(modeSettingIt_key))


#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  banner_driving_script_info(os.path.basename(__file__))

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

  if "PodOdProjectionError" in module.algos[scenario]:
    banner_compute_od_projection_error()
    compute_od_pod_projection_errors(workDir, problem, module, scenario)
  else:
    print("Nothing to do here")
