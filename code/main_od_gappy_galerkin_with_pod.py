
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, subprocess, logging
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
  make_modes_per_tile_dic_with_const_modes_count,\
  compute_total_modes_across_all_tiles, \
  find_all_sample_meshes_for_target_partition_info,\
  find_modes_per_tile_from_target_energy

from py_src.fncs_myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir,\
  load_fom_state_snapshot_matrix, \
  load_fom_rhs_snapshot_matrix, \
  load_basis_from_binary_file, \
  write_dic_to_yaml_file

from py_src.fncs_directory_naming import \
  path_to_partition_info_dir, \
  path_to_state_pod_data_dir, \
  path_to_rhs_pod_data_dir, \
  string_identifier_from_partition_info_dir, \
  path_to_partition_based_full_mesh_dir, \
  path_to_gappy_projector_dir,\
  path_to_phi_on_stencil_dir, \
  string_identifier_from_sample_mesh_dir

from py_src.fncs_compute_phi_on_stencil import *

from py_src.fncs_compute_gappy_projector import \
  compute_gappy_projector_using_factor_of_state_pod_modes

from py_src.fncs_to_extract_from_mesh_info_file import *

from py_src.fncs_impl_od_galerkin_for_all_test_values import \
  run_hr_od_galerkin_for_all_test_values

# -------------------------------------------------------------------
def main_impl(workDir, problem, module, scenario, fomMeshPath):

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

          #*********************************************************************
          if modeSettingIt_key == 'allTilesUseTheSameUserDefinedValue':
          #*********************************************************************
            for numModes in modeSettingIt_val:
              modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, numModes)
              # all tiles have same modes, so find what that is
              numOfModes = modesPerTileDic[0]

              projectorDir = path_to_gappy_projector_dir(workDir, \
                                                         "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         None, numOfModes, smKeyword)
              if os.path.exists(projectorDir):
                logging.info('{} already exists'.format(os.path.basename(projectorDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(projectorDir)))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStateFullPodDir, currRhsFullPodDir,\
                                                                        sampleMeshDirIt, modesPerTileDic,
                                                                        module.numDofsPerCell)

              phiOnStencilDir = path_to_phi_on_stencil_dir(workDir, partitionStringIdentifier, \
                                                           setId, modeSettingIt_key,\
                                                           None, numOfModes, smKeyword)
              if os.path.exists(phiOnStencilDir):
                logging.info('{} already exists'.format(os.path.basename(phiOnStencilDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(phiOnStencilDir)))
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

          #*********************************************************************
          elif modeSettingIt_key in ['findMinValueAcrossTilesUsingEnergyAndUseInAllTiles', \
                                     'findMaxValueAcrossTilesUsingEnergyAndUseInAllTiles']:
          #*********************************************************************
            for energyValue in modeSettingIt_val:
              modesPerTileDicTmp = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                          currStateFullPodDir, energyValue)
              numModesChosen = 0
              if 'min' in modeSettingIt_key:
                numModesChosen = np.min(list(modesPerTileDicTmp.values()))
              else:
                numModesChosen = np.max(list(modesPerTileDicTmp.values()))

              modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, numModesChosen)

              projectorDir = path_to_gappy_projector_dir(workDir, \
                                                         "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         energyValue, None, smKeyword)
              if os.path.exists(projectorDir):
                logging.info('{} already exists'.format(os.path.basename(projectorDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(projectorDir)))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStateFullPodDir, currRhsFullPodDir,\
                                                                        sampleMeshDirIt, modesPerTileDic,
                                                                        module.numDofsPerCell)

              phiOnStencilDir = path_to_phi_on_stencil_dir(workDir, partitionStringIdentifier, \
                                                           setId, modeSettingIt_key,\
                                                           energyValue, None, smKeyword)
              if os.path.exists(phiOnStencilDir):
                logging.info('{} already exists'.format(os.path.basename(phiOnStencilDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(phiOnStencilDir)))
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

          #*********************************************************************
          elif modeSettingIt_key == 'tileSpecificUsingEnergy':
          #*********************************************************************
            for energyValue in modeSettingIt_val:
              modesPerTileDic = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                       currStateFullPodDir, energyValue)
              projectorDir = path_to_gappy_projector_dir(workDir, \
                                                         "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         energyValue, None, smKeyword)
              if os.path.exists(projectorDir):
                logging.info('{} already exists'.format(os.path.basename(projectorDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(projectorDir)))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStateFullPodDir, currRhsFullPodDir,\
                                                                        sampleMeshDirIt, modesPerTileDic,
                                                                        module.numDofsPerCell)

              # compute phi on stencil for each tile if needed
              phiOnStencilDir = path_to_phi_on_stencil_dir(workDir, partitionStringIdentifier, \
                                                           setId, modeSettingIt_key,\
                                                           energyValue, None, smKeyword)
              if os.path.exists(phiOnStencilDir):
                logging.info('{} already exists'.format(os.path.basename(phiOnStencilDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(phiOnStencilDir)))
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

  if "OdGappyGalerkinWithPodBases" in module.algos[scenario]:
    banner_run_pod_od_galerkin_gappy_real()
    fomMeshPath = find_full_mesh_and_ensure_unique(workDir)
    main_impl(workDir, problem, module, scenario, fomMeshPath)

  else:
    logging.info("Nothing to do here")
