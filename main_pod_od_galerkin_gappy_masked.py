
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
  find_all_sample_meshes_for_target_partition_info, \
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

from py_src.fncs_compute_projector import \
  compute_gappy_projector_using_factor_of_state_pod_modes

from py_src.fncs_to_extract_from_mesh_info_file import *

from py_src.fncs_impl_od_galerkin_for_all_test_values import \
  run_masked_gappy_od_galerkin_for_all_test_values

# -------------------------------------------------------------------
def run_od_pod_galerkin_gappy_masked(workDir, problem, module, \
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
      currStatePodDir = path_to_state_pod_data_dir(workDir, partitionStringIdentifier, setId)
      currRhsPodDir   = path_to_rhs_pod_data_dir(workDir, partitionStringIdentifier, setId)

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

              projectorDir = path_to_gappy_projector_dir(workDir, "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         None, numOfModes, smKeyword)
              if os.path.exists(projectorDir):
                logging.info('{} already exists'.format(os.path.basename(projectorDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(projectorDir)))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStatePodDir, currRhsPodDir,\
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
                                       currStatePodDir, sampleMeshDirIt, \
                                       modesPerTileDic, module.numDofsPerCell)

              run_masked_gappy_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                               scenario, partInfoDirIt, \
                                                               fomMeshPath, sampleMeshDirIt, \
                                                               currStatePodDir, projectorDir, \
                                                               modeSettingIt_key, \
                                                               None, modesPerTileDic, \
                                                               setId, smKeyword)

          #*********************************************************************
          elif modeSettingIt_key == 'findMinValueAcrossTilesUsingEnergyAndUseInAllTiles':
          #*********************************************************************
            for energyValue in modeSettingIt_val:
              modesPerTileDicTmp = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                          currStatePodDir, energyValue)
              # find minimum value
              minMumModes = np.min(list(modesPerTileDicTmp.values()))
              modesPerTileDic = make_modes_per_tile_dic_with_const_modes_count(nTiles, minMumModes)

              projectorDir = path_to_gappy_projector_dir(workDir, "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         energyValue, None, smKeyword)
              if os.path.exists(projectorDir):
                logging.info('{} already exists'.format(os.path.basename(projectorDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(projectorDir)))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStatePodDir, currRhsPodDir,\
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
                                       currStatePodDir, sampleMeshDirIt, \
                                       modesPerTileDic, module.numDofsPerCell)


              run_masked_gappy_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                               scenario, partInfoDirIt, \
                                                               fomMeshPath, sampleMeshDirIt, \
                                                               currStatePodDir, projectorDir, \
                                                               modeSettingIt_key, \
                                                               energyValue, modesPerTileDic, \
                                                               setId, smKeyword)

          #*********************************************************************
          elif modeSettingIt_key == 'tileSpecificUsingEnergy':
          #*********************************************************************
            for energyValue in modeSettingIt_val:
              modesPerTileDic = find_modes_per_tile_from_target_energy(module, scenario, \
                                                                       currStatePodDir, energyValue)
              projectorDir = path_to_gappy_projector_dir(workDir, "basedOnFactorOfStateModes",\
                                                         partitionStringIdentifier, \
                                                         setId, modeSettingIt_key, \
                                                         energyValue, None, smKeyword)
              if os.path.exists(projectorDir):
                logging.info('{} already exists'.format(os.path.basename(projectorDir)))
              else:
                logging.info('Generating {}'.format(os.path.basename(projectorDir)))
                os.system('mkdir -p ' + projectorDir)
                compute_gappy_projector_using_factor_of_state_pod_modes(projectorDir, partInfoDirIt, \
                                                                        currStatePodDir, currRhsPodDir,\
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
                                       currStatePodDir, sampleMeshDirIt, \
                                       modesPerTileDic, module.numDofsPerCell)


              run_masked_gappy_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                               scenario, partInfoDirIt, \
                                                               fomMeshPath, sampleMeshDirIt, \
                                                               currStatePodDir, projectorDir, \
                                                               modeSettingIt_key, \
                                                               energyValue, modesPerTileDic, \
                                                               setId, smKeyword)


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

  if "PodOdGalerkinGappyMasked" in module.algos[scenario]:
    banner_run_pod_od_galerkin_gappy_masked()

    # before we move on, we need to ensure that in workDir
    # there is a unique FULL mesh. This is because the mesh is specified
    # via command line argument and must be unique for a scenario.
    # If one wants to run for a different mesh, then they have to
    # run this script again with a different working directory
    fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

    run_od_pod_galerkin_gappy_masked(workDir, problem, module, scenario, fomMeshPath)

  else:
    logging.info("Nothing to do here")
