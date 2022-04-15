
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
  compute_total_modes_across_all_tiles, \
  find_all_sample_meshes_for_target_partition_info

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
  path_to_rhs_pod_data_dir, \
  string_identifier_from_partition_info_dir, \
  path_to_partition_based_full_mesh_dir, \
  path_to_gappy_projector_dir,\
  path_to_phi_on_stencil_dir, \
  string_identifier_from_sample_mesh_dir

from py_src.compute_phi_on_stencil import *

from py_src.compute_gappy_projector import \
  compute_gappy_projector_using_factor_of_state_pod_modes

from py_src.mesh_info_file_extractors import *

from py_src.impl_od_galerkin_for_all_test_values import \
  run_hr_od_galerkin_for_all_test_values

# # -------------------------------------------------------------------
# def run_od_pod_galerkin_quad(workDir, problem, module, \
#                              scenario, fomMeshPath):

#   # -------
#   # loop 1: over all decompositions
#   # ------
#   for partInfoDirIt in find_all_partitions_info_dirs(workDir):
#     partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

#     # -------
#     # loop 2: over all POD computed from various sets of train runs
#     # ------
#     for setId, trainIndices in module.basis_sets[scenario].items():
#       currStateFullPodDir = path_to_state_pod_data_dir(workDir, partitionStringIdentifier, setId)

#       # find all train dirs for current setId
#       trainDirs = [workDir+'/'+d for d in os.listdir(workDir) \
#                    if "train" in d and get_run_id(d) in trainIndices]
#       assert(len(trainDirs) == len(trainIndices))

#       # -------
#       # loop 3: over all target energies
#       # ------
#       for energyValue in module.odrom_energies[scenario]:
#         modesPerTileDic = find_modes_per_tile_from_target_energy(module, scenario, \
#                                                                  currStateFullPodDir, energyValue)

#         # -------
#         # loop 4: over all samples meshes
#         # ------
#         for sampleMeshDirIt in find_all_sample_meshes_for_target_partition_info(workDir, partInfoDirIt):
#           smKeyword = string_identifier_from_sample_mesh_dir(sampleMeshDirIt)

#           # compute projector for each tile if needed
#           projectorDir = path_to_quad_projector_dir(workDir, \
#                                                     partitionStringIdentifier, \
#                                                     setId, \
#                                                     energyValue, \
#                                                     smKeyword)
#           if os.path.exists(projectorDir):
#             print('{} already exists'.format(projectorDir))
#           else:
#             print('Generating {}'.format(projectorDir))
#             os.system('mkdir -p ' + projectorDir)
#             compute_quad_projector(trainDirs, fomMeshPath, \
#                                    projectorDir, partInfoDirIt, \
#                                    currStateFullPodDir, \
#                                    sampleMeshDirIt, modesPerTileDic,
#                                    module.numDofsPerCell)

#           # compute phi on stencil for each tile if needed
#           phiOnStencilDir = path_to_phi_on_stencil_dir(workDir, \
#                                                        partitionStringIdentifier, \
#                                                        setId, \
#                                                        energyValue, \
#                                                        smKeyword)
#           print(phiOnStencilDir)
#           if os.path.exists(phiOnStencilDir):
#             print('{} already exists'.format(phiOnStencilDir))
#           else:
#             print('Generating {}'.format(phiOnStencilDir))
#             os.system('mkdir -p ' + phiOnStencilDir)
#             compute_phi_on_stencil(phiOnStencilDir, partInfoDirIt, \
#                                    currStateFullPodDir, sampleMeshDirIt, \
#                                    modesPerTileDic, module.numDofsPerCell)

#           # -------
#           # loop 5: over all test values
#           # ------
#           run_hr_od_galerkin_for_all_test_values(workDir, problem, module, \
#                                                  scenario, partInfoDirIt, \
#                                                  fomMeshPath, sampleMeshDirIt, \
#                                                  currStateFullPodDir, projectorDir,
#                                                  phiOnStencilDir, \
#                                                  energyValue, modesPerTileDic, \
#                                                  setId, smKeyword, "quad")

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
  banner_run_pod_od_galerkin_quad_real()
  # --------------------------------------
  if "PodOdGalerkinQuad" in module.algos[scenario]:
    sys.exit("PodOdGalerkinQuad: code is here but needs revision")
    #run_od_pod_galerkin_quad(workDir, problem, module, scenario, fomMeshPath)
  print("")
