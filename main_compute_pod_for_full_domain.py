
# standard modules
from argparse import ArgumentParser
import sys, os, importlib
import numpy as np

from py_src.banners_and_prints import *

from py_src.miscellanea import \
  find_full_mesh_and_ensure_unique, \
  get_run_id

from py_src.myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir, \
  load_fom_state_snapshot_matrix, \
  load_fom_rhs_snapshot_matrix

from py_src.directory_naming import \
  path_to_full_domain_state_pod_data_dir, \
  path_to_full_domain_rhs_pod_data_dir

from py_src.fom_run_dirs_detection import \
  find_fom_train_dirs_for_target_set_of_indices

from py_src.mesh_info_file_extractors import *

from py_src.svd import do_svd_py

# -------------------------------------------------------------------
def compute_full_domain_state_pod(workDir, module, scenario, \
                                  setId, dataDirs, fomMesh):
  outDir = path_to_full_domain_state_pod_data_dir(workDir, setId)
  if os.path.exists(outDir):
    print('{} already exists'.format(outDir))
  else:
    fomTotCells = find_total_cells_from_info_file(fomMesh)
    totFomDofs  = fomTotCells*module.numDofsPerCell
    # find from scenario if we want to subtract initial condition
    # from snapshots before doing pod.
    subtractInitialCondition = module.use_ic_reference_state[scenario]

    # load snapshots once
    fomStateSnapsFullDomain = load_fom_state_snapshot_matrix(dataDirs, totFomDofs, \
                                                             module.numDofsPerCell, \
                                                             subtractInitialCondition)
    print("pod: fomStateSnapsFullDomain.shape = ", fomStateSnapsFullDomain.shape)

    os.system('mkdir -p ' + outDir)
    lsvFile = outDir + '/lsv_state_p_0'
    svaFile = outDir + '/sva_state_p_0'
    do_svd_py(fomStateSnapsFullDomain, lsvFile, svaFile)
  print("")

# -------------------------------------------------------------------
def compute_full_domain_rhs_pod(workDir, module, scenario, \
                                setId, dataDirs, fomMesh):
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell
  fomRhsSnapsFullDomain   = load_fom_rhs_snapshot_matrix(dataDirs, totFomDofs, \
                                                         module.numDofsPerCell)
  print("pod: fomRhsSnapsFullDomain.shape = ", fomRhsSnapsFullDomain.shape)

  outDir = path_to_full_domain_rhs_pod_data_dir(workDir, setId)
  if os.path.exists(outDir):
    print('{} already exists'.format(outDir))
  else:
    os.system('mkdir -p ' + outDir)
    lsvFile = outDir + '/lsv_rhs_p_0'
    svaFile = outDir + '/sva_rhs_p_0'
    do_svd_py(fomRhsSnapsFullDomain, lsvFile, svaFile)
  print("")

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
  banner_compute_full_pod()
  # --------------------------------------
  triggers = ["PodStandardGalerkinFull", \
              "PodStandardGalerkinGappy"]
  if any(x in triggers for x in module.algos[scenario]):

    for setId, trainIndices in module.basis_sets[scenario].items():
      print("\033[1;37;46mFULL domain STATE POD for setId = {} {}".format(setId, color_resetter()))
      print("------------------------------------")
      trainDirs = find_fom_train_dirs_for_target_set_of_indices(workDir, trainIndices)
      compute_full_domain_state_pod(workDir, module, scenario, \
                                    setId, trainDirs, fomMeshPath)

      if "PodStandardGalerkinGappy" in module.algos[scenario]:
        print("\033[1;37;46mFULL domain RHS POD for setId = {} {}".format(setId, color_resetter()))
        print("----------------------------------")
        compute_full_domain_rhs_pod(workDir, module, scenario, \
                                    setId, trainDirs, fomMeshPath)
  else:
    print("skipping")
  print("")
