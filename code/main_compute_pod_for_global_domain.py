
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, logging
import numpy as np

from py_src.fncs_banners_and_prints import \
  banner_driving_script_info, \
  banner_import_problem, check_and_print_problem_summary, \
  banner_compute_full_pod, color_resetter

from py_src.fncs_miscellanea import \
  find_full_mesh_and_ensure_unique, \
  get_run_id

from py_src.fncs_myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir, \
  load_fom_state_snapshot_matrix, \
  load_fom_rhs_snapshot_matrix

from py_src.fncs_directory_naming import \
  path_to_full_domain_state_pod_data_dir, \
  path_to_full_domain_rhs_pod_data_dir

from py_src.fncs_fom_run_dirs_detection import \
  find_fom_train_dirs_for_target_set_of_indices

from py_src.fncs_to_extract_from_mesh_info_file import *
from py_src.fncs_svd import do_svd_py

# -------------------------------------------------------------------
def compute_full_domain_state_pod(workDir, module, scenario, \
                                  setId, dataDirs, fomMesh):
  outDir = path_to_full_domain_state_pod_data_dir(workDir, setId)
  if os.path.exists(outDir):
    logging.info('{} already exists'.format(outDir))
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
    logging.debug("fomStateSnapsFullDomain.shape = {}".format(fomStateSnapsFullDomain.shape))

    os.system('mkdir -p ' + outDir)
    lsvFile = outDir + '/lsv_state_p_0'
    svaFile = outDir + '/sva_state_p_0'
    do_svd_py(fomStateSnapsFullDomain, lsvFile, svaFile)
  logging.info("")

# -------------------------------------------------------------------
def compute_full_domain_rhs_pod(workDir, module, scenario, \
                                setId, dataDirs, fomMesh):
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell
  fomRhsSnapsFullDomain   = load_fom_rhs_snapshot_matrix(dataDirs, totFomDofs, \
                                                         module.numDofsPerCell)
  logging.debug("fomRhsSnapsFullDomain.shape = {}".format(fomRhsSnapsFullDomain.shape))

  outDir = path_to_full_domain_rhs_pod_data_dir(workDir, setId)
  if os.path.exists(outDir):
    logging.info('{} already exists'.format(outDir))
  else:
    os.system('mkdir -p ' + outDir)
    lsvFile = outDir + '/lsv_rhs_p_0'
    svaFile = outDir + '/sva_rhs_p_0'
    do_svd_py(fomRhsSnapsFullDomain, lsvFile, svaFile)
  logging.info("")

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

  # we need to compute global POD if any of the following
  # substrings is present in the algo list of target scenario
  matchers = ["GlobalGalerkinWithPodBases", "ProjectionErrorUsingGlobalPodBases"]
  matching = [s for s in module.algos[scenario] if any(xs in s for xs in matchers)]
  if matching or module.odrom_tile_based_or_split_global[scenario]=="SplitGlobal":
    banner_compute_full_pod()

    # before we move on, we need to ensure that in workDir
    # there is a unique FULL mesh. This is because the mesh is specified
    # via command line argument and must be unique for a scenario.
    # If one wants to run for a different mesh, then they have to
    # run this script again with a different working directory
    fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

    for setId, trainIndices in module.basis_sets[scenario].items():
      logging.info("computing STATE POD on FULL domain for setId = {} {}".format(setId, color_resetter()))
      logging.info(55*"-")
      trainDirs = find_fom_train_dirs_for_target_set_of_indices(workDir, trainIndices)
      compute_full_domain_state_pod(workDir, module, scenario, \
                                    setId, trainDirs, fomMeshPath)

      logging.info("computing RHS POD on FULL domain for setId = {} {}".format(setId, color_resetter()))
      logging.info(55*"-")
      compute_full_domain_rhs_pod(workDir, module, scenario, \
                                  setId, trainDirs, fomMeshPath)
