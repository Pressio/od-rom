
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, logging
import numpy as np

from py_src.fncs_banners_and_prints import *

from py_src.fncs_miscellanea import \
  find_full_mesh_and_ensure_unique,\
  get_run_id, \
  find_all_partitions_info_dirs

from py_src.fncs_myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir,\
  load_fom_state_snapshot_matrix, \
  load_fom_rhs_snapshot_matrix

from py_src.fncs_directory_naming import \
  path_to_partition_info_dir, \
  path_to_state_pod_data_dir, \
  path_to_rhs_pod_data_dir, \
  string_identifier_from_partition_info_dir

from py_src.fncs_fom_run_dirs_detection import \
  find_fom_train_dirs_for_target_set_of_indices

from py_src.fncs_to_extract_from_mesh_info_file import *
from py_src.fncs_svd import do_svd_py

# -------------------------------------------------------------------
def compute_partition_based_state_pod(workDir, module, scenario, \
                                      setId, dataDirs, fomMesh):
  '''
  compute pod from STATE snapshosts
  '''
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell

  # find if we want to subtract initial condition before doing pod
  subtractInitialCondition = module.use_ic_reference_state[scenario]

  # logic to ensure snapshots are only loaded once
  needToLoadSnaphots = True
  fomStateSnapsFullDomain = None

  # with the FOM data loaded for a target setId (i.e. set of runs)
  # loop over all partitions and compute local POD.
  for partitionInfoDirIt in find_all_partitions_info_dirs(workDir):
    # need an identifier from this partition directory so that I can
    # use it to uniquely associate a directory where we store the POD
    stringIdentifier = string_identifier_from_partition_info_dir(partitionInfoDirIt)
    nTiles = np.loadtxt(partitionInfoDirIt+"/ntiles.txt", dtype=int)

    outDir = path_to_state_pod_data_dir(workDir, stringIdentifier, setId)
    if os.path.exists(outDir):
      logging.info('{} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # only load snapshots once
      if needToLoadSnaphots:
        needToLoadSnaphots = False
        fomStateSnapsFullDomain = load_fom_state_snapshot_matrix(dataDirs, totFomDofs, \
                                                                 module.numDofsPerCell, \
                                                                 subtractInitialCondition)
        logging.debug("pod: fomStateSnapsFullDomain.shape = {}".format(fomStateSnapsFullDomain.shape))


      # loop over each tile
      for tileId in range(nTiles):
        # I need to compute POD for both STATE and RHS
        # using FOM data LOCAL to myself, so need to load
        # which rows of the FOM state I own and use to slice
        myFile = partitionInfoDirIt + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
        myRowsInFullState = np.loadtxt(myFile, dtype=int)

        # use the row indices to get only the data that belongs to me
        myStateEntries = fomStateSnapsFullDomain[myRowsInFullState, :]
        logging.debug(" state pod for tileId={:>5} with stateSlice.Shape={}".format(tileId, myStateEntries.shape))

        lsvFile = outDir + '/lsv_state_p_'+str(tileId)
        svaFile = outDir + '/sva_state_p_'+str(tileId)
        do_svd_py(myStateEntries, lsvFile, svaFile)

# -------------------------------------------------------------------
def compute_partition_based_rhs_pod(workDir, module, scenario, \
                                    setId, dataDirs, fomMesh):
  '''
  compute pod for RHS snapshots
  '''
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell

  # logic to ensure snapshots are only loaded once
  needToLoadSnaphots = True
  fomRhsSnapsFullDomain = None

  # with the FOM data loaded for a target setId (i.e. set of runs)
  # loop over all partitions and compute local POD.
  for partitionInfoDirIt in find_all_partitions_info_dirs(workDir):
    # need an identifier from this partition directory so that I can
    # use it to uniquely associate a directory where we store the POD
    stringIdentifier = string_identifier_from_partition_info_dir(partitionInfoDirIt)
    nTiles = np.loadtxt(partitionInfoDirIt+"/ntiles.txt", dtype=int)
    #nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

    outDir = path_to_rhs_pod_data_dir(workDir, stringIdentifier, setId)
    if os.path.exists(outDir):
      logging.info('{} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # only load snapshots once
      if needToLoadSnaphots:
        needToLoadSnaphots = False
        fomRhsSnapsFullDomain = load_fom_rhs_snapshot_matrix(dataDirs, totFomDofs, \
                                                             module.numDofsPerCell)
        logging.debug("pod: fomRhsSnapsFullDomain.shape = {}".format(fomRhsSnapsFullDomain.shape))

      # loop over each tile
      for tileId in range(nTiles):
        # I need to compute POD for both STATE and RHS
        # using FOM data LOCAL to myself, so need to load
        # which rows of the FOM state I own and use to slice
        myFile = partitionInfoDirIt + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
        myRowsInFullState = np.loadtxt(myFile, dtype=int)

        # use the row indices to get only the data that belongs to me
        myRhsSlice   = fomRhsSnapsFullDomain[myRowsInFullState, :]
        logging.debug(" rhs pod for tileId={:>5} with rhsSlice.shape={}".format(tileId, myRhsSlice.shape))

        lsvFile = outDir + '/lsv_rhs_p_'+str(tileId)
        svaFile = outDir + '/sva_rhs_p_'+str(tileId)
        do_svd_py(myRhsSlice, lsvFile, svaFile)


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
    sys.exit("Working dir {} does not exist, terminating".format(workDir))

  banner_import_problem()
  scenario = read_scenario_from_dir(workDir)
  problem  = read_problem_name_from_dir(workDir)
  module   = importlib.import_module(problem)
  check_and_print_problem_summary(problem, module)
  logging.info("")

  # before we move on, we need to ensure that in workDir
  # there is a unique FULL mesh. This is because the mesh is specified
  # via command line argument and must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory
  fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

  mustDoPodModesForEachTile = False
  if "PodOdGalerkin"        in module.algos[scenario] or \
     "PodOdGalerkinGappy"   in module.algos[scenario] or \
     "PodOdGalerkinMasked"  in module.algos[scenario] or \
     "PodOdGalerkinQuad"    in module.algos[scenario] or \
     "PodOdProjectionError" in module.algos[scenario]:
    mustDoPodModesForEachTile = True

  # if scenario has PolyOdGalerkin and the poly_order = -1
  # because poly_order = -1 indicates that we compute the poly order
  # in each tile such that we match as possible the number of local pod modes
  if "PolyOdGalerkinFull" in module.algos[scenario] and \
     -1 in module.odrom_poly_order[scenario]:
    mustDoPodModesForEachTile = True

  if mustDoPodModesForEachTile:
    banner_compute_pod_all_partitions()

    for setId, trainIndices in module.basis_sets[scenario].items():
      logging.info("partition-local POD for setId = {}".format(setId))
      print(35*"-")
      trainDirs = find_fom_train_dirs_for_target_set_of_indices(workDir, trainIndices)
      compute_partition_based_state_pod(workDir, module, scenario, \
                                        setId, trainDirs, fomMeshPath)
      logging.info("")
      compute_partition_based_rhs_pod(workDir, module, scenario, \
                                      setId, trainDirs, fomMeshPath)
  else:
    logging.info("Nothing to do here")
