
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, subprocess, logging
import random
import numpy as np
from scipy import linalg as scipyla

from py_src.fncs_banners_and_prints import *

from py_src.fncs_miscellanea import \
  find_full_mesh_and_ensure_unique,\
  get_run_id, \
  find_all_partitions_info_dirs

from py_src.fncs_myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir,\
  load_fom_state_snapshot_matrix, \
  load_fom_rhs_snapshot_matrix, \
  load_basis_from_binary_file

from py_src.fncs_directory_naming import \
  path_to_partition_info_dir, \
  path_to_od_sample_mesh_random, \
  path_to_od_sample_mesh_psampling, \
  path_to_rhs_pod_data_dir, \
  string_identifier_from_partition_info_dir

from py_src.fncs_fom_run_dirs_detection import \
  find_fom_train_dirs_for_target_set_of_indices

from py_src.fncs_to_extract_from_mesh_info_file import *

from py_src.fncs_svd import do_svd_py


# -------------------------------------------------------------------
def process_partitions_sample_mesh_files(pdaDir, fomMeshPath, sampleMeshDir, \
                                         partInfoDir, nTiles):
   logging.info('Generating sample mesh in:')
   logging.info(' {}'.format(sampleMeshDir))
   owd = os.getcwd()
   meshScriptsDir = pdaDir + "/meshing_scripts"
   args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
           "--fullMeshDir", fomMeshPath,
           "--sampleMeshIndices", sampleMeshDir+'/sample_mesh_gids.dat',
           "--outDir", sampleMeshDir+"/pda_sm",
           "--useTilingFrom", partInfoDir)
   popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
   popen.wait()
   output = popen.stdout.read();
   logging.info(output)

   # copy from sampleMeshDir/sm the generated stencil mesh gids file
   args = ("cp", sampleMeshDir+"/pda_sm/stencil_mesh_gids.dat", sampleMeshDir+"/stencil_mesh_gids.dat")
   popen  = subprocess.Popen(args, stdout=subprocess.PIPE); popen.wait()
   output = popen.stdout.read();
   logging.info(output)

   # now we can also figure out the stencil gids for each tile
   stencilGids = np.loadtxt(sampleMeshDir+"/pda_sm/stencil_mesh_gids.dat", dtype=int)
   count = []
   for tileId in range(nTiles):
     myFile     = partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
     myCellGids = np.loadtxt(myFile, dtype=int)
     commonElem = set(stencilGids).intersection(myCellGids)
     commonElem = np.sort(list(commonElem))
     np.savetxt(sampleMeshDir+'/stencil_mesh_gids_p_'+str(tileId)+'.dat', commonElem, fmt='%8i')
     count.append(len(commonElem))

   np.savetxt(sampleMeshDir+'/count_stm.txt', np.array(count), fmt='%8i')


# -------------------------------------------------------------------
def compute_sample_mesh_random_od(workDir, module, scenario, pdaDir, fomMeshPath):
  # get list of RANDOM sample mesh cases from module
  sampleMeshesList = [it for it in module.sample_meshes[scenario]\
                      if "random" in it]
  logging.info(sampleMeshesList)

  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in find_all_partitions_info_dirs(workDir):
    nTiles = np.loadtxt(partInfoDirIt+"/ntiles.txt", dtype=int)
    #nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    #nTiles = nTilesX*nTilesY
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # -------
    # loop 2: over all target sample mesh cases
    # ------
    for sampleMeshCaseIt in sampleMeshesList:
      # extract the fraction, which can be in position 0 or 1
      # so check which one is a string and pick the other
      fractionOfCellsNeeded = \
        sampleMeshCaseIt[0] if isinstance(sampleMeshCaseIt[1], str) \
        else sampleMeshCaseIt[1]

      # create name of directory where to store the sample mesh
      outDir = path_to_od_sample_mesh_random(workDir,\
                                             partitionStringIdentifier, \
                                             fractionOfCellsNeeded)
      if os.path.exists(outDir):
        logging.info('{} already exists'.format(outDir))
      else:
        logging.info('Generating RANDOM od sample mesh in {}'.format(outDir))
        os.system('mkdir -p ' + outDir)

        # loop over tiles
        global_sample_mesh_gids = []
        for tileId in range(nTiles):
          # figure out how many local sample mesh cells
          myFile     = partInfoDirIt + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
          myCellGids = np.loadtxt(myFile, dtype=int)
          myNumCells = len(myCellGids)
          mySampleMeshCount = int(myNumCells * fractionOfCellsNeeded)
          logging.debug(" tileId = {:>5}, myNumCells = {:>5}, mySmCount = {:>5}".format(tileId, \
                                                                                myNumCells, \
                                                                                mySampleMeshCount))


          smCellsIndices = random.sample(range(0, myNumCells), mySampleMeshCount)
          mylocalids = np.sort(smCellsIndices)
          mySampleMeshGidsWrtFullMesh = myCellGids[mylocalids]

          # add to sample mesh global list of gids
          global_sample_mesh_gids += mySampleMeshGidsWrtFullMesh.tolist()
          np.savetxt(outDir+'/sample_mesh_gids_p_'+str(tileId)+'.txt',\
                     mySampleMeshGidsWrtFullMesh, fmt='%8i')

        # now we can write to file the gids of the sample mesh cells over entire domain
        global_sample_mesh_gids = np.sort(global_sample_mesh_gids)
        np.savetxt(outDir+'/sample_mesh_gids.dat', global_sample_mesh_gids, fmt='%8i')

        process_partitions_sample_mesh_files(pdaDir, fomMeshPath, \
                                             outDir, partInfoDirIt, nTiles)


# -------------------------------------------------------------------
def compute_sample_mesh_psampling_od(workDir, module, scenario, pdaDir, fomMeshPath):

  # get list of sample mesh cases, filter only those having "psampling" in it
  sampleMeshesList = [it for it in module.sample_meshes[scenario]\
                      if "psampling" in it]
  logging.info(sampleMeshesList)

  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in find_all_partitions_info_dirs(workDir):
    nTiles = np.loadtxt(partInfoDirIt+"/ntiles.txt", dtype=int)
    #nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    #nTiles = nTilesX*nTilesY
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # -------
    # loop 2: over all setIds
    # ------
    howManySets = len(module.basis_sets[scenario].keys())
    for setId in range(howManySets):
      # for psampling I need to get the rhs modes
      currRhsPodDir = path_to_rhs_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3: over all target sample mesh cases
      # ------
      for sampleMeshCaseIt in sampleMeshesList:
        # extract the fraction, which must be in position 1
        assert(not isinstance(sampleMeshCaseIt[1], str))
        fractionNeeded = sampleMeshCaseIt[1]

        # for psampling sample mesh, I need to use a certain dof/variable
        # for exmaple for swe, there is h,u,v so I need to know which one
        # to use to find the cells
        # this info is provided in the problem
        assert(isinstance(sampleMeshCaseIt[2], int))
        whichDofToUseForFindingCells = sampleMeshCaseIt[2]

        # name of where to generate files
        outDir = path_to_od_sample_mesh_psampling(workDir, partitionStringIdentifier, \
                                                  setId, fractionNeeded, \
                                                  whichDofToUseForFindingCells)
        if os.path.exists(outDir):
          logging.info('{} already exists'.format(outDir))
        else:
          logging.info('Generating psampling OD sample mesh in: \n {}'.format(outDir))
          os.system('mkdir -p ' + outDir)

          # loop over tiles
          tmp = []
          global_sample_mesh_gids = []
          for tileId in range(nTiles):
            # figure out how many local sample mesh cells
            myFile     = partInfoDirIt + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
            myCellGids = np.loadtxt(myFile, dtype=int)
            myNumCells = len(myCellGids)
            mySampleMeshCount = int(myNumCells * fractionNeeded)
            logging.debug(" tileId = {:>5}, myNumCells = {:>5}, mySmCount = {:>5}".format(tileId, \
                                                                                  myNumCells, \
                                                                                  mySampleMeshCount))

            myRhsPodFile = currRhsPodDir + "/lsv_rhs_p_" + str(tileId)
            myRhsPod = load_basis_from_binary_file(myRhsPodFile)[whichDofToUseForFindingCells::module.numDofsPerCell]
            if myRhsPod.shape[1] < mySampleMeshCount:
              s1 = "Warning: psampling in tileId = {:>5}:".format(tileId)
              s1 += "not enough RHS modes, automatically reducing sample mesh count"
              logging.debug(s1)
              mySampleMeshCount = myRhsPod.shape[1]-1

            Q,R,P = scipyla.qr(myRhsPod[:,0:mySampleMeshCount].transpose(), pivoting=True)
            mylocalids = np.array(np.sort(P[0:mySampleMeshCount]))
            mySampleMeshGidsWrtFullMesh = myCellGids[mylocalids]

            # add to sample mesh global list of gids
            global_sample_mesh_gids += mySampleMeshGidsWrtFullMesh.tolist()
            np.savetxt(outDir+'/sample_mesh_gids_p_'+str(tileId)+'.txt',\
                       mySampleMeshGidsWrtFullMesh, fmt='%8i')
            tmp.append(len(mySampleMeshGidsWrtFullMesh))

          np.savetxt(outDir+'/count_sm.txt', np.array(tmp), fmt='%8i')

          # now we can write to file the gids of the sample mesh cells over entire domain
          global_sample_mesh_gids = np.sort(global_sample_mesh_gids)
          np.savetxt(outDir+'/sample_mesh_gids.dat', global_sample_mesh_gids, fmt='%8i')

          process_partitions_sample_mesh_files(pdaDir, fomMeshPath, \
                                               outDir, partInfoDirIt, nTiles)

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
  parser.add_argument("--pdadir", dest="pdadir", required=True)
  args     = parser.parse_args()
  workDir  = args.workdir
  pdaDir   = args.pdadir

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

  # before we move on, we need to ensure that in workDir
  # there is a unique FULL mesh. This is because the mesh is specified
  # via command line argument and must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory
  fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

  banner_make_sample_meshes_all_partitions()

  matchers = ["Gappy", "Quad"]
  matching = [s for s in module.algos[scenario] if any(xs in s for xs in matchers)]
  if matching:
    sampleMeshesList = module.sample_meshes[scenario]

    if any(["random" in it for it in sampleMeshesList]):
      compute_sample_mesh_random_od(workDir, module, scenario, pdaDir, fomMeshPath)

    if any(["psampling" in it for it in sampleMeshesList]):
      compute_sample_mesh_psampling_od(workDir, module, scenario, pdaDir, fomMeshPath)

  else:
    logging.info("skipping")
  logging.info("")
