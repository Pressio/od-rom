
# standard modules
from argparse import ArgumentParser
import sys, os, importlib
import re, time, yaml, random, subprocess, logging
import numpy as np
from scipy import linalg as scipyla
from scipy import optimize as sciop

# try:
#   import pressiotools.linalg as ptla
#   from pressiotools.samplemesh.withLeverageScores import computeNodes
# except ImportError:
#   raise ImportError("Unable to import classes from pressiotools")

from py_src.fncs_banners_and_prints import *

from py_src.fncs_miscellanea import \
  find_full_mesh_and_ensure_unique

from py_src.fncs_myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir, \
  load_fom_rhs_snapshot_matrix, \
  load_basis_from_binary_file

from py_src.fncs_directory_naming import \
  path_to_full_domain_sample_mesh_random, \
  path_to_full_domain_sample_mesh_psampling,\
  path_to_full_domain_rhs_pod_data_dir

from py_src.fncs_to_extract_from_mesh_info_file import *

# -------------------------------------------------------------------
def compute_sample_mesh_random_full_domain(workDir, module, scenario, pdaDir, fomMeshPath):
  # get list of RANDOM sample mesh cases from module
  sampleMeshesList = [it for it in module.sample_meshes[scenario]\
                      if "random" in it]
  logging.debug(sampleMeshesList)

  for sampleMeshCaseIt in sampleMeshesList:
    fractionOfCellsNeeded = sampleMeshCaseIt[1]

    # create name of directory where to store the sample mesh
    outDir = path_to_full_domain_sample_mesh_random(workDir, fractionOfCellsNeeded)
    if os.path.exists(outDir):
      logging.info('{} already exists'.format(outDir))
    else:
      logging.info('Generating random sample mesh in {}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      fomNumCells = find_total_cells_from_info_file(fomMeshPath)
      sampleMeshCount = int(fomNumCells * fractionOfCellsNeeded)
      sample_mesh_gids = random.sample(range(0, fomNumCells), sampleMeshCount)
      sample_mesh_gids = np.sort(sample_mesh_gids)
      logging.debug(" numCellsFullDomain = {}".format(fomNumCells))
      logging.debug(" sampleMeshSize     = {}".format(sampleMeshCount))
      np.savetxt(outDir+'/sample_mesh_gids_p_0.txt', sample_mesh_gids, fmt='%8i')

      owd = os.getcwd()
      meshScriptsDir = pdaDir + "/meshing_scripts"
      args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
              "--fullMeshDir", fomMeshPath,
              "--sampleMeshIndices", outDir+'/sample_mesh_gids_p_0.txt',
              "--outDir", outDir)
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      logging.debug(output)


# -------------------------------------------------------------------
def compute_sample_mesh_psampling_full_domain(workDir, module, scenario, pdaDir, fomMeshPath):
  # get list of sample mesh cases, filter only those having "psampling" in it
  sampleMeshesList = [it for it in module.sample_meshes[scenario]\
                      if "psampling" in it]
  logging.debug(sampleMeshesList)

  # -------
  # loop 2: over all setIds
  # ------
  howManySets = len(module.basis_sets[scenario].keys())
  for setId in range(howManySets):
    # for psampling I need to get the rhs modes
    currRhsPodDir = path_to_full_domain_rhs_pod_data_dir(workDir, setId)

    # -------
    # loop 3: over all target sample mesh cases
    # ------
    for sampleMeshCaseIt in sampleMeshesList:
      # extract the fraction, which must be in position 1
      assert(not isinstance(sampleMeshCaseIt[1], str))
      fractionOfCellsNeeded = sampleMeshCaseIt[1]

      # for psampling sample mesh, I need to use a certain dof/variable
      # for exmaple for swe, there is h,u,v so I need to know which one
      # to use to find the cells
      # this info is provided in the problem
      assert(isinstance(sampleMeshCaseIt[2], int))
      whichDofToUseForFindingCells = sampleMeshCaseIt[2]

      # name of where to generate files
      outDir = path_to_full_domain_sample_mesh_psampling(workDir, setId, fractionOfCellsNeeded)
      if os.path.exists(outDir):
        logging.info('{} already exists'.format(outDir))
      else:
        logging.info('Generating psampling sample mesh in: \n {}'.format(outDir))
        os.system('mkdir -p ' + outDir)

        fomNumCells = find_total_cells_from_info_file(fomMeshPath)
        mySampleMeshCount = int(fomNumCells * fractionOfCellsNeeded)
        rhsPodFile = currRhsPodDir + "/lsv_rhs_p_0"
        myRhsPod = load_basis_from_binary_file(rhsPodFile)[whichDofToUseForFindingCells::module.numDofsPerCell]
        if myRhsPod.shape[1] < mySampleMeshCount:
          logging.warning("Warning: psampling sample mesh for full domain, not enough rhs modes, automatically reducing sample mesh count")
          mySampleMeshCount = myRhsPod.shape[1]-1

        Q,R,P = scipyla.qr(myRhsPod[:,0:mySampleMeshCount].transpose(), pivoting=True)
        mySampleMeshGidsWrtFullMesh = np.sort(P[0:mySampleMeshCount])
        np.savetxt(outDir+'/sample_mesh_gids_p_0.txt',\
                   mySampleMeshGidsWrtFullMesh, fmt='%8i')

        owd = os.getcwd()
        meshScriptsDir = pdaDir + "/meshing_scripts"
        args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
                "--fullMeshDir", fomMeshPath,
                "--sampleMeshIndices", outDir+'/sample_mesh_gids_p_0.txt',
                "--outDir", outDir)
        popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
        popen.wait()
        output = popen.stdout.read();
        logging.info(output)

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

  if "PodStandardGalerkinGappy" in module.algos[scenario]:

    # before we move on, we need to ensure that in workDir
    # there is a unique FULL mesh. This is because the mesh is specified
    # via command line argument and must be unique for a scenario.
    # If one wants to run for a different mesh, then they have to
    # run this script again with a different working directory
    fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

    banner_sample_mesh_full_domain()
    sampleMeshesList = module.sample_meshes[scenario]
    if any(["random" in it for it in sampleMeshesList]):
      compute_sample_mesh_random_full_domain(workDir, module, scenario, pdaDir, fomMeshPath)
      logging.info("")

    if any(["psampling" in it for it in sampleMeshesList]):
      compute_sample_mesh_psampling_full_domain(workDir, module, scenario, pdaDir, fomMeshPath)

  else:
    logging.info("Nothing to do here")
  logging.info("")
