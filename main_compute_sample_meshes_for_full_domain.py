
# standard modules
from argparse import ArgumentParser
import sys, os, importlib
import re, time, yaml, random, subprocess
import numpy as np
from scipy import linalg as scipyla
from scipy import optimize as sciop

# try:
#   import pressiotools.linalg as ptla
#   from pressiotools.samplemesh.withLeverageScores import computeNodes
# except ImportError:
#   raise ImportError("Unable to import classes from pressiotools")

from py_src.banners_and_prints import *

from py_src.miscellanea import \
  find_full_mesh_and_ensure_unique

from py_src.myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir, \
  load_fom_rhs_snapshot_matrix, \
  load_basis_from_binary_file

from py_src.directory_naming import \
  path_to_full_domain_sample_mesh_random, \
  path_to_full_domain_sample_mesh_psampling,\
  path_to_full_domain_rhs_pod_data_dir

from py_src.mesh_info_file_extractors import *

# -------------------------------------------------------------------
def compute_sample_mesh_random_full_domain(workDir, module, scenario, pdaDir, fomMeshPath):
  # get list of RANDOM sample mesh cases from module
  sampleMeshesList = [it for it in module.sample_meshes[scenario]\
                      if "random" in it]
  print(sampleMeshesList)

  for sampleMeshCaseIt in sampleMeshesList:
    fractionOfCellsNeeded = sampleMeshCaseIt[1]

    # create name of directory where to store the sample mesh
    outDir = path_to_full_domain_sample_mesh_random(workDir, fractionOfCellsNeeded)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print('Generating RANDOM sample mesh in {}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      fomNumCells = find_total_cells_from_info_file(fomMeshPath)
      sampleMeshCount = int(fomNumCells * fractionOfCellsNeeded)
      sample_mesh_gids = random.sample(range(0, fomNumCells), sampleMeshCount)
      sample_mesh_gids = np.sort(sample_mesh_gids)
      print(" numCellsFullDomain = ", fomNumCells)
      print(" sampleMeshSize     = ", sampleMeshCount)
      np.savetxt(outDir+'/sample_mesh_gids_p_0.txt', sample_mesh_gids, fmt='%8i')

      print('Generating sample mesh in:')
      print(' {}'.format(outDir))
      owd = os.getcwd()
      meshScriptsDir = pdaDir + "/meshing_scripts"
      args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
              "--fullMeshDir", fomMeshPath,
              "--sampleMeshIndices", outDir+'/sample_mesh_gids_p_0.txt',
              "--outDir", outDir)
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      print(output)


# -------------------------------------------------------------------
def compute_sample_mesh_psampling_full_domain(workDir, module, scenario, pdaDir, fomMeshPath):
  # get list of sample mesh cases, filter only those having "psampling" in it
  sampleMeshesList = [it for it in module.sample_meshes[scenario]\
                      if "psampling" in it]
  print(sampleMeshesList)

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
        print('{} already exists'.format(outDir))
      else:
        print('Generating psampling sample mesh in: \n {}'.format(outDir))
        os.system('mkdir -p ' + outDir)

        fomNumCells = find_total_cells_from_info_file(fomMeshPath)
        mySampleMeshCount = int(fomNumCells * fractionOfCellsNeeded)
        rhsPodFile = currRhsPodDir + "/lsv_rhs_p_0"
        myRhsPod = load_basis_from_binary_file(rhsPodFile)[whichDofToUseForFindingCells::module.numDofsPerCell]
        if myRhsPod.shape[1] < mySampleMeshCount:
          print("Warning: psampling sample mesh for full domain")
          print("         not enough rhs modes, automatically reducing sample mesh count")
          mySampleMeshCount = myRhsPod.shape[1]-1

        Q,R,P = scipyla.qr(myRhsPod[:,0:mySampleMeshCount].transpose(), pivoting=True)
        mySampleMeshGidsWrtFullMesh = np.sort(P[0:mySampleMeshCount])
        np.savetxt(outDir+'/sample_mesh_gids_p_0.txt',\
                   mySampleMeshGidsWrtFullMesh, fmt='%8i')

        print('Generating sample mesh in:')
        print(' {}'.format(outDir))
        owd = os.getcwd()
        meshScriptsDir = pdaDir + "/meshing_scripts"
        args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
                "--fullMeshDir", fomMeshPath,
                "--sampleMeshIndices", outDir+'/sample_mesh_gids_p_0.txt',
                "--outDir", outDir)
        popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
        popen.wait()
        output = popen.stdout.read();
        print(output)


#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  banner_driving_script_info(os.path.basename(__file__))

  parser   = ArgumentParser()
  parser.add_argument("--wdir", dest="workdir", required=True)
  parser.add_argument("--pdadir", dest="pdadir", required=True)
  args     = parser.parse_args()
  workDir  = args.workdir
  pdaDir   = args.pdadir

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

    if any(["psampling" in it for it in sampleMeshesList]):
      compute_sample_mesh_psampling_full_domain(workDir, module, scenario, pdaDir, fomMeshPath)
  else:
    print("Nothing to do here")
  print("")
