
# standard modules
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import re, time, yaml, random, subprocess
import numpy as np
from decimal import Decimal

from py_src.banners_and_prints import *

from py_src.miscellanea import \
  find_full_mesh_and_ensure_unique

from py_src.myio import \
  read_scenario_from_dir, \
  read_problem_name_from_dir

from py_src.directory_naming import \
  path_to_partition_info_dir

from py_src.mesh_info_file_extractors import *

# -------------------------------------------------------------------
def make_rectangular_uniform_partitions_1d(workDir, fullMeshPath, listOfTilings):
  '''
  tile a 1d mesh using uniform partitions if possible
  '''
  this_file_path = pathlib.Path(__file__).parent.absolute()

  for nTilesX in listOfTilings:
    outDir = path_to_partition_info_dir(workDir, nTilesX, 1, "rectangularUniform")
    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for \n{}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(this_file_path)+'/py_src/partition_uniform.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTilesX),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

# -------------------------------------------------------------------
def make_rectangular_uniform_partitions_2d(workDir, fullMeshPath, listOfTilings):
  '''
  tile a 2d mesh using uniform partitions if possible
  '''
  this_file_path = pathlib.Path(__file__).parent.absolute()

  for pIt in listOfTilings:
    nTilesX, nTilesY = pIt[0], pIt[1]

    outDir = path_to_partition_info_dir(workDir, nTilesX, nTilesY, "rectangularUniform")
    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for \n{}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(this_file_path)+'/py_src/partition_uniform.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTilesX), str(nTilesY),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

# -------------------------------------------------------------------
def make_concentric_uniform_partitions_2d(workDir, fullMeshPath, listOfTilings):
  this_file_path = pathlib.Path(__file__).parent.absolute()

  for nTiles in listOfTilings:
    outDir = path_to_partition_info_dir(workDir, nTiles, None, "concentricUniform")
    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for \n{}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(this_file_path)+'/py_src/partition_radial.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTiles),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

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

  triggers = ["PodOdGalerkin", \
              "PodOdProjectionError", \
              "PodOdGalerkinGappy", \
              "PodOdGalerkinGappyMasked", \
              "PodOdGalerkinQuad", \
              "LegendreOdGalerkinFull"]
  if any(x in triggers for x in module.algos[scenario]):
    banner_make_partitions()

    # before we move on, we need to ensure that in workDir
    # there is a unique FULL mesh. This is because the mesh is specified
    # via command line argument and must be unique for a scenario.
    # If one wants to run for a different mesh, then they have to
    # run this script again with a different working directory
    fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

    for key, val in module.odrom_partitions[scenario].items():
      style, listOfTilings = key, val

      if module.dimensionality == 1:
        assert(style == "rectangularUniform")
        make_rectangular_uniform_partitions_1d(workDir, fomMeshPath, listOfTilings)

      if module.dimensionality == 2 and \
         style == "rectangularUniform":
        make_rectangular_uniform_partitions_2d(workDir, fomMeshPath, listOfTilings)

      if module.dimensionality == 2 and \
         style == "concentricUniform":
        make_concentric_uniform_partitions_2d(workDir, fomMeshPath, listOfTilings)
  else:
    print("Nothing to do here")
  print("")
