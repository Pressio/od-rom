
import numpy as np
import sys, os, re, yaml, logging

from .fncs_directory_naming import \
  string_identifier_from_partition_info_dir

from .fncs_cumulative_energy import \
  compute_cumulative_energy

# ----------------------------------------------------------------
def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

# ----------------------------------------------------------------
def get_run_id(runDir):
  return int(runDir.split('_')[-1])

# ----------------------------------------------------------------
def find_full_mesh_and_ensure_unique(workDir):
  # This is because the mesh is specified
  # via command line argument and must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory
  logger = logging.getLogger(__name__)

  fomFullMeshes = [workDir+'/'+d for d in os.listdir(workDir) \
                   # we need to find only dirs that BEGIN with
                   # this string otherwise we pick up other things
                   if "full_mesh" == os.path.basename(d)[0:9]]
  if len(fomFullMeshes) != 1:
    em = "Error: I found multiple full meshes:\n"
    for it in fomFullMeshes:
      em += it + "\n"
    em += "inside the workDir = {} \n".format(workDir)
    em += "You can only have a single FULL mesh the working directory."
    logger.error(em)
    sys.exit(1)
  return fomFullMeshes[0]

# -------------------------------------------------------------------
def find_all_partitions_info_dirs(workDir):
  # To do this, we find in workDir all directories with info about partitions
  # which identifies all possible partitions
  partsInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "od_info_" in d]
  return partsInfoDirs

# -------------------------------------------------------------------
def find_all_sample_meshes_for_target_partition_info(workDir, partInfoDir):
  partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDir)
  stringToFind = "partition_based_" + partitionStringIdentifier + "_sample_mesh"
  sampleMeshesDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                      if stringToFind in d]
  return sampleMeshesDirs

# -------------------------------------------------------------------
def replicate_bases_for_multiple_dofs(M, numDofsPerCell):
  logger = logging.getLogger(__name__)

  if numDofsPerCell == 1:
    return M

  elif numDofsPerCell == 2:
    K1 = np.shape(M)[1]
    K2 = np.shape(M)[1]
    Phi = np.zeros((M.shape[0]*numDofsPerCell, K1+K2))
    Phi[0::2,    0: K1]       = M
    Phi[1::2,   K1: K1+K2]    = M
    return Phi

  elif numDofsPerCell == 3:
    K1 = np.shape(M)[1]
    K2 = np.shape(M)[1]
    K3 = np.shape(M)[1]
    Phi = np.zeros((M.shape[0]*numDofsPerCell, K1+K2+K3))
    Phi[0::3,    0: K1]       = M
    Phi[1::3,   K1: K1+K2]    = M
    Phi[2::3,K1+K2: K1+K2+K3] = M
    return Phi

  elif numDofsPerCell == 4:
    K1 = np.shape(M)[1]
    K2 = np.shape(M)[1]
    K3 = np.shape(M)[1]
    K4 = np.shape(M)[1]
    Phi = np.zeros((M.shape[0]*numDofsPerCell, K1+K2+K3+K4))
    Phi[0::4,       0: K1]       = M
    Phi[1::4,     K1 : K1+K2]    = M
    Phi[2::4,   K1+K2: K1+K2+K3] = M
    Phi[3::4,K1+K2+K3: K1+K2+K3+K4] = M
    return Phi

  else:
    logger.error("replicate_bases_for_multiple_dofs: invalid numDofsPerCell")
    sys.exit(1)


# ----------------------------------------------------------------
def make_modes_per_tile_dic_with_const_modes_count(nTiles, modeCount):
  modesPerTileDic = {}
  for iTile in range(nTiles):
    modesPerTileDic[iTile] = modeCount
  return modesPerTileDic

# -------------------------------------------------------------------
def compute_total_modes_across_all_tiles(modesPerTileDic):
  return np.sum(list(modesPerTileDic.values()))

# -------------------------------------------------------------------
def find_modes_for_full_domain_from_target_energy(module, scenario, podDir, energy):
  singValues = np.loadtxt(podDir+'/sva_state_p_0')
  return compute_cumulative_energy(singValues, energy)


# -------------------------------------------------------------------
def find_modes_per_tile_from_target_energy(module, scenario, podDir, energy):
  def get_tile_id(stringIn):
    return int(stringIn.split('_')[-1])

  modesPerTileDic = {}
  # find all sing values files
  singValsFiles = [podDir+'/'+f for f in os.listdir(podDir) \
                   if "sva_state" in f]
  # sort based on the tile id
  singValsFiles = sorted(singValsFiles, key=get_tile_id)

  for it in singValsFiles:
    singValues = np.loadtxt(it)
    tileId = get_tile_id(it)
    K = compute_cumulative_energy(singValues, energy)
    modesPerTileDic[tileId] = max(K, module.odrom_min_num_modes_per_tile[scenario])

  return modesPerTileDic
