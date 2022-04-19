
import numpy as np
import sys, os, time, logging
from scipy import linalg as scipyla
from scipy import optimize as sciop

from .fncs_myio import \
  load_basis_from_binary_file, \
  load_fom_rhs_snapshot_matrix

from .fncs_to_extract_from_mesh_info_file import \
  find_total_cells_from_info_file

# -------------------------------------------------------------------
def compute_quad_projector_single_tile(fomTrainDirs, fomMesh, outDir, \
                                       partitionInfoDir, statePodDir, \
                                       sampleMeshDir, modesPerTileDic, \
                                       numDofsPerCell):
  logger = logging.getLogger(__name__)

  fomTotCells      = find_total_cells_from_info_file(fomMesh)
  totFomDofs       = fomTotCells*numDofsPerCell
  fSnapsFullDomain = load_fom_rhs_snapshot_matrix(fomTrainDirs, totFomDofs, numDofsPerCell)

  myNumModes = modesPerTileDic[0]

  # load my phi on full mesh
  myPhiFile     = statePodDir + "/lsv_state_p_0"
  myPhiFullMesh = load_basis_from_binary_file(myPhiFile)[:,0:myNumModes]

  mySmMeshGids  = np.loadtxt(sampleMeshDir + "/sample_mesh_gids_p_0.txt", dtype=int)
  mySmCount     = len(mySmMeshGids)
  logger.debug("required = {}".format(numDofsPerCell* mySmCount))
  logger.debug("snaps #  = {}".format(fSnapsFullDomain.shape[1]))
  assert( numDofsPerCell* mySmCount <= fSnapsFullDomain.shape[1] )

  # phi on sample mesh
  myPhiSampleMesh = np.zeros((mySmCount*numDofsPerCell, myPhiFullMesh.shape[1]), order='F')
  for j in range(numDofsPerCell):
    myPhiSampleMesh[j::numDofsPerCell, :] = myPhiFullMesh[numDofsPerCell*mySmMeshGids + j, :]

  # get rhs snaps on sample mesh
  myfSnapsSampleMesh = np.zeros((mySmCount*numDofsPerCell, fSnapsFullDomain.shape[1]), order='F')
  for j in range(numDofsPerCell):
    myfSnapsSampleMesh[j::numDofsPerCell, :] = fSnapsFullDomain[numDofsPerCell*mySmMeshGids + j, :]

  logger.debug("myPhiSampleMesh.shape = {}".format(myPhiSampleMesh.shape))
  logger.debug("myfSnapsSampleMesh.shape = {}".format(myfSnapsSampleMesh.shape))
  logger.debug("fSnapsFullDomain.shape = {}".format(fSnapsFullDomain.shape))

  # setup sequence of ls problem: minimize (Aw - b)
  # initialize weights (weights for each basis vector)
  W = np.zeros_like(myPhiSampleMesh)
  logger.debug(W.shape)
  for j in range(myPhiFullMesh.shape[1]):
    A = myPhiSampleMesh[:,j:j+1] * myfSnapsSampleMesh[:, :]
    logger.debug("A.shape = {}".format(A.shape))
    b = myPhiFullMesh[:,j].transpose() @ fSnapsFullDomain[:, :]
    logger.debug("b.shape = {}".format(b.shape))
    W[:,j], _ = sciop.nnls(A.T, b, maxiter=5000)

  mjop = myPhiSampleMesh * W
  # save mjop to file
  np.savetxt(outDir+'/projector_p_'+str(0)+'.txt', mjop)

# -------------------------------------------------------------------
def make_bases_on_sample_mesh(partInfoDir, tileId, sampleMeshPath, \
                              phiFullMesh, numDofsPerCell):

   myCellGids   = np.loadtxt(partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt",dtype=int)
   mySmMeshGids = np.loadtxt(sampleMeshPath + "/sample_mesh_gids_p_"+str(tileId)+".txt", dtype=int)
   mySmCount    = len(mySmMeshGids)

   commonElem  = set(mySmMeshGids).intersection(myCellGids)
   commonElem  = np.sort(list(commonElem))
   mylocalinds = np.searchsorted(myCellGids, commonElem)
   phiOnSampleMesh = np.zeros((mySmCount*numDofsPerCell, phiFullMesh.shape[1]), order='F')
   for j in range(numDofsPerCell):
     phiOnSampleMesh[j::numDofsPerCell, :] = phiFullMesh[numDofsPerCell*mylocalinds + j, :]

   return phiOnSampleMesh

# -------------------------------------------------------------------
def compute_quad_projector(fomTrainDirs, fomMesh, outDir, \
                           partitionInfoDir, statePodDir, \
                           sampleMeshDir, modesPerTileDic, \
                           numDofsPerCell):

  logger = logging.getLogger(__name__)

  # load f snapshots
  fomTotCells      = find_total_cells_from_info_file(fomMesh)
  totFomDofs       = fomTotCells*numDofsPerCell
  fSnapsFullDomain = load_fom_rhs_snapshot_matrix(fomTrainDirs, totFomDofs, numDofsPerCell)

  nTiles = len(modesPerTileDic)
  for tileId in range(nTiles):
    myNumModes = modesPerTileDic[tileId]

    # load my phi on full mesh
    myPhiFile     = statePodDir + "/lsv_state_p_" + str(tileId)
    myPhiFullMesh = load_basis_from_binary_file(myPhiFile)[:,0:myNumModes]

    # restrict on sample mesh
    myPhiSampleMesh = make_bases_on_sample_mesh(partitionInfoDir, tileId, \
                                                sampleMeshDir, myPhiFullMesh,\
                                                numDofsPerCell)
    assert(myPhiSampleMesh.shape[1] == myPhiFullMesh.shape[1])

    # indexing info
    cellGidsFile   = partitionInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myCellGids     = np.loadtxt(cellGidsFile, dtype=int)
    sampleGidsFile = sampleMeshDir + "/sample_mesh_gids_p_"+str(tileId)+".txt"
    mySmMeshGids   = np.loadtxt(sampleGidsFile, dtype=int)
    mySmCount      = len(mySmMeshGids)
    logger.debug(numDofsPerCell* mySmCount)
    logger.debug(fSnapsFullDomain.shape[1])

    # get rhs snaps on sample mesh
    rowsFile = partitionInfoDir + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
    myRowsInFullState = np.loadtxt(rowsFile, dtype=int)
    myRhsSnaps  = fSnapsFullDomain[myRowsInFullState, :]
    assert( numDofsPerCell* mySmCount <= myRhsSnaps.shape[1] )

    commonElem  = set(mySmMeshGids).intersection(myCellGids)
    commonElem  = np.sort(list(commonElem))
    mylocalinds = np.searchsorted(myCellGids, commonElem)
    myfSnapsSampleMesh = np.zeros((mySmCount*numDofsPerCell, myRhsSnaps.shape[1]), order='F')
    logger.debug(myfSnapsSampleMesh.shape)
    logger.debug(len(mylocalinds))
    for j in range(numDofsPerCell):
      myfSnapsSampleMesh[j::numDofsPerCell, :] = myRhsSnaps[numDofsPerCell*mylocalinds + j, :]

    # setup sequence of ls problem: minimize (Aw - b)
    # initialize weights (weights for each basis vector)
    W = np.zeros_like(myPhiSampleMesh)
    logger.debug(W.shape)

    numModes = myPhiFullMesh.shape[1]
    for j in range(numModes):
      A = myPhiSampleMesh[:,j:j+1] * myfSnapsSampleMesh[:,:]
      b = myPhiFullMesh[:,j].transpose() @ myRhsSnaps[:, :]
      W[:,j],_ = sciop.nnls(A.T, b, maxiter=5000)

    mjop = myPhiSampleMesh * W
    np.savetxt(outDir+'/projector_p_'+str(tileId)+'.txt', mjop)
