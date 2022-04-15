
import numpy as np
import sys, os, time, logging
from scipy import linalg as scipyla

from .fncs_myio import load_basis_from_binary_file

# -------------------------------------------------------------------
def compute_gappy_projector_using_factor_of_state_pod_modes(outDir, partitionInfoDir, \
                                                            statePodDir, rhsPodDir, \
                                                            sampleMeshDir, \
                                                            modesPerTileDic, \
                                                            numDofsPerCell):

  logger = logging.getLogger(__name__)

  nTiles = len(modesPerTileDic)
  maxNumRows = 0
  rowsPerTile = []
  for tileId in range(nTiles):
    myNumStatePodModes = modesPerTileDic[tileId]

    # load my full phi
    myFullPhiFile = statePodDir + "/lsv_state_p_" + str(tileId)
    myFullPhi     = load_basis_from_binary_file(myFullPhiFile)[:,0:myNumStatePodModes]

    # indexing info
    myFile1      = partitionInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myCellGids   = np.loadtxt(myFile1, dtype=int)
    myFile2      = sampleMeshDir + "/sample_mesh_gids_p_"+str(tileId)+".txt"
    mySmMeshGids = np.loadtxt(myFile2, dtype=int)
    mySmCount    = len(mySmMeshGids)

    K = myNumStatePodModes*3 + 1
    logger.debug("tile::: {} {} {}".format(K, myNumStatePodModes, mySmCount))
    if mySmCount*numDofsPerCell < K:
      logger.debug("Cannot have K > mySmCount*numDofsPerCell in tileId = {:>5}, adapting K".format(tileId))
      K = mySmCount*numDofsPerCell - 1

    # K should be larger than myNumStatePodModes
    if K < myNumStatePodModes:
      logger.debug("Cannot have K < myNumStatePodModes in tileId = {:>5}, adapting K".format(tileId))
      K = myNumStatePodModes + 1

    myFullRhsPodFile = rhsPodDir + "/lsv_rhs_p_" + str(tileId)
    myTheta = load_basis_from_binary_file(myFullRhsPodFile)[:,0:K]

    # need to slice pod of rhs (i.e. theta) to get elements on my sample mesh cells
    # note that I need to do the following beacuse the POD modes saved are computed
    # using the FOM rhs local data, which is not in the same order as the indexing
    # for the odrom, since the indexing within each tile has changed.
    # So I cannot just a get a contiguous subview of the theta matrix, but
    # I need to do a bit more maninulation to figure out which row indices to get
    commonElem = set(mySmMeshGids).intersection(myCellGids)
    commonElem = np.sort(list(commonElem))
    mylocalinds = np.searchsorted(myCellGids, commonElem)
    mySlicedTheta = np.zeros((mySmCount*numDofsPerCell, myTheta.shape[1]), order='F')
    for j in range(numDofsPerCell):
      mySlicedTheta[j::numDofsPerCell, :] = myTheta[numDofsPerCell*mylocalinds + j, :]

    A = myFullPhi.transpose() @ myTheta
    projector = A @ scipyla.pinv(mySlicedTheta)
    logger.debug(" tileId = {:>5}, projectorShape = {}".format(tileId, projector.T.shape))

    # write to file
    # here when writing w need to consider that project above is computed
    # such that it is short and wide, so to write it to file we need to
    # "view" it as flipped. the actual num rows is the cols and vice versa.
    numRows = np.int64(projector.shape[1])
    numCols = np.int64(projector.shape[0])
    fileo = open(outDir+'/projector_p_'+str(tileId), "wb")
    np.array([numRows]).tofile(fileo)
    np.array([numCols]).tofile(fileo)
    projector.tofile(fileo)
    fileo.close()
    np.savetxt(outDir+'/projector_p_'+str(tileId)+'.txt', projector.T)
    maxNumRows = max(maxNumRows, projector.T.shape[0])
    rowsPerTile.append(projector.T.shape[0])

  np.savetxt(outDir+'/max_num_rows.txt', np.array([int(maxNumRows)]), fmt="%6d")
  np.savetxt(outDir+'/rows_per_tile.txt', np.array(rowsPerTile), fmt="%6d")

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
def compute_quad_projector(fomTrainDirs, fomMesh, outDir, \
                           partitionInfoDir, statePodDir, \
                           sampleMeshDir, modesPerTileDic, \
                           numDofsPerCell):

  logger = logging.getLogger(__name__)

  # load f snapshots
  fomTotCells      = find_total_cells_from_info_file(fomMesh)
  totFomDofs       = fomTotCells*module.numDofsPerCell
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
