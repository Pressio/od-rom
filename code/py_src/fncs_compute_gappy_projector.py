
import numpy as np
import sys, os, time, logging
from scipy import linalg as scipyla
from scipy import optimize as sciop

from .fncs_myio import load_basis_from_binary_file

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
