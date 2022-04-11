
import numpy as np
import sys, os, time
from scipy import linalg as scipyla

from .myio import load_basis_from_binary_file

# -------------------------------------------------------------------
def compute_phi_on_stencil(outDir, partitionInfoDir, \
                           statePodDir, sampleMeshDir, \
                           modesPerTileDic, numDofsPerCell):

  nTiles = len(modesPerTileDic)
  maxNumRows = 0
  rowsPerTile = []
  for tileId in range(nTiles):
    myNumModes = modesPerTileDic[tileId]

    # load my full phi
    myPhiFile = statePodDir + "/lsv_state_p_" + str(tileId)
    myPhi     = load_basis_from_binary_file(myPhiFile)[:,0:myNumModes]

    # load indices such that we can extract phi on stencil mesh
    myCellGids   = np.loadtxt(partitionInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt",dtype=int)
    myStMeshGids = np.loadtxt(sampleMeshDir + "/stencil_mesh_gids_p_"+str(tileId)+".dat", dtype=int)
    myStCount    = len(myStMeshGids)

    commonElem  = set(myStMeshGids).intersection(myCellGids)
    commonElem  = np.sort(list(commonElem))
    mylocalinds = np.searchsorted(myCellGids, commonElem)
    mySlicedPhi = np.zeros((myStCount*numDofsPerCell, myNumModes), order='F')
    for j in range(numDofsPerCell):
      mySlicedPhi[j::numDofsPerCell, :] = myPhi[numDofsPerCell*mylocalinds + j, :]

    maxNumRows = max(maxNumRows, mySlicedPhi.shape[0])
    rowsPerTile.append(mySlicedPhi.shape[0])
    np.savetxt(outDir+'/phi_on_stencil_p_'+str(tileId)+'.txt', mySlicedPhi)

  np.savetxt(outDir+'/max_num_rows.txt', np.array([int(maxNumRows)]), fmt="%6d")
  np.savetxt(outDir+'/rows_per_tile.txt', np.array(rowsPerTile), fmt="%6d")
