
import numpy as np
import sys, os
from .myio import load_basis_from_binary_file

# -------------------------------------------------------------------
def make_od_rom_initial_condition(workDir, appObjForIc, \
                                  partitionInfoDir, \
                                  basesDir, modesPerTileDic, \
                                  romSizeOverAllPartitions, \
                                  usingIcAsRefState):

  if usingIcAsRefState:
    # dont need to do projection, romState is simply all zeros
    return np.zeros(romSizeOverAllPartitions)
  else:
    nTiles = len(modesPerTileDic.keys())
    fomIc  = appObjForIc.initialCondition()
    romState = np.zeros(romSizeOverAllPartitions)
    romStateSpanStart = 0
    for tileId in range(nTiles):
      myK             = modesPerTileDic[tileId]
      myPhi           = load_basis_from_binary_file(basesDir+"/lsv_state_p_"+str(tileId))[:,0:myK]
      myStateRowsFile = partitionInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
      myStateRows     = np.loadtxt(myStateRowsFile, dtype=int)
      myFomIcSlice    = fomIc[myStateRows]
      tmpyhat         = np.dot(myPhi.transpose(), myFomIcSlice)
      romState[romStateSpanStart:romStateSpanStart+myK] = np.copy(tmpyhat)
      romStateSpanStart += myK
    return romState
