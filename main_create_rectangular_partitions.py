
import numpy as np
import os, sys, re

def find_num_cells_from_info_file(workDir, ns):
  reg = re.compile(r''+ns+'.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

def find_dimensionality_from_info_file(workDir):
  reg = re.compile(r'dim.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# def _ij_from_gid(gid):
#   j = int(gid)/int(nx)
#   i = gid % nx
#   return int(i), int(j)

def _gid_from_ij(i,j, nx_, ny_):
  return int( (j%ny_)*nx_ + i%nx_ )

def _mapCellGidsToStateDofsGids(cellGidsDic, numDofsPerCell):
  d = {}

  if numDofsPerCell == 1:
    for pCount, pCellGids in cellGidsDic.items():
      myl = [int(i) for i in pCellGids]
      d[pCount] = myl

  elif numDofsPerCell == 2:
    for pCount, pCellGids in cellGidsDic.items():
      myl = [None]*len(pCellGids)*2
      myl[0::2] = [int(i*2)   for i in pCellGids]
      myl[1::2] = [int(i*2+1) for i in pCellGids]
      d[pCount] = myl

  elif numDofsPerCell == 3:
    for pCount, pCellGids in cellGidsDic.items():
      myl = [None]*len(pCellGids)*3
      myl[0::3] = [int(i*3)   for i in pCellGids]
      myl[1::3] = [int(i*3+1) for i in pCellGids]
      myl[2::3] = [int(i*3+2) for i in pCellGids]
      d[pCount] = myl

  elif numDofsPerCell == 4:
    for pCount, pCellGids in cellGidsDic.items():
      myl = [None]*len(pCellGids)*4
      myl[0::4] = [int(i*4)   for i in pCellGids]
      myl[1::4] = [int(i*4+1) for i in pCellGids]
      myl[2::4] = [int(i*4+2) for i in pCellGids]
      myl[3::4] = [int(i*4+3) for i in pCellGids]
      d[pCount] = myl

  elif numDofsPerCell == 5:
    for pCount, pCellGids in cellGidsDic.items():
      myl = [None]*len(pCellGids)*5
      myl[0::5] = [int(i*5)   for i in pCellGids]
      myl[1::5] = [int(i*5+1) for i in pCellGids]
      myl[2::5] = [int(i*5+2) for i in pCellGids]
      myl[3::5] = [int(i*5+3) for i in pCellGids]
      myl[4::5] = [int(i*5+4) for i in pCellGids]
      d[pCount] = myl

  return d

# def _mapCellGidsToPieces(numCellsPerBlockX, numCellsPerBlockY, npX, npY, nx, ny):
#   d = {}
#   # loop over all gids and map gids to patch
#   for j in range(ny):
#     for i in range(nx):
#       blockI = int(i/numCellsPerBlockX)
#       blockJ = int(j/numCellsPerBlockY)
#       blockGid = _gid_from_ij(blockI, blockJ, npX, npY)
#       cellGid  = _gid_from_ij(i,j, nx, ny)
#       if blockGid in d:
#         d[blockGid].append(cellGid)
#       else:
#         d[blockGid] = [cellGid]
#   return d

def _create1d_new(totCellsX, xSplits, ndpc):
  blockSizes = {}
  d = {}
  blockGid = 0
  for xgids in xSplits:
    for i in xgids:
      cellGid  = i
      if blockGid in d:
        d[blockGid].append(cellGid)
      else:
        d[blockGid] = [cellGid]
    blockSizes[blockGid] = [len(xgids)]
    blockGid += 1

  stateDofsGidsDic = _mapCellGidsToStateDofsGids(d, ndpc)
  return [blockSizes, d, stateDofsGidsDic]

def _create2d_new(totCellsX, totCellsY, xSplits, ySplits, ndpc):
  blockSizes = {}
  d = {}
  blockGid = 0
  for ygids in ySplits:
    for xgids in xSplits:
      for j in ygids:
        for i in xgids:
          cellGid  = _gid_from_ij(i,j, totCellsX, totCellsY)
          if blockGid in d:
            d[blockGid].append(cellGid)
          else:
            d[blockGid] = [cellGid]
      blockSizes[blockGid] = [len(xgids), len(ygids)]
      blockGid += 1

  stateDofsGidsDic = _mapCellGidsToStateDofsGids(d, ndpc)
  return [blockSizes, d, stateDofsGidsDic]


#----------------------------
if __name__ == '__main__':
#----------------------------
  from argparse import ArgumentParser
  parser   = ArgumentParser()
  parser.add_argument("--wdir", "--workdir",
                      dest="workDir",
                      required=True)
  parser.add_argument("--meshPath",
                      dest="meshPath", \
                      required=True)
  parser.add_argument("--tiles",
                      nargs='+',
                      dest="tiles",
                      type=int,
                      required=True)
  parser.add_argument("--ndpc",
                      dest="numDofsPerCell",
                      type=int,
                      required=True)

  args     = parser.parse_args()
  workDir  = args.workDir
  meshPath = args.meshPath
  ndpc     = args.numDofsPerCell

  if not os.path.exists(workDir):
    os.system('mkdir -p ' + workDir)

  meshDim = find_dimensionality_from_info_file(meshPath)

  # ----------------------
  if meshDim == 1:
  # ----------------------
    assert(len(args.tiles)==1)
    nx = find_num_cells_from_info_file(meshPath, "nx")
    nTilesX = args.tiles[0]
    totTiles = nTilesX
    xSplits = np.array_split(np.arange(nx), nTilesX)

    blockSizes, cellGidsDic, stateDofsGidsDic = _create1d_new(nx, xSplits, ndpc)

    np.savetxt(workDir+"/ntiles.txt", \
               np.array([nTilesX]), fmt='%8d')

    for k in range(totTiles):
      np.savetxt(workDir+"/block_size_p_"+str(k)+".txt", \
                 np.array(blockSizes[k]), fmt='%8d')

    for k,v in cellGidsDic.items():
      np.savetxt(workDir+"/cell_gids_wrt_full_mesh_p_"+str(k)+".txt", \
                 np.array(v), fmt='%8d')

    for k,v in stateDofsGidsDic.items():
      np.savetxt(workDir+"/state_vec_rows_wrt_full_mesh_p_"+str(k)+".txt", \
                 np.array(v), fmt='%8d')

  # ----------------------
  elif meshDim == 2:
  # ----------------------
    assert(len(args.tiles)==2)
    nx = find_num_cells_from_info_file(meshPath, "nx")
    ny = find_num_cells_from_info_file(meshPath, "ny")

    nTilesX = args.tiles[0]
    nTilesY = args.tiles[1]
    totTiles = nTilesX*nTilesY

    xSplits = np.array_split(np.arange(nx), nTilesX)
    ySplits = np.array_split(np.arange(ny), nTilesY)

    blockSizes, cellGidsDic, stateDofsGidsDic = _create2d_new(nx, ny, \
                                                              xSplits, ySplits, \
                                                              ndpc)

    np.savetxt(workDir+"/ntiles.txt", \
               np.array([nTilesX*nTilesY]), fmt='%8d')

    for k in range(totTiles):
      np.savetxt(workDir+"/block_size_p_"+str(k)+".txt", \
                 np.array(blockSizes[k]), fmt='%8d')

    for k,v in cellGidsDic.items():
      np.savetxt(workDir+"/cell_gids_wrt_full_mesh_p_"+str(k)+".txt", \
                 np.array(v), fmt='%8d')

    for k,v in stateDofsGidsDic.items():
      np.savetxt(workDir+"/state_vec_rows_wrt_full_mesh_p_"+str(k)+".txt", \
                 np.array(v), fmt='%8d')

  # ----------------------
  else:
  # ----------------------
    sys.exit("invalid meshDim = {}".format(meshDim))
