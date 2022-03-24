
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

def _gid_from_ij(i,j, nx_, ny_):
  return int( (j%ny_)*nx_ + i%nx_ )

def _ij_from_gid(gid):
  j = int(gid)/int(nx)
  i = gid % nx
  return int(i), int(j)

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

  nx = find_num_cells_from_info_file(meshPath, "nx")
  ny = find_num_cells_from_info_file(meshPath, "ny")

  x = np.loadtxt(meshPath + "/coordinates.dat")[:,1]
  y = np.loadtxt(meshPath + "/coordinates.dat")[:,2]

  #warning: this has to be changed
  # for now constraint to square domains center around origin
  maxRadius = np.max(x)
  cellGidsDic = {}

  rs = [0]
  for i in range(1, args.tiles):
    # https://math.stackexchange.com/questions/270287/how-to-divide-a-circle-into-9-rings-1-inner-circle-with-the-same-area
    ri = maxRadius * np.sqrt(float(i)/args.tiles)
    rs.append(ri)
    cellGidsDic[i-1] = np.where((np.sqrt(x**2 + y**2) >= rs[i-1]) & (np.sqrt(x**2 + y**2) < ri))[0]
  cellGidsDic[args.tiles-1] = np.where(np.sqrt(x**2 + y**2) > rs[-1])[0]

  # rs = np.linspace(0.0, maxRadius, args.tiles).tolist()
  # for it in range(1, args.tiles):
  #   r0 = rs[it-1]
  #   r1 = rs[it]
  #   cellGidsDic[it-1] = np.where((np.sqrt(x**2 + y**2) > r0) & (np.sqrt(x**2 + y**2) < r1))[0]
  #   print(len(cellGidsDic[it-1]))
  # cellGidsDic[args.tiles-1] = np.where(np.sqrt(x**2 + y**2) > rs[-1])[0]

  stateDofsGidsDic = _mapCellGidsToStateDofsGids(cellGidsDic, ndpc)
  np.savetxt(workDir+"/ntiles.txt", \
             np.array([args.tiles]), fmt='%8d')

  for k,v in cellGidsDic.items():
    np.savetxt(workDir+"/cell_gids_wrt_full_mesh_p_"+str(k)+".txt", \
               np.array(v), fmt='%8d')

  for k,v in stateDofsGidsDic.items():
    np.savetxt(workDir+"/state_vec_rows_wrt_full_mesh_p_"+str(k)+".txt", \
               np.array(v), fmt='%8d')
