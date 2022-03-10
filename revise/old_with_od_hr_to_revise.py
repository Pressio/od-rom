
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import random, subprocess
import matplotlib.pyplot as plt
import re, os, time
import numpy as np
from numpy import linalg as LA
import pressiodemoapps as pda
from scipy import linalg

# -------------------------------------------------------------------
def dir_path_to_partition_info(workDir, npx, npy):
  return workDir + "/od"+str(npx)+"x"+str(npy)+"_info"

# -------------------------------------------------------------------
def dir_path_to_pod_data(workDir, npx, npy):
  return workDir + "/od"+str(npx)+"x"+str(npy)+"_full_pod"

# -------------------------------------------------------------------
def dir_path_to_od_sample_mesh(workDir, npx, npy, smPct):
  return workDir + "/od"+str(npx)+"x"+str(npy)+"_pct_"+str(smPct)+"_sample_mesh"

# -------------------------------------------------------------------
def dir_path_to_od_projectors(workDir, npx, npy, smPctString, numBasis):
  a = "od"+str(npx)+"x"+str(npy)
  b = "_projectors"
  c = "_pct_" + smPctString
  d = "_k_"+str(numBasis)
  return workDir+"/"+a+b+c+d

# -------------------------------------------------------------------
def extract_mesh_size(meshPath):
  def _single_n(string):
    reg = re.compile(r''+string+'.+')
    filein = open(meshPath+'/info.dat', 'r')
    result = re.search(reg, filein.read())
    filein.close()
    assert(result)
    return int(result.group().split()[1])
  return _single_n('nx'), _single_n('ny')

# -------------------------------------------------------------------
def write_matrix_to_bin(fileName, M, writeShape, transposeBeforeWriting):
  fileo = open(fileName, "wb")
  # write to beginning of file the extents of the matrix
  if writeShape:
    r=np.int64(M.shape[0])
    np.array([r]).tofile(fileo)
    c=np.int64(M.shape[1])
    np.array([c]).tofile(fileo)
  if transposeBeforeWriting:
    MT = np.transpose(M)
    MT.tofile(fileo)
  else:
    M.tofile(fileo)
  fileo.close()

# -------------------------------------------------------------------
def run_fom(workDir, kind, runId, fomObj, dt, numSteps, samplingFreq):
  runDir = workDir + "/fom_" + kind + "_" + str(runId)

  if not os.path.exists(runDir):
    os.makedirs(runDir)

    yn = fomObj.initialCondition()
    np.savetxt(runDir+'/initial_state.txt', yn)
    numDofs = len(yn)
    stateSamplingFreq = samplingFreq['state']
    rhsSamplingFreq   = samplingFreq['rhs']
    obsO = FomObserver(numDofs, stateSamplingFreq, rhsSamplingFreq, numSteps)
    pda.advanceSSP3(fomObj, yn, dt, numSteps, observer=obsO)
    obsO.write(runDir)
    np.savetxt(runDir+'/final_state.txt', yn)

# -------------------------------------------------------------------
def load_basis_from_binary_file(lsvFile):
  nr, nc  = np.fromfile(lsvFile, dtype=np.int64, count=2)
  M = np.fromfile(lsvFile, offset=np.dtype(np.int64).itemsize*2)
  M = np.reshape(M, (nr, nc), order='F')
  return M

# -------------------------------------------------------------------
def make_partitions(workDir, partitions, nx, ny, numDofsPerCell):
  file_path = pathlib.Path(__file__).parent.absolute()
  #from partitioning import createPartitions

  partitionInfoDirs = {}
  partitionBlockSizes = {}
  for pIt in partitions:
    nTilesX, nTilesY = pIt[0], pIt[1]
    currPartInfoDir = dir_path_to_partition_info(workDir, nTilesX, nTilesY)

    if os.path.exists(currPartInfoDir):
      print('Partition {} already exists'.format(currPartInfoDir))
    else:
      print('Generating partition files for {}'.format(currPartInfoDir))
      os.system('mkdir -p ' + currPartInfoDir)

      args = ("python3", str(file_path)+'/partition.py',
              "--wdir", currPartInfoDir,
              "--mesh", str(nx), str(ny),
              "--tiles", str(nTilesX), str(nTilesY),
              "--ndpc", str(numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE); popen.wait()
      output = popen.stdout.read();
      print(output)

# -------------------------------------------------------------------
def load_fom_state_snapshot_matrix(dataDirs, numTotDofs, numDofsPerCell):
  M = np.zeros((0, numTotDofs))
  for targetDirec in dataDirs:
    print("reading data from {}".format(targetDirec))

    data = np.fromfile(targetDirec+"/fom_snaps_state")
    numTimeSteps = int(np.size(data)/numTotDofs)
    D = np.reshape(data, (numTimeSteps, numTotDofs))
    M = np.append(M, D, axis=0)

  print("state snapshots: shape  : ", M.T.shape)
  print("state snapshots: min/max: ", np.min(M), np.max(M))
  return M.T

# -------------------------------------------------------------------
def load_fom_rhs_snapshot_matrix(dataDirs, numTotDofs, numDofsPerCell):
  M = np.zeros((0, numTotDofs))
  for targetDirec in dataDirs:
    print("reading data from {}".format(targetDirec))

    data = np.fromfile(targetDirec+"/fom_snaps_rhs")
    numTimeSteps = int(np.size(data)/numTotDofs)
    D = np.reshape(data, (numTimeSteps, numTotDofs))
    M = np.append(M, D, axis=0)

  print("rhs snapshots: shape    : ", M.T.shape)
  print("rhs snapshots: min/max  : ", np.min(M), np.max(M))
  return M.T

# -------------------------------------------------------------------
def do_svd_py(mymatrix, lsvFile):
  timing = np.zeros(1)
  start = time.time()
  U,S,_ = linalg.svd(mymatrix, full_matrices=False, lapack_driver='gesdd')
  end = time.time()
  elapsed = end - start
  timing[0] = elapsed
  #print("elapsed ", elapsed)
  # singular values
  #print("Writing sing values to file: {}".format(svaFile))
  #np.savetxt(svaFile, S)

  assert(U.flags['F_CONTIGUOUS'])

  # left singular vectors
  fileo = open(lsvFile, "wb")
  # write to beginning of file the extents of the matrix
  print("  writing POD modes to file: {}".format(lsvFile))
  r=np.int64(U.shape[0])
  np.array([r]).tofile(fileo)
  c=np.int64(U.shape[1])
  np.array([c]).tofile(fileo)
  '''
  NOTE: tofile write an array in rowwise, REGARDLESS of the layout of the matrix.
  So here we need to pass U.T to tofile so that tofile writes U in the proper
  way required format for how we read these later
  '''
  UT = np.transpose(U)
  UT.tofile(fileo)
  fileo.close()

  # # finally print timings to file
  # # find out the directory where we are storing things
  outDir = os.path.dirname(lsvFile)
  np.savetxt(lsvFile+'.txt', U[:,:3])
  # np.savetxt(outDir+'/timings.txt', timing)

# -------------------------------------------------------------------
def compute_full_pod_in_each_tile(workDir, partitions, totFomDofs, numDofsPerCell):
  '''
  compute pod for both state and rhs using fom train data
  '''

  fomDirsData   = [workDir+'/'+d for d in os.listdir(workDir) if "fom_train" in d]
  fomStateSnaps = load_fom_state_snapshot_matrix(fomDirsData, totFomDofs, numDofsPerCell)
  fomRhsSnaps   = load_fom_rhs_snapshot_matrix(fomDirsData,   totFomDofs, numDofsPerCell)
  print("pod: fomStateSnaps.shape = ", fomStateSnaps.shape)
  print("pod:   fomRhsSnaps.shape = ", fomRhsSnaps.shape)
  print("")
  #np.savetxt(workDir+'/FULLsnaps.txt', fomStateSnaps)

  # loop over all partitions
  for pIt in partitions:
    nTilesX, nTilesY = pIt[0], pIt[1]
    currPartInfoDir  = dir_path_to_partition_info(workDir, nTilesX, nTilesY)
    outDir = dir_path_to_pod_data(workDir, nTilesX, nTilesY)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))

    else:
      os.system('mkdir -p ' + outDir)
      # loop over each tile
      for tileId in range(nTilesX*nTilesY):
        # I need to compute POD for both STATE and RHS
        # using FOM data LOCAL to myself so need to
        # find from file which rows of the FOM state I own
        # and slice accordingly
        myFullStateRows = np.loadtxt(currPartInfoDir + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt", dtype=int)

        # with loaded row indices I can slice the FOM STATE and RHS snapshots
        # to get only the data that belongs to me.
        myStateSlice = fomStateSnaps[myFullStateRows, :]
        myRhsSlice   = fomRhsSnaps[myFullStateRows, :]
        print("pod: tileId={}".format(tileId))
        print("  stateSlice.shape={}".format(myStateSlice.shape))
        print("  rhsSlice.shape  ={}".format(myRhsSlice.shape))
        #np.savetxt(outDir+'/snaps_p_'+str(tileId)+'.txt', myStateSlice)

        # svd
        lsvFile = outDir + '/lsv_state_p_'+str(tileId)
        do_svd_py(myStateSlice, lsvFile)
        lsvFile = outDir + '/lsv_rhs_p_'+str(tileId)
        do_svd_py(myRhsSlice, lsvFile)

# -------------------------------------------------------------------
def compute_sample_mesh(workDir, pdaDir, fullMeshPath, partitions,
                        basisCounts, smPercentages, \
                        totFomDofs, numDofsPerCell):
  import scipy.linalg

  # load RHS fom shapshots
  fomDirsFullPath = [workDir+'/'+d for d in os.listdir(workDir) if "fom_train" in d]
  fomRhsSnaps     = load_fom_rhs_snapshot_matrix(fomDirsFullPath, totFomDofs, numDofsPerCell)

  for smPct in smPercentages:
    for pIt in partitions:
      nTilesX, nTilesY = pIt[0], pIt[1]
      currPartInfoDir  = dir_path_to_partition_info(workDir, nTilesX, nTilesY)
      outDir           = dir_path_to_od_sample_mesh(workDir, nTilesX, nTilesY, smPct)
      if os.path.exists(outDir):
        print('{} already exists'.format(outDir))
      else:
        print('Generating OD sample mesh in {}'.format(outDir))
        os.system('mkdir -p ' + outDir)

        global_sample_mesh_gids = []
        for tileId in range(nTilesX*nTilesY):
          # figure out how many sample mesh cells I need
          myCellGids        = np.loadtxt(currPartInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt",dtype=int)
          myNumCells        = len(myCellGids)
          mySampleMeshCount = int(myNumCells * smPct)
          print("tileId = ", tileId)
          print(" numCellsInThisPartition = ",    myNumCells)
          print(" sampleMeshSizePerPartition = ", mySampleMeshCount)

          # ensure the sm count >= basis count
          for K in basisCounts: assert(mySampleMeshCount >= K)

          # using P sampling
          # get only snapshot data that belongs to me: FIX what quantity we use
          fullStateRowsFile = currPartInfoDir + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
          myFullStateRows = np.loadtxt(fullStateRowsFile, dtype=int)
          myRhsSnaps = fomRhsSnaps[myFullStateRows, :][0::numDofsPerCell]
          U,_,_ = np.linalg.svd(myRhsSnaps, full_matrices=False)
          Q,R,P = scipy.linalg.qr(U[:,0:mySampleMeshCount].transpose(), pivoting=True)
          mylocalids = np.array(np.sort(P[0:mySampleMeshCount]))
          mySampleMeshGidsWrtFullMesh = myCellGids[mylocalids]

          # # using lev scores
          # myS = np.reshape(myS, (3*numCellsInThisPartition, nsnaps))
          # myS2 = ptla.MultiVector(np.asfortranarray(myS))
          # mylocalids, pmf = computeNodes(matrix=myS2, numSamples=sampleMeshSizePerPartition, dofsPerMeshNode=3)
          # vals = np.array(cellGids)
          # sample_mesh += vals[mylocalids].tolist()
          # np.savetxt(outDir+'/sample_mesh_gids_p_'+str(tileId)+'.txt', vals[mylocalids], fmt='%8i')

          # # -------------
          # # fully random
          # # -------------
          # # randomly sample target set of indices from 0,myNumCells
          # smCellsIndices = random.sample(range(0, myNumCells), mySampleMeshCount)
          # mylocalids = np.sort(smCellsIndices)
          # mySampleMeshGidsWrtFullMesh = myCellGids[mylocalids]

          # add to sample mesh
          global_sample_mesh_gids += mySampleMeshGidsWrtFullMesh.tolist()
          np.savetxt(outDir+'/sample_mesh_gids_p_'+str(tileId)+'.txt', mySampleMeshGidsWrtFullMesh, fmt='%8i')

        # now we can write to file the gids of the sample mesh cells ACROSS entire domain
        global_sample_mesh_gids = np.sort(global_sample_mesh_gids)
        np.savetxt(outDir+'/sample_mesh_gids.dat', global_sample_mesh_gids, fmt='%8i')

        print('Generating sample mesh in:')
        print(' {}'.format(outDir))
        owd = os.getcwd()
        meshScriptsDir = pdaDir + "/meshing_scripts"
        args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
                "--fullMeshDir", fullMeshPath,
                "--sampleMeshIndices", outDir+'/sample_mesh_gids.dat',
                "--outDir", outDir+"/sm",
                "--useTilingFrom", currPartInfoDir)
        popen  = subprocess.Popen(args, stdout=subprocess.PIPE); popen.wait()
        output = popen.stdout.read();
        print(output)

        # copy from outDir/sm the generated stencil mesh gids file
        args = ("cp", outDir+"/sm/stencil_mesh_gids.dat", outDir+"/stencil_mesh_gids.dat")
        popen  = subprocess.Popen(args, stdout=subprocess.PIPE); popen.wait()
        output = popen.stdout.read();
        print(output)

        stencilGids = np.loadtxt(outDir+"/sm/stencil_mesh_gids.dat", dtype=int)
        for tileId in range(nTilesX*nTilesY):
          myCellGids   = np.loadtxt(currPartInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt",dtype=int)
          commonElem = set(stencilGids).intersection(myCellGids)
          commonElem = np.sort(list(commonElem))
          np.savetxt(outDir+'/stencil_mesh_gids_p_'+str(tileId)+'.dat', commonElem, fmt='%8i')

# -------------------------------------------------------------------
def compute_gappy_projector(workDir, partitions, basisCountList, numDofsPerCell):
  # need to find all sample meshes directories to work with
  sampleMeshDirs = [workDir+'/'+d for d in os.listdir(workDir) if "_sample_mesh" in d]

  for pIt in partitions:
    nTilesX, nTilesY = pIt[0], pIt[1]
    currPartInfoDir  = dir_path_to_partition_info(workDir, nTilesX, nTilesY)
    odPodDataDir     = dir_path_to_pod_data(workDir, nTilesX, nTilesY)

    # loop over all num of basis
    for K in basisCountList:
      print("numModes = ", K)

      # loop over all sample meshes directories
      for smDirIt in sampleMeshDirs:
        pctString = smDirIt.split("_")[3]
        outDir    = dir_path_to_od_projectors(workDir, nTilesX, nTilesY, pctString, K)

        if os.path.exists(outDir):
          print('{} already exists'.format(outDir))
        else:
          print('Generating {}'.format(outDir))
          os.system('mkdir -p ' + outDir)

          for tileId in range(nTilesX*nTilesY):
            myCellGids   = np.loadtxt(currPartInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt",dtype=int)
            myPhi        = load_basis_from_binary_file( odPodDataDir + "/lsv_state_p_" + str(tileId) )[:,0:K]
            myTheta      = load_basis_from_binary_file( odPodDataDir + "/lsv_rhs_p_" + str(tileId) )[:,0:K]
            mySmMeshGids = np.loadtxt(smDirIt + "/sample_mesh_gids_p_"+str(tileId)+".txt", dtype=int)
            mySmCount    = len(mySmMeshGids)
            myStMeshGids = np.loadtxt(smDirIt + "/stencil_mesh_gids_p_"+str(tileId)+".dat", dtype=int)
            myStCount    = len(myStMeshGids)

            commonElem = set(myStMeshGids).intersection(myCellGids)
            commonElem = np.sort(list(commonElem))
            mylocalinds = np.searchsorted(myCellGids, commonElem)
            mySlicedPhi = np.zeros((myStCount*numDofsPerCell, K), order='F')
            for j in range(numDofsPerCell):
              mySlicedPhi[j::numDofsPerCell, :] = myPhi[numDofsPerCell*mylocalinds + j, :]
            np.savetxt(outDir+'/stencil_phi_p_'+str(tileId)+'.txt', mySlicedPhi)

            # need to slice theta to only get elements for my sample mesh cells
            commonElem = set(mySmMeshGids).intersection(myCellGids)
            commonElem = np.sort(list(commonElem))
            mylocalinds = np.searchsorted(myCellGids, commonElem)
            mySlicedTheta = np.zeros((mySmCount*numDofsPerCell, K), order='F')
            for j in range(numDofsPerCell):
              mySlicedTheta[j::numDofsPerCell, :] = myTheta[numDofsPerCell*mylocalinds + j, :]

            A = myPhi.transpose() @ myTheta
            projector = A @ linalg.pinv(mySlicedTheta)
            print("proj.shape = ", projector.shape)
            print("")

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

# -------------------------------------------------------------------
def fom_plot_final(meshPath, fomDir, totFomDofs, numDofsPerCell):
  D = load_fom_state_snapshot_matrix([fomDir], totFomDofs, numDofsPerCell)
  print("D shape = ", D.shape)
  nx,ny = extractN(meshPath, 'nx'), extractN(meshPath, 'ny')
  fomTotDofs = nx*ny*numDofsPerCell
  fomCoords = np.loadtxt(meshPath+'/coordinates.dat', dtype=float)
  x = np.reshape(fomCoords[:,1], (ny,nx))
  y = np.reshape(fomCoords[:,2], (ny,nx))
  fig, axs = plt.subplots(1,1)
  rho  = np.reshape(D[:,-1][0::4], (ny,nx))
  axs.contourf(x, y, rho, 35)
  axs.set_aspect(aspect=1.)
  plt.show()


# -------------------------------------------------------------------
def make_full_mesh_if_needed(workingDir, args):
  pdaMeshDir = args.pdaDir + "/meshing_scripts"
  nx, ny     = args.meshSize[0], args.meshSize[1]
  meshDir    = workingDir + "/full_mesh" + str(nx) + "x" + str(ny)

  if os.path.exists(meshDir):
    # if mesh exists, do nothing
    print('Mesh {} already exists'.format(meshDir))
  else:
    # generate
    print('Generating mesh {}'.format(meshDir))
    # call script
    owd = os.getcwd()
    args = ("python3", pdaMeshDir + '/create_full_mesh.py',\
            "-n", str(nx), str(ny),\
            "--outDir", meshDir,\
            #"--bounds", "0.0", "0.25", "0.0", "1.0",
            "--bounds", "-5.0", "5.0", "-5.0", "5.0",
            "-s", str(args.stencilSize))

    popen  = subprocess.Popen(args, stdout=subprocess.PIPE); popen.wait()
    output = popen.stdout.read();

  return meshDir

# -------------------------------------------------------------------
class OdRom:
  def __init__(self, workDir, numFomDofs, \
               sampleMeshSize, stencilMeshSize, \
               numDofsPerCell, pIt, K, pct):

    nTilesX, nTilesY = pIt[0], pIt[1]
    self.K_       = K
    self.nTilesX_ = nTilesX
    self.nTilesY_ = nTilesY
    self.nTiles_  = nTilesX*nTilesY
    self.fomStateFullMesh_    = np.zeros(totFomDofs)
    self.fomStateStencilMesh_ = np.zeros(stencilMeshSize)
    self.fomVeloSampleMesh_   = np.zeros(sampleMeshSize)

    partInfoDir   = dir_path_to_partition_info(workDir, nTilesX, nTilesY)
    podDataDir    = dir_path_to_pod_data(workDir, nTilesX, nTilesY)
    projectorsDir = dir_path_to_od_projectors(workDir, nTilesX, nTilesY, str(pct), K)
    sampleMeshDir = dir_path_to_od_sample_mesh(workDir, nTilesX, nTilesY, pct)

    self.fullVeloRows_  = {}
    self.fullStateRows_ = {}
    self.fullPhis_  = {} # phi on full mesh
    self.phis_      = {} # phi on stencil mesh
    self.prjs_      = {}
    for tileId in range(nTilesX*nTilesY):
      myFullPhi = load_basis_from_binary_file(podDataDir + "/lsv_state_p_" + str(tileId) )[:,0:K]
      self.fullPhis_[tileId] = myFullPhi

      myStPhi = np.loadtxt(projectorsDir+'/stencil_phi_p_'+str(tileId)+'.txt')
      self.phis_[tileId] = myStPhi

      myPrj = np.loadtxt(projectorsDir+'/projector_p_'+str(tileId)+'.txt')
      self.prjs_[tileId] = myPrj

      srVecFile = partInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
      self.fullStateRows_[tileId] = np.loadtxt(srVecFile, dtype=int)

      mysmgids = np.loadtxt(sampleMeshDir + "/sample_mesh_gids_p_"+str(tileId)+".txt", dtype=int)
      if numDofsPerCell==3:
        myl = [None]*len(mysmgids)*3
        myl[0::3] = [int(i*3)   for i in mysmgids]
        myl[1::3] = [int(i*3+1) for i in mysmgids]
        myl[2::3] = [int(i*3+2) for i in mysmgids]
        self.fullVeloRows_[tileId] = np.array(myl)
      elif numDofsPerCell==4:
        myl = [None]*len(mysmgids)*4
        myl[0::4] = [int(i*4)   for i in mysmgids]
        myl[1::4] = [int(i*4+1) for i in mysmgids]
        myl[2::4] = [int(i*4+2) for i in mysmgids]
        myl[3::4] = [int(i*4+3) for i in mysmgids]
        self.fullVeloRows_[tileId] = np.array(myl)

  def createRomState(self):
    return np.zeros(self.K_ * self.nTiles_)

  def viewFomStateFullMesh(self):
    return self.fomStateFullMesh_

  def reconstructMemberFomStateFullMesh(self, romStateIn):
    K = self.K_
    for tileId in range(self.nTiles_):
      myRomStateSlice = romStateIn[tileId*K:tileId*K+K]
      myPhi = self.fullPhis_[tileId]
      tmpy  = np.dot(myPhi, myRomStateSlice)
      for j,it in enumerate(self.fullStateRows_[tileId]):
        self.fomStateFullMesh_[it] = tmpy[j]

  def reconstructMemberFomStateStencilMesh(self, romStateIn):
    K = self.K_
    startfrom = 0
    for tileId in range(self.nTiles_):
      myRomStateSlice = romStateIn[tileId*K:tileId*K+K]
      myPhi = self.phis_[tileId]
      tmpy  = np.dot(myPhi, myRomStateSlice)
      self.fomStateStencilMesh_[startfrom:startfrom+len(tmpy)] = np.copy(tmpy)
      startfrom += len(tmpy)

  def projectMemberFomVelo(self, romState):
    K = self.K_
    startfrom = 0
    for tileId in range(self.nTiles_):
      #myProjector = self.prjs_[tileId]
      myProjector = self.phis_[tileId]
      n = myProjector.shape[0]
      myRhsSlice  = self.fomVeloSampleMesh_[startfrom:startfrom+n]
      yhattmp     = np.dot(myProjector.T, myRhsSlice)
      romState[tileId*K:tileId*K+K] = np.copy(yhattmp)
      startfrom += n

  def run(self, yhat, fomApp, nSteps, dt):
    romRhs = np.zeros_like(yhat)
    yhat0  = np.zeros_like(yhat)
    yhat1  = np.zeros_like(yhat)

    two = 2.
    oneOverThree  = 1./3.
    oneOverFour   = 1./4.
    threeOverFour = 3./4.

    time = 0.
    for step in range(1, nSteps+1):
      print("norm = {}".format(linalg.norm(yhat)))
      if step % 50 == 0:
        print("step = ", step, "/", nSteps)

      self.reconstructMemberFomStateFullMesh(yhat)
      if step % 50 == 0:
        np.savetxt(workDir+"/y_rec_"+str(step)+".txt", self.fomStateFullMesh_)

      self.reconstructMemberFomStateStencilMesh(yhat)

      fomApp.velocity(self.fomStateStencilMesh_, time, self.fomVeloSampleMesh_)
      self.projectMemberFomVelo(romRhs);
      yhat0[:] = yhat + dt * romRhs

      self.reconstructMemberFomStateStencilMesh(yhat0)
      fomApp.velocity(self.fomStateStencilMesh_, time, self.fomVeloSampleMesh_)
      self.projectMemberFomVelo(romRhs);
      yhat1[:] = threeOverFour*yhat + oneOverFour*yhat0 + oneOverFour*dt*romRhs

      self.reconstructMemberFomStateStencilMesh(yhat1)
      fomApp.velocity(self.fomStateStencilMesh_, time, self.fomVeloSampleMesh_)
      self.projectMemberFomVelo(romRhs);
      yhat[:] = oneOverThree*(yhat + two*yhat1 + two*dt*romRhs)

      time += dt


#----------------------------
if __name__ == '__main__':
#----------------------------
  parser   = ArgumentParser()
  parser.add_argument("--wdir", "--workdir", required=True)
  parser.add_argument("--pdadir", required=True)

  args     = parser.parse_args()
  workDir  = args.workdir
  pdaDir   = args.pdadir

  # read yaml input file

  # decide if fom or odrom



  sys.exit()

  partitions    = [[6,6]]
  smPercentages = [1.0]
  basisCounts   = [10]

  sampFrequency  = {'state': 2, 'rhs': 2 }
  timeStepSize   = 0.005
  finalTime      = 5.0
  numSteps       = int(finalTime/timeStepSize)
  train_values   = [-3.0, -0.5]
  test_values    = [-3.5]
  numDofsPerCell = 3
  print("numSteps = {}".format(numSteps))

  probId = pda.Swe2d.SlipWall #pda.Euler2d.RayleighTaylor
  scheme = None
  if args.stencilSize == 3:
    scheme = pda.InviscidFluxReconstruction.FirstOrder
  elif args.stencilSize == 5:
    scheme = pda.InviscidFluxReconstruction.Weno3
  elif args.stencilSize == 7:
    scheme = pda.InviscidFluxReconstruction.Weno5
  else:
    sys.exit("Invalid stencil size {}".format(args.stencilSize))

  #===============
  # generate and load full mesh
  meshPath   = make_full_mesh_if_needed(workDir, args)
  meshObj    = pda.load_cellcentered_uniform_mesh(meshPath)
  nx, ny     = extract_mesh_size(meshPath)
  totFomDofs = nx*ny*numDofsPerCell
  assert(nx == args.meshSize[0])
  assert(ny == args.meshSize[1])

  #===============
  # train foms
  for i,paramValueIt in enumerate(train_values):
    print("Running fom_train_{} for {}".format(i, paramValueIt))
    appObj = pda.create_problem(meshObj, probId, scheme, 9.8, paramValueIt)
    run_fom(workDir, "train", i, appObj, timeStepSize, numSteps, sampFrequency)
  print("")

  #===============
  # test foms
  for i,paramValueIt in enumerate(test_values):
    print("Running fom_test_{} for  {}".format(i, paramValueIt))
    appObj = pda.create_problem(meshObj, probId, scheme, 9.8, paramValueIt)
    run_fom(workDir, "test", i, appObj, timeStepSize, numSteps, sampFrequency)
  print("")

  print("===================")
  print("Creating partitions")
  print("===================")
  make_partitions(workDir, partitions, nx, ny, numDofsPerCell)
  print("")

  print("===================")
  print("Doing POD")
  print("===================")
  compute_full_pod_in_each_tile(workDir, partitions, totFomDofs, numDofsPerCell)
  print("")

  # print("===================")
  # print("Sample meshes")
  # print("===================")
  # compute_sample_mesh(workDir, pdaDir, meshPath, partitions, basisCounts,\
  #                     smPercentages, totFomDofs, numDofsPerCell)
  # print("")

  # print("===================")
  # print("Doing projectors")
  # print("===================")
  # compute_gappy_projector(workDir, partitions, basisCounts, numDofsPerCell)
  # print("")

  # print("===================")
  # print("Doing masked ODROM")
  # print("===================")
  # parts  = partitions[0]
  # nTiles = parts[0]*parts[1]
  # K      = basisCounts[0]
  # pct    = smPercentages[0]
  # smMeshDir = [workDir+'/'+d for d in os.listdir(workDir) if "_sample_mesh" in d][0]
  # smMeshObj = pda.load_cellcentered_uniform_mesh(smMeshDir+"/sm")

  # for i,paramValueIt in enumerate(test_values):
  #   print("Running od_rom_test_{} for  {}".format(i, paramValueIt))

  #   fomObjForIC = pda.create_problem(meshObj, probId, scheme, 9.8, paramValueIt)
  #   fomIc       = fomObjForIC.initialCondition()
  #   romState    = np.zeros(K*nTiles)
  #   partInfoDir = dir_path_to_partition_info(workDir, parts[0], parts[1])
  #   podDataDir  = dir_path_to_pod_data(workDir, parts[0], parts[1])
  #   for tileId in range(nTiles):
  #     myFullPhi       = load_basis_from_binary_file(podDataDir + "/lsv_state_p_" + str(tileId) )[:,0:K]
  #     myStateRowsFile = partInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
  #     myStateRows     = np.loadtxt(myStateRowsFile, dtype=int)
  #     myFomIcSlice    = fomIc[myStateRows]
  #     tmpyhat         = np.dot(myFullPhi.transpose(), myFomIcSlice)
  #     romState[tileId*K:tileId*K+K] = np.copy(tmpyhat)

  #   fomObj = pda.create_problem(smMeshObj, probId, scheme, 9.8, paramValueIt)
  #   sampleMeshSize  = fomObj.totalDofSampleMesh()
  #   stencilMeshSize = fomObj.totalDofStencilMesh()

  #   odRomObj = OdRom(workDir, totFomDofs, sampleMeshSize, \
  #                    stencilMeshSize, numDofsPerCell, parts, K, pct)
  #   odRomObj.reconstructMemberFomStateFullMesh(romState)
  #   yRIC = odRomObj.viewFomStateFullMesh()
  #   np.savetxt(workDir+"/y_rec_ic.txt", yRIC)

  #   odRomObj.run(romState, fomObj, numSteps, timeStepSize)

  #   odRomObj.reconstructMemberFomStateFullMesh(romState)
  #   yRecon = odRomObj.viewFomStateFullMesh()
  #   np.savetxt(workDir+"/y_rec_final.txt", yRecon)

  #   # fomCoords = np.loadtxt(meshPath+'/coordinates.dat', dtype=float)
  #   # x = np.reshape(fomCoords[:,1], (ny,nx))
  #   # y = np.reshape(fomCoords[:,2], (ny,nx))
  #   # fig = plt.figure()
  #   # ax = plt.gca()
  #   # rho  = np.reshape(fomRIC[0::4], (ny,nx))
  #   # ax.contourf(x, y, rho, 35)
  #   # ax.set_aspect(aspect=1.)
  #   # plt.show()
