
from scipy.special import legendre
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import random, subprocess
import matplotlib.pyplot as plt
import re, os, time, yaml
import numpy as np
from numpy import linalg as LA
import pressiodemoapps as pda
from scipy import linalg
from odrom_full import *
from myio import *
from observer import FomObserver
from entropy_fncs import conservative_to_entropy

#==============================================================
# functions
#==============================================================

def get_run_id(runDir):
  return int(runDir.split('_')[-1])

# -------------------------------------------------------------------
def find_sample_mesh_count_from_info_file(workDir):
  reg = re.compile(r'sampleMeshSize.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_stencil_mesh_count_from_info_file(workDir):
  reg = re.compile(r'stencilMeshSize.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_dimensionality_from_info_file(workDir):
  reg = re.compile(r'dim.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_num_cells_from_info_file(workDir, ns):
  reg = re.compile(r''+ns+'.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_total_cells_from_info_file(workDir):
  dims = find_dimensionality_from_info_file(workDir)
  if dims == 1:
    return find_num_cells_from_info_file(workDir, "nx")
  elif dims==2:
    nx = find_num_cells_from_info_file(workDir, "nx")
    ny = find_num_cells_from_info_file(workDir, "ny")
    return nx*ny
  else:
    sys.exit("Invalid dims = {}".format(dims))

# -------------------------------------------------------------------
def path_to_partition_info_dir(workDir, npx, npy, style):
  return workDir + "/od_info_"+str(npx)+"x"+str(npy)+"_"+style

# -------------------------------------------------------------------
def string_identifier_from_partition_info_dir(infoDir):
  return os.path.basename(infoDir)[8:]

# -------------------------------------------------------------------
def path_to_partition_based_full_mesh_dir(workDir, partitioningKeyword):
  return workDir + "/partition_based_"+partitioningKeyword+"_full_mesh"

# -------------------------------------------------------------------
def path_to_pod_data_dir(workDir, partitioningKeyword, setId):
  return workDir + "/partition_based_"+partitioningKeyword+"_full_pod_set_"+str(setId)

# -------------------------------------------------------------------
def path_to_poly_bases_data_dir(workDir, partitioningKeyword, order, energy=None, setId=None):
  res = workDir + "/partition_based_"+partitioningKeyword+"_full_poly_order_"+str(order)
  if energy != None:
    res += "_"+str(energy)
  if setId != None:
    res += "_set_"+str(setId)

  return res

# -------------------------------------------------------------------
def inviscid_flux_string_to_stencil_size(stringIn):
  if stringIn == "FirstOrder":
    return 3
  elif stringIn == "Weno3":
    return 5
  elif stringIn == "Weno5":
    return 7
  else:
    sys.exit("Invalid scheme detected {}".format(scheme))
    return None

# -------------------------------------------------------------------
def inviscid_flux_string_to_enum(stringIn):
  if stringIn == "FirstOrder":
    return pda.InviscidFluxReconstruction.FirstOrder
  elif stringIn == "Weno3":
    return pda.InviscidFluxReconstruction.Weno3
  elif stringIn == "Weno5":
    return pda.InviscidFluxReconstruction.Weno5
  else:
    sys.exit("Invalid string")

# ----------------------------------------------------------------
def make_fom_mesh_if_not_existing(workDir, problem, \
                                  module, scenario, \
                                  pdaDir, meshSize):

  # store dir to meshing scripts in pressio-demoapps
  pdaMeshDir = pdaDir + "/meshing_scripts"
  meshArgs = None

  if problem in ["2d_swe", "2d_rti"]:
    # need both x and y
    assert(len(args.mesh) == 2)

    # find stencil size needed for the FOM
    schemeString = module.base_dic[scenario]['fom']['inviscidFluxReconstruction']
    stencilSize  = inviscid_flux_string_to_stencil_size(schemeString)

    # bounds depend on problem
    bounds = None
    if problem == "2d_swe":
      bounds = [-5., 5., -5., 5.]
    else:
      bounds = [0., 0.25, 0., 1.]

    nx, ny = meshSize[0], meshSize[1]
    outDir = workDir + "/full_mesh" + str(nx) + "x" + str(ny)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print('Generating mesh {}'.format(outDir))
      meshArgs = ("python3", \
                  pdaMeshDir + '/create_full_mesh.py',\
                  "-n", str(nx), str(ny),\
                  "--outDir", outDir,\
                  "--bounds", str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]),\
                  "-s", str(stencilSize))
      popen  = subprocess.Popen(meshArgs, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

  elif problem  == "2d_gs":

    assert(len(args.mesh) == 2)
    bounds = [-1.25, 1.25, -1.25, 1.25]
    nx, ny = meshSize[0], meshSize[1]
    outDir = workDir + "/full_mesh" + str(nx) + "x" + str(ny)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print('Generating mesh {}'.format(outDir))
      meshArgs = ("python3", \
                  pdaMeshDir + '/create_full_mesh.py',\
                  "-n", str(nx), str(ny),\
                  "--outDir", outDir,\
                  "--bounds", str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]),\
                  "--periodic", "x", "y")
      popen  = subprocess.Popen(meshArgs, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

  elif problem  in ["2d_eulersmooth", "2d_burgers"]:
    assert(len(args.mesh) == 2)

    # find stencil size needed for the FOM
    schemeString = module.base_dic[scenario]['fom']['inviscidFluxReconstruction']
    stencilSize  = inviscid_flux_string_to_stencil_size(schemeString)

    bounds = [-1.0, 1.0, -1.0, 1.0]
    nx, ny = meshSize[0], meshSize[1]
    outDir = workDir + "/full_mesh" + str(nx) + "x" + str(ny)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print('Generating mesh {}'.format(outDir))
      meshArgs = ("python3", \
                  pdaMeshDir + '/create_full_mesh.py',\
                  "-n", str(nx), str(ny),\
                  "--outDir", outDir,\
                  "--bounds", str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]),\
                  "-s", str(stencilSize), \
                  "--periodic", "x", "y")
      popen  = subprocess.Popen(meshArgs, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();

  else:
    sys.exit("make_fom_mesh: invalid problem = {}".format(problemName))

# ----------------------------------------------------------------
def find_full_mesh_and_ensure_unique(workDir):
  # This is because the mesh is specified
  # via command line argument and must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory

  fomFullMeshes = [workDir+'/'+d for d in os.listdir(workDir) \
                   # we need to find only dirs that BEGIN with
                   # this string otherwise we pick up other things
                   if "full_mesh" == os.path.basename(d)[0:9]]
  if len(fomFullMeshes) != 1:
    sys.exit("Error: detected multiple full meshes in a single working dir. \
You can only have a single FULL mesh the working directory.")

  return fomFullMeshes[0]

# -------------------------------------------------------------------
def run_single_fom(runDir, appObj, dic, finalTime):
  # need to update the final time in dic since it was
  # set by the calling function
  del dic['finalTimeTrain']
  del dic['finalTimeTest']
  dic['finalTime'] = float(finalTime)

  # write to yaml the dic to save info for later
  inputFile = runDir + "/input.yaml"
  write_dic_to_yaml_file(inputFile, dic)

  from decimal import Decimal

  # extrac params
  odeScheme         = dic['odeScheme']
  dt                = float(dic['dt'])
  stateSamplingFreq = int(dic['stateSamplingFreq'])
  rhsSamplingFreq   = int(dic['velocitySamplingFreq'])
  numSteps          = int(round(Decimal(finalTime)/Decimal(dt), 8))
  print("numSteps = ", numSteps)

  # run
  yn = appObj.initialCondition()
  np.savetxt(runDir+'/initial_state.txt', yn)
  numDofs = len(yn)

  start = time.time()
  obsO = FomObserver(numDofs, stateSamplingFreq, rhsSamplingFreq, numSteps)
  if odeScheme in ["RungeKutta4", "RK4"]:
    pda.advanceRK4(appObj, yn, dt, numSteps, observer=obsO)
  elif odeScheme == "SSPRK3":
    pda.advanceSSP3(appObj, yn, dt, numSteps, observer=obsO)
  else:
    sys.exit("run_single_fom: invalid ode scheme = {}".format(odeScheme))

  elapsed = time.time() - start
  print("elapsed = {}".format(elapsed))
  f = open(runDir+"/timing.txt", "w")
  f.write(str(elapsed))
  f.close()

  obsO.write(runDir)
  np.savetxt(runDir+'/final_state.txt', yn)


# -------------------------------------------------------------------
def run_foms(workDir, problem, module, scenario, \
             testOrTrainString, fomMesh):
  assert(testOrTrainString in ["train", "test"])

  # load the parameter values to run FOM
  param_values = None
  if testOrTrainString == "train":
    param_values = module.train_points[scenario]
  else:
    param_values = module.test_points[scenario]

  # fom mesh object is loaded in same way for ALL problems
  fomMeshObj = pda.load_cellcentered_uniform_mesh(fomMesh)

  # now we need to create the problem obj and run
  # but how we do this is specific to each problem
  for k,val in param_values.items():
    appObj = None
    # get the dic with base parameters for the FOM
    fomDic   = module.base_dic[scenario]['fom'].copy()
    coeffDic = module.base_dic[scenario]['physicalCoefficients'].copy()

    # ----------------------
    # --- 2d swe specific ---
    # ----------------------
    if problem == "2d_swe":
      probId    = pda.Swe2d.SlipWall
      schemeStr = fomDic['inviscidFluxReconstruction']
      schemeEnu = inviscid_flux_string_to_enum(schemeStr)
      gravity   = coeffDic['gravity']
      coriolis  = coeffDic['coriolis']
      pulse     = coeffDic['pulsemag']

      if scenario == 1:
        coriolis = val
      else:
        sys.exit("invalid scenario {} for 2d_swe".format(scenario))

      fomDic['gravity']  = gravity
      fomDic['coriolis'] = coriolis
      fomDic['pulse']    = pulse
      appObj = pda.create_problem(fomMeshObj, probId, schemeEnu, gravity, coriolis, pulse)

    # ----------------------
    # --- 2d gs specific ---
    # ----------------------
    elif problem == "2d_gs":
      probId   = pda.DiffusionReaction2d.GrayScott
      diff_A   = coeffDic['diffusionA']
      diff_B   = coeffDic['diffusionB']
      feedRate = coeffDic['feedRate']
      killRate = coeffDic['killRate']

      if scenario == 1:
        killRate = val
      elif scenario == 2:
        feedRate = val
      else:
        sys.exit("invalid scenario {} for 2d_gs".format(scenario))

      fomDic['diffusionA'] = diff_A
      fomDic['diffusionB'] = diff_B
      fomDic['feedRate']   = feedRate
      fomDic['killRate']   = killRate
      scheme = pda.ViscousFluxReconstruction.FirstOrder
      appObj = pda.create_problem(fomMeshObj, probId, scheme, diff_A, diff_B, feedRate, killRate)

      # ---------------------------
      # --- 2d burgers specific ---
      # ---------------------------
    elif problem == "2d_burgers":
      probId      = pda.AdvectionDiffusion2d.Burgers
      schemeStr   = fomDic['inviscidFluxReconstruction']
      schemeEnu   = inviscid_flux_string_to_enum(schemeStr)
      pulsemag    = coeffDic['pulsemag']
      pulsespread = coeffDic['pulsespread']
      diffusion   = coeffDic['diffusion']
      pulsecenter = coeffDic['pulsecenter']

      if scenario == 1:
        pulsespread = val
      else:
        sys.exit("invalid scenario {} for 2d_burgers".format(scenario))

      fomDic['pulsemag']    = pulsemag
      fomDic['pulsespread'] = pulsespread
      fomDic['diffusion']   = diffusion
      fomDic['pulsecenter'] = pulsecenter
      appObj  = pda.create_problem(fomMeshObj, probId, schemeEnu, \
                                   pulsemag, pulsespread, diffusion, \
                                   pulsecenter[0], pulsecenter[1])

    # ----------------------
    # --- 2d rti specific ---
    # ----------------------
    elif problem == "2d_rti":
      probId    = pda.Euler2d.RayleighTaylor
      schemeStr = fomDic['inviscidFluxReconstruction']
      schemeEnu = inviscid_flux_string_to_enum(schemeStr)
      amplitude = coeffDic['amplitude']

      if scenario == 1:
        amplitude = val
      else:
        sys.exit("invalid scenario {} for 2d_rti".format(scenario))

      fomDic['amplitude']  = amplitude
      appObj = pda.create_problem(fomMeshObj, probId, schemeEnu, amplitude)

    # --------------------------------
    # --- 2d euler smooth specific ---
    # --------------------------------
    elif problem == "2d_eulersmooth":
      probId    = pda.Euler2d.PeriodicSmooth
      schemeStr = fomDic['inviscidFluxReconstruction']
      schemeEnu = inviscid_flux_string_to_enum(schemeStr)

      if scenario == 1:
        pass
      else:
        sys.exit("invalid scenario {} for 2d_rti".format(scenario))

      appObj = pda.create_problem(fomMeshObj, probId, schemeEnu)

    else:
      sys.exit("invalid problem = {}".format(problemName))

    # note that the train/test simulation time might differ
    # so make sure we pick the right one
    finalTime  = fomDic['finalTimeTrain'] if testOrTrainString == "train" \
      else fomDic['finalTimeTest']

    # now we can do the FOM run for current parameter
    runDir = workDir + "/fom_"+testOrTrainString+"_"+str(k)
    if not os.path.exists(runDir):
      os.makedirs(runDir)
      print("Doing FOM run for {}".format(runDir))
      run_single_fom(runDir, appObj, fomDic, finalTime)
    else:
      print("FOM run {} already exists".format(runDir))

# -------------------------------------------------------------------
def make_uniform_partitions_2d(workDir, module, scenario, fullMeshPath):
  this_file_path = pathlib.Path(__file__).parent.absolute()

  for pIt in module.odrom_partitioning_topol[scenario]:
    nTilesX, nTilesY = pIt[0], pIt[1]

    outDir = path_to_partition_info_dir(workDir, nTilesX, nTilesY, "uniform")
    if os.path.exists(outDir):
      print('Partition {} already exists'.format(outDir))
    else:
      print('Generating partition files for {}'.format(outDir))
      os.system('mkdir -p ' + outDir)

      args = ("python3",    str(this_file_path)+'/partition_uniform.py',
              "--wdir",     outDir,
              "--meshPath", fullMeshPath,
              "--tiles",    str(nTilesX), str(nTilesY),
              "--ndpc",     str(module.numDofsPerCell))
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      print(output)


# -------------------------------------------------------------------
def make_full_mesh_for_odrom_using_partition_based_indexing(workDir, pdaDir, module, fomMesh):
  '''
  for FULL od-rom without HR, for performance reasons,
  we don't/should not use the same full mesh used in the fom.
  We need to make a new full mesh with a new indexing
  that is consistent with the partitions and allows continguous storage
  of the state and rhs within each tile
  '''
  totalCells = find_total_cells_from_info_file(fomMesh)

  # find all existing partitions directories inside the workDir
  partitionInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                       if "od_info_" in d]

  # for each one, make the mesh with correct indexing
  for partitionInfoDirIt in partitionInfoDirs:
    # I need to extract an identifier from the direc so that I can
    # use this string to uniquely create a corresponding directory
    # where to store the new mesh
    stringIdentifier = string_identifier_from_partition_info_dir(partitionInfoDirIt)
    outDir = path_to_partition_based_full_mesh_dir(workDir, stringIdentifier)
    if os.path.exists(outDir):
      print('Partition-based full mesh dir {} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # to make the mesh, I need to make an array of gids
      # which in this case is the full gids
      gids = np.arange(0, totalCells)
      np.savetxt(outDir+'/sample_mesh_gids.dat', gids, fmt='%8i')

      print('Generating sample mesh in:')
      print(' {}'.format(outDir))
      meshScriptsDir = pdaDir + "/meshing_scripts"
      args = ("python3", meshScriptsDir+'/create_sample_mesh.py',
              "--fullMeshDir", fomMesh,
              "--sampleMeshIndices", outDir+'/sample_mesh_gids.dat',
              "--outDir", outDir,
              "--useTilingFrom", partitionInfoDirIt)
      popen  = subprocess.Popen(args, stdout=subprocess.PIPE);
      popen.wait()
      output = popen.stdout.read();
      print(output)


# -------------------------------------------------------------------
def do_svd_py(mymatrix, lsvFile, svaFile):
  timing = np.zeros(1)
  start = time.time()
  U,S,_ = linalg.svd(mymatrix, full_matrices=False, lapack_driver='gesdd')
  end = time.time()
  elapsed = end - start
  timing[0] = elapsed
  #print("elapsed ", elapsed)

  #singular values
  #print("Writing sing values to file: {}".format(svaFile))
  np.savetxt(svaFile, S)

  assert(U.flags['F_CONTIGUOUS'])

  # left singular vectors
  fileo = open(lsvFile, "wb")
  # write to beginning of file the extents of the matrix
  #print("  writing POD modes to file: {}".format(lsvFile))
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
  #outDir = os.path.dirname(lsvFile)
  #np.savetxt(lsvFile+'.txt', U[:,:3])
  # np.savetxt(outDir+'/timings.txt', timing)

# -------------------------------------------------------------------
def mapToReferenceRange(x):
  xmax,xmin = np.max(x),np.min(x)
  xmid = 0.5 * ( xmax + xmin )
  return 2. * (x - xmid) / (xmax - xmin)

# -------------------------------------------------------------------
def create_legendre_basis(x, y, order):
  '''
  Create a 2D legendre polynomial basis at coordinates
  specified by lists x and y. "order" specifies the maximum
  order of polynomials in x and y directions
  Outputs M, an N by N_basis, where N is the number of
  coordinates and N_basis = (order+1)^2  is the number of
  basis polynomials
  '''
  assert(len(x)==len(y))
  N = len(x)
  x = np.array(x)
  y = np.array(y)

  # map x,y to [-1,1]
  x = mapToReferenceRange(x)
  y = mapToReferenceRange(y)

  # create legendre polynomials
  Poly = []
  for p in range(order+1):
    Poly.append(legendre(p))

  # evaluate legendre polynomial products
  M = np.zeros((N,(order+1)**2), order='F')
  for px in range(order+1):
    for py in range(order+1):
      icol = px * (order+1) + py
      M[:,icol] = Poly[px](x) * Poly[py](y)

  return M

# -------------------------------------------------------------------
def compute_partition_based_poly_bases_same_order_in_all_tiles(fomMesh, outDir, \
                                                               partInfoDir, targetOrder):
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell

  xcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,1]
  ycoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,2]

  tiles = np.loadtxt(partInfoDir+"/topo.txt")
  nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

  totalModesCount = 0
  modesPerTile = {}
  # loop over each tile
  for tileId in range(nTilesX*nTilesY):
    myFile = partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myRows = np.loadtxt(myFile, dtype=int)

    myX = xcoords[myRows]
    myY = ycoords[myRows]
    lsvFile = outDir + '/lsv_state_p_'+str(tileId)

    U = create_legendre_basis(myX, myY, targetOrder)
    U,_ = np.linalg.qr(U,mode='reduced')

    print("U shape = ", U.shape)
    fileo = open(lsvFile, "wb")
    r=np.int64(U.shape[0])
    np.array([r]).tofile(fileo)
    c=np.int64(U.shape[1])
    np.array([c]).tofile(fileo)
    UT = np.transpose(U)
    UT.tofile(fileo)
    fileo.close()

    modesPerTile[tileId] = U.shape[1]
    totalModesCount += U.shape[1]

  return totalModesCount, modesPerTile

# -------------------------------------------------------------------
def compute_partition_based_poly_bases_to_match_pod(fomMesh, outDir, \
                                                    partInfoDir, \
                                                    podModesPerTileToMatch, \
                                                    tag):
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell
  xcoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,1]
  ycoords = np.loadtxt(fomMesh+"/coordinates.dat")[:,2]
  tiles = np.loadtxt(partInfoDir+"/topo.txt")
  nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

  totalModesCount = 0
  modesPerTile = {}
  for tileId in range(nTilesX*nTilesY):
    myFile = partInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myRows = np.loadtxt(myFile, dtype=int)

    myX = xcoords[myRows]
    myY = ycoords[myRows]
    lsvFile = outDir + '/lsv_state_p_'+str(tileId)

    order = math.ceil(np.sqrt(podModesPerTileToMatch[tileId])) - 1

    U = create_legendre_basis(myX, myY, order)
    U,_ = np.linalg.qr(U,mode='reduced')

    print("tile = {}, podModesCount = {}, polyModesCount = {}".format(tileId, \
                                                                      podModesPerTileToMatch[tileId],
                                                                      U.shape[1]))
    fileo = open(lsvFile, "wb")
    r=np.int64(U.shape[0])
    np.array([r]).tofile(fileo)
    c=np.int64(U.shape[1])
    np.array([c]).tofile(fileo)
    UT = np.transpose(U)
    UT.tofile(fileo)
    fileo.close()

    if tag == -1:
      # I want to only use as many to match exactly the POD
      modesPerTile[tileId] = podModesPerTileToMatch[tileId]
      totalModesCount += podModesPerTileToMatch[tileId]
    elif tag==-2:
      modesPerTile[tileId] = U.shape[1]
      totalModesCount += U.shape[1]
    else:
      sys.exit("Invalid tag = {}".format(tag))

  return totalModesCount, modesPerTile

# -------------------------------------------------------------------
def compute_partition_based_pod(workDir, setId, dataDirs, module, fomMesh):
  '''
  compute pod for both state and rhs using fom train data
  '''
  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell

  # only load snapshots once
  fomStateSnapsFullDomain = load_fom_state_snapshot_matrix(dataDirs, totFomDofs, module.numDofsPerCell)
  fomRhsSnapsFullDomain   = load_fom_rhs_snapshot_matrix(dataDirs,   totFomDofs, module.numDofsPerCell)
  print("pod: fomStateSnapsFullDomain.shape = ", fomStateSnapsFullDomain.shape)
  print("pod:   fomRhsSnapsFullDomain.shape = ", fomRhsSnapsFullDomain.shape)
  print("")

  # with the FOM data loaded for a target setId (i.e. set of runs)
  # loop over all partitions and compute local POD.
  # To do this, we find in workDir all directories with info about partitions
  # which identifies all possible partitions
  partsInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "od_info_" in d]

  for partitionInfoDirIt in partsInfoDirs:
    # I need to extract an identifier from the direc so that I can
    # use this string to uniquely create a corresponding directory
    # where to store the POD data
    stringIdentifier = string_identifier_from_partition_info_dir(partitionInfoDirIt)
    tiles = np.loadtxt(partitionInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

    outDir = path_to_pod_data_dir(workDir, stringIdentifier, setId)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # loop over each tile
      for tileId in range(nTilesX*nTilesY):
        # I need to compute POD for both STATE and RHS
        # using FOM data LOCAL to myself, so need to load
        # which rows of the FOM state I own and use to slice
        myFile = partitionInfoDirIt + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
        myRowsInFullState = np.loadtxt(myFile, dtype=int)

        # after loading the row indices, slice the FOM STATE and RHS snapshots
        # to get only the data that belongs to me.
        myStateSlice = fomStateSnapsFullDomain[myRowsInFullState, :]
        myRhsSlice   = fomRhsSnapsFullDomain[myRowsInFullState, :]
        print("pod: tileId={}".format(tileId))
        print("  stateSlice.shape={}".format(myStateSlice.shape))
        print("  rhsSlice.shape  ={}".format(myRhsSlice.shape))

        # compute svd
        lsvFile = outDir + '/lsv_state_p_'+str(tileId)
        svaFile = outDir + '/sva_state_p_'+str(tileId)
        do_svd_py(myStateSlice, lsvFile, svaFile)

        lsvFile = outDir + '/lsv_rhs_p_'+str(tileId)
        svaFile = outDir + '/sva_rhs_p_'+str(tileId)
        do_svd_py(myRhsSlice, lsvFile, svaFile)

# -------------------------------------------------------------------
def compute_partition_based_entropy_pod(workDir, setId, dataDirs, module, fomMesh):
  print("compute_partition_based_entropy_pod")

  fomTotCells = find_total_cells_from_info_file(fomMesh)
  totFomDofs  = fomTotCells*module.numDofsPerCell

  # load conservative snapshots
  fomStateSnapsFullDomain = load_fom_state_snapshot_matrix(dataDirs, totFomDofs, \
                                                           module.numDofsPerCell)

  # conver to entropy
  fomStateSnapsFullDomainEntropy = conservative_to_entropy(fomStateSnapsFullDomain, \
                                                           module.numDofsPerCell, \
                                                           module.dimensionality)

  print("pod: fomStateSnapsFullDomain.shape = ", fomStateSnapsFullDomain.shape)
  print("")

  # with the FOM data loaded for a target setId (i.e. set of runs)
  # loop over all partitions and compute local POD.
  # To do this, we find in workDir all directories with info about partitions
  # which identifies all possible partitions
  partsInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "od_info_" in d]

  for partitionInfoDirIt in partsInfoDirs:
    # I need to extract an identifier from the direc so that I can
    # use this string to uniquely create a corresponding directory
    # where to store the POD data
    stringIdentifier = string_identifier_from_partition_info_dir(partitionInfoDirIt)
    tiles = np.loadtxt(partitionInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])

    outDir = path_to_pod_data_dir(workDir, stringIdentifier, setId)
    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      os.system('mkdir -p ' + outDir)

      # loop over each tile
      for tileId in range(nTilesX*nTilesY):
        # I need to compute POD for both STATE and RHS
        # using FOM data LOCAL to myself, so need to load
        # which rows of the FOM state I own and use to slice
        myFile = partitionInfoDirIt + "/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
        myRowsInFullState = np.loadtxt(myFile, dtype=int)

        # after loading the row indices, slice the FOM STATE and RHS snapshots
        # to get only the data that belongs to me.
        myStateSlice = fomStateSnapsFullDomainEntropy[myRowsInFullState, :]
        print("pod: tileId={}".format(tileId))
        print("  stateSlice.shape={}".format(myStateSlice.shape))

        # compute svd
        lsvFile = outDir + '/lsv_state_p_'+str(tileId)
        svaFile = outDir + '/sva_state_p_'+str(tileId)
        do_svd_py(myStateSlice, lsvFile, svaFile)

# -------------------------------------------------------------------
def compute_cumulative_energy(svalues, targetPercentage):

  if targetPercentage == 100.:
    return len(svalues)

  else:
    # convert percentage to decimal
    target = float(targetPercentage)/100.
    sSq = np.square(svalues)
    den = np.sum(sSq)
    rsum = 0.
    for i in range(0, len(svalues)):
      rsum += sSq[i]
      ratio = (rsum/den)
      if ratio >= target:
        return i
    return len(svalues)

# -------------------------------------------------------------------
def find_modes_per_tile_from_target_energy(podDir, energy):
  def get_tile_id(stringIn):
    return int(stringIn.split('_')[-1])

  result = {}
  # find all sing values files
  singValsFiles = [podDir+'/'+f for f in os.listdir(podDir) \
                   if "sva_state" in f]
  # sort based on the tile id
  singValsFiles = sorted(singValsFiles, key=get_tile_id)

  for it in singValsFiles:
    singValues = np.loadtxt(it)
    tileId = get_tile_id(it)
    K = compute_cumulative_energy(singValues, energy)
    result[tileId] = max(K, 5)

  totalModesCount = 0
  for k,v in result.items():
    totalModesCount += int(v)

  return totalModesCount, result


# -------------------------------------------------------------------
def make_od_rom_initial_condition(workDir, appObjForIc, \
                                  partitionInfoDir, \
                                  basesDir, modesPerTileDic, \
                                  romSizeOverAllPartitions, \
                                  useEntropy):
  nTiles = len(modesPerTileDic.keys())
  fomIc = None
  if useEntropy:
    fomIcCons = appObjForIc.initialCondition()
    fomIc     = conservative_to_entropy(fomIcCons, \
                                        module.numDofsPerCell, \
                                        module.dimensionality)
  else:
    fomIc = appObjForIc.initialCondition()

  romState = np.zeros(romSizeOverAllPartitions)
  romStateStart = 0
  for tileId in range(nTiles):
    myK             = modesPerTileDic[tileId]
    myPhi           = load_basis_from_binary_file(basesDir + "/lsv_state_p_" + str(tileId) )[:,0:myK]
    myStateRowsFile = partitionInfoDir+"/state_vec_rows_wrt_full_mesh_p_"+str(tileId)+".txt"
    myStateRows     = np.loadtxt(myStateRowsFile, dtype=int)
    myFomIcSlice    = fomIc[myStateRows]
    tmpyhat         = np.dot(myPhi.transpose(), myFomIcSlice)
    romState[romStateStart:romStateStart+myK] = np.copy(tmpyhat)
    romStateStart += myK
  return romState

# -------------------------------------------------------------------
def run_full_od_galerkin_for_all_test_values(workDir, problem, module, \
                                             scenario, fomMeshPath, \
                                             partInfoDir, basesDir, \
                                             energyValue,
                                             polyOrder, \
                                             romSizeOverAllPartitions, \
                                             modesPerTileDic, \
                                             romMeshObj, \
                                             nTiles, setId, \
                                             useEntropy, \
                                             basesKind):

  # this is odrom WITHOUT HR, so the following should hold:
  stencilDofsCount = romMeshObj.stencilMeshSize()*module.numDofsPerCell
  sampleDofsCount  = romMeshObj.sampleMeshSize()*module.numDofsPerCell
  assert(stencilDofsCount == sampleDofsCount)
  fomTotalDofs = stencilDofsCount

  fomMeshObj = pda.load_cellcentered_uniform_mesh(fomMeshPath)
  partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDir)

  param_values = module.test_points[scenario]
  for k, val in param_values.items():

    outDir = workDir + "/odrom_full_"+partitionStringIdentifier+"_"+basesKind
    if polyOrder != None:
      outDir += "_"+str(polyOrder)

    if energyValue != None:
      outDir += "_"+str(energyValue)

    if setId != None:
      outDir += "_set_"+str(setId)

    outDir += "_"+str(k)

    if os.path.exists(outDir):
      print('{} already exists'.format(outDir))
    else:
      print("Running odrom in {}".format(os.path.basename(outDir)))
      os.system('mkdir -p ' + outDir)
      romRunDic    = module.base_dic[scenario]['odrom'].copy()
      coeffDic     = module.base_dic[scenario]['physicalCoefficients'].copy()
      appObjForIc  = None
      appObjForRom = None

      # ----------------------
      # --- 2d swe specific ---
      # ----------------------
      if problem == "2d_swe":
        probId    = pda.Swe2d.SlipWall
        schemeStr = romRunDic['inviscidFluxReconstruction']
        schemeEnu = inviscid_flux_string_to_enum(schemeStr)

        # preset values from problem dic
        gravity  = coeffDic['gravity']
        coriolis = coeffDic['coriolis']
        pulse    = coeffDic['pulsemag']
        # change if something needs to be changed
        if scenario == 1:
          coriolis = val
        else:
          sys.exit("invalid scenario {} for 2d_swe".format(scenario))

        # store
        romRunDic['gravity']  = gravity
        romRunDic['coriolis'] = coriolis
        romRunDic['pulse']    = pulse
        appObjForIc  = pda.create_problem(fomMeshObj, probId, schemeEnu, gravity, coriolis, pulse)
        appObjForRom = pda.create_problem(romMeshObj, probId, schemeEnu, gravity, coriolis, pulse)

      # ----------------------
      # --- 2d gs specific ---
      # ----------------------
      elif problem == "2d_gs":
        probId   = pda.DiffusionReaction2d.GrayScott
        diff_A   = coeffDic['diffusionA']
        diff_B   = coeffDic['diffusionB']
        feedRate = coeffDic['feedRate']
        killRate = coeffDic['killRate']

        if scenario == 1:
          killRate = val
        elif scenario == 2:
          feedRate = val
        else:
          sys.exit("invalid scenario {} for 2d_gs".format(scenario))

        romRunDic['diffusionA'] = diff_A
        romRunDic['diffusionB'] = diff_B
        romRunDic['feedRate']   = feedRate
        romRunDic['killRate']   = killRate
        scheme = pda.ViscousFluxReconstruction.FirstOrder
        appObjForIc  = pda.create_problem(fomMeshObj, probId, scheme, diff_A, diff_B, feedRate, killRate)
        appObjForRom = pda.create_problem(romMeshObj, probId, scheme, diff_A, diff_B, feedRate, killRate)

      # ---------------------------
      # --- 2d burgers specific ---
      # ---------------------------
      elif problem == "2d_burgers":
        probId      = pda.AdvectionDiffusion2d.Burgers
        schemeStr   = romRunDic['inviscidFluxReconstruction']
        schemeEnu   = inviscid_flux_string_to_enum(schemeStr)
        pulsemag    = coeffDic['pulsemag']
        pulsespread = coeffDic['pulsespread']
        diffusion   = coeffDic['diffusion']
        pulsecenter = coeffDic['pulsecenter']

        if scenario == 1:
          pulsespread = val
        else:
          sys.exit("invalid scenario {} for 2d_burgers".format(scenario))

        romRunDic['pulsemag']    = pulsemag
        romRunDic['pulsespread'] = pulsespread
        romRunDic['diffusion']   = diffusion
        romRunDic['pulsecenter'] = pulsecenter
        appObjForIc  = pda.create_problem(fomMeshObj, probId, schemeEnu, \
                                          pulsemag, pulsespread, diffusion, \
                                          pulsecenter[0], pulsecenter[1])
        appObjForRom = pda.create_problem(romMeshObj, probId, schemeEnu, \
                                          pulsemag, pulsespread, diffusion, \
                                          pulsecenter[0], pulsecenter[1])

      # ----------------------
      # --- 2d rti specific ---
      # ----------------------
      elif problem == "2d_rti":
        probId    = pda.Euler2d.RayleighTaylor
        schemeStr = romRunDic['inviscidFluxReconstruction']
        schemeEnu = inviscid_flux_string_to_enum(schemeStr)
        amplitude = coeffDic['amplitude']

        if scenario == 1:
          amplitude = val
        else:
          sys.exit("invalid scenario {} for 2d_rti".format(scenario))

        romRunDic['amplitude']  = amplitude
        appObjForIc  = pda.create_problem(fomMeshObj, probId, schemeEnu, amplitude)
        appObjForRom = pda.create_problem(romMeshObj, probId, schemeEnu, amplitude)

      # --------------------------------
      # --- 2d euler smooth specific ---
      # --------------------------------
      elif problem == "2d_eulersmooth":
        probId    = pda.Euler2d.PeriodicSmooth
        schemeStr = romRunDic['inviscidFluxReconstruction']
        schemeEnu = inviscid_flux_string_to_enum(schemeStr)

        if scenario == 1:
          pass
        else:
          sys.exit("invalid scenario {} for 2d_rti".format(scenario))

        appObjForIc  = pda.create_problem(fomMeshObj, probId, schemeEnu)
        appObjForRom = pda.create_problem(romMeshObj, probId, schemeEnu)

      else:
        sys.exit("Invalid problem = {}".format(problem))

      # ----------------------------------------
      # nothing should be changed below because
      # the following is problem-independent
      f = open(outDir+"/rom_dofs_count.txt", "w")
      f.write(str(romSizeOverAllPartitions))
      f.close()

      np.savetxt(outDir+"/modes_per_tile.txt", \
                 np.array(list(modesPerTileDic.values())),
                 fmt="%5d")

      # if basesKind == "using_pod_bases":
      #   romRunDic['energy'] = energyValueOrPolyOrder
      # elif basesKind == "using_poly_bases":
      #   romRunDic['poly_order'] = energyValueOrPolyOrder
      # else:
      #   sys.exit("Invalid kind = {}".format(basesKind))

      romRunDic['partioningInfo'] = partInfoDir
      inputFile = outDir + "/input.yaml"
      write_dic_to_yaml_file(inputFile, romRunDic)

      # these should be valid
      assert(appObjForIc  != None)
      assert(appObjForRom != None)

      romState = make_od_rom_initial_condition(workDir, appObjForIc, partInfoDir, \
                                               basesDir, modesPerTileDic, \
                                               romSizeOverAllPartitions, \
                                               useEntropy)

      # construct odrom object
      odRomObj = OdRomFull(basesDir, fomTotalDofs, modesPerTileDic, \
                           module.dimensionality, module.numDofsPerCell, \
                           partInfoDir, useEntropy)
      # initial condition
      odRomObj.reconstructMemberFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_ic.txt", odRomObj.viewFomState())

      # time loop
      dt         = romRunDic['dt']
      finalTime  = romRunDic['finalTime']
      numSteps   = int(finalTime/dt)
      odeScheme  = romRunDic['odeScheme']

      pTimeStart = time.time()
      if odeScheme == "SSPRK3":
        odRomObj.runSSPRK3(outDir, romState, appObjForRom, numSteps, dt)
      elif odeScheme in ["RungeKutta4", "RK4"]:
        assert(useEntropy == False)
        odRomObj.runRK4(outDir, romState, appObjForRom, numSteps, dt)
      elif odeScheme in ["RungeKutta2", "RK2"]:
        assert(useEntropy == False)
        odRomObj.runRK2(outDir, romState, appObjForRom, numSteps, dt)

      elapsed = time.time() - pTimeStart
      print("elapsed = {}".format(elapsed))
      f = open(outDir+"/timing.txt", "w")
      f.write(str(elapsed))
      f.close()

      # reconstruct final state
      odRomObj.reconstructMemberFomStateFullMeshOrdering(romState)
      np.savetxt(outDir+"/y_rec_final.txt", odRomObj.viewFomState())


# -------------------------------------------------------------------
def run_full_od_galerkin_pod_bases(workDir, problem, module, scenario, \
                                   fomMeshPath, useEntropy=False):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in partInfoDirs:
    tiles = np.loadtxt(partInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    nTiles = nTilesX*nTilesY
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # for each one, find the associated full mesh with the indexiging
    # suitable for doing the ODROM. Note: each decomposition should
    # have a unique full mesh associated with it.
    topoString = str(nTilesX)+"x"+str(nTilesY)
    odMeshDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if topoString in d and "full_mesh" in d]
    assert(len(odMeshDirs)==1)
    odRomMeshDir = odMeshDirs[0]
    # the mesh found becomes the mesh to use for the odrom.
    # we can make a mesh object for this decomposition once
    # and used for all the runs below
    odRomMeshObj = pda.load_cellcentered_uniform_mesh(odRomMeshDir)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    howManySets = len(module.odrom_basis_sets[scenario].keys())
    for setId in range(howManySets):
      currPodDir = path_to_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3: over all target energies
      # ------
      for energyValue in module.odrom_energies[scenario]:
        totalRomSize, modesPerTileDic = find_modes_per_tile_from_target_energy(currPodDir, \
                                                                               energyValue)
        print(totalRomSize)
        print(modesPerTileDic)

        # -------
        # loop 4: over all test values
        # ------
        run_full_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                 scenario, fomMeshPath, \
                                                 partInfoDirIt, currPodDir, \
                                                 energyValue, None, \
                                                 totalRomSize, modesPerTileDic, \
                                                 odRomMeshObj, \
                                                 nTiles, setId, \
                                                 useEntropy, \
                                                 "using_pod_bases")

# -------------------------------------------------------------------
def run_od_galerkin_same_poly_bases_in_all_tiles(workDir, problem, module, scenario, \
                                                 fomMeshPath, polyOrders):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in partInfoDirs:
    tiles = np.loadtxt(partInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    nTiles = nTilesX*nTilesY
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # for each one, find the associated full mesh with the indexiging
    # suitable for doing the ODROM. Note: each decomposition should
    # have a unique full mesh associated with it.
    topoString = str(nTilesX)+"x"+str(nTilesY)
    odMeshDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if topoString in d and "full_mesh" in d]
    assert(len(odMeshDirs)==1)
    odRomMeshDir = odMeshDirs[0]
    # the mesh found becomes the mesh to use for the odrom.
    # we can make a mesh object for this decomposition once
    # and used for all the runs below
    odRomMeshObj = pda.load_cellcentered_uniform_mesh(odRomMeshDir)

    # -------
    # loop 2: over all target orders
    # ------
    for orderIt in polyOrders:
      polyBasesDir = path_to_poly_bases_data_dir(workDir, partitionStringIdentifier, orderIt)
      if os.path.exists(polyBasesDir):
        print('{} already exists'.format(polyBasesDir))
      else:
        os.system('mkdir -p ' + polyBasesDir)
        totalRomSize, modesPerTileDic = compute_partition_based_poly_bases_same_order_in_all_tiles(fomMeshPath,\
                                                                                                   polyBasesDir, \
                                                                                                   partInfoDirIt, \
                                                                                                   orderIt)

        # -------
        # loop 3: over all test values
        # ------
        run_full_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                 scenario, fomMeshPath, \
                                                 partInfoDirIt, polyBasesDir, \
                                                 None, orderIt, \
                                                 totalRomSize, modesPerTileDic, \
                                                 odRomMeshObj, \
                                                 nTiles,
                                                 None, \
                                                 False, \
                                                 "using_poly_bases")


# -------------------------------------------------------------------
def run_od_galerkin_poly_bases_variable_order_per_tile(workDir, problem, \
                                                       module, scenario, \
                                                       fomMeshPath, tag):

  partInfoDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if "od_info" in d]
  # -------
  # loop 1: over all decompositions
  # ------
  for partInfoDirIt in partInfoDirs:
    tiles = np.loadtxt(partInfoDirIt+"/topo.txt")
    nTilesX, nTilesY = int(tiles[0]), int(tiles[1])
    nTiles = nTilesX*nTilesY
    partitionStringIdentifier = string_identifier_from_partition_info_dir(partInfoDirIt)

    # for each one, find the associated full mesh with the indexiging
    # suitable for doing the ODROM. Note: each decomposition should
    # have a unique full mesh associated with it.
    topoString = str(nTilesX)+"x"+str(nTilesY)
    odMeshDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                  if topoString in d and "full_mesh" in d]
    assert(len(odMeshDirs)==1)
    odRomMeshDir = odMeshDirs[0]
    # the mesh found becomes the mesh to use for the odrom.
    # we can make a mesh object for this decomposition once
    # and used for all the runs below
    odRomMeshObj = pda.load_cellcentered_uniform_mesh(odRomMeshDir)

    # -------
    # loop 2: over all POD computed from various sets of train runs
    # ------
    howManySets = len(module.odrom_basis_sets[scenario].keys())
    for setId in range(howManySets):
      currPodDir = path_to_pod_data_dir(workDir, partitionStringIdentifier, setId)

      # -------
      # loop 3: over all target POD energies
      # ------
      for energyValue in module.odrom_energies[scenario]:
        totalRomSizePod, modesPerTilePod = find_modes_per_tile_from_target_energy(currPodDir, \
                                                                                  energyValue)
        print(totalRomSizePod)
        print(modesPerTilePod)

        # now that we know POD modes per tile, we create local poly bases
        # such that the order yields num of modes that matches POD bases
        polyBasesDir = path_to_poly_bases_data_dir(workDir, partitionStringIdentifier, tag, energyValue, setId)
        if os.path.exists(polyBasesDir):
          print('{} already exists'.format(polyBasesDir))
        else:
          os.system('mkdir -p ' + polyBasesDir)

          totalRomSize, modesPerTileDic = compute_partition_based_poly_bases_to_match_pod(fomMeshPath,\
                                                                                          polyBasesDir, \
                                                                                          partInfoDirIt, \
                                                                                          modesPerTilePod,\
                                                                                          tag)
          # -------
          # loop 4: over all test values
          # ------
          run_full_od_galerkin_for_all_test_values(workDir, problem, module, \
                                                   scenario, fomMeshPath, \
                                                   partInfoDirIt, polyBasesDir, \
                                                   energyValue, tag, totalRomSize, \
                                                   modesPerTileDic, \
                                                   odRomMeshObj, \
                                                   nTiles,
                                                   setId, \
                                                   False, \
                                                   "using_poly_bases")

#==============================================================
# main
#==============================================================
if __name__ == '__main__':
  parser   = ArgumentParser()
  parser.add_argument("--wdir", "--workdir",
                      dest="workdir",
                      required=True)
  parser.add_argument("--pdadir",
                      dest="pdadir",
                      required=True)
  parser.add_argument("--problem",
                      dest="problem",
                      required=True)
  parser.add_argument("--scenario", "-s",
                      dest="scenario",
                      type=int,
                      required=True)
  parser.add_argument("--mesh",
                      nargs=2, \
                      dest="mesh", \
                      type=int, \
                      required=True)

  args     = parser.parse_args()
  workDir  = args.workdir
  pdaDir   = args.pdadir
  problem  = args.problem
  scenario = args.scenario
  meshSize = args.mesh

  if not os.path.exists(workDir):
    print("Working dir {} does not exist, creating it".format(workDir))
    os.system('mkdir -p ' + workDir)

  # write scenario id, problem to file
  write_scenario_to_file(scenario, workDir)
  write_problem_name_to_file(problem, workDir)

  print("========================")
  print("Importing problem module")
  print("========================")
  module = importlib.import_module(problem)

  try:    print("{} has dimensionality = {}".format(problem, module.dimensionality))
  except: sys.exit("Missing dimensionality in problem's module")

  try:    print("{} has numDofsPerCell = {}".format(problem, module.numDofsPerCell))
  except: sys.exit("Missing numDofsPerCell in problem's module")

  print(module)
  print("")

  print("========================")
  print("Make full mesh ")
  print("========================")
  make_fom_mesh_if_not_existing(workDir, problem, \
                                module, scenario, \
                                pdaDir, meshSize)
  print("")

  # before we move on, we need to ensure that in workDir
  # there is a unique FULL mesh. This is because the mesh is specified
  # via command line argument and must be unique for a scenario.
  # If one wants to run for a different mesh, then they have to
  # run this script again with a different working directory
  fomMeshPath = find_full_mesh_and_ensure_unique(workDir)

  print("========================")
  print("Train runs")
  print("========================")
  run_foms(workDir, problem, module, scenario, "train", fomMeshPath)
  print("")

  print("========================")
  print("Test runs")
  print("========================")
  run_foms(workDir, problem, module, scenario, "test", fomMeshPath)
  print("")

  print("========================")
  print("Compute partitions")
  print("========================")
  # loop over all target styles for partitioning:
  # the simplest and currently only supported is uniform
  # we can add more. We might add a new partitioning methods
  # that use FOM train data so the partitioning stage
  # makes sense to be after the FOM runs
  for partitioningStyleIt in module.odrom_partitioning_style[scenario]:
    if module.dimensionality == 2 and partitioningStyleIt == "uniform":
      make_uniform_partitions_2d(workDir, module, scenario, fomMeshPath)
    else:
      sys.exit("Invalid dimensionality or partitionStyle = {}".format(module.dimensionality, \
                                                                      partitioningStyleIt))
  print("")

  print("=====================================")
  print("Compute partition-based full meshes ")
  print("=====================================")
  # for FULL od-rom without HR, for performance reasons,
  # we don't/should not use the same full mesh used in the fom.
  # We need to make a new full mesh with a new indexing
  # that is consistent with the partitions and allows continguous storage
  # of the state and rhs within each tile
  make_full_mesh_for_odrom_using_partition_based_indexing(workDir, pdaDir, module, fomMeshPath)
  print("")


  '''
  The pod modes for each tile must be computed in two cases:
  1. we explicitly want podGalerkin
  2. if PolyGalerkin is on and the poly_order = -1, -2, in which case
     we computed the poly order in each tile based on the local pod modes
  '''
  mustComputeTiledPodModes = False
  if "PodGalerkinFull" in module.odrom_algos[scenario]:
    mustComputeTiledPodModes = True

  if "PolyGalerkinFull" in module.odrom_algos[scenario] and \
     (-1 in module.odrom_poly_order[scenario] or \
      -2 in module.odrom_poly_order[scenario]):
    mustComputeTiledPodModes = True

  if mustComputeTiledPodModes:
    print("=====================================")
    print("Compute partition-based POD")
    print("=====================================")
    for setId, trainIndices in module.odrom_basis_sets[scenario].items():
      print("")
      print("------------------------")
      print("Handling setId = {}".format(setId))
      print("------------------------")
      trainDirs = [workDir+'/'+d for d in os.listdir(workDir) \
                   if "train" in d and get_run_id(d) in trainIndices]
      assert(len(trainDirs) == len(trainIndices))

      # if "GalerkinEntropy" in module.odrom_algos[scenario]:
      #   assert(problem == "2d_rti")
      #   compute_partition_based_entropy_pod(workDir, setId, trainDirs, module, fomMeshPath)
      # else:
      compute_partition_based_pod(workDir, setId, trainDirs, module, fomMeshPath)

    print("")


  print("=====================================")
  print("Run FULL od-galerkin if needed ")
  print("=====================================")
  if "PolyGalerkinFull" in module.odrom_algos[scenario]:
    # first run the case where each partition has same poly bases,
    # so make list of all the orders we want that are != -1
    polyOrders = [i for i in module.odrom_poly_order[scenario] if i > 0]
    run_od_galerkin_same_poly_bases_in_all_tiles(workDir, problem, module, scenario, \
                                                 fomMeshPath, polyOrders)

    # check if -1 is is part of the order list, and if so that means
    # we need to use polyn bases such that in each tile we decide the order
    # based on the number of POD modes to have a fair comparison
    if -1 in module.odrom_poly_order[scenario]:
      run_od_galerkin_poly_bases_variable_order_per_tile(workDir, problem, module, \
                                                         scenario, \
                                                         fomMeshPath,
                                                         -1)
    if -2 in module.odrom_poly_order[scenario]:
      run_od_galerkin_poly_bases_variable_order_per_tile(workDir, problem, module, \
                                                         scenario, \
                                                         fomMeshPath,
                                                         -2)


  if "PodGalerkinFull" in module.odrom_algos[scenario]:
    run_full_od_galerkin_pod_bases(workDir, problem, module, scenario, fomMeshPath)

  # elif "GalerkinEntropy" in module.odrom_algos[scenario]:
  #   run_full_od_galerkin(workDir, problem, module, scenario, fomMeshPath, useEntropy=True)
  # else:
  #   sys.exit()
  print("")
