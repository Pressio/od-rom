

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
