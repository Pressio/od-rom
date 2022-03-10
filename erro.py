# standard modules
from argparse import ArgumentParser
import sys, os, importlib, pathlib, math
import random, subprocess
import matplotlib.pyplot as plt
import re, os, time, yaml
import numpy as np
from scipy import linalg as scipyla
from decimal import Decimal
from scipy import optimize as sciop

from myio import *


if __name__ == '__main__':
  wdir = "/"

  fomRhsDir = wdir + '/fom_train_0'
  projecDir = wdir + '/partition_based_2x2_uniform_quad_projector_99.9999999_set_0_psampling_.05'
  phiPodDir = wdir + '/partition_based_2x2_uniform_full_state_pod_set_0'

  fSnapsFullDomain = load_fom_rhs_snapshot_matrix(fomTrainDirs, totFomDofs, numDofsPerCell)

  nTiles =
  for tileId in range(nTiles):
    myK = 

    myPhiFullMesh = load_basis_from_binary_file(phiFile)[:,0:myK]


    myProjector = np.loadtxt(projectorDir+'/projector_p_'+str(tileId)+'.txt')


    # indexing info
    cellGidsFile   = partitionInfoDir + "/cell_gids_wrt_full_mesh_p_"+str(tileId)+".txt"
    myCellGids     = np.loadtxt(cellGidsFile, dtype=int)
    sampleGidsFile = sampleMeshDir + "/sample_mesh_gids_p_"+str(tileId)+".txt"
    mySmMeshGids   = np.loadtxt(sampleGidsFile, dtype=int)
    mySmCount      = len(mySmMeshGids)
    print(numDofsPerCell* mySmCount)
    print(fSnapsFullDomain.shape[1])
    commonElem  = set(mySmMeshGids).intersection(myCellGids)
    commonElem  = np.sort(list(commonElem))
    mylocalinds = np.searchsorted(myCellGids, commonElem)
    myfSnapsSampleMesh = np.zeros((mySmCount*numDofsPerCell, myRhsSnaps.shape[1]), order='F')
    for j in range(numDofsPerCell):
      myfSnapsSampleMesh[j::numDofsPerCell, :] = myRhsSnaps[numDofsPerCell*mylocalinds + j, :]


    A = phi.T * myfFullMesh
    B = myProjector.transpose() * myfSampleMesh
