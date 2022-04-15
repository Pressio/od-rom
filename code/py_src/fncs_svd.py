
import numpy as np
import sys, os, time, logging
from scipy import linalg as scipyla

# -------------------------------------------------------------------
def do_svd_py(mymatrix, lsvFile, svaFile):
  logger = logging.getLogger(__name__)

  timing = np.zeros(1)
  start = time.time()
  U,S,_ = scipyla.svd(mymatrix, full_matrices=False, lapack_driver='gesdd')
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
