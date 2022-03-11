
import numpy as np

# -------------------------------------------------------------------
def entropy_to_conservative(V0, numDofsPerCell, physDim):
  rankIn = V0.ndim
  ncells = int(V0.shape[0]/numDofsPerCell)

  if rankIn == 1:
    V = np.reshape(V0, (numDofsPerCell, ncells), order='F')

  elif rankIn == 2:
    ncols  = V0.shape[1]
    V = np.reshape(V0, (numDofsPerCell, ncells, ncols), order='F')

  U = np.zeros(np.shape(V))
  gamma = 5./3.
  gamma1 = gamma - 1.
  igamma1 = 1./gamma1
  gmogm1 = gamma*igamma1

  if (physDim == 2):
    iu3 = 1./V[3]  #- p / rho
    u = -iu3*V[1]
    v = -iu3*V[2]
    t0 = -0.5*iu3*(V[1]**2 + V[2]**2)
    t1 = V[0] - gmogm1 + t0
    t2 =np.exp(-igamma1*np.log(-V[3]) )
    t3 = np.exp(t1)
    U[0] = t2*t3
    H = -iu3*(gmogm1 + t0)
    E = (H + iu3)
    U[1] = U[0]*u
    U[2] = U[0]*v
    U[3] = U[0]*E

  elif (physDim == 1):
    iu3 = 1./V[2]  #- p / rho
    u = -iu3*V[1]
    t0 = -0.5*iu3*(V[1]**2)
    t1 = V[0] - gmogm1 + t0
    t2 =np.exp(-igamma1*np.log(-V[2]) )
    t3 = np.exp(t1)
    U[0] = t2*t3
    H = -iu3*(gmogm1 + t0)
    E = (H + iu3)
    U[1] = U[0]*u
    U[2] = U[0]*E

  U = np.reshape(U, V0.shape, order='F')
  return U

# -------------------------------------------------------------------
def conservative_to_entropy(Min, numDofsPerCell, physDim):
  # fix this
  gamma = 5./3.
  gammaMinusOne = gamma - 1.

  ncells = int(Min.shape[0]/numDofsPerCell)
  rankIn = Min.ndim
  if rankIn == 1:
    M = np.reshape(Min, (numDofsPerCell, ncells), order='F')

  elif rankIn == 2:
    ncols  = Min.shape[1]
    M = np.reshape(Min, (numDofsPerCell, ncells, ncols), order='F')

  else:
    sys.exit("dmfasfd")

  V = np.zeros(np.shape(M))
  if (physDim == 2):
    assert(M.shape[0] == 4)
    p = (gammaMinusOne)*(M[3] - 0.5*M[1]**2/M[0] - 0.5*M[2]**2/M[0] )
    s = np.log(p) - gamma*np.log(M[0])
    V[0] = -s/(gammaMinusOne) + (gamma + 1.)/(gammaMinusOne) - M[3]/p
    V[1] = M[1]/p
    V[2] = M[2]/p
    V[3] = -M[0]/p

  elif (physDim == 1):
    assert(M.shape[0] == 4)
    p = (gammaMinusOne)*(M[2] - 0.5*M[1]**2/M[0])
    s = np.log(p) - gamma*np.log(M[0])
    V[0] = -s/(gammaMinusOne) + (gamma + 1.)/(gammaMinusOne) - M[2]/p
    V[1] = M[1]/p
    V[2] = -M[0]/p

  V = np.reshape(V, Min.shape, order='F')
  return V


# -------------------------------------------------------------------
def computedUdV(U, numDofsPerCell, physicalDim):
  V = conservative_to_entropy(U, numDofsPerCell, physicalDim)

  nCells = int(V.size / numDofsPerCell)
  V = V.reshape( (numDofsPerCell, nCells),order='F')
  U = U.reshape( (numDofsPerCell, nCells),order='F')

  if (U.ndim > 1):
    nSpace = np.shape(U)[-1]
  else:
    nSpace = 0

  gamma = 5./3.
  if (physicalDim == 2):
    if (nSpace == 0):
      dUdV = np.zeros((4,4))
    else:
      dUdV = np.zeros((4,4,nSpace))

    p = (gamma - 1.)*(U[3] - 0.5*U[1]**2/U[0] - 0.5*U[2]**2/U[0] )
    H = (U[3] + p) / U[0]
    asqr = gamma*p/U[0]
    dUdV[0,:] = U[:]
    dUdV[1,0] = dUdV[0,1]
    dUdV[1,1] = U[1]**2/U[0] + p
    dUdV[1,2] = -U[1]*V[2]/V[3]
    dUdV[1,3] = -U[1]/V[3] - V[1]*U[3]/V[3]
    dUdV[2,0] = dUdV[0,2]
    dUdV[2,1] = dUdV[1,2]
    dUdV[2,2] = U[2]**2/U[0] + p
    dUdV[2,3] = -U[2]/V[3] - V[2]*U[3]/V[3]
    dUdV[3,0] = dUdV[0,3]
    dUdV[3,1] = dUdV[1,3]
    dUdV[3,2] = dUdV[2,3]
    dUdV[3,3] = U[0]*H**2 - asqr*p/(gamma - 1.)

  elif physicalDim == 1:
    if (nSpace == 0):
      dUdV = np.zeros((3,3))
    else:
      dUdV = np.zeros((3,3,nSpace))
    p = (gamma - 1.)*(U[2] - 0.5*U[1]**2/U[0] )
    H = (U[2] + p) / U[0]
    asqr = gamma*p/U[0]
    dUdV[0,:] = U[:]
    dUdV[1,0] = dUdV[0,1]
    dUdV[1,1] = U[1]**2/U[0] + p
    dUdV[1,2] = -U[1]/V[2] - V[1]*U[2]/V[2]
    dUdV[2,0] = dUdV[0,2]
    dUdV[2,1] = dUdV[1,2]
    dUdV[2,2] = U[0]*H**2 - asqr*p/(gamma - 1.)

  return dUdV
