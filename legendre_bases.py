
import numpy as np
from scipy.special import legendre

# -------------------------------------------------------------------
def mapToReferenceRange(x):
  xmax,xmin = np.max(x),np.min(x)
  xmid = 0.5 * ( xmax + xmin )
  return 2. * (x - xmid) / (xmax - xmin)

# -------------------------------------------------------------------
class LegendreBases2d:
  def __init__(self, policy):
    self.policy_ = policy
    assert(policy in ["maxOrder", "totalOrder"])

  def totalBasesCount(self, order):
    if self.policy_ == "maxOrder":
      return (order+1)**2
    elif self.policy_ == "totalOrder":
      return int((order+2)*(order+1)/2)

  def findClosestOrderToMatchTargetBasesCount(self, count):
    if self.policy_ == "maxOrder":
      return np.ceil(np.sqrt(targetCount))-1
    elif self.policy_ == "totalOrder":
      return int((-3 + np.sqrt(9. + 4.*count*2))/2)

  def __call__(self, order, x, y):
    x, y = np.array(x), np.array(y)
    assert(len(x)==len(y))
    N = len(x)
    x,y = mapToReferenceRange(x), mapToReferenceRange(y)

    Poly = []
    for p in range(order+1):
      Poly.append(legendre(p))

    if self.policy_ == "maxOrder":
      N_basis = self.totalBasesCount(order)
      M = np.zeros((N, N_basis), order='F')
      for px in range(order+1):
        for py in range(order+1):
          icol = px * (order+1) + py
          M[:,icol] = Poly[px](x) * Poly[py](y)
      return M

    elif self.policy_ == "totalOrder":
      N_basis = self.totalBasesCount(order)
      M = np.zeros((N,N_basis), order='F')
      icol = 0
      for px in range(order+1):
        for py in range(order+1-px):
          M[:,icol] = Poly[px](x) * Poly[py](y)
          icol += 1
      return M
