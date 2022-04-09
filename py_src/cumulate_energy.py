
import numpy as np
import sys, os
from decimal import Decimal

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
