#!/usr/bin/env python

#==============================================================
# imports
#==============================================================

from argparse import ArgumentParser
import numpy as np

#==============================================================
# functions
#==============================================================

def compute_cumulative_energy(s, target):
  sSq = np.square(s)
  den = np.sum(sSq)
  rsum = 0.
  for i in range(0, len(s)):
    rsum += sSq[i]
    ratio = (rsum/den)
    if ratio >= target:
      return i
  return len(s)

#==============================================================
# main
#==============================================================
if __name__== "__main__":
  parser = ArgumentParser()
  parser.add_argument("--singvals",
                      dest="svFile",
                      required=True,
                      help="Path to file with sing values.")

  parser.add_argument("--percent", "-p",
                      dest="pct",
                      required=True,
                      type=np.float,
                      help="Target percetange: e.g. 99.99")
  # parse all args
  args = parser.parse_args()
  assert(args.svFile != "empty")

  # convert percentage to decimal
  target = float(args.pct)/100.

  # load data
  sv = np.loadtxt(args.svFile)

  # compute cumulative energy
  if (target == 1.):
    n = len(sv)
    print ("Nv: {}".format(n) )
  else:
    n = compute_cumulative_energy(sv, target)

  print ("Nv: {}".format(n) )
  print ("numBasis: {}".format(n) )
