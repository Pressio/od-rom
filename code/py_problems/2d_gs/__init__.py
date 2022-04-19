
import pressiodemoapps as pda
import sys
from .scenarios import *

dimensionality = 2
numDofsPerCell = 2

# -------------------------------------------------------------------
def create_problem_for_scenario(scenario, meshObj, coeffDic, dicIn, val):
  diff_A   = coeffDic['diffusionA']
  diff_B   = coeffDic['diffusionB']
  feedRate = coeffDic['feedRate']
  killRate = coeffDic['killRate']

  if scenario in [1,2]:
    feedRate = val
  else:
    sys.exit("invalid scenario {} for 2d_gs".format(scenario))

  dicIn['diffusionA'] = diff_A
  dicIn['diffusionB'] = diff_B
  dicIn['feedRate']   = feedRate
  dicIn['killRate']   = killRate
  scheme = pda.ViscousFluxReconstruction.FirstOrder
  appObj = pda.create_gray_scott_2d_problem(meshObj, scheme, \
                                            diff_A, diff_B, \
                                            feedRate, killRate)
  return appObj

# -------------------------------------------------------------------
def custom_tuple_args_for_fom_mesh_generation(scenario):
  mypart = ("--bounds", str(-1.25), str(1.25), str(-1.25), str(1.25), \
            "--periodic", "x", "y")
  return mypart
