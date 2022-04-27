
import sys, logging
import pressiodemoapps as pda
from .final_scenarios import *
from .wip_scenarios import *

dimensionality = 2
numDofsPerCell = 2

# -------------------------------------------------------------------
def create_problem_for_scenario(scenario, meshObj, coeffDic, dicIn, val):
  logger = logging.getLogger(__name__)

  diffA   = coeffDic['diffusionA']
  diffB   = coeffDic['diffusionB']
  feedRate = coeffDic['feedRate']
  killRate = coeffDic['killRate']

  if isinstance(val, list):
    # if here, val is a list so multiple parameters are changing
    # if a param needs to be set, it is defined in the scenario
    # something line: p-k where k is the index we need to use
    # to find the corresponding value in val
    if isinstance(diffA, str):
      index = int(diffA[-1])
      diffA = val[index]
    if isinstance(diffB, str):
      index = int(diffB[-1])
      diffB = val[index]
    if isinstance(feedRate, str):
      index = int(feedRate[-1])
      feedRate = val[index]
    if isinstance(killRate, str):
      index = int(killRate[-1])
      killRate = val[index]

  else:
    # if here, one one param is varying, find which one
    if isinstance(diffA, str):
      diffA = val
    if isinstance(diffB, str):
      diffB = val
    if isinstance(feedRate, str):
      feedRate = val
    if isinstance(killRate, str):
      killRate = val

  if not isinstance(diffA, float):
    logger.error("2d_gs: diffA not a float")
    sys.exit(1)
  if not isinstance(diffB, float):
    logger.error("2d_gs: diffB not a float")
    sys.exit(1)
  if not isinstance(feedRate, float):
    logger.error("2d_gs: feedRate not a float")
    sys.exit(1)
  if not isinstance(killRate, float):
    logger.error("2d_gs: killRate not a float")
    sys.exit(1)

  dicIn['diffusionA'] = diffA
  dicIn['diffusionB'] = diffB
  dicIn['feedRate']   = feedRate
  dicIn['killRate']   = killRate
  scheme = pda.ViscousFluxReconstruction.FirstOrder
  appObj = pda.create_gray_scott_2d_problem(meshObj, scheme, \
                                            diffA, diffB, feedRate, killRate)
  return appObj

# -------------------------------------------------------------------
def custom_tuple_args_for_fom_mesh_generation(scenario):
  mypart = ("--bounds", str(-1.25), str(1.25), str(-1.25), str(1.25), \
            "--periodic", "x", "y")
  return mypart
