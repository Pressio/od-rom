
import sys, logging
import pressiodemoapps as pda
from .final_scenarios import *
from .wip_scenarios import *

dimensionality = 2
numDofsPerCell = 3

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

# -------------------------------------------------------------------
def create_problem_for_scenario(scenario, meshObj, coeffDic, dicIn, val):
  logger = logging.getLogger(__name__)

  schemeStr = dicIn['inviscidFluxReconstruction']
  schemeEnu = inviscid_flux_string_to_enum(schemeStr)
  # preset values from problem dic
  gravity  = coeffDic['gravity']
  coriolis = coeffDic['coriolis']
  pulse    = coeffDic['pulsemag']

  if isinstance(val, list):
    # if here, val is a list so multiple parameters are changing
    # if a param needs to be set, it is defined in the scenario
    # something line: p-k where k is the index we need to use
    # to find the corresponding value in val
    if isinstance(gravity, str):
      index = int(gravity[-1])
      gravity = val[index]
    if isinstance(coriolis, str):
      index = int(coriolis[-1])
      coriolis = val[index]
    if isinstance(pulse, str):
      index = int(pulse[-1])
      pulse = val[index]

  else:
    # if here, one one param is varying, find which one
    if isinstance(gravity, str):
      gravity = val
    elif isinstance(coriolis, str):
      coriolis = val
    elif isinstance(pulse, str):
      pulse = val

  if not isinstance(gravity, float):
    logger.error("2d_swe: gravity not a float")
    sys.exit(1)
  if not isinstance(coriolis, float):
    logger.error("2d_swe: coriolis not a float")
    sys.exit(1)
  if not isinstance(pulse, float):
    logger.error("2d_swe: pulse not a float")
    sys.exit(1)

  dicIn['gravity']  = gravity
  dicIn['coriolis'] = coriolis
  dicIn['pulse']    = pulse
  appObj  = pda.create_slip_wall_swe_2d_problem(meshObj, schemeEnu, \
                                               gravity, coriolis, pulse)
  return appObj

# -------------------------------------------------------------------
def custom_tuple_args_for_fom_mesh_generation(scenario):
  schemeString = base_dic[scenario]['fom']['inviscidFluxReconstruction']
  stencilSize  = inviscid_flux_string_to_stencil_size(schemeString)
  mypart = ("--bounds", str(-5.0), str(5.0), str(-5.0), str(5.0), \
            "-s", str(stencilSize))
  return mypart
