
import sys
import pressiodemoapps as pda
from .scenarios import *

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
  probId    = pda.Swe2d.SlipWall
  schemeStr = dicIn['inviscidFluxReconstruction']
  schemeEnu = inviscid_flux_string_to_enum(schemeStr)

  # preset values from problem dic
  gravity  = coeffDic['gravity']
  coriolis = coeffDic['coriolis']
  pulse    = coeffDic['pulsemag']

  if scenario == 1:
    coriolis = val
  elif scenario == 2:
    coriolis = val
  elif scenario == 3:
    coriolis = val
  elif scenario == -1:
    coriolis = val
  else:
    sys.exit("__init__: invalid scenario {} for 2d_swe".format(scenario))

  # store
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
