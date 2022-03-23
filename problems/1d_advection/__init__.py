
import pressiodemoapps as pda
from .scenarios import *

dimensionality = 1
numDofsPerCell = 1

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
  schemeStr   = dicIn['inviscidFluxReconstruction']
  schemeEnu   = inviscid_flux_string_to_enum(schemeStr)
  velocity    = coeffDic['velocity']
  myinitialcond = 3

  if scenario <= 3:
    velocity = val
  else:
    sys.exit("invalid scenario {} for 1d_advection".format(scenario))

  dicIn['velocity'] = velocity
  appObj = pda.create_linear_advection_1d_problem(meshObj, schemeEnu, \
                                                  velocity, myinitialcond)

  return appObj

# -------------------------------------------------------------------
def custom_tuple_args_for_fom_mesh_generation(scenario):
  schemeString = base_dic[scenario]['fom']['inviscidFluxReconstruction']
  stencilSize  = inviscid_flux_string_to_stencil_size(schemeString)
  xMin, xMax = 0.0, 12.0
  mypart = ("--bounds", str(xMin), str(xMax), \
            "-s", str(stencilSize), \
            "--periodic", "x")
  return mypart
