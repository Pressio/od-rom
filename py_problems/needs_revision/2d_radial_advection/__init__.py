
import pressiodemoapps as pda
from .scenarios import *

dimensionality = 2
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
  omega       = coeffDic['omega']

  if scenario == 1:
    omega = val
  else:
    sys.exit("invalid scenario {} for 2d_radial_advection".format(scenario))

  dicIn['omega'] = omega
  appObj = pda.create_radial_advection_2d_problem(meshObj, schemeEnu, omega)

  return appObj

# -------------------------------------------------------------------
def custom_tuple_args_for_fom_mesh_generation(scenario):
  schemeString  = base_dic[scenario]['fom']['inviscidFluxReconstruction']
  stencilSize   = inviscid_flux_string_to_stencil_size(schemeString)
  mypart = ("--bounds", str(-1.0), str(1.0), str(-1.0), str(1.0), \
            "-s", str(stencilSize), \
            "--periodic", "x", "y")

  return mypart
