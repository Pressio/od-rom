
import numpy as np
import sys, os, re, yaml

# -------------------------------------------------------------------
def path_to_partition_info_dir(workDir, npx, npy, style):
  s1 = workDir + "/od_info"
  if npy == None:
    s2 = str(npx)
  else:
    s2 = str(npx)+"x"+str(npy)
  s3 = "style_"+style
  return s1+"_"+s2+"_"+s3

# -------------------------------------------------------------------
def string_identifier_from_partition_info_dir(infoDir):
  return os.path.basename(infoDir)[8:]

# -------------------------------------------------------------------
def path_to_partition_based_full_mesh_dir(workDir, partitioningKeyword):
  s1 = workDir + "/partition_based_"+partitioningKeyword
  return s1 + "_" + "full_mesh"

# -------------------------------------------------------------------
def path_to_state_pod_data_dir(workDir, partitioningKeyword, setId):
  s1 = workDir + "/partition_based_"+partitioningKeyword
  s2 = "full_state_pod_set_"+str(setId)
  return s1 + "_" + s2

# -------------------------------------------------------------------
def path_to_full_domain_state_pod_data_dir(workDir, setId):
  s1 = workDir + "/full_domain"
  s2 = "full_state_pod_set_"+str(setId)
  return s1 + "_" + s2

# -------------------------------------------------------------------
def path_to_rhs_pod_data_dir(workDir, partitioningKeyword, setId):
  s1 = "/partition_based_"+partitioningKeyword
  s2 = "full_rhs_pod_set_"+str(setId)
  return workDir + s1 + "_" + s2

# -------------------------------------------------------------------
def path_to_full_domain_rhs_pod_data_dir(workDir, setId):
  s1 = "/full_domain"
  s2 = "full_rhs_pod_set_"+str(setId)
  return workDir + s1 + "_" + s2

# -------------------------------------------------------------------
def path_to_poly_bases_data_dir(workDir, partitioningKeyword, \
                                order, energy=None, setId=None):
  s1 = workDir + "/partition_based_"+partitioningKeyword
  s2 = "full_poly_order_"+str(order)
  result = s1 + "_" + s2
  if energy != None:
    result += "_"+str(energy)
  if setId != None:
    result += "_set_"+str(setId)
  return result

# -------------------------------------------------------------------
def path_to_od_sample_mesh_random(workDir, partitioningKeyword, fraction):
  s1 = workDir + "/partition_based_"+partitioningKeyword
  s2 = "sample_mesh_random_{:3.3f}".format(fraction)
  return s1 + "_" + s2

# -------------------------------------------------------------------
def path_to_full_domain_sample_mesh_random(workDir, fraction):
  s1 = workDir + "/full_domain"
  s2 = "sample_mesh_random_{:3.3f}".format(fraction)
  return s1 + "_" + s2

# -------------------------------------------------------------------
def path_to_od_sample_mesh_psampling(workDir, partitioningKeyword, setId, \
                                     fraction, dofToUseForFindingCells):
  s1 = workDir + "/partition_based_"+partitioningKeyword
  s2 = "sample_mesh_psampling_set_"+str(setId)
  s3 = "dofid_{}".format(dofToUseForFindingCells)
  s4 = "fraction_{:3.3f}".format(fraction)
  return s1 + "_" + s2 + "_" + s3 + "_" + s4

# -------------------------------------------------------------------
def path_to_full_domain_sample_mesh_psampling(workDir, setId, fraction):
  s1 = workDir + "/full_domain"
  s2 = "sample_mesh_psampling_set_"+str(setId)
  s3 = "fraction_{:3.3f}".format(fraction)
  return s1 + "_" + s2 + "_" + s3

# -------------------------------------------------------------------
def string_identifier_from_sample_mesh_dir(sampleMeshDir):
  if "sample_mesh_random" in sampleMeshDir:
    return "random_"+sampleMeshDir[-5:]
  elif "sample_mesh_psampling" in sampleMeshDir:
    return "psampling_"+sampleMeshDir[-22:]

# # -------------------------------------------------------------------
# def path_to_gappy_projector_dir(workDir, partitioningKeyword, \
#                                 setId, energyValue, smKeyword):
#   s1 = workDir + "/partition_based_"+partitioningKeyword
#   s2 = "gappy_projector"
#   s3 = str(energyValue)
#   s4 = "using_"+smKeyword
#   s5 = "set_"+ str(setId)
#   sep = "_"
#   return s1 + sep + s2 + sep + s3 + sep + s4 + sep + s5

# -------------------------------------------------------------------
def path_to_gappy_projector_dir(workDir, gappyPolicyName, \
                                partitioningKeyword, \
                                setId, modeSettingPolicy, \
                                energyValue, numModes, \
                                smKeyword):
  s1 = workDir + "/partition_based_"+partitioningKeyword
  s2 = "gappy_projector_" + gappyPolicyName
  s3 = modeSettingPolicy
  if energyValue != None:
    s4 = str(energyValue)
  if numModes != None:
    s4 = str(numModes)
  s5 = "using_"+smKeyword
  s6 = "set_"+ str(setId)
  sep = "_"
  return s1 + sep + s2 + sep + s3 + sep + s4 + sep + s5 + sep + s6

# -------------------------------------------------------------------
def path_to_phi_on_stencil_dir(workDir, partitioningKeyword, \
                               setId, modeSettingPolicy, \
                               energyValue, numModes, smKeyword):
  s1 = workDir + "/partition_based_"+partitioningKeyword
  s2 = "phi_on_stencil"
  s3 = modeSettingPolicy
  if energyValue != None:
    s4 = str(energyValue)
  if numModes != None:
    s4 = str(numModes)

  s5 = "using_"+smKeyword
  s6 = "set_"+ str(setId)
  sep = "_"
  return s1 + sep + s2 + sep + s3 + sep + s4 + sep + s5 + sep + s6

# -------------------------------------------------------------------
def path_to_quad_projector_dir(workDir, partitioningKeyword, \
                               setId, energyValue, smKeyword):
  s1 = workDir + "/partition_based_"+partitioningKeyword
  s2 = "quad_projector"
  s3 = str(energyValue)
  s4 = "set_"+ str(setId)
  s5 = smKeyword
  sep = "_"
  return s1 + sep + s2 + sep + s3 + sep + s4 + sep + s5
