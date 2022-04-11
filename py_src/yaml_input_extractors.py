
import re, os, time, yaml

# -------------------------------------------------------------------
def find_state_sampling_frequency_from_input_file(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["stateSamplingFreq"]

# -------------------------------------------------------------------
def find_numdofspercell_from_input_file(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["numDofsPerCell"]

# -------------------------------------------------------------------
def find_meshdir_from_input_file(runDir):
  with open(runDir+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
  return ifile["meshDir"]

# -------------------------------------------------------------------
def find_dimensionality_from_info_file(workDir):
  reg = re.compile(r'dim.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_num_cells_from_info_file(workDir, ns):
  reg = re.compile(r''+ns+'.+')
  file1 = open(workDir+'/info.dat', 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[1])

# -------------------------------------------------------------------
def find_total_cells_from_info_file(workDir):
  dims = find_dimensionality_from_info_file(workDir)
  if dims == 1:
    return find_num_cells_from_info_file(workDir, "nx")
  elif dims==2:
    nx = find_num_cells_from_info_file(workDir, "nx")
    ny = find_num_cells_from_info_file(workDir, "ny")
    return nx*ny
  else:
    sys.exit("Invalid dims = {}".format(dims))
