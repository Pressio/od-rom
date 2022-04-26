
#==============================================================
base_dic = {}
#==============================================================
'''
base_dic MUST ALWAYS be present
'''

#==============================================================
train_points = {}
test_points  = {}
#==============================================================
'''
add description
'''

#==============================================================
algos = {}
#==============================================================
'''
list of strings to set which algos to run

choices:
  GlobalGalerkinWithPodBases             # standard global galerkin with pod
  ProjectionErrorUsingGlobalPod          # projection error full domain

  ProjectionErrorUsingTileLocalPodBases       # projection error for odrom
  OdGalerkinWithTileLocalPodBases             # od galerkin
  OdGappyGalerkinWithTileLocalPodBases        # od gappy galerkin with real HR
  OdMaskedGappyGalerkinWithTileLocalPodBases  # od gappy galerkin via mask
  OdQuadGalerkinWithTileLocalPodBases         # od galerkin with quad projector
'''

#==============================================================
basis_sets = {}
#==============================================================
'''
the sets of training runs to use for doing pod
'''

#==============================================================
use_ic_reference_state = {}
#==============================================================
'''
True/False to set initial condition as ref state

if true:
- state snapshots POD are computed on snapshots after
  subtracted corresponding initial condition
- for each test run, the corresponding rom is done
  using the initial state of that test run as reference state

if false: FOM initial condition is used to compute rom ic
'''

#==============================================================
standardrom_modes_setting_policies = {}
odrom_modes_setting_policies       = {}
odrom_min_num_modes_per_tile       = {}
#==============================================================
'''
add description
'''

#==============================================================
odrom_partitions = {}
#==============================================================
'''
defines the target tiling structure and how this is done

currently support two main types of decompositions:
- "rectangularUniform"
  tries to decompose into rectangular as uniform as possible

- "concentricUniform"
  tries to decompose into concentric annuli using
  the center of the full domain as "center" aiming for as
  uniform distribution as possible


for 1d:
  - the two methods are the same, so only one choice is valid
  - format: list of integers
  - ex: odrom_partitioning[1] = { 'rectangularUniform' : [2, 4] }

for 2d:
  - format: list of [nx,ny]
  - ex:
   odrom_partitioning[1] = {
     'rectangularUniform' : [[2,2], [4,4]],
     'concentricUniform' : [4, 5]
   }
'''

#==============================================================
sample_meshes = {}
#==============================================================
'''
Example of sample meshes:
sample_meshes[1] = [["random", 0.25], ["psampling", 0.25, 0] ]

for random   : [string, fraction]
for psampling: [string, fraction, int of dof to use to compute sm]
'''
