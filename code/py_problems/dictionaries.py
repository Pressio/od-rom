
###############################################################
base_dic = {}
###############################################################
'''
base_dic MUST ALWAYS be present
'''

###############################################################
train_points = {}
test_points  = {}
###############################################################
'''
say something
'''

###############################################################
algos = {}
###############################################################
'''
list of strings to set which algos to run

choices:
  PodStandardGalerkin        # standard galerkin
  PodStandardGalerkinGappy   # gappy galerkin
  PodStandardProjectionError # projection error full domain

  PodOdProjectionError      # projection error for odrom
  PodOdGalerkin             # od galerkin
  PodOdGalerkinGappy        # od gappy galerkin with real HR
  PodOdGalerkinGappyMasked  # od gappy galerkin via mask
  PodOdGalerkinQuad         # od galerkin with quad projector

  LegendreOdGalerkinFull
'''

###############################################################
basis_sets = {}
###############################################################

'''
the sets of training runs to use for doing pod
'''

###############################################################
use_ic_reference_state = {}
###############################################################
'''
True/False to set initial condition as ref state

if true:
- state snapshots POD are computed on snapshots
  after subtracted corresponding initial condition
- for each test run, the corresponding rom is done
  using the initial state of that test run as reference state

if false:
- FOM initial condition is used to compute rom ic
'''

###############################################################
standardrom_modes_setting_policies = {}
odrom_modes_setting_policies       = {}
odrom_min_num_modes_per_tile       = {}
###############################################################
'''
fill
'''

###############################################################
odrom_partitions = {}
###############################################################
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
  - ex:
   odrom_partitioning[1] = { 'rectangularUniform' : [2, 4] }

for 2d:
  - format: list of [nx,ny]
  - ex:
   odrom_partitioning[1] = {
     'rectangularUniform' : [[2,2], [4,4]],
     'concentricUniform' : [4, 5]
   }
'''

###############################################################
sample_meshes = {}
###############################################################
'''
Example of sample meshes:
sample_meshes[1] = [["random", 0.25], ["psampling", 0.25, 0] ]

for random   : [string, fraction]
for psampling: [string, fraction, int of dof to use to compute sm]
'''

###############################################################
odrom_poly_order = {}
###############################################################
'''
Example:
  odrom_poly_order[1] = [-1, 1, 4, 6]

where:
# -1: compute orders of the poly bases to match pod modes
#     and truncate to have a full poly order
# int>0: we use same poly order in each tile
'''
