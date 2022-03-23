

base_dic              = {}
'''
- base_dic MUST ALWAYS be present
- should look like this:

    base_dic[1] = {
      'fom' : {
        'meshSize'             : <list with mesh sizes along each axis> ,
        'finalTimeTrain'       : <self-explanatory>,
        'finalTimeTest'        : <self-explanatory>,
        'odeScheme'            : <RK2, RK4, SSPRK3>,
        'dt'                   : <self-explanatory>,
        'stateSamplingFreq'    : <self-explanatory>,
        'velocitySamplingFreq' : <self-explanatory>,
        ...
        any other field else needed by the problem
      },

      'odrom' : {
        'finalTime'         : <self-explanatory>,
        'odeScheme'         : <RK2, RK4, SSPRK3>,
        'dt'                : <self-explanatory>,
        'stateSamplingFreq' : <self-explanatory>,
        ...
        <any other field else needed by the problem>
      },

      'physicalCoefficients' : {
        <any param specifci to the target problem>
      }
    }
'''

train_points = {}
test_points  = {}


use_ic_reference_state = {}
'''
True/False to set initial condition as ref state

if true:

- state snapshots POD are computed on snapshots
  after subtracted corresponding initial condition

- for each test run, the corresponding rom is done
  using the initial state of that test run as reference state
'''


algos = {}
'''
list of strings to set which algos to run

choices:
PodStandardGalerkinFull
PodStandardGalerkinGappy
PodOdGalerkinFull
PodOdGalerkinGappy
PodOdGalerkinMasked
LegendreOdGalerkinFull
'''


standardrom_energies = {}
odrom_energies = {}
'''
list of energies to use to set the num of modes

note that we keep separate the list for odrom
and standard galerkin because it is likely that for
standard rom we will want to push to higher energies
'''


odrom_partitions = {}
'''
defines the target tiling structure and how this is done

currently support two main types of decompositions:
- rectangularUniform
  tries to decompose into rectangular as uniform as possible

- concentricUniform
  tries to decompose into concentric annuli using
  the center of the full domain as "center" aiming for as
  uniform distribution as possible


for 1d:
  - the two methods are the same, so only one choice is valid
  - format: list of integers
  - ex:
   odrom_partitioning[1] = {
    'rectangularUniform' : [2, 4]
   }

for 2d:
  - format: list of [nx,ny]
  - ex:
   odrom_partitioning[1] = {
     'rectangularUniform' : [[2,2], [4,4]],
     'concentricUniform' : [4, 5]
   }
'''


basis_sets = {}
'''
the sets of training runs to use for doing pod
'''


sample_meshes = {}
'''
Example of sample meshes:
sample_meshes[1] = [["random", 0.25], \
                    ["psampling", 0.25, 0] ]

for random   : [string, fraction]
for psampling: [string, fraction, int of dof to use to compute sm]
'''


odrom_poly_order = {}
'''
Example:
  odrom_poly_order[1] = [-1, 1, 4, 6]

where:
# -1: compute orders of the poly bases to match pod modes
#     and truncate to have a full poly order
# int>0: we use same poly order in each tile
'''
