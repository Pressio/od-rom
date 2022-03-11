

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

# True/False to set initial condition as ref state
use_ic_reference_state = {}

'''
"PodStandardGalerkinFull"
"PodStandardGalerkinGappy"
"PodOdGalerkinFull"
"PodOdGalerkinGappy"
"LegendreOdGalerkinFull"
'''
algos = {}

'''
energies are kept separate between odrom and standard
because it is likely that for standard rom
we will want to push to higher energies
'''
standardrom_energies = {}
odrom_energies = {}

odrom_partitioning_topol = {}
odrom_partitioning_style = {}

basis_sets = {}

'''
Example of sample meshes:
od_rom_sample_meshes[1] = [["random", 0.25], \
                          ["psampling", 0.25, 0] ]

for random   : [string, fraction]
for psampling: [string, fraction, int of dof to use to compute sm]
'''
sample_meshes   = {}

'''
Example:
  odrom_poly_order[1] = [-1, 1, 4, 6]

where:
# -1: compute orders of the poly bases to match pod modes
#     and truncate to have a full poly order
# int>0: we use same poly order in each tile
'''
odrom_poly_order      = {}
