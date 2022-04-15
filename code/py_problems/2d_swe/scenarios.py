
import numpy as np
from py_problems.dictionaries import *

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

reproductive case where we use multiple training runs
so that we can see if/how the dynamics is complex enough
that bases get affected as compared to using a single training run

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
base_dic[1] = {
  'fom' : {
    'meshSize': [256, 256],
    'finalTimeTrain': 8.0,
    'finalTimeTest' : 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025,
    'stateSamplingFreqTrain': 5,
    'velocitySamplingFreq'  : 5
  },
  'stateSamplingFreqTest' : 400,
  'odrom' : {
    'finalTime': 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005
  },
  'physicalCoefficients' : {
    'gravity': 9.8, 'coriolis':"tbd", 'pulsemag': 0.125
  }
}

train_points[1] = { 0: -3.0, 1: -1.5, 2: -0.5 }
test_points[1]  = train_points[1]

algos[1] = ["PodStandardGalerkin", "PodStandardProjectionError", \
            "PodOdGalerkin", "PodOdProjectionError"]

use_ic_reference_state[1] = True
basis_sets[1] = { 0: [0,1,2] }
standardrom_modes_setting_policies[1] = {'userDefinedValue' : [10, 50, 150]}

odrom_modes_setting_policies[1] = { 'allTilesUseTheSameUserDefinedValue' : [10, 20, 25]}
odrom_min_num_modes_per_tile[1] = 5
odrom_partitions[1] = {'rectangularUniform' : [[3,3], [7,7], [8,8], [10,10]]}



'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

it seems that in the reproductive case above,
things dont work well for 8x8 or 10x10.
Maybe this is due to the coriolis range above being too aggressive?
So here we try a reproductive using a smaller variation of Coriolis

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
base_dic[2]     = base_dic[1]
train_points[2] = { 0: -3.0, 1: -2.5, 2: -2.0 }
test_points[2]  = train_points[2]

use_ic_reference_state[2] = True
basis_sets[2] = { 0: [0,1,2] }

algos[2] = ["PodOdGalerkin"]
odrom_modes_setting_policies[2] = {'allTilesUseTheSameUserDefinedValue' : [10, 20, 25],
                                   'tileSpecificUsingEnergy' : [99.992, 99.995]}

odrom_min_num_modes_per_tile[2] = 5
odrom_partitions[2] = {'rectangularUniform' : [[3,3],[8,8],[10,10]]}


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

predict in both param space (Coriolis) and time

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
base_dic[3] = {
  'fom' : {
    'meshSize': [256, 256],
    'finalTimeTrain': 8.0,
    'finalTimeTest' : 10.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025,
    'stateSamplingFreqTrain': 4,
    'velocitySamplingFreq'  : 4
  },

  'stateSamplingFreqTest' : 20,

  'odrom' : {
    'finalTime': 10.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005
  },

  'physicalCoefficients' : {
    'gravity' : 9.8, 'coriolis'  : "tbd", 'pulsemag'  : 0.125
  }
}
train_points[3] = {0: -4.0, 1: -3.0, 2: -2.0 }
test_points[3]  = {0: -2.5, 1: -1.5}

algos[3] = ["PodStandardProjectionError", "PodOdProjectionError", \
            "PodStandardGalerkin", "PodOdGalerkin"]

use_ic_reference_state[3] = True
basis_sets[3] = { 0: [0,1,2] }

standardrom_modes_setting_policies[3] = {'userDefinedValue' : [25, 50, 100, 200]}
#'energyBased' : [99.999, 99.9999] }

odrom_modes_setting_policies[3] = {'allTilesUseTheSameUserDefinedValue' : [10, 20, 25]}
#'tileSpecificUsingEnergy' : [99.999, 99.9999]}

odrom_min_num_modes_per_tile[3] = 3

odrom_partitions[3] = {'rectangularUniform' : [[4,4], [6,6], [7,7], [8,8], [10,10]]}


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

this scenario is meant to check that on the training data
we see that, for a fixed number of total modes, regular POD
is always better than od, and the convergence should be better.
For testing data, we should see that this does not hold.

we only want projection error here.
Coriolis as parameter, same simulation time for traing and test

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
base_dic[4] = {
  'fom' : {
    'meshSize': [256, 256],
    'finalTimeTrain': 8.0,
    'finalTimeTest' : 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025,
    'stateSamplingFreqTrain': 4,
    'velocitySamplingFreq'  : 4
  },

  'stateSamplingFreqTest' : 20,

  # 'odrom' : {
  #   'finalTime': 8.0,
  #   'inviscidFluxReconstruction' : "Weno5",
  #   'odeScheme': "RK4",
  #   'dt' : 0.005
  # },

  'physicalCoefficients' : {
    'gravity' : 9.8, 'coriolis'  : "tbd", 'pulsemag'  : 0.125
  }
}
algos[4]        = ["PodStandardProjectionError", "PodOdProjectionError"]
train_points[4] = {0: -4.0, 1: -3.0, 2: -2.0 }
test_points[4]  = {0: -2.5, 1: -3.0}
basis_sets[4]   = { 0: [0,1,2] }

use_ic_reference_state[4] = True
standardrom_modes_setting_policies[4] = {'userDefinedValue' : [8*8*5, 8*8*10, 8*8*20, 8*8*25]}
odrom_modes_setting_policies[4] = {'allTilesUseTheSameUserDefinedValue' : [5, 10, 20, 25]}
#odrom_min_num_modes_per_tile[4] = 3
odrom_partitions[4] = {'rectangularUniform' : [[8,8]]}



'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
this scenario -1 is my playground, DO NOT RELY ON IT
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''
base_dic[-1] = {
  'fom' : {
    'meshSize': [128, 128],
    'finalTimeTrain': 4.0,
    'finalTimeTest' : 4.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreqTrain': 5,
    'velocitySamplingFreq'  : 5
  },
  'stateSamplingFreqTest' : 10,
  'odrom' : {
    'finalTime': 4.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.008
  },
  'physicalCoefficients' : {
    'gravity': 9.8, 'coriolis':"tbd", 'pulsemag': 0.125
  }
}

train_points[-1] = { 0: -3.0}
test_points[-1]  = train_points[-1]
use_ic_reference_state[-1] = True
basis_sets[-1] = { 0: [0]}
algos[-1] = ["PodStandardProjectionError", \
             "PodStandardGalerkin", \
             "PodStandardGalerkinGappy", \
             "PodOdProjectionError",\
             "PodOdGalerkin",\
             "PodOdGalerkinGappy",\
             "PodOdGalerkinGappyMasked"]

standardrom_modes_setting_policies[-1] = {'userDefinedValue' : [20],\
                                          'energyBased' : [99.9999]}

odrom_modes_setting_policies[-1] = { 'tileSpecificUsingEnergy' : [99.999], \
                                     'allTilesUseTheSameUserDefinedValue' : [20],
                                     'findMinValueAcrossTilesUsingEnergyAndUseInAllTiles': [99.95]}
odrom_min_num_modes_per_tile[-1] = 5
odrom_partitions[-1] = {'rectangularUniform' : [[7,7]]}
sample_meshes[-1] = [["psampling", 0.5, 0], ['random', 0.8]]
