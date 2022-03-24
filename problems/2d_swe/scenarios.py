
import numpy as np
from problems.dictionaries import *

'''
reproductive case but with a twist
- use multiple training runs so that we can see if/how
  the dynamics is complex enough that bases get affected
  as compared to using a single training run
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
use_ic_reference_state[1] = True
basis_sets[1] = { 0: [0,1,2] }
algos[1] = ["PodOdGalerkinFull", "PodStandardGalerkinFull", "PodOdProjectionError"]
standardrom_modes_setting_policies[1] = {'energyBased' : [99.99],
                                         'userDefinedValue' : [10, 20, 25]}

odrom_modes_setting_policies[1] = { 'allTilesUseTheSameUserDefinedValue' : [10, 20, 25],
                                    'tileSpecificUsingEnergy' : [99.992, 99.995]}
odrom_min_num_modes_per_tile[1] = 5
odrom_partitions[1] = {'rectangularUniform' : [[3,3],[6,6], [7,7], [8,8], [10,10]]}



'''
it seems that in the reproductive case above,
things dont work well for 8x8 or 10x10 for some reason.
I am wondering if this is due to the coriolis range being too aggressive.
So let's try here to do a reproductive using a smaller range
'''
base_dic[2] = {
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

train_points[2] = { 0: -3.0, 1: -2.5, 2: -2.0 }
test_points[2]  = train_points[2]

use_ic_reference_state[2] = True
basis_sets[2] = { 0: [0,1,2] }

algos[2] = ["PodOdGalerkinFull"]

standardrom_modes_setting_policies[2] = {'userDefinedValue' : [10, 20, 25]}

odrom_modes_setting_policies[2] = {'allTilesUseTheSameUserDefinedValue' : [10, 20, 25],
                                   'tileSpecificUsingEnergy' : [99.992, 99.995]}

odrom_min_num_modes_per_tile[2] = 5
odrom_partitions[2] = {'rectangularUniform' : [[3,3],[8,8],[10,10]]}


'''
-------------------------------------------------------
predict in both param space (Coriolis) and time
-------------------------------------------------------
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

  'stateSamplingFreqTest' : 500,

  'odrom' : {
    'finalTime': 10.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005
  },

  'physicalCoefficients' : {
    'gravity'   : 9.8, 'coriolis'  : "tbd", 'pulsemag'  : 0.125
  }
}
train_points[3] = { 0: -3.0, 1: -1.5, 2: -0.5 }
test_points[3] = {0: -2.0, 1: -1.1}

use_ic_reference_state[3] = True
basis_sets[3] = { 0: [0,1,2] }

algos[3] = ["PodOdGalerkinFull", "PodStandardGalerkinFull"]

standardrom_modes_setting_policies[3] = {'userDefinedValue' : [10], \
                                         'energyBased' : [99.99, 99.9999] }

odrom_modes_setting_policies[3] = { 'allTilesUseTheSameUserDefinedValue' : [7, 10, 15, 20],
                                    'tileSpecificUsingEnergy' : [99.99, 99.9999],
                                    'findMinValueAcrossTilesUsingEnergyAndUseInAllTiles' : [99.99]}

odrom_min_num_modes_per_tile[3] = 3

odrom_partitions[3] = {'rectangularUniform' : [[5,5], [6,6], [7,7], [8,8], [10,10]]}
                       #'concentricUniform' : [8,12] }

sample_meshes[3] = [["psampling", 0.5, 0]]
