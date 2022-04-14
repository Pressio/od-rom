import numpy as np

from py_problems.dictionaries import *

base_dic[1] = {
  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 2000.,
    'finalTimeTest' : 2000.,
    'odeScheme': "RK4",
    'dt' : 0.2,
    'stateSamplingFreqTrain' : 10,
    'velocitySamplingFreq'   : 10
  },
  'stateSamplingFreqTest' : 10,
  'physicalCoefficients' : {
    'diffusionA' : 0.0002,
    'diffusionB' : 0.00005,
    'killRate'   : 0.062,
    'feedRate'   : "tbd"
  }
}

train_points[1] = {0: 0.03, 1: 0.05, 2: 0.07}
test_points[1]  = {0: 0.04, 1: 0.06, 2: 0.075, 3: 0.025}
use_ic_reference_state[1] = True
basis_sets[1] = { 0: [0,1,2] }

algos[1] = ["PodStandardProjectionError", "PodOdProjectionError"]

standardrom_modes_setting_policies[1] = {'userDefinedValue' : [10, 50, 100, 200, 500]}

odrom_modes_setting_policies[1] = { 'tileSpecificUsingEnergy' : [99.99], \
                                    'allTilesUseTheSameUserDefinedValue' : [10, 15, 20],
                                    'findMinValueAcrossTilesUsingEnergyAndUseInAllTiles': [99.99]}
odrom_min_num_modes_per_tile[1] = 5
odrom_partitions[1] = {'rectangularUniform' : [[4,4], [7,7], [10,10], [15,15]]}
