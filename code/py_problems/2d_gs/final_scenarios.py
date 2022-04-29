import numpy as np
from py_problems.dictionaries import *

base_dic[1] = {
  'fom' : {
    'meshSize': [256, 256],
    'finalTimeTrain': 3000.,
    'finalTimeTest' : 3000.,
    'odeScheme': "RK4",
    'dt' : 0.2,
    'stateSamplingFreqTrain' : 8,
    'velocitySamplingFreq'   : 8
  },
  'stateSamplingFreqTest' : 10,
  'physicalCoefficients' : {
    'diffusionA' : 0.00002,
    'diffusionB' : 0.00001,
    'killRate'   : 0.055,
    'feedRate'   : "p0"
  }
}

algos[1] = ["ProjectionErrorUsingGlobalPodBases", \
            "ProjectionErrorUsingTiledPodBases"]

train_points[1] = {0: 0.025, 1: 0.026, 2: 0.027}
test_points[1]  = {0: 0.024, 1: 0.028, 2: 0.025, 3: 0.027, 4: 0.0255}
basis_sets[1]   = {0: [0,1], 1: [0,1,2]}

use_ic_reference_state[1] = True
odrom_partitions[1] = {'rectangularUniform': \
                       [[3,3], [4,4], [5,5], [6,6], \
                        [7,7], [8,8], [9,9], [10,10] ]}
odrom_min_num_modes_per_tile[1] = 1
odrom_tile_based_or_split_global[1] = "TileBased"

odrom_modes_setting_policies[1] = {'tileSpecificUsingEnergy': \
                                   [90., 92.5, 95., 99., 99.5, \
                                    99.9, 99.95, 99.999, 99.9995]}

standardrom_modes_setting_policies[1] = {'userDefinedValue': \
                                         [10, 25, 50, 100, 250, \
                                          500, 1000, 1500, 2000]}
