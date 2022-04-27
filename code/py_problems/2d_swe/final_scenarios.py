
import numpy as np
from py_problems.dictionaries import *

'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
this scenario checks that on the training data, for a fixed
number of total modes, regular POD is always better than od,
and the convergence is better.
For testing data, we should see that this does not hold.

We only want projection error here.
Coriolis as parameter, same simulation time for traing and test
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''

assert(1 not in base_dic)
base_dic[1] = {
  'fom' : {
    'meshSize': [256, 256],
    'finalTimeTrain': 10.0, 'finalTimeTest' : 10.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4", 'dt' : 0.005,
    'stateSamplingFreqTrain': 5, 'velocitySamplingFreq': 5
  },

  'stateSamplingFreqTest' : 20,
  'physicalCoefficients' : {
    'gravity' : 9.8, 'coriolis': "p0", 'pulsemag': 0.125
  }
}
algos[1]        = ["ProjectionErrorUsingTiledPodBases", \
                   "ProjectionErrorUsingGlobalPodBases"]
train_points[1] = {0: -4.0,  1: -3.0, 2: -2.0, 3: -1.0, 4: 0.0}
test_points[1]  = {0: -4.25, 1: 0.25, 2: -4.0, 3: 0.0,  4: -2.5}
basis_sets[1]   = {0: [0,4], 1: [0,2,4], 2: [0,1,2,3,4]}

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
