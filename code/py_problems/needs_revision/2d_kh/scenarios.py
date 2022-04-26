
import numpy as np
from py_problems.dictionaries import *

base_dic[1] = {
  'fom' : {
    'meshSize': [256, 256],
    'finalTimeTrain': 25.0,
    'finalTimeTest' : 30.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "SSPRK3",
    'dt' : 0.005,
    'stateSamplingFreqTrain': 5,
    'velocitySamplingFreq'  : 5
  },
  'stateSamplingFreqTest' : 20,
  'odrom' : {
    'finalTime': 30.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "SSPRK3",
    'dt' : 0.005
  },
  'physicalCoefficients' : {}
}

train_points[1] = { 0: 0.0 }
test_points[1]  = train_points[1]

algos[1] = [ "PodStandardProjectionError", "PodOdProjectionError"]
# "PodStandardGalerkin", "PodOdGalerkin"]

use_ic_reference_state[1] = True
basis_sets[1] = { 0: [0] }
standardrom_modes_setting_policies[1] = {'userDefinedValue' : [50, 250, 500, 1000]}

odrom_modes_setting_policies[1] = {'tileSpecificUsingEnergy' : [99.99, 99.999, 99.9999]}
odrom_min_num_modes_per_tile[1] = 5
odrom_partitions[1] = {'rectangularUniform' : [[3,3], [4,4], [5,3], [7,7], [10,10]], \
                       'concentricUniform' : [10]}
