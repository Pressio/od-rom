import numpy as np

from py_problems.dictionaries import *


'''
same as [1] but we run roms for real
'''
base_dic[2] = {
  'fom' : {
    'meshSize': [200, 200],
    'finalTimeTrain': 2000.,
    'finalTimeTest' : 2000.,
    'odeScheme': "RK4",
    'dt' : 0.2,
    'stateSamplingFreqTrain' : 10,
    'velocitySamplingFreq'   : 10
  },
  'rom' : {
    'finalTime': 2000.0,
    'odeScheme': "RK4",
    'dt' : 0.4
  },

  'stateSamplingFreqTest' : 10,
  'physicalCoefficients' : {
    'diffusionA' : 0.0002,
    'diffusionB' : 0.00005,
    'killRate'   : 0.062,
    'feedRate'   : "p0"
  }
}

train_points[2] = train_points[1]
test_points[2]  = test_points[1]
use_ic_reference_state[2] = use_ic_reference_state[1]
basis_sets[2] = basis_sets[1]
algos[2] = ["OdGalerkinWithPodBases", "OdGappyGalerkinWithPodBases"]

odrom_tile_based_or_split_global[2] = "TileBased"
odrom_modes_setting_policies[2] = { 'tileSpecificUsingEnergy' : [99.99], \
                                    'allTilesUseTheSameUserDefinedValue' : [10, 15, 20],
                                    'findMinValueAcrossTilesUsingEnergyAndUseInAllTiles': [99.99]}
odrom_min_num_modes_per_tile[2] = 5
odrom_partitions[2] = {'rectangularUniform' : [[10,10], [15,15]]}
sample_meshes[2] = [["psampling", 0.1, 1]]
