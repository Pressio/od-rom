import numpy as np

from py_problems.dictionaries import *

'''
this scenario is interesting because the param space
is such that the dynamics has a kind of bifurcation,
so this is pretty hard.
odrom performs pretty well while regular gal does not.
'''
base_dic[1] = {
  'fom' : {
    'meshSize': [200, 200],
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

train_points[1] = {0: 0.025, 1: 0.027}
test_points[1]  = {0: 0.024, 1: 0.025, 2: 0.026, 3: 0.028}
use_ic_reference_state[1] = True
basis_sets[1] = { 0: [0,1] }

algos[1] = ["ProjectionErrorUsingGlobalPodBases", "ProjectionErrorUsingTiledPodBases"]

standardrom_modes_setting_policies[1] = {'userDefinedValue': \
                                         [10, 50, 100, 200, 500, 1000, 3000]}

odrom_tile_based_or_split_global[1] = "TileBased"
odrom_modes_setting_policies[1] = { 'tileSpecificUsingEnergy' : [99.99, 99.999, 99.9999]}
odrom_min_num_modes_per_tile[1] = 5
odrom_partitions[1] = {'rectangularUniform' : [[4,4], [7,7], [10,10], [15,15]]}


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
