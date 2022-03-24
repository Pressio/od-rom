import numpy as np
from problems.dictionaries import *

'''
-------------------------------------------------------
predict in param only
-------------------------------------------------------
'''
base_dic[1] = {

  'fom' : {
    'meshSize': [150, 150],
    'finalTimeTrain': 10.,
    'finalTimeTest' : 10.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025,
    'stateSamplingFreqTrain' : 4,
    'velocitySamplingFreq'   : 4
  },

  'odrom' : {
    'finalTime': 8.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025
  },

  'stateSamplingFreqTest' : 800,

  'physicalCoefficients' : {
    'pulsemag'    : 0.5,
    'pulsespread' : "tbd",
    'diffusion'   : 0.0001,
    "pulsecenter" : [-0.15, -0.3]
  }
}

train_points[1] = {
  0: 0.2,
  1: 0.3
  #2: 0.4
}

test_points[1]  = {
  0: 0.15,
  # 1: 0.25,
  # 2: 0.35,
  # 3: 0.45
}

use_ic_reference_state[1] = True

basis_sets[1] = { 0: [0,1] }

algos[1] = ["PodOdGalerkinGappy", "PodOdGalerkinFull"]

standardrom_energies[1] = [1.]
odrom_energies[1] = [99.999, 99.9999]

odrom_partitions[1] = { 'rectangularUniform' : [[5,5], [12,12]] }

sample_meshes[1] = [["psampling", 0.2, 0]]



'''
'''
base_dic[2] = {

  'fom' : {
    'meshSize': [250, 250],
    'finalTimeTrain': 5.,
    'finalTimeTest' : 5.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025,
    'stateSamplingFreqTrain' : 2,
    'velocitySamplingFreq'   : 2
  },

  'odrom' : {
    'finalTime': 5.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005
  },

  'stateSamplingFreqTest' : 5,

  'physicalCoefficients' : {
    'pulsemag'    : "tbd",
    'pulsespread' : 0.3,
    'diffusion'   : 0.0001,
    "pulsecenter" : [-0.15, -0.3]
  }
}

train_points[2] = {
  0: 0.2,
  1: 0.4,
  2: 0.6
}

test_points[2]  = {
  0: 0.15,
  1: 0.35,
  2: 0.65,
}

use_ic_reference_state[2] = True

basis_sets[2] = { 0: [0,1,2] }

algos[2] = ["PodOdGalerkinGappy", "PodOdGalerkinFull", "PodStandardGalerkinFull"]

standardrom_energies[2] = [1.]
odrom_energies[2] = [99.9993]

odrom_partitions[2] = { 'rectangularUniform' : [[8,8], [10,10]] }

sample_meshes[2] = [["psampling", 0.5, 0], ["psampling", 0.20, 0]]


'''
'''
base_dic[3] = {

  'fom' : {
    'meshSize': [400, 400],
    'finalTimeTrain': 5.,
    'finalTimeTest' : 5.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0010,
    'stateSamplingFreqTrain' : 4,
    'velocitySamplingFreq'   : 4
  },

  'odrom' : {
    'finalTime': 5.,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025
  },

  'stateSamplingFreqTest' : 500,

  'physicalCoefficients' : {
    'pulsemag'    : "tbd",
    'pulsespread' : "tbd",
    'diffusion'   : 0.0005,
    "pulsecenter" : [-0.15, -0.3]
  }
}

train_points[3] = {
  #  mag spread
  0: [0.5,  0.2],
  1: [0.5,  0.5],
  # 2: [0.5,  0.7],
  # 3: [0.8,  0.7],
  # 4: [0.65, 0.5]
}

test_points[3]  = { 0: [0.5, 0.8] }

use_ic_reference_state[3] = True
basis_sets[3] = { 0: [0,1] }

algos[3] = ["PodOdGalerkinGappy"]

standardrom_energies[3] = [1.]
odrom_energies[3] = [99.999, 99.9999]

odrom_partitions[3] = { 'rectangularUniform' : [[8,8], [12,12], [16,16]] }

'''
sample meshed worked using this to compute gappy projector:
  K = compute_cumulative_energy(rhsSingVals, 99.9999)
  if mySmCount*numDofsPerCell < K: K = mySmCount*numDofsPerCell - 1
  if K < myNumModes: K = myNumModes + 1
'''
sample_meshes[3] = [["psampling", 0.2, 0], ["psampling", 0.1, 0]]
