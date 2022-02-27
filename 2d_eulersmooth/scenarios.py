import numpy as np

base_dic              = {}
train_points          = {}
test_points           = {}

sample_mesh_fractions = {}
leverage_scores_betas = {}

odrom_energies        = {}
odrom_basis_sets      = {}
odrom_algos           = {}

odrom_partitioning_topol = {}
odrom_partitioning_style = {}

################################

base_dic[1] = {
  'general' : {
    'problem': "2d_eulersmooth",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 2.0,
    'finalTimeTest' : 3.5,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "SSPRK3",
    'dt' : 0.001,
    'stateSamplingFreq' : 5,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 3.5,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "SSPRK3",
    'dt' : 0.005,
    'stateSamplingFreq' : 5
  },

  'physicalCoefficients' : {}
}

train_points[1] = { 0: 0.}

test_points[1]  = { 0: 0.}

odrom_algos[1]        = ["PodGalerkinFull"]
odrom_energies[1]     = [99.9, 99.99, 99.999, 99.9999, 99.99999]
odrom_basis_sets[1]   = { 0: [0] }

odrom_partitioning_topol[1] = [[1,1], [2,2], [5,5], [10,10], [20,20]]
odrom_partitioning_style[1] = ['uniform']

'''
male list of valid scenarios, figure out from keys in dic
'''
valid_scenarios_ids = list(base_dic.keys())
