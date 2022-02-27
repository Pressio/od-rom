import numpy as np

base_dic              = {}
train_points          = {}
test_points           = {}

sample_mesh_fractions = {}
leverage_scores_betas = {}

odrom_energies        = {}
odrom_basis_sets      = {}
odrom_algos           = {}

'''
A decomposition is defined by two things:
a. topology: how many pieces
b. style   : how the pieces are computed
so we need to provide two things
to specify a decomposition.

For a., we can do:
in 2d we can have [[2,2], [2,3]]

For b. choices are:
1. uniform: self-explanatory
2. add more
'''
odrom_partitioning_topol = {}
odrom_partitioning_style = {}

################################

base_dic[1] = {
  'general' : {
    'problem': "2d_swe",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 5.0,
    'finalTimeTest' : 7.5,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "SSPRK3",
    'dt' : 0.0025,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 7.5,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "SSPRK3",
    'dt' : 0.01,
    'stateSamplingFreq' : 4
  },

  'physicalCoefficients' : {
    'gravity'   : 9.8,
    'coriolis'  : "tbd",
    'pulsemag'  : 0.125
  }
}

train_points[1] = {
  0: -3.5,
  1: -3.0,
  2: -2.5
}

test_points[1]  = {
  0: -2.11,
  1: -3.76
}

odrom_algos[1]        = ["PodGalerkinFull"]
odrom_energies[1]     = [99.99, 99.999, 99.999999]
odrom_basis_sets[1]   = {
  0: [0,1,2]
}

odrom_partitioning_topol[1] = [[5,5], [10,10]]
odrom_partitioning_style[1] = ['uniform']

'''
male list of valid scenarios, figure out from keys in dic
'''
valid_scenarios_ids = list(base_dic.keys())