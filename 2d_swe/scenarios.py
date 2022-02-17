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
    'finalTime': 7.0,
    'inviscidFluxReconstruction' : "FirstOrder",
    'odeScheme': "RungeKutta4",
    'dt' : 0.01,
    'stateSamplingFreq' : 2,
    'velocitySamplingFreq' : 2
  },

  'odrom' : {
    'finalTime': 7.0,
    'inviscidFluxReconstruction' : "FirstOrder",
    'odeScheme': "RungeKutta4",
    'dt' : 0.01,
    'stateSamplingFreq' : 2
  },

  'physicalCoefficients' : {
    'gravity'   : 9.8,
    'coriolis'  : "tbd",
    'pulsemag'  : 0.05
  }
}

train_points[1] = {
  0: -4.0,
  1: -0.5
}

test_points[1]  = {
  0: -1.5
}

odrom_algos[1]        = ["GalerkinFull"]
odrom_energies[1]     = [99.9, 99.95]
odrom_basis_sets[1]   = {
  0: [0],
  1: [0,1]
}

odrom_partitioning_topol[1] = [[2,2], [2,5]]
odrom_partitioning_style[1] = ['uniform']

'''
male list of valid scenarios, figure out from keys in dic
'''
valid_scenarios_ids = list(base_dic.keys())
