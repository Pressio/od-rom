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
    'problem': "2d_gs",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 1000.,
    'finalTimeTest' : 1000.,
    'odeScheme': "RK4",
    'dt' : 0.2,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 1000.0,
    'odeScheme': "RK4",
    'dt' : 0.4,
    'stateSamplingFreq' : 4
  },

  'physicalCoefficients' : {
    'diffusionA' : 0.0002,
    'diffusionB' : 0.00005,
    'killRate'   : "tbd",
    'feedRate'   : 0.03
  }
}

train_points[1] = {
  0: 0.060,
  1: 0.065,
  2: 0.070,
}

test_points[1]  = {
  0: 0.0625,
  1: 0.0725,
  2: 0.0750,
  3: 0.055,
  4: 0.068
}

odrom_algos[1]        = ["GalerkinFull"]
odrom_energies[1]     = [99.9999, 99.999995, 99.99999999]
odrom_basis_sets[1]   = {
  0: [0,1,2]
}

odrom_partitioning_topol[1] = [[1,1], [5,5], [10,10], [20,20]]
odrom_partitioning_style[1] = ['uniform']


#----------------------------------------------

base_dic[2] = {
  'general' : {
    'problem': "2d_gs",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 1000.,
    'finalTimeTest' : 1000.,
    'odeScheme': "RK4",
    'dt' : 0.2,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 1000.0,
    'odeScheme': "RK4",
    'dt' : 0.4,
    'stateSamplingFreq' : 4
  },

  'physicalCoefficients' : {
    'diffusionA' : 0.0002,
    'diffusionB' : 0.00005,
    'killRate'   : 0.062,
    'feedRate'   : "tbd"
  }
}

train_points[2] = {
  0: 0.03,
  1: 0.05,
  2: 0.07,
}

test_points[2]  = {
  0: 0.04,
  1: 0.06,
  2: 0.075,
  3: 0.025
}

odrom_algos[2]        = ["PodGalerkinFull"]
odrom_energies[2]     = [99.9999, 99.999995, 99.99999999]
odrom_basis_sets[2]   = {
  0: [0,1,2]
}

odrom_partitioning_topol[2] = [[1,1], [5,5], [10,10], [20,20]]
odrom_partitioning_style[2] = ['uniform']


'''
male list of valid scenarios, figure out from keys in dic
'''
valid_scenarios_ids = list(base_dic.keys())
