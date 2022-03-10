import numpy as np

base_dic              = {}
train_points          = {}
test_points           = {}

sample_mesh_fractions = {}
leverage_scores_betas = {}

# True/False to set initial condition as ref state
odrom_use_ic_reference_state = {}

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
    'problem': "2d_rti",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 1.8,
    'finalTimeTest' : 1.8,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "SSPRK3",
    'dt' : 0.0005,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 50
  },

  'odrom' : {
    'finalTime': 1.8,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "SSPRK3",
    'dt' : 0.002,
    'stateSamplingFreq' : 4
  },

  'physicalCoefficients' : {
    'amplitude' : "tbd",
  }
}

train_points[1] = {
  0: 0.020,
  1: 0.040
}

test_points[1]  = {
  0: 0.030,
}

odrom_use_ic_reference_state[1] = False

odrom_algos[1]        = ["PodGalerkinFull"]
odrom_energies[1]     = [99.99999]
odrom_basis_sets[1]   = {
  0: [0,1]
}

odrom_partitioning_topol[1] = [[2,7]]
odrom_partitioning_style[1] = ['uniform']

'''
male list of valid scenarios, figure out from keys in dic
'''
valid_scenarios_ids = list(base_dic.keys())
