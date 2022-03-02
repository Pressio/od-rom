import numpy as np

base_dic              = {}
train_points          = {}
test_points           = {}

sample_mesh_fractions = {}
leverage_scores_betas = {}

odrom_energies        = {}
odrom_basis_sets      = {}
odrom_algos           = {}
odrom_poly_order      = {}

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
    'finalTimeTest' : 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
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
  0: -3.0,
  1: -0.5
}

test_points[1]  = {
  0: -1.0,
  1: -3.25,
  2: 0.5
}

odrom_algos[1]        = ["PodGalerkinFull", "PolyGalerkinFull"]
odrom_energies[1]     = [99.99, 99.999, 99.9999, 99.99999, 99.999999]
odrom_basis_sets[1]   = {
  0: [0,1]
}

# -1: compute orders of the poly bases to match pod modes and truncate to have a full poly order
# int>0: we use same poly order in each tile
odrom_poly_order[1]   = [-1]#, 1, 2, 4, 6]

odrom_partitioning_topol[1] = [[12,12], [6,6]]
odrom_partitioning_style[1] = ['uniform']


#---------------------------------------------

base_dic[2] = {
  'general' : {
    'problem': "2d_swe",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 5.0,
    'finalTimeTest' : 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.0025,
    'stateSamplingFreq' : 4,
    'velocitySamplingFreq' : 100
  },

  'odrom' : {
    'finalTime': 8.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.01,
    'stateSamplingFreq' : 4
  },

  'physicalCoefficients' : {
    'gravity'   : 9.8,
    'coriolis'  : "tbd",
    'pulsemag'  : "tbd"
  }
}

train_points[2] = {
  0: [-3.0, 0.125],
  1: [-0.5, 0.125],
  2: [-3.0, 0.155],
  3: [-0.5, 0.155],
}

test_points[2]  = {
  0: [-1.0,  0.135],
  1: [-3.25, 0.135],
  2: [ 0.5,  0.135],
  3: [-1.0,  0.170]
}

odrom_algos[2]        = ["PodGalerkinFull", "PolyGalerkinFull"]
odrom_energies[2]     = [99.99, 99.995, 99.999, 99.9995, 99.99999, 99.999999]
odrom_basis_sets[2]   = {
  0: [0,1]
}

# -1: compute orders of the poly bases to match pod modes and truncate to have a full poly order
# int>0: we use same poly order in each tile
odrom_poly_order[2]   = [-1]#, 1, 2, 4, 6]

odrom_partitioning_topol[2] = [[12,12], [6,6]]
odrom_partitioning_style[2] = ['uniform']


'''
male list of valid scenarios, figure out from keys in dic
'''
valid_scenarios_ids = list(base_dic.keys())
