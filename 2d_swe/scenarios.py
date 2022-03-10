import numpy as np

base_dic              = {}
train_points          = {}
test_points           = {}

# True/False to set initial condition as ref state
odrom_use_ic_reference_state = {}

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

odrom_energies        = {}
odrom_basis_sets      = {}
odrom_sample_meshes   = {}
odrom_poly_order      = {}

################################

base_dic[1] = {
  'general' : {
    'problem': "2d_swe",
    'meshDir': "tbd"
  },

  'fom' : {
    'finalTimeTrain': 5.0,
    'finalTimeTest' : 6.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.005,
    'stateSamplingFreq' : 2,
    'velocitySamplingFreq' : 2
  },

  'odrom' : {
    'finalTime': 6.0,
    'inviscidFluxReconstruction' : "Weno5",
    'odeScheme': "RK4",
    'dt' : 0.025,
    'stateSamplingFreq' : 1
  },

  'physicalCoefficients' : {
    'gravity'   : 9.8,
    'coriolis'  : "tbd",
    'pulsemag'  : 0.125
  }
}

train_points[1] = {
  0: -2.0,
  1: -0.5
}

test_points[1]  = {
  0: -2.0,
  1: -1.25
}

odrom_use_ic_reference_state[1] = True

odrom_algos[1]      = ["PodGalerkinGappy"]

odrom_partitioning_topol[1] = [[10,10]]
odrom_partitioning_style[1] = ['uniform']

odrom_energies[1]   = [99.999, 99.999999, 99.9999999]
odrom_basis_sets[1] = { 0: [0,1] }

odrom_sample_meshes[1] = [["psampling", 0.1], ["psampling", 0.2]]

# -1: compute orders of the poly bases to match pod modes
#     and truncate to have a full poly order
# int>0: we use same poly order in each tile
odrom_poly_order[1]   = [-1]#, 1, 2, 4, 6]
