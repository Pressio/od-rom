import numpy as np
from legendre_bases import LegendreBases2d

polyObj = LegendreBases2d("totalOrder")
#order = 2

#M = polyObj(order, np.linspace(-1.,1., 10), np.linspace(-1.,1., 10))

o = polyObj.findClosestOrderToMatchTargetBasesCount(20)
print(o)
M = polyObj(o, np.linspace(-1.,1., 10), np.linspace(-1.,1., 10))
print(M.shape[1])
