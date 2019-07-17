#!/usr/bin/python

# Copyright 2019, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

from gurobipy import *

try:

    # Create a new model
    m = Model("mip1")

    # Create variables
    x1 = m.addVar(vtype=GRB.CONTINUOUS, name="x1")
    x2 = m.addVar(vtype=GRB.CONTINUOUS, name="x2")
    x3 = m.addVar(vtype=GRB.CONTINUOUS, name="x3")
    x4 = m.addVar(vtype=GRB.CONTINUOUS, name="x4")

    # Set objective
    m.setObjective(-4 * x1 - 2 * x2, GRB.MINIMIZE)

    # Set constraints
    m.addConstr(x1 +  x2 +  x3 == 5.0, "c0")
    m.addConstr(2 * x1 + 0.5 * x2 + x4 == 8.0, "c1")
    m.addConstr(x1 >= 0, "c2")
    m.addConstr(x2 >= 0, "c3")
    m.addConstr(x3 >= 0, "c4")
    m.addConstr(x4 >= 0, "c5")

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')