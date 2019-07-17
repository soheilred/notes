#!/usr/bin/python


import numpy as np
from gurobipy import *

# defining matrix A in A v >= r
# A = I_(nm*n) - P_(nm*n)
n = 3 # number of states
m = 2 # number of actions
# building I_(nm*n)
I = np.zeros([n*m,n])
for i in range(m):
  I[i*n:(i+1)*n,:] = np.eye(n)

gamma = 0.95 # discount factor
# transition probability
P = np.array([[1/3,2/3,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0],[0,0,1]])
A = I - gamma*P
# Initial state distribution
alpha = np.zeros(n)
alpha[0] = 1
# reward function r(s,a), with first incrementing over s then a
r = np.array([-10, 1, -1, 10, 0, 0])

# Model
model = Model("mdp")

# Create variables for value function
u = model.addVars(n*m, name="u", lb=0)
t = model.addVar(name="t")


# The objective is to minimize the costs
model.setObjective(t, GRB.MAXIMIZE)


# Constraints
model.addConstrs(
                (quicksum(A[j,i] * u[j] for j in range(n*m)) ==
                alpha[i] for i in range(n) ),
                 name="C0")

model.addConstrs((t <= r[i]*u[i] for i in range(n*m)), name="C1")

# Checking the correctness of constraints
model.write("reward.lp")


# Solve
model.optimize()
# printSolution()

def printSolution():
    if model.status == GRB.Status.OPTIMAL:
        print('\nCost: %g' % model.objVal)
        print('\nBuy:')
        buyx = model.getAttr('x', buy)
        for f in foods:
            if buy[f].x > 0.0001:
                print('%s %g' % (f, buyx[f]))
    else:
        print('No solution')
