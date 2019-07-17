#!/usr/bin/python


import numpy as np
from gurobipy import *
from numpy import linalg as LA

def printSolution():
    if model.status == GRB.Status.OPTIMAL:
        print('\nReward: %g' % model.objVal)
        for v in model.getVars():
            print('%s %g' % (v.varName, v.x))
    else:
        print('No solution')

# defining matrix A in A v >= r
# A = I_(nm*n) - P_(nm*n)
n = 1 # number of states
m = 2 # number of actions
# building I_(nm*n)
I = np.zeros([n*m,n])
for i in range(m):
  I[i*n:(i+1)*n,:] = np.eye(n)
gamma = 0.95 # discount factor
# transition probability
P = np.array([[1],[1]])
A = I - gamma*P
# Initial state distribution
alpha = np.zeros(n)
alpha[0] = 1
# alpha[1] = 1/3
# alpha[2] = 1/3
# reward function r(s,a), with first incrementing over s then a
# r = np.array([5,-5])
# Defining C, Cr < d is constraint over the rewards
C = np.array(   [[-1,0],
                [1,0],
                [0,-1],
                [0,1]])

d = np.array([-5, 10, -1, 5])
l = len(d) # number of reward's constraints
# Model
model = Model("mdp")

# Create variables for value function
u = model.addVars(n*m, name="u", lb=0)
# r = model.addVars(n*m, name="r", lb=0)
t = model.addVars(l, name="t", lb=0)
# t = model.addVar(name="t")


# The objective is to maximize the rewards
model.setObjective((quicksum(d[i] * t[i] for i in range(l))), GRB.MAXIMIZE)
# model.setObjective(t, GRB.MAXIMIZE)

# Constraints
model.addConstrs(
                (quicksum(A[j,i] * u[j] for j in range(n*m)) ==
                alpha[i] for i in range(n) ),
                name="C0")

# model.addConstrs((t <= r[i]*u[i] for i in range(n*m)), name="C1")
# model.addConstrs(
#                 (quicksum(C[i,j] * r[j] for j in range(n*m)) <=
#                 d[i] for i in range(l) ),
#                 name="C2")
model.addConstrs(
                (quicksum(C[j,i] * t[j] for j in range(l)) ==
                u[i] for i in range(n*m) ),
                name="C1")

model.addConstrs(
                (t[i] >= 0 for i in range(l)),
                name="C2")

# Checking the correctness of constraints
model.write("reward_1state.lp")


# Solve
model.optimize()
printSolution()



# solve them separately

