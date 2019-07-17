\ Model mdp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  20 u[1] + 5 u[2] + 50 u[3] + 10 u[4] + 10 u[5]
Subject To
 C0[0]: 0.6833333333333333 u[0] + u[3] = 1
 C0[1]: - 0.6333333333333333 u[0] + 0.05 u[1] + 0.05 u[4] = 0
 C0[2]: 0.05 u[2] - 0.95 u[3] + 0.05 u[5] = 0
Bounds
End
