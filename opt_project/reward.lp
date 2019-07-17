\ Model mdp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  t
Subject To
 C0[0]: 0.6833333333333333 u[0] + u[3] = 1
 C0[1]: - 0.6333333333333333 u[0] + 0.05 u[1] + 0.05 u[4] = 0
 C0[2]: 0.05 u[2] - 0.95 u[3] + 0.05 u[5] = 0
 C1[0]: t <= 0
 C1[1]: - 20 u[1] + t <= 0
 C1[2]: - 5 u[2] + t <= 0
 C1[3]: - 50 u[3] + t <= 0
 C1[4]: - 10 u[4] + t <= 0
 C1[5]: - 10 u[5] + t <= 0
Bounds
End
