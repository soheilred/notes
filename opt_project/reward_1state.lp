\ Model mdp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 5 t[0] + 10 t[1] - t[2] + 5 t[3]
Subject To
 C0[0]: 0.05 u[0] + 0.05 u[1] = 1
 C1[0]: - u[0] - t[0] + t[1] = 0
 C1[1]: - u[1] - t[2] + t[3] = 0
 C2[0]: t[0] >= 0
 C2[1]: t[1] >= 0
 C2[2]: t[2] >= 0
 C2[3]: t[3] >= 0
Bounds
End
