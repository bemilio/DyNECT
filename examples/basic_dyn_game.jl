using Revise
using DyNECT

A = [1 1; 0 0.9]
B = [[1.; 0.;;], [0.; 1.;;]]
x0 = [1; 1]
game = DyNEP(A, B)
