
import active_subspaces as ac
import active_subspaces.subspaces
# import active_subspaces.utils
import numpy as np
import matplotlib.pyplot as plt
from wing_functions import *

M = 1000 #This is the number of data points to use

#Sample the input space according to the distributions in the table above
Sw = np.random.uniform(150, 200, (M, 1))
Wfw = np.random.uniform(220, 300, (M, 1))
A = np.random.uniform(6, 10, (M, 1))
L = np.random.uniform(-10, 10, (M, 1))
q = np.random.uniform(16, 45, (M, 1))
l = np.random.uniform(.5, 1, (M, 1))
tc = np.random.uniform(.08, .18, (M, 1))
Nz = np.random.uniform(2.5, 6, (M, 1))
Wdg = np.random.uniform(1700, 2500, (M, 1))
Wp = np.random.uniform(.025, .08, (M, 1))

#The input matrix
x = np.hstack((Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp))

#The function's output
f = wing(x)

#Upper and lower limits for inputs
ub = np.array([150, 220, 6, -10, 16, .5, .08, 2.5, 1700, .025]).reshape((1, 10))
lb = np.array([200, 300, 10, 10, 45, 1, .18, 6, 2500, .08]).reshape((1, 10))

#We normalize the inputs to the interval [-1, 1]: 
XX = 2.*(x - lb)/(ub - lb) - 1.0

#Instantiate a subspace object
ss = active_subspaces.subspaces.Subspaces()

#Compute the subspace with a global linear model (sstype='OLS') and 100 bootstrap replicates
ss.compute(X=XX, f=f, nboot=100, sstype='OLS')

#This plots the eigenvalues (ss.eigenvals) with bootstrap ranges (ss.e_br)
active_subspaces.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br)

#This plots subspace errors with bootstrap ranges (all contained in ss.sub_br)
ac.utils.plotters.subspace_errors(ss.sub_br)

#This makes sufficient summary plots with the active variables (XX.dot(ss.W1)) and output (f)
ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), f)

ss.compute(X=XX, f=f, nboot=100, sstype='QPHD')
ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br)
ac.utils.plotters.subspace_errors(ss.sub_br)
ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), f)

#quadratic polynomial approximation
RS = ac.utils.response_surfaces.PolynomialApproximation(2)

#Train the surface with active variable values (y = XX.dot(ss.W1)) and function values (f)
y = XX.dot(ss.W1)
RS.train(y, f)
print('The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr))

#Plot the data and response surface prediction
plt.figure(figsize=(7, 7))
y0 = np.linspace(-2, 2, 200)
plt.plot(y, f, 'bo', y0, RS.predict(y0[:,None])[0], 'k-', linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Active Variable Value', fontsize=18)
plt.ylabel('Output', fontsize=18)