# Python modules
import numpy as np
from scipy.stats import norm, multivariate_normal

# Copied from https://github.com/probml/pyprobml/blob/master/scripts/logreg_laplace_demo.py in order to reproduce.
np.random.seed(135)
#Creating data
N = 30
D = 2
mu1 = np.hstack((np.ones((N,1)), 5 * np.ones((N, 1))))
mu2 = np.hstack((-5 * np.ones((N,1)), np.ones((N, 1))))
class1_std = 1
class2_std = 1.1
X_1 = np.add(class1_std*np.random.randn(N,2), mu1)
X_2 = np.add(2*class2_std*np.random.randn(N,2), mu2)
X = np.vstack((X_1,X_2))
t = np.vstack((np.ones((N,1)),np.zeros((N,1))))

# Export data
import pandas as pd
import os
os.chdir('./content/post/2021-10-27-bayesian-logistic-regression/')
dt = pd.DataFrame({'y': t.flatten().tolist(), 'x_1': X[:,0].tolist(), 'x_2': X[:,1].tolist()})
dt.to_csv('data/toy_data.csv', index=False)
