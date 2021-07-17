import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.linalg import eigh # eigendecomposition
from numpy.linalg import svd # SVD


url = 'https://vincentarelbundock.github.io/Rdatasets/csv/MASS/crabs.csv'
crabs = pd.read_csv(url)
X = crabs[['FL', 'RW', 'CL', 'CW', 'BD']]

#sample covariance matrix
S = X.cov()

eigenvals, eigenvecs = eigh(S)

X_centered = X - X.mean()

U, d, V = svd(X_centered)     #D is actually an array of the diagonal entries

dim1 = U.shape[1]
dim2 = V.shape[0]     #making D a 5x5 diagonal matrix

D = np.zeros(shape = (dim1, dim2))
np.fill_diagonal(D, val = d)
V = V.T
#columns of V are the eigenvectors of 

(U @ D) @ (V.T)   #sanity check for SVD

pca_projections = X_centered @ V

plt.figure()
sns.pairplot(data = pca_projections)  #pair plots of prpjections onto the 5 PC

B = X_centered @ X_centered.T  #gram matrix

U2, d2, V2 = svd(B)  #V2 is U.T because matrix symmetric

D2 = np.zeros (shape = (dim1, dim2))
np.fill_diagonal(D2,np.sqrt(d2[:5]))  #SVD decomposition of X has 200x5 diagonalt matrix
# equal to D2

pca_projections_2 = U2 @ D2

plt.figure()
sns.pairplot(data = pd.DataFrame(pca_projections_2))





