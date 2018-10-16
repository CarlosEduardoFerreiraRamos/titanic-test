"""
There are two types of Dimension Reduction

1 - Feature selection: backward elimination, forward elimination, bidirectional Elimination, Score Compaison
and more.

2 - Feature extraction: Principal components Analysis(PCA), Linear Discriminat Analysis (LDA), Kernel PCA,
Quadratic Discrimination Analysis

* And we will cover the Future extraction part.
"""

"""
Principal components Analysis(PCA)
# Most popular reduction

1 - Identify patters in data (using correlation)
2 - detect the correlation between the variables

break down of the pca processes:
# Stadardize teh data
# Obtain Eigenvector and Eigenvalues from the covariace matrix or correlation matrix, or perform Singular Vector Decomposition
# Sort eigenvlues in descending orider and choose the k eigevector that correspon to the largest eigenvalues where k is the
  number of dimension of the new feature subspace
# Construct the projection matrix W from the selected k eigenvectors 
# Transform the original dataset x via W to obtain a k-simentional feature subspace Y

pca exemple
http://setosa.io/ev/principal-component-analysis/

Learn about the relationship between x and y values
find list of principal axes

cons: hight afected by outliers in the data
"""