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
# Stadardize the data
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

"""
Linear Discirminant Analysis - LDA

used as dimensionality reduction technique
used in the pre-processing step for patter classification
has the goal to project a dataset onto a lower-dimensional space

LDA differs from PCA in that LDA try to find max axis that differentiate the classes

the goal of PDA is to project a feature space onto a small subspace while maintaining
the class-discriminatory information.

So both LDA and PDA are linear transformation techniques for dimension reduction.
PDA is unsurpevised.
LDA is supervised, becouse of it's relation to it'd dependent varible

break down of the LDA processes:
# Compute of d-dimentional mean vectors for the different classes from the database 
# conpute the scatter matrix
# Compute the eigenvectors and correspondent eigevalues for the scatter matrix. 
# Sort eigenvectors by decresing eigevalues and choose the k eigevector that correspon to the largest eigenvalues
  to form a d x k dimensional matrix W (where every colum represents a eigevector) 
# Use this d x k eigenvectors matrix to trasform  the samples onto  the new subspace.
  This can be summarized by the matrix multiplication: Y = X * W
  (wheres X is the n xd dimensional matrix representing the n samples, and y it's the n x k dimensional
  samples in the new subspace)  
"""