import numpy as np

# ========= Matrix-related functions =========

# Adds ones column to the left of the matrix
# ex.
# [ 0.5, 0.10 ] -> [ 1, 0.5, 0.10 ] 
# [ 0.9, 0.20 ]    [ 1, 0.9, 0.20 ]
def addBiasColumn(matrix):
    # Create column of 1
    ones_column = np.ones((matrix.shape[0], 1))

    # Create bias column in the left
    return np.hstack((ones_column, matrix))

# Removes first column
def removeBiasColumn(matrix):
    return matrix[:,  1:]

# Removes first row
def removeBiasRow(matrix):
    return matrix[1:, :]