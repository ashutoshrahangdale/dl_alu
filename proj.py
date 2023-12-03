

import tensorflow as tf
def matrix_multiply(a, b):
    result = tf.matmul(a, b)
    return result
def convolution(input_data, filters):
    result = tf.nn.conv2d(input_data, filters, strides=[1, 1, 1, 1], padding='SAME')
    return result
# Example usage:
# Create random matrices for matrix multiplication
print('Please enter the order of Matrix 1')
row1 = int(input('Enter Number of Rows of Matrix 1:'))
column1 = int(input('ENter the number of Columns of the Matrix 1:'))
matrix_a = tf.random.normal((row1, column1))
print(matrix_a)
print('Please enter the order of Matrix 2')
row2 = int(input(' Number of Rows of Matrix 2:'))
column2 = int(input('ENter the number of Columns of the Matrix 2:'))
matrix_b = tf.random.normal((row2, column2))
print(matrix_b)
# Perform matrix multiplication
result_matrix_multiply = matrix_multiply(matrix_a, matrix_b)
print("Matrix Multiplication Result:")
print(result_matrix_multiply)
# Create random input data and filters for convolution
input_data = tf.random.normal((1, 32, 32, 3))  # Batch size 1, height 32, width 32, 3 channels
filters = tf.random.normal((3, 3, 3, 64))  # Filter size 3x3, 3 input channels, 64 output channels
# Perform convolution
result_convolution = convolution(input_data, filters)
print("\nConvolution Result:")
print(result_convolution)

