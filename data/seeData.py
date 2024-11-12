import numpy as np

# Load the .npy file
data = np.load('./agnews/trainX.npy')
data1 = np.load('./agnews/validX.npy')

# Display the data
print("trainY", data[:1])  # Display the first 10 elements
print("validY", data1[:1])  # Display the first 10 elements
