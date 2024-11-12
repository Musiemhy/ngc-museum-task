import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# File paths
TRAIN_FILE_PATH = 'train_sampled.csv'
TEST_FILE_PATH = 'test_sampled.csv'

# Load Data
data = pd.read_csv(TRAIN_FILE_PATH)
testdata = pd.read_csv(TEST_FILE_PATH)

# Set Column Names
data.columns = ['ClassIndex', 'Title', 'Description']
testdata.columns = ['ClassIndex', 'Title', 'Description']

# Combine Title and Description for both train and test sets
X_data = data['Title'] + " " + data['Description']
X_test = testdata['Title'] + " " + testdata['Description']

# Adjust class indices to start from 0
y_data = data['ClassIndex'].values - 1
y_test = testdata['ClassIndex'].values - 1

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# Initialize the BERT model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the train, validation, and test text data using BERT in batches for efficiency
X_train_embedded = model.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
X_valid_embedded = model.encode(X_valid.tolist(), batch_size=32, show_progress_bar=True)
X_test_embedded = model.encode(X_test.tolist(), batch_size=32, show_progress_bar=True)

# Convert labels to one-hot encoded vectors
label_binarizer = LabelBinarizer()
Y_train = label_binarizer.fit_transform(y_train)
Y_valid = label_binarizer.transform(y_valid)
Y_test = label_binarizer.transform(y_test)

# Check shapes (optional)
print("X_train shape:", X_train_embedded.shape)
print("Y_train shape:", Y_train.shape)
print("X_valid shape:", X_valid_embedded.shape)
print("Y_valid shape:", Y_valid.shape)
print("X_test shape:", X_test_embedded.shape)
print("Y_test shape:", Y_test.shape)

# Create the agnews folder if it doesn't exist
output_dir = "agnews"
os.makedirs(output_dir, exist_ok=True)

# Save arrays as .npy files in the agnews folder
np.save(os.path.join(output_dir, "trainX.npy"), X_train_embedded)
np.save(os.path.join(output_dir, "trainY.npy"), Y_train)
np.save(os.path.join(output_dir, "validX.npy"), X_valid_embedded)
np.save(os.path.join(output_dir, "validY.npy"), Y_valid)
np.save(os.path.join(output_dir, "testX.npy"), X_test_embedded)
np.save(os.path.join(output_dir, "testY.npy"), Y_test)
