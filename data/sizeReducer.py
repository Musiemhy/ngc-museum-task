import pandas as pd

# Load the dataset
train_data = pd.read_csv('train.csv')

# Sample 20% of each class for the training data
train_data_sampled = train_data.groupby('Class Index', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))

# Shuffle the sampled data to randomize the order
train_data_sampled = train_data_sampled.sample(frac=1, random_state=42)

# Save the sampled training data to a new file
train_data_sampled.to_csv('train_sampled.csv', index=False)

# Load the test dataset
test_data = pd.read_csv('test.csv')

# Sample 20% of each class for the test data
test_data_sampled = test_data.groupby('Class Index', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))

# Shuffle the sampled data to randomize the order
test_data_sampled = test_data_sampled.sample(frac=1, random_state=42)

# Save the sampled test data to a new file
test_data_sampled.to_csv('test_sampled.csv', index=False)

print("Sampling complete. Files saved as 'train_sampled.csv' and 'test_sampled.csv'")
