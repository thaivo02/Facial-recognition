import os
from sklearn.model_selection import train_test_split
import shutil

# Set the path to your dataset folder
dataset_folder = "data"

# List all files in the dataset folder
all_files = os.listdir(dataset_folder)

# Split the data into training and test sets
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Create folders for training and test sets
train_folder = os.path.join(dataset_folder, "train")
test_folder = os.path.join(dataset_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Move files to their respective folders
for file in train_files:
    src_path = os.path.join(dataset_folder, file)
    dst_path = os.path.join(train_folder, file)
    shutil.move(src_path, dst_path)

for file in test_files:
    src_path = os.path.join(dataset_folder, file)
    dst_path = os.path.join(test_folder, file)
    shutil.move(src_path, dst_path)

print("Dataset split into training and test sets.")
