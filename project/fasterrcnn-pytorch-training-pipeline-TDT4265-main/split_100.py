import os
import random
import shutil

images_folder = 'data/LiDAR/archive/images'  #images
labels_folder = 'data/LiDAR/archive/outputs' #output should have the .XML files (ong Pascal Voc formate)
output_folder = 'data/LiDAR/archive/data' #empty folder where the shuffled val, train and and test data will be

# Ensure output directories exist
categories = ['train', 'test', 'val']
for category in categories:
    os.makedirs(os.path.join(output_folder, 'images', category), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels', category), exist_ok=True)

# List and sort files
image_files = sorted(os.listdir(images_folder))
label_files = sorted(os.listdir(labels_folder))

# Pairing files assuming names match except extensions
pairs = []
for image in image_files:
    label = image.replace('.PNG', '.xml')  # Adjust extension as necessary
    if label in label_files:
        pairs.append((image, label))

# Split pairs into sections of 50
section_size = 100
sections = [pairs[i:i + section_size] for i in range(0, len(pairs), section_size)]
random.shuffle(sections)  # Shuffle sections

# Allocate sections to train, test, val (70%, 20%, 10%)
train_index = int(len(sections) * 0.7)
test_index = int(len(sections) * 0.9)

train_data = [item for section in sections[:train_index] for item in section]
test_data = [item for section in sections[train_index:test_index] for item in section]
val_data = [item for section in sections[test_index:] for item in section]

# Shuffle data within sections and copy
def shuffle_and_copy(data, category):
    random.shuffle(data)  # Shuffle data inside each section
    for image_name, label_name in data:
        src_image_path = os.path.join(images_folder, image_name)
        src_label_path = os.path.join(labels_folder, label_name)
        dst_image_path = os.path.join(output_folder, 'images', category, image_name)
        dst_label_path = os.path.join(output_folder, 'labels', category, label_name)
        shutil.copy(src_image_path, dst_image_path)
        shutil.copy(src_label_path, dst_label_path)

# Copying files to respective folders
shuffle_and_copy(train_data, 'train')
shuffle_and_copy(test_data, 'test')
shuffle_and_copy(val_data, 'val')

print("Data processing complete. Files copied successfully.")