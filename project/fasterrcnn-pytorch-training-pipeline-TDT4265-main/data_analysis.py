import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
"""
This code was used to print an image with all the b-boxes from the training set. Used to analyse the common position of b-boxes.
This code is made with help from ChatGPT
"""

# Function to read XML label files and extract bounding box center points
def read_label_file(label_file):
    tree = ET.parse(label_file)
    root = tree.getroot()

    # Parse XML to extract bounding box center points
    center_points = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        # Calculate centroid of the bounding box
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_points.append((center_x, center_y))
    return center_points

# Function to find all label files in a directory and read their bounding box center points
def process_labels(labels_dir):
    all_center_points = []
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.xml'):
            label_file_path = os.path.join(labels_dir, label_file)
            center_points = read_label_file(label_file_path)
            all_center_points.extend(center_points)
    return all_center_points

# Directories for labels and images
train_label_path = "data/LiDAR/archive/data/crop_labels/train"
train_image_path = "data/LiDAR/archive/data/crop_images/train"
#train_label_path = "data/LiDAR/archive/data/labels/train"
#train_image_path = "data/LiDAR/archive/data/images/train"

# Function to find an example image and its corresponding labels
def find_example_image(labels_dir, images_dir):
    # Find the first XML label file
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.xml'):
            label_file_path = os.path.join(labels_dir, label_file)
            image_file = label_file.replace('.xml', '.PNG')
            image_file_path = os.path.join(images_dir, image_file)
            return image_file_path, label_file_path
    return None, None

# Find an example image and its corresponding labels
image_path, label_path = find_example_image(train_label_path, train_image_path)

# Read center points from all label files
center_points = process_labels(train_label_path)

# Plot the image and center points
if image_path is not None and label_path is not None:
    # Read and plot the image
    image = plt.imread(image_path)
    plt.imshow(image)

    # Plot center points
    center_points = np.array(center_points)
    plt.scatter(center_points[:, 0], center_points[:, 1], color='red', marker='o')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Center Points of Bounding Boxes')
    plt.grid(True)
    plt.show()
else:
    print("No example image found.")
