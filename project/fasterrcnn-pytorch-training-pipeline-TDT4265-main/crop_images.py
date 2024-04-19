import os
import cv2
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm

"""
thsi code ewas used to generate croiped versions of the images. Because we noticed alot of space on the top of our images without ovbjects.
This code also adjustes the position of the bounding boxes in the labels files (.xml).
makes sure to use the correct paths.
Run the function by the calls that are commented out at the end of code.
This code is meda with help by ChatGPT
"""

# Directories
train_im_path = "data/LiDAR/archive/data/images/train"
test_im_path = "data/LiDAR/archive/data/images/test"
val_im_path = "data/LiDAR/archive/data/images/val"

train_label_path = "data/LiDAR/archive/data/labels/train"
test_label_path = "data/LiDAR/archive/data/labels/test"
val_label_path = "data/LiDAR/archive/data/labels/val"

crop_train_im_path = "data/LiDAR/archive/data/crop_images/train"
crop_test_im_path = "data/LiDAR/archive/data/crop_images/test"
crop_val_im_path = "data/LiDAR/archive/data/crop_images/val"

crop_train_label_path = "data/LiDAR/archive/data/crop_labels/train"
crop_test_label_path = "data/LiDAR/archive/data/crop_labels/test"
crop_val_label_path = "data/LiDAR/archive/data/crop_labels/val"

# Function to read XML label files and extract bounding box coordinates
def read_label_file(label_file):
    tree = ET.parse(label_file)
    root = tree.getroot()

    # Parse XML to extract bounding box coordinates and class names
    bounding_boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        class_name = obj.find('name').text
        bounding_boxes.append(([(xmin, ymin), (xmax, ymax)], class_name))
    return bounding_boxes


# Function to crop images and adjust bounding box coordinates
def crop_images_and_labels(src_im_path, src_label_path, dest_im_path, dest_label_path):
    for filename in tqdm(os.listdir(src_im_path)):
        if filename.endswith(".PNG"):
            im_file = os.path.join(src_im_path, filename)
            label_file = os.path.join(src_label_path, filename.replace('.PNG', '.xml'))

            # Read image
            image = cv2.imread(im_file)
            height, width, _ = image.shape
            
            # Compute crop height (to crop top 30% of image)
            crop_height = int(0.3 * height)

            # Crop the top portion of the image
            cropped_image = image[crop_height:, :]

            # Read bounding box coordinates and class names from label file
            bounding_boxes = read_label_file(label_file)

            # Adjust bounding box coordinates after cropping
            adjusted_boxes = []
            class_names = []  # Store class names
            for bbox, class_name in bounding_boxes:
                adjusted_box = [
                    (bbox[0][0], max(0, bbox[0][1] - crop_height)),
                    (bbox[1][0], max(0, bbox[1][1] - crop_height))
                ]
                adjusted_boxes.append(adjusted_box)
                class_names.append(class_name)

            # Write the cropped image
            cropped_im_file = os.path.join(dest_im_path, filename)
            cv2.imwrite(cropped_im_file, cropped_image)

            # Write adjusted bounding box coordinates to label file
            cropped_label_file = os.path.join(dest_label_path, filename.replace('.PNG', '.xml'))
            with open(cropped_label_file, 'w') as f:
                f.write('<annotation>\n')
                for bbox, class_name in zip(adjusted_boxes, class_names):
                    f.write('\t<object>\n')
                    f.write('\t\t<name>{}</name>\n'.format(class_name))  # Write original class name
                    f.write('\t\t<bndbox>\n')
                    f.write('\t\t\t<xmin>{}</xmin>\n'.format(bbox[0][0]))
                    f.write('\t\t\t<ymin>{}</ymin>\n'.format(bbox[0][1]))
                    f.write('\t\t\t<xmax>{}</xmax>\n'.format(bbox[1][0]))
                    f.write('\t\t\t<ymax>{}</ymax>\n'.format(bbox[1][1]))
                    f.write('\t\t</bndbox>\n')
                    f.write('\t</object>\n')
                f.write('</annotation>')

# Show one or two images with bounding boxes to verify
def show_images_with_bboxes(image_path, label_path, num_images=2):
    images = os.listdir(image_path)
    for i in range(num_images):
        image_file = images[i]
        label_file = os.path.join(label_path, image_file.replace('.PNG', '.xml'))
        bounding_boxes = read_label_file(label_file)
        image = cv2.imread(os.path.join(image_path, image_file))
        for bbox, class_name in bounding_boxes:
            pt1 = bbox[0]
            pt2 = bbox[1]
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(image, class_name, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow(f'Image with Bounding Boxes {image_file}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




# Crop images and labels for training set

#crop_images_and_labels(train_im_path, train_label_path, crop_train_im_path, crop_train_label_path)
#crop_images_and_labels(val_im_path, val_label_path, crop_val_im_path, crop_val_label_path)
#crop_images_and_labels(test_im_path, test_label_path, crop_test_im_path, crop_test_label_path)
show_images_with_bboxes(crop_train_im_path, crop_train_label_path)
