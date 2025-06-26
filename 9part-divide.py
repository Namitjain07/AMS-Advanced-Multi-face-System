from PIL import Image
import os
import pyheif
import cv2
import matplotlib.pyplot as plt
# import numpy as np
from retinaface import RetinaFace  # Ensure RetinaFace is installed
# from arcface import ArcFace
# import os
# from PIL import Image
# import pillow_heif
import os
import pyheif
from PIL import Image



source_dir = "DL Project Data/28-01-2025"



# Traverse directory and find .heic files
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(".heic"):
            heic_path = os.path.join(root, file)
            jpg_path = os.path.splitext(heic_path)[0] + ".jpg"

            # Convert HEIC to JPG
            heif_file = pyheif.read(heic_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode
            )

            # Save as JPG
            image.save(jpg_path, "JPEG")
            print(f"Converted: {heic_path} → {jpg_path}")


image_files = []
for root, _, files in os.walk('DL Project Data/07-01-2025'):
    for file in files:
      if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".png"):
        image_files.append(os.path.join(root, file))

print(image_files)

# --- Step 0: Configuration ---
input_image_path = image_files[0]  # Replace with your image file path.
output_folder = "9_segmented_images"
os.makedirs(output_folder, exist_ok=True)

# --- Step 1: Load the image and pad it to a square ---
im = Image.open(input_image_path)
width, height = im.size

# Use the maximum dimension for the square side (padding instead of cropping)
new_side = max(width, height)

# Create a new square image with a black background.
# Change (0, 0, 0) to a different RGB tuple if you prefer another background color.
new_im = Image.new(im.mode, (new_side, new_side), (0, 0, 0))

# Paste the original image into the center of the new square image.
paste_x = (new_side - width) // 2
paste_y = (new_side - height) // 2
new_im.paste(im, (paste_x, paste_y))

# This is your square image with padding.
im_square = new_im

# --- Step 2: Define the 4x4 grid and extract the 2x2 blocks ---
# The square is conceptually divided into a 4x4 grid.
cell_size = new_side // 4

# Mapping from segment number to the top-left cell (row, col) of its 2x2 block.
segments = {
    1: (0, 0),  # cells 1,2,5,6
    2: (0, 2),  # cells 3,4,7,8
    3: (2, 0),  # cells 9,10,13,14
    4: (2, 2),  # cells 11,12,15,16
    5: (0, 1),  # cells 2,3,6,7
    6: (1, 0),  # cells 5,6,9,10
    7: (2, 1),  # cells 10,11,14,15
    8: (1, 2),  # cells 7,8,11,12
    9: (1, 1),  # cells 6,7,10,11
}

# Loop over each segment, crop it from the square image, and save it.
for seg_num, (grid_row, grid_col) in segments.items():
    # Calculate pixel coordinates for the 2x2 block.
    left_coord = grid_col * cell_size
    top_coord = grid_row * cell_size
    right_coord = left_coord + (2 * cell_size)
    bottom_coord = top_coord + (2 * cell_size)

    # Crop the segment.
    seg_im = im_square.crop((left_coord, top_coord, right_coord, bottom_coord))

    # Save the segment.
    seg_filename = os.path.join(output_folder, f"segment{seg_num}.jpg")
    seg_im.save(seg_filename)
    print(f"Saved segment {seg_num} to {seg_filename}")

image_files = []
for root, _, files in os.walk('9_segmented_images'):
    for file in files:
      if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".png"):
        image_files.append(os.path.join(root, file))

print(image_files)

# Folder to store extracted faces
output_folder = "face_images"
os.makedirs(output_folder, exist_ok=True)

all_faces = []  # Store extracted faces if needed

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Create a copy of the original image for visualization
    img_with_rectangles = img.copy()

    # Detect faces
    faces = RetinaFace.detect_faces(img)
    print(f'Image: {img_path} | Faces detected: {len(faces)}')

    for key in faces.keys():
        identity = faces[key]
        facial_area = identity["facial_area"]
        x1, y1, x2, y2 = facial_area

        # Ensure valid coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x1 >= x2 or y1 >= y2:
            continue

        # ----------------------------
        # FIRST: Extract face from ORIGINAL image (no border)
        # ----------------------------
        face_img = img[y1:y2, x1:x2]  # Use original image for extraction
        if face_img.size == 0:
            continue

        # Save the extracted face (no border)
        face_filename = os.path.join(output_folder, f"face_{key}_{os.path.basename(img_path)}")
        cv2.imwrite(face_filename, face_img)

        # Convert to RGB for storage if needed
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        all_faces.append(face_rgb)

        # ----------------------------
        # THEN: Draw rectangle on COPY
        # ----------------------------
        cv2.rectangle(img_with_rectangles, (x1, y1), (x2, y2), (255, 255, 255), 10)

    # Display the image with rectangles
    plt.figure(figsize=[15, 15])
    plt.imshow(cv2.cvtColor(img_with_rectangles, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()