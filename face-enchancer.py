import os
# Force CPU usage (add this at the very beginning)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import shutil
from retinaface import RetinaFace


def has_proper_features(image_path):
    """Check if image contains faces with all required features using RetinaFace"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return False

    try:
        # Detect faces with enhanced confidence threshold
        faces = RetinaFace.detect_faces(image, threshold=0.999)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

    if not isinstance(faces, dict):  # RetinaFace returns dict when faces are found
        return False

    height, width = image.shape[:2]

    for face_id, face_data in faces.items():
        # Check facial keypoints
        kps = face_data['landmarks']
        required_points = {
            'left_eye': kps['left_eye'],
            'right_eye': kps['right_eye'],
            'nose': kps['nose'],
            'mouth_left': kps['mouth_left'],
            'mouth_right': kps['mouth_right']
        }

        # Verify all points are within image boundaries
        valid = True
        for name, point in required_points.items():
            x, y = point
            if x < 0 or x > width or y < 0 or y > height:
                valid = False
                break

        if valid:
            return True

    return False


# Configure paths
input_folder = "results/face_images_0.7/restored_faces"
output_folder = "detected_faces"

# Create output directory if not exists
os.makedirs(output_folder, exist_ok=True)

# Process images
valid_count = 0
total_count = 0

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        total_count += 1
        image_path = os.path.join(input_folder, filename)

        if has_proper_features(image_path):
            output_path = os.path.join(output_folder, filename)
            shutil.copy(image_path, output_path)
            valid_count += 1
            print(f"✅ Valid: {filename}")
        else:
            print(f"❌ Rejected: {filename}")

print(f"\nProcessing complete!")
print(f"Total images processed: {total_count}")
print(f"Valid images detected: {valid_count}")
print(f"Output folder: {os.path.abspath(output_folder)}")