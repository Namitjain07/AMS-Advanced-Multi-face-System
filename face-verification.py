import os
import shutil
import cv2
import mediapipe as mp
from PIL import Image
import logging
import warnings
from datetime import datetime

# Setup logging to file
log_filename = f"face_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, "w", encoding="utf-8")

def log_and_print(message):
    print(message)
    log_file.write(message + "\n")

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)

# Suppress OpenCV logs if possible
try:
    if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except:
    pass

mp_face_mesh = mp.solutions.face_mesh

def has_proper_features(image_path):
    log_and_print(f"🔍 Processing: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        log_and_print(f"⚠️  Could not read image: {image_path}")
        return False

    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        log_and_print(f"❌ No face detected in: {os.path.basename(image_path)}")
        return False

    for face_landmarks in results.multi_face_landmarks:
        required_indices = {
            'right_eye': 33,
            'left_eye': 263,
            'nose': 1,
            'mouth_right': 61,
            'mouth_left': 291
        }

        for name, idx in required_indices.items():
            if idx >= len(face_landmarks.landmark):
                log_and_print(f"❌ Missing landmark {name} in: {os.path.basename(image_path)}")
                return False

            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            if not (0 <= x <= width and 0 <= y <= height):
                log_and_print(f"❌ Invalid position for {name}: ({x}, {y})")
                return False

        log_and_print(f"✅ Valid face found in: {os.path.basename(image_path)}")
        return True

    log_and_print(f"❌ Features not valid in: {os.path.basename(image_path)}")
    return False


# Setup
output_folder = 'verified_face_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    log_and_print(f"📁 Created folder: {output_folder}")

image_files = []
for root, _, files in os.walk('face_images'):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(os.path.join(root, file))

log_and_print(f"🧾 Found {len(image_files)} image(s) to process.\n")

# Process images
verified_count = 0
skipped_count = 0

for image_path in image_files:
    if has_proper_features(image_path):
        destination_path = os.path.join(output_folder, os.path.basename(image_path))
        shutil.copy(image_path, destination_path)
        log_and_print(f"📥 Copied to verified folder.\n")
        verified_count += 1
    else:
        log_and_print(f"⏭️  Skipped: does not meet criteria.\n")
        skipped_count += 1

# Final summary
log_and_print("📊 Processing Summary:")
log_and_print(f"✅ Verified images: {verified_count}")
log_and_print(f"⏭️  Skipped images: {skipped_count}")
log_and_print(f"📁 Output saved to folder: {output_folder}")
log_and_print(f"📝 Log saved to file: {log_filename}")

log_file.close()
