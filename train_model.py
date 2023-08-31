import os
import json
import face_recognition
import logging

# Set up logging configuration
log_file_path = os.path.join("/Users/joshuasaji/Desktop/Face recognition system/Logs", "model_training_log.json")
logging.basicConfig(level=logging.INFO, filename=log_file_path, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Directory containing face images for training
training_dir = "Faces/"

# Load existing model data if available
model_path = os.path.join("models", "face_recognition_model.json")
if os.path.exists(model_path):
    with open(model_path, "r") as model_file:
        model_data = json.load(model_file)
else:
    model_data = {
        "encodings": [],
        "names": []
    }

# Initialize lists for face encodings and corresponding names
face_encodings = []
face_names = []

# Loop through each image in the training directory
for image_name in os.listdir(training_dir):
    if image_name.endswith(".jpg"):
        name = image_name.split(".")[0]
        if name not in model_data["names"]:
            image_path = os.path.join(training_dir, image_name)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)
            if face_encoding:
                face_encodings.append(face_encoding[0].tolist())  # Convert ndarray to list
                face_names.append(name)
                print(f"Training model on {name}...")
            else:
                print(f"No face found in {name}, skipping...")

# Update the model data with new encodings and names
model_data["encodings"] += face_encodings
model_data["names"] += face_names

# Save the updated model data to the model file using JSON
with open(model_path, "w") as model_file:
    json.dump(model_data, model_file)
    logging.info(f"Training model on {face_names}")
