from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import json
import face_recognition
import logging
import os
import ui_camera_selection

# Set up logging configuration
log_file_path = os.path.join("/Users/joshuasaji/Desktop/Face recognition system/Logs", "app_log.json")
logging.basicConfig(level=logging.INFO, filename=log_file_path, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

camera = None

app = Flask(__name__)

# Default resolution
selected_resolution = (640, 360)

# Load the trained model data using JSON
model_path = os.path.join("models", "face_recognition_model.json")
with open(model_path, "r") as model_file:
    model_data = json.load(model_file)

known_face_encodings = model_data["encodings"]
known_face_names = model_data["names"]

# Cache known face encodings for better performance
face_encoding_cache = {}

# Variables for frame skipping
frame_skip_counter = 0
frame_skip_interval = 15  # Process every nth frame

# Display resolution options
resolutions = {
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360)
}
current_resolution = resolutions["480p"]  # Default resolution


def gen_frames():
    global frame_skip_counter
    
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Resize frame to the selected resolution
        frame = cv2.resize(frame, selected_resolution)
        
        # Apply frame skipping
        frame_skip_counter = (frame_skip_counter + 1) % frame_skip_interval
        if frame_skip_counter != 0:
            continue
        
        #print("Generating frames...")
        frame = recognize_face(frame)  # Perform face recognition on the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield boundary + frame_bytes + b'\r\n'

def recognize_face(frame): 
    #print("Recognizing faces...")
    # Find face locations and encodings in the current frame
    logging.info(f"Recognizing faces at {selected_resolution[0]}x{selected_resolution[1]} resolution")
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if not face_encodings:  # Check if no faces are detected
        #print("No face encodings detected")
        #print("Loaded model data:", model_data)
        return frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        if face_encoding.tobytes() in face_encoding_cache:
            name = face_encoding_cache[face_encoding.tobytes()]
            #print(f"Known face detected: {name}")
        else:
            #print("Number of known face encodings:", len(known_face_encodings))
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = int(face_distance.argmin())
            #print("Number of known face encodings:", len(known_face_encodings))

            if face_distance[best_match_index] < 0.6:  # Adjust the threshold as needed
                name = known_face_names[best_match_index]
                accuracy_percentage = f"{100 - face_distance[best_match_index] * 100:.2f}%"
                face_encoding_cache[face_encoding.tobytes()] = name
                #print(f"Known face detected: {name} ({accuracy_percentage} accuracy)")
            else:
                name = "Unknown"
                accuracy_percentage = ""
                #print("Unknown face detected")
        logging.info(f"Detected face: {name} at ({left},{top}) - ({right},{bottom})")
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Draw rectangle around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} {accuracy_percentage}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def restart_camera():
    global camera
    if camera.isOpened():
        camera.release()
    camera = cv2.VideoCapture(0)

@app.route('/set_resolution/<resolution>')
def set_resolution(resolution):
    global selected_resolution
    if resolution == '720p':
        selected_resolution = (1280, 720)
    elif resolution == '480p':
        selected_resolution = (854, 480)
    elif resolution == '360p':
        selected_resolution = (640, 360)
    
    restart_camera()  # Restart the camera with the new resolution
    print("The camera is restarting...")
    return redirect(url_for('ui'))

@app.route('/add_face', methods=['POST'])
def add_face():
    try:    
        if request.method == 'POST':
            name = request.form['name']
            image = request.files['image']
            if name and image:
                image_path = f'Faces/{name}.jpg'
                image.save(image_path)
                logging.info(f"Saved face: {name}")
                # Load and encode the newly added face
                new_face_image = face_recognition.load_image_file(image_path)
                new_face_encodings = face_recognition.face_encodings(new_face_image)
                if new_face_encodings:
                    new_face_encoding = new_face_encodings[0]
                    known_face_encodings.append(new_face_encoding)
                    known_face_names.append(name)
                    
                    # Retrain the model
                    try:
                        # Restart the camera
                        restart_camera()
                        print("Restarting camera...")
                        #ui_camera_selection.show_reloading_animation()
                        # Run the training script to retrain the model
                        logging.info(f"Retraining model for {name}'s face")
                        os.system("python train_model.py")
                        
                        # Update the model data in app.py with the new encodings and names
                        model_data['encodings'] = known_face_encodings
                        model_data['names'] = known_face_names
                        with open(model_path, "w") as model_file:
                            json.dump(model_data, model_file)
                            logging.info(f"Encoding dumped for {name}'s face")
                    except Exception as e:
                        logging.error(f"An error occurred during retraining and model update: {e}")
                else:
                    logging.warning("No face encoding detected in the added image.")
                
                return redirect(url_for('ui'))
        return render_template('ui.html', resolutions=resolutions, current_resolution=current_resolution)
    except Exception as e:
        logging.error(f"An exception occurred: {e}")
        return "An error occurred while processing your request."


@app.route('/kill')
def kill_program():
    camera.release()
    cv2.destroyAllWindows()
    print("Program terminated.")
    os._exit(0)  # Terminate the program

@app.route('/')
def ui():
    return render_template('ui.html', resolutions=resolutions, current_resolution=current_resolution)

if __name__ == "__main__":
    camera_choice = ui_camera_selection.get_camera_choice()
    if camera_choice == '2':
        camera_link = ui_camera_selection.get_ip_camera_info()
        camera = cv2.VideoCapture(camera_link)
    else:
        camera = cv2.VideoCapture(0)
    
    print("The camera is starting...")

    app.run(debug=True)

