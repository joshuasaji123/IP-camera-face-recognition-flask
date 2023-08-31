import os
import time


# Set up the camera globally
camera = None


def prompt_to_add_photos():
    print("Before running the training, make sure to add photos to the 'Faces' folder.")
    input("Add the photos to the 'Faces' or press Enter to continue...")
    

def main():
    prompt_to_add_photos()
    #show_starting_animation()
    #show_flask_animation()
    print("Initializing...")
    # Run the training script
    os.system("python train_model.py")
    # Run the Flask app
    os.system("python app.py")

    

if __name__ == "__main__":
    main()
