# Face recognition 

This is a Flask-based face recognition system that trains a model to recognize faces in a video stream.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Setting Up Virtual Environment

1. Open a terminal/command prompt.
2. Navigate to the project directory:
    cd path/to/your/project
3. Create a virtual environment:
    python3 -m venv venv
4. Activate the virtual environment:

On macOS and Linux:
    source venv/bin/activate

On Windows:
    venv\Scripts\activate


### Installing Dependencies

Install the required dependencies using the provided `requirements.txt` file:
    pip install -r requirements.txt


### Running the App
Make sure that the Faces folder is populated with the .jpg files with faces you want added to the recognition dataset. And update the paths for the logs and put the path to the log folder. 

To run the program, execute the following command:
    python run_app.py


This will start the app, and you can access it in your web browser at `http://localhost:5000`.

### Resolutions

The app allows you to adjust the resolution of the video stream. You can click on the provided resolution buttons to change the resolution.

### Adding New Faces

1. Open your browser and navigate to `http://localhost:5000`.
2. Use the "Add Face" section to upload an image and provide a name for the new face.
3. The app will automatically recognize the new face when encountered in the video stream.

### Exiting the App

To stop the app, press `Ctrl + C` in the terminal/command prompt where the app is running. Don't forget to deactivate the virtual environment when you're done:
    deactivate


## Compatibility

Tested on macOS, Windows, and Linux.

## Demo

![GIFMaker_me](https://github.com/joshuasaji123/IP-camera-face-recognition-flask/assets/143409651/6186333c-c083-43a2-8207-8ab8fbb1b73d)

