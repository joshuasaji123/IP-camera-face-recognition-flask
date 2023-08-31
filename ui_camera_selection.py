import time
import cv2


def get_camera_choice():
    print("Select camera:")
    print("1. Default Camera")
    print("2. IP Camera")
    return input("Enter choice (1/2): ")

def get_ip_camera_info():
    username = input("Enter username for IP Camera: ")
    password = input("Enter password for IP Camera: ")
    ip_link = input("Enter IP camera link: ")
    camera_link = f"http://{username}:{password}@{ip_link}/video"
    return camera_link

def setup_camera(camera_choice):
    if camera_choice == '2':
        return cv2.VideoCapture(get_ip_camera_info())
    else:
        return cv2.VideoCapture(0)

            
def main():
    camera_choice = get_camera_choice()
    print("The camera is starting...")
    camera = setup_camera(camera_choice)
    print("Camera setup complete.")

if __name__ == "__main__":
    main()
