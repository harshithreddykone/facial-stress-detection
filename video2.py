import cv2  # Added import for OpenCV

import os
import numpy as np
from keras.preprocessing import image
import warnings
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
from keras.models import load_model

# Set the threshold for stress detection
threshold = 100  # Adjust the threshold as needed

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the model
model = load_model("d:/best_model.keras")

# Load the face cascade classifier
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a function to capture and process video frames
def process_frame():
    ret, test_img = cap.read()
    if not ret:
        return

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224))

        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        img_pixels = image.img_to_array(roi_color)
        img_pixels = img_pixels / 255.0

        predictions = model.predict(np.expand_dims(img_pixels, axis=0))
        stress_probability = predictions[0][0] * 100

        if stress_probability < threshold:
            stress_detected = True
            label = f"Stress Detected: {stress_probability:.2f}%"
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Use a larger font for stress detection
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.5  # Adjust the size as needed
            font_color = (255, 255, 255)
            line_type = 2

            # Display the label above the face rectangle
            text_size = cv2.getTextSize(label, font, font_scale, line_type)[0]
            text_x = x + (w - text_size[0]) // 2
            cv2.putText(test_img, label, (text_x, y - 20), font, font_scale, font_color, line_type)

            
        else:
            label = f"No Stress: {stress_probability:.2f}%"
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(test_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    img = ImageTk.PhotoImage(image=img)
    video_label.img = img
    video_label.config(image=img)
    video_label.after(10, process_frame)

# Function to start the video feed
def start_video():
    global cap
    cap = cv2.VideoCapture(0)
    process_frame()
    detection_frame.pack()
    

# Function to stop the video feed
def stop_video():
    global cap
    if cap is not None:
        cap.release()
        video_label.config(image=None)
        detection_frame.pack_forget()
        

# Create the main application window
root = tk.Tk()
root.title("Stress Detection")

# Set a background color and window size
root.configure(bg='#E0E0E0')
root.geometry("1200x800")

# Load the background image (change 'background_image.png' to your image file)
bg_image = Image.open("C:/Users/saket/OneDrive/Pictures/OMEN_PortalWP_05.jpg")  # Change to your image file path
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label to display the background image
bg_label = Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# Create a label for displaying the video feed
video_label = Label(root)
video_label.pack(pady=20)

# Detection Frame
detection_frame = Frame(root, bg='white')
detect_label = Label(detection_frame, text="Stress Detection", bg='white', font=('Helvetica', 16))
detect_label.pack(pady=20)
start_button = Button(detection_frame, text="Start Video", command=start_video, bg='#4CAF50', fg='white', font=('Helvetica', 14))
start_button.pack(side=tk.LEFT)
stop_button = Button(detection_frame, text="Stop Video", command=stop_video, bg='#D32F2F', fg='white', font=('Helvetica', 14))
stop_button.pack(side=tk.LEFT)

# Start the video feed immediately
start_video()

root.mainloop()
