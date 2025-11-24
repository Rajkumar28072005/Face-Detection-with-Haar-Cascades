# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  
### Program
NAME: RAJKUMAR G
REG NO : 212223230166
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# --- 1. Load Images Safely ---
def load_image(path, flag=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(path, flag)
    if img is None:
        print(f"⚠ Could not load: {path}, using dummy image instead.")
        img = np.zeros((200, 200), dtype=np.uint8)
    return img

model = load_image("img1.jpg")
withglass = load_image("image_02.png")
group = load_image("image_03.png")   # <-- FIXED missing extension

# --- 2. Load Haar Cascades ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- 3. Display Loaded Images ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(model, cmap='gray')
plt.title("Model")

plt.subplot(1, 3, 2)
plt.imshow(withglass, cmap='gray')
plt.title("With Glasses")

plt.subplot(1, 3, 3)
plt.imshow(group, cmap='gray')
plt.title("Group")
plt.show()

# --- 4. Detection Functions ---
def detect_face(img):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_color, (x, y), (x+w, y+h), (255,255,255), 2)
    return img_color

def detect_eyes(img):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    eyes = eye_cascade.detectMultiScale(img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img_color, (x, y), (x+w, y+h), (255,255,255), 2)
    return img_color

# --- 5. Apply Detection and Show Results ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(detect_face(withglass))
plt.title("Face Detection (Glasses)")

plt.subplot(1, 2, 2)
plt.imshow(detect_face(group))
plt.title("Face Detection (Group)")
plt.show()

### OUTPUT
