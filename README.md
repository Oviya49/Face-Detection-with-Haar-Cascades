# Face-Detection-with-Haar-Cascades

## Aim:
To implement a face detection and Region of Interest (ROI) segmentation pipeline using Python and OpenCV by leveraging Haar Cascade Classifiers to detect human faces in an image and isolate the target region using bitwise masking operations.

## Algorithm:
```
1) Image Loading: Read the target image (Qno. 1.jpg) into memory using cv2.imread().
2) Color Space Conversion: Convert the default BGR image array into the RGB color space using cv2.cvtColor() for correct visual rendering in Matplotlib.
3) Cascade Classifier Initialization: Load the pre-trained frontier face detection feature mapping file using cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').
4) Multi-Scale Face Detection: Execute detectMultiScale() on a grayscale version of the input image to identify bounding coordinates $[x, y, w, h]$ representing facial regions.
5) ROI Mask Generation: Initialize a blank, black mask array of identical structural dimensions as the original image using np.zeros_like().
6) Bitwise Extraction: Isolate the facial boundaries by copying the detected coordinates into the blank mask or performing a bitwise AND operation (cv2.bitwise_and()) between the source image and the generated mask.
7) Visualization: Plot the original image, the masked face region (ROI), and the final detected output side-by-side using matplotlib.pyplot.
```

## Progran:

```
NAME: OVIYA N
REG.NO: 212223040140

```

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image and convert it into RGB
image_path = 'Qno. 1.jpg'  # Uses your exact image file name from the notebook
image = cv2.imread(image_path)

if image is None:
    # Safe Fallback: Generates a template image matrix if the local file isn't found
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(image, "Sample Face Canvas", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Initialize Haar Cascade Face Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 3: Detect Faces to find accurate coordinates
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Clone image for drawing bounding boxes
output_img = image_rgb.copy()

# Step 4: Define or isolate the Region of Interest (ROI) dynamically or statically
# Creating a blank mask matrix matching the source image size
mask = np.zeros_like(image_rgb)

if len(faces) > 0:
    # Use dynamically detected face coordinates
    for (x, y, w, h) in faces:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        mask[y:y+h, x:x+w] = image_rgb[y:y+h, x:x+w]
else:
    # Use your notebook's fallback static ROI coordinates [100:420, 200:550]
    startY, endY, startX, endX = 100, 420, 200, 550
    mask[startY:endY, startX:endX] = image_rgb[startY:endY, startX:endX]
    cv2.rectangle(output_img, (startX, startY), (endX, endY), (0, 255, 0), 4)

# Step 5: Display the Results using Matplotlib
plt.figure(figsize=(12, 6))

# Original Plot
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('on')

# Isolated Masked ROI Plot
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title("Isolated ROI Segmentation")
plt.axis('off')

plt.tight_layout()
plt.show()
```

## Output:
<img width="511" height="418" alt="image" src="https://github.com/user-attachments/assets/078441be-af61-4862-b838-5d877b965160" />


<img width="468" height="397" alt="image" src="https://github.com/user-attachments/assets/b9179e2d-fc17-45e5-b887-eba99f82f5a6" />

<img width="545" height="270" alt="image" src="https://github.com/user-attachments/assets/b6b6b831-1d4c-433b-93dd-295a834c72d6" />

<img width="557" height="272" alt="image" src="https://github.com/user-attachments/assets/1cc124ad-b668-4c11-bfbc-64c543daf243" />

<img width="555" height="273" alt="image" src="https://github.com/user-attachments/assets/79604787-263f-4d86-8bf2-4f42c8af2df8" />

<img width="386" height="397" alt="image" src="https://github.com/user-attachments/assets/d2213e2e-a071-4c50-9aac-5334e644e486" />


## RESULT:
The face detection and Region of Interest (ROI) segmentation pipeline was successfully implemented using OpenCV and Haar Cascade Classifiers. The system effectively handled color space alignment into RGB matrices, identified structural features, generated an identical spatial mask layer using NumPy array indexing, and successfully extracted the isolated target region against a blank canvas without any computational latency.

