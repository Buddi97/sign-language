import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize the video capture and the hand detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

detector = HandDetector(maxHands=1)  # Initialize the hand detector
classifier = None

# Try to load the ASL classifier model and handle exceptions
try:
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
except Exception as e:
    print("Error loading model:", e)
    exit()

# Parameters for image processing
offset = 20
imgSize = 300
labels = ["A", "B", "C"]  # Example labels, adjust according to your model

while True:
    # Read frame from camera
    success, img = cap.read()
    if not success:
        print("Error: Camera feed lost.")
        break

    imgOutput = img.copy()  # Copy the original image for display
    hands, img = detector.findHands(img)  # Detect hand(s) in the frame

    if hands:
        # Get the bounding box of the hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a blank white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the region of interest around the hand with some offset
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if the crop region is valid
        if imgCrop.size == 0:
            print("Warning: Invalid crop region.")
            continue

        # Resize image maintaining the aspect ratio
        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                # Height is greater than width
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                # Width is greater than height
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Perform prediction using the model
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]
            print(f"Prediction: {label} - {prediction}")

            # Display prediction on the output image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Display intermediate images for debugging
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print("Error during prediction processing:", e)

    # Display the output image
    cv2.imshow("Image", imgOutput)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
