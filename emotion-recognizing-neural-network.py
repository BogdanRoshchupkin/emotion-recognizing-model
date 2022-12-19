import cv2
import numpy as np
import face_recognition

# Load accessible emotions for your model
with open("emotion_file.txt", "r") as f:
    class_names = f.read().split(",")

# Initialize random colors for every emotion
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the neural network model
model = cv2.dnn.readNetFromTensorflow("frozen_graph.pb")
# Webcam input
cap = cv2.VideoCapture(0)

w_frame = int(cap.get(3))
h_frame = int(cap.get(4))
face_locations = []
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_locations = face_recognition.face_locations(frame)
        for face in face_locations:
            top, right, bottom, left = face
            x, y, h, w = left, top, right, bottom
            crop_frame = frame[y:w, x:h]
            count += 1

            # Create a 4D blob from image
            blob = cv2.dnn.blobFromImage(image=crop_frame, size=(224, 224), mean=(104, 117, 123), swapRB=True)
            # Set the input blob to the neural network
            model.setInput(blob)
            # Run the forward pass image blob through the model
            output = model.forward()
            new_output = output.reshape(len(output[0][:]))
            expanded = np.exp(new_output - np.max(new_output))
            prob = expanded / expanded.sum()
            conf = np.max(prob)
            index = np.argmax(prob)
            # Set the label of emotion
            label = class_names[index]
            # Set the color to bounding box and label
            color = COLORS[index]

            # Create a bounding box with text on top
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # cv2.imshow('crop', crop_frame)
            cv2.imshow('image', frame)
        # Stop the program if 'q' button was pushed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()