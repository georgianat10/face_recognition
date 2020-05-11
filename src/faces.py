import numpy as np
import cv2
import pickle

# Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # convert the frame to because cascade works off of grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #print(x, y, w, h)
        # roi = region of interest = face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #implement recognizer
        _id, conf = recognizer.predict(roi_gray) #predict the region of interest
        if conf >= 4 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[_id]
            color = (255, 255, 255)
            cv2.putText(frame, name, (x, y), font, 1, color, 2, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)

        # draw a rectangle arouf the face
        color = (255, 0, 0)  # BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
        
        eyes = eye_cascade.detectMultiScale(roi_color) 
        for(ey, ex, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # display the result frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
